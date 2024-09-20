import torch
from motiondiff.models.mdm_v3.mdm_base import MDMBase
from motiondiff.utils.tools import import_type_from_str
from motiondiff.data_pipeline.humanml.utils.word_vectorizer import WordVectorizer


""" 
Main Model
"""


class ControlMDM(MDMBase):

    def __init__(self, cfg, is_inference=False, preload_checkpoint=True):
        super().__init__(cfg, is_inference)
        self.denoiser = import_type_from_str(self.model_cfg.denoiser.type)(pl_module=self, **self.model_cfg.denoiser)
        self.init_diffusion()
        if preload_checkpoint:
            self.load_pretrain_checkpoint()
        return

    def configure_optimizers(self):

        optimizer_cfg = self.cfg.train.optimizer
        params = list(self.denoiser.control_seqTransEncoder.parameters())
        params += list(self.denoiser.hint_process.parameters())
        # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay)
        optimizer = torch.optim.AdamW(
            params,
            # self.parameters(),
            lr=optimizer_cfg.lr,
            weight_decay=optimizer_cfg.weight_decay,
        )
        scheduler_cfg = self.cfg.train.get('scheduler', None)
        if scheduler_cfg is not None:
            type = scheduler_cfg.pop("type")
            lt_kwargs = dict(scheduler_cfg.pop("lt_kwargs", {}))
            scheduler = import_type_from_str(type)(optimizer, **scheduler_cfg)
            lt_kwargs["scheduler"] = scheduler
            return {"optimizer": optimizer, "lr_scheduler": lt_kwargs}
        else:
            return optimizer

    def load_pretrain_checkpoint(self):
        if 'pretrained_checkpoint' in self.model_cfg:
            cp_cfg = self.model_cfg.pretrained_checkpoint
            state_dict = torch.load(cp_cfg.path, map_location='cpu')['state_dict']
            filter_keys = cp_cfg.get('filter_keys', [])
            if len(filter_keys) > 0:
                print(f'Filtering checkpoint keys: {filter_keys}')
                skipped_keys = [k for k in state_dict.keys() if any(key in k for key in filter_keys)]
                print(f'Skipped keys: {skipped_keys}')
                state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in filter_keys)}
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=cp_cfg.get('strict', True))

            if len(missing_keys) > 0:
                for k in missing_keys:
                    assert k.startswith('denoiser.control_seqTransEncoder') or k.startswith('denoiser.hint_process') or k.startswith('smpl.'), k
                print(f'Missing keys: {missing_keys}')
            if len(unexpected_keys) > 0:
                print(f'Unexpected keys: {unexpected_keys}')
            assert 'denoiser.control_seqTransEncoder.hint_layers.0.linear1.weight' in self.state_dict().keys()
            if 'denoiser.control_seqTransEncoder.hint_layers.0.linear1.weight' not in state_dict.keys():
                print("Copy denoiser for controlnet")
                seqTransEncoder_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('denoiser.seqTransEncoder'):
                        new_k = k.replace('denoiser.seqTransEncoder.layers', 'hint_layers')
                        seqTransEncoder_state_dict[new_k] = v.clone()

                missing_keys, unexpected_keys = self.denoiser.control_seqTransEncoder.load_state_dict(seqTransEncoder_state_dict, strict=False)
                assert len(unexpected_keys) == 0
                for k in missing_keys:
                    assert k.startswith('zero_conv')
