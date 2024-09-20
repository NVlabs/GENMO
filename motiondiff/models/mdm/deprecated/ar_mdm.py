import torch
from motiondiff.models.mdm.mdm_base import MDMBase
from motiondiff.utils.tools import import_type_from_str
from motiondiff.data_pipeline.humanml.utils.word_vectorizer import WordVectorizer
from motiondiff.utils.scheduler import update_scheduled_params
from motiondiff.utils.torch_utils import tensor_to
from motiondiff.data_pipeline.tensors import collate
from motiondiff.diffusion.gaussian_diffusion import ModelMeanType


""" 
Main Model
"""


class ARMDM(MDMBase):

    def __init__(self, cfg, is_inference=False, preload_checkpoint=True):
        super().__init__(cfg, is_inference)
        self.denoiser = import_type_from_str(self.model_cfg.denoiser.type)(pl_module=self, **self.model_cfg.denoiser)
        self.init_diffusion()
        if preload_checkpoint:
            self.load_pretrain_checkpoint()
        return

    def training_step(self, batch, batch_idx):
        schedule = self.cfg.get('schedule', dict())
        update_scheduled_params(self, schedule, self.global_step)
        ar_window = self.model_cfg.ar_window
        motion_prefix_len = self.model_cfg.motion_prefix_len

        data = {}
        motion, cond = batch
        if motion.device != self.device:
            motion, cond = tensor_to([motion, cond], device=self.device)
        if self.motion_rep == 'position':
            motion = motion[:, :67]
        
        lengths = cond['y']['lengths']
        motion_prefix = []
        motion_ar = []
        mask = []

        for i in range(motion.shape[0]):
            motion_start = torch.randint(max(lengths[i].item() - ar_window, 1), (1,)).item()
            motion_ar_i = motion[i, ..., motion_start:motion_start + ar_window]
            mask_i = cond['y']['mask'][i, ..., motion_start:motion_start + ar_window]
            motion_prefix_i = motion[i, ..., :motion_start]
            if motion_prefix_i.shape[-1] > motion_prefix_len:
                motion_prefix_i = motion_prefix_i[..., -motion_prefix_len:]
            else:
                motion_prefix_i = torch.cat([torch.zeros((*motion_prefix_i.shape[:-1], motion_prefix_len - motion_prefix_i.shape[-1]), device=self.device), motion_prefix_i], dim=-1)
            motion_ar.append(motion_ar_i)
            motion_prefix.append(motion_prefix_i)
            mask.append(mask_i)
        data['motion'] = torch.stack(motion_ar, dim=0)
        cond['y']['motion_prefix'] = torch.stack(motion_prefix, dim=0)
        data['cond'] = cond
        data['mask'] = torch.stack(mask, dim=0)

        if self.augment_text:
            self.augment_data_text(data)

        t, t_weights = self.schedule_sampler.sample(motion.shape[0], self.device)
        data = self.get_diffusion_pred_target(data, t)
        loss, loss_dict, loss_uw_dict = self.compute_loss(data, t, t_weights)

        self.log('loss/train_all', loss, on_step=True, on_epoch=True, sync_dist=True)
        for key, val in loss_uw_dict.items():
            self.log(f'loss/train_{key}', val, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def infer_texts(self, texts, num_frames, progress=True):
        diffusion = self.test_diffusion
        batch_size = len(texts)
        ar_window = self.model_cfg.ar_window
        motion_prefix_len = self.model_cfg.motion_prefix_len

        cur_sample = None
        while cur_sample is None or cur_sample.shape[-1] < num_frames:
            _, cond = collate(
                [{'inp': torch.tensor([[0.]]), 'target': 0, 'text': txt, 'tokens': None, 'lengths': ar_window} for txt in texts]
            )
            cond = tensor_to(cond, device=self.device)

            if cur_sample is None:
                motion_prefix = torch.zeros(batch_size, self.denoiser.njoints, self.denoiser.nfeats, motion_prefix_len, device=self.device)
            elif cur_sample.shape[-1] > motion_prefix_len:
                motion_prefix = cur_sample[..., -motion_prefix_len:]
            else:
                motion_prefix = torch.cat([torch.zeros((*cur_sample.shape[:-1], motion_prefix_len - cur_sample.shape[-1]), device=self.device), cur_sample], dim=-1)
            cond['y']['motion_prefix'] = motion_prefix

            denoiser = self.guided_denoiser
            cond['y']['scale'] = torch.ones(batch_size, device=self.device) * self.cfg.model.diffusion.guidance_param
            
            diff_sampler = self.cfg.model.diffusion.get('sampler', 'ddim')
            if diff_sampler == 'ddim':
                sample_fn = diffusion.ddim_sample_loop
                kwargs = {'eta': self.cfg.model.diffusion.ddim_eta}
            else:
                sample_fn = diffusion.p_sample_loop
                kwargs = {}

            samples = sample_fn(
                denoiser,
                (batch_size, self.denoiser.njoints, self.denoiser.nfeats, ar_window),
                clip_denoised=False,
                model_kwargs=cond,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=progress,
                dump_steps=None,
                noise=None,
                const_noise=False,
                **kwargs
            )
            if cur_sample is None:
                cur_sample = samples
            else:
                cur_sample = torch.cat([cur_sample, samples], dim=-1)

        samples = cur_sample[..., :num_frames]
        return samples
    
    def validate_loss(self, batch, batch_idx):
        with torch.no_grad():
            training = self.training
            self.train()
            ar_window = self.model_cfg.ar_window
            motion_prefix_len = self.model_cfg.motion_prefix_len
            data = {}
            motion, cond = batch
            batch_size = motion.shape[0]
            if motion.device != self.device:
                motion, cond = tensor_to([motion, cond], device=self.device)
            if self.motion_rep == 'position':
                motion = motion[:, :67]

            lengths = cond['y']['lengths']
            motion_prefix = []
            motion_ar = []
            mask = []

            for i in range(motion.shape[0]):
                motion_start = torch.randint(max(lengths[i].item() - ar_window, 1), (1,)).item()
                motion_ar_i = motion[i, ..., motion_start:motion_start + ar_window]
                mask_i = cond['y']['mask'][i, ..., motion_start:motion_start + ar_window]
                motion_prefix_i = motion[i, ..., :motion_start]
                if motion_prefix_i.shape[-1] > motion_prefix_len:
                    motion_prefix_i = motion_prefix_i[..., -motion_prefix_len:]
                else:
                    motion_prefix_i = torch.cat([torch.zeros((*motion_prefix_i.shape[:-1], motion_prefix_len - motion_prefix_i.shape[-1]), device=self.device), motion_prefix_i], dim=-1)
                motion_ar.append(motion_ar_i)
                motion_prefix.append(motion_prefix_i)
                mask.append(mask_i)
            data['motion'] = torch.stack(motion_ar, dim=0)
            cond['y']['motion_prefix'] = torch.stack(motion_prefix, dim=0)
            data['cond'] = cond
            data['mask'] = torch.stack(mask, dim=0)

            t, t_weights = self.schedule_sampler.sample(motion.shape[0], self.device)
            data = self.get_diffusion_pred_target(data, t)
            loss, loss_dict, loss_uw_dict = self.compute_loss(data, t, t_weights)
            self.train(training)
        return loss, loss_uw_dict, batch_size

