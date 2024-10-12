from hmr4d.network.mdm.mdm_base import MDMBase, import_type_from_str
from hmr4d.network.mdm.mdm_2mode import MDMBase2Mode


""" 
Main Model
"""


class MDM(MDMBase):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.denoiser = import_type_from_str(self.model_cfg.denoiser.type)(
            pl_module=self, **self.model_cfg.denoiser
        )
        self.init_diffusion()
        if self.model_cfg.get('preload_checkpoint', True):
            self.load_pretrain_checkpoint()
        return


class MDM2Mode(MDMBase2Mode):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.denoiser = import_type_from_str(self.model_cfg.denoiser.type)(
            pl_module=self, **self.model_cfg.denoiser
        )
        self.init_diffusion()
        if self.model_cfg.get("preload_checkpoint", True):
            self.load_pretrain_checkpoint()
        return
