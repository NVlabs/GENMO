from motiondiff.data_pipeline.humanml.utils.word_vectorizer import WordVectorizer
from motiondiff.models.mdm_v2.mdm_base import MDMBase
from motiondiff.utils.tools import import_type_from_str

"""
Main Model
"""


class MDM(MDMBase):
    def __init__(self, cfg, is_inference=False, preload_checkpoint=True):
        super().__init__(cfg, is_inference)
        self.denoiser = import_type_from_str(self.model_cfg.denoiser.type)(
            pl_module=self, **self.model_cfg.denoiser
        )
        self.init_diffusion()
        if preload_checkpoint:
            self.load_pretrain_checkpoint()
        return
