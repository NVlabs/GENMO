import torch.nn as nn
from yacs.config import CfgNode

from .smpl_head import SMPLTransformerDecoderHead
from .vit import ViT


class HMR2(nn.Module):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        self.backbone = ViT(
            img_size=(256, 192),
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.55,
        )
        self.smpl_head = SMPLTransformerDecoderHead(cfg)

    def forward(self, batch):
        """Run HMR2 in feature-extraction mode.

        Args:
            batch: dict with key "img" of shape (B, 3, 256, 256).

        Returns:
            token_out: (B, 1024) feature vector.
        """
        # Backbone
        x = batch["img"][:, :, :, 32:-32]
        vit_feats = self.backbone(x)

        # Output head -- feat_mode only
        token_out = self.smpl_head(vit_feats, only_return_token_out=True)  # (B, 1024)
        return token_out
