import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from timm.models.vision_transformer import Mlp

from hmr4d.configs import MainStore, builds
from hmr4d.network.base_arch.transformer.encoder_rope import (
    DecoderRoPEBlock,
    EncoderRoPEBlock,
)
from hmr4d.network.base_arch.transformer.layer import zero_module
from hmr4d.utils.net_utils import length_to_mask
from motiondiff.models.mdm.modules import PositionalEncoding
from phc.learning.mlp import MLP


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])


class NetworkEncoderRoPE(nn.Module):
    def __init__(
        self,
        # x
        output_dim=151,
        xt_dim=157,
        max_len=120,
        # condition
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        # intermediate
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        # output
        pred_cam_dim=3,
        static_conf_dim=6,
        # training
        dropout=0.1,
        # other
        avgbeta=True,
        njoints=None,
        encoded_text_dim=1024,
        use_text_pos_enc=True,
        text_encoder_cfg={},
        motion_text_pos_enc=None,
        text_mask_prob=0.0,
        input_remove_global=False,
        input_remove_condition=False,
        allow_autoregressive=True,
        use_raw_humanoid_obs=False,
        **kwargs,
    ):
        super().__init__()

        # input
        self.output_dim = output_dim
        self.max_len = max_len

        # condition
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim

        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.njoints = njoints
        self.nfeats = 1
        self.encoded_text_dim = encoded_text_dim
        self.text_mask_prob = text_mask_prob
        self.use_text_pos_enc = use_text_pos_enc
        self.input_remove_global = input_remove_global
        self.input_remove_condition = input_remove_condition
        self.allow_autoregressive = allow_autoregressive
        self.use_raw_humanoid_obs = use_raw_humanoid_obs

        units = [2048, 1024, 512]  # Example hidden layer size
        mlp_in_dim = 934 if self.use_raw_humanoid_obs else self.latent_dim
        self.model = MLP(mlp_in_dim, 69, units, "silu")

    def forward(
        self,
        xt,
        timesteps,
        y=None,
        inputs=None,
        observed_motion_3d=None,
        motion_mask_3d=None,
        rm_text_flag=None,
        **kwargs,
    ):
        """
        Args:
            x: None we do not use it
            timesteps: (B,)
            length: (B), valid length of x, if None then use x.shape[2]
            f_imgseq: (B, L, C)
            f_cliffcam: (B, L, 3), CLIFF-Cam parameters (bbx-detection in the full-image)
            f_noisyobs: (B, L, C), nosiy pose observation
            f_cam_angvel: (B, L, 6), Camera angular velocity
        """
        L = xt.size(1)
        x = y["f_cond"]
        if self.use_raw_humanoid_obs:
            x = inputs["f_condition"]["humanoid_obs"][:, :L]
        sample = self.model(x)

        output = {
            "pred_x": sample,
            "pred_x_start": sample,
        }
        return output


# Add to MainStore
group_name = "network/mv2d"
MainStore.store(
    name="relative_transformer",
    node=builds(NetworkEncoderRoPE, populate_full_signature=True),
    group=group_name,
)
