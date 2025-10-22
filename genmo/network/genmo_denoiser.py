import torch
import torch.nn as nn
from einops import repeat
from timm.models.vision_transformer import Mlp

from genmo.network.base_arch.embeddings.pe import PositionalEncoding
from genmo.network.base_arch.transformer.encoder_rope import (
    DecoderRoPEBlock,
    EncoderRoPEBlock,
)
from genmo.network.base_arch.transformer.layer import zero_module
from genmo.utils.net_utils import length_to_mask


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

        # ===== build model ===== #
        # Input (Kp2d)
        # Main token: map d_obs 2 to 32
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32,
            hidden_features=self.latent_dim * 2,
            out_features=self.latent_dim,
            drop=dropout,
        )

        self._build_condition_embedder()

        # Transformer
        self.blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(
                    self.latent_dim,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )
        self.embed_text = nn.Linear(self.encoded_text_dim, self.latent_dim)
        self.text_encoder_cfg = text_encoder_cfg
        text_encode_mode = text_encoder_cfg.get("mode", "first")
        if text_encode_mode == "first":
            self.text_encode_layer_idx = [0]
        elif text_encode_mode == "all":
            self.text_encode_layer_idx = list(range(num_layers))
        elif text_encode_mode.startswith("every_"):
            self.text_encode_layer_idx = list(
                range(0, num_layers, int(text_encode_mode.split("_")[1]))
            )
        elif text_encode_mode == "none":
            self.text_encode_layer_idx = []
        else:
            raise ValueError(f"Invalid text_encode_mode {text_encode_mode}")
        use_self_attn = text_encoder_cfg.get("use_self_attn", False)
        net_type = text_encoder_cfg.get("net_type", "rope_decoder")
        cross_attn_type = text_encoder_cfg.get("cross_attn_type", "rope")
        pos_enc_dropout = text_encoder_cfg.get("pos_enc_dropout", 0.0)
        self.text_encoder_layers = nn.ModuleDict()
        for idx in self.text_encode_layer_idx:
            if net_type == "rope_decoder":
                text_block = DecoderRoPEBlock(
                    self.latent_dim,
                    self.num_heads,
                    use_self_attn=use_self_attn,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    cross_attn_type=cross_attn_type,
                    pos_enc_dropout=pos_enc_dropout,
                )
            else:
                raise ValueError(f"Invalid net_type {net_type}")
            self.text_encoder_layers[f"{idx}"] = text_block

        self.motion_text_pos_enc = motion_text_pos_enc

        # Output heads
        self.final_layer = Mlp(self.latent_dim, out_features=self.output_dim)
        self.pred_cam_head = (
            pred_cam_dim > 0
        )  # keep extra_output for easy-loading old ckpt
        if self.pred_cam_head:
            self.pred_cam_head = Mlp(self.latent_dim, out_features=pred_cam_dim)
            self.register_buffer(
                "pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False
            )
            self.register_buffer(
                "pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False
            )

        self.static_conf_head = static_conf_dim > 0
        if self.static_conf_head:
            self.static_conf_head = Mlp(self.latent_dim, out_features=static_conf_dim)

        self.add_cond_linear = nn.Linear(xt_dim + self.latent_dim, self.latent_dim)

        self.avgbeta = avgbeta

    def _build_condition_embedder(self):
        latent_dim = self.latent_dim
        dropout = self.dropout
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(self.cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        if self.cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if self.imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
            )

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
        x = y["f_cond"]
        length = y["length"]
        multi_text_data = y.get("multi_text_data", None)
        L = xt.size(1)
        B = xt.size(0)

        if self.input_remove_condition:
            x = torch.zeros_like(x)

        if self.input_remove_global:
            xt[..., -15:] = 0

        if motion_mask_3d is not None:
            xt = xt * (1 - motion_mask_3d) + observed_motion_3d * motion_mask_3d

        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        x = x + emb

        x = self.add_cond_linear(torch.cat([x, xt], dim=-1))

        if "encoded_text" in y and len(self.text_encode_layer_idx) > 0:
            enc_text = y["encoded_text"].clone()
            if self.training and self.text_mask_prob > 0:
                mask = torch.rand((B,), device=x.device) < self.text_mask_prob
                enc_text = enc_text * (1 - mask[:, None, None].float())
            if rm_text_flag is not None:
                enc_text = enc_text * (1 - rm_text_flag[:, None, None].float())
            emb_text = self.embed_text(enc_text)
            if self.use_text_pos_enc:
                emb_text = self.sequence_pos_encoder(emb_text, batch_first=True)

        if multi_text_data is not None:
            multi_text_data["text_embed_feats"] = self.embed_text(
                multi_text_data["text_embed"]
            )
            if self.use_text_pos_enc:
                multi_text_data["text_embed_feats"] = self.sequence_pos_encoder(
                    multi_text_data["text_embed_feats"], batch_first=True
                )

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L)

        if L > self.max_len:
            attnmask = torch.ones((B, L, L), device=x.device, dtype=torch.bool)
            attnmask_noar = torch.ones((L, L), device=x.device, dtype=torch.bool)
            attnmask_ar = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - self.max_len // 2)
                max_ind = min(L, i + self.max_len // 2)
                eff_max_len = min(self.max_len, L)
                max_ind_exp = max(eff_max_len, max_ind)
                min_ind_exp = min(L - eff_max_len, min_ind)
                attnmask_ar[i, min_ind:max_ind] = False
                attnmask_noar[i, min_ind_exp:max_ind_exp] = False
            attnmask[:] = attnmask_noar.unsqueeze(0)
        else:
            attnmask = None

        # Transformer
        for i, block in enumerate(self.blocks):
            if i in self.text_encode_layer_idx:
                text_block = self.text_encoder_layers[f"{i}"]
                x = text_block(
                    x,
                    emb_text,
                    attn_mask=attnmask,
                    tgt_key_padding_mask=pmask,
                    multi_text_data=multi_text_data,
                    motion_text_pos_enc=self.motion_text_pos_enc,
                )
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        # Output
        sample = self.final_layer(x)  # (B, L, C)
        if self.avgbeta:  # TODO: fix based on beta dims
            betas = (sample[..., 126:136] * (~pmask[..., None])).sum(1) / length[
                :, None
            ]  # (B, C)
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)

        # Output (extra)
        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(
                pred_cam[..., 0], 0.25
            )  # min_clamp s to 0.25 (prevent negative prediction)

        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x)  # (B, L, C')

        output = {
            "pred_context": x,
            "pred_x": sample,
            "pred_x_start": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
        }
        return output
