# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal ViTPose-H COCO-17 model builder for inference.

Architecture matches the ViTPose checkpoint from GVHMR
(vitpose-h-multi-coco.pth). Adapted from the ViTPose-pytorch implementation
(OpenMMLab / ViTAE-Transformer).
"""
from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

# ---------------------------------------------------------------------------
# ViT backbone (custom, NOT from timm — matches ViTPose checkpoint keys)
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio**2)
        self.patch_shape = (
            int(img_size[0] // patch_size[0] * ratio),
            int(img_size[1] // patch_size[1] * ratio),
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=4 + 2 * (ratio // 2 - 1),
        )

    def forward(self, x, **kwargs):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        ratio=1,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            ratio=ratio,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.last_norm = norm_layer(embed_dim)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)
        x = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return x


# ---------------------------------------------------------------------------
# Heatmap head (TopdownHeatmapSimpleHead, inference-only)
# ---------------------------------------------------------------------------


class TopdownHeatmapSimpleHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=None,
    ):
        super().__init__()
        self.in_channels = in_channels

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers, num_deconv_filters, num_deconv_kernels
            )
        else:
            self.deconv_layers = nn.Identity()

        kernel_size, padding = 1, 0
        if extra is not None and "final_conv_kernel" in extra:
            kernel_size = extra["final_conv_kernel"]
            padding = 1 if kernel_size == 3 else 0

        conv_channels = num_deconv_filters[-1] if num_deconv_layers > 0 else in_channels
        self.final_layer = nn.Conv2d(
            conv_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding
        )

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        in_ch = self.in_channels
        for i in range(num_layers):
            kernel = num_kernels[i]
            if kernel == 4:
                padding, output_padding = 1, 0
            elif kernel == 3:
                padding, output_padding = 1, 1
            elif kernel == 2:
                padding, output_padding = 0, 0
            else:
                raise ValueError(f"Unsupported deconv kernel size: {kernel}")
            planes = num_filters[i]
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_ch,
                        planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = planes
        return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Full model wrapper
# ---------------------------------------------------------------------------


class VitPoseModel(nn.Module):
    def __init__(self, backbone, keypoint_head):
        super().__init__()
        self.backbone = backbone
        self.keypoint_head = keypoint_head

    def forward(self, x):
        features = self.backbone(x)
        return self.keypoint_head(features)


def build_vitpose_coco17(checkpoint_path: str) -> VitPoseModel:
    """Build ViTPose-H COCO-17 model and load checkpoint.

    Architecture: ViT-Huge (patch16, 256x192) + 2-layer deconv head -> 17 keypoints.
    """
    backbone = ViT(
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
    )
    head = TopdownHeatmapSimpleHead(
        in_channels=1280,
        out_channels=17,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1),
    )
    model = VitPoseModel(backbone, head)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[ViTPose] WARNING: {len(missing)} missing keys")
    if unexpected:
        print(f"[ViTPose] WARNING: {len(unexpected)} unexpected keys")
    return model
