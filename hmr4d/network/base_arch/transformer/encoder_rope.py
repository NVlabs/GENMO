import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Mlp
from typing import Optional, Tuple
from einops import einsum, rearrange, repeat
from hmr4d.network.base_arch.embeddings.rotary_embedding import ROPE


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.rope = ROPE(self.head_dim, max_seq_len=4096)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, context=None, attn_mask=None, key_padding_mask=None):
        # x: (B, L, C)
        # context: (B, L_ctx, C) or None
        # attn_mask: (L, L) or (L, L_ctx)
        # key_padding_mask: (B, L) or (B, L_ctx)
        B, L, _ = x.shape
        if context is None:
            context = x
        L_ctx = context.shape[1]

        xq = self.query(x)
        xk = self.key(context)
        xv = self.value(context)

        xq = xq.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xk = xk.reshape(B, L_ctx, self.num_heads, -1).transpose(1, 2)
        xv = xv.reshape(B, L_ctx, self.num_heads, -1).transpose(1, 2)

        xq = self.rope.rotate_queries_or_keys(xq)  # B, N, L, C
        xk = self.rope.rotate_queries_or_keys(xk)  # B, N, L_ctx, C

        attn_score = einsum(xq, xk, "b n i c, b n j c -> b n i j") / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(1, 1, L, L_ctx).expand(B, self.num_heads, -1, -1)
            attn_score = attn_score.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.reshape(B, 1, 1, L_ctx).expand(-1, self.num_heads, L, -1)
            attn_score = attn_score.masked_fill(key_padding_mask, float("-inf"))

        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = self.dropout(attn_score)
        output = einsum(attn_score, xv, "b n i j, b n j c -> b n i c")  # B, N, L, C
        output = output.transpose(1, 2).reshape(B, L, -1)  # B, L, C
        output = self.proj(output)  # B, L, C
        return output


class EncoderRoPEBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)

        self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Zero-out adaLN modulation layers
        nn.init.constant_(self.gate_msa, 0)
        nn.init.constant_(self.gate_mlp, 0)

    def forward(self, x, attn_mask=None, tgt_key_padding_mask=None):
        x = x + self.gate_msa * self._sa_block(
            self.norm1(x), attn_mask=attn_mask, key_padding_mask=tgt_key_padding_mask
        )
        x = x + self.gate_mlp * self.mlp(self.norm2(x))
        return x

    def _sa_block(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, C)
        x = self.attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x


class DecoderRoPEBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, use_self_attn=True, cross_attn_type='rope', **block_kwargs):
        super().__init__()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
            self.self_attn = RoPEAttention(hidden_size, num_heads, dropout)
            self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.constant_(self.gate_msa, 0)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.cross_attn_type = cross_attn_type
        if cross_attn_type == 'rope':
            self.cross_attn = RoPEAttention(hidden_size, num_heads, dropout)
        elif cross_attn_type == 'mha':
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)

        self.gate_cross_attn = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Zero-out adaLN modulation layers
        nn.init.constant_(self.gate_cross_attn, 0)
        nn.init.constant_(self.gate_mlp, 0)

    def forward(self, x, context, attn_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if self.use_self_attn:
            x = x + self.gate_msa * self._sa_block(
                self.norm1(x), attn_mask=attn_mask, key_padding_mask=tgt_key_padding_mask
            )
        x = x + self.gate_cross_attn * self._ca_block(
            self.norm2(x), context=context, key_padding_mask=memory_key_padding_mask
        )
        x = x + self.gate_mlp * self.mlp(self.norm3(x))
        return x

    def _sa_block(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, C)
        x = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x

    def _ca_block(self, x, context, key_padding_mask=None):
        # x: (B, L, C)
        if self.cross_attn_type == 'rope':
            x = self.cross_attn(x, context=context, key_padding_mask=key_padding_mask)
        elif self.cross_attn_type == 'mha':
            x = self.cross_attn(x, context, context, key_padding_mask=key_padding_mask)[0]
        else:
            raise ValueError(f"Invalid cross_attn_type: {self.cross_attn_type}")
        return x