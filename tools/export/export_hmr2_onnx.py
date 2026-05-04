# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the HMR2 ViT feature extractor to ONNX.

The exported model takes ImageNet-normalized 256x256 RGB crops and produces
1024-D pose tokens (the SMPL-head's per-frame ``token_out``). The 256x256 →
256x192 horizontal crop is baked into the graph, matching the runtime
preprocessing in ``gem/network/hmr2/hmr2.py``.

Usage::

    cd /path/to/gem-smpl
    python tools/export/export_hmr2_onnx.py \
        --hmr2_ckpt inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt \
        --output inputs/onnx/hmr2.onnx
"""
# ruff: noqa: E402
import argparse
import functools
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

torch.load = functools.partial(torch.load, weights_only=False)


class HMR2FeatureWrapper(nn.Module):
    """Wraps an HMR2 model so its forward signature is ONNX-friendly.

    Input  imgs : (B, 3, 256, 256) ImageNet-normalized RGB.
    Output f_imgseq : (B, 1024) ViT pose token.
    """

    def __init__(self, hmr2_model: nn.Module):
        super().__init__()
        self.model = hmr2_model.eval()

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # Mirror gem/network/hmr2/hmr2.py:36 — slice 256x256 → 256x192 inside the graph.
        x = imgs[:, :, :, 32:-32]
        vit_feats = self.model.backbone(x)
        token_out = self.model.smpl_head(vit_feats, only_return_token_out=True)
        return token_out


def parse_args():
    parser = argparse.ArgumentParser(description="Export HMR2 ViT to ONNX")
    parser.add_argument(
        "--hmr2_ckpt", type=str,
        default="inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
    )
    parser.add_argument("--output", type=str, default="inputs/onnx/hmr2.onnx")
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    if not os.path.exists(args.hmr2_ckpt):
        sys.exit(f"[Export] HMR2 checkpoint not found: {args.hmr2_ckpt}")

    print(f"[Export] Loading HMR2 from {args.hmr2_ckpt} ...")
    from gem.network.hmr2 import load_hmr2

    model = load_hmr2(args.hmr2_ckpt).eval().cuda()
    wrapper = HMR2FeatureWrapper(model).eval().cuda()

    dummy = torch.randn(1, 3, 256, 256, device="cuda")
    feat = wrapper(dummy)
    print(f"[Export] Forward OK: f_imgseq={tuple(feat.shape)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        args.output,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["imgs"],
        output_names=["f_imgseq"],
        dynamic_axes={"imgs": {0: "B"}, "f_imgseq": {0: "B"}},
    )
    print(f"[Export] Wrote {args.output}")

    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(
            args.output, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        ort_out = sess.run(None, {"imgs": dummy.cpu().numpy()})[0]
        max_diff = np.abs(feat.cpu().numpy() - ort_out).max()
        print(f"[Export] ONNX validation OK: f_imgseq={ort_out.shape}, max_diff={max_diff:.6f}")
    except ImportError:
        print("[Export] onnxruntime not installed; skipping validation")


if __name__ == "__main__":
    main()
