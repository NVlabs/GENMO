# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the ViTPose-H COCO-17 backbone + heatmap head to ONNX.

The exported ONNX model takes ImageNet-normalized 256x192 RGB crops and
produces 17-channel heatmaps. Used by the real-time webcam demo for fast
2D keypoint extraction.

Usage::

    cd /path/to/gem-smpl
    python tools/export/export_vitpose_onnx.py \
        --ckpt inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth \
        --output inputs/onnx/vitpose_coco17.onnx
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

torch.load = functools.partial(torch.load, weights_only=False)

from gem.utils.vitpose_model import build_vitpose_coco17


def parse_args():
    parser = argparse.ArgumentParser(description="Export ViTPose-H COCO-17 to ONNX")
    parser.add_argument("--ckpt", type=str,
                        default="inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth")
    parser.add_argument("--output", type=str, default="inputs/onnx/vitpose_coco17.onnx")
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


@torch.no_grad()
def main():
    import tempfile

    args = parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit(f"[Export] ViTPose checkpoint not found: {args.ckpt}")

    print(f"[Export] Loading ViTPose-H COCO-17 from {args.ckpt} ...")
    model = build_vitpose_coco17(args.ckpt).eval().cuda()

    dummy = torch.randn(1, 3, 256, 192, device="cuda")
    heatmaps = model(dummy)
    print(f"[Export] Forward OK: heatmaps={tuple(heatmaps.shape)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Export to a tempdir first; ViT-H is >2GB so torch.onnx.export emits many
    # per-tensor external-data files. We then load and re-save with a single
    # .onnx.data sidecar so the model is clean on disk and easy to upload.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = str(Path(tmpdir) / "model.onnx")
        torch.onnx.export(
            model, (dummy,), tmp_path,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["imgs"],
            output_names=["heatmaps"],
            dynamic_axes={"imgs": {0: "B"}, "heatmaps": {0: "B"}},
        )

        import onnx
        loaded = onnx.load(tmp_path, load_external_data=True)
        data_filename = Path(args.output).stem + ".onnx.data"
        onnx.save_model(
            loaded, args.output,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=1024,
            convert_attribute=False,
        )
    print(f"[Export] Wrote {args.output} (+ {Path(args.output).name}.data)")

    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(
            args.output, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        ort_out = sess.run(None, {"imgs": dummy.cpu().numpy()})[0]
        max_diff = np.abs(heatmaps.cpu().numpy() - ort_out).max()
        print(f"[Export] ONNX validation OK: heatmaps={ort_out.shape}, max_diff={max_diff:.6f}")
    except ImportError:
        print("[Export] onnxruntime not installed; skipping validation")


if __name__ == "__main__":
    main()
