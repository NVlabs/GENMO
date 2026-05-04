# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-module latency benchmark for the GEM-SMPL real-time pipeline.

Measures each module in isolation with proper CUDA synchronization. Reports
mean / p50 / p95 / throughput so you can pinpoint bottlenecks without the
async pipeline overlap of demo_webcam.py.

Usage::

    python tools/benchmark/benchmark_modules.py
    python tools/benchmark/benchmark_modules.py --no_imgfeat --iters 200
    python tools/benchmark/benchmark_modules.py --modules denoiser vitpose
"""
# ruff: noqa: E402
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "demo"))

from gem.utils.cam_utils import estimate_K
from gem.utils.geo_transform import compute_cam_angvel
from gem.utils.motion_utils import init_rollout_w_Rt_state, rollout_step_w_Rt
from onnx_runners import (
    load_denoiser, load_hmr2, load_vitpose, load_yolox,
    run_hmr2_single_frame, run_vitpose_single_frame,
)


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_USE_CUDA = _DEVICE.type == "cuda"


def time_block(fn, iters: int, warmup: int):
    """Run ``fn`` ``iters`` times after ``warmup`` warmup iterations.

    Returns dict {mean_ms, p50_ms, p95_ms, throughput_hz}.
    Uses cuda.synchronize() around each timed iteration on GPU; pure perf_counter on CPU.
    """
    for _ in range(warmup):
        fn()
    if _USE_CUDA:
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        if _USE_CUDA:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if _USE_CUDA:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(samples)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "throughput_hz": 1000.0 / float(arr.mean()),
        "n": iters,
    }


def fmt(name, stats, backend=""):
    bk = f" [{backend}]" if backend else ""
    return (f"  {name + bk:<28} "
            f"mean={stats['mean_ms']:7.2f}ms  "
            f"p50={stats['p50_ms']:7.2f}ms  "
            f"p95={stats['p95_ms']:7.2f}ms  "
            f"~{stats['throughput_hz']:6.1f} Hz")


# ---- Per-module benchmarks ------------------------------------------------


def bench_yolox(args):
    detector, _tracker, backend = load_yolox()
    H, W = args.height, args.width
    img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    return time_block(lambda: detector.detect(img), args.iters, args.warmup), backend


def bench_vitpose(args):
    runner, backend = load_vitpose()
    if runner is None:
        return None, "missing"
    frame = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    bbx = torch.tensor([args.width / 2, args.height / 2, 256.0])
    fn = lambda: run_vitpose_single_frame(runner, backend, frame, bbx, flip_test=False)
    return time_block(fn, args.iters, args.warmup), backend


def bench_hmr2(args):
    runner, backend = load_hmr2()
    if runner is None:
        return None, "missing"
    frame = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    bbx = torch.tensor([args.width / 2, args.height / 2, 256.0])
    fn = lambda: run_hmr2_single_frame(runner, backend, frame, bbx)
    return time_block(fn, args.iters, args.warmup), backend


def bench_denoiser(args):
    runner, backend = load_denoiser(no_imgfeat=args.no_imgfeat)
    if runner is None:
        return None, "missing"
    L = args.context_frames
    K = estimate_K(args.width, args.height)
    cam_angvel = compute_cam_angvel(torch.eye(3).unsqueeze(0).repeat(L, 1, 1)).unsqueeze(0)

    batch = {
        "obs": torch.randn(1, L, 17, 3),
        "bbx_xys": torch.tensor([args.width / 2, args.height / 2, 256.0]).expand(1, L, 3).contiguous(),
        "K_fullimg": K.unsqueeze(0).unsqueeze(0).expand(1, L, 3, 3).contiguous(),
        "f_imgseq": torch.zeros(1, L, 1024) if args.no_imgfeat else torch.randn(1, L, 1024),
        "f_cam_angvel": cam_angvel,
    }
    fn = lambda: runner(**batch)
    return time_block(fn, args.iters, args.warmup), backend


def bench_endecoder_decode(args):
    from gem.network.endecoder import EnDecoder
    enc = EnDecoder(
        stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM",
        encode_type="gvhmr", feat_dim=151, clip_std=True,
    ).eval().to(_DEVICE)
    enc.build_obs_indices_dict()
    pred_x = torch.randn(1, args.context_frames, 151, device=_DEVICE)
    fn = lambda: enc.decode(pred_x)
    return time_block(fn, args.iters, args.warmup), "pytorch"


def bench_rollout_step(args):
    gv0 = torch.zeros(3, device=_DEVICE)
    gc0 = torch.zeros(3, device=_DEVICE)
    state = init_rollout_w_Rt_state(gv0, gc0)
    cam = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=_DEVICE)
    lv = torch.zeros(3, device=_DEVICE)

    def fn():
        nonlocal state
        _, state = rollout_step_w_Rt(
            state, gv0, gc0,
            cam_angvel_prev=cam, local_transl_vel_prev=lv,
        )
    return time_block(fn, args.iters, args.warmup), "pytorch"


_REGISTRY = {
    "yolox": bench_yolox,
    "vitpose": bench_vitpose,
    "hmr2": bench_hmr2,
    "denoiser": bench_denoiser,
    "decode": bench_endecoder_decode,
    "rollout": bench_rollout_step,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Per-module latency benchmark for GEM-SMPL")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--width", type=int, default=1280, help="Frame width for image inputs")
    parser.add_argument("--height", type=int, default=720, help="Frame height for image inputs")
    parser.add_argument("--context_frames", type=int, default=120, help="Denoiser seq_len")
    parser.add_argument("--no_imgfeat", action="store_true",
                        help="Use no-imgfeat denoiser variant (zeros for f_imgseq)")
    parser.add_argument("--modules", nargs="+", default=list(_REGISTRY.keys()),
                        choices=list(_REGISTRY.keys()),
                        help="Subset of modules to benchmark")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n[Bench] device={_DEVICE}, iters={args.iters}, warmup={args.warmup}, "
          f"input={args.width}x{args.height}, ctx={args.context_frames}, "
          f"no_imgfeat={args.no_imgfeat}\n")

    results = {}
    for name in args.modules:
        try:
            stats, backend = _REGISTRY[name](args)
            if stats is None:
                print(f"  {name:<28} [MISSING] — checkpoint or ONNX file not found")
                continue
            results[name] = (stats, backend)
            print(fmt(name, stats, backend))
        except Exception as e:
            print(f"  {name:<28} [FAILED] {e!r}")

    # Summary: estimated end-to-end with no_imgfeat path (yolox/N + vitpose + denoiser + decode + rollout)
    if results:
        print()
        print("─" * 75)
        # Per-frame compute cost in steady state (yolox amortized over yolo_period=5)
        sum_ms = 0.0
        if "yolox" in results: sum_ms += results["yolox"][0]["mean_ms"] / 5  # amortized
        if "vitpose" in results: sum_ms += results["vitpose"][0]["mean_ms"]
        if not args.no_imgfeat and "hmr2" in results:
            sum_ms += results["hmr2"][0]["mean_ms"]
        if "denoiser" in results: sum_ms += results["denoiser"][0]["mean_ms"]
        if "decode" in results: sum_ms += results["decode"][0]["mean_ms"]
        if "rollout" in results: sum_ms += results["rollout"][0]["mean_ms"]
        print(f"  Estimated steady-state per-frame (synchronous): "
              f"{sum_ms:.2f}ms  ~{1000.0/sum_ms:.1f} FPS")
        print("  (yolox amortized over yolo_period=5; async pipeline can overlap further)")
    print()


if __name__ == "__main__":
    main()
