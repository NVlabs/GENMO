# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared ONNX/TRT runner infrastructure for the GEM-SMPL demos.

Hosts:
- ``OnnxRunner`` — drop-in inference session with CUDA IOBinding fast-path
- ``TRTRunner``  — TensorRT engine runner (drop-in API)
- ``load_vitpose`` / ``load_denoiser`` / ``load_hmr2`` / ``load_yolox`` —
  prefer ONNX/TRT, fall back to PyTorch automatically
- ``vitpose_postprocess_coco17`` — argmax + subpixel decode shared by both
  the ONNX path and the PyTorch ``CocoPoseExtractor``
- ``vitpose_preprocess_256x192`` — affine crop helper

Mirrors the structure of ``gem/scripts/demo/demo_soma_fast.py`` but adapted
for SMPL: 17 COCO joints (not 77 SOMA), 151-D denoiser output, HMR2 features
(not SAM-3D-Body).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from gem.utils.pylogger import Log


_PREFER_QUANTIZED = sys.platform == "darwin" or not torch.cuda.is_available()
_IS_MACOS = sys.platform == "darwin"


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _torch_to_np_dtype(dtype):
    _map = {
        torch.float32: np.float32, torch.float16: np.float16,
        torch.int32: np.int32, torch.int64: np.int64,
        torch.float64: np.float64, torch.int8: np.int8,
    }
    return _map.get(dtype, np.float32)


# ──────────────────────────────────────────────────────────────────────
#  ONNX / TRT runners
# ──────────────────────────────────────────────────────────────────────


class OnnxRunner:
    """Generic ONNX Runtime inference session with GPU IOBinding support."""

    def __init__(self, onnx_path: str, device: str = "cuda"):
        import onnxruntime as ort

        providers = []
        if device == "cuda":
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
        if sys.platform == "darwin":
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")

        real_path = str(Path(onnx_path).resolve())

        # External-data ONNX models cannot use CoreML EP. Models without
        # external data benefit from full graph optimization on every EP.
        has_external_data = Path(real_path + ".data").exists()
        so = ort.SessionOptions()
        if _PREFER_QUANTIZED:
            _ncpu = os.cpu_count() or 4
            so.intra_op_num_threads = _ncpu
            so.inter_op_num_threads = max(1, _ncpu // 2)
        if has_external_data:
            providers = [p for p in providers if p != "CoreMLExecutionProvider"]
            _ort_ver = tuple(int(x) for x in ort.__version__.split(".")[:2])
            if _ort_ver >= (1, 17):
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            else:
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        else:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(real_path, sess_options=so, providers=providers)
        self.input_names = [inp.name for inp in self.sess.get_inputs()]
        self.output_names = [out.name for out in self.sess.get_outputs()]
        active_ep = self.sess.get_providers()
        self._use_cuda = "CUDAExecutionProvider" in active_ep
        Log.info(f"[ONNX] Loaded {onnx_path} (EP={active_ep}, inputs={self.input_names})")

    def __call__(self, **kwargs) -> dict:
        if self._use_cuda:
            return self._run_cuda(**kwargs)
        return self._run_cpu(**kwargs)

    def _run_cpu(self, **kwargs) -> dict:
        feed = {}
        for name in self.input_names:
            val = kwargs[name]
            feed[name] = val.cpu().numpy() if isinstance(val, torch.Tensor) else val
        ort_out = self.sess.run(self.output_names, feed)
        return {name: torch.from_numpy(arr) for name, arr in zip(self.output_names, ort_out)}

    def _run_cuda(self, **kwargs) -> dict:
        io = self.sess.io_binding()
        bound_inputs = []
        for name in self.input_names:
            val = kwargs[name]
            if isinstance(val, torch.Tensor):
                t = val.contiguous().cuda()
            else:
                t = torch.from_numpy(np.ascontiguousarray(val)).cuda()
            bound_inputs.append(t)
            io.bind_input(
                name, "cuda", 0, _torch_to_np_dtype(t.dtype),
                list(t.shape), t.data_ptr(),
            )

        for name in self.output_names:
            io.bind_output(name, "cuda", 0)

        self.sess.run_with_iobinding(io)
        del bound_inputs
        ort_outputs = io.get_outputs()
        result = {}
        for name, ort_val in zip(self.output_names, ort_outputs):
            arr = ort_val.numpy()
            result[name] = torch.from_numpy(arr)
        return result


class TRTRunner:
    """TensorRT engine runner (for models with fixed input/output names)."""

    def __init__(self, engine_path: str, input_names: list, output_shapes: dict, device: str = "cuda"):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.device = torch.device(device)
        self.input_names = input_names
        self.output_shapes = output_shapes
        Log.info(f"[TRT] Loaded {engine_path}")

    @torch.inference_mode()
    def __call__(self, **kwargs) -> dict:
        for name in self.input_names:
            tensor = kwargs[name].contiguous().to(self.device)
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        outputs = {}
        for name, shape_fn in self.output_shapes.items():
            shape = tuple(self.context.get_tensor_shape(name))
            if shape[0] <= 0:
                shape = shape_fn(kwargs)
            out = torch.empty(shape, device=self.device, dtype=torch.float32)
            self.context.set_tensor_address(name, int(out.data_ptr()))
            outputs[name] = out

        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)
        torch.cuda.current_stream().synchronize()
        return outputs


# ──────────────────────────────────────────────────────────────────────
#  Loaders with auto-fallback
# ──────────────────────────────────────────────────────────────────────


def _try_hf_download(name: str) -> Optional[str]:
    """Download an ONNX from HuggingFace. Returns local path on success, else None."""
    try:
        from gem.utils.hf_utils import download_onnx_model
        Log.info(f"[HF] Downloading ONNX model '{name}' from HuggingFace ...")
        return download_onnx_model(name)
    except Exception as e:
        Log.warn(f"[HF] Download failed for '{name}': {e}")
        return None


def load_vitpose(
    onnx_path: str = "inputs/onnx/vitpose_coco17.onnx",
    trt_path: str = "outputs/trt/vitpose_coco17.engine",
    ckpt_path: str = "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
):
    """Load 2D pose model (COCO-17). Prefer TRT > ONNX (auto-DL) > PyTorch fallback."""
    if os.path.exists(trt_path):
        try:
            return TRTRunner(
                trt_path,
                input_names=["imgs"],
                output_shapes={"heatmaps": lambda kw: (kw["imgs"].shape[0], 17, 64, 48)},
            ), "trt"
        except Exception as e:
            Log.warn(f"[TRT] VitPose load failed: {e}")

    if not os.path.exists(onnx_path):
        _try_hf_download(Path(onnx_path).stem)

    if os.path.exists(onnx_path):
        try:
            return OnnxRunner(onnx_path), "onnx"
        except Exception as e:
            Log.warn(f"[ONNX] VitPose load failed: {e}")

    Log.info("[VitPose] Falling back to PyTorch CocoPoseExtractor")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from demo_utils import CocoPoseExtractor
    return CocoPoseExtractor(device=_auto_device().split(":")[0]), "pytorch"


def load_denoiser(
    onnx_path: str = "inputs/onnx/gem_smpl_denoiser.onnx",
    trt_path: str = "outputs/trt/gem_smpl_denoiser.engine",
    no_imgfeat: bool = False,
):
    """Load GEM-SMPL denoiser. Prefer TRT > ONNX (auto-DL) > None.

    When *no_imgfeat* is True, the patched no-imgfeat variant is loaded.
    """
    if no_imgfeat:
        nif_onnx = onnx_path.replace(
            "gem_smpl_denoiser.onnx", "gem_smpl_denoiser_no_imgfeat.onnx"
        )
        if not os.path.exists(nif_onnx):
            _try_hf_download(Path(nif_onnx).stem)
        if os.path.exists(nif_onnx):
            try:
                return OnnxRunner(nif_onnx), "onnx"
            except Exception as e:
                Log.warn(f"[ONNX] No-imgfeat denoiser load failed: {e}")

    if os.path.exists(trt_path):
        try:
            return TRTRunner(
                trt_path,
                input_names=["obs", "bbx_xys", "K_fullimg", "f_imgseq", "f_cam_angvel"],
                output_shapes={
                    "pred_x": lambda kw: (kw["obs"].shape[0], kw["obs"].shape[1], 151),
                    "pred_cam": lambda kw: (kw["obs"].shape[0], kw["obs"].shape[1], 3),
                },
            ), "trt"
        except Exception as e:
            Log.warn(f"[TRT] Denoiser load failed: {e}")

    if not os.path.exists(onnx_path):
        _try_hf_download(Path(onnx_path).stem)

    if os.path.exists(onnx_path):
        try:
            return OnnxRunner(onnx_path), "onnx"
        except Exception as e:
            Log.warn(f"[ONNX] Denoiser load failed: {e}")

    return None, "pytorch"


def load_hmr2(
    onnx_path: str = "inputs/onnx/hmr2.onnx",
    trt_path: str = "outputs/trt/hmr2.engine",
    ckpt_path: str = "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
):
    """Load HMR2 ViT feature extractor. Prefer TRT > ONNX (auto-DL) > PyTorch."""
    if os.path.exists(trt_path):
        try:
            return TRTRunner(
                trt_path,
                input_names=["imgs"],
                output_shapes={"f_imgseq": lambda kw: (kw["imgs"].shape[0], 1024)},
            ), "trt"
        except Exception as e:
            Log.warn(f"[TRT] HMR2 load failed: {e}")

    if not os.path.exists(onnx_path):
        _try_hf_download(Path(onnx_path).stem)

    if os.path.exists(onnx_path):
        try:
            return OnnxRunner(onnx_path), "onnx"
        except Exception as e:
            Log.warn(f"[ONNX] HMR2 load failed: {e}")

    if os.path.exists(ckpt_path):
        Log.info("[HMR2] Falling back to PyTorch")
        from gem.network.hmr2 import load_hmr2 as _load_hmr2_pt
        model = _load_hmr2_pt(ckpt_path).to(_auto_device()).eval()
        return model, "pytorch"

    Log.warn(f"[HMR2] No checkpoint found at {ckpt_path}; image features unavailable")
    return None, "pytorch"


def load_yolox():
    """Load YOLOX detector + ByteTracker. Returns (detector, tracker, backend)."""
    from gem.utils.yolox_detector import ByteTracker, YOLOXDetector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLOXDetector(device=device)
    tracker = ByteTracker(max_lost=30)
    return detector, tracker, "onnx"


# ──────────────────────────────────────────────────────────────────────
#  VitPose preprocessing / postprocessing (COCO-17)
# ──────────────────────────────────────────────────────────────────────


_VITPOSE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_VITPOSE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_COCO17_FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16),
]


def vitpose_preprocess_256x192(frame_bgr: np.ndarray, bbx_xys_single: torch.Tensor) -> np.ndarray:
    """Affine-crop a single BGR frame to (3, 256, 192) ImageNet-normalized RGB."""
    cx = float(bbx_xys_single[0])
    cy = float(bbx_xys_single[1])
    s = float(bbx_xys_single[2])
    hs = s / 2
    src = np.array([[cx - hs, cy - hs], [cx + hs, cy - hs], [cx, cy]], dtype=np.float32)
    dst = np.array([[0, 0], [255, 0], [127.5, 127.5]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(frame_bgr, M, (256, 256), flags=cv2.INTER_LINEAR)
    crop = crop[:, 32:224, ::-1].astype(np.float32) / 255.0
    crop = (crop - _VITPOSE_MEAN) / _VITPOSE_STD
    return crop.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 256, 192)


def vitpose_postprocess_coco17(
    heatmaps: np.ndarray, bbx_xys: torch.Tensor
) -> torch.Tensor:
    """Argmax + subpixel refinement on COCO-17 heatmaps.

    Args:
        heatmaps: (N, 17, H, W) float32, ViTPose-H output (typically H=64, W=48)
        bbx_xys: (N, 3) tensor [cx, cy, size]

    Returns:
        kp2d: (N, 17, 3) tensor [x, y, confidence]
    """
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    idx = flat.argmax(-1)  # (N, K)
    px = (idx % W).astype(np.float32)
    py = (idx // W).astype(np.float32)
    # Subpixel refinement (loop is fine for N*K small)
    for n in range(N):
        for k in range(K):
            hm = heatmaps[n, k]
            x, y = int(px[n, k]), int(py[n, k])
            if 1 < x < W - 1:
                px[n, k] += np.sign(hm[y, x + 1] - hm[y, x - 1]) * 0.25
            if 1 < y < H - 1:
                py[n, k] += np.sign(hm[y + 1, x] - hm[y - 1, x]) * 0.25

    preds = np.stack([px, py], axis=-1)  # (N, K, 2)
    bbx_np = bbx_xys.cpu().numpy() if isinstance(bbx_xys, torch.Tensor) else np.asarray(bbx_xys)
    cx = bbx_np[:, 0:1]
    cy = bbx_np[:, 1:2]
    sz = bbx_np[:, 2:3]
    # The 256x192 crop covers (size, size*192/256) of the input image
    preds[..., 0] = preds[..., 0] / W * sz * (192.0 / 256.0) + (cx - sz * (192.0 / 256.0) / 2)
    preds[..., 1] = preds[..., 1] / H * sz + (cy - sz / 2)

    maxvals_raw = flat.max(-1)
    maxvals = (1.0 / (1.0 + np.exp(-maxvals_raw)))[..., np.newaxis]  # sigmoid → [0, 1]

    out = np.concatenate([preds, maxvals], axis=-1).astype(np.float32)
    return torch.from_numpy(out)


def flip_heatmap_coco17(hm: torch.Tensor) -> torch.Tensor:
    """Flip heatmaps horizontally and swap COCO-17 left/right joint channels."""
    hm = hm.flip(-1)
    out = hm.clone()
    for left, right in _COCO17_FLIP_PAIRS:
        out[:, left] = hm[:, right]
        out[:, right] = hm[:, left]
    return out


def run_vitpose_single_frame(
    runner, backend: str, frame_bgr: np.ndarray, bbx_xys_single: torch.Tensor,
    flip_test: bool = False,
) -> torch.Tensor:
    """Run ViTPose on one frame; returns (17, 3) tensor [x, y, conf].

    Used in the webcam demo's tight loop. Skips flip-test by default for speed.
    """
    if backend == "pytorch":
        # CocoPoseExtractor.extract expects (L, H, W, 3) RGB
        frames_rgb = frame_bgr[..., ::-1].copy()[None]
        return runner.extract(frames_rgb, bbx_xys_single.unsqueeze(0))[0]

    batch = vitpose_preprocess_256x192(frame_bgr, bbx_xys_single)
    if backend == "trt":
        batch = torch.from_numpy(batch).cuda()
    out = runner(imgs=batch)
    hm = out["heatmaps"]
    if isinstance(hm, torch.Tensor):
        hm = hm.cpu().numpy()
    if flip_test:
        batch_flip = batch[..., ::-1].copy() if isinstance(batch, np.ndarray) else batch.flip(-1)
        out_flip = runner(imgs=batch_flip)
        hm_flip = out_flip["heatmaps"]
        if isinstance(hm_flip, torch.Tensor):
            hm_flip = hm_flip.cpu().numpy()
        hm_flip = flip_heatmap_coco17(torch.from_numpy(hm_flip)).numpy()
        hm = (hm + hm_flip) * 0.5

    bbx_xys_t = bbx_xys_single.unsqueeze(0)
    kp2d = vitpose_postprocess_coco17(hm, bbx_xys_t)
    return kp2d[0]


# ──────────────────────────────────────────────────────────────────────
#  HMR2 single-frame helper
# ──────────────────────────────────────────────────────────────────────


def hmr2_preprocess_256x256(frame_rgb: np.ndarray, bbx_xys_single: torch.Tensor) -> np.ndarray:
    """Crop+resize a single RGB frame to (3, 256, 256) ImageNet-normalized.

    Mirrors gem-smpl/gem/network/hmr2/utils/preproc.py:crop_and_resize but in
    a streaming-friendly form (no Gaussian blur — fine for live feeds).
    """
    from gem.network.hmr2.utils.preproc import IMAGE_MEAN, IMAGE_STD, crop_and_resize

    cx = float(bbx_xys_single[0])
    cy = float(bbx_xys_single[1])
    s = float(bbx_xys_single[2])
    img, _ = crop_and_resize(frame_rgb, np.array([cx, cy], dtype=np.float32), s, 256, enlarge_ratio=1.0)
    img = img.astype(np.float32) / 255.0
    mean = IMAGE_MEAN.numpy() if isinstance(IMAGE_MEAN, torch.Tensor) else np.asarray(IMAGE_MEAN)
    std = IMAGE_STD.numpy() if isinstance(IMAGE_STD, torch.Tensor) else np.asarray(IMAGE_STD)
    img = (img - mean) / std
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def run_hmr2_single_frame(
    runner, backend: str, frame_rgb: np.ndarray, bbx_xys_single: torch.Tensor,
) -> torch.Tensor:
    """Run HMR2 on one frame; returns (1024,) tensor."""
    if runner is None:
        return torch.zeros(1024)

    if backend == "pytorch":
        batch = hmr2_preprocess_256x256(frame_rgb, bbx_xys_single)
        with torch.no_grad():
            t = torch.from_numpy(batch).to(next(runner.parameters()).device)
            feat = runner({"img": t})  # (1, 1024)
        return feat[0].detach().cpu()

    batch = hmr2_preprocess_256x256(frame_rgb, bbx_xys_single)
    if backend == "trt":
        batch = torch.from_numpy(batch).cuda()
    out = runner(imgs=batch)
    feat = out["f_imgseq"] if "f_imgseq" in out else out[list(out.keys())[0]]
    if isinstance(feat, torch.Tensor):
        return feat[0].cpu()
    return torch.from_numpy(feat[0])
