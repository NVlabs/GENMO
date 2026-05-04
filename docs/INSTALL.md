# Installation Guide

## Step 1 — Clone the repository

```bash
git clone https://github.com/NVlabs/GENMO.git
cd GENMO
```

## Step 2 — Create virtual environment with uv

```bash
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
```

## Step 3 — Install PyTorch with CUDA

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Step 4 — Install GEM-SMPL and dependencies

```bash
bash scripts/install_env.sh
```

## Step 5 — Download SMPLX body model

1. Register and download **SMPLX_NEUTRAL.npz** from [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)
2. Place the file at:
   ```
   inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
   ```

## Prerequisites for Demo

The following steps are only required if you plan to run the demo. They are **not needed** for training.

### Step 6 — Download HMR2 checkpoint

The demo uses [HMR2](https://github.com/shubham-goel/4D-Humans) for image feature extraction. Download the checkpoint from [GVHMR's Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) and place it at:

```
inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt
```

### Step 7 — Download ViTPose checkpoint

The demo uses [ViTPose-H](https://github.com/ViTAE-Transformer/ViTPose) for 2D keypoint extraction. Download `vitpose-h-multi-coco.pth` from [GVHMR's Google Drive](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD) and place it at:

```
inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth
```

## Prerequisites for Real-Time Webcam Demo

The following steps are only required for `scripts/demo/demo_webcam.py` (real-time inference via ONNX). The offline video demos (`demo_smpl.py`, `demo_smpl_hpe.py`) do not need them.

### Step 8 — Install ONNX Runtime

For CUDA GPU:

```bash
uv pip install onnxruntime-gpu nvidia-cudnn-cu12
```

For CPU only:

```bash
uv pip install onnxruntime
```

Verify the CUDA execution provider is available:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include 'CUDAExecutionProvider'
```

### Step 9 — (Optional) Install Viser for global 3D rendering

Required only for `--render_mode viser` (web-based 3D world view). The OpenCV mesh-overlay mode (`--render_mode opencv`) does not need this.

```bash
uv pip install viser
```

### Step 10 — Export ONNX models

The webcam demo runs detection, 2D pose, image features, and the GEM denoiser via ONNX Runtime. Three of the four ONNX files are exported from local checkpoints; YOLOX is auto-downloaded from the OpenMMLab CDN on first use.

```bash
# GEM-SMPL denoiser — both with-imgfeat and no-imgfeat variants
python tools/export/export_denoiser_onnx.py --ckpt <path/to/gem_smpl.ckpt> --exp gem_smpl
python tools/export/export_denoiser_onnx.py --ckpt <path/to/gem_smpl.ckpt> --exp gem_smpl --no-imgfeat

# ViTPose-H COCO-17
python tools/export/export_vitpose_onnx.py

# HMR2 ViT (skip if you only ever use --no_imgfeat)
python tools/export/export_hmr2_onnx.py
```

Outputs land in `inputs/onnx/`:

| File | Size | Used when |
|---|---|---|
| `gem_smpl_denoiser.onnx` (+ `.data`) | ~1.7 GB | default mode |
| `gem_smpl_denoiser_no_imgfeat.onnx` (+ `.data`) | ~1.7 GB | `--no_imgfeat` |
| `vitpose_coco17.onnx` | ~2.4 GB | always |
| `hmr2.onnx` | ~2.4 GB | default mode |

**Note**: each denoiser ONNX bakes in `seq_len=120` from the export trace; `demo_webcam.py --context_frames` must match this value. Re-export with `--seq_len <N>` to change.

## Step 11 — Verify installation

```bash
python -c "import gem; print('Installation successful')"
```

Optional: benchmark per-module latency to confirm GPU EP is active and tune expectations:

```bash
python tools/benchmark/benchmark_modules.py --no_imgfeat
```
