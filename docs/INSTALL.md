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

## Step 8 — Verify installation

```bash
python -c "import gem; print('Installation successful')"
```
