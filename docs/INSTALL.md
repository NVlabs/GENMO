# Install

## Environment
Setup python env:
```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/dair/projects/genmo.git --recursive
cd genmo

conda create -y -n genmo python=3.10
conda activate genmo
pip install -r requirements.txt
pip install -e .

# DROID-SLAM
cd third-party/DROID-SLAM
export CUDA_HOME=/usr/local/cuda-12.1/
export PATH=$PATH:/usr/local/cuda-12.1/bin/
python setup.py install
```

Setup pre-commit for code formatting:
```
pip install pre-commit
pre-commit install
```

## Inputs & Outputs

**Prepare pretrained models**

```bash
mkdir inputs
rsync -avzP -m cs-oci-ord-dc-03:/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/GVHMR/smpl_data ./inputs/
rsync -avzP -m cs-oci-ord-dc-03:/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/GVHMR/checkpoints ./inputs/
```
<!-- 
**Weights**

```bash
inputs/checkpoints/
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt
``` -->

**Training Data**

We provide preprocessed data for training and evaluation.
Note that we do not intend to distribute the original datasets, and you need to download them (annotation, videos, etc.) from the original websites.
*We're unable to provide the original data due to the license restrictions.*
By downloading the preprocessed data, you agree to the original dataset's terms of use and use the data for research purposes only.

You can download them from [Google-Drive](https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link). Please place them in the "inputs" folder and execute the following commands:

```bash
cd inputs
# Train
tar -xzvf AMASS_hmr4d_support.tar.gz
tar -xzvf BEDLAM_hmr4d_support.tar.gz
tar -xzvf H36M_hmr4d_support.tar.gz
# Test
tar -xzvf 3DPW_hmr4d_support.tar.gz
tar -xzvf EMDB_hmr4d_support.tar.gz
tar -xzvf RICH_hmr4d_support.tar.gz

# The folder structure should be like this:
inputs/
├── AMASS/hmr4d_support/
├── BEDLAM/hmr4d_support/
├── H36M/hmr4d_support/
├── 3DPW/hmr4d_support/
├── EMDB/hmr4d_support/
└── RICH/hmr4d_support/
```
