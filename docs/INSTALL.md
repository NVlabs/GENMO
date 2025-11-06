# Install

## Environment
Setup python env:
```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/dair/projects/genmo.git --recursive
cd genmo

conda create -y -n genmo python=3.10
conda activate genmo
pip install -r requirements.txt
pip install open3d scenepic moviepy imageio einops dnspython dill librosa gdown colorama numpy==1.23.5 av==12.1.0
pip install git+https://github.com/google/aistplusplus_api.git
pip install git+https://github.com/facebookresearch/detectron2.git@a59f05630a8f205756064244bf5beb8661f96180
pip install -e .
pip install torch-scatter -f "https://data.pyg.org/whl/torch-2.3.0+cu121.html"

# DROID-SLAM
cd third-party/DROID-SLAM
export CUDA_HOME=/usr/local/cuda-12.1/
export PATH=$PATH:/usr/local/cuda-12.1/bin/
python setup.py install
```

You may need to install ffmpeg in your machine.
```bash
sudo apt-get install ffmpeg
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
