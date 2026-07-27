# Dataset Preparation

GEM uses preprocessed datasets from [GVHMR](https://github.com/zju3dv/GVHMR). Download the `*_hmr4d_support.tar.gz` archives from [GVHMR's Google Drive](https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link) and extract them under `inputs/`:

```
inputs/
├── AMASS/hmr4d_support/
├── BEDLAM/hmr4d_support/
├── H36M/hmr4d_support/
├── 3DPW/hmr4d_support/
├── EMDB/hmr4d_support/
└── RICH/hmr4d_support/
```

> By downloading these files you agree to the original dataset licenses. The preprocessed data is for **research use only**.

## Evaluation Datasets

Required for running `task=test`:

| Dataset | Archive | Config |
|---------|---------|--------|
| **EMDB** | `EMDB_hmr4d_support.tar.gz` | `configs/test_datasets/emdb1_v1_fliptest.yaml` |
| **3DPW** | `3DPW_hmr4d_support.tar.gz` | `configs/test_datasets/3dpw_fliptest.yaml` |
| **RICH** | `RICH_hmr4d_support.tar.gz` | `configs/test_datasets/rich_all.yaml` |

### Supplemental EMDB / RICH / 3DPW Test Files

Some GEM-SMPL test dataloaders also require supplemental files that may be
missing from the GVHMR archives:

```
inputs/EMDB/hmr4d_support/emdb_vimo.pt
inputs/EMDB/hmr4d_support/emdb_slam_traj.pt
inputs/RICH/hmr4d_support/rich_test_vimo_preproc.pt
inputs/3DPW/hmr4d_support/test_3dpw_vimo_labels.pt
inputs/3DPW/hmr4d_support/3dpw_test_slam_traj.pt
```

Download them from the GEM-X HuggingFace repository and merge the included
`inputs/` directory into the repository root:

```bash
huggingface-cli download nvidia/GEM-X \
  --include "gem_smpl/missing_hmr4d_support/**" \
  --local-dir .

mkdir -p inputs
cp -a gem_smpl/missing_hmr4d_support/inputs/. inputs/
```

## Training Datasets

Required for training `gem_smpl_regression`:

| Dataset | Archive | Config |
|---------|---------|--------|
| **AMASS** | `AMASS_hmr4d_support.tar.gz` | `configs/train_datasets/amass_v11.yaml` |
| **BEDLAM** | `BEDLAM_hmr4d_support.tar.gz` | `configs/train_datasets/bedlam_v2.yaml` |
| **Human3.6M** | `H36M_hmr4d_support.tar.gz` | `configs/train_datasets/h36m_v1.yaml` |
| **3DPW** | `3DPW_hmr4d_support.tar.gz` (train split) | `configs/train_datasets/3dpw_v1.yaml` |

## Additional Datasets

Required for the full model `gem_smpl` (regression + text/audio generation):

| Dataset | Source |
|---------|--------|
| **AIST++** | [https://google.github.io/aistplusplus_dataset/](https://google.github.io/aistplusplus_dataset/) — dance motion |
| **Beat2** | [https://pantomatrix.github.io/BEAT/](https://pantomatrix.github.io/BEAT/) — gesture dataset |
| **HumanML3D** | [https://github.com/EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D) — text-to-motion |
