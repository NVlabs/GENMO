# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

HF_REPO_ID = "nvidia/GEM-X"
DEFAULT_CKPT_FILENAME = "gem_smpl.ckpt"
DEFAULT_CKPT_DIR = "inputs/pretrained"


def download_checkpoint(
    repo_id=HF_REPO_ID, filename=DEFAULT_CKPT_FILENAME, local_dir=DEFAULT_CKPT_DIR
):
    """Download checkpoint from HuggingFace Hub if not already cached."""
    local_path = Path(local_dir) / filename
    if local_path.exists():
        return str(local_path)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    return path
