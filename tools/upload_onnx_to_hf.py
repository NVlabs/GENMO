# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upload exported ONNX models to the GEM-X HuggingFace repo.

Run this once after exporting the ONNX models locally; downstream users will
auto-download via ``gem.utils.hf_utils.download_onnx_model`` on first run of
``demo_webcam.py``.

Usage::

    huggingface-cli login           # one time
    python tools/upload_onnx_to_hf.py
    python tools/upload_onnx_to_hf.py --repo nvidia/GEM-X --dry-run
"""
# ruff: noqa: E402
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.utils.hf_utils import HF_REPO_ID, ONNX_HF_PREFIX

DEFAULT_ONNX_DIR = PROJECT_ROOT / "inputs" / "onnx"
DEFAULT_FILES = [
    "gem_smpl_denoiser.onnx",
    "gem_smpl_denoiser_no_imgfeat.onnx",
    "vitpose_coco17.onnx",
    "vitpose_coco17.onnx.data",
    "hmr2.onnx",
    "hmr2.onnx.data",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Upload GEM-SMPL ONNX models to HuggingFace")
    parser.add_argument("--repo", default=HF_REPO_ID, help="HF repo id")
    parser.add_argument("--prefix", default=ONNX_HF_PREFIX, help="Subdirectory inside the repo")
    parser.add_argument("--onnx_dir", default=str(DEFAULT_ONNX_DIR), help="Local ONNX dir")
    parser.add_argument(
        "--files", nargs="+", default=DEFAULT_FILES,
        help="Filenames inside --onnx_dir to upload (default: all 4 models + 2 .data sidecars)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List planned uploads without sending")
    parser.add_argument(
        "--commit_message", default="Upload GEM-SMPL ONNX models for real-time webcam demo",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from huggingface_hub import HfApi, whoami

    try:
        info = whoami()
        Log = lambda msg: print(f"[Upload] {msg}")
        Log(f"Authenticated as {info.get('name', '?')} ({info.get('email', '?')})")
    except Exception as e:
        sys.exit(f"[Upload] Not logged in to HuggingFace: {e}\n"
                 "  Run: huggingface-cli login")

    onnx_dir = Path(args.onnx_dir)
    if not onnx_dir.is_dir():
        sys.exit(f"[Upload] {onnx_dir} not found")

    api = HfApi()

    plan = []
    for filename in args.files:
        local = onnx_dir / filename
        if not local.exists():
            print(f"  SKIP    {filename} (not found in {onnx_dir})")
            continue
        size_gb = local.stat().st_size / 1e9
        path_in_repo = f"{args.prefix}/{filename}"
        plan.append((local, path_in_repo, size_gb))
        print(f"  UPLOAD  {filename:48} -> {args.repo}:{path_in_repo}  ({size_gb:.2f} GB)")

    if not plan:
        sys.exit("[Upload] Nothing to upload.")

    total_gb = sum(p[2] for p in plan)
    print(f"\n[Upload] Total: {len(plan)} files, {total_gb:.2f} GB")

    if args.dry_run:
        print("[Upload] --dry-run set; exiting without uploading.")
        return

    for local, path_in_repo, _ in plan:
        print(f"[Upload] Pushing {local.name} ...")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"{args.commit_message}: {local.name}",
        )
        print(f"[Upload] Done: {path_in_repo}")

    print(f"\n[Upload] All files pushed to https://huggingface.co/{args.repo}/tree/main/{args.prefix}/")


if __name__ == "__main__":
    main()
