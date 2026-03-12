# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Video-to-SMPL inference pipeline (4 stages).

Usage:
    python scripts/demo/demo_smpl_hpe.py --video /path/to/video.mp4 --ckpt_path /path/to/checkpoint.ckpt

Stages:
    1. Preprocessing  - Person detection/tracking (YOLO), 2D keypoints (ViTPose-H COCO17),
                        image features (HMR2 ViT), camera (static or SLAM).
    2. Data Assembly  - Build data dict for GEM.predict().
    3. Model Inference - Load GEM model, run prediction.
    4. Visualization   - In-camera render, global render, side-by-side merge.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from demo_utils import (
    CocoPoseExtractor,
    detect_and_track,
    get_camera_static,
    get_image_features,
    load_model,
    normalize_global_verts,
    render_2d_keypoints,
    render_global_frames,
    render_incam_frames,
    render_side_by_side,
    run_inference,
)

# ===========================================================================
# Stage 2: Assemble data dict
# ===========================================================================


def assemble_data(
    kp2d: torch.Tensor,  # (L, 17, 3)
    bbx_xys: torch.Tensor,  # (L, 3)
    K_fullimg: torch.Tensor,  # (L, 3, 3)
    cam_angvel: torch.Tensor,  # (L, 6)
    cam_tvel: torch.Tensor,  # (L, 3)
    R_w2c: torch.Tensor,  # (L, 3, 3)
    f_imgseq: torch.Tensor,  # (L, 1024)
    has_img_mask: torch.Tensor,  # (L,) bool
    static_cam: bool,
) -> dict:
    """Build the data dict matching GEM.predict() expectations."""
    from gem.utils.net_utils import get_valid_mask

    L = kp2d.shape[0]

    cam_valid_len = L if not static_cam else 0

    data = {
        "kp2d": kp2d,
        "bbx_xys": bbx_xys,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "cam_tvel": cam_tvel,
        "R_w2c": R_w2c,
        "f_imgseq": f_imgseq,
        "has_text": torch.tensor([False]),
        "mask": {
            "has_img_mask": has_img_mask,
            "has_2d_mask": get_valid_mask(L, L),
            "has_cam_mask": get_valid_mask(L, cam_valid_len),
            "has_audio_mask": get_valid_mask(L, 0),
            "has_music_mask": get_valid_mask(L, 0),
        },
        "length": torch.tensor(L),
        "meta": [{"mode": "default"}],
    }
    return data


# ===========================================================================
# Stage 4: Visualization
# ===========================================================================


def render_results(
    pred: dict,
    video_path: str,
    output_dir: str,
    no_render: bool = False,
    kp2d: torch.Tensor | None = None,
    bbx_xys: torch.Tensor | None = None,
):
    """Render in-camera overlay, global view, and side-by-side video."""
    os.makedirs(output_dir, exist_ok=True)

    # Save SMPL parameters to disk
    params_path = os.path.join(output_dir, "smpl_params.pt")
    save_dict = {}
    for key_group in ("body_params_incam", "body_params_global"):
        if key_group in pred:
            save_dict[key_group] = {k: v.cpu() for k, v in pred[key_group].items()}
    if "K_fullimg" in pred:
        save_dict["K_fullimg"] = pred["K_fullimg"].cpu()
    torch.save(save_dict, params_path)
    print(f"[Stage 4] SMPL parameters saved to {params_path}")

    # Render 2D keypoint overlay (always, even with --no_render)
    if kp2d is not None and bbx_xys is not None:
        render_2d_keypoints(video_path, kp2d, bbx_xys, output_dir)

    if no_render:
        print("[Stage 4] Skipping rendering (--no_render)")
        return

    # ---- Rendering ----
    try:
        from gem.utils.smplx_utils import make_smplx
        from gem.utils.video_io_utils import (
            get_video_lwh,
            read_video_np,
            save_video,
        )
    except ImportError as e:
        print(f"[Stage 4] Rendering dependencies not available: {e}")
        print("[Stage 4] Skipping rendering. SMPL parameters were saved.")
        return

    _, W, H = get_video_lwh(video_path)
    frames = read_video_np(video_path)

    # Create full SMPL body model for rendering (full vertices)
    body_model = make_smplx("supermotion")
    body_model.cuda().eval()
    smpl_faces = torch.from_numpy(body_model.faces.astype(np.int32)).long()

    # ---- In-camera rendering ----
    print("[Stage 4] Rendering in-camera overlay ...")
    bp_incam = pred["body_params_incam"]
    K_fullimg = pred["K_fullimg"]

    with torch.no_grad():
        smpl_out = body_model(
            body_pose=bp_incam["body_pose"].cuda(),
            global_orient=bp_incam["global_orient"].cuda(),
            transl=bp_incam["transl"].cuda(),
            betas=bp_incam.get("betas", torch.zeros(bp_incam["body_pose"].shape[0], 10)).cuda(),
        )
    verts_incam = smpl_out.vertices.cpu()  # (L, V, 3)

    K_render = K_fullimg[0]  # use first frame intrinsics (3, 3)
    incam_frames = render_incam_frames(frames, verts_incam, smpl_faces, K_render, W, H)

    incam_path = os.path.join(output_dir, "1_incam.mp4")
    save_video(incam_frames, incam_path, fps=30)
    print(f"[Stage 4] In-camera video saved to {incam_path}")

    # ---- Global rendering ----
    print("[Stage 4] Rendering global view ...")
    bp_global = pred.get("body_params_global")
    if bp_global is None:
        print("[Stage 4] No global body params available; skipping global render.")
        return

    verts_global = normalize_global_verts(body_model, bp_global)
    global_frames = render_global_frames(verts_global, smpl_faces, W, H)

    global_path = os.path.join(output_dir, "2_global.mp4")
    save_video(global_frames, global_path, fps=30)
    print(f"[Stage 4] Global video saved to {global_path}")

    # ---- Side-by-side ----
    print("[Stage 4] Creating side-by-side video ...")
    sbs_frames = render_side_by_side(incam_frames, global_frames)

    sbs_path = os.path.join(output_dir, "3_incam_global_horiz.mp4")
    save_video(sbs_frames, sbs_path, fps=30)
    print(f"[Stage 4] Side-by-side video saved to {sbs_path}")

    # Force-exit to avoid Open3D Filament VertexBuffer double-free crash during cleanup
    os._exit(0)


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="GEM-SMPL: Video-to-SMPL inference")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Pretrained checkpoint path (auto-downloads from HuggingFace if not provided)",
    )
    parser.add_argument(
        "--output_root",
        default="outputs/demo_smpl",
        help="Output directory (default: outputs/demo_smpl)",
    )
    parser.add_argument(
        "-s",
        "--static_cam",
        action="store_true",
        help="Assume static camera (disables camera motion estimation)",
    )
    parser.add_argument(
        "--hmr2_ckpt",
        default="inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
        help="Path to HMR2 checkpoint for image feature extraction",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Skip video rendering, only save SMPL parameters",
    )
    args = parser.parse_args()

    video_path = args.video
    assert os.path.exists(video_path), f"Video not found: {video_path}"

    video_name = Path(video_path).stem
    output_dir = os.path.join(args.output_root, video_name)
    preprocess_dir = os.path.join(output_dir, "preprocess")
    os.makedirs(preprocess_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GEM-SMPL Demo Pipeline")
    print(f"  Video:      {video_path}")
    print(f"  Checkpoint: {args.ckpt_path}")
    print(f"  HMR2 ckpt:  {args.hmr2_ckpt}")
    print(f"  Output:     {output_dir}")
    print(f"  Static cam: {args.static_cam}")
    print("=" * 60)

    t0 = time.time()

    # ------------------------------------------------------------------
    # Stage 1: Preprocessing
    # ------------------------------------------------------------------
    from gem.utils.video_io_utils import get_video_lwh, read_video_np

    L, W, H = get_video_lwh(video_path)
    print(f"\n[Info] Video: {L} frames, {W}x{H}")

    # 1a. Person detection + tracking
    bbx_xys = detect_and_track(video_path, preprocess_dir)

    # 1b. 2D Keypoints (COCO-17)
    vitpose_cache = os.path.join(preprocess_dir, "vitpose.pt")
    if os.path.exists(vitpose_cache):
        print(f"[Stage 1b] Loading cached keypoints from {vitpose_cache}")
        kp2d = torch.load(vitpose_cache, map_location="cpu")
    else:
        print("[Stage 1b] Extracting COCO-17 keypoints ...")
        frames = read_video_np(video_path)
        extractor = CocoPoseExtractor(device="cuda")
        kp2d = extractor.extract(frames, bbx_xys, batch_size=32)
        torch.save(kp2d, vitpose_cache)
        print(f"[Stage 1b] Keypoints saved to {vitpose_cache}")
        del frames  # free memory

    # 1c. Image features (HMR2 ViT)
    print("[Stage 1c] Image features: HMR2 ViT")
    f_imgseq, has_img_mask = get_image_features(video_path, bbx_xys, args.hmr2_ckpt, preprocess_dir)

    # 1d. Camera
    print(f"[Stage 1d] Camera: {'static' if args.static_cam else 'static (default)'}")
    R_w2c, cam_angvel, cam_tvel, K_fullimg = get_camera_static(L, W, H)

    print(f"\n[Stage 1] Preprocessing complete ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # Stage 2: Assemble data dict
    # ------------------------------------------------------------------
    print("\n[Stage 2] Assembling data dict ...")
    data = assemble_data(
        kp2d=kp2d,
        bbx_xys=bbx_xys,
        K_fullimg=K_fullimg,
        cam_angvel=cam_angvel,
        cam_tvel=cam_tvel,
        R_w2c=R_w2c,
        f_imgseq=f_imgseq,
        has_img_mask=has_img_mask,
        static_cam=args.static_cam,
    )

    # ------------------------------------------------------------------
    # Stage 3: Model Inference
    # ------------------------------------------------------------------
    print("\n[Stage 3] Loading model and running inference ...")
    ckpt_path = args.ckpt_path
    if ckpt_path is None or not Path(ckpt_path).exists():
        from gem.utils.hf_utils import download_checkpoint

        print("[Checkpoint] Not found locally. Downloading from HuggingFace...")
        ckpt_path = download_checkpoint()
    model = load_model(ckpt_path)
    pred = run_inference(model, data, static_cam=args.static_cam)
    print(f"[Stage 3] Inference complete ({time.time() - t0:.1f}s total)")

    # ------------------------------------------------------------------
    # Stage 4: Visualization
    # ------------------------------------------------------------------
    print("\n[Stage 4] Visualization ...")
    render_results(
        pred=pred,
        video_path=video_path,
        output_dir=output_dir,
        no_render=args.no_render,
        kp2d=kp2d,
        bbx_xys=bbx_xys,
    )

    print(f"\n{'=' * 60}")
    print(f"Done! Total time: {time.time() - t0:.1f}s")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
