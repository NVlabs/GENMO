# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-modal GEM-SMPL demo: mixed video + text conditioning.

Usage:
    # Video + inline text:
    python scripts/demo/demo_smpl.py \\
        --input_list video1.mp4 "text:a person walks forward" video2.mp4 \\
        --ckpt_path path/to/gem_smpl.ckpt

    # Video + text file:
    python scripts/demo/demo_smpl.py \\
        --input_list video1.mp4 prompt.txt video2.mp4 \\
        --ckpt_path path/to/gem_smpl.ckpt

    # With options:
    python scripts/demo/demo_smpl.py \\
        --input_list v1.mp4 "text:dancing" v2.mp4 \\
        --ckpt_path ckpt.ckpt --text_length 300 --static_cam --output_root outputs

    # More than one text and video:
    python scripts/demo/demo_smpl.py \\
        --input_list video1.mp4 "text:a person walks forward" video2.mp4 "text:a person dances" \\
        --ckpt_path path/to/gem_smpl.ckpt
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from demo_utils import (
    CocoPoseExtractor,
    create_text_frames,
    detect_and_track,
    get_camera_static,
    get_image_features,
    load_model,
    normalize_global_verts,
    render_global_frames,
    render_incam_frames,
    render_side_by_side,
    run_inference,
)

# ===========================================================================
# Input parsing
# ===========================================================================


def parse_input_list(input_list: list[str]) -> list[dict]:
    """Classify each input item as video or text.

    Returns:
        List of segment dicts with keys: type, path/caption, name
    """
    VIDEO_EXTS = {".mp4", ".avi", ".mov"}
    segments = []
    text_counter = 0

    for item in input_list:
        ext = Path(item).suffix.lower() if not item.startswith("text:") else ""

        if ext in VIDEO_EXTS:
            segments.append(
                {
                    "type": "video",
                    "path": str(Path(item).resolve()),
                    "name": Path(item).stem,
                }
            )
        elif item.startswith("text:"):
            caption = item[len("text:") :]
            segments.append(
                {
                    "type": "text",
                    "caption": caption.strip(),
                    "name": f"text_{text_counter}",
                }
            )
            text_counter += 1
        elif ext == ".txt":
            with open(item) as f:
                caption = f.read().strip()
            segments.append(
                {
                    "type": "text",
                    "caption": caption,
                    "name": Path(item).stem,
                }
            )
        else:
            raise ValueError(
                f"Unrecognized input: {item}. "
                "Expected .mp4/.avi/.mov, .txt, or 'text:...' prefix."
            )

    # Validate: at least one video
    has_video = any(s["type"] == "video" for s in segments)
    if not has_video:
        raise ValueError(
            "At least one video input is required (for reference resolution & intrinsics)."
        )

    return segments


# ===========================================================================
# Preprocessing
# ===========================================================================


def preprocess_video_segment(
    video_path: str,
    preprocess_dir: str,
    hmr2_ckpt: str,
    static_cam: bool,
) -> dict:
    """Run Stage 1 preprocessing on one video segment.

    Returns:
        Segment dict with all feature tensors and metadata.
    """
    from gem.utils.video_io_utils import get_video_lwh, read_video_np

    L, W, H = get_video_lwh(video_path)

    # 1a. Detection + tracking
    bbx_xys = detect_and_track(video_path, preprocess_dir)

    # 1b. 2D Keypoints
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
        del frames

    # 1c. Image features
    f_imgseq, has_img_mask = get_image_features(video_path, bbx_xys, hmr2_ckpt, preprocess_dir)

    # 1d. Camera
    R_w2c, cam_angvel, cam_tvel, K_fullimg = get_camera_static(L, W, H)

    return {
        "type": "video",
        "length": L,
        "width": W,
        "height": H,
        "bbx_xys": bbx_xys,
        "kp2d": kp2d,
        "f_imgseq": f_imgseq,
        "has_img_mask": has_img_mask,
        "R_w2c": R_w2c,
        "cam_angvel": cam_angvel,
        "cam_tvel": cam_tvel,
        "K_fullimg": K_fullimg,
    }


def create_text_segment(
    caption: str,
    text_length: int,
    ref_K_fullimg: torch.Tensor,
) -> dict:
    """Create zero-feature segment for text-conditioned generation.

    Args:
        caption: text prompt
        text_length: number of frames for this text segment
        ref_K_fullimg: (3, 3) intrinsics from first video (avoids NaN in compute_bbox_info_bedlam)

    Returns:
        Segment dict with zeroed visual features and text metadata.
    """
    from gem.utils.geo_transform import compute_cam_angvel

    L = text_length

    R_w2c = torch.eye(3).unsqueeze(0).expand(L, -1, -1).clone()
    cam_angvel = compute_cam_angvel(R_w2c, padding_last=True)
    cam_tvel = torch.zeros(L, 3)
    K_fullimg = ref_K_fullimg.unsqueeze(0).expand(L, -1, -1).clone()

    return {
        "type": "text",
        "caption": caption,
        "length": L,
        "bbx_xys": torch.zeros(L, 3),
        "kp2d": torch.zeros(L, 17, 3),
        "f_imgseq": torch.zeros(L, 1024),
        "has_img_mask": torch.zeros(L, dtype=torch.bool),
        "R_w2c": R_w2c,
        "cam_angvel": cam_angvel,
        "cam_tvel": cam_tvel,
        "K_fullimg": K_fullimg,
    }


# ===========================================================================
# Data assembly
# ===========================================================================


def assemble_mixed_data(
    segments: list[dict],
    static_cam: bool,
) -> tuple[dict, list[dict]]:
    """Concatenate all segments and build the unified data dict for model.predict().

    Returns:
        data: dict ready for model.predict()
        segment_info: list of dicts with (start, end, type, ...) per segment
    """
    from gem.utils.net_utils import get_valid_mask

    # Build segment_info and collect tensors
    segment_info = []
    current_idx = 0
    for seg in segments:
        L = seg["length"]
        info = {
            "start": current_idx,
            "end": current_idx + L,
            "type": seg["type"],
        }
        if seg["type"] == "video":
            info["video_path"] = seg.get("video_path", None)
        else:
            info["caption"] = seg["caption"]
        segment_info.append(info)
        current_idx += L

    tot_length = current_idx

    # Concatenate all tensors along time
    bbx_xys = torch.cat([s["bbx_xys"] for s in segments], dim=0)
    kp2d = torch.cat([s["kp2d"] for s in segments], dim=0)
    f_imgseq = torch.cat([s["f_imgseq"] for s in segments], dim=0)
    K_fullimg = torch.cat([s["K_fullimg"] for s in segments], dim=0)
    cam_angvel = torch.cat([s["cam_angvel"] for s in segments], dim=0)
    cam_tvel = torch.cat([s["cam_tvel"] for s in segments], dim=0)
    R_w2c = torch.cat([s["R_w2c"] for s in segments], dim=0)

    # Per-frame condition masks
    has_img_mask = torch.cat([s["has_img_mask"] for s in segments], dim=0)

    # has_2d_mask and has_cam_mask: True for video frames, False for text frames
    has_2d_mask = torch.zeros(tot_length, dtype=torch.bool)
    has_cam_mask = torch.zeros(tot_length, dtype=torch.bool)
    for seg, info in zip(segments, segment_info):
        if seg["type"] == "video":
            has_2d_mask[info["start"] : info["end"]] = True
            if not static_cam:
                has_cam_mask[info["start"] : info["end"]] = True

    # Build multi_text_data for text segments
    text_segments = [
        (i, seg, info)
        for i, (seg, info) in enumerate(zip(segments, segment_info))
        if seg["type"] == "text"
    ]

    # First text caption for the top-level caption field
    first_text_caption = text_segments[0][1]["caption"] if text_segments else ""

    multi_text_data = {
        "vid": [],
        "caption": [],
        "text_ind": [],
        "window_start": [],
        "window_end": [],
    }
    for idx, (_seg_idx, seg, info) in enumerate(text_segments):
        multi_text_data["vid"].append(f"text{idx}")
        multi_text_data["caption"].append(seg["caption"])
        multi_text_data["text_ind"].append(idx)
        multi_text_data["window_start"].append(info["start"] / tot_length)
        multi_text_data["window_end"].append(info["end"] / tot_length)

    multi_text_data["window_start"] = torch.tensor(multi_text_data["window_start"])
    multi_text_data["window_end"] = torch.tensor(multi_text_data["window_end"])

    data = {
        "kp2d": kp2d,
        "bbx_xys": bbx_xys,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "cam_tvel": cam_tvel,
        "R_w2c": R_w2c,
        "f_imgseq": f_imgseq,
        "has_text": torch.tensor([True]),
        "caption": first_text_caption,
        "mask": {
            "has_img_mask": has_img_mask,
            "has_2d_mask": has_2d_mask,
            "has_cam_mask": has_cam_mask,
            "has_audio_mask": get_valid_mask(tot_length, 0),
            "has_music_mask": get_valid_mask(tot_length, 0),
        },
        "length": torch.tensor(tot_length),
        "meta": [
            {
                "mode": "default",
                "multi_text_data": multi_text_data,
            }
        ],
    }

    return data, segment_info


# ===========================================================================
# Rendering
# ===========================================================================


def render_results_mixed(
    pred: dict,
    segments: list[dict],
    segment_info: list[dict],
    input_segments: list[dict],
    output_dir: str,
    no_render: bool = False,
    ref_width: int = 1280,
    ref_height: int = 720,
):
    """Segment-aware rendering for mixed video + text inputs."""
    os.makedirs(output_dir, exist_ok=True)

    # Save SMPL parameters
    params_path = os.path.join(output_dir, "smpl_params.pt")
    save_dict = {}
    for key_group in ("body_params_incam", "body_params_global"):
        if key_group in pred:
            save_dict[key_group] = {k: v.cpu() for k, v in pred[key_group].items()}
    if "K_fullimg" in pred:
        save_dict["K_fullimg"] = pred["K_fullimg"].cpu()
    save_dict["segment_info"] = segment_info
    torch.save(save_dict, params_path)
    print(f"[Stage 4] SMPL parameters saved to {params_path}")

    if no_render:
        print("[Stage 4] Skipping rendering (--no_render)")
        return

    try:
        from gem.utils.smplx_utils import make_smplx
        from gem.utils.video_io_utils import (
            get_video_lwh,
            read_video_np,
            save_video,
        )
    except ImportError as e:
        print(f"[Stage 4] Rendering dependencies not available: {e}")
        return

    body_model = make_smplx("supermotion")
    body_model.cuda().eval()
    smpl_faces = torch.from_numpy(body_model.faces.astype(np.int32)).long()

    bp_incam = pred["body_params_incam"]
    K_fullimg = pred["K_fullimg"]

    with torch.no_grad():
        smpl_out = body_model(
            body_pose=bp_incam["body_pose"].cuda(),
            global_orient=bp_incam["global_orient"].cuda(),
            transl=bp_incam["transl"].cuda(),
            betas=bp_incam.get("betas", torch.zeros(bp_incam["body_pose"].shape[0], 10)).cuda(),
        )
    verts_incam = smpl_out.vertices.cpu()

    # ---- In-camera per segment ----
    print("[Stage 4] Rendering in-camera overlay per segment ...")
    incam_parts = []
    for seg, info in zip(input_segments, segment_info):
        start, end = info["start"], info["end"]
        seg_verts = verts_incam[start:end]
        seg_K = K_fullimg[start]

        if seg["type"] == "video":
            video_path = seg["path"]
            L_vid, W, H = get_video_lwh(video_path)
            frames = read_video_np(video_path)
            seg_frames = render_incam_frames(frames, seg_verts, smpl_faces, seg_K, W, H)
            if seg_frames.shape[1] != ref_height or seg_frames.shape[2] != ref_width:
                h, w = seg_frames.shape[1], seg_frames.shape[2]
                scale = min(ref_width / w, ref_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                pad_x = (ref_width - new_w) // 2
                pad_y = (ref_height - new_h) // 2

                def _letterbox(f, _new_w=new_w, _new_h=new_h, _pad_x=pad_x, _pad_y=pad_y):
                    resized = cv2.resize(f, (_new_w, _new_h))
                    canvas = np.zeros((ref_height, ref_width, 3), dtype=f.dtype)
                    canvas[_pad_y : _pad_y + _new_h, _pad_x : _pad_x + _new_w] = resized
                    return canvas

                seg_frames = np.stack([_letterbox(f) for f in seg_frames])
            incam_parts.append(seg_frames)
        else:
            # Text segment: create text card frames
            font_size = int(min(ref_width, ref_height) * 0.1)
            text_frames = create_text_frames(
                seg["caption"], end - start, ref_width, ref_height, font_size=font_size
            )
            incam_parts.append(text_frames)

    incam_all = np.concatenate(incam_parts, axis=0)
    incam_path = os.path.join(output_dir, "1_incam.mp4")
    save_video(incam_all, incam_path, fps=30)
    print(f"[Stage 4] In-camera video saved to {incam_path}")

    # ---- Global render ----
    print("[Stage 4] Rendering global view ...")
    bp_global = pred.get("body_params_global")
    if bp_global is None:
        print("[Stage 4] No global body params; skipping global render.")
        return

    verts_global = normalize_global_verts(body_model, bp_global)
    W, H = ref_width, ref_height
    global_frames = render_global_frames(verts_global, smpl_faces, W, H)

    global_path = os.path.join(output_dir, "2_global.mp4")
    save_video(global_frames, global_path, fps=30)
    print(f"[Stage 4] Global video saved to {global_path}")

    # ---- Side-by-side ----
    print("[Stage 4] Creating side-by-side video ...")
    sbs_frames = render_side_by_side(incam_all, global_frames)
    sbs_path = os.path.join(output_dir, "3_incam_global_horiz.mp4")
    save_video(sbs_frames, sbs_path, fps=30)
    print(f"[Stage 4] Side-by-side video saved to {sbs_path}")

    # Force-exit to avoid Open3D Filament double-free crash
    os._exit(0)


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="GEM-SMPL Multi-Modal Demo: Video + Text conditioning"
    )
    parser.add_argument(
        "--input_list",
        nargs="+",
        required=True,
        help="Input list: .mp4/.avi/.mov files, .txt files, or 'text:prompt' strings",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Pretrained checkpoint path (auto-downloads from HuggingFace if not provided)",
    )
    parser.add_argument(
        "--output_root",
        default="outputs",
        help="Output directory (default: outputs)",
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
        "--text_length",
        type=int,
        default=300,
        help="Number of frames for each text segment (default: 300 = 10s at 30fps)",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Skip video rendering, only save SMPL parameters",
    )
    args = parser.parse_args()

    t0 = time.time()

    # ------------------------------------------------------------------
    # Parse inputs
    # ------------------------------------------------------------------
    input_segments = parse_input_list(args.input_list)

    # Derive output name from first video
    first_video = next(s for s in input_segments if s["type"] == "video")
    run_name = first_video["name"] + "_mix"
    output_dir = os.path.join(args.output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GEM-SMPL Multi-Modal Demo Pipeline")
    print(f"  Inputs:     {[s.get('name', s.get('caption', '?')) for s in input_segments]}")
    print(f"  Checkpoint: {args.ckpt_path}")
    print(f"  HMR2 ckpt:  {args.hmr2_ckpt}")
    print(f"  Output:     {output_dir}")
    print(f"  Static cam: {args.static_cam}")
    print(f"  Text len:   {args.text_length} frames")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Stage 1: Preprocess each segment
    # ------------------------------------------------------------------
    print("\n[Stage 1] Preprocessing segments ...")
    processed_segments = []
    ref_K = None
    ref_width = None
    ref_height = None

    for seg in input_segments:
        if seg["type"] == "video":
            video_path = seg["path"]
            assert os.path.exists(video_path), f"Video not found: {video_path}"

            preprocess_dir = os.path.join(output_dir, f"preprocess_{seg['name']}")
            os.makedirs(preprocess_dir, exist_ok=True)

            print(f"\n  Processing video: {seg['name']}")
            proc = preprocess_video_segment(
                video_path, preprocess_dir, args.hmr2_ckpt, args.static_cam
            )
            proc["video_path"] = video_path

            # Capture reference intrinsics from first video
            if ref_K is None:
                ref_K = proc["K_fullimg"][0]  # (3, 3)
                ref_width = proc["width"]
                ref_height = proc["height"]

            processed_segments.append(proc)

        elif seg["type"] == "text":
            # Defer text segment creation until we have ref_K
            processed_segments.append(seg)

    # Now create text segments with reference intrinsics
    for i, seg in enumerate(processed_segments):
        if isinstance(seg, dict) and seg.get("type") == "text" and "K_fullimg" not in seg:
            print(f"\n  Creating text segment: {input_segments[i]['name']} -> \"{seg['caption']}\"")
            processed_segments[i] = create_text_segment(seg["caption"], args.text_length, ref_K)

    print(f"\n[Stage 1] Preprocessing complete ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # Stage 2: Assemble mixed data
    # ------------------------------------------------------------------
    print("\n[Stage 2] Assembling mixed data dict ...")
    data, segment_info = assemble_mixed_data(processed_segments, args.static_cam)

    tot_length = int(data["length"])
    print(f"  Total sequence length: {tot_length} frames ({tot_length / 30:.1f}s)")
    for i, info in enumerate(segment_info):
        seg_type = info["type"]
        seg_len = info["end"] - info["start"]
        label = input_segments[i].get("name", input_segments[i].get("caption", "?"))
        print(
            f"  Segment {i}: [{info['start']}:{info['end']}] {seg_type} ({seg_len} frames) - {label}"
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
    model = load_model(ckpt_path, load_text_encoder=True)
    pred = run_inference(model, data, static_cam=args.static_cam)
    print(f"[Stage 3] Inference complete ({time.time() - t0:.1f}s total)")

    # ------------------------------------------------------------------
    # Stage 4: Rendering
    # ------------------------------------------------------------------
    print("\n[Stage 4] Rendering ...")
    render_results_mixed(
        pred=pred,
        segments=processed_segments,
        segment_info=segment_info,
        input_segments=input_segments,
        output_dir=output_dir,
        no_render=args.no_render,
        ref_width=ref_width,
        ref_height=ref_height,
    )

    print(f"\n{'=' * 60}")
    print(f"Done! Total time: {time.time() - t0:.1f}s")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
