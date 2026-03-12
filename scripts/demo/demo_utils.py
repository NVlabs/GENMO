# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for GEM-SMPL demo scripts.

"""
from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repo path setup (mirrors scripts/train.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ===========================================================================
# Stage 1 helpers
# ===========================================================================

# ---- 1a. Person Detection + Tracking (YOLOv8) ----------------------------


def detect_and_track(video_path: str, preprocess_dir: str) -> torch.Tensor:
    """Run YOLOv8 person detection, select top-1 track, smooth bboxes.

    Returns:
        bbx_xys: (L, 3) tensor of [center_x, center_y, bbox_size]
    """
    cache_path = os.path.join(preprocess_dir, "bbx.pt")
    if os.path.exists(cache_path):
        print(f"[Stage 1a] Loading cached bounding boxes from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print("[Stage 1a] Running YOLOv8 person detection ...")
    from ultralytics import YOLO

    from gem.utils.video_io_utils import get_video_lwh, read_video_np

    L, W, H = get_video_lwh(video_path)
    frames = read_video_np(video_path)  # (L, H, W, 3) RGB uint8

    model = YOLO("yolov8x.pt")

    # Collect per-frame person detections: list of (L,) lists of [x1,y1,x2,y2,conf]
    all_boxes: list[list[np.ndarray]] = [[] for _ in range(len(frames))]
    for i, frame in enumerate(tqdm(frames, desc="YOLO detect", leave=False)):
        frame_bgr = frame[..., ::-1].copy()
        results = model(frame_bgr, classes=[0], verbose=False)  # class 0 = person
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                all_boxes[i].append(np.append(xyxy, conf))

    # Simple greedy tracking: pick the person with the largest total area across frames.
    # For each frame take the largest-area detection.
    bbx_xyxy_list = []
    for i in range(len(frames)):
        if len(all_boxes[i]) == 0:
            # No detection: repeat last or use zeros
            if len(bbx_xyxy_list) > 0:
                bbx_xyxy_list.append(bbx_xyxy_list[-1].copy())
            else:
                bbx_xyxy_list.append(np.array([W / 4, H / 4, 3 * W / 4, 3 * H / 4]))
        else:
            boxes_np = np.stack(all_boxes[i])  # (N, 5)
            areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
            best = areas.argmax()
            bbx_xyxy_list.append(boxes_np[best, :4])

    bbx_xyxy = torch.tensor(np.stack(bbx_xyxy_list), dtype=torch.float32)  # (L, 4)

    # Clamp to image bounds
    bbx_xyxy[:, [0, 2]] = bbx_xyxy[:, [0, 2]].clamp(0, W - 1)
    bbx_xyxy[:, [1, 3]] = bbx_xyxy[:, [1, 3]].clamp(0, H - 1)

    # Convert xyxy -> xys
    from gem.utils.geo_transform import get_bbx_xys_from_xyxy

    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2)  # (L, 3)

    # Smooth with moving average
    from gem.utils.net_utils import moving_average_smooth

    bbx_xys = moving_average_smooth(bbx_xys, window_size=5, dim=0)

    os.makedirs(preprocess_dir, exist_ok=True)
    torch.save(bbx_xys, cache_path)
    print(f"[Stage 1a] Bounding boxes saved to {cache_path} ({len(bbx_xys)} frames)")
    return bbx_xys


# ---- 1b. COCO-17 2D Keypoint Extraction (ViTPose-H) ----------------------


class CocoPoseExtractor:
    """Lightweight COCO-17 ViTPose-H extractor.

    Uses custom ViT-H backbone + deconv head matching the ViTPose checkpoint.
    If the checkpoint is unavailable, falls back to zero-confidence keypoints.
    """

    CKPT_PATH = "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # COCO-17 left-right flip pairs for flip-test
    FLIP_PAIRS = [
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
    ]

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._fallback = False

        ckpt_path = self.CKPT_PATH
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ViTPose-H COCO checkpoint not found at {ckpt_path}")

        self.model = self._build_model(ckpt_path)
        self.model.to(device).eval()

    # ------------------------------------------------------------------
    @staticmethod
    def _build_model(ckpt_path: str) -> nn.Module:
        """Build ViT-H + deconv head for COCO-17 heatmaps."""
        from gem.utils.vitpose_model import build_vitpose_coco17

        return build_vitpose_coco17(ckpt_path)

    # ------------------------------------------------------------------
    def _preprocess(self, frames: np.ndarray, bbx_xys: torch.Tensor) -> torch.Tensor:
        """Crop person regions to 256x256 square, then slice to 256x192."""
        T = len(frames)
        out = np.zeros((T, 3, 256, 192), dtype=np.float32)
        for i in range(T):
            cx, cy, s = float(bbx_xys[i, 0]), float(bbx_xys[i, 1]), float(bbx_xys[i, 2])
            hs = s / 2
            # Affine transform: map bbox square -> 256x256
            src = np.array([[cx - hs, cy - hs], [cx + hs, cy - hs], [cx, cy]], dtype=np.float32)
            dst = np.array([[0, 0], [255, 0], [127.5, 127.5]], dtype=np.float32)
            M = cv2.getAffineTransform(src, dst)
            crop256 = cv2.warpAffine(frames[i], M, (256, 256), flags=cv2.INTER_LINEAR)
            # Slice center 192 columns (matching old repo's [32:224])
            crop = crop256[:, 32:224]  # (256, 192, 3) RGB
            # Normalize (frames are RGB, no channel reversal needed)
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - self.MEAN) / self.STD
            out[i] = crop.transpose(2, 0, 1)
        return torch.from_numpy(out)

    def _heatmap_to_keypoints(
        self, heatmaps: np.ndarray, bbx_xys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Argmax + subpixel refinement -> image-space keypoints.

        Args:
            heatmaps: (N, 17, H, W)
            bbx_xys: (N, 3)
        Returns:
            preds: (N, 17, 2)
            maxvals: (N, 17, 1)
        """
        N, K, H, W = heatmaps.shape
        flat = heatmaps.reshape(N, K, -1)
        idx = flat.argmax(-1)  # (N, K)
        px = (idx % W).astype(np.float32)
        py = (idx // W).astype(np.float32)
        # Subpixel refinement
        for n in range(N):
            for k in range(K):
                hm = heatmaps[n, k]
                x, y = int(px[n, k]), int(py[n, k])
                if 1 < x < W - 1:
                    px[n, k] += np.sign(hm[y, x + 1] - hm[y, x - 1]) * 0.25
                if 1 < y < H - 1:
                    py[n, k] += np.sign(hm[y + 1, x] - hm[y - 1, x]) * 0.25

        preds = np.stack([px, py], axis=-1)  # (N, K, 2)
        # Map from heatmap coords to image coords using bbx_xys
        cx = bbx_xys[:, 0:1]  # (N, 1)
        cy = bbx_xys[:, 1:2]
        sz = bbx_xys[:, 2:3]
        # heatmap (H, W) corresponds to crop of size (sz, sz*192/256)
        preds[..., 0] = preds[..., 0] / W * sz * (192.0 / 256.0) + (cx - sz * (192.0 / 256.0) / 2)
        preds[..., 1] = preds[..., 1] / H * sz + (cy - sz / 2)
        maxvals_raw = flat.max(-1)
        maxvals = (1 / (1 + np.exp(-maxvals_raw)))[..., np.newaxis]  # sigmoid -> [0,1]
        return preds, maxvals

    @staticmethod
    def _flip_heatmap_coco17(hm: torch.Tensor) -> torch.Tensor:
        """Flip heatmaps horizontally and swap left-right joint channels."""
        hm = hm.flip(-1)  # horizontal flip
        pairs = [
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (9, 10),
            (11, 12),
            (13, 14),
            (15, 16),
        ]
        hm_new = hm.clone()
        for left, right in pairs:
            hm_new[:, left] = hm[:, right]
            hm_new[:, right] = hm[:, left]
        return hm_new

    @torch.no_grad()
    def extract(
        self, frames: np.ndarray, bbx_xys: torch.Tensor, batch_size: int = 16
    ) -> torch.Tensor:
        """Extract COCO-17 keypoints.

        Args:
            frames: (L, H, W, 3) uint8 RGB
            bbx_xys: (L, 3)
        Returns:
            kp2d: (L, 17, 3) tensor [x, y, confidence]
        """
        L = len(frames)
        if self._fallback:
            print("[Stage 1b] Returning zero keypoints (no ViTPose checkpoint)")
            return torch.zeros(L, 17, 3)

        imgs = self._preprocess(frames, bbx_xys)
        results = []
        for j in tqdm(range(0, L, batch_size), desc="ViTPose COCO17", leave=False):
            batch = imgs[j : j + batch_size].to(self.device)
            # Flip test
            heatmap_normal = self.model(batch)
            heatmap_flipped = self.model(batch.flip(-1))
            heatmap_flipped = self._flip_heatmap_coco17(heatmap_flipped)
            heatmap = (heatmap_normal + heatmap_flipped) * 0.5

            hm_np = heatmap.cpu().numpy()
            bxy_np = bbx_xys[j : j + batch_size].numpy()
            preds, maxvals = self._heatmap_to_keypoints(hm_np, bxy_np)
            kp = np.concatenate([preds, maxvals], axis=-1)  # (B, 17, 3)
            results.append(torch.from_numpy(kp).float())

        return torch.cat(results, dim=0)


# ---- 1c. Image Features (HMR2 ViT) ---------------------------------------


def get_image_features(
    video_path: str,
    bbx_xys: torch.Tensor,
    hmr2_ckpt: str,
    preprocess_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract HMR2 ViT features, with caching.

    Args:
        video_path: path to input video
        bbx_xys: (L, 3) bounding boxes
        hmr2_ckpt: path to HMR2 checkpoint
        preprocess_dir: directory for caching preprocessed data

    Returns:
        f_imgseq: (L, 1024) image features
        has_img_mask: (L,) bool mask (all True)
    """
    cache_path = os.path.join(preprocess_dir, "vit_features.pt")
    if os.path.exists(cache_path):
        print(f"[Stage 1c] Loading cached HMR2 features from {cache_path}")
        f_imgseq = torch.load(cache_path, map_location="cpu")
        has_img_mask = torch.ones(f_imgseq.shape[0], dtype=torch.bool)
        return f_imgseq, has_img_mask

    if not os.path.exists(hmr2_ckpt):
        print(
            f"\n[ERROR] HMR2 checkpoint not found at {hmr2_ckpt}.\n"
            "Download from GVHMR's Google Drive: "
            "https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD\n"
            "Place hmr2/epoch=10-step=25000.ckpt under inputs/checkpoints/\n"
        )
        sys.exit(1)

    print(f"[Stage 1c] Extracting HMR2 features (ckpt: {hmr2_ckpt}) ...")
    from gem.utils.hmr2_extractor import HMR2FeatureExtractor

    extractor = HMR2FeatureExtractor(hmr2_ckpt, device="cuda")
    f_imgseq = extractor.extract_video_features(video_path, bbx_xys, batch_size=16)

    os.makedirs(preprocess_dir, exist_ok=True)
    torch.save(f_imgseq, cache_path)
    print(f"[Stage 1c] HMR2 features saved to {cache_path} ({f_imgseq.shape})")

    has_img_mask = torch.ones(f_imgseq.shape[0], dtype=torch.bool)
    return f_imgseq, has_img_mask


# ---- 1d. Camera (static or SLAM) -----------------------------------------


def get_camera_static(L: int, width: int, height: int):
    """Static camera: identity rotations, zero velocity.

    Returns:
        R_w2c: (L, 3, 3)
        cam_angvel: (L, 6)
        cam_tvel: (L, 3)
        K_fullimg: (L, 3, 3)
    """
    from gem.utils.cam_utils import estimate_K
    from gem.utils.geo_transform import compute_cam_angvel

    R_w2c = torch.eye(3).unsqueeze(0).expand(L, -1, -1).clone()
    cam_angvel = compute_cam_angvel(R_w2c, padding_last=True)  # (L, 6) -- all identity 6d
    cam_tvel = torch.zeros(L, 3)
    K = estimate_K(width, height)  # (3, 3)
    K_fullimg = K.unsqueeze(0).expand(L, -1, -1).clone()

    return R_w2c, cam_angvel, cam_tvel, K_fullimg


# ===========================================================================
# Stage 3: Model loading and inference
# ===========================================================================


def load_model(ckpt_path: str, load_text_encoder: bool = False):
    """Load the GEM model with Hydra config composition.

    Args:
        ckpt_path: path to pretrained checkpoint
        load_text_encoder: if True, load T5 text encoder for text-conditioned generation

    Returns the model (on CUDA, eval mode).
    """
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", builtins.eval, replace=True)

    import hydra
    from hydra import compose, initialize_config_dir

    config_dir = str(_REPO_ROOT / "configs")

    overrides = [
        "exp=gem_smpl",
        "ckpt_path=null",
        "video_name=demo",
    ]
    if load_text_encoder:
        overrides.append("model.model_cfg.text_encoder.load_llm=true")

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="demo",
            overrides=overrides,
        )

    print("[Stage 3] Instantiating GEM model ...")
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)

    # Load checkpoint weights
    from gem.utils.net_utils import load_pretrained_model

    load_pretrained_model(model, ckpt_path)
    print(f"[Stage 3] Loaded checkpoint: {ckpt_path}")

    model.cuda().eval()
    return model


def run_inference(model, data: dict, static_cam: bool) -> dict:
    """Run GEM.predict() and return results."""
    print("[Stage 3] Running model inference ...")
    with torch.no_grad():
        pred = model.predict(data, static_cam=static_cam)
    return pred


# ===========================================================================
# Stage 4: Visualization helpers
# ===========================================================================


def render_2d_keypoints(
    video_path: str,
    kp2d: torch.Tensor,  # (L, 17, 3)
    bbx_xys: torch.Tensor,  # (L, 3)
    output_dir: str,
    fps: int = 30,
):
    """Render 2D keypoints overlay on the original video."""
    out_path = os.path.join(output_dir, "0_kp2d_overlay.mp4")
    if os.path.exists(out_path):
        print(f"[Stage 4] 2D keypoint overlay already exists: {out_path}")
        return

    from gem.utils.video_io_utils import read_video_np, save_video
    from gem.utils.vis.cv2_utils import draw_bbx_xys_on_image, draw_coco17_skeleton

    frames = read_video_np(video_path)  # (L, H, W, 3) RGB uint8
    kp2d_np = kp2d.cpu().numpy()  # (L, 17, 3)
    bbx_np = bbx_xys.cpu().numpy()  # (L, 3)

    if kp2d_np[..., 2].max() == 0:
        print(
            "[Stage 4] WARNING: All keypoint confidences are 0 — ViTPose checkpoint likely missing. "
            f"Ensure it exists at: {CocoPoseExtractor.CKPT_PATH}"
        )
        return

    out_frames = []
    for i in range(min(len(frames), len(kp2d_np))):
        img = draw_coco17_skeleton(frames[i], kp2d_np[i], conf_thr=0.5)
        img = draw_bbx_xys_on_image(bbx_np[i], img)
        out_frames.append(img)

    save_video(np.stack(out_frames), out_path, fps=fps)
    print(f"[Stage 4] 2D keypoint overlay saved to {out_path}")


def render_incam_frames(
    frames: np.ndarray,
    verts_incam: torch.Tensor,
    smpl_faces: torch.Tensor,
    K_render: torch.Tensor,
    W: int,
    H: int,
) -> np.ndarray:
    """Render SMPL mesh overlay on video frames (in-camera space).

    Args:
        frames: (L, H, W, 3) RGB uint8 video frames
        verts_incam: (L, V, 3) SMPL vertices in camera space
        smpl_faces: (F, 3) face indices
        K_render: (3, 3) camera intrinsics
        W, H: frame dimensions

    Returns:
        incam_frames: (L, H, W, 3) RGB uint8 composited frames
    """
    import open3d as o3d

    from gem.utils.vis.o3d_render import Settings, create_meshes

    mat_settings = Settings()
    lit_mat = mat_settings._materials[Settings.LIT]

    color_incam = torch.tensor([0.69019608, 0.39215686, 0.95686275])

    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])
    renderer.scene.set_lighting(
        renderer.scene.LightingProfile.SOFT_SHADOWS, np.array([0.0, 0.7, 0.7])
    )
    renderer.scene.camera.set_projection(
        K_render.cpu().double().numpy(), 0.01, 100.0, float(W), float(H)
    )
    # OpenCV camera convention: camera at origin, looking +Z, Y-down
    eye = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, -1.0, 0.0])
    renderer.scene.camera.look_at(target, eye, up)

    incam_frames = []
    for i in tqdm(range(len(frames)), desc="In-camera render", leave=False):
        mesh = create_meshes(verts_incam[i], smpl_faces, color_incam)
        mesh_name = f"mesh_{i}"
        if i > 0:
            renderer.scene.remove_geometry(f"mesh_{i - 1}")
        renderer.scene.add_geometry(mesh_name, mesh, lit_mat)
        rendered = np.array(renderer.render_to_image())  # (H, W, 3) uint8
        depth = np.asarray(renderer.render_to_depth_image())  # (H, W) float32
        mask = (depth < 1.0).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=1.0)
        alpha = mask[..., np.newaxis]
        composite = rendered.astype(np.float32) * alpha + frames[i].astype(np.float32) * (
            1.0 - alpha
        )
        incam_frames.append(composite.clip(0, 255).astype(np.uint8))

    return np.stack(incam_frames)


def render_global_frames(
    verts_global: torch.Tensor,
    smpl_faces: torch.Tensor,
    W: int,
    H: int,
) -> np.ndarray:
    """Render SMPL mesh in global space with ground plane.

    Args:
        verts_global: (L, V, 3) SMPL vertices in normalized global space
        smpl_faces: (F, 3) face indices
        W, H: frame dimensions

    Returns:
        global_frames: (L, H, W, 3) RGB uint8
    """
    import open3d as o3d

    from gem.utils.cam_utils import create_camera_sensor
    from gem.utils.vis.o3d_render import Settings, create_meshes, get_ground
    from gem.utils.vis.renderer import (
        get_global_cameras_static_v2,
        get_ground_params_from_points,
    )

    L = verts_global.shape[0]

    mat_settings = Settings()
    lit_mat = mat_settings._materials[Settings.LIT]

    # Ground plane parameters
    root_points = verts_global.mean(1)  # (L, 3)
    scale, cx, cz = get_ground_params_from_points(root_points, verts_global)

    # Global camera
    _, _, K_global = create_camera_sensor(W, H, fov_deg=24)
    position, target_center, up_vec = get_global_cameras_static_v2(
        verts_global.clone(),
        beta=4.5,
        cam_height_degree=30,
    )

    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.set_lighting(
        renderer.scene.LightingProfile.NO_SHADOWS, np.array([0.577, -0.577, -0.577])
    )
    renderer.scene.camera.set_projection(
        K_global.cpu().double().numpy(), 0.1, 100.0, float(W), float(H)
    )
    renderer.scene.camera.look_at(
        target_center.cpu().numpy(), position.cpu().numpy(), up_vec.cpu().numpy()
    )

    # Add ground mesh
    ground = get_ground(max(scale, 3) * 1.5, cx, cz)
    gv, gf, gc = ground
    ground_mesh = create_meshes(gv, gf, gc[..., :3])
    ground_mat = o3d.visualization.rendering.MaterialRecord()
    ground_mat.shader = Settings.LIT
    renderer.scene.add_geometry("mesh_ground", ground_mesh, ground_mat)

    color_global = torch.tensor([0.69019608, 0.39215686, 0.95686275])
    global_frames = []
    for i in tqdm(range(L), desc="Global render", leave=False):
        mesh = create_meshes(verts_global[i], smpl_faces, color_global)
        if i > 0:
            renderer.scene.remove_geometry(f"mesh_{i - 1}")
        renderer.scene.add_geometry(f"mesh_{i}", mesh, lit_mat)
        global_frames.append(np.array(renderer.render_to_image()))

    return np.stack(global_frames)


def normalize_global_verts(body_model, bp_global):
    """Run SMPL forward pass on global params and normalize to ground-centered coords.

    Args:
        body_model: SMPL body model (on CUDA)
        bp_global: dict of global body params

    Returns:
        verts_global: (L, V, 3) normalized vertices (feet on ground, centered, face-Z)
    """
    from gem.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay

    with torch.no_grad():
        smpl_out_global = body_model(
            body_pose=bp_global["body_pose"].cuda(),
            global_orient=bp_global["global_orient"].cuda(),
            transl=bp_global["transl"].cuda(),
            betas=bp_global.get("betas", torch.zeros(bp_global["body_pose"].shape[0], 10)).cuda(),
        )
    verts_global = smpl_out_global.vertices.cpu()  # (L, V, 3)
    joints_global = smpl_out_global.joints.cpu()  # (L, J, 3)

    # Normalize: feet on ground, centered at origin, face Z
    offset = joints_global[0, 0].clone()  # root joint at frame 0
    offset[1] = verts_global[:, :, 1].min()  # align feet to Y=0
    verts_global = verts_global - offset
    joints_global = joints_global - offset
    T_ay2ayfz = compute_T_ayfz2ay(joints_global[[0]], inverse=True)
    verts_global = apply_T_on_points(verts_global, T_ay2ayfz)

    return verts_global


def render_side_by_side(
    incam_frames: np.ndarray,
    global_frames: np.ndarray,
) -> np.ndarray:
    """Create side-by-side composition of in-camera and global frames.

    Args:
        incam_frames: (L1, H, W, 3) RGB uint8
        global_frames: (L2, H2, W2, 3) RGB uint8

    Returns:
        sbs_frames: (min(L1,L2), H, W+W', 3) RGB uint8
    """
    # Resize global frames to match incam height if needed
    if incam_frames.shape[1] != global_frames.shape[1]:
        target_h = incam_frames.shape[1]
        scale_ratio = target_h / global_frames.shape[1]
        target_w = int(global_frames.shape[2] * scale_ratio)
        global_frames_resized = np.stack(
            [cv2.resize(f, (target_w, target_h)) for f in global_frames]
        )
    else:
        global_frames_resized = global_frames

    # Ensure same frame count (use min)
    min_len = min(len(incam_frames), len(global_frames_resized))
    sbs_frames = np.concatenate([incam_frames[:min_len], global_frames_resized[:min_len]], axis=2)
    return sbs_frames


def create_text_frames(
    text: str,
    num_frames: int,
    width: int,
    height: int,
    font_size: int = 60,
) -> np.ndarray:
    """Create frames with centered text on black background (word-wrapped).

    Args:
        text: text to display
        num_frames: number of frames to generate
        width, height: frame dimensions
        font_size: font size for text

    Returns:
        frames: (num_frames, H, W, 3) RGB uint8
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)

    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Word-wrap
    max_text_width = width * 0.9
    words = text.split()
    if not words:
        words = [""]
    lines = [words[0]]
    for word in words[1:]:
        test_line = lines[-1] + " " + word
        test_width = draw.textbbox((0, 0), test_line, font=font)[2]
        if test_width <= max_text_width:
            lines[-1] = test_line
        else:
            lines.append(word)

    # Draw centered
    line_height = font_size * 1.2
    total_text_height = len(lines) * line_height
    y_position = (height - total_text_height) / 2

    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=font)[2]
        x_position = (width - line_width) / 2
        draw.text((x_position, y_position), line, font=font, fill=(255, 255, 255))
        y_position += line_height

    frame = np.array(img)  # RGB
    # Repeat for all frames
    return np.stack([frame] * num_frames)
