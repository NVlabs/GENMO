# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Camera utility functions.
"""

import math

import torch

from gem.utils.geo_transform import convert_bbx_xys_to_lurb, cvt_to_bi01_p2d


def perspective_projection(points, K):
    """Standard pinhole camera projection.

    Args:
        points: (*, J, 3)
        K: (*, 3, 3)
    Returns:
        p2d: (*, J, 2)
    """
    projected = points / points[..., -1:].clamp(min=1e-6)
    projected = torch.einsum("...ij,...kj->...ki", K, projected.float())
    return projected[..., :2]


def compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg):
    """Convert weak-perspective camera parameters to full-perspective translation.

    Args:
        pred_cam: (*, 3)  [s, tx, ty]
        bbx_xys: (*, 3)  [cx, cy, size]
        K_fullimg: (*, 3, 3)
    Returns:
        cam_t: (*, 3)
    """
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = K_fullimg[..., 0, 0]
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)
    return torch.stack([tx + cx, ty + cy, tz], dim=-1)


def get_a_pred_cam(transl, bbx_xys, K_fullimg):
    """Inverse of compute_transl_full_cam: full translation → weak-perspective params.

    Args:
        transl: (*, 3)
        bbx_xys: (*, 3)
        K_fullimg: (*, 3, 3)  one extra leading dim compared to transl/bbx_xys
    Returns:
        pred_cam: (*, 3)  [s, tx, ty]
    """
    f = K_fullimg[..., 0, 0]
    cx = K_fullimg[..., 0, 2]
    cy = K_fullimg[..., 1, 2]
    gt_s = 2 * f / (transl[..., 2] * bbx_xys[..., 2])
    gt_x = transl[..., 0] - transl[..., 2] / f * (bbx_xys[..., 0] - cx)
    gt_y = transl[..., 1] - transl[..., 2] / f * (bbx_xys[..., 1] - cy)
    return torch.stack([gt_s, gt_x, gt_y], dim=-1)


def compute_bbox_info_bedlam(bbx_xys, K_fullimg):
    """Compute BEDLAM-style bounding box info (normalised by focal length).

    Args:
        bbx_xys: (*, 3)  [cx, cy, size]
        K_fullimg: (*, 3, 3)
    Returns:
        bbox_info: (*, 3)
    """
    fl = K_fullimg[..., 0, 0].unsqueeze(-1)
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    cx, cy, b = bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2]
    bbox_info = torch.stack([cx - icx, cy - icy, b], dim=-1)
    return bbox_info / fl


def safely_render_x3d_K(x3d, K_fullimg, thr=0.3):
    """Project 3D vertices to 2D, clamping near-zero depths for safety.

    Args:
        x3d: (*, V, 3)
        K_fullimg: (*, 3, 3)
        thr: minimum depth threshold
    Returns:
        i_x2d: (*, V, 2)
    """
    x3d = x3d.clone()
    x3d[..., 2] = x3d[..., 2].clamp(min=thr)
    return perspective_projection(x3d, K_fullimg)


def estimate_K(width, height):
    """Estimate default pinhole intrinsics from image dimensions.

    Uses max(w, h) as focal length — a reasonable approximation when
    the true focal length is unknown.

    Args:
        width, height: image dimensions (int or float)
    Returns:
        K: (3, 3) camera intrinsics tensor
    """
    focal = float(max(width, height))
    K = torch.eye(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = float(width) / 2.0
    K[1, 2] = float(height) / 2.0
    return K


def resize_K(K, scale):
    """Scale camera intrinsics for a resized image.

    Args:
        K: (*, 3, 3) intrinsics tensor
        scale: scalar scale factor
    Returns:
        K_new: (*, 3, 3) scaled intrinsics
    """
    K_new = K.clone().float()
    K_new[..., :2, :] = K_new[..., :2, :] * scale
    return K_new


def create_camera_sensor(width, height, fov_deg=60):
    """Create camera intrinsics from image dimensions and field-of-view angle.

    Args:
        width, height: image dimensions
        fov_deg: horizontal field of view in degrees
    Returns:
        (width, height, K)  where K is (3, 3)
    """
    focal = float(width) / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    K = torch.eye(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = float(width) / 2.0
    K[1, 2] = float(height) / 2.0
    return width, height, K


def get_bbx_xys_from_xyxy(xyxy, base_zoom=1.0, minimal_size=10):
    """Convert bounding box from [x1, y1, x2, y2] to [cx, cy, size].

    Args:
        xyxy: (..., 4) bounding box in xyxy format
        base_zoom: zoom factor applied to the box size
        minimal_size: minimum allowed box size
    Returns:
        xys: (..., 3) [cx, cy, size]
    """
    x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    size = torch.max(x2 - x1, y2 - y1) * base_zoom
    size = size.clamp(min=minimal_size)
    return torch.stack([cx, cy, size], dim=-1)


def project_to_bi01(points, bbx_xys, K_fullimg):
    """Project 3D points to bounding-box-normalised [0,1] 2D coordinates.

    Args:
        points: (*, J, 3)
        bbx_xys: (*, 3)
        K_fullimg: (*, 3, 3)
    Returns:
        p2d_bi01: (*, J, 2)
    """
    p2d = perspective_projection(points, K_fullimg)
    bbx_lurb = convert_bbx_xys_to_lurb(bbx_xys)
    return cvt_to_bi01_p2d(p2d, bbx_lurb)
