# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch

from gem.utils.rotation_conversions import axis_angle_to_matrix


def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

    roi_image[mask] = image[mask]
    out_image[bbox[1] : bbox[3], bbox[0] : bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype

    K = torch.zeros((K_org.shape[0], 4, 4)).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1

    image_sizes = []
    for idx, bbox_item in enumerate(bbox):
        left, upper, right, lower = bbox_item
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R is not None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T is not None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = right - left
    height = bottom - top

    new_left = torch.clamp(cx - width / 2 * scaleFactor, min=0, max=img_w - 1)
    new_right = torch.clamp(cx + width / 2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h - 1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = (
        torch.stack(
            (
                new_left.detach(),
                new_top.detach(),
                new_right.detach(),
                new_bottom.detach(),
            )
        )
        .int()
        .float()
        .T
    )

    return bbox


def get_global_cameras_static_v2(
    verts,
    beta=4.0,
    cam_height_degree=30,
    target_center_height=1.0,
    use_long_axis=False,
    vec_rot=45,
    device="cuda",
):
    L, V, _ = verts.shape

    # Compute target trajectory, denote as center + scale
    targets = verts.mean(1)  # (L, 3)
    targets[:, 1] = 0  # project to xz-plane
    target_center = targets.mean(0)  # (3,)
    target_scale, target_idx = torch.norm(targets - target_center, dim=-1).max(0)

    # a 45 degree vec from longest axis
    if use_long_axis:
        long_vec = targets[target_idx] - target_center  # (x, 0, z)
        long_vec = long_vec / torch.norm(long_vec)
        R = axis_angle_to_matrix(torch.tensor([0, np.pi / 4, 0])).to(long_vec)
        vec = R @ long_vec
    else:
        vec_rad = vec_rot / 180 * np.pi
        vec = torch.tensor([np.sin(vec_rad), 0, np.cos(vec_rad)]).float()
        vec = vec / torch.norm(vec)

    # Compute camera position (center + scale * vec * beta) + y=4
    target_scale = max(target_scale, 1.0) * beta
    position = target_center + vec * target_scale
    position[1] = target_scale * np.tan(np.pi * cam_height_degree / 180) + target_center_height

    target_center[1] = target_center_height
    up = torch.tensor([0, 1, 0])
    return position, target_center, up


def get_ground_params_from_points(root_points, vert_points):
    """xz-plane is the ground plane
    Args:
        root_points: (L, 3), to decide center
        vert_points: (L, V, 3), to decide scale
    """
    root_max = root_points.max(0)[0]  # (3,)
    root_min = root_points.min(0)[0]  # (3,)
    cx, _, cz = (root_max + root_min) / 2.0

    vert_max = vert_points.reshape(-1, 3).max(0)[0]  # (L, 3)
    vert_min = vert_points.reshape(-1, 3).min(0)[0]  # (L, 3)
    scale = (vert_max - vert_min)[[0, 2]].max()
    return float(scale), float(cx), float(cz)
