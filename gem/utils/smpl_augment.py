# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import torch

import gem.utils.matrix as matrix
from gem.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
)

_BODY_MODEL_DIR = Path(__file__).parent / "body_model"

COCO17_AUG = {
    k: v.flatten()
    for k, v in torch.load(_BODY_MODEL_DIR / "coco_aug_dict.pth", weights_only=False).items()
}
COCO17_AUG_CUDA = {}
COCO17_TREE = [
    [5, 6],
    0,
    0,
    1,
    2,
    -1,
    -1,
    5,
    6,
    7,
    8,
    -1,
    -1,
    11,
    12,
    13,
    14,
    15,
    15,
    15,
    16,
    16,
    16,
]

COCO12_AUG = {
    k: v.flatten()[5:]
    for k, v in torch.load(_BODY_MODEL_DIR / "coco_aug_dict.pth", weights_only=False).items()
}
COCO12_AUG_CUDA = {}
COCO12_TREE = [
    -1,
    -1,
    0,
    1,
    2,
    3,
    -1,
    -1,
    6,
    7,
    8,
    9,
]


def gaussian_augment(body_pose, std_angle=10.0, to_R=True):
    """
    Args:
        body_pose torch.Tensor: (..., J, 3) axis-angle if to_R is True, else rotmat (..., J, 3, 3)
        std_angle: scalar or list, in degree
    """

    body_pose = body_pose.clone()

    if to_R:
        body_pose_R = axis_angle_to_matrix(body_pose)  # (B, L, J, 3, 3)
    else:
        body_pose_R = body_pose
    shape = body_pose_R.shape[:-2]
    device = body_pose.device

    # 1. Simulate noise
    # angle:
    std_angle = torch.tensor(std_angle).to(device).reshape(-1)  # allow scalar or list
    noise_angle = torch.randn(shape, device=device) * std_angle * torch.pi / 180

    # axis: avoid zero vector
    noise_axis = torch.rand((*shape, 3), device=device)
    mask_ = torch.norm(noise_axis, dim=-1) < 1e-6
    noise_axis[mask_] = 1

    noise_axis = noise_axis / torch.norm(noise_axis, dim=-1, keepdim=True)
    noise_aa = noise_angle[..., None] * noise_axis  # (B, L, J, 3)
    noise_R = axis_angle_to_matrix(noise_aa)  # (B, L, J, 3, 3)

    # 2. Add noise to body pose
    new_body_pose_R = matrix.get_mat_BfromA(body_pose_R, noise_R)  # (B, L, J, 3, 3)
    new_body_pose_r6d = matrix_to_rotation_6d(new_body_pose_R)  # (B, L, J, 6)
    new_body_pose_aa = matrix_to_axis_angle(new_body_pose_R)  # (B, L, J, 3)

    return new_body_pose_R, new_body_pose_r6d, new_body_pose_aa


# ========= Augment Joint 3D ======== #


def get_jitter(shape=(8, 120), s_jittering=5e-2, num_J=12):
    """Guassian jitter modeling."""
    if num_J == 17:
        AUG_DICT = COCO17_AUG
    elif num_J == 12:
        AUG_DICT = COCO12_AUG
    else:
        raise ValueError(f"num_J: {num_J} is not supported")
    jittering_noise = (
        torch.normal(
            mean=torch.zeros((*shape, num_J, 3)),
            std=AUG_DICT["jittering"].reshape(1, 1, num_J, 1).expand(*shape, -1, 3),
        )
        * s_jittering
    )
    return jittering_noise


def get_jitter_cuda(shape=(8, 120), s_jittering=5e-2, num_J=12, device="cuda"):
    if num_J == 17:
        AUG_DICT = COCO17_AUG
        AUG_DICT_CUDA = COCO17_AUG_CUDA
    elif num_J == 12:
        AUG_DICT = COCO12_AUG
        AUG_DICT_CUDA = COCO12_AUG_CUDA
    else:
        raise ValueError(f"num_J: {num_J} is not supported")
    if "jittering" not in AUG_DICT_CUDA:
        AUG_DICT_CUDA["jittering"] = AUG_DICT["jittering"].to(device).reshape(1, 1, num_J, 1)
    jittering = AUG_DICT_CUDA["jittering"]
    jittering_noise = torch.randn((*shape, num_J, 3), device=device) * jittering * s_jittering
    return jittering_noise


def get_lfhp(shape=(8, 120), s_peak=3e-1, s_peak_mask=5e-3, num_J=12):
    """Low-frequency high-peak noise modeling."""
    if num_J == 17:
        AUG_DICT = COCO17_AUG
    elif num_J == 12:
        AUG_DICT = COCO12_AUG
    else:
        raise ValueError(f"num_J: {num_J} is not supported")

    def get_peak_noise_mask():
        peak_noise_mask = torch.rand(*shape, num_J) * AUG_DICT["pmask"]
        peak_noise_mask = peak_noise_mask < s_peak_mask
        return peak_noise_mask

    peak_noise_mask = get_peak_noise_mask()  # (B, L, num_J)
    peak_noise = peak_noise_mask.float().unsqueeze(-1).repeat(1, 1, 1, 3)
    peak_noise = peak_noise * torch.randn(3) * AUG_DICT["peak"].reshape(num_J, 1) * s_peak
    return peak_noise


def get_lfhp_cuda(shape=(8, 120), s_peak=3e-1, s_peak_mask=5e-3, num_J=12, device="cuda"):
    if num_J == 17:
        AUG_DICT = COCO17_AUG
        AUG_DICT_CUDA = COCO17_AUG_CUDA
    elif num_J == 12:
        AUG_DICT = COCO12_AUG
        AUG_DICT_CUDA = COCO12_AUG_CUDA
    else:
        raise ValueError(f"num_J: {num_J} is not supported")
    if "peak" not in AUG_DICT_CUDA:
        AUG_DICT_CUDA["pmask"] = AUG_DICT["pmask"].to(device)
        AUG_DICT_CUDA["peak"] = AUG_DICT["peak"].to(device).reshape(num_J, 1)

    pmask = AUG_DICT_CUDA["pmask"]
    peak = AUG_DICT_CUDA["peak"]
    peak_noise_mask = torch.rand(*shape, num_J, device=device) * pmask < s_peak_mask
    peak_noise = (
        peak_noise_mask.float().unsqueeze(-1).expand(-1, -1, -1, 3)
        * torch.randn(3, device=device)
        * peak
        * s_peak
    )
    return peak_noise


def get_bias(shape=(8, 120), s_bias=1e-1, num_J=12):
    """Bias noise modeling."""
    if num_J == 17:
        AUG_DICT = COCO17_AUG
    elif num_J == 12:
        AUG_DICT = COCO12_AUG
    else:
        raise ValueError(f"num_J: {num_J} is not supported")
    b, seq_len = shape
    bias_noise = (
        torch.normal(mean=torch.zeros((b, num_J, 3)), std=AUG_DICT["bias"].reshape(1, num_J, 1))
        * s_bias
    )
    bias_noise = bias_noise[:, None].expand(
        -1, seq_len, -1, -1
    )  # (B, L, J, 3), the whole sequence is moved by the same bias
    return bias_noise


def get_bias_cuda(shape=(8, 120), s_bias=1e-1, num_J=12, device="cuda"):
    if num_J == 17:
        AUG_DICT = COCO17_AUG
        AUG_DICT_CUDA = COCO17_AUG_CUDA
    elif num_J == 12:
        AUG_DICT = COCO12_AUG
        AUG_DICT_CUDA = COCO12_AUG_CUDA
    else:
        raise ValueError(f"num_J: {num_J} is not supported")
    if "bias" not in AUG_DICT_CUDA:
        AUG_DICT_CUDA["bias"] = AUG_DICT["bias"].to(device).reshape(1, num_J, 1)

    bias = AUG_DICT_CUDA["bias"]
    bias_noise = torch.randn((shape[0], num_J, 3), device=device) * bias * s_bias
    bias_noise = bias_noise[:, None].expand(-1, shape[1], -1, -1).clone()
    return bias_noise


def get_wham_aug_kp3d(shape=(8, 120), num_J=12, device="cuda"):
    aug = (
        get_bias_cuda(shape, num_J=num_J, device=device)
        + get_lfhp_cuda(shape, num_J=num_J, device=device)
        + get_jitter_cuda(shape, num_J=num_J, device=device)
    )
    return aug


def get_visible_mask(shape=(8, 120), num_J=12, s_mask=0.03, device=None):
    """Mask modeling."""
    # Per-frame and joint
    mask = torch.rand(*shape, num_J, device=device) < s_mask
    visible = (~mask).clone()  # (B, L, num_J)

    visible = visible.reshape(-1, num_J)  # (BL, num_J)
    for child in range(num_J):
        if num_J == 17:
            parent = COCO17_TREE[child]
        elif num_J == 12:
            parent = COCO12_TREE[child]
        else:
            raise ValueError(f"num_J: {num_J} is not supported")
        if parent == -1:
            continue
        if isinstance(parent, list):
            visible[:, child] *= visible[:, parent[0]] * visible[:, parent[1]]
        else:
            visible[:, child] *= visible[:, parent]
    visible = visible.reshape(*shape, num_J).clone()  # (B, L, J)
    return visible


def get_invisible_legs_mask(shape, num_J=12, s_mask=0.03, device=None):
    """
    Both legs are invisible for a random duration.
    """
    B, L = shape
    starts = torch.randint(0, L - 90, (B,), device=device)
    ends = starts + torch.randint(30, 90, (B,), device=device)
    mask_range = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    mask_to_apply = (mask_range >= starts.unsqueeze(1)) & (mask_range < ends.unsqueeze(1))
    mask_to_apply = mask_to_apply.unsqueeze(2).expand(-1, -1, num_J).clone()
    if num_J == 17:
        mask_to_apply[:, :, :11] = False  # only both legs are invisible
    elif num_J == 12:
        mask_to_apply[:, :, :6] = False  # only both legs are invisible
    else:
        raise ValueError(f"num_J: {num_J} is not supported")
    mask_to_apply = mask_to_apply & (torch.rand(B, 1, 1, device=device) < s_mask)
    return mask_to_apply


def randomly_modify_hands_legs(j3d, num_J=12, device=None):
    if num_J == 17:
        lhand = 9
        rhand = 10
        lleg = 15
        rleg = 16
    elif num_J == 12:
        lhand = 4
        rhand = 5
        lleg = 10
        rleg = 11
    else:
        raise ValueError(f"num_J: {num_J} is not supported")

    hands = [lhand, rhand]
    legs = [lleg, rleg]

    B, L, J, _ = j3d.shape
    p_switch_hand = 0.001
    p_switch_leg = 0.001
    p_wrong_hand0 = 0.001
    p_wrong_hand1 = 0.001
    p_wrong_leg0 = 0.001
    p_wrong_leg1 = 0.001

    mask = torch.rand(B, L, device=device) < p_switch_hand
    j3d[mask][:, hands] = j3d[mask][:, hands[::-1]]
    mask = torch.rand(B, L, device=device) < p_switch_leg
    j3d[mask][:, legs] = j3d[mask][:, legs[::-1]]
    mask = torch.rand(B, L, device=device) < p_wrong_hand0
    j3d[mask][:, lhand] = j3d[mask][:, rhand]
    mask = torch.rand(B, L, device=device) < p_wrong_hand1
    j3d[mask][:, rhand] = j3d[mask][:, lhand]
    mask = torch.rand(B, L, device=device) < p_wrong_leg0
    j3d[mask][:, lleg] = j3d[mask][:, rleg]
    mask = torch.rand(B, L, device=device) < p_wrong_leg1
    j3d[mask][:, rleg] = j3d[mask][:, lleg]

    return j3d
