# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Global motion utilities.
"""

import torch
import torch.nn.functional as F

from gem.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)

# Coordinate-system transform axis-angles (rotations around fixed axes)
_tsf_axisangle = {
    "ay->ay": [0, 0, 0],
    "any->ay": [0, 0, torch.pi],
    "az->ay": [-torch.pi / 2, 0, 0],
    "ay->any": [0, 0, torch.pi],
}


def get_local_transl_vel(transl, global_orient):
    """Translation velocity expressed in the body-local (root) coordinate frame.

    Args:
        transl: (*, L, 3)
        global_orient: (*, L, 3)  axis-angle
    Returns:
        local_transl_vel: (*, L, 3)  last frame is repeat of second-to-last
    """
    global_orient_R = axis_angle_to_matrix(global_orient)  # (*, L, 3, 3)
    transl_vel = transl[..., 1:, :] - transl[..., :-1, :]  # (*, L-1, 3)
    transl_vel = torch.cat([transl_vel, transl_vel[..., [-1], :]], dim=-2)  # (*, L, 3)
    # v_local = R^T @ v_global
    local_transl_vel = torch.einsum("...lij,...lj->...li", global_orient_R, transl_vel)
    return local_transl_vel


def rollout_local_transl_vel(local_transl_vel, global_orient, transl_0=None):
    """Integrate local-frame velocity back to global translation.

    Args:
        local_transl_vel: (*, L, 3)
        global_orient: (*, L, 3)  axis-angle
        transl_0: (*, 1, 3)  starting position; zeros if None
    Returns:
        transl: (*, L, 3)
    """
    global_orient_R = axis_angle_to_matrix(global_orient)
    transl_vel = torch.einsum("...lij,...lj->...li", global_orient_R, local_transl_vel)

    if transl_0 is None:
        transl_0 = transl_vel[..., :1, :].clone().detach().zero_()
    transl_ = torch.cat([transl_0, transl_vel[..., :-1, :]], dim=-2)
    transl = torch.cumsum(transl_, dim=-2)
    return transl


def get_static_joint_mask(w_j3d, vel_thr=0.25, smooth=False, repeat_last=False):
    """Boolean mask: True where a joint is approximately stationary (30 fps assumed).

    Args:
        w_j3d: (*, L, J, 3)
        vel_thr: velocity threshold in m/s  (HuMoR uses 0.15)
        smooth: unused, kept for API compatibility
        repeat_last: if True, repeat the last frame so shape matches w_j3d
    Returns:
        static_joint_mask: (*, L-1, J)  or (*, L, J) if repeat_last
    """
    joint_v = (w_j3d[..., 1:, :, :] - w_j3d[..., :-1, :, :]).pow(2).sum(-1).sqrt() / 0.033
    static_joint_mask = joint_v < vel_thr  # True = stationary

    if repeat_last:
        static_joint_mask = torch.cat([static_joint_mask, static_joint_mask[..., [-1], :]], dim=-2)
    return static_joint_mask


def get_c_rootparam(global_orient_w, transl_w, T_w2c, offset=None):
    """Convert world-space root parameters to camera-space.

    Args:
        global_orient_w: (*, 3)  axis-angle in world space
        transl_w: (*, 3)  translation in world space
        T_w2c: (*, 4, 4)  world-to-camera transform
        offset: (3,) optional offset added to transl_w before transforming
    Returns:
        global_orient_c: (*, 3)
        transl_c: (*, 3)
    """
    R_w2c = T_w2c[..., :3, :3]
    t_w2c = T_w2c[..., :3, 3]
    global_orient_R_c = R_w2c @ axis_angle_to_matrix(global_orient_w)
    global_orient_c = matrix_to_axis_angle(global_orient_R_c)
    tw = transl_w if offset is None else transl_w + offset
    transl_c = torch.einsum("...ij,...j->...i", R_w2c, tw) + t_w2c
    if offset is not None:
        transl_c = transl_c - offset
    return global_orient_c, transl_c


def get_R_c2gv(R_w2c, axis_gravity_in_w=None):
    """Rotation from camera frame to gravity-aligned view (gv).

    The gv y-axis points up (opposite gravity).  The gv z-axis is the
    camera forward direction projected onto the horizontal plane.

    Args:
        R_w2c: (*, 3, 3)  world-to-camera rotation
        axis_gravity_in_w: (3,) gravity direction in world coords,
            default [0, -1, 0] (gravity along -y / y-up world)
    Returns:
        R_c2gv: (*, 3, 3)
    """
    device = R_w2c.device
    if axis_gravity_in_w is None:
        axis_gravity_in_w = torch.tensor([0.0, -1.0, 0.0], device=device)
    g_c = torch.einsum("...ij,j->...i", R_w2c.float(), axis_gravity_in_w.to(device).float())

    y_c = -g_c / g_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Project camera forward [0,0,1] onto the plane perp to y_c
    fwd = torch.zeros(*R_w2c.shape[:-2], 3, device=device)
    fwd[..., 2] = 1.0
    fwd_proj = fwd - (fwd * y_c).sum(-1, keepdim=True) * y_c
    norm = fwd_proj.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(fwd_proj)
    fallback[..., 0] = 1.0
    z_c = torch.where(norm > 1e-6, fwd_proj / norm.clamp(min=1e-8), fallback)

    x_c = torch.linalg.cross(y_c, z_c, dim=-1)
    x_c = x_c / x_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return torch.stack([x_c, y_c, z_c], dim=-2)  # rows = gv axes in camera coords


def get_tgtcoord_rootparam(
    global_orient, transl, gravity_vec=None, tgt_gravity_vec=None, tsf="ay->ay"
):
    """Rotate root parameters to a target coordinate frame.

    Args:
        global_orient: (*, 3)  axis-angle
        transl: (*, 3)
        tsf: one of 'ay->ay', 'any->ay', 'az->ay', 'ay->any'
    Returns:
        tgt_global_orient: (*, 3)
        tgt_transl: (*, 3)
        R_g2tg: (3, 3)
    """
    device = global_orient.device
    aa = torch.tensor(_tsf_axisangle[tsf], dtype=torch.float32).to(device)
    R_g2tg = axis_angle_to_matrix(aa)  # (3, 3)

    global_orient_R = axis_angle_to_matrix(global_orient)  # (*, 3, 3)
    tgt_global_orient = matrix_to_axis_angle(R_g2tg @ global_orient_R)
    tgt_transl = torch.einsum("ij,...j->...i", R_g2tg, transl)
    return tgt_global_orient, tgt_transl, R_g2tg


def get_T_w2c_from_wcparams(global_orient_w, transl_w, global_orient_c, transl_c, offset):
    """Compute T_w2c from both world-space and camera-space root params.

    Args:
        global_orient_w: torch.tensor, (F, 3)
        transl_w: torch.tensor, (F, 3)
        global_orient_c: torch.tensor, (F, 3)
        transl_c: torch.tensor, (F, 3)
        offset: torch.tensor, (*, 3)
    Returns:
        T_w2c: torch.tensor, (F, 4, 4)
    """
    assert global_orient_w.shape == transl_w.shape and len(global_orient_w.shape) == 2
    assert global_orient_c.shape == transl_c.shape and len(global_orient_c.shape) == 2

    R_w = axis_angle_to_matrix(global_orient_w)  # (F, 3, 3)
    t_w = transl_w  # (F, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (F, 3, 3)
    t_c = transl_c  # (F, 3)

    R_w2c = R_c @ R_w.transpose(-1, -2)  # (F, 3, 3)
    t_w2c = t_c + offset - torch.einsum("fij,fj->fi", R_w2c, t_w + offset)  # (F, 3)
    T_w2c = torch.eye(4, device=global_orient_w.device).repeat(R_w.size(0), 1, 1)  # (F, 4, 4)
    T_w2c[..., :3, :3] = R_w2c  # (F, 3, 3)
    T_w2c[..., :3, 3] = t_w2c  # (F, 3)
    return T_w2c


# ──────────────────────────────────────────────────────────────────────
#  Streaming (frame-by-frame) rollout for real-time inference
# ──────────────────────────────────────────────────────────────────────


def _as_identity(R):
    """Snap near-identity rotations to exact identity to avoid numerical drift."""
    is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
    if is_I.any():
        R[is_I] = torch.eye(3, device=R.device)[None].expand(is_I.sum(), -1, -1)
    return R


@torch.no_grad()
def init_rollout_w_Rt_state(global_orient_gv_0, global_orient_c_0, device=None):
    """Initialize streaming rollout state for incremental world-pose computation.

    Args:
        global_orient_gv_0: (3,) or (B, 3)  axis-angle at first frame in GV coords
        global_orient_c_0:  (3,) or (B, 3)  axis-angle at first frame in camera coords
        device: optional torch device

    Returns:
        dict with keys R_t_to_0, global_orient_pre, transl_pre,
        last_global_orient_gv, last_global_orient_c
    """
    if global_orient_gv_0.dim() == 1:
        global_orient_gv_0 = global_orient_gv_0.unsqueeze(0)
    if global_orient_c_0.dim() == 1:
        global_orient_c_0 = global_orient_c_0.unsqueeze(0)
    if device is None:
        device = global_orient_gv_0.device
    B = global_orient_gv_0.shape[0]

    R_t_to_0 = torch.eye(3, device=device).reshape(1, 3, 3).repeat(B, 1, 1)
    R_gv0 = axis_angle_to_matrix(global_orient_gv_0)
    global_orient_pre_0 = matrix_to_axis_angle(R_t_to_0 @ R_gv0)
    transl_pre_0 = torch.zeros((B, 3), device=device)

    return {
        "R_t_to_0": R_t_to_0,
        "global_orient_pre": global_orient_pre_0,
        "transl_pre": transl_pre_0,
        "last_global_orient_gv": global_orient_gv_0.detach().clone(),
        "last_global_orient_c": global_orient_c_0.detach().clone(),
    }


@torch.no_grad()
def rollout_step_w_Rt(
    state,
    global_orient_gv_curr,
    global_orient_c_curr,
    cam_angvel_prev=None,
    local_transl_vel_prev=None,
    local_transl_vel_curr=None,
):
    """One-step streaming rollout to compute world pose at the current frame.

    Mirrors the per-step logic of ``get_body_params_w_Rt_v2`` but maintains
    a running state dict so we never re-process the full sequence.

    Args:
        state: dict from ``init_rollout_w_Rt_state`` or a previous call
        global_orient_gv_curr: (3,) or (B, 3)  GV-frame axis-angle at current frame
        global_orient_c_curr:  (3,) or (B, 3)  camera-frame axis-angle at current frame
        cam_angvel_prev:       (6,) or (B, 6)  6D rotation from t-1 to t (None → identity)
        local_transl_vel_prev: (3,) or (B, 3)  body-local velocity at t-1
        local_transl_vel_curr: (3,) or (B, 3)  body-local velocity at t (used when prev is None)

    Returns:
        body_params_curr: {"global_orient": (B, 3), "transl": (B, 3)} in AY coords
        new_state: updated state dict for the next call

    Note:
        ``local_transl_vel_curr`` is only consumed as a fallback when
        ``local_transl_vel_prev`` is None (typically at frame 0 of a stream).
        In that case the rollout uses the current-frame velocity to take its
        single forward step, which introduces a one-frame phase error in
        ``transl`` that disappears once frame-1 supplies a real ``prev``
        velocity. Inference-only (``@torch.no_grad``); for differentiable
        rollout, use ``get_body_params_w_Rt_v2``.
    """
    if global_orient_gv_curr.dim() == 1:
        global_orient_gv_curr = global_orient_gv_curr.unsqueeze(0)
    if global_orient_c_curr.dim() == 1:
        global_orient_c_curr = global_orient_c_curr.unsqueeze(0)
    if cam_angvel_prev is not None and cam_angvel_prev.dim() == 1:
        cam_angvel_prev = cam_angvel_prev.unsqueeze(0)
    if local_transl_vel_prev is not None and local_transl_vel_prev.dim() == 1:
        local_transl_vel_prev = local_transl_vel_prev.unsqueeze(0)
    if local_transl_vel_curr is not None and local_transl_vel_curr.dim() == 1:
        local_transl_vel_curr = local_transl_vel_curr.unsqueeze(0)

    device = global_orient_gv_curr.device
    B = global_orient_gv_curr.shape[0]

    # --- Incremental yaw rotation from camera angular velocity ---
    if cam_angvel_prev is None:
        R_t_to_tp1 = torch.eye(3, device=device).reshape(1, 3, 3).repeat(B, 1, 1)
    else:
        R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel_prev.to(device))
        R_t_to_tp1 = _as_identity(R_t_to_tp1)

    last_gv = state["last_global_orient_gv"]
    last_c = state["last_global_orient_c"]
    if last_gv.dim() == 1:
        last_gv = last_gv.unsqueeze(0)
    if last_c.dim() == 1:
        last_c = last_c.unsqueeze(0)

    R_gv_prev = axis_angle_to_matrix(last_gv.to(device))
    R_c_prev = axis_angle_to_matrix(last_c.to(device))
    R_c2gv_prev = R_gv_prev @ R_c_prev.mT

    R_cnext2gv = R_c2gv_prev @ R_t_to_tp1.mT
    view_axis_gv = R_c2gv_prev[..., 2]
    view_axis_gv_next = R_cnext2gv[..., 2]
    if view_axis_gv.dim() == 1:
        view_axis_gv = view_axis_gv.unsqueeze(0)
    if view_axis_gv_next.dim() == 1:
        view_axis_gv_next = view_axis_gv_next.unsqueeze(0)

    vec1 = view_axis_gv.clone()
    vec1[:, 1] = 0
    vec1 = F.normalize(vec1, dim=-1)
    vec2 = view_axis_gv_next.clone()
    vec2[:, 1] = 0
    vec2 = F.normalize(vec2, dim=-1)

    axis = vec2.cross(vec1, dim=-1)
    axis = F.normalize(axis, dim=-1)
    angle = torch.acos(torch.clamp((vec1 * vec2).sum(-1, keepdim=True), -1.0, 1.0))
    aa_tp1_to_t = axis * angle
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT

    R_t_to_0_new = state["R_t_to_0"].to(device) @ R_tp1_to_t
    R_gv_curr = axis_angle_to_matrix(global_orient_gv_curr)
    global_orient_pre_curr = matrix_to_axis_angle(R_t_to_0_new @ R_gv_curr)

    gop = state["global_orient_pre"]
    if gop.dim() == 1:
        gop = gop.unsqueeze(0)
    R_prev_world = axis_angle_to_matrix(gop.to(device))

    if local_transl_vel_prev is None:
        assert local_transl_vel_curr is not None
        delta_transl = torch.einsum("bij,bj->bi", R_prev_world, local_transl_vel_curr.to(device))
    else:
        delta_transl = torch.einsum("bij,bj->bi", R_prev_world, local_transl_vel_prev.to(device))

    transl_pre_prev = state["transl_pre"]
    if transl_pre_prev.dim() == 1:
        transl_pre_prev = transl_pre_prev.unsqueeze(0)
    transl_pre_curr = transl_pre_prev.to(device) + delta_transl

    global_orient_ay, transl_ay, _ = get_tgtcoord_rootparam(
        global_orient_pre_curr, transl_pre_curr, tsf="ay->ay"
    )

    body_params_curr = {"global_orient": global_orient_ay, "transl": transl_ay}
    new_state = {
        "R_t_to_0": R_t_to_0_new.detach(),
        "global_orient_pre": global_orient_pre_curr.detach(),
        "transl_pre": transl_pre_curr.detach(),
        "last_global_orient_gv": global_orient_gv_curr.detach().clone(),
        "last_global_orient_c": global_orient_c_curr.detach().clone(),
    }
    return body_params_curr, new_state
