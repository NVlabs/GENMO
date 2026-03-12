# Copyright (c) 2018-present, Facebook, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
# This code is derived from Facebook Research

import torch


def qinv(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qnormalize(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=1e-8)


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.reshape(-1, 4, 1), q.reshape(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qpow(q0, t, dtype=torch.float):
    """q0 : tensor of quaternions
    t: tensor of powers
    """
    q0 = qnormalize(q0)
    theta0 = torch.acos(q0[..., :1])

    # if theta0 is close to zero, add epsilon to avoid NaNs
    mask = (theta0 <= 10e-10) * (theta0 >= -10e-10)
    mask = mask.float()
    theta0 = (1 - mask) * theta0 + mask * 10e-10
    v0 = q0[..., 1:] / torch.sin(theta0)

    if isinstance(t, torch.Tensor):
        q = torch.zeros(t.shape + q0.shape, device=q0.device)
        theta = t.view(-1, 1) * theta0.view(1, -1)
    else:
        q = torch.zeros(q0.shape, device=q0.device)
        theta = t * theta0

    q[..., :1] = torch.cos(theta)
    q[..., 1:] = v0 * torch.sin(theta)

    return q.to(dtype)


def qslerp(q0, q1, t):
    """
    q0: starting quaternion
    q1: ending quaternion
    t: array of points along the way

    Returns:
    Tensor of Slerps: t.shape + q0.shape
    """
    q0 = qnormalize(q0)
    q1 = qnormalize(q1)
    q_ = qpow(qmul(q1, qinv(q0)), t)

    return qmul(q_, q0)


def qbetween(v0, v1):
    """
    find the quaternion used to rotate v0 to v1
    """
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    v = torch.cross(v0, v1, dim=-1)

    w = torch.sqrt((v0**2).sum(dim=-1, keepdim=True) * (v1**2).sum(dim=-1, keepdim=True)) + (
        v0 * v1
    ).sum(dim=-1, keepdim=True)
    y_vec = torch.zeros_like(v)
    y_vec[..., 1] = 1.0
    mask = v.norm(dim=-1) == 0
    mask2 = w.sum(dim=-1).abs() <= 1e-4
    mask = torch.logical_and(mask, mask2)
    v[mask] = y_vec[mask]

    return qnormalize(torch.cat([w, v], dim=-1))
