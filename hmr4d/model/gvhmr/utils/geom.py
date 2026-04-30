import numpy as np
import torch


def norm_np_arr(arr):
    return arr / np.linalg.norm(arr)


def lookat_correct(eye, at, up):
    zaxis = norm_np_arr(at - eye)
    xaxis = norm_np_arr(np.cross(up, zaxis))
    yaxis = np.cross(zaxis, xaxis)

    _viewMatrix = np.array(
        [
            [xaxis[0], yaxis[0], zaxis[0], eye[0]],
            [xaxis[1], yaxis[1], zaxis[1], eye[1]],
            [xaxis[2], yaxis[2], zaxis[2], eye[2]],
            [0, 0, 0, 1],
        ]
    )
    return _viewMatrix


def spherical_to_cartesian(r, azimuth, elevation):
    # Convert degrees to radians
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)

    return np.array([x, y, z])


def safe_inverse(x, epsilon=1e-12):
    return x / (x**2 + epsilon)


class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        U, S, V = torch.linalg.svd(A, full_matrices=False)
        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = V.transpose(-2, -1)
        Ut = U.transpose(-2, -1)
        M = U.size(-2)
        N = V.size(-1)
        NS = S.size(-1)

        F = S.unsqueeze(-1) - S.unsqueeze(-2)
        F = safe_inverse(F)
        F.diagonal(dim1=-2, dim2=-1).fill_(0)

        G = S.unsqueeze(-1) + S.unsqueeze(-2)
        G.diagonal(dim1=-2, dim2=-1).fill_(np.inf)
        G = 1 / G

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F + G) * (UdU - UdU.transpose(-2, -1)) / 2
        Sv = (F - G) * (VdV - VdV.transpose(-2, -1)) / 2

        dA = U @ (Su + Sv + torch.diag_embed(dS)) @ Vt
        if M > NS:
            eye_M = torch.eye(M, dtype=dU.dtype, device=dU.device).expand(
                *dA.shape[:-2], M, M
            )
            dA = dA + (eye_M - U @ Ut) @ (dU / S.unsqueeze(-2)) @ Vt
        if N > NS:
            eye_N = torch.eye(N, dtype=dU.dtype, device=dU.device).expand(
                *dA.shape[:-2], N, N
            )
            dA = dA + (U / S.unsqueeze(-2)) @ dV.transpose(-2, -1) @ (eye_N - V @ Vt)
        return dA


def perspective_projection(points, cam_intrinsics, rotation=None, translation=None):
    K = cam_intrinsics
    if rotation is not None:
        points = torch.matmul(rotation, points.transpose(1, 2)).transpose(1, 2)
    if translation is not None:
        points = points + translation.unsqueeze(1)
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points.float())
    return projected_points[:, :, :-1]


def batch_triangulate(keypoints_, Pall, keypoints_pre=None, lamb=1e3):
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    if keypoints_.shape[-1] == 2:
        keypoints_ = np.concatenate(
            [keypoints_, np.ones((keypoints_.shape[0], keypoints_.shape[1], 1))],
            axis=-1,
        )
    v = (keypoints_[:, :, -1] > 0).sum(axis=0)
    valid_joint = np.where(v > 1)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0) / v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    if keypoints_pre is not None:
        # keypoints_pre: (nJoints, 4)
        B = np.eye(4)[None, :, :].repeat(A.shape[0], axis=0)
        B[:, :3, 3] = -keypoints_pre[valid_joint, :3]
        confpre = lamb * keypoints_pre[valid_joint, 3]
        # 1, 0, 0, -x0
        # 0, 1, 0, -y0
        # 0, 0, 1, -z0
        # 0, 0, 0,   0
        B[:, 3, 3] = 0
        B = B * confpre[:, None, None]
        A = np.hstack((A, B))
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d
    return result


def batch_triangulate_torch_single(keypoints_, Pall, keypoints_pre=None, lamb=1e3):
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    if keypoints_.shape[-1] == 2:
        keypoints_ = torch.cat(
            [
                keypoints_,
                torch.ones(
                    (keypoints_.shape[0], keypoints_.shape[1], 1),
                    device=keypoints_.device,
                ),
            ],
            dim=-1,
        )
    v = (keypoints_[:, :, -1] > 0).sum(dim=0)
    valid_joint = torch.where(v > 1)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(dim=0) / v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = torch.cat([Au, Av], dim=1)
    if keypoints_pre is not None:
        # keypoints_pre: (nJoints, 4)
        B = torch.eye(4, device=keypoints_.device)[None, :, :].repeat(A.shape[0], 1, 1)
        B[:, :3, 3] = -keypoints_pre[valid_joint, :3]
        confpre = lamb * keypoints_pre[valid_joint, 3]
        # 1, 0, 0, -x0
        # 0, 1, 0, -y0
        # 0, 0, 1, -z0
        # 0, 0, 0,   0
        B[:, 3, 3] = 0
        B = B * confpre[:, None, None]
        A = torch.cat((A, B), dim=1)
    u, s, v = torch.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = torch.zeros((keypoints_.shape[1], 4), device=keypoints_.device)
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d
    return result


def batch_triangulate_torch(
    keypoints_, Pall, keypoints_pre=None, lamb=1e3, use_custom_svd="default"
):
    # keypoints: (batch_size, nViews, nJoints, 3)
    # Pall: (batch_size, nViews, 3, 4)
    # A: (batch_size, nJoints, nViewsx2, 4), x: (batch_size, nJoints, 4, 1); b: (batch_size, nJoints, nViewsx2, 1)
    batch_size = keypoints_.shape[0]
    if keypoints_.shape[-1] == 2:
        keypoints_ = torch.cat(
            [
                keypoints_,
                torch.ones(
                    (batch_size, keypoints_.shape[1], keypoints_.shape[2], 1),
                    device=keypoints_.device,
                ),
            ],
            dim=-1,
        )
    valid_joint = keypoints_[:, :, :, -1].mean((0, 1)) > 0
    keypoints = keypoints_[:, :, valid_joint]
    conf3d = keypoints[:, :, :, -1].sum(dim=1) / keypoints.shape[1]
    # P2: P矩阵的最后一行：(batch_size, 1, nViews, 1, 4)
    P0 = Pall[:, None, :, 0, :]
    P1 = Pall[:, None, :, 1, :]
    P2 = Pall[:, None, :, 2, :]
    # uP2: x坐标乘上P2: (batch_size, nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, :, 0].permute(0, 2, 1)[:, :, :, None] * P2
    vP2 = keypoints[:, :, :, 1].permute(0, 2, 1)[:, :, :, None] * P2
    conf = keypoints[:, :, :, 2].permute(0, 2, 1)[:, :, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = torch.cat([Au, Av], dim=2)
    if keypoints_pre is not None:
        # keypoints_pre: (batch_size, nJoints, 4)
        B = torch.eye(4, device=keypoints_.device)[None, None, :, :].repeat(
            batch_size, A.shape[1], 1, 1
        )
        B[:, :, :3, 3] = -keypoints_pre[:, valid_joint, :3]
        confpre = lamb * keypoints_pre[:, valid_joint, 3]
        # 1, 0, 0, -x0
        # 0, 1, 0, -y0
        # 0, 0, 1, -z0
        # 0, 0, 0,   0
        B[:, :, 3, 3] = 0
        B = B * confpre[:, :, None, None]
        A = torch.cat((A, B), dim=2)
    if use_custom_svd == "custom":
        u, s, v = SVD.apply(A)
    elif use_custom_svd in {"gesvd"}:
        u, s, v = torch.linalg.svd(A, driver=use_custom_svd)
    else:
        u, s, v = torch.linalg.svd(A)
    X = v[:, :, -1, :]
    X = X / X[:, :, 3:]
    # out: (batch_size, nJoints, 4)
    result = torch.zeros((batch_size, keypoints_.shape[2], 4), device=keypoints_.device)
    result[:, valid_joint, :3] = X[:, :, :3]
    result[:, valid_joint, 3] = conf3d
    return result
