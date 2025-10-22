import numpy as np
import torch

if __name__ != "__main__":
    from .konia_transform import (
        angle_axis_to_quaternion,
        angle_axis_to_rotation_matrix,
        quaternion_to_angle_axis,
        quaternion_to_rotation_matrix,
        rotation_matrix_to_angle_axis,
        rotation_matrix_to_quaternion,
    )
else:
    from konia_transform import (
        angle_axis_to_quaternion,
        angle_axis_to_rotation_matrix,
        quaternion_to_rotation_matrix,
        rotation_matrix_to_angle_axis,
        rotation_matrix_to_quaternion,
    )


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:].clone()
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 0:1].clone() * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_angle(a, eps: float = 1e-6):
    shape = a.shape
    a = a.reshape(-1, 4)
    s = 2 * (a[:, 0] ** 2) - 1
    s = s.clamp(-1 + eps, 1 - eps)
    s = s.acos()
    return s.view(shape[:-1])


@torch.jit.script
def quat_angle_diff(quat1, quat2):
    return quat_angle(quat_mul(quat1, quat_conjugate(quat2)))


@torch.jit.script
def torch_safe_atan2(y, x, eps: float = 1e-8):
    y = y.clone()
    y[(y.abs() < eps) & (x.abs() < eps)] += eps
    return torch.atan2(y, x)


@torch.jit.script
def ypr_euler_from_quat(
    q, handle_singularity: bool = False, eps: float = 1e-6, singular_eps: float = 1e-6
):
    """
    convert quaternion to yaw-pitch-roll euler angles
    """
    yaw_atany = 2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2])
    yaw_atanx = 1 - 2 * (q[..., 2] * q[..., 2] + q[..., 3] * q[..., 3])
    roll_atany = 2 * (q[..., 0] * q[..., 1] + q[..., 2] * q[..., 3])
    roll_atanx = 1 - 2 * (q[..., 1] * q[..., 1] + q[..., 2] * q[..., 2])
    yaw = torch_safe_atan2(yaw_atany, yaw_atanx, eps)
    pitch = torch.asin(
        torch.clamp(
            2 * (q[..., 0] * q[..., 2] - q[..., 1] * q[..., 3]),
            min=-1 + eps,
            max=1 - eps,
        )
    )
    roll = torch_safe_atan2(roll_atany, roll_atanx, eps)

    if handle_singularity:
        """ handle two special cases """
        test = q[..., 0] * q[..., 2] - q[..., 1] * q[..., 3]
        # north pole, pitch ~= 90 degrees
        np_ind = test > 0.5 - singular_eps
        if torch.any(np_ind):
            # print('ypr_euler_from_quat singularity -- north pole!')
            roll[np_ind] = 0.0
            pitch[np_ind].clamp_max_(0.5 * np.pi)
            yaw_atany = q[..., 3][np_ind]
            yaw_atanx = q[..., 0][np_ind]
            yaw[np_ind] = 2 * torch_safe_atan2(yaw_atany, yaw_atanx, eps)
        # south pole, pitch ~= -90 degrees
        sp_ind = test < -0.5 + singular_eps
        if torch.any(sp_ind):
            # print('ypr_euler_from_quat singularity -- south pole!')
            roll[sp_ind] = 0.0
            pitch[sp_ind].clamp_min_(-0.5 * np.pi)
            yaw_atany = q[..., 3][sp_ind]
            yaw_atanx = q[..., 0][sp_ind]
            yaw[sp_ind] = 2 * torch_safe_atan2(yaw_atany, yaw_atanx, eps)

    return torch.stack([roll, pitch, yaw], dim=-1)


@torch.jit.script
def quat_from_ypr_euler(angles):
    """
    convert yaw-pitch-roll euler angles to quaternion
    """
    half_ang = angles * 0.5
    sin = torch.sin(half_ang)
    cos = torch.cos(half_ang)
    q = torch.stack(
        [
            cos[..., 0] * cos[..., 1] * cos[..., 2]
            + sin[..., 0] * sin[..., 1] * sin[..., 2],
            sin[..., 0] * cos[..., 1] * cos[..., 2]
            - cos[..., 0] * sin[..., 1] * sin[..., 2],
            cos[..., 0] * sin[..., 1] * cos[..., 2]
            + sin[..., 0] * cos[..., 1] * sin[..., 2],
            cos[..., 0] * cos[..., 1] * sin[..., 2]
            - sin[..., 0] * sin[..., 1] * cos[..., 2],
        ],
        dim=-1,
    )
    return q


def quat_between_two_vec(v1, v2, eps: float = 1e-6):
    """
    quaternion for rotating v1 to v2
    """
    orig_shape = v1.shape
    v1 = v1.reshape(-1, 3)
    v2 = v2.reshape(-1, 3)
    dot = (v1 * v2).sum(-1)
    cross = torch.cross(v1, v2, dim=-1)
    out = torch.cat([(1 + dot).unsqueeze(-1), cross], dim=-1)
    # handle v1 & v2 with same direction
    sind = dot > 1 - eps
    out[sind] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=v1.device)
    # handle v1 & v2 with opposite direction
    nind = dot < -1 + eps
    if torch.any(nind):
        vx = torch.tensor([1.0, 0.0, 0.0], device=v1.device)
        vxdot = (v1 * vx).sum(-1).abs()
        nxind = nind & (vxdot < 1 - eps)
        if torch.any(nxind):
            out[nxind] = angle_axis_to_quaternion(
                normalize(torch.cross(vx.expand_as(v1[nxind]), v1[nxind], dim=-1))
                * np.pi
            )
        # handle v1 & v2 with opposite direction and they are parallel to x axis
        pind = nind & (vxdot >= 1 - eps)
        if torch.any(pind):
            vy = torch.tensor([0.0, 1.0, 0.0], device=v1.device)
            out[pind] = angle_axis_to_quaternion(
                normalize(torch.cross(vy.expand_as(v1[pind]), v1[pind], dim=-1)) * np.pi
            )
    # normalize and reshape
    out = normalize(out).view(orig_shape[:-1] + (4,))
    return out


@torch.jit.script
def get_yaw(q, eps: float = 1e-6):
    yaw_atany = 2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2])
    yaw_atanx = 1 - 2 * (q[..., 2] * q[..., 2] + q[..., 3] * q[..., 3])
    yaw = torch_safe_atan2(yaw_atany, yaw_atanx, eps)
    return yaw


@torch.jit.script
def get_yaw_q(q):
    yaw = get_yaw(q)
    angle_axis = torch.cat(
        [torch.zeros(yaw.shape + (2,), device=q.device), yaw.unsqueeze(-1)], dim=-1
    )
    heading_q = angle_axis_to_quaternion(angle_axis)
    return heading_q


@torch.jit.script
def get_heading(q, eps: float = 1e-6):
    heading_atany = q[..., 3]
    heading_atanx = q[..., 0]
    heading = 2 * torch_safe_atan2(heading_atany, heading_atanx, eps)
    return heading


def get_heading_q(q):
    q_new = q.clone()
    q_new[..., 1] = 0
    q_new[..., 2] = 0
    q_new = normalize(q_new)
    return q_new


def get_y_heading_q(q):
    q_new = q.clone()
    q_new[..., 1] = 0
    q_new[..., 3] = 0
    q_new = normalize(q_new)
    return q_new


@torch.jit.script
def heading_to_vec(h_theta):
    v = torch.stack([torch.cos(h_theta), torch.sin(h_theta)], dim=-1)
    return v


@torch.jit.script
def vec_to_heading(h_vec):
    h_theta = torch_safe_atan2(h_vec[..., 1], h_vec[..., 0])
    return h_theta


@torch.jit.script
def heading_to_quat(h_theta):
    angle_axis = torch.cat(
        [
            torch.zeros(h_theta.shape + (2,), device=h_theta.device),
            h_theta.unsqueeze(-1),
        ],
        dim=-1,
    )
    heading_q = angle_axis_to_quaternion(angle_axis)
    return heading_q


def deheading_quat(q, heading_q=None):
    if heading_q is None:
        heading_q = get_heading_q(q)
    dq = quat_mul(quat_conjugate(heading_q), q)
    return dq


@torch.jit.script
def rotmat_to_rot6d(mat):
    rot6d = torch.cat([mat[..., 0], mat[..., 1]], dim=-1)
    return rot6d


@torch.jit.script
def rot6d_to_rotmat(rot6d, eps: float = 1e-8):
    a1 = rot6d[..., :3].clone()
    a2 = rot6d[..., 3:].clone()
    ind = torch.norm(a1, dim=-1) < eps
    a1[ind] = torch.tensor([1.0, 0.0, 0.0], device=a1.device)
    b1 = normalize(a1)

    b2 = normalize(a2 - (b1 * a2).sum(dim=-1).unsqueeze(-1) * b1)
    ind = torch.norm(b2, dim=-1) < eps
    b2[ind] = torch.tensor([0.0, 1.0, 0.0], device=b2.device)

    b3 = torch.cross(b1, b2, dim=-1)
    mat = torch.stack([b1, b2, b3], dim=-1)
    return mat


@torch.jit.script
def angle_axis_to_rot6d(aa):
    return rotmat_to_rot6d(angle_axis_to_rotation_matrix(aa))


@torch.jit.script
def rot6d_to_angle_axis(rot6d):
    return rotation_matrix_to_angle_axis(rot6d_to_rotmat(rot6d))


@torch.jit.script
def quat_to_rot6d(q):
    return rotmat_to_rot6d(quaternion_to_rotation_matrix(q))


@torch.jit.script
def rot6d_to_quat(rot6d):
    return rotation_matrix_to_quaternion(rot6d_to_rotmat(rot6d))


@torch.jit.script
def make_transform(rot, trans, rot_type: str = "rotmat"):
    if rot_type == "axis_angle":
        rot = angle_axis_to_rotation_matrix(rot)
    elif rot_type == "6d":
        rot = rot6d_to_rotmat(rot)
    transform = torch.eye(4).to(trans.device).repeat(rot.shape[:-2] + (1, 1))
    transform[..., :3, :3] = rot
    transform[..., :3, 3] = trans
    return transform


@torch.jit.script
def transform_trans(transform_mat, trans):
    trans = torch.cat((trans, torch.ones_like(trans[..., :1])), dim=-1)[..., None, :]
    while len(transform_mat.shape) < len(trans.shape):
        transform_mat = transform_mat.unsqueeze(-3)
    trans_new = torch.matmul(trans, transform_mat.transpose(-2, -1))[..., 0, :3]
    return trans_new


@torch.jit.script
def transform_rot(transform_mat, rot):
    rot_qmat = angle_axis_to_rotation_matrix(rot)
    while len(transform_mat.shape) < len(rot_qmat.shape):
        transform_mat = transform_mat.unsqueeze(-3)
    rot_qmat_new = torch.matmul(transform_mat[..., :3, :3], rot_qmat)
    rot_new = rotation_matrix_to_angle_axis(rot_qmat_new)
    return rot_new


@torch.jit.script
def inverse_transform(transform_mat):
    transform_inv = torch.zeros_like(transform_mat)
    transform_inv[..., :3, :3] = transform_mat[..., :3, :3].transpose(-2, -1)
    transform_inv[..., :3, 3] = -torch.matmul(
        transform_mat[..., :3, 3].unsqueeze(-2), transform_mat[..., :3, :3]
    ).squeeze(-2)
    transform_inv[..., 3, 3] = 1.0
    return transform_inv


def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    if len(S1.shape) > 3:
        orig_shape = S1.shape
        S1 = S1.reshape(-1, *S1.shape[-2:])
        S2 = S2.reshape(-1, *S2.shape[-2:])
    else:
        orig_shape = None

    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    if orig_shape is not None:
        S1_hat = S1_hat.reshape(orig_shape)

    return S1_hat
