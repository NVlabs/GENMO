import enum
import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from numpy import pi
from torch import Tensor, tensor

__all__ = [
    # functional api
    "rad2deg",
    "deg2rad",
    "pol2cart",
    "cart2pol",
    "convert_points_from_homogeneous",
    "convert_points_to_homogeneous",
    "convert_affinematrix_to_homography",
    "convert_affinematrix_to_homography3d",
    "angle_axis_to_rotation_matrix",
    "angle_axis_to_quaternion",
    "rotation_matrix_to_angle_axis",
    "rotation_matrix_to_quaternion",
    "quaternion_to_angle_axis",
    "quaternion_to_rotation_matrix",
    "quaternion_log_to_exp",
    "quaternion_exp_to_log",
    "denormalize_pixel_coordinates",
    "normalize_pixel_coordinates",
    "normalize_quaternion",
    "denormalize_pixel_coordinates3d",
    "normalize_pixel_coordinates3d",
]


def KORNIA_CHECK_SHAPE(x, shape: List[str]) -> None:
    # Desired shape here is list and not tuple, because torch.jit
    # does not like variable-length tuples
    KORNIA_CHECK_IS_TENSOR(x)
    if "*" == shape[0]:
        start_idx: int = 1
        x_shape_to_check = x.shape[-len(shape) - 1 :]
    else:
        start_idx = 0
        x_shape_to_check = x.shape

    for i in range(start_idx, len(shape)):
        # The voodoo below is because torchscript does not like
        # that dim can be both int and str
        dim_: str = shape[i]
        if not dim_.isnumeric():
            continue
        dim = int(dim_)
        if x_shape_to_check[i] != dim:
            raise TypeError(f"{x} shape should be must be [{shape}]. Got {x.shape}")


def KORNIA_CHECK_IS_TENSOR(x, msg: Optional[str] = None):
    if not isinstance(x, Tensor):
        raise TypeError(f"Not a Tensor type. Got: {type(x)}.\n{msg}")


class QuaternionCoeffOrder(enum.Enum):
    XYZW = "xyzw"
    WXYZ = "wxyz"


def torch_safe_atan2(y, x, eps: float = 1e-6):
    y = y.clone()
    y[(y.abs() < eps) & (x.abs() < eps)] += eps
    return torch.atan2(y, x)


def rad2deg(tensor: torch.Tensor) -> torch.Tensor:
    r"""Function that converts angles from radians to degrees.

    Args:
        tensor: Tensor of arbitrary shape.

    Returns:
        Tensor with same shape as input.

    Example:
        >>> input = torch.tensor(3.1415926535) * torch.rand(1, 3, 3)
        >>> output = rad2deg(input)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    return 180.0 * tensor / pi.to(tensor.device).type(tensor.dtype)


def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    r"""Function that converts angles from degrees to radians.

    Args:
        tensor: Tensor of arbitrary shape.

    Returns:
        tensor with same shape as input.

    Examples:
        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = deg2rad(input)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.0


def pol2cart(rho: torch.Tensor, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Function that converts polar coordinates to cartesian coordinates.

    Args:
        rho: Tensor of arbitrary shape.
        phi: Tensor of same arbitrary shape.

    Returns:
        Tensor with same shape as input.

    Example:
        >>> rho = torch.rand(1, 3, 3)
        >>> phi = torch.rand(1, 3, 3)
        >>> x, y = pol2cart(rho, phi)
    """
    if not (isinstance(rho, torch.Tensor) & isinstance(phi, torch.Tensor)):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}, {}".format(type(rho), type(phi))
        )

    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return x, y


def cart2pol(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function that converts cartesian coordinates to polar coordinates.

    Args:
        rho: Tensor of arbitrary shape.
        phi: Tensor of same arbitrary shape.
        eps: To avoid division by zero.

    Returns:
        Tensor with same shape as input.

    Example:
        >>> x = torch.rand(1, 3, 3)
        >>> y = torch.rand(1, 3, 3)
        >>> rho, phi = cart2pol(x, y)
    """
    if not (isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor)):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}, {}".format(type(x), type(y))
        )

    rho = torch.sqrt((x**2 + y**2).clamp_min(eps))
    phi = torch_safe_atan2(y, x)
    return rho, phi


def convert_points_from_homogeneous(
    points: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    r"""Function that converts points from homogeneous to Euclidean space.

    Args:
        points: the points to be transformed.
        eps: to avoid division by zero.

    Returns:
        the points in Euclidean space.

    Examples:
        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = convert_points_from_homogeneous(input)  # BxNx2
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(points)))

    if len(points.shape) < 2:
        raise ValueError(
            "Input must be at least a 2D tensor. Got {}".format(points.shape)
        )

    # we check for points at max_val
    z_vec: torch.Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: torch.Tensor = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

    return scale * points[..., :-1]


def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.

    Args:
        points: the points to be transformed.

    Returns:
        the points in homogeneous coordinates.

    Examples:
        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = convert_points_to_homogeneous(input)  # BxNx4
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(points)))
    if len(points.shape) < 2:
        raise ValueError(
            "Input must be at least a 2D tensor. Got {}".format(points.shape)
        )

    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)


def _convert_affinematrix_to_homography_impl(A: torch.Tensor) -> torch.Tensor:
    H: torch.Tensor = torch.nn.functional.pad(A, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0
    return H


def convert_affinematrix_to_homography(A: torch.Tensor) -> torch.Tensor:
    r"""Function that converts batch of affine matrices.

    Args:
        A: the affine matrix with shape :math:`(B,2,3)`.

    Returns:
         the homography matrix with shape of :math:`(B,3,3)`.

    Examples:
        >>> input = torch.rand(2, 2, 3)  # Bx2x3
        >>> output = convert_affinematrix_to_homography(input)  # Bx3x3
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(A)))
    if not (len(A.shape) == 3 and A.shape[-2:] == (2, 3)):
        raise ValueError("Input matrix must be a Bx2x3 tensor. Got {}".format(A.shape))
    return _convert_affinematrix_to_homography_impl(A)


def convert_affinematrix_to_homography3d(A: torch.Tensor) -> torch.Tensor:
    r"""Function that converts batch of 3d affine matrices.

    Args:
        A: the affine matrix with shape :math:`(B,3,4)`.

    Returns:
         the homography matrix with shape of :math:`(B,4,4)`.

    Examples:
        >>> input = torch.rand(2, 3, 4)  # Bx3x4
        >>> output = convert_affinematrix_to_homography3d(input)  # Bx4x4
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(A)))
    if not (len(A.shape) == 3 and A.shape[-2:] == (3, 4)):
        raise ValueError("Input matrix must be a Bx3x4 tensor. Got {}".format(A.shape))
    return _convert_affinematrix_to_homography_impl(A)


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    r"""Convert 3d vector of axis-angle rotation to 3x3 rotation matrix.

    Args:
        angle_axis: tensor of 3d vector of axis-angle rotations.

    Returns:
        tensor of 3x3 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 3, 3)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = angle_axis_to_rotation_matrix(input)  # Nx3x3
    """
    if not isinstance(angle_axis, torch.Tensor):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(angle_axis))
        )

    if not angle_axis.shape[-1] == 3:
        raise ValueError(
            "Input size must be a (*, 3) tensor. Got {}".format(angle_axis.shape)
        )

    orig_shape = angle_axis.shape
    angle_axis = angle_axis.reshape(-1, 3)

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2.clamp_min(eps))
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1
        )
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1
        )
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(3).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 3, 3).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = (
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    )

    rotation_matrix = rotation_matrix.view(orig_shape[:-1] + (3, 3))
    return rotation_matrix  # Nx3x3


def rotation_matrix_to_angle_axis(rotation_matrix: torch.Tensor) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to Rodrigues vector.

    Args:
        rotation_matrix: rotation matrix.

    Returns:
        Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 3)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 3)  # Nx3x3
        >>> output = rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(
            f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}"
        )

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}"
        )
    quaternion: torch.Tensor = rotation_matrix_to_quaternion(
        rotation_matrix, order=QuaternionCoeffOrder.WXYZ
    )
    return quaternion_to_angle_axis(quaternion, order=QuaternionCoeffOrder.WXYZ)


def safe_zero_division(
    numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1.0e-6
) -> torch.Tensor:
    denominator = denominator.clone()
    denominator[denominator.abs() < eps] += eps
    return numerator / denominator


def rotation_matrix_to_quaternion(
    rotation_matrix: torch.Tensor,
    eps: float = 1.0e-6,
    order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ,
) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (w, x, y, z) or (x, y, z, w) format.

    .. note::
        The (x, y, z, w) order is going to be deprecated in favor of efficiency.

    Args:
        rotation_matrix: the rotation matrix to convert.
        eps: small value to avoid zero division.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        the rotation in quaternion.

    Shape:
        - Input: :math:`(*, 3, 3)`
        - Output: :math:`(*, 4)`

    Example:
        >>> input = torch.rand(4, 3, 3)  # Nx3x3
        >>> output = rotation_matrix_to_quaternion(input, eps=torch.finfo(input.dtype).eps,
        ...                                        order=QuaternionCoeffOrder.WXYZ)  # Nx4
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(
            f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}"
        )

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}"
        )

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(
                f"order must be one of {QuaternionCoeffOrder.__members__.keys()}"
            )

    if order == QuaternionCoeffOrder.XYZW:
        warnings.warn(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )

    rotation_matrix_vec: torch.Tensor = rotation_matrix.reshape(
        *rotation_matrix.shape[:-2], 9
    )

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(
        rotation_matrix_vec, chunks=9, dim=-1
    )

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt((trace + 1.0).clamp_min(eps)) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1():
        sq = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(eps)) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2():
        sq = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(eps)) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3():
        sq = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(eps)) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        if order == QuaternionCoeffOrder.XYZW:
            return torch.cat((qx, qy, qz, qw), dim=-1)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion


def normalize_quaternion(
    quaternion: torch.Tensor, eps: float = 1.0e-12
) -> torch.Tensor:
    r"""Normalizes a quaternion.

    The quaternion should be in (x, y, z, w) format.

    Args:
        quaternion: a tensor containing a quaternion to be normalized.
          The tensor can be of shape :math:`(*, 4)`.
        eps: small value to avoid division by zero.

    Return:
        the normalized quaternion of shape :math:`(*, 4)`.

    Example:
        >>> quaternion = torch.tensor((1., 0., 1., 0.))
        >>> normalize_quaternion(quaternion)
        tensor([0.7071, 0.0000, 0.7071, 0.0000])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape)
        )
    return F.normalize(quaternion, p=2.0, dim=-1, eps=eps)


# based on:
# https://github.com/matthew-brett/transforms3d/blob/8965c48401d9e8e66b6a8c37c65f2fc200a076fa/transforms3d/quaternions.py#L101
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py#L247


def quaternion_to_rotation_matrix(
    quaternion: torch.Tensor, order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ
) -> torch.Tensor:
    r"""Converts a quaternion to a rotation matrix.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape :math:`(*, 4)`.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        the rotation matrix of shape :math:`(*, 3, 3)`.

    Example:
        >>> quaternion = torch.tensor((0., 0., 0., 1.))
        >>> quaternion_to_rotation_matrix(quaternion, order=QuaternionCoeffOrder.WXYZ)
        tensor([[-1.,  0.,  0.],
                [ 0., -1.,  0.],
                [ 0.,  0.,  1.]])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}"
        )

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(
                f"order must be one of {QuaternionCoeffOrder.__members__.keys()}"
            )

    if order == QuaternionCoeffOrder.XYZW:
        warnings.warn(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )

    # normalize the input quaternion
    quaternion_norm: torch.Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    if order == QuaternionCoeffOrder.XYZW:
        x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)
    else:
        w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: torch.Tensor = torch.tensor(1.0)

    matrix: torch.Tensor = torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    ).view(quaternion.shape[:-1] + (3, 3))

    # if len(quaternion.shape) == 1:
    #     matrix = torch.squeeze(matrix, dim=0)
    return matrix


def quaternion_to_angle_axis(
    quaternion: torch.Tensor,
    eps: float = 1.0e-6,
    order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ,
) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion: tensor with quaternions.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            f"Input must be a tensor of shape Nx4 or 4. Got {quaternion.shape}"
        )

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(
                f"order must be one of {QuaternionCoeffOrder.__members__.keys()}"
            )

    if order == QuaternionCoeffOrder.XYZW:
        warnings.warn(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )
    # unpack input and compute conversion
    q1: torch.Tensor = torch.tensor([])
    q2: torch.Tensor = torch.tensor([])
    q3: torch.Tensor = torch.tensor([])
    cos_theta: torch.Tensor = torch.tensor([])

    if order == QuaternionCoeffOrder.XYZW:
        q1 = quaternion[..., 0]
        q2 = quaternion[..., 1]
        q3 = quaternion[..., 2]
        cos_theta = quaternion[..., 3]
    else:
        cos_theta = quaternion[..., 0]
        q1 = quaternion[..., 1]
        q2 = quaternion[..., 2]
        q3 = quaternion[..., 3]

    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt((sin_squared_theta).clamp_min(eps))
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch_safe_atan2(-sin_theta, -cos_theta),
        torch_safe_atan2(sin_theta, cos_theta),
    )

    k_pos: torch.Tensor = safe_zero_division(two_theta, sin_theta, eps)
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def quaternion_log_to_exp(
    quaternion: torch.Tensor,
    eps: float = 1.0e-6,
    order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ,
) -> torch.Tensor:
    r"""Applies exponential map to log quaternion.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape :math:`(*, 3)`.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        the quaternion exponential map of shape :math:`(*, 4)`.

    Example:
        >>> quaternion = torch.tensor((0., 0., 0.))
        >>> quaternion_log_to_exp(quaternion, eps=torch.finfo(quaternion.dtype).eps,
        ...                       order=QuaternionCoeffOrder.WXYZ)
        tensor([1., 0., 0., 0.])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 3:
        raise ValueError(
            f"Input must be a tensor of shape (*, 3). Got {quaternion.shape}"
        )

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(
                f"order must be one of {QuaternionCoeffOrder.__members__.keys()}"
            )

    if order == QuaternionCoeffOrder.XYZW:
        warnings.warn(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )

    # compute quaternion norm
    norm_q: torch.Tensor = torch.norm(quaternion, p=2, dim=-1, keepdim=True).clamp(
        min=eps
    )

    # compute scalar and vector
    quaternion_vector: torch.Tensor = quaternion * torch.sin(norm_q) / norm_q
    quaternion_scalar: torch.Tensor = torch.cos(norm_q)

    # compose quaternion and return
    quaternion_exp: torch.Tensor = torch.tensor([])
    if order == QuaternionCoeffOrder.XYZW:
        quaternion_exp = torch.cat((quaternion_vector, quaternion_scalar), dim=-1)
    else:
        quaternion_exp = torch.cat((quaternion_scalar, quaternion_vector), dim=-1)

    return quaternion_exp


def quaternion_exp_to_log(
    quaternion: torch.Tensor,
    eps: float = 1.0e-6,
    order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ,
) -> torch.Tensor:
    r"""Applies the log map to a quaternion.

    The quaternion should be in (x, y, z, w) format.

    Args:
        quaternion: a tensor containing a quaternion to be converted.
          The tensor can be of shape :math:`(*, 4)`.
        eps: A small number for clamping.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        the quaternion log map of shape :math:`(*, 3)`.

    Example:
        >>> quaternion = torch.tensor((1., 0., 0., 0.))
        >>> quaternion_exp_to_log(quaternion, eps=torch.finfo(quaternion.dtype).eps,
        ...                       order=QuaternionCoeffOrder.WXYZ)
        tensor([0., 0., 0.])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(quaternion)}")

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            f"Input must be a tensor of shape (*, 4). Got {quaternion.shape}"
        )

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(
                f"order must be one of {QuaternionCoeffOrder.__members__.keys()}"
            )

    if order == QuaternionCoeffOrder.XYZW:
        warnings.warn(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )

    # unpack quaternion vector and scalar
    quaternion_vector: torch.Tensor = torch.tensor([])
    quaternion_scalar: torch.Tensor = torch.tensor([])

    if order == QuaternionCoeffOrder.XYZW:
        quaternion_vector = quaternion[..., 0:3]
        quaternion_scalar = quaternion[..., 3:4]
    else:
        quaternion_scalar = quaternion[..., 0:1]
        quaternion_vector = quaternion[..., 1:4]

    # compute quaternion norm
    norm_q: torch.Tensor = torch.norm(
        quaternion_vector, p=2, dim=-1, keepdim=True
    ).clamp(min=eps)

    # apply log map
    quaternion_log: torch.Tensor = (
        quaternion_vector
        * torch.acos(torch.clamp(quaternion_scalar, min=-1.0 + eps, max=1.0 - eps))
        / norm_q
    )

    return quaternion_log


# based on:
# https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L138


def angle_axis_to_quaternion(
    angle_axis: torch.Tensor,
    eps: float = 1.0e-6,
    order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ,
) -> torch.Tensor:
    r"""Convert an angle axis to a quaternion.

    The quaternion vector has components in (x, y, z, w) or (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis: tensor with angle axis.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        tensor with quaternion.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)`

    Example:
        >>> angle_axis = torch.rand(2, 3)  # Nx3
        >>> quaternion = angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)  # Nx4
    """

    if not angle_axis.shape[-1] == 3:
        raise ValueError(
            f"Input must be a tensor of shape Nx3 or 3. Got {angle_axis.shape}"
        )

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(
                f"order must be one of {QuaternionCoeffOrder.__members__.keys()}"
            )

    if order == QuaternionCoeffOrder.XYZW:
        warnings.warn(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )

    # unpack input and compute conversion
    a0: torch.Tensor = angle_axis[..., 0:1]
    a1: torch.Tensor = angle_axis[..., 1:2]
    a2: torch.Tensor = angle_axis[..., 2:3]
    theta_squared: torch.Tensor = a0 * a0 + a1 * a1 + a2 * a2

    theta: torch.Tensor = torch.sqrt((theta_squared).clamp_min(eps))
    half_theta: torch.Tensor = theta * 0.5

    mask: torch.Tensor = theta_squared > 0.0
    ones: torch.Tensor = torch.ones_like(half_theta)

    k_neg: torch.Tensor = 0.5 * ones
    k_pos: torch.Tensor = safe_zero_division(torch.sin(half_theta), theta, eps)
    k: torch.Tensor = torch.where(mask, k_pos, k_neg)
    w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)

    quaternion: torch.Tensor = torch.zeros(
        size=angle_axis.shape[:-1] + (4,),
        dtype=angle_axis.dtype,
        device=angle_axis.device,
    )
    if order == QuaternionCoeffOrder.XYZW:
        quaternion[..., 0:1] = a0 * k
        quaternion[..., 1:2] = a1 * k
        quaternion[..., 2:3] = a2 * k
        quaternion[..., 3:4] = w
    else:
        quaternion[..., 1:2] = a0 * k
        quaternion[..., 2:3] = a1 * k
        quaternion[..., 3:4] = a2 * k
        quaternion[..., 0:1] = w
    return quaternion


# based on:
# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L65-L71


def normalize_pixel_coordinates(
    pixel_coordinates: torch.Tensor, height: int, width: int, eps: float = 1e-8
) -> torch.Tensor:
    r"""Normalize pixel coordinates between -1 and 1.

    Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

    Args:
        pixel_coordinates: the grid with pixel coordinates. Shape can be :math:`(*, 2)`.
        width: the maximum width in the x-axis.
        height: the maximum height in the y-axis.
        eps: safe division by zero.

    Return:
        the normalized pixel coordinates.
    """
    if pixel_coordinates.shape[-1] != 2:
        raise ValueError(
            "Input pixel_coordinates must be of shape (*, 2). Got {}".format(
                pixel_coordinates.shape
            )
        )
    # compute normalization factor
    hw: torch.Tensor = torch.stack(
        [
            torch.tensor(
                width, device=pixel_coordinates.device, dtype=pixel_coordinates.dtype
            ),
            torch.tensor(
                height, device=pixel_coordinates.device, dtype=pixel_coordinates.dtype
            ),
        ]
    )

    factor: torch.Tensor = torch.tensor(
        2.0, device=pixel_coordinates.device, dtype=pixel_coordinates.dtype
    ) / (hw - 1).clamp(eps)

    return factor * pixel_coordinates - 1


def denormalize_pixel_coordinates(
    pixel_coordinates: torch.Tensor, height: int, width: int, eps: float = 1e-8
) -> torch.Tensor:
    r"""Denormalize pixel coordinates.

    The input is assumed to be -1 if on extreme left, 1 if on extreme right (x = w-1).

    Args:
        pixel_coordinates: the normalized grid coordinates. Shape can be :math:`(*, 2)`.
        width: the maximum width in the x-axis.
        height: the maximum height in the y-axis.
        eps: safe division by zero.

    Return:
        the denormalized pixel coordinates.
    """
    if pixel_coordinates.shape[-1] != 2:
        raise ValueError(
            "Input pixel_coordinates must be of shape (*, 2). Got {}".format(
                pixel_coordinates.shape
            )
        )
    # compute normalization factor
    hw: torch.Tensor = (
        torch.stack([torch.tensor(width), torch.tensor(height)])
        .to(pixel_coordinates.device)
        .to(pixel_coordinates.dtype)
    )

    factor: torch.Tensor = torch.tensor(2.0) / (hw - 1).clamp(eps)

    return torch.tensor(1.0) / factor * (pixel_coordinates + 1)


def normalize_pixel_coordinates3d(
    pixel_coordinates: torch.Tensor,
    depth: int,
    height: int,
    width: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Normalize pixel coordinates between -1 and 1.

    Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

    Args:
        pixel_coordinates: the grid with pixel coordinates. Shape can be :math:`(*, 3)`.
        depth: the maximum depth in the z-axis.
        height: the maximum height in the y-axis.
        width: the maximum width in the x-axis.
        eps: safe division by zero.

    Return:
        the normalized pixel coordinates.
    """
    if pixel_coordinates.shape[-1] != 3:
        raise ValueError(
            "Input pixel_coordinates must be of shape (*, 3). Got {}".format(
                pixel_coordinates.shape
            )
        )
    # compute normalization factor
    dhw: torch.Tensor = (
        torch.stack([torch.tensor(depth), torch.tensor(width), torch.tensor(height)])
        .to(pixel_coordinates.device)
        .to(pixel_coordinates.dtype)
    )

    factor: torch.Tensor = torch.tensor(2.0) / (dhw - 1).clamp(eps)

    return factor * pixel_coordinates - 1


def denormalize_pixel_coordinates3d(
    pixel_coordinates: torch.Tensor,
    depth: int,
    height: int,
    width: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Denormalize pixel coordinates.

    The input is assumed to be -1 if on extreme left, 1 if on extreme right (x = w-1).

    Args:
        pixel_coordinates: the normalized grid coordinates. Shape can be :math:`(*, 3)`.
        depth: the maximum depth in the x-axis.
        height: the maximum height in the y-axis.
        width: the maximum width in the x-axis.
        eps: safe division by zero.

    Return:
        the denormalized pixel coordinates.
    """
    if pixel_coordinates.shape[-1] != 3:
        raise ValueError(
            "Input pixel_coordinates must be of shape (*, 3). Got {}".format(
                pixel_coordinates.shape
            )
        )
    # compute normalization factor
    dhw: torch.Tensor = (
        torch.stack([torch.tensor(depth), torch.tensor(width), torch.tensor(height)])
        .to(pixel_coordinates.device)
        .to(pixel_coordinates.dtype)
    )

    factor: torch.Tensor = torch.tensor(2.0) / (dhw - 1).clamp(eps)

    return torch.tensor(1.0) / factor * (pixel_coordinates + 1)


def Rt_to_matrix4x4(R: Tensor, t: Tensor) -> Tensor:
    r"""Combines 3x3 rotation matrix R and 1x3 translation vector t into 4x4 extrinsics.
    Args:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.
    Returns:
        the extrinsics :math:`(B, 4, 4)`.
    Example:
        >>> R, t = torch.eye(3)[None], torch.ones(3).reshape(1, 3, 1)
        >>> Rt_to_matrix4x4(R, t)
        tensor([[[1., 0., 0., 1.],
                 [0., 1., 0., 1.],
                 [0., 0., 1., 1.],
                 [0., 0., 0., 1.]]])
    """
    KORNIA_CHECK_SHAPE(R, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(t, ["B", "3", "1"])
    Rt = torch.cat([R, t], dim=2)
    return convert_affinematrix_to_homography3d(Rt)


def matrix4x4_to_Rt(extrinsics: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Converts 4x4 extrinsics into 3x3 rotation matrix R and 1x3 translation vector ts.
    Args:
        extrinsics: pose matrix :math:`(B, 4, 4)`.
    Returns:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.
    Example:
        >>> ext = torch.eye(4)[None]
        >>> matrix4x4_to_Rt(ext)
        (tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]]), tensor([[[0.],
                 [0.],
                 [0.]]]))
    """
    KORNIA_CHECK_SHAPE(extrinsics, ["B", "4", "4"])
    R, t = extrinsics[:, :3, :3], extrinsics[:, :3, 3:]
    return R, t


def camtoworld_graphics_to_vision_4x4(extrinsics_graphics: Tensor) -> Tensor:
    r"""Converts graphics coordinate frame (e.g. OpenGL) to vision coordinate frame (e.g. OpenCV.), , i.e. flips y
    and z axis. Graphics convention: [+x, +y, +z] == [right, up, backwards]. Vision convention: [+x, +y, +z] ==
    [right, down, forwards]
    Args:
        extrinsics: pose matrix :math:`(B, 4, 4)`.
    Returns:
        extrinsics: pose matrix :math:`(B, 4, 4)`.
    Example:
        >>> ext = torch.eye(4)[None]
        >>> camtoworld_graphics_to_vision_4x4(ext)
        tensor([[[ 1.,  0.,  0.,  0.],
                 [ 0., -1.,  0.,  0.],
                 [ 0.,  0., -1.,  0.],
                 [ 0.,  0.,  0.,  1.]]])
    """
    KORNIA_CHECK_SHAPE(extrinsics_graphics, ["B", "4", "4"])
    invert_yz = tensor(
        [[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.0]]],
        dtype=extrinsics_graphics.dtype,
        device=extrinsics_graphics.device,
    )
    return extrinsics_graphics @ invert_yz


def camtoworld_graphics_to_vision_Rt(R: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Converts graphics coordinate frame (e.g. OpenGL) to vision coordinate frame (e.g. OpenCV.), , i.e. flips y
    and z axis. Graphics convention: [+x, +y, +z] == [right, up, backwards]. Vision convention: [+x, +y, +z] ==
    [right, down, forwards]
    Args:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.
    Returns:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.
    Example:
        >>> R, t = torch.eye(3)[None], torch.ones(3).reshape(1, 3, 1)
        >>> camtoworld_graphics_to_vision_Rt(R, t)
        (tensor([[[ 1.,  0.,  0.],
                 [ 0., -1.,  0.],
                 [ 0.,  0., -1.]]]), tensor([[[1.],
                 [1.],
                 [1.]]]))
    """
    KORNIA_CHECK_SHAPE(R, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(t, ["B", "3", "1"])
    mat4x4 = camtoworld_graphics_to_vision_4x4(Rt_to_matrix4x4(R, t))
    return matrix4x4_to_Rt(mat4x4)


def camtoworld_vision_to_graphics_4x4(extrinsics_vision: Tensor) -> Tensor:
    r"""Converts vision coordinate frame (e.g. OpenCV) to graphics coordinate frame (e.g. OpenGK.), i.e. flips y and
    z axis Graphics convention: [+x, +y, +z] == [right, up, backwards]. Vision convention: [+x, +y, +z] == [right,
    down, forwards]
    Args:
        extrinsics: pose matrix :math:`(B, 4, 4)`.
    Returns:
        extrinsics: pose matrix :math:`(B, 4, 4)`.
    Example:
        >>> ext = torch.eye(4)[None]
        >>> camtoworld_vision_to_graphics_4x4(ext)
        tensor([[[ 1.,  0.,  0.,  0.],
                 [ 0., -1.,  0.,  0.],
                 [ 0.,  0., -1.,  0.],
                 [ 0.,  0.,  0.,  1.]]])
    """
    KORNIA_CHECK_SHAPE(extrinsics_vision, ["B", "4", "4"])
    invert_yz = torch.tensor(
        [[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.0]]],
        dtype=extrinsics_vision.dtype,
        device=extrinsics_vision.device,
    )
    return extrinsics_vision @ invert_yz


def camtoworld_vision_to_graphics_Rt(R: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Converts graphics coordinate frame (e.g. OpenGL) to vision coordinate frame (e.g. OpenCV.), , i.e. flips y
    and z axis. Graphics convention: [+x, +y, +z] == [right, up, backwards]. Vision convention: [+x, +y, +z] ==
    [right, down, forwards]
    Args:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.
    Returns:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.
    Example:
        >>> R, t = torch.eye(3)[None], torch.ones(3).reshape(1, 3, 1)
        >>> camtoworld_vision_to_graphics_Rt(R, t)
        (tensor([[[ 1.,  0.,  0.],
                 [ 0., -1.,  0.],
                 [ 0.,  0., -1.]]]), tensor([[[1.],
                 [1.],
                 [1.]]]))
    """
    KORNIA_CHECK_SHAPE(R, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(t, ["B", "3", "1"])
    mat4x4 = camtoworld_vision_to_graphics_4x4(Rt_to_matrix4x4(R, t))
    return matrix4x4_to_Rt(mat4x4)


def camtoworld_to_worldtocam_Rt(R: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Converts camtoworld, i.e. projection from camera coordinate system to world coordinate system, to worldtocam
    frame i.e. projection from world to the camera coordinate system (used in Colmap).
    See
    long-url: https://colmap.github.io/format.html#output-format
    Args:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.
    Returns:
        Rinv: Rotation matrix, :math:`(B, 3, 3).`
        tinv: Translation matrix :math:`(B, 3, 1)`.
    Example:
        >>> R, t = torch.eye(3)[None], torch.ones(3).reshape(1, 3, 1)
        >>> camtoworld_to_worldtocam_Rt(R, t)
        (tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]]), tensor([[[-1.],
                 [-1.],
                 [-1.]]]))
    """
    KORNIA_CHECK_SHAPE(R, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(t, ["B", "3", "1"])

    R_inv = R.transpose(1, 2)
    new_t: Tensor = -R_inv @ t

    return (R_inv, new_t)


def worldtocam_to_camtoworld_Rt(R: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Converts worldtocam frame i.e. projection from world to the camera coordinate system (used in Colmap) to
    camtoworld, i.e. projection from camera coordinate system to world coordinate system.
    Args:
        R: Rotation matrix, :math:`(B, 3, 3).`
        t: Translation matrix :math:`(B, 3, 1)`.
    Returns:
        Rinv: Rotation matrix, :math:`(B, 3, 3).`
        tinv: Translation matrix :math:`(B, 3, 1)`.
    Example:
        >>> R, t = torch.eye(3)[None], torch.ones(3).reshape(1, 3, 1)
        >>> worldtocam_to_camtoworld_Rt(R, t)
        (tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]]]), tensor([[[-1.],
                 [-1.],
                 [-1.]]]))
    """
    KORNIA_CHECK_SHAPE(R, ["B", "3", "3"])
    KORNIA_CHECK_SHAPE(t, ["B", "3", "1"])

    R_inv = R.transpose(1, 2)
    new_t: Tensor = -R_inv @ t

    return (R_inv, new_t)


def ARKitQTVecs_to_ColmapQTVecs(qvec: Tensor, tvec: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Converts output of Apple ARKit screen pose (in quaternion representation) to the camera-to-world
    transformation, expected by Colmap, also in quaternion representation.
    Args:
        qvec: ARKit rotation quaternion :math:`(B, 4)`, [x, y, z, w] format.
        tvec: translation vector :math:`(B, 3, 1)`, [x, y, z]
    Returns:
        qvec: Colmap rotation quaternion :math:`(B, 4)`, [w, x, y, z] format.
        tvec: translation vector :math:`(B, 3, 1)`, [x, y, z]
    Example:
        >>> q, t = torch.tensor([0, 1, 0, 1.])[None], torch.ones(3).reshape(1, 3, 1)
        >>> ARKitQTVecs_to_ColmapQTVecs(q, t)
        (tensor([[0.7071, 0.0000, 0.7071, 0.0000]]), tensor([[[-1.0000],
                 [-1.0000],
                 [ 1.0000]]]))
    """
    # ToDo:  integrate QuaterniaonAPI

    Rcg = quaternion_to_rotation_matrix(qvec, order=QuaternionCoeffOrder.WXYZ)
    Rcv, Tcv = camtoworld_graphics_to_vision_Rt(Rcg, tvec)
    R_colmap, t_colmap = camtoworld_to_worldtocam_Rt(Rcv, Tcv)
    t_colmap = t_colmap.reshape(-1, 3, 1)
    q_colmap = rotation_matrix_to_quaternion(
        R_colmap.contiguous(), order=QuaternionCoeffOrder.WXYZ
    )
    return q_colmap, t_colmap


def rotation_about_x(angle: float) -> Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor(
        [[1, 0, 0, 0], [0, cos, -sin, 0], [0, sin, cos, 0], [0, 0, 0, 1]]
    )


def rotation_about_y(angle: float) -> Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor(
        [[cos, 0, sin, 0], [0, 1, 0, 0], [-sin, 0, cos, 0], [0, 0, 0, 1]]
    )


def rotation_about_z(angle: float) -> Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor(
        [[cos, -sin, 0, 0], [sin, cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )


def mat4x4_inverse(mat4x4):
    R = mat4x4[:, :3, :3].clone()
    t = mat4x4[:, :3, 3].clone()
    R = R.transpose(-1, -2)
    t = -R @ t[..., None]
    mat4x4[:, :3, :3] = R
    mat4x4[:, :3, 3] = t[..., 0]
    return mat4x4


def gl_world_to_camera(camtoworld: Tensor) -> Tensor:
    orig_shape = camtoworld.shape[:-2]
    camtoworld = camtoworld.reshape(-1, 4, 4)
    camtoworld = camtoworld @ rotation_about_x(math.pi)[None].to(camtoworld)
    w2c = mat4x4_inverse(camtoworld)
    return w2c.reshape(*orig_shape, 4, 4)


def gl_camera_to_world(worldtocam: Tensor) -> Tensor:
    orig_shape = worldtocam.shape[:-2]
    worldtocam = mat4x4_inverse(worldtocam.reshape(-1, 4, 4))
    c2w = worldtocam @ rotation_about_x(math.pi)[None].to(worldtocam)
    return c2w.reshape(*orig_shape, 4, 4)


def gl_camera_to_world_wo_x_rot(worldtocam: Tensor) -> Tensor:
    orig_shape = worldtocam.shape[:-2]
    c2w = mat4x4_inverse(worldtocam.reshape(-1, 4, 4))
    # c2w = worldtocam @ rotation_about_x(math.pi)[None].to(worldtocam)
    return c2w.reshape(*orig_shape, 4, 4)


def slam_to_opencv(slam_cam_poses: Tensor) -> Tensor:
    assert slam_cam_poses.shape[-2:] == (4, 4), (
        f"Expected tensor shape to be (4,4) but found {slam_cam_poses.shape[-2:]}"
    )
    orig_shape = slam_cam_poses.shape[:-2]
    slam_cam_poses = slam_cam_poses.reshape(-1, 4, 4)

    tmp = slam_cam_poses.clone()
    slam_cam_poses = gl_world_to_camera(slam_cam_poses)

    # Important bug fix here
    R, t = matrix4x4_to_Rt(slam_cam_poses)
    R = (
        R
        @ rotation_about_z(math.pi)[None, :3, :3]
        @ rotation_about_x(-math.pi / 2)[None, :3, :3]
    )

    slam_cam_poses = Rt_to_matrix4x4(R, t)

    slam_cam_poses[:, 1:3] *= -1

    slam_cam_poses = slam_cam_poses.reshape(*orig_shape, 4, 4)

    return slam_cam_poses
