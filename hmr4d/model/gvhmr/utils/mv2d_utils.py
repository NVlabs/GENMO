import os
import shutil

import cv2
import numpy as np
import torch

from motiondiff.utils.torch_transform import (
    angle_axis_to_quaternion,
    angle_axis_to_rotation_matrix,
    inverse_transform,
    make_transform,
    quat_mul,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_angle_axis,
)
from motiondiff.utils.vis import images_to_video

from .geom import (
    batch_triangulate_torch,
    lookat_correct,
    perspective_projection,
    spherical_to_cartesian,
)

coco_joint_parent_mapping = {
    "L_Hip": "Pelvis",
    "R_Hip": "Pelvis",
    "L_Knee": "L_Hip",
    "R_Knee": "R_Hip",
    "L_Ankle": "L_Knee",
    "R_Ankle": "R_Knee",
    "L_Shoulder": "Pelvis",
    "R_Shoulder": "Pelvis",
    "L_Elbow": "L_Shoulder",
    "R_Elbow": "R_Shoulder",
    "L_Wrist": "L_Elbow",
    "R_Wrist": "R_Elbow",
    "L_Ear": "Nose",
    "R_Ear": "Nose",
    "L_Eye": "Nose",
    "R_Eye": "Nose",
    "Nose": "Pelvis",
}

coco_joint_names = [
    "Nose",
    "L_Eye",
    "R_Eye",
    "L_Ear",
    "R_Ear",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
    "Pelvis",
]
coco_joint_parents = [
    coco_joint_names.index(coco_joint_parent_mapping[j])
    if j in coco_joint_parent_mapping
    else -1
    for j in coco_joint_names
]


def create_cam(cam_intrinsics, azimuths, elevations, radius):
    P = []
    for azimuth, elevation, r in zip(
        azimuths.ravel(), elevations.ravel(), radius.ravel()
    ):
        eye = spherical_to_cartesian(r, azimuth, elevation)
        at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        c2w = torch.tensor(lookat_correct(eye, at, up)).float()
        w2c = torch.inverse(c2w)
        # Assuming cam_intrinsics is defined elsewhere in your context
        P_i = torch.matmul(cam_intrinsics, w2c[:3, :])
        P.append(P_i)
    P = torch.stack(P).view(*azimuths.shape, 3, 4)
    cam_dict = {"P": P}
    return cam_dict


def draw_mv_imgs(
    kp2d,
    joint_parents,
    img_w,
    img_h,
    bone_color=(0, 255, 0),
    kp_color=(255, 0, 0),
    base_img=None,
    mask=None,
    show_joints=None,
    add_coco_root=False,
    unnormalize=False,
    line_thickness=1,
    circle_radius=3,
    highlight_view=None,
):
    kp2d = kp2d.cpu()
    if add_coco_root and kp2d.shape[-2] == 17:
        kp2d = torch.cat(
            [kp2d, (kp2d[..., [11], :] + kp2d[..., [12], :]) * 0.5], dim=-2
        )
        if kp2d.shape[-1] == 3:
            kp2d[..., -1, -1] = 1.0
    if kp2d.shape[-1] == 3:
        if mask is None:
            mask = kp2d[..., 2] > 0.5
        kp2d = kp2d[..., :2]
    if unnormalize:
        kp2d = (kp2d + 1) * 0.5 * torch.tensor([img_w, img_h], device=kp2d.device)
    mv_imgs = []
    if base_img is not None:
        base_img = np.split(base_img, kp2d.shape[0], axis=1)
    for v in range(kp2d.shape[0]):
        if base_img is not None:
            img = base_img[v]
        else:
            img = np.zeros((img_h, img_w, 3), np.uint8)

        # Draw skeleton
        for i in range(kp2d.shape[1]):
            if joint_parents[i] == -1 or (
                show_joints is not None and i not in show_joints
            ):
                continue
            bone_color_v = (
                ((255, 255, 0))
                if mask is not None and torch.any(mask[v, i])
                else bone_color
            )
            parent = joint_parents[i]
            x1, y1 = kp2d[v, i].numpy()
            x2, y2 = kp2d[v, parent].numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(img, (x1, y1), (x2, y2), bone_color_v, line_thickness)

        # Draw keypoints
        for i in range(kp2d.shape[1]):
            if show_joints is not None and i not in show_joints:
                continue
            kp_color_v = (
                ((255, 255, 0))
                if mask is not None and torch.any(mask[v, i])
                else kp_color
            )
            x, y = kp2d[v, i].numpy()
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), circle_radius, kp_color_v, -1)

        if v == highlight_view:
            img = cv2.rectangle(img, (0, 0), (img_w, img_h), (0, 255, 0), 4)

        mv_imgs.append(img)
    mv_imgs = np.concatenate(mv_imgs, axis=1)
    return mv_imgs


def draw_motion_2d(
    motion_2d, fname, joint_parents, img_w, img_h, fps=30, show_joints=None, mask=None
):
    frame_dir = os.path.splitext(fname)[0] + "_frames"
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir, exist_ok=True)
    # create blank image
    for t in range(motion_2d.shape[0]):
        mv_imgs = draw_mv_imgs(
            motion_2d[t],
            joint_parents,
            img_w,
            img_h,
            show_joints=show_joints,
            mask=mask[t] if mask is not None else None,
        )
        cv2.imwrite(f"{frame_dir}/{t:06d}.jpg", mv_imgs[..., ::-1])

    images_to_video(frame_dir, fname, fps=fps)


def recon_from_2d_and_project(
    samples,
    cam_P,
    img_h,
    img_w,
    num_views,
    mean=None,
    std=None,
    use_custom_svd="default",
):
    """unnormalization"""
    samples_perm = samples.permute(0, 2, 3, 1)[
        :, 0
    ]  # now permuted to: [batch, 1, seq_len, nfeat]
    if mean is not None:
        samples_perm = samples_perm * std + mean
    local_kpt2d = (samples_perm + 1) * 0.5
    local_kpt2d = local_kpt2d.reshape(
        *local_kpt2d.shape[:2], num_views, -1, 2
    ) * torch.tensor([img_w, img_h], device=local_kpt2d.device)
    local_kpt2d[..., 1] = img_h - local_kpt2d[..., 1]
    local_kpt2d_rs = local_kpt2d.reshape(-1, *local_kpt2d.shape[2:])
    """ triangulate """
    cam_P = cam_P.repeat_interleave(local_kpt2d.shape[1], dim=0)
    kpt3d_recon = batch_triangulate_torch(
        local_kpt2d_rs, cam_P, use_custom_svd=use_custom_svd
    )
    """ reprojection """
    kpt3d_recon_pad = torch.cat(
        (kpt3d_recon[..., :3], torch.ones_like(kpt3d_recon[..., :1])), dim=-1
    )
    local_kpt2d_recon = (cam_P @ kpt3d_recon_pad[:, None].transpose(-1, -2)).transpose(
        -1, -2
    )
    local_kpt2d_recon = local_kpt2d_recon[..., :2] / local_kpt2d_recon[..., 2:]
    # sanity check: local_kpt2d_recon - local_kpt2d_rs ~= 0
    """ normalization """
    local_kpt2d_recon = local_kpt2d_recon.view_as(local_kpt2d)
    local_kpt2d_recon[..., 1] = img_h - local_kpt2d_recon[..., 1]
    samples_recon = (
        local_kpt2d_recon / torch.tensor([img_w, img_h]).to(local_kpt2d_recon) - 0.5
    ) * 2
    samples_recon = samples_recon.reshape(*samples_recon.shape[:2], -1)
    if mean is not None:
        samples_recon = (samples_recon - mean) / std
    samples_recon = samples_recon.permute(0, 2, 1).unsqueeze(2)
    kpt3d_recon = kpt3d_recon.reshape(local_kpt2d.shape[:2] + kpt3d_recon.shape[1:])
    return samples_recon, kpt3d_recon[..., :3]


def project_3d_to_2d(kpt3d_recon, cam_P, img_h, img_w, num_views, mean=None, std=None):
    """reprojection"""
    kpt3d_recon_shape = kpt3d_recon.shape
    kpt3d_recon = kpt3d_recon.reshape(-1, *kpt3d_recon.shape[2:])
    kpt3d_recon_pad = torch.cat(
        (kpt3d_recon[..., :3], torch.ones_like(kpt3d_recon[..., :1])), dim=-1
    )
    local_kpt2d_recon = (cam_P @ kpt3d_recon_pad[:, None].transpose(-1, -2)).transpose(
        -1, -2
    )
    local_kpt2d_recon = local_kpt2d_recon[..., :2] / local_kpt2d_recon[..., 2:]
    # sanity check: local_kpt2d_recon - local_kpt2d_rs ~= 0
    """ normalization """
    local_kpt2d_recon = local_kpt2d_recon.view(
        kpt3d_recon_shape[:2] + (num_views, -1, 2)
    )
    local_kpt2d_recon[..., 1] = img_h - local_kpt2d_recon[..., 1]
    samples_recon = (
        local_kpt2d_recon / torch.tensor([img_w, img_h]).to(local_kpt2d_recon) - 0.5
    ) * 2
    samples_recon = samples_recon.reshape(*samples_recon.shape[:2], -1)
    if mean is not None:
        samples_recon = (samples_recon - mean) / std
    samples_recon = samples_recon.permute(0, 2, 1).unsqueeze(2)
    return samples_recon


def generate_cam(mv2d_cam_params, num_views=4):
    device = mv2d_cam_params["elevations"].device
    orig_shape = mv2d_cam_params["elevations"].shape[:2]
    elevations = mv2d_cam_params["elevations"].view(-1, 1)
    radius = mv2d_cam_params["radius"].view(-1, 1)
    tilt = mv2d_cam_params["tilt"].view(-1, 1)

    def get_naive_intrinsics(res, focal_scale):
        # Assume 45 degree FOV
        img_w, img_h = res
        focal_length = (img_w * img_w + img_h * img_h) ** 0.5 * focal_scale
        cam_intrinsics = torch.eye(3).repeat(1, 1, 1).float()
        cam_intrinsics[:, 0, 0] = focal_length
        cam_intrinsics[:, 1, 1] = focal_length
        cam_intrinsics[:, 0, 2] = img_w / 2.0
        cam_intrinsics[:, 1, 2] = img_h / 2.0
        return cam_intrinsics

    def lookat_correct(eye, at, up):
        zaxis = (at - eye) / torch.norm(at - eye, dim=-1, keepdim=True)
        xaxis = torch.cross(up, zaxis, dim=-1)
        xaxis = xaxis / torch.norm(xaxis, dim=-1, keepdim=True)
        yaxis = torch.cross(zaxis, xaxis, dim=-1)
        view_matrix = torch.zeros((eye.shape[0], 4, 4), device=device)
        view_matrix[:, :3, 0] = xaxis
        view_matrix[:, :3, 1] = yaxis
        view_matrix[:, :3, 2] = zaxis
        view_matrix[:, :3, 3] = eye
        view_matrix[:, 3, 3] = 1.0
        return view_matrix

    def spherical_to_cartesian(r, azimuth, elevation):
        azimuth = torch.deg2rad(azimuth)
        elevation = torch.deg2rad(elevation)
        x = r * torch.cos(elevation) * torch.cos(azimuth)
        y = r * torch.cos(elevation) * torch.sin(azimuth)
        z = r * torch.sin(elevation)
        return torch.stack([x, y, z], dim=-1)

    # elevations = (torch.ones((1, num_views)) * 0.0).to(device)
    # radius = (torch.ones((1, num_views)) * 8.0).to(device)
    # tilt = (torch.ones((1, num_views))).to(device)
    elevations = elevations.repeat(1, num_views)
    radius = radius.repeat(1, num_views)
    tilt = tilt.repeat(1, num_views)
    azimuths = torch.linspace(0, 360, num_views + 1)[:num_views].to(device)
    azimuths = azimuths.unsqueeze(0).expand(elevations.shape[0], -1)
    eyes = spherical_to_cartesian(radius, azimuths, elevations)

    eyes_flat = eyes.view(-1, 3)
    at = torch.zeros((eyes_flat.shape[0], 3), device=device)
    up = torch.tensor([0.0, 0.0, 1.0], device=device)[None, :].expand(
        eyes_flat.shape[0], -1
    )
    c2w = lookat_correct(eyes_flat, at, up).reshape(eyes.shape[0], num_views, 4, 4)
    if torch.norm(tilt) > 0:
        tilt_rot = angle_axis_to_rotation_matrix(
            torch.stack(
                [torch.zeros_like(tilt), torch.zeros_like(tilt), torch.deg2rad(tilt)],
                dim=-1,
            )
        )
        c2w = c2w @ make_transform(tilt_rot, torch.zeros(3).to(tilt_rot))
    w2c = inverse_transform(c2w)

    cam_intrinsics = get_naive_intrinsics((1024, 1024), 2)
    intrinsics = (
        cam_intrinsics.to(device).unsqueeze(0).repeat(eyes.shape[0], num_views, 1, 1)
    )
    P = torch.matmul(intrinsics, w2c[..., :3, :])

    cam_dict = {
        "c2w": c2w.reshape(orig_shape + c2w.shape[1:]),
        "w2c": w2c.reshape(orig_shape + w2c.shape[1:]),
        "intrinsics": intrinsics.reshape(orig_shape + intrinsics.shape[1:]),
        "P": P.reshape(orig_shape + P.shape[1:]),
        "azimuths": azimuths.reshape(orig_shape + azimuths.shape[1:]),
        "elevations": elevations.reshape(orig_shape + elevations.shape[1:]),
        "radius": radius.reshape(orig_shape + radius.shape[1:]),
    }

    return cam_dict


def project_keypoints(kpt3d, cam_P, img_h):
    kpt3d_pad = torch.cat((kpt3d, torch.ones_like(kpt3d[..., :1])), dim=-1)
    local_kpt2d_new = (cam_P @ kpt3d_pad[:, :, None].transpose(-1, -2)).transpose(
        -1, -2
    )
    local_kpt2d_new = local_kpt2d_new[..., :2] / local_kpt2d_new[..., 2:]
    local_kpt2d_new[..., 1] = img_h - local_kpt2d_new[..., 1]
    return local_kpt2d_new


def recon_from_2d(samples, cam_P, img_h, img_w):
    """unnormalization"""
    local_kpt2d = (
        (samples + 1) * 0.5 * torch.tensor([img_w, img_h], device=samples.device)
    )
    local_kpt2d_orig = local_kpt2d.clone()
    local_kpt2d[..., 1] = img_h - local_kpt2d[..., 1]
    """ triangulate """
    local_kpt2d_rs = local_kpt2d.reshape(-1, *local_kpt2d.shape[2:])
    cam_P_rs = cam_P.reshape(-1, *cam_P.shape[2:])
    kpt3d_recon = batch_triangulate_torch(local_kpt2d_rs, cam_P_rs)[..., :3]
    kpt3d_recon = kpt3d_recon.reshape(local_kpt2d.shape[:2] + kpt3d_recon.shape[1:])
    """ reprojection """
    local_kpt2d_recon = project_keypoints(kpt3d_recon, cam_P, img_h)
    # sanity check: local_kpt2d_recon - local_kpt2d_orig ~= 0
    """ normalization """
    samples_recon = (
        local_kpt2d_recon / torch.tensor([img_w, img_h]).to(local_kpt2d_recon)
    ) * 2 - 1
    return kpt3d_recon, samples_recon
