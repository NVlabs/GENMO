import numpy as np
import torch
import os
import cv2
import shutil
from motiondiff.utils.vis import images_to_video
from .geom import batch_triangulate_torch, perspective_projection, lookat_correct, spherical_to_cartesian


coco_joint_parent_mapping = {
    'L_Hip': 'Pelvis',
    'R_Hip': 'Pelvis',
    'L_Knee': 'L_Hip',
    'R_Knee': 'R_Hip',
    'L_Ankle': 'L_Knee',
    'R_Ankle': 'R_Knee',
    'L_Shoulder': 'Pelvis',
    'R_Shoulder': 'Pelvis',
    'L_Elbow': 'L_Shoulder',
    'R_Elbow': 'R_Shoulder',
    'L_Wrist': 'L_Elbow',
    'R_Wrist': 'R_Elbow',
    'L_Ear': 'Nose',
    'R_Ear': 'Nose',
    'L_Eye': 'Nose',
    'R_Eye': 'Nose',
    'Nose': 'Pelvis',
}

coco_joint_names = ['Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis']
coco_joint_parents = [coco_joint_names.index(coco_joint_parent_mapping[j]) if j in coco_joint_parent_mapping else -1 for j in coco_joint_names]




def create_cam(cam_intrinsics, azimuths, elevations, radius):
    P = []
    for azimuth, elevation, r in zip(azimuths.ravel(), elevations.ravel(), radius.ravel()):
        eye = spherical_to_cartesian(r, azimuth, elevation)
        at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        c2w = torch.tensor(lookat_correct(eye, at, up)).float()
        w2c = torch.inverse(c2w)
        # Assuming cam_intrinsics is defined elsewhere in your context
        P_i = torch.matmul(cam_intrinsics, w2c[:3, :])
        P.append(P_i)
    P = torch.stack(P).view(*azimuths.shape, 3, 4)
    cam_dict = {'P': P}
    return cam_dict

    
def draw_mv_imgs(kp2d, joint_parents, img_w, img_h, bone_color=(0, 255, 0), kp_color=(255, 0, 0), base_img=None, mask=None, show_joints=None):
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
            if joint_parents[i] == -1 or (show_joints is not None and i not in show_joints):
                continue
            bone_color_v = ((255, 255, 0)) if mask is not None and torch.any(mask[v, i]) else bone_color
            parent = joint_parents[i]
            x1, y1 = kp2d[v, i].numpy()
            x2, y2 = kp2d[v, parent].numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(img, (x1, y1), (x2, y2), bone_color_v, 2)
        
        # Draw keypoints
        for i in range(kp2d.shape[1]):
            if show_joints is not None and i not in show_joints:
                continue
            kp_color_v = ((255, 255, 0)) if mask is not None and torch.any(mask[v, i]) else kp_color
            x, y = kp2d[v, i].numpy()
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 6, kp_color_v, -1)
        
        mv_imgs.append(img)
    mv_imgs = np.concatenate(mv_imgs, axis=1)
    return mv_imgs


def draw_motion_2d(motion_2d, fname, joint_parents, img_w, img_h, fps=30, show_joints=None, mask=None):
    frame_dir = os.path.splitext(fname)[0] + '_frames'
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir, exist_ok=True)
    # create blank image
    for t in range(motion_2d.shape[0]):
        mv_imgs = draw_mv_imgs(motion_2d[t], joint_parents, img_w, img_h, show_joints=show_joints, mask=mask[t])
        cv2.imwrite(f'{frame_dir}/{t:06d}.jpg', mv_imgs[..., ::-1])

    images_to_video(frame_dir, fname, fps=fps)
    

def recon_from_2d_and_project(samples, cam_P, img_h, img_w, num_views, mean=None, std=None, use_custom_svd='default'):
    """ unnormalization """
    samples_perm = samples.permute(0, 2, 3, 1)[:, 0] # now permuted to: [batch, 1, seq_len, nfeat]
    if mean is not None:
        samples_perm = samples_perm * std + mean
    local_kpt2d = (samples_perm + 1) * 0.5
    local_kpt2d = local_kpt2d.reshape(*local_kpt2d.shape[:2], num_views, -1, 2) * torch.tensor([img_w, img_h], device=local_kpt2d.device)
    local_kpt2d[..., 1] = img_h - local_kpt2d[..., 1]
    local_kpt2d_rs = local_kpt2d.reshape(-1, *local_kpt2d.shape[2:])
    """ triangulate """
    cam_P = cam_P.repeat_interleave(local_kpt2d.shape[1], dim=0)
    kpt3d_recon = batch_triangulate_torch(local_kpt2d_rs, cam_P, use_custom_svd=use_custom_svd)
    """ reprojection """
    kpt3d_recon_pad = torch.cat((kpt3d_recon[..., :3], torch.ones_like(kpt3d_recon[..., :1])), dim=-1)
    local_kpt2d_recon = (cam_P @ kpt3d_recon_pad[:, None].transpose(-1, -2)).transpose(-1, -2)
    local_kpt2d_recon = local_kpt2d_recon[..., :2] / local_kpt2d_recon[..., 2:]
    # sanity check: local_kpt2d_recon - local_kpt2d_rs ~= 0
    """ normalization """
    local_kpt2d_recon = local_kpt2d_recon.view_as(local_kpt2d)
    local_kpt2d_recon[..., 1] = img_h - local_kpt2d_recon[..., 1]
    samples_recon = (local_kpt2d_recon / torch.tensor([img_w, img_h]).to(local_kpt2d_recon) - 0.5) * 2
    samples_recon = samples_recon.reshape(*samples_recon.shape[:2], -1)
    if mean is not None:
        samples_recon = (samples_recon - mean) / std
    samples_recon = samples_recon.permute(0, 2, 1).unsqueeze(2)
    kpt3d_recon = kpt3d_recon.reshape(local_kpt2d.shape[:2] + kpt3d_recon.shape[1:])
    return samples_recon, kpt3d_recon[..., :3]


def project_3d_to_2d(kpt3d_recon, cam_P, img_h, img_w, num_views, mean=None, std=None):
    """ reprojection """
    kpt3d_recon_shape = kpt3d_recon.shape
    kpt3d_recon = kpt3d_recon.reshape(-1, *kpt3d_recon.shape[2:])
    kpt3d_recon_pad = torch.cat((kpt3d_recon[..., :3], torch.ones_like(kpt3d_recon[..., :1])), dim=-1)
    local_kpt2d_recon = (cam_P @ kpt3d_recon_pad[:, None].transpose(-1, -2)).transpose(-1, -2)
    local_kpt2d_recon = local_kpt2d_recon[..., :2] / local_kpt2d_recon[..., 2:]
    # sanity check: local_kpt2d_recon - local_kpt2d_rs ~= 0
    """ normalization """
    local_kpt2d_recon = local_kpt2d_recon.view(kpt3d_recon_shape[:2] + (num_views, -1, 2))
    local_kpt2d_recon[..., 1] = img_h - local_kpt2d_recon[..., 1]
    samples_recon = (local_kpt2d_recon / torch.tensor([img_w, img_h]).to(local_kpt2d_recon) - 0.5) * 2
    samples_recon = samples_recon.reshape(*samples_recon.shape[:2], -1)
    if mean is not None:
        samples_recon = (samples_recon - mean) / std
    samples_recon = samples_recon.permute(0, 2, 1).unsqueeze(2)
    return samples_recon

    