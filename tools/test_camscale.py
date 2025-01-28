import os
import random

import torch

from hmr4d.dataset.pure_motion.cam_traj_utils import CameraAugmentorV11
from hmr4d.utils.eval.eval_utils import batch_compute_scale_trans_torch
from hmr4d.utils.geo.hmr_cam import create_camera_sensor, perspective_projection
from hmr4d.utils.geo.hmr_global import get_c_rootparam, get_R_c2gv
from hmr4d.utils.smplx_utils import make_smplx
from motiondiff.models.mdm.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from motiondiff.utils.vis_scenepic import ScenepicVisualizer


def as_identity(R):
    is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
    R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
    return R


def estimate_scale_and_offset(t0, t1):
    """
    Estimate optimal scale s and offset t such that t0 ≈ s*t1 + t

    Args:
    t0: tensor of shape (B, k)
    t1: tensor of shape (B, k)

    Returns:
    s: tensor scale of shape (1,)
    t: tensor offset of shape (k,)
    """
    B, k = t0.shape
    # Construct the design matrix A
    A = torch.cat([t1, torch.ones_like(t1)], dim=1)

    solution = torch.linalg.lstsq(A, t0).solution

    s_mat = solution[:k]
    sx, sy, sz = s_mat[0, 0], s_mat[1, 1], s_mat[2, 2]
    optimal_s = torch.stack((sx, sy, sz))
    optimal_t = solution[k:].sum(0)

    return optimal_s, optimal_t, solution


def ransac_scale_and_offset(
    t0,
    t1,
    num_iterations=100,
    min_inliers_ratio=0.6,
    sample_ratio=0.1,
):
    """
    Estimate optimal scale s and offset t such that t0 ≈ s*t1 + t using RANSAC

    Args:
    t0: tensor of shape (B, k)
    t1: tensor of shape (B, k)
    num_iterations: number of RANSAC iterations
    min_inliers_ratio: minimum ratio of inliers required for a model to be considered good
    sample_ratio: number of points to sample in each iteration (minimum 100 for 3D data)

    Returns:
    s: tensor scale of shape (k,)
    t: tensor offset of shape (k,)
    """
    B, k = t0.shape

    sample_size = int(B * sample_ratio)
    sample_size = max(sample_size, 100)  # Ensure at least 100 points are sampled
    sall, t, R = batch_compute_scale_trans_torch(t0[None], t1[None])
    t0_hat = (
        sall.unsqueeze(-1).unsqueeze(-1) * R.bmm(t0[None].permute(0, 2, 1)) + t
    ).permute(0, 2, 1)

    s_list = []
    t_list = []
    error_list = []
    for _ in range(num_iterations):
        # Randomly select sample_size points
        idx = random.sample(range(B), sample_size)

        # Estimate s and t using these points
        sample_t0 = t0[idx]
        sample_t1 = t1[idx]

        s, t, R = batch_compute_scale_trans_torch(sample_t0[None], sample_t1[None])
        import ipdb

        ipdb.set_trace()
        s_all_k, t, solution = estimate_scale_and_offset(sample_t0, sample_t1)
        s = s_all_k.mean()

        # Count inliers
        errors = torch.norm(t0 - (t1 * s + t[None]), dim=1)
        s_list.append(s)
        t_list.append(t)
        error_list.append(errors.mean())

    # Check if the best model has enough inliers
    if best_inliers_count / total_points < min_inliers_ratio:
        print(
            f"Warning: Best model has only {best_inliers_count}/{total_points} inliers."
        )

    return best_s, best_t


def estimate_camscale(smpl_param_c, smpl_param_w, T_w2c, gravity_vec):
    bs = T_w2c.shape[0]
    # 1. align the camera view direction
    # R_w2c = R_c @ R_w.mT
    R_c = smpl_param_c["global_orient"]
    t_c = smpl_param_c["transl"]
    R_w = smpl_param_w["global_orient"]
    t_w = smpl_param_w["transl"]

    est_R_w2c = R_c @ R_w.mT  # estiamte from motion
    slam_R_w2c = T_w2c[:, :3, :3]
    slam_t_w2c = T_w2c[:, :3, 3]
    # Find optimal R0 using SVD (Kabsch algorithm)
    H = (slam_R_w2c.mT @ est_R_w2c).sum(dim=0)
    U, _, Vt = torch.linalg.svd(H)
    R0 = Vt.mT @ U.mT

    # Verify the result
    aligned_R_w2c = slam_R_w2c @ R0.mT
    error = torch.norm(aligned_R_w2c - est_R_w2c, dim=(1, 2))
    print(f"Alignment R_w2c error: {error.mean().item():.6f}")

    # 2. compute the cam_traj estiamted from motion
    est_t_w2c = t_c + offset - torch.einsum("fij,fj->fi", est_R_w2c, t_w + offset)

    # 3. align the slam cam_traj
    est_t_c2w = (-est_R_w2c.mT @ est_t_w2c[..., None])[..., 0]
    slam_t_c2w = (-est_R_w2c.mT @ slam_t_w2c[..., None])[..., 0]

    vel_est = est_t_c2w[1:] - est_t_c2w[:-1]
    vel_slam = slam_t_c2w[1:] - slam_t_c2w[:-1]
    scale_c2w = vel_est / vel_slam

    scale = scale_c2w.median(0).values
    scale = scale[1]
    aligned_t_c2w = slam_t_c2w * scale
    delta_t = (aligned_t_c2w - est_t_c2w).mean(0)
    aligned_t_c2w = aligned_t_c2w - delta_t

    error = torch.norm(aligned_t_c2w - est_t_c2w, dim=1)
    print(f"Alignment t_c2w error: {error.mean().item():.6f}")

    aligned_t_w2c = (-est_R_w2c @ aligned_t_c2w[..., None])[..., 0]
    error = torch.norm(aligned_t_w2c - est_t_w2c, dim=1)
    print(f"Alignment t_w2c error: {error.mean().item():.6f}")

    res_T_w2c = torch.eye(4)[None].repeat(bs, 1, 1).to(device)
    res_T_w2c[:, :3, :3] = aligned_R_w2c
    res_T_w2c[:, :3, 3] = aligned_t_w2c
    return res_T_w2c


device = "cuda:0"
sp_visualizer = ScenepicVisualizer("inputs/checkpoints/body_models/smpl", device=device)
smpl = sp_visualizer.smpl_dict["neutral"]
smplx = make_smplx("supermotion_v437coco17").to(device)

# data_pt = torch.load("inputs/EMDB/hmr4d_support/emdb_vit_v4.pt")
fns = torch.load("inputs/BEDLAM/hmr4d_support/mid_to_valid_range_all60.pt")
feat_dir = "imgfeats/bedlam_all60"
data_pt = torch.load("inputs/BEDLAM/hmr4d_support/smplpose_v2.pth")

vid_names = list(data_pt.keys())

# for i, name in enumerate(vid_names):
#     if 'run' in name:
#         print(i, name)
#         import ipdb; ipdb.set_trace()
#     if 'walk' in name:
#         print(i, name)
#         import ipdb; ipdb.set_trace()
wham_cam_augmentor = CameraAugmentorV11()
width, height, K_fullimg = create_camera_sensor(1000, 1000, 43.3)  # WHAM
K_fullimg = K_fullimg.to(device)
gravity_vec = torch.tensor([0, -1, 0], dtype=torch.float32)  # (3), BEDLAM is ay

idxes = [19280, 17986]
# while True:
#     idx = random.randint(0, len(vid_names))
for idx in idxes:
    vid = vid_names[idx]
    data = data_pt[vid]
    start, end = fns[vid]

    # bedlam
    body_pose = data["pose"][:, 3:].to(device)
    betas = data["beta"].to(device)
    global_orient = data["pose"][:, :3].to(device)
    transl = data["trans"].to(device)
    if betas.ndim == 1:
        betas = betas[None].expand(body_pose.shape[0], -1)

    offset = torch.sqrt((transl[:, [0, 2]] ** 2).sum(dim=1))
    offset = offset - offset[:1]
    print("max offset", torch.max(offset.abs()))
    if torch.max(offset.abs()) < 1:
        continue

    body_pose = body_pose[start:end]
    betas = betas[start:end]
    global_orient = global_orient[start:end]
    transl = transl[start:end]
    if body_pose.shape[1] < 69:
        bs = body_pose.shape[0]
        pad_size = 69 - body_pose.shape[1]
        body_pose = torch.cat(
            (body_pose, torch.zeros(bs, pad_size).to(body_pose)), dim=1
        )

    # # align the first frame
    # transl = transl - transl[:1]

    smpl_out = smpl(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        orig_joints=True,
    )
    joints = smpl_out.joints
    length = joints.shape[0]

    gt_T_w2c = wham_cam_augmentor(joints.cpu(), length)  # (F, 4, 4)
    gt_T_w2c = gt_T_w2c.to(joints)

    offset = smplx.get_skeleton(betas[0])[0]  # (3)
    global_orient_c, transl_c = get_c_rootparam(
        global_orient,
        transl,
        gt_T_w2c,
        offset,
    )
    _, local_joints = smplx(
        betas=betas,
        body_pose=body_pose[:, :63],
        global_orient=global_orient_c,
        transl=transl_c,
    )
    local_joints_2d = perspective_projection(local_joints, K_fullimg)  # (B, L, J, 2)

    # R_w2c = R_c2w.mT
    # t_w2c = - R_c2w.mT @ t_c2w
    # R_c2w = R_w2c.mT  (cam orient)
    # t_c2w = - R_w2c.mT @ t_w2c  (cam position)
    est_T_c2w = torch.eye(4)[None].repeat(bs, 1, 1).to(device)
    gt_T_c2w = gt_T_w2c.inverse()
    R_c2w = as_identity(gt_T_c2w[:, :3, :3])
    t_c2w = gt_T_c2w[:, :3, 3]
    # align the first frame
    R0_c2w = R_c2w[:1]
    t0_c2w = t_c2w[:1]
    est_R_c2w = R0_c2w.mT @ R_c2w
    est_t_c2w = (R0_c2w.mT @ (t_c2w - t0_c2w)[..., None])[..., 0]
    vel_scale = torch.norm(est_t_c2w[1:] - est_t_c2w[:-1], dim=1).mean()
    scale = max(vel_scale, 0.1)
    print("GT cam scale:", scale)
    est_t_c2w = est_t_c2w / scale
    est_T_c2w[:, :3, :3] = est_R_c2w
    est_T_c2w[:, :3, 3] = est_t_c2w
    est_T_w2c = est_T_c2w.inverse()
    est_T_w2c[:, :3, :3] = as_identity(est_T_w2c[:, :3, :3])
    est_T_w2c[:, 3, :3] = 0

    smpl_param_c = {
        "global_orient": axis_angle_to_matrix(global_orient_c).to(device),
        "transl": transl_c.to(device),
        "betas": betas,
        "offset": offset,
    }
    smpl_param_w = {
        "global_orient": axis_angle_to_matrix(global_orient).to(device),
        "transl": transl.to(device),
        "betas": betas,
        "offset": offset,
    }
    est_T_w2c2 = estimate_camscale(smpl_param_c, smpl_param_w, est_T_w2c, gravity_vec)
    vis = True
    if vis:
        # move y-axis to z-axis for vis
        vis_mat = torch.tensor(((1, 0, 0), (0, 0, -1), (0, 1, 0))).to(device).float()
        joints_vis = (vis_mat[None] @ joints.mT).mT
        gt_T_w2c_vis = gt_T_w2c.clone()
        gt_T_w2c_vis[:, :3, :3] = gt_T_w2c[:, :3, :3] @ vis_mat[None].mT
        res = {
            "text": "",
            "joints_pos": joints_vis.detach(),
            "T_w2c": gt_T_w2c_vis.detach(),
            "vis_all_cam": True,
        }
        est_T_w2c2_vis = est_T_w2c2.clone()
        est_T_w2c2_vis[:, :3, :3] = est_T_w2c2[:, :3, :3] @ vis_mat[None].mT
        res_est = {
            "text": "",
            "joints_pos": joints_vis.detach(),
            "T_w2c": est_T_w2c2_vis.detach(),
            "vis_all_cam": True,
        }
        res = {"gt": res, "est": res_est}
        html_path = f"tmp/{idx}_{os.path.basename(vid)}.html"
        sp_visualizer.vis_smpl_scene(res, html_path, window_size=(600, 600))
        import ipdb

        ipdb.set_trace()
