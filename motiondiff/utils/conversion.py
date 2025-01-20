import torch
from motiondiff.data_pipeline.humanml.scripts.motion_process import recover_from_ric, extract_features
from motiondiff.data_pipeline.humanml.common.quaternion import cont6d_to_matrix
from motiondiff.data_pipeline.humanml.utils.paramUtil import *
from scipy.interpolate import interp1d
from motiondiff.utils.hybrik import batch_inverse_kinematics_transform, batch_rigid_transform
from motiondiff.utils.torch_transform import quat_conjugate, quaternion_to_angle_axis, angle_axis_to_quaternion, angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis, quat_mul, quat_apply, rot6d_to_angle_axis
from motiondiff.models.common.smpl import SMPL


smpl_rest_joints = {}

def joints_to_smpl(smpl_joint_pos, smpl, device):
    ''' 
    runs IK on joint positions to recover pose params
    - smpl_joint_pos : must have hand joints [B, nframes, 24, 3]
    '''
    global smpl_rest_joints
    base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device)

    shape_in = smpl_joint_pos.shape

    smpl_body_pos = smpl_joint_pos.clone()
    smpl_body_pos = smpl_body_pos.reshape(-1, *smpl_body_pos.shape[2:])

    # inverse kinematics
    if smpl.gender not in smpl_rest_joints:
        smpl_rest_joints[smpl.gender] = smpl.get_joints(
            global_orient=torch.zeros((1, 3)).to(device),
            body_pose=torch.zeros((1, 69)).to(device),
            betas=torch.zeros((1, 10)).to(device),
            root_trans=torch.zeros((1, 3)).to(device)
        )
    rest_body_pos = smpl_rest_joints[smpl.gender].repeat_interleave(smpl_body_pos.shape[0], dim=0).to(device)
    phis = torch.tensor([1.0, 0.0], device=device).expand(smpl_body_pos.shape[0], 23, -1)
    leaf_thetas = torch.eye(3, device=device).expand(smpl_body_pos.shape[0], 5, -1, -1)
    root_diff = rest_body_pos[:, [0]] - smpl_body_pos[:, [0]]
    smpl_body_pos += root_diff
    smpl_rot_mats, global_rot_mat, global_body_pos = batch_inverse_kinematics_transform(smpl_body_pos, None, phis, rest_body_pos, smpl.children_map, smpl.parents, leaf_thetas, False)
    smpl_rot_mats[:, -4:] = torch.eye(3, device=device)

    smpl_trans = smpl_joint_pos[:, :, 0].clone()
    smpl_pose = rotation_matrix_to_angle_axis(smpl_rot_mats.contiguous()).reshape(*shape_in[:2], 72)

    smpl_trans = quat_apply(base_rot.expand(smpl_trans.shape[:-1] + (4,)), smpl_trans)
    smpl_pose[..., :3] = quaternion_to_angle_axis(quat_mul(base_rot.expand(smpl_pose.shape[:-1] + (4,)), angle_axis_to_quaternion(smpl_pose[..., :3])))

    return smpl_pose, smpl_trans


def humanml_to_smpl(sample_raw, mean, std, smpl, debug=False, time=False, render=False, interp=True, source_fps=20, target_fps=30, get_orig_rot=False):
    global smpl_rest_joints

    device = sample_raw.device
    base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device)
    # sample_raw: [batch, nfeat, 1, seq_len]
    sample_raw_norm = sample_raw.permute(0, 2, 3, 1) * std + mean # sample_raw_norm: [batch, 1, seq_len, nfeat]
    sample_recon, r_rot_quat = recover_from_ric(sample_raw_norm, 22, return_r_rot=True)
    sample_recon, r_rot_quat = sample_recon[:, 0], r_rot_quat[:, 0]
    r_rot_aa = quaternion_to_angle_axis(r_rot_quat)

    if interp:
        orig_len = sample_recon.shape[1]
        new_len = int(orig_len / source_fps * target_fps)
        T = orig_len / source_fps
        f = interp1d(np.linspace(0, T, orig_len), sample_recon.cpu().numpy(), axis=1, assume_sorted=True, fill_value="extrapolate")
        sample_recon = torch.from_numpy(f(np.linspace(0, T, new_len))).type_as(sample_recon)
        if get_orig_rot:
            f = interp1d(np.linspace(0, T, orig_len), r_rot_aa.cpu().numpy(), axis=1, assume_sorted=True, fill_value="extrapolate")
            r_rot_aa = torch.from_numpy(f(np.linspace(0, T, new_len))).type_as(r_rot_aa)
            f = interp1d(np.linspace(0, T, orig_len), sample_raw_norm.cpu().numpy(), axis=2, assume_sorted=True, fill_value="extrapolate")
            sample_raw_norm = torch.from_numpy(f(np.linspace(0, T, new_len))).type_as(sample_raw_norm)

    sample = sample_recon
    
    if smpl is None:
        hand_joints = sample[:, :, -2:].clone()
        hand_joints[..., 1] += 1e-2
        smpl_body_pos = torch.cat([sample, hand_joints], dim=2)
        joints_3d = quat_apply(base_rot.expand(smpl_body_pos.shape[:-1] + (4,)), smpl_body_pos)
        return None, None, joints_3d

    if get_orig_rot:
        smpl_trans = sample[:, :, 0].clone()
        rot6d = sample_raw_norm[:, 0, :, 67:193].reshape(sample.shape[0], -1, 21, 6)
        rotmat = cont6d_to_matrix(rot6d)
        smpl_pose = rotation_matrix_to_angle_axis(rotmat)
        smpl_pose = torch.cat([r_rot_aa.unsqueeze(2), smpl_pose, torch.zeros_like(smpl_pose[:, :, :2])], dim=2).reshape(sample.shape[0], -1, 72)
        joints_3d = None

    else:
        hand_joints = sample[:, :, -2:].clone()
        hand_joints[..., 1] += 1e-2
        smpl_body_pos = torch.cat([sample, hand_joints], dim=2)
        smpl_body_pos_orig = smpl_body_pos.clone()
        smpl_body_pos = smpl_body_pos.view(-1, *smpl_body_pos.shape[2:])

        # inverse kinematics
        if smpl.gender not in smpl_rest_joints:
            smpl_rest_joints[smpl.gender] = smpl.get_joints(
                global_orient=torch.zeros((1, 3)).to(sample_raw.device),
                body_pose=torch.zeros((1, 69)).to(sample_raw.device),
                betas=torch.zeros((1, 10)).to(sample_raw.device),
                root_trans=torch.zeros((1, 3)).to(sample_raw.device)
            )
        rest_body_pos = smpl_rest_joints[smpl.gender].repeat_interleave(smpl_body_pos.shape[0], dim=0).to(device)
        phis = torch.tensor([1.0, 0.0], device=device).expand(smpl_body_pos.shape[0], 23, -1)
        leaf_thetas = torch.eye(3, device=device).expand(smpl_body_pos.shape[0], 5, -1, -1)
        root_diff = rest_body_pos[:, [0]] - smpl_body_pos[:, [0]]
        smpl_body_pos += root_diff
        smpl_rot_mats, global_rot_mat, global_body_pos = batch_inverse_kinematics_transform(smpl_body_pos, None, phis, rest_body_pos, smpl.children_map, smpl.parents, leaf_thetas, False)
        smpl_rot_mats[:, -4:] = torch.eye(3, device=device)

        smpl_trans = smpl_body_pos_orig[:, :, 0].clone()
        smpl_pose = rotation_matrix_to_angle_axis(smpl_rot_mats.contiguous()).reshape(*sample.shape[:2], 72)
        joints_3d = quat_apply(base_rot.expand(smpl_body_pos_orig.shape[:-1] + (4,)), smpl_body_pos_orig)
    
    smpl_trans = quat_apply(base_rot.expand(smpl_trans.shape[:-1] + (4,)), smpl_trans)
    smpl_pose[..., :3] = quaternion_to_angle_axis(quat_mul(base_rot.expand(smpl_pose.shape[:-1] + (4,)), angle_axis_to_quaternion(smpl_pose[..., :3])))
    
    return smpl_pose, smpl_trans, joints_3d
