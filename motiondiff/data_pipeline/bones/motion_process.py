import os
import random
import sys

import numpy as np
import torch

sys.path.append("./")
from motiondiff.data_pipeline.humanml.common.quaternion import *
from motiondiff.data_pipeline.humanml.utils.paramUtil import *
from motiondiff.utils.hybrik import batch_rigid_transform
from motiondiff.utils.torch_transform import (
    get_y_heading_q,
    quat_conjugate,
    quat_mul,
    rotation_matrix_to_quaternion,
)


def foot_detect(positions, vel_thres, height_thresh):
    fid_l, fid_r = [27, 28], [23, 24]
    velfactor, heightfactor = (
        np.array([vel_thres, vel_thres]),
        np.array([height_thresh, height_thresh]),
    )

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1, fid_l, 1]
    feet_l = np.logical_and(
        (feet_l_x + feet_l_y + feet_l_z) < velfactor, feet_l_h < heightfactor
    ).astype(float)
    # feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(float)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1, fid_r, 1]
    feet_r = np.logical_and(
        (feet_r_x + feet_r_y + feet_r_z) < velfactor, feet_r_h < heightfactor
    ).astype(float)
    # feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(float)
    return feet_l, feet_r


def get_rifke(positions, heading_quat_inv, use_joint_local_height=True):
    """Local pose"""
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    if use_joint_local_height:
        positions[..., 1] -= positions[:, 0:1, 1]
    """All pose face Z+"""
    positions = qrot_np(heading_quat_inv, positions)
    return positions


def get_cont6d_params(positions, body_joint_quat, heading_quat, heading_quat_inv):
    """Quaternion to continuous 6D"""
    cont_6d_params = quaternion_to_cont6d_np(body_joint_quat)
    # (seq_len, 4)
    r_rot = heading_quat[:, 0].copy()
    #     print(r_rot[0])
    """Root Linear Velocity"""
    # (seq_len - 1, 3)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = np.concatenate([velocity, velocity[[-1]]], axis=0)
    #     print(r_rot.shape, velocity.shape)
    velocity = qrot_np(heading_quat_inv[:, 0], velocity)
    """Root Angular Velocity"""
    # (seq_len - 1, 4)
    r_angles = np.arctan2(r_rot[:, 2:3], r_rot[:, :1]) * 2
    r_velocity = r_angles[1:] - r_angles[:-1]
    r_velocity[r_velocity > np.pi] -= 2 * np.pi
    r_velocity[r_velocity < -np.pi] += 2 * np.pi
    r_velocity = np.concatenate([r_velocity, r_velocity[[-1]]], axis=0)
    # (seq_len, joints_num, 4)
    return cont_6d_params, r_velocity, velocity, r_rot


def extract_features(
    joint_rot_mats,
    posed_joints,
    use_root_local_rot=False,
    use_joint_local_height=False,
    use_joint_local_vel=False,
):
    posed_joints = posed_joints[::6]  # 120 -> 20fps
    joint_rot_mats = joint_rot_mats[::6]

    positions = posed_joints.copy()
    joint_quat = rotation_matrix_to_quaternion(torch.tensor(joint_rot_mats))
    root_quat = joint_quat[:, 0].clone()
    heading_quat = get_y_heading_q(root_quat)
    root_quat_wo_heading = quat_mul(quat_conjugate(heading_quat), root_quat).numpy()

    heading_quat = heading_quat.unsqueeze(1).repeat(1, joint_quat.shape[1], 1)

    root_quat = root_quat.numpy()
    heading_quat = heading_quat.numpy()
    heading_quat_inv = qinv_np(heading_quat)
    init_heading_quat_inv = np.repeat(
        heading_quat_inv[[0]], heading_quat_inv.shape[0], axis=0
    )

    """XZ at origin"""
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    """All initially face Z+"""
    positions = qrot_np(init_heading_quat_inv, positions)
    heading_quat = qmul_np(
        heading_quat, init_heading_quat_inv
    )  # normalize heading coordiante, so the heading is 0 for the first frame
    heading_quat_inv = qinv_np(heading_quat)

    global_positions = positions.copy()

    feet_l, feet_r = foot_detect(positions, 0.0015, 0.10)
    body_joint_quat = joint_quat[:, 1:].numpy()
    rot_joint_quat = body_joint_quat.copy()
    if use_root_local_rot:
        rot_joint_quat = np.concatenate(
            [root_quat_wo_heading[:, None], rot_joint_quat], axis=1
        )
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(
        positions, rot_joint_quat, heading_quat, heading_quat_inv
    )
    positions = get_rifke(positions, heading_quat_inv, use_joint_local_height)

    """Root height"""
    root_y = global_positions[:, 0, 1:2]

    """Root rotation and linear velocity"""
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y], axis=-1)

    """Get Joint Rotation Representation"""
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params.reshape(len(cont_6d_params), -1)

    """Get Joint Rotation Invariant Position Represention"""
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    """Get Joint Velocity Representation"""
    # (seq_len-1, joints_num*3)
    if not use_joint_local_vel:
        local_vel = qrot_np(
            heading_quat_inv[:-1], global_positions[1:] - global_positions[:-1]
        )
    else:
        local_vel = qrot_np(heading_quat_inv[:-1], positions[1:] - positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    local_vel = np.concatenate([local_vel, local_vel[[-1]].copy()], axis=0)
    feet_l = np.concatenate([feet_l, feet_l[[-1]].copy()], axis=0)
    feet_r = np.concatenate([feet_r, feet_r[[-1]].copy()], axis=0)

    data = root_data
    data = np.concatenate(
        [data, ric_data, rot_data, local_vel, feet_l, feet_r], axis=-1
    )
    return data


def recover_root_rot_pos(data, return_r_rot_ang=False):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    """Get Y-axis rotation from rotation velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang / 2)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang / 2)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    """Add Y-axis rotation to root position"""
    r_pos = qrot(r_rot_quat, r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    if return_r_rot_ang:
        return r_rot_quat, r_pos, r_rot_ang
    return r_rot_quat, r_pos


def recover_root_rot_pos_264(data, return_r_rot_ang=False):
    sin_theta = data[..., 0]
    cos_theta = data[..., 1]
    rot_vel = torch.atan2(sin_theta, cos_theta)

    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    """Get Y-axis rotation from rotation velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang / 2)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang / 2)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 2:4]
    """Add Y-axis rotation to root position"""
    r_pos = qrot(r_rot_quat, r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 4]
    if return_r_rot_ang:
        return r_rot_quat, r_pos, r_rot_ang
    return r_rot_quat, r_pos


def recover_from_ric(
    data,
    joints_num,
    r_rot_quat=None,
    r_pos=None,
    return_r_rot=False,
    use_global_joint_height=True,
):
    if r_rot_quat is None or r_pos is None:
        r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    """Add Y-axis rotation to local joints"""
    positions = qrot(
        r_rot_quat[..., None, :].expand(positions.shape[:-1] + (4,)), positions
    )

    """Add root XZ to joints"""
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    if not use_global_joint_height:
        positions[..., 1] += r_pos[..., 1:2]

    """Concate root and joints"""
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    if return_r_rot:
        return positions, r_rot_quat
    else:
        return positions


def recover_from_ric_264(
    data, joints_num, r_rot_quat=None, r_pos=None, return_r_rot=True
):
    """
    entry 0 and 1 are sin(ang_vel), cos(ang_vel) instead of usual ang_vel
    """
    if r_rot_quat is None or r_pos is None:
        r_rot_quat, r_pos = recover_root_rot_pos_264(data)
    positions = data[..., 5 : (joints_num - 1) * 3 + 5]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    """Add Y-axis rotation to local joints"""
    positions = qrot(
        r_rot_quat[..., None, :].expand(positions.shape[:-1] + (4,)), positions
    )

    """Add root XZ to joints"""
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    """Concate root and joints"""
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    if return_r_rot:
        return positions, r_rot_quat
    else:
        return positions


def recover_from_ric_with_joint_rot(
    data,
    joints,
    parents,
    root_dim=4,
    r_rot_quat=None,
    r_pos=None,
    return_r_rot=False,
    return_joint_rot=False,
):
    """
    data : [..., seq_len, nfeat]
    """
    data_shape = data.shape
    data = data.reshape(-1, *data_shape[-2:])
    joints_num = len(parents)
    if r_rot_quat is None or r_pos is None:
        if root_dim == 5:
            raise NotImplementedError(
                "ERROR (recover_from_ric_with_joint_rot) : must feed in r_pos and r_rot_quat when using this function with global reps"
            )
        r_rot_quat, r_pos = recover_root_rot_pos(data)
    else:
        r_rot_quat = r_rot_quat.reshape(-1, *r_rot_quat.shape[-2:])
        r_pos = r_pos.reshape(-1, *r_pos.shape[-2:])

    start_index = root_dim + (joints_num - 1) * 3
    end_index = start_index + joints_num * 6
    rot_6d = data[..., start_index:end_index].view(data.shape[:2] + (-1, 6))
    rotmat = cont6d_to_matrix(rot_6d)
    heading_rot_mat = quaternion_to_matrix(r_rot_quat)
    root_rot = torch.matmul(heading_rot_mat, rotmat[:, :, 0])
    final_rotmat = torch.cat(
        [root_rot.unsqueeze(2), rotmat[:, :, 1:]], dim=2
    )  # [bsize, num_frames, num_joints, 3, 3]
    final_rotmat_flat = final_rotmat.view(-1, joints_num, 3, 3)
    positions_flat = batch_rigid_transform(
        final_rotmat_flat, joints.repeat(final_rotmat_flat.shape[0], 1, 1), parents
    )[0]
    positions = positions_flat.reshape(
        final_rotmat.shape[:2] + positions_flat.shape[1:]
    )
    positions = positions + r_pos.unsqueeze(2)

    positions = positions.reshape(data_shape[:-2] + positions.shape[1:])

    output = [positions]
    if return_r_rot:
        r_rot_quat = r_rot_quat.reshape(data_shape[:-2] + r_rot_quat.shape[1:])
        output += [r_rot_quat]

    if return_joint_rot:
        joint_rot_quat = rotation_matrix_to_quaternion(final_rotmat)
        output += [joint_rot_quat]

    if len(output) > 1:
        return tuple(output)
    else:
        return output[0]


def receover_single_mano_hand(data):
    SMPL_PARAM_NUM = 102 // 2
    JOINT_PARAM_NUM = 126 // 2

    smpl_params = data[..., :SMPL_PARAM_NUM]
    joint_params = data[..., SMPL_PARAM_NUM : SMPL_PARAM_NUM + JOINT_PARAM_NUM]
    assert data.shape[-1] == SMPL_PARAM_NUM + JOINT_PARAM_NUM


def recover_from_mano_with_joint_rot(
    data,
    root_dim=4,
    r_rot_quat=None,
    r_pos=None,
    return_r_rot=False,
    return_joint_rot=False,
):
    """
    data : [..., seq_len, nfeat]
    """
    rots = data[:, :, :102]
    joints = data[:, :, 102:]
    assert joints.shape[-1] == 126
    left_rots = rots[:, :, :51]
    right_rots = rots[:, :, 51:]
    left_joints = joints[:, :, :63]
    right_joints = joints[:, :, 63:]
    joints_pos = (left_joints, right_joints)
    joints_rot = (left_rots, right_rots)
    return joints_pos, joints_rot
    data_shape = data.shape
    data = data.reshape(-1, *data_shape[-2:])
    joints_num = len(parents)
    if r_rot_quat is None or r_pos is None:
        if root_dim == 5:
            raise NotImplementedError(
                "ERROR (recover_from_mano_with_joint_rot) : must feed in r_pos and r_rot_quat when using this function with global reps"
            )
        r_rot_quat, r_pos = recover_root_rot_pos(data)
    else:
        r_rot_quat = r_rot_quat.reshape(-1, *r_rot_quat.shape[-2:])
        r_pos = r_pos.reshape(-1, *r_pos.shape[-2:])

    start_index = root_dim + (joints_num - 1) * 3
    end_index = start_index + joints_num * 6
    rot_6d = data[..., start_index:end_index].view(data.shape[:2] + (-1, 6))
    rotmat = cont6d_to_matrix(rot_6d)
    heading_rot_mat = quaternion_to_matrix(r_rot_quat)
    root_rot = torch.matmul(heading_rot_mat, rotmat[:, :, 0])
    final_rotmat = torch.cat(
        [root_rot.unsqueeze(2), rotmat[:, :, 1:]], dim=2
    )  # [bsize, num_frames, num_joints, 3, 3]
    final_rotmat_flat = final_rotmat.view(-1, joints_num, 3, 3)
    positions_flat = batch_rigid_transform(
        final_rotmat_flat, joints.repeat(final_rotmat_flat.shape[0], 1, 1), parents
    )[0]
    positions = positions_flat.reshape(
        final_rotmat.shape[:2] + positions_flat.shape[1:]
    )
    positions = positions + r_pos.unsqueeze(2)

    positions = positions.reshape(data_shape[:-2] + positions.shape[1:])

    output = [positions]
    if return_r_rot:
        r_rot_quat = r_rot_quat.reshape(data_shape[:-2] + r_rot_quat.shape[1:])
        output += [r_rot_quat]

    if return_joint_rot:
        joint_rot_quat = rotation_matrix_to_quaternion(final_rotmat)
        output += [joint_rot_quat]

    if len(output) > 1:
        return tuple(output)
    else:
        return output[0]


def recover_local_joint_pos_with_joint_rot(data, joints, parents, r_pos, root_dim=4):
    """
    Returns local joint positions based on joint rots in rep (with root height added from r_pos)
    This assumes the rep is either "orig_rot" or "global_root_local_joints_root_rot"
    data : [..., seq_len, nfeat]
    r_pos : [..., seq_len, 3]
    """
    data_shape = data.shape
    data = data.reshape(-1, *data_shape[-2:])
    joints_num = len(parents)

    if r_pos is None:
        raise NotImplementedError(
            "ERROR (recover_from_ric_with_joint_rot) : must feed in r_pos and r_rot_quat when using this function with global reps"
        )
    else:
        r_pos = r_pos.reshape(-1, *r_pos.shape[-2:])

    start_index = root_dim + (joints_num - 1) * 3
    end_index = start_index + joints_num * 6
    rot_6d = data[..., start_index:end_index].view(data.shape[:2] + (-1, 6))
    rotmat = cont6d_to_matrix(rot_6d)
    rotmat_flat = rotmat.view(-1, joints_num, 3, 3)
    # FK
    positions_flat = batch_rigid_transform(
        rotmat_flat, joints.repeat(rotmat_flat.shape[0], 1, 1), parents
    )[0]
    positions = positions_flat.reshape(rotmat.shape[:2] + positions_flat.shape[1:])

    # need to add in height because height is baked into local joint positions of rep...
    up_mask = torch.stack(
        [
            torch.zeros(r_pos.shape[:-1]),
            torch.ones(r_pos.shape[:-1]),
            torch.zeros(r_pos.shape[:-1]),
        ],
        dim=-1,
    )  # y up
    r_height = r_pos * up_mask.to(r_pos.device)  # [..., 3]
    positions = positions + r_height[..., None, :]

    return positions


if __name__ == "__main__":
    from motiondiff.data_pipeline.scripts.vis_bones import SMPLVisualizer
    from motiondiff.data_pipeline.utils.skeleton import Skeleton, load_bvh_animation

    bvh = np.load(
        "/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_joints/P1/joint_rot_mats/talking_with_child_001__A034_M.npy"
    )
    # bvh_file = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw/P1/BVH/talking_with_child_001__A034_M.bvh'
    # joint_rot_mats = np.load('/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_joints/P1/joint_rot_mats/talking_with_child_001__A034_M.npy')
    # posed_joints = np.load('/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_joints/P1/posed_joints/talking_with_child_001__A034_M.npy')
    bvh_file = "/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw/P5/BVH/sad_idle_turn_360_002__A174.bvh"
    joint_rot_mats = np.load(
        "/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_joints/P5/joint_rot_mats/sad_idle_turn_360_002__A174.npy"
    )
    posed_joints = np.load(
        "/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_joints/P5/posed_joints/sad_idle_turn_360_002__A174.npy"
    )

    skeleton = Skeleton()
    joints_name = skeleton.load_from_bvh(bvh_file)
    parent_indices = skeleton.get_parent_indices()
    joints = skeleton.get_neutral_joints().astype(np.float32) * 0.01
    joints -= joints[[0]]
    parents = torch.LongTensor(parent_indices)
    joints = torch.tensor(joints).unsqueeze(0)
    # torch.save(joints, 'assets/bones/skeleton/joints.p')
    # torch.save(parents, 'assets/bones/skeleton/parents.p')

    # posed_joints, global_rot_mat = batch_rigid_transform(rot_mats, joints, parents)

    """ extract data. data: [seq_len, F] """
    data = extract_features(joint_rot_mats, posed_joints, use_root_local_rot=True)
    """ recover data. positions: [seq_len, joints_num (29), 3] """
    positions = recover_from_ric(torch.tensor(data).float(), 29)
    positions_rot = recover_from_ric_with_joint_rot(
        torch.tensor(data).float(), joints, parents
    )
    print("Done.")

    positions = torch.stack(
        [-positions[..., 0], positions[..., 2], positions[..., 1]], dim=-1
    )
    positions_rot = torch.stack(
        [-positions_rot[..., 0], positions_rot[..., 2], positions_rot[..., 1]], dim=-1
    )

    smpl_seq = {
        "gt": {
            "joints_pos": positions,
        },
        "rot": {
            "joints_pos": positions_rot,
        },
    }

    parents = [
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        4,
        7,
        8,
        9,
        10,
        10,
        12,
        4,
        14,
        15,
        16,
        17,
        17,
        19,
        0,
        21,
        22,
        23,
        0,
        25,
        26,
        27,
    ]
    vis = SMPLVisualizer(joint_parents=parents, distance=7, elevation=10, verbose=False)
    # print('saving test videos...')
    # os.makedirs('out', exist_ok=True)
    # vis.save_animation_as_video('out/test_bones_preprocess.mp4', init_args={'smpl_seq': smpl_seq, 'mode': 'gt', 'camera': {'azimuth': -90}}, window_size=(800, 800), fps=20, crf=15)
    # print(f'test videos saved to out/test_bones_preprocess.mp4')
    vis.show_animation(
        init_args={"smpl_seq": smpl_seq, "mode": "gt", "camera": {"azimuth": 90}},
        window_size=(1500, 1500),
        fps=30,
    )
    exit()

    data_dir = "/mnt/nvr_torontoai_humanmotionfm/datasets/bones_full_joints/P1/"
    data_dir = "../bones_data/bones_joints/P11"

    sampled_paths = random.sample(
        os.listdir(os.path.join(data_dir, "posed_joints")), 10
    )
    for path in sampled_paths:
        posed_joints = np.load(os.path.join(data_dir, "posed_joints", path))
        joint_rot_mats = np.load(os.path.join(data_dir, "joint_rot_mats", path))

        """ extract data. data: [seq_len, 347] """
        data = extract_features(joint_rot_mats, posed_joints)
        """ recover data. positions: [seq_len, joints_num (29), 3] """
        positions = recover_from_ric(torch.tensor(data).float(), 29)
        print("Done.")

        positions = torch.stack(
            [-positions[..., 0], positions[..., 2], positions[..., 1]], dim=-1
        )
        foot_contacts = data[:, -4:]
        smpl_seq = {
            "gt": {
                "joints_pos": positions,
                "foot_contacts": foot_contacts,
            }
        }

        parents = [
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            4,
            7,
            8,
            9,
            10,
            10,
            12,
            4,
            14,
            15,
            16,
            17,
            17,
            19,
            0,
            21,
            22,
            23,
            0,
            25,
            26,
            27,
        ]
        vis = SMPLVisualizer(
            joint_parents=parents, distance=7, elevation=10, verbose=False
        )
        print("saving test videos...")
        os.makedirs("out/bones_feature_conversion", exist_ok=True)
        out_path = f"out/bones_feature_conversion/{path[:-3]}.mp4"
        vis.save_animation_as_video(
            out_path,
            init_args={"smpl_seq": smpl_seq, "mode": "gt", "camera": {"azimuth": -90}},
            window_size=(800, 800),
            fps=20,
            crf=15,
        )
        print(f"test videos saved to {out_path}")
        # vis.show_animation(init_args={'smpl_seq': smpl_seq, 'mode': 'gt', 'camera': {'azimuth': -90}}, window_size=(1500, 1500), fps=30)
