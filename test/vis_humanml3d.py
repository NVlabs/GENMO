import torch
import torch
from motiondiff.utils.vis_scenepic import ScenepicVisualizer
import os
from motiondiff.models.mdm.rotation_conversions import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
)

# motion_files = torch.load("inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose.pth")
motion_files = torch.load("humanml3d_smplhpose.pth")

device = torch.device('cuda')
sp_visualizer = ScenepicVisualizer("/home/jiefengl/git/physdiff_megm/data/smpl_data", device=device)
smpl_dict = sp_visualizer.smpl_dict


right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]  # , 23, 22]
SMPL_JOINTS_FLIP_PERM_BODY = [i - 1 for i in SMPL_JOINTS_FLIP_PERM][1:]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3 * i)
    SMPL_POSE_FLIP_PERM.append(3 * i + 1)
    SMPL_POSE_FLIP_PERM.append(3 * i + 2)


for idx, vid in enumerate(motion_files):
    if not vid.startswith('M'):
        continue
    motion_data = motion_files[vid]
    smpl_layer = smpl_dict['neutral']
    
    pose = motion_data["pose"].reshape(-1, 22, 3)
    global_orient = pose[:, :1, :]
    body_pose = pose[:, 1:, :]
    beta = motion_data['beta'][None].repeat(pose.shape[0], 1)
    trans = motion_data['trans']

    pose_pad = torch.cat([body_pose, torch.zeros_like(body_pose[:, :2])], dim=1)
    smpl_output = smpl_layer(
        betas=beta.to(device),
        global_orient=global_orient.to(device),
        body_pose=pose_pad.to(device),
        transl=trans.to(device),
        orig_joints=True,
        pose2rot=True,
    )
    j3d = smpl_output.joints.detach()

    smpl_seq = {
        "text": '',
        "joints_pos": j3d,
        # "joints_pos": torch.from_numpy(data_jts[:, :24]).float(),
    }

    pose_flip = motion_data["pose"].reshape(-1, 22, 3).clone()
    global_orient_flip = pose_flip[:, :1, :]
    body_pose_flip = pose_flip[:, 1:, :]
    global_orient_flip[:, :, 1::3] = -global_orient_flip[:, :, 1::3].clone()
    global_orient_flip[:, :, 2::3] = -global_orient_flip[:, :, 2::3].clone()
    body_pose_flip[:, :, 1::3] = -body_pose_flip[:, :, 1::3].clone()
    body_pose_flip[:, :, 2::3] = -body_pose_flip[:, :, 2::3].clone()
    # global_orient_mat = axis_angle_to_matrix(global_orient_flip)
    # body_pose_mat = axis_angle_to_matrix(body_pose_flip)
    # global_orient_mat[:, :, 1:] = -global_orient_mat[:, :, 1:]
    # body_pose_mat[:, :, 1:] = -body_pose_mat[:, :, 1:]
    # global_orient_flip = matrix_to_axis_angle(global_orient_mat)
    # body_pose_flip = matrix_to_axis_angle(body_pose_mat)

    trans_flip = motion_data["trans"].clone()
    trans_flip[:, 0] = -trans_flip[:, 0]

    body_pose_flip[:, :] = body_pose_flip[:, SMPL_JOINTS_FLIP_PERM_BODY].clone()
    pose_pad_flip = torch.cat([body_pose_flip, torch.zeros_like(body_pose_flip[:, :2])], dim=1)
    smpl_output_flip = smpl_layer(
        betas=beta.to(device),
        global_orient=global_orient_flip.to(device),
        body_pose=pose_pad_flip.to(device),
        transl=trans_flip.to(device),
        orig_joints=True,
        pose2rot=True,
    )
    j3d_flip = smpl_output_flip.joints.detach()

    smpl_seq_flip = {
        "text": "",
        "joints_pos": j3d_flip,
        # "joints_pos": torch.from_numpy(data_jts[:, :24]).float(),
    }

    smpl_res = {
        "smpl_seq": smpl_seq,
        "smpl_seq_flip": smpl_seq_flip,
    }
    html_file = f"out/vis_humanml3d/{vid}.html"
    os.makedirs(os.path.dirname(html_file), exist_ok=True)
    sp_visualizer.vis_smpl_scene(smpl_res, html_file)
    print(vid)
    # if idx > 100:
    #     break