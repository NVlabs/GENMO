import sys
import torch
import numpy as np
import os
sys.path.append('./')
from motiondiff.data_pipeline.utils.skeleton import Skeleton, load_bvh_animation
from motiondiff.utils.hybrik import batch_rigid_transform
from motiondiff.data_pipeline.bones.vis import SMPLVisualizer


# bvh_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw'
# bvh_dir = 'dataset/bones_full_raw'
bvh_dir = '/mnt/nvr_torontoai_humanmotionfm/datasets/bones_full_raw'
file_names = [
    'P8/BVH/old_itching_left_forearm_R_max_003__A269_M.bvh',
    'P4/BVH/lift_crate_walk_ff_stop_225_001__A162_M.bvh',
    'P4/BVH/big_heavy_two_hands_put_down_front_high_R_001__A526_M.bvh',
]

for fname in file_names:
    print(f'Processing {fname}')
    bvh_file = f'{bvh_dir}/{fname}'
    skeleton = Skeleton()
    joints_name = skeleton.load_from_bvh(bvh_file)
    root_trans, joint_rot_mats = load_bvh_animation(bvh_file, skeleton)
    parent_indices = skeleton.get_parent_indices()
    joints = skeleton.get_neutral_joints()

    ###
    # length = np.array([ 85.,  95.,  95.,  95., 251., 130., 184., 159., 295., 233.,  70., 57.,  70., 184., 159., 295., 233.,  70.,  57.,  70.,  99., 412., 456., 171.,  99., 412., 456., 171.])
    # for i in range(1, 29):
    #     print(f"{joints_name[i]}-{joints_name[parent_indices[i]]} : {length[i-1]}")
    ###

    rot_mats = torch.tensor(joint_rot_mats)
    joints = torch.tensor(joints).unsqueeze(0).repeat(rot_mats.shape[0], 1, 1)
    parents = torch.LongTensor(parent_indices)

    joints -= joints[:, [0]]
    
    posed_joints, global_rot_mat = batch_rigid_transform(rot_mats, joints, parents)
    posed_joints += torch.tensor(root_trans).unsqueeze(1)
    posed_joints = torch.stack([posed_joints[:, :, 0], posed_joints[:, :, 2], posed_joints[:, :, 1]], dim=-1)

    vis = SMPLVisualizer(joint_parents=parents, distance=7, elevation=10)
    smpl_seq = {
        'gt':{
            'joints_pos': posed_joints.float() * 0.01
        }
    }
    video_path = f'out/bones/{os.path.basename(fname)[:-4]}.mp4'
    frame_dir = f'out/bones/frames/{os.path.basename(fname)[:-4]}'
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    vis.save_animation_as_video(video_path, init_args={'smpl_seq': smpl_seq, 'mode': 'gt'}, window_size=(1500, 1500), frame_dir=frame_dir, fps=120)