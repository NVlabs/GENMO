import sys
import torch
import numpy as np
import os
sys.path.append('./')
from motiondiff.data_pipeline.utils.skeleton import Skeleton, load_bvh_animation
from motiondiff.utils.hybrik import batch_rigid_transform
from motiondiff.data_pipeline.scripts.vis_bones import SMPLVisualizer


bvh_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw'
file_names = [
    'P8/BVH/big_light_one_hand_behind_high_to_right_side_high_R_001__A527_M.bvh',
    'P8/BVH/crouch_ff_stop_315_R_003__A245_M.bvh',
    'P8/BVH/jog_ff_start_180_R_002__A247_M.bvh',
    'P3/BVH/sending_kisses_R_003__A097.bvh',
    'P3/BVH/thinking_R_003__A101.bvh',
    'P7/BVH/jump_ff_180_R_002__A232.bvh',
    'P7/BVH/sensual_jump_ff_180_R_001__A229.bvh',
    'P7/BVH/threaten_fist_R_001__A235_M.bvh',
    'P11/BVH/spear_hit_head_R_001__A386_M.bvh',
    'P11/BVH/spear_turn_walk_pass_start_360_R_003__A386_M.bvh'
]

for fname in file_names:
    print(f'Processing {fname}')
    bvh_file = f'{bvh_dir}/{fname}'
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvh_file)
    root_trans, joint_rot_mats = load_bvh_animation(bvh_file, skeleton)
    parent_indices = skeleton.get_parent_indices()
    joints = skeleton.get_neutral_joints()

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