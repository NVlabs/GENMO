import sys
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import json
import torch
import argparse
import pdb
from torchmin import minimize

sys.path.append('./')
from motiondiff.data_pipeline.utils.skeleton import Skeleton, load_bvh_animation
from motiondiff.utils.hybrik import batch_rigid_transform
from motiondiff.models.common.smpl import SMPL
from motiondiff.data_pipeline.bones.vis_v2 import SMPLVisualizer
from motiondiff.data_pipeline.bones.skeleton_params import BONESPose, SMPLPose, smpl_ear_indices, smpl_eye_indices


if __name__ == '__main__':

    ''' Find a sequence with first frame close to T pose'''
    bvh_path_full = 'dataset/bones/bones_full_raw/P4/BVH/big_heavy_two_hands_put_down_front_high_R_001__A526_M.bvh'
    # bvh_path_full = 'dataset/bones/bones_full_raw/P4/BVH/idle_spelling_idle_start_001__A150.bvh'
    
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvh_path_full)
    root_trans, joint_rot_mats = load_bvh_animation(bvh_path_full, skeleton)

    parent_indices = skeleton.get_parent_indices()
    bones_parents = torch.LongTensor(parent_indices)
    joints = skeleton.get_neutral_joints()
    joints *= 0.01
    joints = torch.tensor(joints).unsqueeze(0)
    joints -= joints[:, [0]]
    
    # rot_mats = torch.zeros((joints.shape[0], joints.shape[1], 3, 3))
    # rot_mats[:, :] = torch.eye(3)
    rot_mats = torch.tensor(joint_rot_mats)[[0]].float()
    '''Set head and neck rotation to all 0'''
    rot_mats[:, [BONESPose.Head]] = torch.eye(3).float()
    rot_mats[:, [BONESPose.Neck]] = torch.eye(3).float()

    bones_joints, global_rot_mat = batch_rigid_transform(rot_mats, joints, bones_parents)

    smpl = SMPL('data/smpl_data', create_transl=False, gender='neutral').to('cuda')
    global_orient = torch.zeros((1, 3), dtype=torch.float32, device=torch.device('cuda:0'))
    body_pose = torch.zeros((1, 23*3), dtype=torch.float32, device=torch.device('cuda:0'))
    root_trans = torch.zeros((1, 3), dtype=torch.float32, device=torch.device('cuda:0'))

    def get_smpl_joints(betas):
        smpl_motion = smpl(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            root_trans=root_trans,
            return_full_pose=True,
            orig_joints=True
        )
        smpl_joints = smpl_motion.joints
        return smpl_joints

    def f(betas):
        smpl_joints = get_smpl_joints(betas)

        weights = [1, 0.1]
        loss = 0
        # First group
        bones_dist = [torch.abs(bones_joints[0, BONESPose.Neck, 1] - bones_joints[0, BONESPose.Pelvis, 1]),
                     torch.abs(bones_joints[0, BONESPose.LeftToeBase, 1] - bones_joints[0, BONESPose.Pelvis, 1]),
                     torch.abs(bones_joints[0, BONESPose.RightToeBase, 1] - bones_joints[0, BONESPose.Pelvis, 1]),
                     torch.abs(bones_joints[0, BONESPose.LeftLeg, 1] - bones_joints[0, BONESPose.Pelvis, 1]),
                     torch.abs(bones_joints[0, BONESPose.RightLeg, 1] - bones_joints[0, BONESPose.Pelvis, 1])]
        smpl_dist  = [torch.abs(smpl_joints[0, SMPLPose.Neck, 1] - smpl_joints[0, SMPLPose.Pelvis, 1]),
                     torch.abs(smpl_joints[0, SMPLPose.LToe, 1] - smpl_joints[0, SMPLPose.Pelvis, 1]),
                     torch.abs(smpl_joints[0, SMPLPose.RToe, 1] - smpl_joints[0, SMPLPose.Pelvis, 1]),
                     torch.abs(smpl_joints[0, SMPLPose.LKnee, 1] - smpl_joints[0, SMPLPose.Pelvis, 1]),
                     torch.abs(smpl_joints[0, SMPLPose.RKnee, 1] - smpl_joints[0, SMPLPose.Pelvis, 1])]
        for (d1, d2) in zip(bones_dist, smpl_dist):
            loss += torch.abs(d1 - d2) * weights[0]

        # Second group
        bones_dist = [torch.norm(bones_joints[0, BONESPose.RightForeArm] - bones_joints[0, BONESPose.RightHand]),
                     torch.norm(bones_joints[0, BONESPose.LeftForeArm] - bones_joints[0, BONESPose.LeftHand]),
                     torch.norm(bones_joints[0, BONESPose.RightLeg] - bones_joints[0, BONESPose.RightFoot]),
                     torch.norm(bones_joints[0, BONESPose.LeftLeg] - bones_joints[0, BONESPose.LeftFoot])]
        smpl_dist  = [torch.norm(smpl_joints[0, SMPLPose.RElbow] - smpl_joints[0, SMPLPose.RWrist]),
                     torch.norm(smpl_joints[0, SMPLPose.LElbow] - smpl_joints[0, SMPLPose.LWrist]),
                     torch.norm(smpl_joints[0, SMPLPose.RKnee] - smpl_joints[0, SMPLPose.RAnkle]),
                     torch.norm(smpl_joints[0, SMPLPose.LKnee] - smpl_joints[0, SMPLPose.LAnkle])]
        for (d1, d2) in zip(bones_dist, smpl_dist):
            loss += torch.abs(d1 - d2) * weights[1]

        print(loss)

        return loss

    betas = torch.zeros((1, 10), dtype=torch.float32, device=torch.device('cuda:0'))
    result = minimize(f, betas, method='bfgs')
    betas_star = result.x.detach()
    print(betas_star.cpu())

    
    smpl_motion = smpl(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas_star,
            root_trans=root_trans,
            return_full_pose=True,
            orig_joints=True
        )
    
    # ''' Add jaw joint'''
    smpl_joints = smpl_motion.joints
    jaw_joints = smpl_joints[0, SMPLPose.Head]
    jaw_joints_rest = (global_rot_mat[0, BONESPose.Head].T @ (jaw_joints.cpu() - bones_joints[0, BONESPose.Pelvis]).T).T + joints[0, BONESPose.Pelvis]
    print('Jaw joints at rest:', jaw_joints_rest)
    bones_joints = torch.concat([bones_joints, smpl_joints.cpu()[:1, SMPLPose.Head].unsqueeze(0)], dim=1)
    bones_parents = torch.concat([bones_parents, torch.LongTensor([BONESPose.Head])], dim=-1)

    # ''' Add ear joints'''
    # ear_joints = smpl_motion.vertices[0][smpl_ear_indices]
    # ear_joints_rest = (global_rot_mat[0, BONESPose.Head].T @ (ear_joints.cpu() - bones_joints[0, BONESPose.Pelvis]).T).T + joints[0, BONESPose.Pelvis]
    # print('Ear joints at rest:', ear_joints_rest)
    # bones_joints = torch.concat([bones_joints, ear_joints.cpu().unsqueeze(0)], dim=1)
    # bones_parents = torch.concat([bones_parents, torch.LongTensor([BONESPose.Head,  BONESPose.Head])], dim=-1)

    # ''' Add eye joints'''
    # eye_joints = smpl_motion.vertices[0][smpl_eye_indices]
    # eye_joints_rest = (global_rot_mat[0, BONESPose.Head].T @ (eye_joints.cpu() - bones_joints[0, BONESPose.Pelvis]).T).T + joints[0, BONESPose.Pelvis]
    # print('Eye joints at rest:', eye_joints_rest)
    # bones_joints = torch.concat([bones_joints, eye_joints.cpu().unsqueeze(0)], dim=1)
    # bones_parents = torch.concat([bones_parents, torch.LongTensor([BONESPose.Head,  BONESPose.Head])], dim=-1)

    ''' For vis '''
    bones_joints = bones_joints.repeat(120, 1, 1)
    bones_joints = torch.stack([bones_joints[:, :, 0], bones_joints[:, :, 2], bones_joints[:, :, 1]], dim=-1)
    bones_joints[:, :, 2] += 1

    smpl_joints = smpl_motion.joints
    smpl_joints = smpl_joints.repeat(120, 1, 1)
    smpl_joints = torch.stack([smpl_joints[:, :, 0], smpl_joints[:, :, 2], smpl_joints[:, :, 1]], dim=-1)
    smpl_joints[:, :, 2] += 1
    smpl_joints[:, :, 1] += 0.1

    vis = SMPLVisualizer(joint_parents=None, distance=4, elevation=2)
    smpl_seq = {
        'gt':{
            'joints_pos': bones_joints.float(),
            'parents': bones_parents,
        },
        'smpl_joints':{
            'joints_pos': smpl_joints.float(),
            'parents': smpl.parents,
        }
    }

    video_path = f'out/bones/bones_to_smpl/optimize_smpl_betas_jaw.mp4'
    frame_dir = f'out/bones/frames/optimize_smpl_betas'
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    vis.save_animation_as_video(video_path, init_args={'smpl_seq': smpl_seq, 'mode': 'gt'}, window_size=(1500, 1500), frame_dir=frame_dir, fps=120)