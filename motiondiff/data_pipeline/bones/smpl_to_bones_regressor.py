import sys
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import torch
import argparse
import pdb
import random
import time

sys.path.append('./')
from motiondiff.models.common.smpl import SMPL
from motiondiff.models.mdm.rotation_conversions import axis_angle_to_matrix
from motiondiff.utils.hybrik import batch_inverse_kinematics_transform_bones, batch_rigid_transform


'''Skeleton parameters'''
from motiondiff.data_pipeline.bones.skeleton_params import BONESPose, bones_beta, bones_parents, bones_children_map, bones_joints_rest_eye
bones_parents = torch.LongTensor(bones_parents)
bones_children_map = torch.LongTensor(bones_children_map)
bones_beta = torch.FloatTensor([bones_beta]).to('cuda')
bones_joints_rest_eye = torch.FloatTensor(bones_joints_rest_eye)


'''Optimization and help functions'''
smpl = SMPL('data/smpl_data', create_transl=False, gender='neutral').to('cuda')

def vis(visualizer, smpl_motion, bones_joints, bones_joints_retarget, bones_joints_retarget_ik, video_path, fps=120):
    base_rot = torch.FloatTensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).to('cuda')
    smpl_joints = smpl_motion.joints @ base_rot
    smpl_verts = smpl_motion.vertices @ base_rot
    bones_joints = bones_joints @ base_rot
    bones_joints_retarget = bones_joints_retarget @ base_rot
    bones_joints_retarget_ik = bones_joints_retarget_ik @ base_rot

    smpl_verts_v2 = smpl_verts.clone()
    smpl_verts[:, :, 0] += 0.8 # 1
    bones_joints_retarget[:, :, 0] += 0.8 # 1
    # bones_joints_retarget[:, :, 0] += 0 # 2
    bones_joints[:, :, 0] -= 0.8 # 3
    smpl_joints[:, :, 0] -= 1.6 # 4
    smpl_verts_v2[:, :, 0] -= 1.6 # 4

    smpl_seq = {
        'bones_joints_retarget':{
            'joints_pos': bones_joints_retarget.float(),
            'parents': bones_parents,
        },
        'smpl_mesh':{
            'smpl_verts': smpl_verts.float(),
            'show_smpl_mesh': True,
            'opacity': 0.8,
        },
        'bones_joints_retarget_ik':{
            'joints_pos': bones_joints_retarget_ik.float(),
            'parents': bones_parents,
        },
        'gt':{
            'joints_pos': bones_joints.float(),
            'parents': bones_parents,
        },
        'smpl_joints':{
            'joints_pos': smpl_joints.float(),
            'parents': smpl.parents,
        },
        'smpl_mesh_v2':{
            'smpl_verts': smpl_verts_v2.float(),
            'show_smpl_mesh': True,
            'opacity': 0.8,
        }
    }
    frame_dir = f'out/bones/frames/bones_to_smpl+{random.randint(0, 1e9)}'
    visualizer.save_animation_as_video(video_path, init_args={'smpl_seq': smpl_seq, 'mode': 'gt'}, window_size=(1500, 1500), frame_dir=frame_dir, fps=fps)


def solve_regressor(verts, joints, L=200):
    from scipy.optimize import minimize
    verts = verts.astype(np.float32)
    joints = joints.astype(np.float32)

    J = joints.shape[1]
    V = verts.shape[1]
    regressor = np.zeros((J, V), dtype=np.float32)

    for j in range(1, njoints):
        print(f"Solving regressor for {j}th joint ...")
        A = verts # n x V
        b = joints[:, [j]] # n
        dist = np.linalg.norm(A - b, axis=0) # V
        inds = np.argsort(dist)[:L]

        A_sub = verts[:, inds] # n x L

        def objective(x, A, b):
            return np.linalg.norm(A @ x - b[:, 0])

        # Define the constraints
        def constraint_sum_to_one(x):
            return np.sum(x) - 1

        # Define the bounds (x >= 0)
        bounds = [(0, None) for _ in range(L)]

        # Initial guess for x
        x0 = np.ones(L, dtype=np.float32) / L

        # Define the constraints dictionary
        constraints = {'type': 'eq', 'fun': constraint_sum_to_one}

        # Perform the optimization
        result = minimize(objective, x0, args=(A_sub, b), bounds=bounds, constraints=constraints)

        # The solution is in result.x
        x = result.x
        print(x)
        print("Sum:", x.sum())

        regressor[j, inds] = x

    return regressor.T


parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, default='metadata_240527_v014.csv')
parser.add_argument('--data_dir', type=str, default='bones_full_raw_v14')
parser.add_argument('--bones_joints_dir', type=str, default='bones_full_joints_v14')
parser.add_argument('--out_dir', type=str, default='bones_to_smpl/bones_to_smpl_v14.5')
parser.add_argument('--downsample_rate', type=int, default=4)
parser.add_argument('--add_joints', type=str, default='eye')
parser.add_argument('--retarget_loss_thresh', type=float, default=0.4)
parser.add_argument('--regress_ver', type=str, default='split_xyz')
parser.add_argument('--test_split_ratio', type=float, default=0.1)
parser.add_argument('--recompute', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--debug_vis', action='store_true', default=False)
parser.add_argument('--max_train_seqs', type=int, default=1000)
args = parser.parse_args()


if __name__ == '__main__':

    csv = pd.read_csv(os.path.join('dataset/bones', args.data_dir, args.csv_file))
    args.bones_joints_dir = os.path.join('/mnt/nvr_torontoai_humanmotionfm/datasets', args.bones_joints_dir)
    # args.bones_to_smpl_dir = os.path.join('/mnt/nvr_torontoai_humanmotionfm/datasets', args.out_dir)
    args.bones_to_smpl_dir = os.path.join('dataset/bones', args.out_dir)
    args.out_dir = os.path.join('dataset/bones', args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    bones_rest_joints = torch.from_numpy(np.load(os.path.join(args.bones_joints_dir, 'neutral_pose.npy')))
    if args.add_joints is not None:
        bones_rest_joints = torch.cat([bones_rest_joints, bones_joints_rest_eye.unsqueeze(0).cpu()], dim=1)
        bones_parents = torch.cat([bones_parents, torch.LongTensor([BONESPose.Head,  BONESPose.Head])], dim=-1)
        bones_children_map = torch.cat([bones_children_map, torch.LongTensor([-1,  -1])], dim=-1)
        bones_children_map[BONESPose.Head] = BONESPose.LEye
    njoints = bones_parents.shape[0]

    # bvh_path_all = [path for path in csv.move_bvh_path if os.path.exists(os.path.join(args.bones_to_smpl_dir, path.replace('BVH', 'smpl').replace('bvh', 'npz')))]
    # print(f'Collected {len(bvh_path_all)} bones-smpl data pairs')
    smpl_path_all = []
    for subdir in tqdm(os.listdir(os.path.join(args.bones_to_smpl_dir, 'smpl'))):
        for path in os.listdir(os.path.join(args.bones_to_smpl_dir, 'smpl', subdir)):
            smpl_path = os.path.join('smpl', subdir, path)
            smpl_path_all.append(smpl_path)
    print(f'Collected {len(smpl_path_all)} bones-smpl data pairs')
    smpl_path_all = sorted(smpl_path_all)

    random.seed(0)
    num_train = int(len(smpl_path_all) * (1 - args.test_split_ratio))
    smpl_path_train = smpl_path_all[:num_train]
    random.shuffle(smpl_path_train)
    smpl_path_train = smpl_path_train[:args.max_train_seqs]
    smpl_path_test = smpl_path_all[min(num_train, args.max_train_seqs):]
    random.shuffle(smpl_path_test)
    
    if args.vis:
        from motiondiff.data_pipeline.bones.vis_v2 import SMPLVisualizer
        visualizer = SMPLVisualizer(joint_parents=None, distance=7, elevation=2, smpl=smpl, verbose=False)
    
    regressor_path = os.path.join(args.out_dir, f'smpl_to_bones_regressor_{args.regress_ver}.npy')
    if not os.path.exists(regressor_path) or args.recompute:
        smpl_verts_train = []
        bones_joints_train = []
        for idx, smpl_path in enumerate(tqdm(smpl_path_train)):
            # if idx % int(1 /args.test_split_ratio) == 0: continue

            smpl_res = np.load(os.path.join(args.bones_to_smpl_dir, smpl_path)) 
            smpl_thetas = smpl_res['thetas']
            smpl_losses = smpl_res['losses']
            valid_frames = smpl_losses < args.retarget_loss_thresh
            if valid_frames.sum() < 30: continue
            
            ''' Compute SMPL mesh '''
            smpl_thetas = smpl_thetas[valid_frames]
            smpl_thetas = torch.from_numpy(smpl_thetas).to('cuda')
            smpl_motion_no_root = smpl(
                global_orient=smpl_thetas[:, :3],
                body_pose=smpl_thetas[:, 3:],
                betas=bones_beta,
                root_trans=torch.zeros_like(smpl_thetas[:, :3]),
                return_full_pose=True,
                orig_joints=True
            )

            ''' Load pre-saved Bones joints '''
            # joint_path = os.path.join(args.bones_joints_dir, smpl_path.replace('smpl', 'posed_joints').replace('npz', 'npy'))
            # bones_joints = np.load(joint_path)[::args.downsample_rate][:smpl_res['thetas'].shape[0]][valid_frames]
            # bones_joints_ori = np.stack([bones_joints[..., 0], bones_joints[..., 2], bones_joints[..., 1]], axis=-1)
            # bones_joints_noroot = bones_joints_ori - bones_joints_ori[:, [0]]

            ''' Compute Bones joints via FK'''
            rotmat_path = os.path.join(args.bones_joints_dir, smpl_path.replace('smpl', 'joint_rot_mats').replace('npz', 'npy'))
            bones_rotmats = torch.from_numpy(np.load(rotmat_path)[::args.downsample_rate])[:smpl_res['thetas'].shape[0]][valid_frames]
            if args.add_joints is not None:
                bones_rotmats = torch.cat([bones_rotmats, bones_rotmats[:, -2:]], dim=1) # any rotations for eyes
            bones_rest_joints_seq = bones_rest_joints.repeat_interleave(bones_rotmats.shape[0], dim=0)
            bones_joints_noroot, _ = batch_rigid_transform(bones_rotmats, bones_rest_joints_seq, bones_parents)
            bones_joints_noroot = bones_joints_noroot.cpu().numpy()

            if args.regress_ver in ['concat_xyz', 'split_xyz']:
                smpl_verts_train += [smpl_motion_no_root.vertices.cpu().numpy()]
                bones_joints_train += [bones_joints_noroot]
            elif args.regress_ver == 'wo_glb_orient_split_xyz':
                rot_mats = axis_angle_to_matrix(smpl_thetas[:, :3]).transpose(1, 2)
                smpl_verts = torch.matmul(smpl_motion_no_root.vertices, rot_mats)
                bones_joints = torch.matmul(torch.from_numpy(bones_joints_noroot).to('cuda'), rot_mats)
                smpl_verts_train += [smpl_verts.cpu().numpy()]
                bones_joints_train += [bones_joints.cpu().numpy()]

            if args.debug_vis and idx > 10: break

        smpl_verts_train = np.vstack(smpl_verts_train)
        bones_joints_train = np.vstack(bones_joints_train)
        assert smpl_verts_train.shape[0] == bones_joints_train.shape[0]

        ''' Compute the least squares solution '''
        print(f"Computing regressor matrix given {smpl_verts_train.shape[0]} paired frames ...")
        start_time = time.time()
        if args.regress_ver == 'concat_xyz':
            ''' 6890*3 x 29*3'''
            smpl_verts_train = smpl_verts_train.reshape(smpl_verts_train.shape[0], -1)
            bones_joints_train = bones_joints_train.reshape(bones_joints_train.shape[0], -1)
        elif args.regress_ver == 'split_xyz' or args.regress_ver == 'wo_glb_orient_split_xyz':
            ''' 6890 x 29'''
            smpl_verts_train = np.transpose(smpl_verts_train, [0, 2, 1]).reshape(-1, smpl_verts_train.shape[1])
            bones_joints_train = np.transpose(bones_joints_train, [0, 2, 1]).reshape(-1, bones_joints_train.shape[1])

        # regressor, residuals, rank, s = np.linalg.lstsq(smpl_verts_train, bones_joints_train, rcond=None)
        regressor = solve_regressor(smpl_verts_train, bones_joints_train)
        print(f"Regressor finished in {time.time() - start_time}s")
        np.save(regressor_path, regressor)
    
    else:
        regressor = np.load(regressor_path)

    ''' Vis / Metric'''
    errors_regressor = []
    errors_ik = []
    errors_regressor_ik = []
    for idx, smpl_path in enumerate(tqdm(smpl_path_test)):
        # if idx % int(1 /args.test_split_ratio) != 0: continue

        print(f"Vis {idx}th seq: {smpl_path}")
        video_path = os.path.join(args.out_dir, f'video_regress_test_{args.regress_ver}', os.path.basename(smpl_path).replace('npz', 'mp4'))
        rotmat_path = os.path.join(args.bones_joints_dir, smpl_path.replace('smpl', 'joint_rot_mats').replace('npz', 'npy'))
        smpl_path = os.path.join(args.bones_to_smpl_dir, smpl_path)

        smpl_params = np.load(smpl_path)
        smpl_thetas = torch.from_numpy(smpl_params['thetas']).to('cuda')
        smpl_trans = torch.from_numpy(smpl_params['root_trans']).to('cuda')

        bones_rotmats = torch.from_numpy(np.load(rotmat_path)[::args.downsample_rate]).to('cuda')
        bones_rotmats = bones_rotmats[:smpl_thetas.shape[0]]
        if args.add_joints is not None:
            bones_rotmats = torch.cat([bones_rotmats, bones_rotmats[:, -2:]], dim=1) # any rotations for eyes
        bones_rest_joints_seq = bones_rest_joints.repeat_interleave(bones_rotmats.shape[0], dim=0).to('cuda')
        bones_joints_noroot_gt, _ = batch_rigid_transform(bones_rotmats, bones_rest_joints_seq, bones_parents)

        smpl_motion_noroot = smpl(
            global_orient=smpl_thetas[:, :3],
            body_pose=smpl_thetas[:, 3:],
            betas=bones_beta,
            root_trans=torch.zeros_like(smpl_thetas[:, :3]),
            return_full_pose=True,
            orig_joints=True
        )
        smpl_verts = smpl_motion_noroot.vertices

        ''' run regressor '''
        if args.regress_ver == 'concat_xyz':
            smpl_verts = smpl_verts.reshape(smpl_verts.shape[0], -1).cpu().numpy()
            bones_joints_retarget = smpl_verts.dot(regressor)
            bones_joints_retarget = torch.from_numpy(bones_joints_retarget.reshape(-1, njoints, 3)).to('cuda')
        elif args.regress_ver == 'split_xyz':
            smpl_verts = smpl_verts.transpose(1, 2).reshape(-1, smpl_verts.shape[1]).cpu().numpy()
            bones_joints_retarget = smpl_verts.dot(regressor)
            bones_joints_retarget = torch.from_numpy(bones_joints_retarget.reshape(-1, 3, njoints)).transpose(1, 2).to('cuda')
        elif args.regress_ver == 'wo_glb_orient_split_xyz':
            rot_mats = axis_angle_to_matrix(smpl_thetas[:, :3]).transpose(1, 2)
            smpl_verts = torch.matmul(smpl_verts, rot_mats)
            smpl_verts = smpl_verts.transpose(1, 2).reshape(-1, smpl_verts.shape[1]).cpu().numpy()
            bones_joints_retarget = smpl_verts.dot(regressor)
            bones_joints_retarget = torch.from_numpy(bones_joints_retarget.reshape(-1, 3, njoints)).transpose(1, 2).to('cuda')
            bones_joints_retarget = torch.matmul(bones_joints_retarget, rot_mats.transpose(1, 2))
        
        ''' IK '''
        phis = torch.tensor([1.0, 0.0], device='cuda').expand(bones_rotmats.shape[0], bones_rotmats.shape[1], -1)
        if args.add_joints is not None:
            leaf_thetas = torch.eye(3, device='cuda').expand(bones_rotmats.shape[0], 8, -1, -1)
        else:
            leaf_thetas = torch.eye(3, device='cuda').expand(bones_rotmats.shape[0], 7, -1, -1)
        bones_rot_mats_ik, _, _ = batch_inverse_kinematics_transform_bones(bones_joints_retarget, None, phis, bones_rest_joints_seq, bones_children_map, bones_parents, leaf_thetas, False, add_eye=args.add_joints is not None)
        bones_rot_mats_ik[:, [BONESPose.LeftHand, BONESPose.LeftHandThumb1, BONESPose.RightHand, BONESPose.RightHandThumb1]] = torch.eye(3, device='cuda')
        bones_joints_retarget_ik, _ = batch_rigid_transform(bones_rot_mats_ik.to('cuda'), bones_rest_joints_seq, bones_parents)

        ''' Compute metric given joints (no hand joints) '''
        selected_joints = np.array([i for i in range(bones_rotmats.shape[1]) if i not in [BONESPose.LeftHandEnd, BONESPose.LeftHandThumb1, BONESPose.LeftHandThumb2, BONESPose.RightHandEnd, BONESPose.RightHandThumb1, BONESPose.RightHandThumb2]])
        err_regressor = np.linalg.norm((bones_joints_noroot_gt - bones_joints_retarget).cpu().numpy()[:, selected_joints], axis=2).mean()
        print(f"Retarget error regressor = {err_regressor}")
        err_ik = np.linalg.norm((bones_joints_retarget - bones_joints_retarget_ik).cpu().numpy()[:, selected_joints], axis=2).mean()
        print(f"Retarget error IK = {err_ik}")
        err_regressor_ik = np.linalg.norm((bones_joints_noroot_gt - bones_joints_retarget_ik).cpu().numpy()[:, selected_joints], axis=2).mean()
        print(f"Retarget error regressor and IK = {err_regressor_ik}")
        errors_regressor += [err_regressor]
        errors_ik += [err_ik]
        errors_regressor_ik += [err_regressor_ik]

        ''' Vis '''
        smpl_motion = smpl(
            global_orient=smpl_thetas[:, :3],
            body_pose=smpl_thetas[:, 3:],
            betas=bones_beta,
            root_trans=smpl_trans,
            return_full_pose=True,
            orig_joints=True
        )
        print(f"Rendering video ...")
        bones_joints_retarget += smpl_trans.unsqueeze(1)
        bones_joints_retarget_ik += smpl_trans.unsqueeze(1)
        bones_joints_gt = bones_joints_noroot_gt + smpl_trans.unsqueeze(1)
        vis(visualizer, smpl_motion, bones_joints_gt, bones_joints_retarget, bones_joints_retarget_ik, video_path, fps=120 / args.downsample_rate)
    
    print(f"{args.regress_ver} produces average errors regressor = {np.average(errors_regressor)}")
    print(f"{args.regress_ver} produces average errors IK = {np.average(errors_ik)}")
    print(f"{args.regress_ver} produces average errors regressor + IK = {np.average(errors_regressor_ik)}")