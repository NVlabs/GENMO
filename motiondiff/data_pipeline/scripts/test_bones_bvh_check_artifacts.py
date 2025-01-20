import sys
import torch
import numpy as np
import os
import csv
import multiprocessing
from scipy.spatial.transform import Rotation  
sys.path.append('./')
from motiondiff.data_pipeline.utils.skeleton import Skeleton, load_bvh_animation
from motiondiff.utils.hybrik import batch_rigid_transform
from motiondiff.data_pipeline.scripts.vis_bones import SMPLVisualizer
from motiondiff.data_pipeline.humanml.common.quaternion import qinv, qmul

FILTER_MIRRORS = True

VIZ = False # viz all sequences

num_joints = 29 #27

CHECK_FLOOR_PEN = True
FLOOR_PEN_THRESH = 5.0 # min cm below the floor
CHECK_JITTERS = True
JITTER_THRESH = 15.0 # max cm to move in 1/120 of a sec
CHECK_BONE_FLIP = True
BONE_FLIP_THRESH = 25.0 # max deg to rotate in 1/120 of a sec
CHECK_NO_MOVE = False
NO_MOVE_THRESH = 1e-5 # max cm to move in 1/120 of a sec
CHECK_FOOT_SKATE = False
FOOT_SKATE_HEIGHT_THRESH = 2.0 # max cm above the floor to be considered in "contact"
FOOT_SKATE_VEL_THRESH = 7.5 # max cm to move in 1/120 of sec when in "contact"

bvh_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw_v14'
out_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_check_artifacts_v14'
# bvh_dir = '../bones_data/foundation_data/v014_retarget'
# out_dir = './out/bones_check_artifacts_v14'
# bvh_dir = '../bones_data/foundation_data/sample__orig_skeleton_markers/231120'
# file_names = [
#     'P8/BVH/big_light_one_hand_behind_high_to_right_side_high_R_001__A527_M.bvh',
#     'P8/BVH/crouch_ff_stop_315_R_003__A245_M.bvh',
#     'P8/BVH/jog_ff_start_180_R_002__A247_M.bvh',
#     'P3/BVH/sending_kisses_R_003__A097.bvh',
#     'P3/BVH/thinking_R_003__A101.bvh',
#     'P7/BVH/jump_ff_180_R_002__A232.bvh',
#     'P7/BVH/sensual_jump_ff_180_R_001__A229.bvh',
#     'P7/BVH/threaten_fist_R_001__A235_M.bvh',
#     'P11/BVH/spear_hit_head_R_001__A386_M.bvh',
#     'P11/BVH/spear_turn_walk_pass_start_360_R_003__A386_M.bvh'
# ]

# bad examples for debugging
# file_names = [
#     'P8/BVH/leaning_idle_003__A245.bvh',
#     'P2/BVH/jog_ff_loop_225_001__A052.bvh',
#     'P1/BVH/sit_croos_legged_loop__A024.bvh',
#     'P8/BVH/leaning_idle_003__A246.bvh',
#     'P1/BVH/sit_croos_legged_stop_001__A022.bvh',
#     'P8/BVH/sword_safety_roll_180_R_007__A250_M.bvh',
#     'P1/BVH/sit_croos_legged_stop_003__A021.bvh',
#     'P2/BVH/jog_ff_loop_225_001__A049.bvh',
#     'P1/BVH/sit_on_heels_loop_003__A026_M.bvh',
#     'P1/BVH/sit_on_heels_loop_002__A022.bvh',
#     'P1/BVH/sit_croos_legged_loop_003__A026_M.bvh',
#     'P8/BVH/leaning_idle_003__A247.bvh',
#     ]

# # skel samples
# file_names = [
#     'big_heavy_one_hand_behind_high_to_behind_high_R_001__A524.bvh',
#     'big_heavy_one_hand_behind_high_to_behind_high_R_001__A525.bvh',
#     'big_heavy_one_hand_behind_high_to_behind_high_R_001__A526.bvh',
#     'big_heavy_one_hand_behind_high_to_behind_high_R_001__A527.bvh',
# ]

# file_names = [
#     'BVH/230228/come_down_50cm_box_R_003__A224.bvh',
#     'BVH/230228/come_up_50cm_box_R_001__A228_M.bvh',
#     'BVH/230228/dancing_routine_V002_003__A225_M.bvh',
#     'BVH/230228/jog_ff_loop_180_R_003__A228.bvh',
#     'BVH/230228/jog_ff_stop_180_R_001__A225.bvh',
# ]

file_names = []
# meta_csv = os.path.join(bvh_dir, 'Metadata - 350 000 moves.csv')
meta_csv = os.path.join(bvh_dir, 'metadata_240527_v014.csv')
print('Loading metadata...')
with open(meta_csv, 'r') as f:
    reader = csv.DictReader(f)
    for ri, row in enumerate(reader):
        bvh_path = row['move_bvh_path']
        if FILTER_MIRRORS:
            if os.path.splitext(bvh_path)[0][-2:] != '_M':
                file_names.append(bvh_path)
        else:
            file_names.append(bvh_path)

print(f'{len(file_names)} entries in metadata (excluding mirrors).')

def write_line_to_file(fname, line, lock):
    # Acquire the lock before writing to the file
    lock.acquire()
    try:
        # Open the file in append mode and write the result
        with open(fname, 'a') as f:
            f.write(line + '\n')
    finally:
        # Release the lock after writing to the file
        lock.release()

def process_bvh_files(fname, all_files_out, check_artifacts_out, lock): 
    print(f'Processing {fname}...')
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

    # print(posed_joints.size()) # [T, 29, 3]

    # NOTES:
    #   - joint positions are in cm
    #   - Z is up axis after pose_joints is re-ordered

    seq_name = os.path.basename(fname)

    viz_check = False
    check_results = ''
    if CHECK_FLOOR_PEN:
        heights = posed_joints[:,:,2] # z val
        min_height = torch.min(heights).item()
        # print(min_height)
        if min_height < -FLOOR_PEN_THRESH:
            print(f'Detected floor penetration with height {min_height}!')
            check_results += f",floor_pen({min_height})"

    if CHECK_JITTERS or CHECK_NO_MOVE:
        vel = torch.norm(posed_joints[1:] - posed_joints[:-1], dim=-1) # [T-1, 29]
        if CHECK_JITTERS:
            max_vel = torch.max(vel).item()
            # print(max_vel)
            if max_vel > JITTER_THRESH:
                print(f'Detected jitter with velocity {max_vel}!')
                check_results += f",jitter({max_vel})"
        if CHECK_NO_MOVE:
            min_vel = torch.min(vel).item()
            print(min_vel)
            if min_vel < NO_MOVE_THRESH:
                print(f'Detected no move with velocity {min_vel}!')
                check_results += f",no_move({min_vel})"

    if CHECK_BONE_FLIP:
        # local joint_rot_mats [T, 29, 3, 3]
        T, J, _, _ = rot_mats.size()
        rot_quat = torch.tensor(Rotation.from_matrix(joint_rot_mats.reshape((T*J, 3, 3))).as_quat().reshape((T, num_joints, 4)))
        rot_quat = torch.concatenate([rot_quat[..., [3]], rot_quat[..., :3]], dim=-1) # w,x,y,z format
        qdiff = qmul(qinv(rot_quat[:-1]), rot_quat[1:])
        rot_vel = 2 * torch.atan2(torch.norm(qdiff[...,1:], dim=-1), qdiff[...,0]) # [T, 29]
        rot_vel[rot_vel > np.pi] -= 2 * np.pi
        rot_vel[rot_vel < -np.pi] += 2 * np.pi
        rot_vel = torch.rad2deg(rot_vel)
        max_rot = torch.max(rot_vel).item()
        # print(max_rot)
        if max_rot > BONE_FLIP_THRESH:
            print(f'Detected bone flip with rot {max_rot}!')
            check_results += f",bone_flip({max_rot})"  

    if CHECK_FOOT_SKATE:
        toe_idx = [24, 28]
        toe_pos = posed_joints[:,toe_idx]
        toe_heights = toe_pos[..., 2] # z val
        if torch.sum(toe_heights < FOOT_SKATE_HEIGHT_THRESH).item() > 0:
            floor_height = torch.median(toe_heights[toe_heights < FOOT_SKATE_HEIGHT_THRESH]) # the floor may vary by a few cm and cause false postives
        else:
            floor_height = FOOT_SKATE_HEIGHT_THRESH
        # print(f'floor_height={floor_height}')
        toe_heights = toe_heights - floor_height
        contact_idx = torch.logical_and(toe_heights < FOOT_SKATE_HEIGHT_THRESH, toe_heights > -FOOT_SKATE_HEIGHT_THRESH)
        # print(f'toe_height={torch.min(toe_heights).item()}')
        # print(torch.sum(contact_idx))

        if torch.sum(contact_idx).item() > 0:
            toe_vel = torch.norm(toe_pos[1:] - toe_pos[:-1], dim=-1) # [T-1, 2]
            toe_vel = torch.cat([toe_vel, toe_vel[-1:]], dim=0)
            max_ctc_vel = torch.max(toe_vel[contact_idx]).item()
            # print(f'ctc_vel={max_ctc_vel}')
            if max_ctc_vel > FOOT_SKATE_VEL_THRESH:
                print(f'Detected foot skate with vel {max_ctc_vel}!')
                check_results += f",foot_skate({max_ctc_vel})" 
        

    if check_results != '':
        write_line_to_file(check_artifacts_out, fname + check_results, lock)

    if VIZ or check_results != '':
        print(f'Visualizing {fname}...')
        vis = SMPLVisualizer(joint_parents=parents, distance=7, elevation=10)
        smpl_seq = {
            'gt':{
                'joints_pos': posed_joints.float() * 0.01
            }
        }
        video_path = os.path.join(out_dir, f'videos/{fname[:-4]}.mp4')
        frame_dir = os.path.join(out_dir, f'videos/frames/{os.path.basename(fname)[:-4]}')
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        vis.save_animation_as_video(video_path, init_args={'smpl_seq': smpl_seq, 'mode': 'gt'}, window_size=(1500, 1500), frame_dir=frame_dir, fps=120)

    write_line_to_file(all_files_out, fname, lock)

all_files_out = os.path.join(out_dir, 'check_finished.csv')
check_results_out = os.path.join(out_dir, 'check_artifacts.csv')

os.makedirs(os.path.dirname(all_files_out), exist_ok=True)

if os.path.exists(all_files_out):
    # already checked some files, load these in and don't repeat
    prechecked = set()
    with open(all_files_out, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    prechecked = set(lines)
    print(f'Already checked {len(prechecked)} from previous run, skipping these.')
    file_names = [name for name in file_names if name not in prechecked]
    print(f'Processing {len(file_names)} entries remaining entries...')

# # serial
# for fname in file_names:
#     process_bvh_files(fname, check_floor_pen_out, None)

# parallel
num_proc = multiprocessing.cpu_count()
print(f'Using {num_proc} processes...')
# Create a multiprocessing manager
manager = multiprocessing.Manager()
# Create a lock for synchronization
lock = manager.Lock()
with multiprocessing.Pool(processes=num_proc) as pool:
    results = pool.starmap(process_bvh_files, 
                           [(arg, all_files_out, check_results_out, lock) 
                            for arg in file_names])

