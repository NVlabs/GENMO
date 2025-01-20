import sys
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import json
import torch
import argparse
from multiprocessing import Pool
import pdb
sys.path.append('./')
from motiondiff.data_pipeline.utils.skeleton import Skeleton, load_bvh_animation
from motiondiff.utils.hybrik import batch_rigid_transform


data_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw'
# data_dir = 'dataset/bones_full_raw'
csv_path = os.path.join(data_dir, 'Metadata - 350 000 moves.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--num_process', type=int, default=1)
parser.add_argument('--num_nodes', type=int, default=1)
parser.add_argument('--node_idx', type=int, default=0)
args = parser.parse_args()

def get_neutral_joints(bvh_path):
    bvh_path_full = os.path.join(data_dir, bvh_path)
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvh_path_full)
    joints = skeleton.get_neutral_joints()
    parent_indices = skeleton.get_parent_indices()

    joints_par = joints[parent_indices[1:]]
    bone_length = np.linalg.norm(joints[1:] - joints_par, axis=1) * 10
    return np.round(bone_length)

bone_length_oracle = np.array([ 85.,  95.,  95.,  95., 251., 130., 184., 159., 295., 233.,  70.,
        57.,  70., 184., 159., 295., 233.,  70.,  57.,  70.,  99., 412.,
       456., 171.,  99., 412., 456., 171.]) # mm

if __name__ == '__main__':

    csv = pd.read_csv(csv_path)
    root_trans_all = []
    joint_rot_mat_all = []
    manifest = []
    base = 0

    todo_path_list = csv.move_bvh_path[args.node_idx : len(csv.move_bvh_path) : args.num_nodes]

    f_wrong_proportion = open('bones_wrong_proportion.txt', 'w')
    for batch_idx in tqdm(range(len(todo_path_list) // args.num_process + 1)):
        bvh_path_batch = todo_path_list[args.num_process * batch_idx: args.num_process * (batch_idx + 1)]
        if len(bvh_path_batch) == 0: break

        bvh_path_batch = [path for path in bvh_path_batch if os.path.exists(os.path.join(data_dir, path))]
        if len(bvh_path_batch) < args.num_process: continue

        try:
            if args.num_process == 1:
                output_batch = [get_neutral_joints(bvh_path_batch[0])]
            else:
                with Pool(args.num_process) as p:
                    output_batch = p.map(get_neutral_joints, bvh_path_batch)
            for i, bl in enumerate(output_batch):
                if (np.abs(bl - bone_length_oracle) > 1).sum() > 0:
                    print("Wrong proportion", bvh_path_batch[i])
                    f_wrong_proportion.write(bvh_path_batch[i] + '\n')
        except Exception:
            print(f"Batch {batch_idx} failed!")
            continue
    f_wrong_proportion.close()