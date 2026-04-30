import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append("./")
from motiondiff.data_pipeline.utils.skeleton import Skeleton, load_bvh_animation
from motiondiff.utils.hybrik import batch_rigid_transform

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets",
)
parser.add_argument("--data_dir", type=str, default="bones_full_raw_v14")
parser.add_argument("--csv_file", type=str, default="metadata_240527_v014.csv")
parser.add_argument("--out_dir", type=str, default="bones_full_joints_v14")
parser.add_argument("--num_process", type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--node_idx", type=int, default=0)
parser.add_argument("--debug", action="store_true", default=False)
args = parser.parse_args()


def parse_bvh(bvh_path):
    bvh_path_full = os.path.join(args.data_dir, bvh_path)
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvh_path_full)
    root_trans, joint_rot_mats = load_bvh_animation(bvh_path_full, skeleton)

    parent_indices = skeleton.get_parent_indices()
    joints = skeleton.get_neutral_joints()

    rot_mats = torch.tensor(joint_rot_mats)
    joints = torch.tensor(joints).unsqueeze(0).repeat(rot_mats.shape[0], 1, 1)
    parents = torch.LongTensor(parent_indices)
    joints -= joints[:, [0]]

    posed_joints, global_rot_mat = batch_rigid_transform(rot_mats, joints, parents)
    posed_joints += torch.tensor(root_trans).unsqueeze(1)
    # posed_joints = torch.stack([posed_joints[:, :, 0], posed_joints[:, :, 2], posed_joints[:, :, 1]], dim=-1) # no need to swap Y and Z
    posed_joints *= 0.01

    joint_rot_mats_path = os.path.join(
        args.out_dir, bvh_path.replace("BVH", "joint_rot_mats").replace("bvh", "npy")
    )
    posed_joints_path = os.path.join(
        args.out_dir, bvh_path.replace("BVH", "posed_joints").replace("bvh", "npy")
    )
    joint_rot_mats_dir = os.path.dirname(joint_rot_mats_path)
    posed_joints_dir = os.path.dirname(posed_joints_path)
    if not os.path.exists(joint_rot_mats_dir):
        os.makedirs(joint_rot_mats_dir, exist_ok=True)
    if not os.path.exists(posed_joints_dir):
        os.makedirs(posed_joints_dir, exist_ok=True)
    np.save(joint_rot_mats_path, joint_rot_mats.astype(np.float32))
    np.save(posed_joints_path, posed_joints.numpy().astype(np.float32))


if __name__ == "__main__":
    if args.debug:
        csv = pd.read_csv(os.path.join("dataset", args.data_dir, args.csv_file))
        args.data_dir = os.path.join(
            "/mnt/nvr_torontoai_humanmotionfm/datasets", args.data_dir
        )
        args.out_dir = os.path.join("dataset", args.out_dir)
    else:
        args.data_dir = os.path.join(args.dataset_dir, args.data_dir)
        args.out_dir = os.path.join(args.dataset_dir, args.out_dir)
        csv = pd.read_csv(os.path.join(args.data_dir, args.csv_file))
    root_trans_all = []
    joint_rot_mat_all = []
    manifest = []
    base = 0

    job_list_all = [path for path in csv.move_bvh_path]
    job_list_node = job_list_all[args.node_idx : len(job_list_all) : args.num_nodes]
    todo_job_list = [
        path
        for path in job_list_node
        if not os.path.exists(
            os.path.join(
                args.out_dir, path.replace("BVH", "posed_joints").replace("bvh", "npy")
            )
        )
    ]

    print(f"Node {args.node_idx} started working with {len(todo_job_list)} files...")
    num_batch = len(todo_job_list) // args.num_process
    if len(todo_job_list) != args.num_process * num_batch:
        num_batch += 1
    for batch_idx in tqdm(range(num_batch)):
        bvh_path_batch = todo_job_list[
            args.num_process * batch_idx : args.num_process * (batch_idx + 1)
        ]
        if len(bvh_path_batch) == 0:
            break

        try:
            if args.num_process == 1:
                parse_bvh(bvh_path_batch[0])
            else:
                with Pool(args.num_process) as p:
                    p.map(parse_bvh, bvh_path_batch)
        except Exception:
            print(f"Batch {batch_idx} failed!")
            continue

    print(f"Node {args.node_idx} finished working ...")
