import argparse
import os
import sys
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("./")
from motiondiff.data_pipeline.bones.motion_process import extract_features
from motiondiff.data_pipeline.bones.skeleton_params import bones_parents

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets",
)
parser.add_argument("--data_dir", type=str, default="bones_full_raw_v14")
parser.add_argument("--joints_dir", type=str, default="bones_full_joints_v14")
parser.add_argument("--out_dir", type=str, default="bones_full353_v2.0")
parser.add_argument("--csv_file", type=str, default="metadata_240527_v014.csv")
parser.add_argument("--use_joint_local_height", action="store_true", default=False)
parser.add_argument("--use_joint_local_vel", action="store_true", default=False)
parser.add_argument("--num_process", type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--node_idx", type=int, default=0)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--vis", action="store_true", default=False)
args = parser.parse_args()


def process_one_sequence(params):
    idx, bvh_path = params

    npy_path = bvh_path.replace("bvh", "npy")
    posed_joints = np.load(
        os.path.join(args.joints_dir, npy_path.replace("BVH", "posed_joints"))
    )
    joint_rot_mats = np.load(
        os.path.join(args.joints_dir, npy_path.replace("BVH", "joint_rot_mats"))
    )

    feature = extract_features(
        joint_rot_mats,
        posed_joints,
        use_root_local_rot=True,
        use_joint_local_height=args.use_joint_local_height,
        use_joint_local_vel=args.use_joint_local_vel,
    )

    feature_path = os.path.join(args.out_dir, "new_joint_vecs", f"{idx:06d}.npy")
    np.save(feature_path, feature.astype(np.float32))


if __name__ == "__main__":
    if args.debug:
        csv = pd.read_csv(os.path.join("dataset/bones", args.data_dir, args.csv_file))
        args.data_dir = os.path.join(
            "/mnt/nvr_torontoai_humanmotionfm/datasets", args.data_dir
        )
        args.joints_dir = os.path.join(
            "/mnt/nvr_torontoai_humanmotionfm/datasets", args.joints_dir
        )
        args.out_dir = os.path.join("dataset/bones", args.out_dir)
    else:
        args.data_dir = os.path.join(args.dataset_dir, args.data_dir)
        args.joints_dir = os.path.join(args.dataset_dir, args.joints_dir)
        args.out_dir = os.path.join(args.dataset_dir, args.out_dir)
        csv = pd.read_csv(os.path.join(args.data_dir, args.csv_file))

    """ Process motion features """
    if not args.vis:
        job_list_all = [(idx, path) for idx, path in enumerate(csv.move_bvh_path)]
        job_list_node = job_list_all[
            args.node_idx : len(csv.move_bvh_path) : args.num_nodes
        ]
        todo_job_list = [
            (idx, path)
            for (idx, path) in job_list_node
            if not os.path.exists(
                os.path.join(args.out_dir, "new_joint_vecs", f"{idx:06d}.npy")
            )
        ]

        print(
            f"Node {args.node_idx} started working with {len(todo_job_list)} files..."
        )
        os.makedirs(os.path.join(args.out_dir, "new_joint_vecs"), exist_ok=True)
        num_batch = len(todo_job_list) // args.num_process
        if len(todo_job_list) != args.num_process * num_batch:
            num_batch += 1
        for batch_idx in tqdm(range(num_batch)):
            job_batch = todo_job_list[
                args.num_process * batch_idx : args.num_process * (batch_idx + 1)
            ]
            if len(job_batch) == 0:
                break

            try:
                if args.num_process == 1:
                    process_one_sequence(job_batch[0])
                else:
                    with Pool(args.num_process) as p:
                        p.map(process_one_sequence, job_batch)
            except Exception:
                print(f"Batch {batch_idx} failed!")
                print(traceback.format_exc())
                continue

        print(f"Node {args.node_idx} finished working ...")

    """ Visualize converted feature """
    if args.debug and args.vis:
        import random

        import torch

        from motiondiff.data_pipeline.bones.motion_process import (
            recover_from_ric,
            recover_from_ric_with_joint_rot,
        )
        from motiondiff.data_pipeline.bones.vis_v2 import SMPLVisualizer

        neutral_joints = torch.load("assets/bones/skeleton/joints.p")
        joint_parents = torch.load("assets/bones/skeleton/parents.p")

        sampled_paths = random.sample(
            os.listdir(os.path.join(args.out_dir, "new_joint_vecs")), 10
        )
        # sampled_paths = [f'{i:06d}.npy' for i in range(10)]
        base_rot = torch.FloatTensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        for path in sampled_paths:
            feature = np.load(os.path.join(args.out_dir, "new_joint_vecs", path))
            # positions = recover_from_ric(torch.tensor(feature).float(), 29, use_global_joint_height=not args.use_local_height)
            positions, joints_rot = recover_from_ric_with_joint_rot(
                torch.tensor(feature).float(),
                neutral_joints,
                joint_parents,
                return_joint_rot=True,
            )
            positions = positions @ base_rot  # Y-up to Z-up

            smpl_seq = {"gt": {"joints_pos": positions}}

            vis = SMPLVisualizer(
                joint_parents=bones_parents, distance=7, elevation=2, verbose=False
            )
            print("saving test videos...")
            os.makedirs(f"out/bones/{os.path.basename(args.out_dir)}", exist_ok=True)
            out_path = f"out/bones/{os.path.basename(args.out_dir)}/{path[:-4]}.mp4"
            frame_dir = "out/frames"
            vis.save_animation_as_video(
                out_path,
                init_args={"smpl_seq": smpl_seq, "mode": "gt", "view_dir": "Y-"},
                window_size=(800, 800),
                frame_dir=frame_dir,
                fps=20,
                crf=15,
            )
            print(f"test videos saved to {out_path}")
