import argparse
import os
import sys

import numpy as np
import torch

sys.path.append("./")
from motiondiff.data_pipeline.bones.vis import SMPLVisualizer
from motiondiff.data_pipeline.utils.skeleton import Skeleton, load_bvh_animation
from motiondiff.utils.hybrik import batch_rigid_transform

# data_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_joints'
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets",
)
parser.add_argument("--joints_dir", type=str, default="bones_full_raw_v14")
parser.add_argument("--skel_version", type=str, default="29-joints")
parser.add_argument("--num_vis", type=int, default=10)
args = parser.parse_args()

if args.skel_version == "29-joints":
    parents = torch.LongTensor(
        [
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
    )
elif args.skel_version == "27-joints":
    parents = torch.LongTensor(
        [
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
            4,
            13,
            14,
            15,
            16,
            16,
            0,
            19,
            20,
            21,
            0,
            23,
            24,
            25,
        ]
    )

joints_dir = os.path.join(args.dataset_dir, args.joints_dir, "posed_joints")
file_paths = []
for subdir in os.listdir(joints_dir):
    for file in os.listdir(os.path.join(joints_dir, subdir)):
        file_paths.append(os.path.join(subdir, file))

for i, fname in enumerate(file_paths):
    print(f"Processing {fname}")

    posed_joints = torch.from_numpy(np.load(os.path.join(joints_dir, fname)))

    vis = SMPLVisualizer(joint_parents=parents, distance=7, elevation=10)
    smpl_seq = {"gt": {"joints_pos": posed_joints.float()}}
    video_path = f"out/bones/{args.joints_dir}/{os.path.basename(fname)[:-4]}.mp4"
    frame_dir = f"out/bones/frames/{os.path.basename(fname)[:-4]}"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    vis.save_animation_as_video(
        video_path,
        init_args={"smpl_seq": smpl_seq, "mode": "gt"},
        window_size=(1500, 1500),
        frame_dir=frame_dir,
        fps=120,
    )

    if i == args.num_vis - 1:
        break
