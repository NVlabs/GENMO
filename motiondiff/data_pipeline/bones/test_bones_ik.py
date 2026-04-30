import argparse
import os
import pdb
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append("./")
from motiondiff.data_pipeline.bones.vis_v2 import SMPLVisualizer
from motiondiff.models.common.smpl import SMPL
from motiondiff.utils.hybrik import (
    batch_inverse_kinematics_transform_bones,
    batch_rigid_transform,
)

"""Skeleton parameters"""
from motiondiff.data_pipeline.bones.skeleton_params import (
    BONESPose,
    bones_beta,
    bones_children_map,
    bones_joints_rest_eye,
    bones_parents,
)

bones_parents = torch.LongTensor(bones_parents)
bones_children_map = torch.LongTensor(bones_children_map)
bones_beta = torch.FloatTensor([bones_beta]).to("cuda")
bones_joints_rest_eye = torch.FloatTensor(bones_joints_rest_eye)

"""Optimization and help functions"""
smpl = SMPL("data/smpl_data", create_transl=False, gender="neutral").to("cuda")


def vis(visualizer, bones_joints, bones_joints_ik, video_path, fps=120):
    base_rot = torch.FloatTensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).to("cuda")
    bones_joints = bones_joints @ base_rot
    bones_joints_ik = bones_joints_ik @ base_rot

    bones_joints[:, :, 0] -= 0.8
    # bones_joints_ik[:, :, 0] += 0.8

    smpl_seq = {
        "bones_joints_retarget_ik": {
            "joints_pos": bones_joints_ik.float(),
            "parents": bones_parents,
        },
        "gt": {
            "joints_pos": bones_joints.float(),
            "parents": bones_parents,
        },
    }
    frame_dir = f"out/bones/frames/bones_to_smpl+{random.randint(0, 1e9)}"
    visualizer.save_animation_as_video(
        video_path,
        init_args={"smpl_seq": smpl_seq, "mode": "gt"},
        window_size=(1500, 1500),
        frame_dir=frame_dir,
        fps=fps,
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets",
)
parser.add_argument("--data_dir", type=str, default="bones_full_raw_v14")
parser.add_argument("--joints_dir", type=str, default="bones_full_joints_v14")
parser.add_argument("--csv_file", type=str, default="metadata_240527_v014.csv")
parser.add_argument("--out_dir", type=str, default="out/bones/bones_ik_test/add_eye")
parser.add_argument("--downsample_rate", type=int, default=4)
parser.add_argument("--add_joints", type=str, default="eye")
args = parser.parse_args()


if __name__ == "__main__":
    visualizer = SMPLVisualizer(
        joint_parents=None, distance=7, elevation=2, smpl=smpl, verbose=False
    )

    csv = pd.read_csv(os.path.join("dataset/bones", args.data_dir, args.csv_file))
    args.data_dir = os.path.join(
        "/mnt/nvr_torontoai_humanmotionfm/datasets", args.data_dir
    )
    args.joints_dir = os.path.join(
        "/mnt/nvr_torontoai_humanmotionfm/datasets", args.joints_dir
    )
    os.makedirs(args.out_dir, exist_ok=True)

    bones_rest_joints = torch.from_numpy(
        np.load(os.path.join(args.joints_dir, "neutral_pose.npy"))
    )
    if args.add_joints is not None:
        bones_rest_joints = torch.cat(
            [bones_rest_joints, bones_joints_rest_eye.unsqueeze(0).cpu()], dim=1
        )
        bones_parents = torch.cat(
            [bones_parents, torch.LongTensor([BONESPose.Head, BONESPose.Head])], dim=-1
        )
        bones_children_map = torch.cat(
            [bones_children_map, torch.LongTensor([-1, -1])], dim=-1
        )
        bones_children_map[BONESPose.Head] = BONESPose.LEye

    errors_ik = []
    for idx, bvh_path in enumerate(tqdm(csv.move_bvh_path)):
        if idx % 100 != 0:
            continue
        print(f"Vis {idx}th seq: {bvh_path}")
        video_path = os.path.join(
            args.out_dir, os.path.basename(bvh_path).replace("bvh", "mp4")
        )

        joint_path = os.path.join(
            args.joints_dir,
            bvh_path.replace("BVH", "posed_joints").replace("bvh", "npy"),
        )
        rotmat_path = os.path.join(
            args.joints_dir,
            bvh_path.replace("BVH", "joint_rot_mats").replace("bvh", "npy"),
        )
        bones_joints = torch.from_numpy(
            np.load(joint_path)[:: args.downsample_rate]
        ).to("cuda")
        # ''' Bones joints are accidently swapped between Y and Z when saving'''
        bones_joints_ori = torch.stack(
            [bones_joints[..., 0], bones_joints[..., 2], bones_joints[..., 1]], dim=-1
        )
        bones_trans = bones_joints_ori[:, [0]]

        bones_rotmats = torch.from_numpy(
            np.load(rotmat_path)[:: args.downsample_rate]
        ).to("cuda")
        if args.add_joints is not None:
            bones_rotmats = torch.cat(
                [bones_rotmats, bones_rotmats[:, -2:]], dim=1
            )  # any rotations for eyes
        bones_rest_joints_seq = bones_rest_joints.repeat_interleave(
            bones_rotmats.shape[0], dim=0
        ).to("cuda")
        bones_joints_noroot, _ = batch_rigid_transform(
            bones_rotmats, bones_rest_joints_seq, bones_parents
        )

        """IK"""
        phis = torch.tensor([1.0, 0.0], device="cuda").expand(
            bones_rotmats.shape[0], bones_rotmats.shape[1], -1
        )
        if args.add_joints is not None:
            leaf_thetas = torch.eye(3, device="cuda").expand(
                bones_rotmats.shape[0], 8, -1, -1
            )
        else:
            leaf_thetas = torch.eye(3, device="cuda").expand(
                bones_rotmats.shape[0], 7, -1, -1
            )
        bones_rotmats_ik, _, _ = batch_inverse_kinematics_transform_bones(
            bones_joints_noroot,
            None,
            phis,
            bones_rest_joints_seq,
            bones_children_map,
            bones_parents,
            leaf_thetas,
            False,
            add_eye=args.add_joints is not None,
        )
        bones_joints_ik, _ = batch_rigid_transform(
            bones_rotmats_ik.to("cuda"), bones_rest_joints_seq, bones_parents
        )

        """ Compute metric given joints (no hand joints) """
        selected_joints = np.array(
            [
                i
                for i in range(bones_rotmats.shape[1])
                if i
                not in [
                    BONESPose.LeftHandEnd,
                    BONESPose.LeftHandThumb1,
                    BONESPose.LeftHandThumb2,
                    BONESPose.RightHandEnd,
                    BONESPose.RightHandThumb1,
                    BONESPose.RightHandThumb2,
                ]
            ]
        )
        err_ik = np.linalg.norm(
            (bones_joints_noroot - bones_joints_ik).cpu().numpy()[:, selected_joints],
            axis=2,
        ).mean()
        print(f"Retarget error IK = {err_ik}")
        errors_ik += [err_ik]

        """ Vis """
        bones_joints = bones_joints_noroot + bones_trans
        bones_joints_ik = bones_joints_ik + bones_trans
        vis(
            visualizer,
            bones_joints,
            bones_joints_ik,
            video_path,
            fps=120 / args.downsample_rate,
        )

    print(f"{args.regress_ver} produces average errors IK = {np.average(errors_ik)}")
