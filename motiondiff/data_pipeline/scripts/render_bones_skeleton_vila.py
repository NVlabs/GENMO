import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append("./")
from motiondiff.data_pipeline.bones.vis_v2 import SMPLVisualizer
from motiondiff.data_pipeline.humanml.common.quaternion import *
from motiondiff.utils.torch_transform import (
    get_y_heading_q,
    rotation_matrix_to_quaternion,
)


def canonicalize(posed_joints, joint_rot_mats):
    positions = posed_joints.numpy()
    joint_quat = rotation_matrix_to_quaternion(joint_rot_mats)
    root_quat = joint_quat[:, 0].clone()
    heading_quat = get_y_heading_q(root_quat)
    # root_quat_wo_heading = quat_mul(quat_conjugate(heading_quat), root_quat).numpy()

    heading_quat = heading_quat.unsqueeze(1).repeat(1, joint_quat.shape[1], 1)

    root_quat = root_quat.numpy()
    heading_quat = heading_quat.numpy()
    heading_quat_inv = qinv_np(heading_quat)
    init_heading_quat_inv = np.repeat(
        heading_quat_inv[[0]], heading_quat_inv.shape[0], axis=0
    )

    """XZ at origin"""
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    """All initially face Z+"""
    positions = qrot_np(init_heading_quat_inv, positions)
    heading_quat = qmul_np(
        heading_quat, init_heading_quat_inv
    )  # normalize heading coordiante, so the heading is 0 for the first frame
    heading_quat_inv = qinv_np(heading_quat)

    return torch.Tensor(positions)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets",
)
parser.add_argument("--data_dir", type=str, default="bones_full_raw_v14")
parser.add_argument("--joints_dir", type=str, default="bones_full_joints_v14")
parser.add_argument("--csv_file", type=str, default="metadata_240527_v014.csv")
parser.add_argument("--out_dir", type=str, default="bones_for_vila/bones_skeleton_v1.0")
parser.add_argument("--skel_version", type=str, default="29-joints")
parser.add_argument("--num_vis", type=int, default=100000)
parser.add_argument("--max_seq_seconds", type=int, default=10)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--resolution", type=int, default=640)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--node_idx", type=int, default=0)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--debug_vis", action="store_true", default=False)
parser.add_argument("--gen_meta", action="store_true", default=False)
args = parser.parse_args()


if __name__ == "__main__":
    if args.debug:
        args.joints_dir = os.path.join(
            "/mnt/nvr_torontoai_humanmotionfm/datasets", args.joints_dir
        )
        args.out_dir = os.path.join("dataset/bones", args.out_dir)
        csv = pd.read_csv(os.path.join("dataset/bones", args.data_dir, args.csv_file))
    else:
        args.joints_dir = os.path.join(args.dataset_dir, args.joints_dir)
        args.out_dir = os.path.join(args.dataset_dir, args.out_dir)
        csv = pd.read_csv(os.path.join(args.dataset_dir, args.data_dir, args.csv_file))

    os.makedirs(os.path.join(args.out_dir, "log"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(args.out_dir, "log", f"node_{args.node_idx}_log.txt")
            ),
            logging.StreamHandler(),
        ],
    )

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

    """Select bones sequences by unique content name """
    job_list_all = []
    row_list_all = []
    content_set = set()
    for i in range(len(csv.move_bvh_path)):
        if csv.content_name[i] not in content_set:
            job_list_all += [csv.move_bvh_path[i]]
            row_list_all += [i]
            content_set.add(csv.content_name[i])

    if args.gen_meta:
        random.seed(0)
        random.shuffle(row_list_all)
        num_train = int(len(row_list_all) * 0.9)
        meta_train, meta_test, meta_all = [], [], []
        for idx in range(len(row_list_all)):
            meta = {
                "id": idx,
                "video": csv.move_bvh_path[row_list_all[idx]]
                .replace("BVH", "video")
                .replace("bvh", "mp4"),
                "conversations": [
                    {
                        "from": "human",
                        "value": "<video>\n Can you describe the motion of the person in the video.",
                    },
                    {
                        "from": "gpt",
                        "value": csv.content_natural_desc_1[row_list_all[idx]],
                    },
                ],
            }
            if idx < num_train:
                meta_train += [meta]
            else:
                meta_test += [meta]
            meta_all += [meta]
        json.dump(
            meta_train, open(os.path.join(args.out_dir, "train.json"), "w"), indent=4
        )
        json.dump(
            meta_test, open(os.path.join(args.out_dir, "test.json"), "w"), indent=4
        )
        json.dump(meta_all, open(os.path.join(args.out_dir, "all.json"), "w"), indent=4)
        exit()

    job_list_all = job_list_all[: args.num_vis]
    job_list_node = job_list_all[args.node_idx : len(job_list_all) : args.num_nodes]
    job_list_todo = [
        path
        for path in job_list_node
        if not os.path.exists(
            os.path.join(
                args.out_dir, path.replace("BVH", "video").replace("bvh", "mp4")
            )
        )
    ]
    # if not args.debug:
    #     random.shuffle(job_list_todo)

    visualizer = SMPLVisualizer(
        joint_parents=parents,
        distance=6,
        elevation=2,
        use_floor_texture=False,
        verbose=False,
        display_num=f":{args.node_idx + 500}",
    )
    if args.debug_vis:
        job_list_todo = job_list_todo[:1]

    logging.info(
        f"Node {args.node_idx} started working with {len(job_list_todo)} files..."
    )
    base_rot = torch.FloatTensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    for idx, bvh_path in enumerate(tqdm(job_list_todo)):
        logging.info(
            f"Node {args.node_idx} started working with {idx} / {len(job_list_todo)} job: {bvh_path}"
        )

        joint_path = os.path.join(
            args.joints_dir,
            bvh_path.replace("BVH", "posed_joints").replace("bvh", "npy"),
        )
        bones_joints = torch.from_numpy(np.load(joint_path))[:: (120 // args.fps)]
        bones_joints = bones_joints[: args.max_seq_seconds * args.fps]

        """ Bones joints are accidently swapped between Y and Z when saving"""
        bones_joints = torch.stack(
            [bones_joints[..., 0], bones_joints[..., 2], bones_joints[..., 1]], dim=-1
        )

        rotmat_path = os.path.join(
            args.joints_dir,
            bvh_path.replace("BVH", "joint_rot_mats").replace("bvh", "npy"),
        )
        bones_rotmats = torch.from_numpy(np.load(rotmat_path))[:: (120 // args.fps)]
        bones_rotmats = bones_rotmats[: args.max_seq_seconds * args.fps]

        """ Canonicalize the facing direction to Z+"""
        bones_joints = canonicalize(bones_joints.float(), bones_rotmats.float())

        """ Convert to Z up for rendering"""
        bones_joints = bones_joints @ base_rot

        if args.debug_vis:
            bones_joints = bones_joints[:120]

        smpl_seq = {"gt": {"joints_pos": bones_joints}}
        video_path = os.path.join(
            args.out_dir, bvh_path.replace("BVH", "video").replace("bvh", "mp4")
        )
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        frame_dir = f"out/bones/frames/{bvh_path}"
        visualizer.save_animation_as_video(
            video_path,
            init_args={"smpl_seq": smpl_seq, "mode": "gt", "view_dir": "Y-"},
            window_size=(1000, 1000),
            frame_dir=frame_dir,
            fps=args.fps,
            enable_shadow=True,
        )
        if args.resolution != 1000:
            video_path_tmp = video_path[:-4] + "_tmp.mp4"
            os.system(
                f"ffmpeg -i {video_path} -s {args.resolution}x{args.resolution} -c:a copy {video_path_tmp} -hide_banner -loglevel error"
            )
            os.rename(video_path_tmp, video_path)
