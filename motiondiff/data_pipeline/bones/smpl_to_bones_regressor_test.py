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
from motiondiff.models.common.smpl import SMPL
from motiondiff.models.mdm.rotation_conversions import axis_angle_to_matrix
from motiondiff.utils.conversion import humanml_to_smpl
from motiondiff.utils.hybrik import (
    batch_inverse_kinematics_transform_bones,
    batch_rigid_transform,
)
from motiondiff.utils.torch_utils import tensor_to

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


def vis(
    visualizer,
    smpl_motion,
    bones_joints_retarget,
    bones_joints_retarget_ik,
    video_path,
    fps=120,
):
    base_rot = torch.FloatTensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).to("cuda")
    smpl_joints = smpl_motion.joints
    smpl_verts = smpl_motion.vertices
    bones_joints_retarget = bones_joints_retarget @ base_rot
    bones_joints_retarget_ik = bones_joints_retarget_ik @ base_rot

    smpl_verts_v2 = smpl_verts.clone()
    smpl_verts[:, :, 0] += 0.8  # 1
    bones_joints_retarget[:, :, 0] += 0.8  # 1
    # bones_joints_retarget_ik[:, :, 0] += 0 # 2
    smpl_joints[:, :, 0] -= 0.8  # 3
    smpl_verts_v2[:, :, 0] -= 0.8  # 3

    smpl_seq = {
        "bones_joints_retarget": {
            "joints_pos": bones_joints_retarget.float(),
            "parents": bones_parents,
        },
        "smpl_mesh": {
            "smpl_verts": smpl_verts.float(),
            "show_smpl_mesh": True,
            "opacity": 0.8,
        },
        "bones_joints_retarget_ik": {
            "joints_pos": bones_joints_retarget_ik.float(),
            "parents": bones_parents,
        },
        "smpl_joints": {
            "joints_pos": smpl_joints.float(),
            "parents": smpl.parents,
        },
        "smpl_mesh_v2": {
            "smpl_verts": smpl_verts_v2.float(),
            "show_smpl_mesh": True,
            "opacity": 0.8,
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
parser.add_argument("--dataset", type=str, default="HumanML3D")
parser.add_argument("--joints_dir", type=str, default="bones_full_joints_v14")
parser.add_argument(
    "--out_dir", type=str, default="dataset/bones/bones_to_smpl/bones_to_smpl_v14.6"
)
parser.add_argument("--add_joints", type=str, default="eye")
parser.add_argument("--regress_ver", type=str, default="split_xyz")
parser.add_argument("--vis", action="store_true", default=False)
args = parser.parse_args()


if __name__ == "__main__":
    if args.vis:
        from motiondiff.data_pipeline.bones.vis_v2 import SMPLVisualizer

        visualizer = SMPLVisualizer(
            joint_parents=None, distance=7, elevation=2, smpl=smpl, verbose=False
        )

    args.joints_dir = os.path.join(
        "/mnt/nvr_torontoai_humanmotionfm/datasets", args.joints_dir
    )
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
    njoints = bones_parents.shape[0]

    regressor_path = os.path.join(
        args.out_dir, f"smpl_to_bones_regressor_{args.regress_ver}.npy"
    )
    regressor = np.load(regressor_path)

    motion_dir = os.path.join("dataset", args.dataset, "new_joint_vecs")
    motion_files = os.listdir(motion_dir)
    mean = torch.zeros(263).float().to("cuda")
    std = torch.ones(263).float().to("cuda")

    errors_ik = []
    base_rot_to_y_up = torch.FloatTensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).to("cuda")
    for i in range(len(motion_files)):
        motion_path = os.path.join(motion_dir, f"{i:06d}.npy")
        print(motion_path)
        motion = np.load(motion_path)
        samples = motion.T.reshape(
            1, motion.shape[-1], 1, -1
        )  # [batch, nfeat, 1, seq_len]
        samples = torch.from_numpy(samples).float().to("cuda")
        smpl_thetas, smpl_trans, _ = humanml_to_smpl(
            samples, mean=mean, std=std, smpl=smpl
        )  # Z-up
        smpl_thetas = smpl_thetas[0]
        smpl_trans = smpl_trans[0]

        smpl_motion_noroot = smpl(
            global_orient=smpl_thetas[:, :3],
            body_pose=smpl_thetas[:, 3:],
            betas=bones_beta,
            root_trans=torch.zeros_like(smpl_thetas[:, :3]),
            return_full_pose=True,
            orig_joints=True,
        )
        smpl_verts = smpl_motion_noroot.vertices
        smpl_verts = smpl_verts @ base_rot_to_y_up

        """ run regressor """
        if args.regress_ver == "concat_xyz":
            njoints = regressor.shape[1] / 3
            smpl_verts = smpl_verts.reshape(smpl_verts.shape[0], -1).cpu().numpy()
            bones_joints_retarget = smpl_verts.dot(regressor)
            bones_joints_retarget = torch.from_numpy(
                bones_joints_retarget.reshape(-1, njoints, 3)
            ).to("cuda")
        elif args.regress_ver == "split_xyz":
            njoints = regressor.shape[1]
            smpl_verts = (
                smpl_verts.transpose(1, 2)
                .reshape(-1, smpl_verts.shape[1])
                .cpu()
                .numpy()
            )
            bones_joints_retarget = smpl_verts.dot(regressor)
            bones_joints_retarget = (
                torch.from_numpy(bones_joints_retarget.reshape(-1, 3, njoints))
                .transpose(1, 2)
                .to("cuda")
            )
        elif args.regress_ver == "wo_glb_orient_split_xyz":
            njoints = regressor.shape[1]
            rot_mats = axis_angle_to_matrix(smpl_thetas[:, :3]).transpose(1, 2)
            smpl_verts = torch.matmul(smpl_verts, rot_mats)
            smpl_verts = (
                smpl_verts.transpose(1, 2)
                .reshape(-1, smpl_verts.shape[1])
                .cpu()
                .numpy()
            )
            bones_joints_retarget = smpl_verts.dot(regressor)
            bones_joints_retarget = (
                torch.from_numpy(bones_joints_retarget.reshape(-1, 3, njoints))
                .transpose(1, 2)
                .to("cuda")
            )
            bones_joints_retarget = torch.matmul(
                bones_joints_retarget, rot_mats.transpose(1, 2)
            )

        """ IK """
        phis = torch.tensor([1.0, 0.0], device="cuda").expand(
            smpl_thetas.shape[0], njoints, -1
        )
        if args.add_joints is not None:
            leaf_thetas = torch.eye(3, device="cuda").expand(
                smpl_thetas.shape[0], 8, -1, -1
            )
        else:
            leaf_thetas = torch.eye(3, device="cuda").expand(
                smpl_thetas.shape[0], 7, -1, -1
            )
        bones_rest_joints_seq = bones_rest_joints.repeat_interleave(
            smpl_thetas.shape[0], dim=0
        ).to("cuda")
        bones_rot_mats_ik, _, _ = batch_inverse_kinematics_transform_bones(
            bones_joints_retarget,
            None,
            phis,
            bones_rest_joints_seq,
            bones_children_map,
            bones_parents,
            leaf_thetas,
            False,
            add_eye=args.add_joints is not None,
        )
        bones_rot_mats_ik[
            :,
            [
                BONESPose.LeftHand,
                BONESPose.LeftHandThumb1,
                BONESPose.RightHand,
                BONESPose.RightHandThumb1,
            ],
            :,
        ] = torch.eye(3, device="cuda")
        bones_joints_retarget_ik, _ = batch_rigid_transform(
            bones_rot_mats_ik.to("cuda"), bones_rest_joints_seq, bones_parents
        )

        """ Compute metric given joints (no hand joints) """
        err_ik = np.linalg.norm(
            (bones_joints_retarget - bones_joints_retarget_ik).cpu().numpy(), axis=2
        ).mean()
        print(f"Retarget error IK = {err_ik}")
        errors_ik += [err_ik]

        """ Vis """
        smpl_motion = smpl(
            global_orient=smpl_thetas[:, :3],
            body_pose=smpl_thetas[:, 3:],
            betas=bones_beta,
            root_trans=smpl_trans,
            return_full_pose=True,
            orig_joints=True,
        )
        smpl_trans = smpl_trans @ base_rot_to_y_up

        print(f"Rendering video ...")
        bones_joints_retarget += smpl_trans.unsqueeze(1)
        bones_joints_retarget_ik += smpl_trans.unsqueeze(1)
        video_path = os.path.join(
            args.out_dir, f"video_regress_test_{args.dataset}", f"{i:06d}.mp4"
        )
        vis(
            visualizer,
            smpl_motion,
            bones_joints_retarget,
            bones_joints_retarget_ik,
            video_path,
            fps=30,
        )

    print(f"{args.regress_ver} produces average errors IK = {np.average(errors_ik)}")
