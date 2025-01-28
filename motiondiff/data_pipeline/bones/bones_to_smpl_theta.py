import argparse
import json
import logging
import os
import pdb
import random
import sys
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from torchmin import minimize
from tqdm import tqdm

sys.path.append("./")
from motiondiff.models.common.smpl import SMPL
from motiondiff.utils.hybrik import batch_rigid_transform

sys.path.append("../human_body_prior/src")
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model

"""Skeleton parameters"""
from motiondiff.data_pipeline.bones.skeleton_params import (
    BONESPose,
    SMPLPose,
    bones_beta,
    bones_joints_rest_ear,
    bones_joints_rest_eye,
    bones_joints_rest_jaw,
    bones_parents,
    smpl_ear_indices,
    smpl_eye_indices,
)

bones_parents = torch.LongTensor(bones_parents)
bones_beta = torch.FloatTensor([bones_beta]).to("cuda")
bones_joints_rest_jaw = torch.FloatTensor(bones_joints_rest_jaw)
bones_joints_rest_ear = torch.FloatTensor(bones_joints_rest_ear)
bones_joints_rest_eye = torch.FloatTensor(bones_joints_rest_eye)


"""Optimization constraints and parameters"""
bones_joint_idx = torch.LongTensor(
    [
        BONESPose.Neck,
        BONESPose.LeftArm,
        BONESPose.RightArm,
        BONESPose.LeftForeArm,
        BONESPose.RightForeArm,
        BONESPose.LeftHand,
        BONESPose.RightHand,
        BONESPose.LeftHandEnd,
        BONESPose.RightHandEnd,
        BONESPose.LeftLeg,
        BONESPose.RightLeg,
        BONESPose.LeftFoot,
        BONESPose.RightFoot,
        BONESPose.LeftToeBase,
        BONESPose.RightToeBase,
    ]
)
smpl_joint_idx = torch.LongTensor(
    [
        SMPLPose.Neck,
        SMPLPose.LShoulder,
        SMPLPose.RShoulder,
        SMPLPose.LElbow,
        SMPLPose.RElbow,
        SMPLPose.LWrist,
        SMPLPose.RWrist,
        SMPLPose.LHand,
        SMPLPose.RHand,
        SMPLPose.LKnee,
        SMPLPose.RKnee,
        SMPLPose.LAnkle,
        SMPLPose.RAnkle,
        SMPLPose.LToe,
        SMPLPose.RToe,
    ]
)
weights = torch.FloatTensor(
    [
        0.5,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
).to("cuda")


"""Optimization and help functions"""
smpl = SMPL("data/smpl_data", create_transl=False, gender="neutral").to("cuda")


def get_smpl_joints(theta, trans, beta):
    smpl_motion = smpl(
        global_orient=theta[:, :3],
        body_pose=theta[:, 3:],
        betas=beta,
        root_trans=trans,
        return_full_pose=True,
        orig_joints=True,
    )
    smpl_joints = smpl_motion.joints
    return smpl_joints


def get_smpl_motion(theta, trans, beta):
    smpl_motion = smpl(
        global_orient=theta[:, :3],
        body_pose=theta[:, 3:],
        betas=beta,
        root_trans=trans,
        return_full_pose=True,
        orig_joints=True,
    )
    return smpl_motion


def optimize_naive(bones_joints):
    def loss_naive(theta):
        smpl_joints = get_smpl_joints(theta, root_trans, bones_beta)[0]

        bones_joints_rel = (
            bones_joints[bones_joint_idx] - bones_joints[BONESPose.Pelvis]
        )
        smpl_joints_rel = smpl_joints[smpl_joint_idx] - smpl_joints[SMPLPose.Pelvis]
        dists = torch.norm(bones_joints_rel - smpl_joints_rel, dim=-1) * weights
        loss = dists.sum()

        return loss

    root_trans = torch.zeros((1, 3), dtype=torch.float32, device=torch.device("cuda:0"))
    smpl_thetas = []
    for i, bones_joints in enumerate(tqdm(bones_joints)):
        if i == 0:
            theta = torch.zeros(
                (1, 24 * 3), dtype=torch.float32, device=torch.device("cuda:0")
            )
        if i % 30 == 0:
            result = minimize(loss_naive, theta, method="bfgs")
        else:
            result = minimize(loss_naive, theta, method="bfgs", max_iter=100)
        theta = result.x.detach()
        logging.info(f"Step {i}: loss={result.fun}")
        smpl_thetas += [theta[0]]
        # if i > 30: break
    smpl_thetas = torch.stack(smpl_thetas)
    return smpl_thetas


def optimize_vposer(bones_joints):
    def loss_naive(theta):
        smpl_joints = get_smpl_joints(theta, root_trans, bones_beta)[0]

        bones_joints_rel = (
            bones_joints[bones_joint_idx] - bones_joints[BONESPose.Pelvis]
        )
        smpl_joints_rel = smpl_joints[smpl_joint_idx] - smpl_joints[SMPLPose.Pelvis]
        dists = torch.norm(bones_joints_rel - smpl_joints_rel, dim=-1) * weights
        loss = dists.sum()

        return loss

    def loss_vposer(latent):
        loss = 0

        pose_body = vp.decode(latent[:, 3:])["pose_body"].contiguous().view(1, 63)
        theta = torch.cat(
            [
                latent[:, :3],
                pose_body,
                torch.zeros((1, 6), dtype=torch.float32).to("cuda"),
            ],
            dim=1,
        )
        smpl_motion = get_smpl_motion(theta, root_trans, bones_beta)
        if args.add_joints is None:
            smpl_joints = smpl_motion.joints[0]
        elif args.add_joints == "ear":
            smpl_joints = torch.concat(
                [smpl_motion.joints[0], smpl_motion.vertices[0, smpl_ear_indices]],
                dim=0,
            )
        elif args.add_joints == "eye":
            smpl_joints = torch.concat(
                [smpl_motion.joints[0], smpl_motion.vertices[0, smpl_eye_indices]],
                dim=0,
            )

        bones_joints_rel = (
            bones_joints[bones_joint_idx] - bones_joints[BONESPose.Pelvis]
        )
        smpl_joints_rel = smpl_joints[smpl_joint_idx] - smpl_joints[SMPLPose.Pelvis]
        dists = torch.norm(bones_joints_rel - smpl_joints_rel, dim=-1) * weights
        loss += dists.sum()

        # ''' Minimize Torso bend '''
        # loss += torch.abs(theta[0, [SMPLPose.Torso*3, SMPLPose.Spine*3, SMPLPose.Chest*3]]).max() * 0.1

        # print(loss)
        return loss

    vp, ps = load_model(
        "data/vposer_v2_05",
        model_code=VPoser,
        remove_words_in_model_weights="vp_model.",
        disable_grad=True,
    )
    vp = vp.to("cuda")

    root_trans = torch.zeros((1, 3), dtype=torch.float32, device=torch.device("cuda:0"))
    smpl_thetas = []
    losses = []
    for i, bones_joints in enumerate(tqdm(bones_joints)):
        if i % 30 == 0:
            if i == 0:
                latent = torch.zeros(
                    (1, 3 + 32), dtype=torch.float32, device=torch.device("cuda:0")
                )
            else:
                latent[:, 3:] = 0
            result = minimize(loss_vposer, latent, method="bfgs")
        else:
            result = minimize(loss_vposer, latent, method="bfgs", max_iter=100)

        num_try = 0
        while result.fun > args.retarget_loss_thresh and num_try < args.max_try:
            logging.info(f"Trying {num_try} times ...")
            # latent = torch.zeros((1, 3 + 32), dtype=torch.float32, device=torch.device('cuda:0'))
            latent[:, 3:] = 0
            result = minimize(loss_vposer, latent, method="bfgs")
            num_try += 1

        # if result.fun > args.retarget_loss_thresh:
        #     logging.info(f"Job failed with loss={result.fun}")
        #     return None
        logging.info(f"Step {i}: loss={result.fun}")

        latent = result.x.detach()
        pose_body = vp.decode(latent[:, 3:])["pose_body"].contiguous().view(1, 63)

        # Refine leaf joints without VPoser
        # refine_joint_idx = [SMPLPose.LElbow, SMPLPose.RElbow, SMPLPose.LWrist, SMPLPose.RWrist, SMPLPose.LKnee, SMPLPose.RKnee, SMPLPose.LAnkle, SMPLPose.RAnkle]
        # result = minimize(loss_naive, theta, method='bfgs', max_iter=100)
        # theta = result.x.detach()
        # logging.info(f'Step {i} refine: loss={result.fun}')

        theta = torch.cat(
            [
                latent[:, :3],
                pose_body,
                torch.zeros((1, 6), dtype=torch.float32).to("cuda"),
            ],
            dim=1,
        )
        theta[:, SMPLPose.LWrist * 3 : (SMPLPose.LWrist + 1) * 3] = 0
        theta[:, SMPLPose.RWrist * 3 : (SMPLPose.RWrist + 1) * 3] = 0
        """With additional joints on face, no need to set head/neck rotations to 0"""
        # theta[:, SMPLPose.Neck*3 : (SMPLPose.Neck+1)*3] = 0
        # theta[:, SMPLPose.Head*3 : (SMPLPose.Head+1)*3] = 0
        smpl_thetas += [theta[0]]
        losses += [result.fun.item()]
        if args.debug_vis and i > 30:
            break
    smpl_thetas = torch.stack(smpl_thetas)
    losses = np.stack(losses)
    return smpl_thetas, losses


def vis(visualizer, bones_joints, smpl_thetas, video_path, fps=120):
    bones_joints = bones_joints[: smpl_thetas.shape[0]]
    bones_root_trans = bones_joints[:, 0].clone()

    base_rot = torch.FloatTensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).to("cuda")
    bones_joints = bones_joints @ base_rot

    smpl_motion = smpl(
        global_orient=smpl_thetas[:, :3],
        body_pose=smpl_thetas[:, 3:],
        betas=bones_beta,
        root_trans=bones_root_trans,
        return_full_pose=True,
        orig_joints=True,
    )
    smpl_joints = smpl_motion.joints @ base_rot
    smpl_verts = smpl_motion.vertices @ base_rot
    bones_joints_v2 = bones_joints.clone()
    smpl_verts_v2 = smpl_verts.clone()

    smpl_verts_v2[:, :, 0] += 0.8  # 1
    bones_joints_v2[:, :, 0] += 0.8  # 1
    # bones_joints[:, :, 0] += 0 # 2
    smpl_joints[:, :, 0] -= 0.8  # 3
    smpl_verts[:, :, 0] -= 1.6  # 4

    smpl_seq = {
        "gt": {
            "joints_pos": bones_joints.float(),
            "parents": bones_parents,
        },
        "smpl_joints": {
            "joints_pos": smpl_joints.float(),
            "parents": smpl.parents,
        },
        "smpl_mesh": {
            "smpl_verts": smpl_verts.float(),
            "show_smpl_mesh": True,
            "opacity": 1.0,
        },
        "gt_v2": {
            "joints_pos": bones_joints_v2.float(),
            "parents": bones_parents,
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
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets",
)
parser.add_argument("--data_dir", type=str, default="bones_full_raw_v14")
parser.add_argument("--joints_dir", type=str, default="bones_full_joints_v14")
parser.add_argument("--csv_file", type=str, default="metadata_240527_v014.csv")
parser.add_argument("--out_dir", type=str, default="bones_to_smpl/bones_to_smpl_v14.6")
parser.add_argument("--downsample_rate", type=int, default=4)
parser.add_argument("--retarget_loss_thresh", type=float, default=0.4)
parser.add_argument("--max_seq_length", type=int, default=600)
parser.add_argument("--max_try", type=int, default=1)
parser.add_argument("--add_joints", type=str, default="eye")
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--node_idx", type=int, default=0)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--vis", action="store_true", default=False)
parser.add_argument("--vis_only", action="store_true", default=False)
parser.add_argument("--debug_vis", action="store_true", default=False)
args = parser.parse_args()


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

        sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
        from userlib.auto_resume import AutoResume

        AutoResume.init()

    if args.vis_only:
        args.vis = True
    os.makedirs(args.out_dir, exist_ok=True)
    bones_rest_joints = torch.from_numpy(
        np.load(os.path.join(args.joints_dir, "neutral_pose.npy"))
    )
    if args.add_joints == "ear":
        bones_rest_joints = torch.cat(
            [bones_rest_joints, bones_joints_rest_ear.unsqueeze(0).cpu()], dim=1
        )
        bones_parents = torch.cat(
            [bones_parents, torch.LongTensor([BONESPose.Head, BONESPose.Head])], dim=-1
        )
        bones_joint_idx = torch.cat(
            [torch.LongTensor([BONESPose.LEar, BONESPose.REar])]
        )
        smpl_joint_idx = torch.cat(
            [smpl_joint_idx, torch.LongTensor([SMPLPose.LEar, SMPLPose.REar])]
        )
        weights = torch.cat([weights, torch.FloatTensor([0.5, 0.5]).to("cuda")])
    elif args.add_joints == "eye":
        bones_rest_joints = torch.cat(
            [bones_rest_joints, bones_joints_rest_eye.unsqueeze(0).cpu()], dim=1
        )
        bones_parents = torch.cat(
            [bones_parents, torch.LongTensor([BONESPose.Head, BONESPose.Head])], dim=-1
        )
        bones_joint_idx = torch.cat(
            [bones_joint_idx, torch.LongTensor([BONESPose.LEye, BONESPose.REye])]
        )
        smpl_joint_idx = torch.cat(
            [smpl_joint_idx, torch.LongTensor([SMPLPose.LEye, SMPLPose.REye])]
        )
        weights = torch.cat([weights, torch.FloatTensor([0.5, 0.5]).to("cuda")])

    # job_list_all = [path for path in csv.move_bvh_path]
    """Select bones sequences by unique content name """
    job_list_all = []
    content_set = set()
    for i in range(len(csv.move_bvh_path)):
        if csv.content_name[i] not in content_set:
            job_list_all += [csv.move_bvh_path[i]]
            content_set.add(csv.content_name[i])

    # job_list_todo = [path for path in job_list_all if 'angry_sword_walk_ff_loop_000_R_001__A378.' in path]
    job_list_node = job_list_all[args.node_idx : len(job_list_all) : args.num_nodes]
    if args.vis_only:
        job_list_todo = [
            path
            for path in job_list_node
            if os.path.exists(
                os.path.join(
                    args.out_dir, path.replace("BVH", "smpl").replace("bvh", "npz")
                )
            )
        ]
    else:
        job_list_todo = [
            path
            for path in job_list_node
            if not os.path.exists(
                os.path.join(
                    args.out_dir, path.replace("BVH", "smpl").replace("bvh", "npz")
                )
            )
        ]
    if not args.debug or args.vis_only:
        random.shuffle(job_list_todo)

    if args.vis:
        from motiondiff.data_pipeline.bones.vis_v2 import SMPLVisualizer

        visualizer = SMPLVisualizer(
            joint_parents=None, distance=7, elevation=2, smpl=smpl, verbose=False
        )
    if args.debug_vis:
        job_list_todo = job_list_todo[:1]
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(args.out_dir, f"node_{args.node_idx}_log.txt")
            ),
            logging.StreamHandler(),
        ],
    )

    logging.info(
        f"Node {args.node_idx} started working with {len(job_list_todo)} files..."
    )
    for idx, bvh_path in enumerate(job_list_todo):
        logging.info(
            f"Node {args.node_idx} started working with {idx} / {len(job_list_todo)} job: {bvh_path}"
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
        bones_rotmats = torch.from_numpy(
            np.load(rotmat_path)[:: args.downsample_rate]
        ).to("cuda")
        bones_joints = bones_joints[: args.max_seq_length]
        bones_rotmats = bones_rotmats[: args.max_seq_length]
        bones_joints_trans = bones_joints[:, [0]]

        if args.add_joints == "ear":
            bones_rotmats = torch.cat(
                [bones_rotmats, bones_rotmats[:, -2:]], dim=1
            )  # any rotations for ears
        elif args.add_joints == "eye":
            bones_rotmats = torch.cat(
                [bones_rotmats, bones_rotmats[:, -2:]], dim=1
            )  # any rotations for eyes

        bones_rest_joints_seq = bones_rest_joints.repeat_interleave(
            bones_rotmats.shape[0], dim=0
        ).to("cuda")
        bones_joints, global_rot_mat = batch_rigid_transform(
            bones_rotmats, bones_rest_joints_seq, bones_parents
        )
        bones_joints += bones_joints_trans

        out_smpl_path = os.path.join(
            args.out_dir, bvh_path.replace("BVH", "smpl").replace("bvh", "npz")
        )
        video_path = os.path.join(
            args.out_dir, "video", os.path.basename(bvh_path).replace("bvh", "mp4")
        )

        if not args.vis_only:
            # smpl_thetas = optimize_naive(bones_joints)
            smpl_thetas, losses = optimize_vposer(bones_joints)
            if smpl_thetas is None:
                continue
            smpl_thetas = gaussian_filter1d(smpl_thetas.cpu().numpy(), 1, axis=0)
            smpl_thetas = torch.from_numpy(smpl_thetas).to("cuda")
        else:
            smpl_res = np.load(out_smpl_path)
            smpl_thetas = torch.from_numpy(smpl_res["thetas"]).to("cuda")

        if not args.vis_only:
            os.makedirs(os.path.dirname(out_smpl_path), exist_ok=True)
            np.savez(
                out_smpl_path,
                thetas=smpl_thetas.cpu().numpy(),
                root_trans=bones_joints[:, 0].cpu().numpy(),
                losses=losses,
            )
        if args.vis:
            vis(
                visualizer,
                bones_joints,
                smpl_thetas,
                video_path,
                fps=120 / args.downsample_rate,
            )

        if not args.debug and AutoResume.termination_requested():
            details = {"node_idx": args.node_idx, "job_idx": idx}
            message = f"[Auto Resume] Terminateing with node {args.node_idx} on {idx} / {len(job_list_todo)}th job"
            logging.critical(message)
            AutoResume.request_resume(details, message=message)
            exit()
