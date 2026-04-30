import copy
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch

sys.path.append("./")
import codecs as cs
import pickle
import random
from os.path import join as pjoin

import pandas as pd
from pycocotools.coco import COCO
from torchvision.transforms import transforms
from tqdm import tqdm

from motiondiff.callbacks.whamlib.models import build_body_model
from motiondiff.data_pipeline.datasets.kp2d_dataset_v2 import KP2DDatasetV2
from motiondiff.data_pipeline.humanml.utils.get_opt import get_opt
from motiondiff.data_pipeline.humanml.utils.word_vectorizer import WordVectorizer
from motiondiff.models.common.utils.human_models import smpl, smpl_x
from motiondiff.models.common.utils.preprocessing import transform_joint_to_other_db
from motiondiff.utils.torch_transform import (
    angle_axis_to_quaternion,
    angle_axis_to_rotation_matrix,
    quat_mul,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_angle_axis,
)


class Bones2DDatasetV2(KP2DDatasetV2):
    def __init__(
        self,
        datapath="/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_to_smpl/bones_to_smpl_v14.7/smpl",
        meta_file="/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw_v14/metadata_240527_v014.csv",
        split_folder="assets/mv2d/splits/v1",
        num_keypoints=14,
        num_frames=81,
        split="train",
        debug=False,
        rng=None,
        img_w=1024,
        img_h=1024,
        num_views=4,
        cam_radius=8,
        cam_elevation=0,
        focal_scale=2,
        synthetic_view_type="even",
        normalize_type="image_size",
        bbox_scale=1.4,
        use_coco_pelvis=False,
        normalize_stats_dir=None,
        sample_beta=False,
        cam_aug_cfg={},
        use_our_normalization=False,
        always_start_from_first_frame=False,
        **kwargs,
    ):
        super().__init__(
            num_frames,
            img_w,
            img_h,
            num_views,
            cam_radius,
            cam_elevation,
            focal_scale,
            synthetic_view_type,
            normalize_type,
            bbox_scale,
            normalize_stats_dir,
            cam_aug_cfg,
            use_our_normalization,
        )
        self.datapath = datapath
        meta = pd.read_csv(meta_file)
        bvh_files = meta["move_bvh_path"].values
        self.motion_names = [x[4:].replace(".bvh", "") for x in bvh_files]
        self.all_index = np.load(pjoin(split_folder, f"filtered_smpl_ind.npy"))
        self.split_index = np.load(pjoin(split_folder, f"{split}_index.npy"))
        self.rng_dict = pickle.load(open(pjoin(split_folder, f"rng_dict.pkl"), "rb"))
        rng_mapping = {x: {} for x in self.all_index}
        for k, rng_arr in self.rng_dict.items():
            for i, val in enumerate(rng_arr):
                rng_mapping[self.all_index[i]][k] = val
        self.rng_mapping = rng_mapping

        self.debug = debug
        self.get_coco_keypoints = num_keypoints == 25
        self.sample_beta = sample_beta
        self.mean = None
        self.std = None
        self.split = split
        self.num_keypoints = num_keypoints
        self.use_coco_pelvis = use_coco_pelvis
        self.always_start_from_first_frame = always_start_from_first_frame
        self.data_files = sorted(
            [f for f in os.listdir(datapath) if f.endswith(".npz")]
        )
        self.wham_regressor = torch.tensor(np.load("data/smpl/J_regressor_wham.npy"))
        self.smpl_joints = smpl.all_joints_name
        self.coco_joints = [
            "Nose",
            "L_Eye",
            "R_Eye",
            "L_Ear",
            "R_Ear",
            "L_Shoulder",
            "R_Shoulder",
            "L_Elbow",
            "R_Elbow",
            "L_Wrist",
            "R_Wrist",
            "L_Hip",
            "R_Hip",
            "L_Knee",
            "R_Knee",
            "L_Ankle",
            "R_Ankle",
        ]
        self.smplx_valid_joints = [
            i
            for i, j in enumerate(smpl_x.joints_name[:num_keypoints])
            if j in self.coco_joints or j in self.smpl_joints
        ]
        self.coco_smplx_ind = [
            smpl_x.joints_name.index(j) for j in self.coco_joints
        ]  # [24, 22, 23, 20, 21, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 6]
        self.base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        self.base_rot_mat = quaternion_to_rotation_matrix(
            torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        )
        self.smpl = build_body_model("cpu", num_frames)

    def get_global_rot_augmentation(self, rng):
        """Global coordinate augmentation. Random rotation around y-axis"""
        if rng is not None and "angle_y" in rng:
            rng_val = rng["angle_y"]
        else:
            rng_val = torch.rand(1)
        angle_y = rng_val * 2 * np.pi
        aa = torch.tensor([0.0, angle_y, 0.0]).float().unsqueeze(0)
        rmat = angle_axis_to_rotation_matrix(aa)
        return rmat

    def get_motion(self, item):
        name = self.motion_names[self.split_index[item]]
        rng = self.rng_mapping[self.split_index[item]]
        fname = osp.join(self.datapath, f"{name}.npz")
        data = np.load(osp.join(self.datapath, fname))
        pose = torch.tensor(data["thetas"])
        if pose.shape[0] > self.num_frames:
            if self.always_start_from_first_frame:
                idx = 0
            else:
                idx = np.random.randint(0, pose.shape[0] - self.num_frames + 1)
            pose = pose[idx : idx + self.num_frames]
        pose = angle_axis_to_rotation_matrix(pose.view(-1, 24, 3))
        rmat = self.get_global_rot_augmentation(rng)
        rmat = self.base_rot_mat @ rmat
        pose[:, 0] = rmat @ pose[:, 0]
        if self.sample_beta:
            if rng is not None and "shape" in rng:
                shape = torch.tensor(rng["shape"]).float()
            else:
                shape = torch.randn(10)
            shape = shape.unsqueeze(0).repeat(pose.shape[0], 1)
        else:
            shape = torch.zeros((pose.shape[0], 10))
        target = {
            "rng": rng,
            "pose": pose,
            "betas": shape,
            "res": torch.tensor([self.img_w, self.img_h]).float(),
        }

        output = self.smpl.get_output(
            body_pose=target["pose"][:, 1:],
            global_orient=target["pose"][:, :1],
            betas=target["betas"],
            pose2rot=False,
        )

        smpl_joint_world = output.orig_joints[:, : len(smpl.all_joints_name)].numpy()
        smplx_joint_world = transform_joint_to_other_db(
            smpl_joint_world.transpose(1, 0, 2),
            smpl.all_joints_name,
            smpl_x.joints_name[:25],
        ).transpose(1, 0, 2)
        coco_joints = output.joints[:, :17]
        smplx_joint_world[:, self.coco_smplx_ind] = coco_joints
        smplx_joint_world[:, 0] = (
            smplx_joint_world[:, 1] + smplx_joint_world[:, 2]
        ) / 2
        smplx_joint_world -= smplx_joint_world[:, :1]
        target["kp3d"] = torch.tensor(smplx_joint_world).float()

        self.get_input(target)
        self.pad_motion(target)

        text = ""
        motion = target["gt_kp2d"]
        cam_dict = target["cam_dict"]
        m_length = motion.shape[0]
        info = {
            "smpl_valid_joints": self.smplx_valid_joints,
            "dataset_name": "bones2d_v2",
        }
        aux_data = {
            "kpt3d": target["kp3d"],
            "obs_kpt2d": target["obs_kp2d"],
            "raw_gt_kpt2d": target["raw_gt_kp2d"],
            "smpl_pose": target["pose"],
            "smpl_shape": target["betas"],
        }
        return text, motion, m_length, cam_dict, aux_data, info

    def __getitem__(self, item):
        return self.get_motion(item)

    def __len__(self):
        return len(self.split_index)


class Bones2DDatasetV2SingleView(Bones2DDatasetV2):
    def __init__(self, num_views=1, **kwargs):
        super().__init__(num_views=num_views, **kwargs)

    def __getitem__(self, item):
        text, motion, m_length, cam_dict, aux_data, info = self.get_motion(item)
        motion_2d = motion
        mask = np.zeros((self.num_frames,))
        mask[:m_length] = 1
        conf = mask[:, None].repeat(self.num_keypoints, axis=-1)
        return {
            "text": text,
            "motion_2d": motion_2d.astype(np.float32),
            "obs_kpt2d": aux_data["obs_kpt2d"].astype(np.float32),
            "motion_mask": mask.astype(np.float32),
            "vit_feats": np.zeros((self.num_frames, 1024), dtype=np.float32),
            "conf": conf,
            "lengths": m_length,
            "smpl_valid_joints": np.array(self.smplx_valid_joints),
            "is_2d": True,
            "dataset_name": "bones2d_singleview_v2",
        }


if __name__ == "__main__":
    cam_aug_cfg = {
        "elevation_mean": 5,
        "elevation_std": 22.5,
        "radius_min": 2,
        "radius_max": 16,
    }
    dataset = Bones2DDatasetV2(
        debug=True,
        normalize_type="bbox_frame",
        num_keypoints=25,
        use_coco_pelvis=True,
        use_our_normalization=False,
        cam_aug_cfg=cam_aug_cfg,
    )
    # dataset = Bones2DDatasetV2SingleView(debug=True, normalize_type='bbox_frame', num_keypoints=25, use_coco_pelvis=True, use_our_normalization=False, sample_beta=True)
    for i in range(len(dataset)):
        print(dataset[i])
        # break

    # import os, sys
    # sys.path.append(os.path.join(os.getcwd()))
    # from motiondiff.utils.config import create_config
    # from omegaconf import OmegaConf

    # conf = create_config('gen2d_mv_test_mask_fc_st_2d_fix_w3d_25kp_kungfu_norm')
    # dataset_cfg = conf.train_dataset.copy()
    # add_cfg = dataset_cfg.pop('dataset_kwargs')
    # dataset_cfg.update(add_cfg.get('humanml3d', {}))
    # dataset = Bones2DDatasetV2SingleView(**dataset_cfg, debug=True)
