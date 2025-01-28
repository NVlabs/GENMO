import copy
import glob
import json
import os
import os.path as osp
import sys

import cv2
import joblib
import numpy as np
import torch

sys.path.append("./")
from pycocotools.coco import COCO
from torchvision.transforms import transforms

from motiondiff.data_pipeline.datasets.kp2d_dataset import KP2DDataset
from motiondiff.models.common.utils.human_models import (
    smpl,
    smpl_x,
    smplx_joint_parent_mapping,
)
from motiondiff.models.common.utils.preprocessing import transform_joint_to_other_db
from motiondiff.models.mv2d.mv2d_utils import draw_motion_2d, draw_mv_imgs
from motiondiff.utils.torch_utils import (
    interp_scipy_ndarray,
    slerp_joint_rots,
    tensor_to,
)

sequence_mapping = {
    "train": ["idea400"],
    "kungfu": ["kungfu"],
}

joint_parent_mapping = {
    "L_Hip": "Pelvis",
    "R_Hip": "Pelvis",
    "L_Knee": "L_Hip",
    "R_Knee": "R_Hip",
    "L_Ankle": "L_Knee",
    "R_Ankle": "R_Knee",
    "Neck": "Pelvis",
    "L_Shoulder": "Neck",
    "R_Shoulder": "Neck",
    "L_Elbow": "L_Shoulder",
    "R_Elbow": "R_Shoulder",
    "L_Wrist": "L_Elbow",
    "R_Wrist": "R_Elbow",
    "L_Big_toe": "L_Ankle",
    "L_Small_toe": "L_Ankle",
    "L_Heel": "L_Ankle",
    "R_Big_toe": "R_Ankle",
    "R_Small_toe": "R_Ankle",
    "R_Heel": "R_Ankle",
    "L_Ear": "Neck",
    "R_Ear": "Neck",
    "L_Eye": "Neck",
    "R_Eye": "Neck",
    "Nose": "Neck",
    "Head": "Neck",
}

coco_joint_parent_mapping = {
    "L_Hip": "Pelvis",
    "R_Hip": "Pelvis",
    "L_Knee": "L_Hip",
    "R_Knee": "R_Hip",
    "L_Ankle": "L_Knee",
    "R_Ankle": "R_Knee",
    "L_Shoulder": "Pelvis",
    "R_Shoulder": "Pelvis",
    "L_Elbow": "L_Shoulder",
    "R_Elbow": "R_Shoulder",
    "L_Wrist": "L_Elbow",
    "R_Wrist": "R_Elbow",
    "L_Ear": "Nose",
    "R_Ear": "Nose",
    "L_Eye": "Nose",
    "R_Eye": "Nose",
    "Nose": "Pelvis",
}


skip_indices = {
    "v1": [
        230,
        677,
        819,
        885,
        1671,
        2193,
        2745,
        2985,
        3004,
        3092,
        3267,
        3750,
        4095,
        4286,
        4381,
        4612,
        4694,
        4974,
        5072,
        5308,
        6171,
        6210,
        6392,
        6516,
        6570,
        7289,
        7314,
        7685,
        7795,
        7900,
        8054,
        9192,
        9232,
        10386,
        10601,
        10812,
        11247,
        11483,
        11677,
        11767,
    ],
    "kungfu_v1": [33, 84, 85, 205, 279, 288, 462, 494, 737],
}


class MotionX:
    def __init__(
        self,
        data_dir="dataset/motion-x",
        split="train",
        num_keypoints=25,
        num_frames=196,
        normalize_type="bbox_frame",
        bbox_scale=1.4,
        kp_conf_threshold=0.1,
        skip_ind_version=None,
        exclude_pelvis=False,
        normalize_stats_dir=None,
        **kwargs,
    ):
        self.num_keypoints = num_keypoints
        self.num_frames = num_frames
        self.normalize_type = normalize_type
        self.bbox_scale = bbox_scale
        self.sequences = sequence_mapping[split]
        self.motion_list = []
        self.kp_conf_threshold = kp_conf_threshold
        for seq in self.sequences:
            kp_dir = f"{data_dir}/pipeline/track_2d/{seq}"
            motion_files = glob.glob(f"{kp_dir}/*.pkl")
            motion_names = sorted(
                [osp.splitext(osp.basename(f))[0] for f in motion_files]
            )
            for motion_name in motion_names:
                self.motion_list.append((seq, motion_name))
        if skip_ind_version is not None:
            self.valid_indices = np.array(
                [
                    i
                    for i in range(len(self.motion_list))
                    if i not in skip_indices[skip_ind_version]
                ]
            )
        else:
            self.valid_indices = np.arange(len(self.motion_list))
        self.joint_names = [
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
            "Head",
            "Neck",
            "Pelvis",
            "L_Big_toe",
            "R_Big_toe",
            "L_Small_toe",
            "R_Small_toe",
            "L_Heel",
            "R_Heel",
        ]
        self.joint_parents = [
            self.joint_names.index(joint_parent_mapping[j])
            if j in joint_parent_mapping
            else -1
            for j in self.joint_names
        ]
        self.smplx_bbox_joints = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        self.smplx_valid_joints = [
            i for i, j in enumerate(smpl_x.joints_name) if j in self.joint_names
        ]
        if exclude_pelvis:
            self.smplx_valid_joints = self.smplx_valid_joints[1:]
        self.smplx_joint_parents = torch.load("assets/mv2d/smplx_joint_parents.p")
        if normalize_stats_dir is not None:
            self.motion_mean = np.load(f"{normalize_stats_dir}/mean.npy")
            self.motion_std = np.load(f"{normalize_stats_dir}/std.npy")
        else:
            self.motion_mean = None
            self.motion_std = None
        self.normalize_motion = self.motion_mean is not None

    def __len__(self):
        return 100000  # len(self.valid_indices)

    def pad_motion(self, motion, conf):
        if motion.shape[0] >= self.num_frames:
            m_length = self.num_frames
            idx = np.random.randint(0, motion.shape[0] - self.num_frames + 1)
            motion = motion[idx : idx + self.num_frames]
            conf_pad = conf[idx : idx + self.num_frames]
        else:
            m_length = motion.shape[0]
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.num_frames - motion.shape[0], motion.shape[1])),
                ],
                axis=0,
            )
            conf_pad = np.concatenate(
                [conf, np.zeros((self.num_frames - conf.shape[0], conf.shape[1]))],
                axis=0,
            )
        return motion, conf_pad, m_length

    def __getitem__(self, i):
        idx = self.valid_indices[i % len(self.valid_indices)]
        seq, motion_name = self.motion_list[idx]
        text_file = f"dataset/motion-x/text/semantic_label/{seq}/{motion_name}.txt"
        motion_file = f"dataset/motion-x/pipeline/track_2d/{seq}/{motion_name}.pkl"

        try:
            text = open(text_file, "r").read()
            motion_dict_new = joblib.load(motion_file)
            body_kpts_coco = motion_dict_new["tracks"][0]["keyp"]
            body_kpts_coco = interp_scipy_ndarray(body_kpts_coco, scale=2 / 3, dim=0)
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            print(f"motion_file: {motion_file}  text_file: {text_file}")
            return self.__getitem__((idx + 1) % len(self))

        # body_kpts_coco_vis = body_kpts_coco[..., :2] - body_kpts_coco[:, :1, :2]
        # body_kpts_coco_vis += np.array([1920/2, 1920/2])
        # draw_motion_2d(torch.tensor(body_kpts_coco_vis[:, None]), 'out/vis/test_motionx_new.mp4', self.joint_parents, 1920, 1920, fps=30)
        body_kpts_smplx = transform_joint_to_other_db(
            body_kpts_coco.transpose(1, 0, 2), self.joint_names, smpl_x.joints_name
        ).transpose(1, 0, 2)
        body_kpts_smplx = body_kpts_smplx[:, : self.num_keypoints]
        motion_2d = np.zeros((body_kpts_smplx.shape[0], self.num_keypoints, 2))
        conf = body_kpts_smplx[:, :, 2]
        if self.normalize_type in {"bbox_frame", "bbox_seq"}:
            bbox_size_arr = []
            center_arr = []
            for t in range(motion_2d.shape[0]):
                front_view = torch.tensor(
                    body_kpts_smplx[t, self.smplx_bbox_joints, :2]
                )  # only use first 14 joints for bbox size
                visible = conf[t, self.smplx_bbox_joints] > self.kp_conf_threshold
                front_view = front_view[visible]
                bbox_min = front_view.min(dim=0)[0]
                bbox_max = front_view.max(dim=0)[0]
                if visible[0] and visible[1]:
                    center = (front_view[0] + front_view[1]) * 0.5
                elif visible[0]:
                    center = front_view[0]
                elif visible[1]:
                    center = front_view[1]
                else:
                    center = front_view.mean(dim=0)
                body_kpts_smplx[t, 0, :2] = center
                bbox_size = (bbox_max - bbox_min).max()
                center_arr.append(center)
                bbox_size_arr.append(bbox_size)
            center = torch.stack(center_arr)
            bbox_size = torch.stack(bbox_size_arr)

            motion_2d = torch.tensor(body_kpts_smplx[:, self.smplx_valid_joints, :2])
            if self.normalize_type == "bbox_seq":
                normalize_size = bbox_size.mean() * self.bbox_scale
                motion_2d = (motion_2d - center[:, None]) / normalize_size * 2
            elif self.normalize_type == "bbox_frame":
                normalize_size = bbox_size[:, None, None] * self.bbox_scale
                motion_2d = (motion_2d - center[:, None]) / normalize_size * 2

            # vis
            # motion_2d_vis = (motion_2d + 1) * 0.5 * torch.tensor([1000, 1000]).float()
            # pelvis = (motion_2d_vis[:, [0]] + motion_2d_vis[:, [1]]) * 0.5
            # motion_2d_vis = torch.cat([pelvis, motion_2d_vis], dim=1)
            # draw_motion_2d(motion_2d_vis[:, None], 'out/vis/test_motionx_bbox_new.mp4', self.smplx_joint_parents, 1000, 1000, fps=30)

            motion_2d_tmp = motion_2d.numpy()
            motion_2d = np.zeros((motion_2d_tmp.shape[0], self.num_keypoints, 2))
            motion_2d[:, self.smplx_valid_joints] = motion_2d_tmp
        else:
            raise ValueError

        motion_2d = motion_2d.reshape(motion_2d.shape[0], -1)
        if self.normalize_motion:
            motion_2d = (motion_2d - self.motion_mean) / self.motion_std
        motion_2d, conf_pad, m_length = self.pad_motion(motion_2d, conf)
        mask = np.zeros((self.num_frames,))
        mask[:m_length] = 1
        return {
            "text": text,
            "motion_2d": motion_2d.astype(np.float32),
            "motion_mask": mask.astype(np.float32),
            "conf": conf_pad.astype(np.float32),
            "lengths": m_length,
            "smpl_valid_joints": np.array(self.smplx_valid_joints),
            "is_2d": True,
            "dataset_name": "motionx",
        }


if __name__ == "__main__":
    # dataset = MotionX(split='kungfu', skip_ind_version='kungfu_v1')
    # dataset = MotionX(skip_ind_version='v1')
    dataset = MotionX(
        skip_ind_version="v1",
        normalize_stats_dir="assets/mv2d/stats/gen2d_mv_test_mask_fc_st_2d_fix_w3d_25kp",
    )
    # data = dataset[326]
    # for i in range(10):
    #     batch = dataset[i]
    #     print(f'data {i}')
    skip_indices = []
    print(len(dataset))
    for i, batch in enumerate(dataset):
        if type(batch) == int:
            print(f"Error loading data at index {batch}")
            skip_indices.append(batch)
        else:
            print(f"data {batch['lengths']}")
        pass
    print(skip_indices)
