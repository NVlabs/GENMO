import os
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils import data

from motiondiff.data_pipeline.bones.motion_process import (
    recover_from_ric,
    recover_from_ric_with_joint_rot,
    recover_root_rot_pos,
)
from motiondiff.data_pipeline.tensors import collate
from motiondiff.models.mv2d.mv2d_utils import draw_motion_2d, draw_mv_imgs
from motiondiff.utils.geom import (
    batch_triangulate,
    batch_triangulate_torch,
    batch_triangulate_torch_single,
    lookat_correct,
    perspective_projection,
    spherical_to_cartesian,
)
from motiondiff.utils.tools import wandb_run_exists
from motiondiff.utils.vis import images_to_video


# an adapter to our collate func
def bones_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [
        {
            "inp": torch.tensor(b[1].T)
            .float()
            .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
            "target": 0,
            "text": b[0],  # b[0]['caption']
            "lengths": b[2],
            "cam_dict": b[3],
        }
        for b in batch
    ]
    return collate(adapted_batch)


def read_text_augment_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


class Bones2DDataset(data.Dataset):
    def __init__(
        self,
        split,
        num_frames,
        meta_file="/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full347_v1.0/meta_240416_v3.csv",
        # meta_file='out/meta.csv',   # TODO
        motion_feature_dir="/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full347_v1.0/new_joint_vecs",
        text_augment_dir=None,
        augment_text=False,
        aug_text_ind_range=None,
        aug_text_prob=1.0,
        use_natural_desc=True,
        use_short_desc=True,
        use_technical_desc=False,
        split_file_pattern="assets/bones/splits/v1/%s_index.npy",
        stats_folder="assets/bones/stats/v1",
        normalize_motion=True,
        skip_idx_file=None,
        img_w=1024,
        img_h=1024,
        num_views=1,
        cam_radius=4,
        cam_elevation=10,
        traj_center_xy_range=0,
        traj_center_z_range=0,
        focal_scale=1.0,
        kp3d_norm_type="per_frame_center",
        synthetic_view_type="even",
        name=None,  # for compatibility with other datasets
    ):
        self.meta = pd.read_csv(meta_file)
        self.split = split
        self.normalize_motion = normalize_motion
        if split == "all":
            self.index = np.arange(len(self.meta))
        else:
            self.index = np.load(split_file_pattern % split)
        # remove data indices with artifacts
        self.skip_idx = (
            set(np.load(skip_idx_file).tolist()) if skip_idx_file is not None else set()
        )
        self.index = np.array([i for i in self.index if i not in self.skip_idx])

        if self.normalize_motion:
            self.mean = np.load(os.path.join(stats_folder, "mean.npy"))
            self.std = np.load(os.path.join(stats_folder, "std.npy"))
        self.num_frames = num_frames
        self.motion_feature_dir = motion_feature_dir
        self.text_augment_dir = text_augment_dir
        self.augment_text = augment_text
        self.aug_text_ind_range = aug_text_ind_range
        self.aug_text_prob = aug_text_prob
        self.use_natural_desc = use_natural_desc
        self.use_short_desc = use_short_desc
        self.use_technical_desc = use_technical_desc
        assert self.use_natural_desc or self.use_short_desc or self.use_technical_desc
        self.motion_paths = self.meta["feature_path"].values
        if self.use_natural_desc:
            self.natural_desc = [
                self.meta[f"natural_desc_{i}"].values for i in range(1, 4)
            ]
        if self.use_short_desc:
            self.short_desc = self.meta[f"short_description"].values
        if self.use_technical_desc:
            self.technical_desc = self.meta[f"technical_description"].values

        focal_length = (img_w * img_w + img_h * img_h) ** 0.5
        self.cam_intrinsics = torch.eye(3).float()
        self.cam_intrinsics[0, 0] = focal_length * focal_scale
        self.cam_intrinsics[1, 1] = focal_length * focal_scale
        self.cam_intrinsics[0, 2] = img_w / 2.0
        self.cam_intrinsics[1, 2] = img_h / 2.0
        self.img_w = img_w
        self.img_h = img_h
        self.num_views = num_views
        self.cam_radius = cam_radius
        self.cam_elevation = cam_elevation
        self.kp3d_norm_type = kp3d_norm_type
        self.traj_center_xy_range = traj_center_xy_range
        self.traj_center_z_range = traj_center_z_range
        self.synthetic_view_type = synthetic_view_type

        self.neutral_joints = torch.load("assets/bones/skeleton/joints.p")
        self.joint_parents = torch.load("assets/bones/skeleton/parents.p")
        return

    def __len__(self):
        return len(self.index)

    def normalize(self, motion):
        return (motion - self.mean) / self.std

    def generate_eyes(self):
        if self.synthetic_view_type == "even":
            azimuths = (
                np.linspace(0, 360, self.num_views, endpoint=False)
                + np.random.rand() * 360 / self.num_views
            )
            elevations = np.ones(self.num_views) * self.cam_elevation
            radius = np.ones(self.num_views) * self.cam_radius
        else:
            raise NotImplementedError
        eyes = np.stack(
            [
                spherical_to_cartesian(r, azimuth, elevation)
                for azimuth, elevation, r in zip(azimuths, elevations, radius)
            ],
            axis=0,
        )
        return eyes

    def generate_cam_and_2d_motion(self, motion):
        cam_dict = []
        joints_pos, joints_rot = recover_from_ric_with_joint_rot(
            torch.tensor(motion),
            self.neutral_joints,
            self.joint_parents,
            return_joint_rot=True,
        )
        joints_pos = torch.stack(
            [-joints_pos[..., 0], joints_pos[..., 2], joints_pos[..., 1]], dim=-1
        )
        kpt3d = self.extract_kpt3d(joints_pos)
        kpt3d_pad = torch.cat((kpt3d, torch.ones_like(kpt3d[:, :, :1])), dim=-1)

        local_kpt2d_arr = []
        eyes = self.generate_eyes()
        for i in range(self.num_views):
            # r = 4  # Distance from the origin
            # azimuth = np.random.rand() * 360  # Azimuth angle in degrees
            # elevation = 10  # Elevation angle in degrees
            # eye = spherical_to_cartesian(r, azimuth, elevation)
            eye = eyes[i]
            at = np.array([0, 0, 0])  # Look at the origin
            up = np.array([0, 0, 1])  # Up direction

            c2w = torch.tensor(lookat_correct(eye, at, up)).float()
            w2c = torch.inverse(c2w)
            P = torch.matmul(self.cam_intrinsics, w2c[:3, :])
            local_kpt2d = (P @ kpt3d_pad.transpose(-1, -2)).transpose(-1, -2)
            local_kpt2d = local_kpt2d[..., :2] / local_kpt2d[..., 2:]
            # local_kpt3d = torch.matmul(w2c, kpt3d_pad.transpose(1, 2)).transpose(1, 2)[..., :3]
            # local_kpt2d = perspective_projection(local_kpt3d, self.cam_intrinsics)
            local_kpt2d[:, :, 1] = self.img_h - local_kpt2d[:, :, 1]
            cam_dict.append(
                {
                    "c2w": c2w,
                    "w2c": w2c,
                    "intrinsics": self.cam_intrinsics,
                    "P": P,
                }
            )
            local_kpt2d_arr.append(local_kpt2d)
        cam_dict = {k: torch.stack([x[k] for x in cam_dict]) for k in cam_dict[0]}
        local_kpt2d = torch.stack(local_kpt2d_arr, dim=1)
        # draw_motion_2d(local_kpt2d, 'out/vis/test_new.mp4', self.joint_parents, self.img_w, self.img_h)
        motion_2d = (
            local_kpt2d / torch.tensor([self.img_w, self.img_h]).float() - 0.5
        ) * 2
        motion_2d = motion_2d.reshape(motion_2d.shape[0], -1).numpy()
        # test recon
        # P_all = cam_dict['P']
        # local_kpt2d[..., 1] = self.img_h - local_kpt2d[..., 1]
        # kpt3d_recon = batch_triangulate(local_kpt2d[0].numpy(), P_all.numpy())
        # kpt3d_arr = []
        # for i in range(local_kpt2d.shape[0]):
        #     kpt3d = batch_triangulate_torch_single(local_kpt2d[i], P_all)
        #     kpt3d_arr.append(kpt3d)
        # kpt3d_recon_arr = torch.stack(kpt3d_arr, axis=0)
        # kpt3d_recon2 = batch_triangulate_torch(local_kpt2d, P_all.repeat(local_kpt2d.shape[0], 1, 1, 1))
        # diff = (kpt3d_recon_arr - kpt3d_recon2).norm()
        return motion_2d, cam_dict

    def extract_kpt3d(self, joints_pos):
        if self.kp3d_norm_type == "per_frame_center":
            kpt3d = joints_pos - joints_pos[:, :1]
        elif self.kp3d_norm_type == "xyz_avg":
            kpt3d = joints_pos - joints_pos[:, 0].mean(dim=0)

        if self.traj_center_xy_range > 0:
            theta = np.random.rand() * 2 * np.pi
            kpt3d[..., :2] += (
                self.traj_center_xy_range
                * np.random.rand(1)
                * np.array([np.cos(theta), np.sin(theta)])
            )
        if self.traj_center_z_range > 0:
            kpt3d[..., 2] += (np.random.rand() - 0.5) * self.traj_center_z_range
        return kpt3d

    def __getitem__(self, idx):
        item = self.index[idx]
        motion_path = os.path.join(self.motion_feature_dir, self.motion_paths[item])
        motion = np.load(motion_path)
        motion, cam_dict = self.generate_cam_and_2d_motion(motion)

        text_list = []
        if self.use_natural_desc:
            text_list += [(self.natural_desc[i][item], i) for i in range(3)]
        if self.use_technical_desc:
            text_list.append((self.technical_desc[item], 3))
        if self.use_short_desc:
            text_list.append((self.short_desc[item], 4))

        text, text_sub_ind = text_list[np.random.choice(len(text_list))]
        if type(text) not in [str, np.str_]:
            text = ""
        if self.augment_text and text != "" and np.random.rand() < self.aug_text_prob:
            try:
                text_augment_path = os.path.join(
                    self.text_augment_dir, f"{item:06d}-{text_sub_ind}.txt"
                )
                if os.path.exists(text_augment_path):
                    aug_texts = read_text_augment_file(text_augment_path)
                    if self.aug_text_ind_range is not None:
                        aug_texts = aug_texts[
                            self.aug_text_ind_range[0] : min(
                                self.aug_text_ind_range[1], len(aug_texts)
                            )
                        ]
                    text = np.random.choice(
                        aug_texts
                    )  # augmented files also include the original text annotation
                    if text[-1] == "." and np.random.rand() < 0.5:
                        # random drop of period at the end
                        text = text[:-1]
                    if np.random.rand() < 0.5:
                        # randomly remove capitalization
                        text = text.lower()
            except Exception as e:
                print(f"Error in text augmentation: {e}")
                print(
                    f"item: {item}, text_sub_ind: {text_sub_ind}, {text_augment_path}"
                )
                if wandb_run_exists():
                    wandb.alert(
                        title=f"[{item}-{text_sub_ind}]",
                        text=f"[{item}-{text_sub_ind}] {text_augment_path}\n" + str(e),
                        level=wandb.AlertLevel.ERROR,
                    )
                # raise

        if self.num_frames == -1:  # no truncation or padding
            m_length = motion.shape[0]
            return text, motion, m_length, cam_dict

        if motion.shape[0] >= self.num_frames:
            m_length = self.num_frames
            idx = np.random.randint(0, motion.shape[0] - self.num_frames + 1)
            motion = motion[idx : idx + self.num_frames]
        else:
            m_length = motion.shape[0]
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.num_frames - motion.shape[0], motion.shape[1])),
                ],
                axis=0,
            )

        return text, motion, m_length, cam_dict
