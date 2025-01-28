import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

sys.path.append("./")

from motiondiff.callbacks.whamlib.data.utils.augmentor import VideoAugmentor
from motiondiff.callbacks.whamlib.models import build_body_model
from motiondiff.callbacks.whamlib.utils import transforms
from motiondiff.data_pipeline.datasets.kp2d_dataset_v2 import KP2DDatasetV2
from motiondiff.data_pipeline.utils.augment import augment_betas
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


class ThreedpwSmplDataset(KP2DDatasetV2):
    def __init__(
        self,
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
        size_multiplier=1,
        hand_leg_aug=False,
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
            hand_leg_aug,
        )
        self.debug = debug
        self.get_coco_keypoints = num_keypoints == 25
        self.sample_beta = sample_beta
        self.mean = None
        self.std = None
        self.split = split
        self.num_keypoints = num_keypoints
        self.use_coco_pelvis = use_coco_pelvis
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
        self.video_augmentor = VideoAugmentor(
            coco_smplx_ind=self.coco_smplx_ind, num_frames=num_frames
        )
        self.base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        self.base_rot_mat = quaternion_to_rotation_matrix(
            torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        )
        self.smpl = build_body_model("cpu", num_frames)

        # Path
        self.hmr4d_support_dir = Path("dataset/GVHMR/3DPW/hmr4d_support")
        self.dataset_name = "3DPW"

        # Setting
        self.min_motion_frames = 60
        self.max_motion_frames = num_frames
        self._load_dataset()
        self._get_idx2meta()  # -> Set self.idx2meta
        self.total_len = len(self.idx2meta) * size_multiplier

    def _load_dataset(self):
        self.train_labels = torch.load(
            self.hmr4d_support_dir / "train_3dpw_gt_labels.pt"
        )
        self.refit_smplx = torch.load(self.hmr4d_support_dir / "train_refit_smplx.pt")
        if True:  # Remove clips that have obvious error
            update_list = {
                "courtyard_basketball_00_1": [(0, 300), (340, 468)],
                "courtyard_laceShoe_00_0": [(0, 620), (780, 931)],
                "courtyard_rangeOfMotions_00_1": [(0, 370), (410, 601)],
                "courtyard_shakeHands_00_1": [(0, 100), (120, 391)],
            }
            for k, v in update_list.items():
                self.refit_smplx[k]["valid_range_list"] = v

        self.f_img_folder = self.hmr4d_support_dir / "imgfeats/3dpw_train_smplx_refit"

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []
        for vid in self.refit_smplx:
            valid_range_list = self.refit_smplx[vid]["valid_range_list"]
            for start, end in valid_range_list:
                seq_length = end - start
                num_samples = max(seq_length // self.max_motion_frames, 1)
                seq_lengths.append(seq_length)
                self.idx2meta.extend([(vid, start, end)] * num_samples)
        minutes = sum(seq_lengths) / 25 / 60

    def _load_data(self, idx):
        data = {}
        vid, range1, range2 = self.idx2meta[idx]
        mlength = range2 - range1

        if mlength > self.num_frames:
            start = np.random.randint(range1, range2 - self.num_frames + 1)
            end = start + self.num_frames
        else:
            start = range1
            end = range2

        data["meta"] = {
            "data_name": self.dataset_name,
            "idx": idx,
            "vid": vid,
            "start_end": (start, end),
        }

        # Select motion subset
        # data["smplx_params_incam"] = {k: v[start:end] for k, v in self.refit_smplx[vid]["smplx_params_incam"].items()}
        data["smpl_params_global"] = {
            k: v[start:end] for k, v in self.train_labels[vid]["smpl_params"].items()
        }
        data["K_fullimg"] = self.train_labels[vid]["K_fullimg"]
        data["T_w2c"] = self.train_labels[vid]["T_w2c"][start:end]
        # Img (as feature):
        f_img_dict = torch.load(self.f_img_folder / f"{vid}.pt")
        data["bbx_xys"] = f_img_dict["bbx_xys"][start:end]  # (F, 3)
        data["f_imgseq"] = f_img_dict["features"][start:end].float()  # (F, 3)
        data["img_wh"] = f_img_dict["img_wh"]  # (2)
        data["kp2d"] = torch.zeros(
            (end - start), 17, 3
        )  # (L, 17, 3)  # do not provide kp2d

        return data

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

    def _process_data(self, data, idx):
        # SMPL params in world
        smpl_params_w = data["smpl_params_global"].copy()  # in az
        # World params
        pose = smpl_params_w["body_pose"]
        grot = smpl_params_w["global_orient"]
        grot_mat = angle_axis_to_rotation_matrix(grot)
        rmat = self.get_global_rot_augmentation(None)
        rmat = self.base_rot_mat @ rmat
        grot_world = rmat @ grot_mat
        grot_world = rotation_matrix_to_angle_axis(grot_world)
        pose = torch.cat([grot_world, pose], dim=-1)
        shape = smpl_params_w["betas"]
        shape = augment_betas(shape, std=0.1)
        target = {
            "rng": None,
            "pose": angle_axis_to_rotation_matrix(pose.view(-1, 24, 3)),
            "betas": shape,
            "res": torch.tensor([self.img_w, self.img_h]).float(),
            "f_imgseq": data["f_imgseq"].float(),
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
        target["pose"] = transforms.matrix_to_rotation_6d(target["pose"])

        self.get_input(target)
        self.pad_motion(target)

        text = ""
        motion = target["gt_kp2d"]
        cam_dict = target["cam_dict"]
        m_length = motion.shape[0]
        info = {"smpl_valid_joints": self.smplx_valid_joints, "dataset_name": "3dpw"}
        aux_data = {
            "kpt3d": target["kp3d"],
            "obs_kpt2d": target["obs_kp2d"],
            "raw_gt_kpt2d": target["raw_gt_kp2d"],
            "smpl_pose": target["pose"],
            "smpl_shape": target["betas"],
            "img_features": target["f_imgseq"],
        }
        return text, motion, m_length, cam_dict, aux_data, info

    def __getitem__(self, idx):
        idx = idx % len(self.idx2meta)
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data

    def __len__(self):
        return self.total_len


# 3DPW
if __name__ == "__main__":
    np.random.seed(0)
    dataset = ThreedpwSmplDataset(use_our_normalization=False)
    print(len(dataset))
    for i in range(10):
        a = dataset[i]
        print(i)
