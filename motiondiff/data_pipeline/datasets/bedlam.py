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


def mid2featname(mid):
    """featname = {scene}/{seqsubj}, Note that it ends with .pt (extra)"""
    # mid example: "inputs/bedlam/bedlam_download/20221011_1_250_batch01hand_closeup_suburb_a/mp4/seq_000001.mp4-rp_emma_posed_008"
    # -> featname: 20221011_1_250_batch01hand_closeup_suburb_a/seq_000001.mp4-rp_emma_posed_008.pt
    scene = mid.split("/")[-3]
    seqsubj = mid.split("/")[-1]
    featname = f"{scene}/{seqsubj}.pt"
    return featname


class BedlamDatasetV2(KP2DDatasetV2):
    """mid_to_valid_range and features are newly generated."""

    MIDINDEX_TO_LOAD = {
        "all60": ("mid_to_valid_range_all60.pt", "imgfeats/bedlam_all60"),
        "maxspan60": ("mid_to_valid_range_maxspan60.pt", "imgfeats/bedlam_maxspan60"),
    }

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
        mid_indices = ["all60", "maxspan60"]
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
        self.root = Path("dataset/GVHMR/BEDLAM/hmr4d_support")
        self.dataset_name = "BEDLAM"
        self.max_motion_frames = num_frames
        # speficify mid_index to handle
        if not isinstance(mid_indices, list):
            mid_indices = [mid_indices]
        self.mid_indices = mid_indices
        assert all([m in self.MIDINDEX_TO_LOAD for m in mid_indices])
        self._load_dataset()
        self._get_idx2meta()  # -> Set self.idx2meta
        self.total_len = len(self.idx2meta) * size_multiplier

    def _load_dataset(self):
        # Load mid to valid range
        self.mid_to_valid_range = {}
        self.mid_to_imgfeat_dir = {}
        for m in self.mid_indices:
            fn, feat_dir = self.MIDINDEX_TO_LOAD[m]
            mid_to_valid_range_ = torch.load(self.root / fn)
            self.mid_to_valid_range.update(mid_to_valid_range_)
            self.mid_to_imgfeat_dir.update(
                {mid: self.root / feat_dir for mid in mid_to_valid_range_}
            )

        # Load motionfiles
        self.motion_files = torch.load(self.root / "smplpose_v2.pth")

    def _get_idx2meta(self):
        # sum_frame = sum([e-s for s, e in self.mid_to_valid_range.values()])
        self.idx2meta = list(self.mid_to_valid_range.keys())

    def _load_data(self, idx):
        mid = self.idx2meta[idx]
        # neutral smplx : "pose": (F, 63), "trans": (F, 3), "beta": (10),
        #           and : "skeleton": (J, 3)
        data = self.motion_files[mid].copy()
        # Random select a subset
        range1, range2 = self.mid_to_valid_range[mid]  # [range1, range2)
        mlength = range2 - range1

        if mlength > self.num_frames:
            start = np.random.randint(range1, range2 - self.num_frames + 1)
            end = start + self.num_frames
        else:
            start = range1
            end = range2
        length = end - start
        data["start_end"] = (start, end)
        data["length"] = length

        # Update data to a subset
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and len(v.shape) > 1 and k != "skeleton":
                data[k] = v[start:end]

        # Load img(as feature) : {mid -> 'features', 'bbx_xys', 'img_wh', 'start_end'}
        imgfeat_dir = self.mid_to_imgfeat_dir[mid]
        f_img_dict = torch.load(imgfeat_dir / mid2featname(mid))

        # remap (start, end)
        start_mapped = start - f_img_dict["start_end"][0]
        end_mapped = end - f_img_dict["start_end"][0]

        data["f_imgseq"] = f_img_dict["features"][
            start_mapped:end_mapped
        ].float()  # (L, 1024)
        data["bbx_xys"] = f_img_dict["bbx_xys"][
            start_mapped:end_mapped
        ].float()  # (L, 4)
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
        rng_val = 0.25
        angle_y = rng_val * 2 * np.pi
        aa = torch.tensor([0.0, angle_y, 0.0]).float().unsqueeze(0)
        rmat = angle_axis_to_rotation_matrix(aa)
        return rmat

    def _process_data(self, data, idx):
        length = data["length"]
        # SMPL params in cam
        body_pose = data["pose"][:, 3:]  # (F, 63)
        betas = data["beta"].repeat(length, 1)  # (F, 10)
        # SMPL params in world
        global_orient_w = data["pose"][:, :3]  # (F, 3)
        transl_w = data["trans"]  # (F, 3)
        smpl_params_w = {
            "body_pose": body_pose,
            "betas": betas,
            "transl": transl_w,
            "global_orient": global_orient_w,
        }
        # World params
        pose = smpl_params_w["body_pose"]
        grot = smpl_params_w["global_orient"]
        grot_mat = angle_axis_to_rotation_matrix(grot)
        rmat = self.get_global_rot_augmentation(None)
        rmat = self.base_rot_mat @ rmat
        grot_world = rmat @ grot_mat
        grot_world = rotation_matrix_to_angle_axis(grot_world)
        pose = torch.cat([grot_world, pose, torch.zeros_like(pose[..., :6])], dim=-1)
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
        info = {"smpl_valid_joints": self.smplx_valid_joints, "dataset_name": "bedlam"}
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
    dataset = BedlamDatasetV2(use_our_normalization=False)
    print(len(dataset))
    for i in range(10):
        a = dataset[i]
        print(i)
