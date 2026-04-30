import os
import random
from copy import copy

import numpy as np
import pandas as pd
import torch
import wandb
from pycocotools.coco import COCO
from torch.utils import data
from torchvision.transforms import transforms

from motiondiff.data_pipeline.datasets.kp2d_dataset import KP2DDataset
from motiondiff.data_pipeline.humanml.utils.get_opt import get_opt
from motiondiff.data_pipeline.humanml.utils.word_vectorizer import WordVectorizer
from motiondiff.data_pipeline.tensors import collate
from motiondiff.models.common.utils.human_models import mano, smpl, smpl_x
from motiondiff.models.common.utils.preprocessing import transform_joint_to_other_db
from motiondiff.utils.tools import wandb_run_exists
from motiondiff.utils.torch_transform import (
    angle_axis_to_quaternion,
    quat_mul,
    quaternion_to_angle_axis,
    rotation_matrix_to_angle_axis,
)


# an adapter to our collate func
def hand_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [
        {
            "inp": torch.tensor(b[0].T)
            .float()
            .unsqueeze(1),  # [seqlen, J] -> [J, 1, seqlen]
            "target": 0,
            # 'text': b[0], #b[0]['caption']
            "text": "",
            "lengths": b[1],
        }
        for b in batch
    ]
    return collate(adapted_batch)


# def read_text_augment_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         lines = [x.strip() for x in lines]
#     return lines


# The last five joints are not from MANO forward, but from the tip vertex selection
JOINT_NAMES = [
    "wrist",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
    "thumb_tip",  # 16
    "index_tip",  # 17
    "middle_tip",  # 18
    "ring_tip",  # 19
    "pinky_tip",  # 20
]


class HandDataset(data.Dataset):
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
        use_natural_desc=False,
        use_short_desc=False,
        use_technical_desc=False,
        split_file_pattern="assets/bones/splits/v1/%s_index.npy",
        stats_folder="assets/bones/stats/v1",
        normalize_motion=True,
        skip_idx_file=None,
        name=None,  # for compatibility with other datasets
    ):
        self.meta = pd.read_csv(meta_file)
        self.split = split
        self.normalize_motion = normalize_motion
        # if split == 'all':
        #     self.index = np.arange(len(self.meta))
        # else:
        #     self.index = np.load(split_file_pattern % split)
        # # remove data indices with artifacts
        # self.skip_idx = set(np.load(skip_idx_file).tolist()) if skip_idx_file is not None else set()
        # self.index = np.array([i for i in self.index if i not in self.skip_idx])

        if self.normalize_motion:
            self.mean = np.load(os.path.join(stats_folder, "mean.npy"))
            self.std = np.load(os.path.join(stats_folder, "std.npy"))
        self.num_frames = num_frames
        self.motion_feature_dir = motion_feature_dir
        self.seqfiles = os.listdir(motion_feature_dir)

        self.text_augment_dir = text_augment_dir
        self.augment_text = augment_text
        self.aug_text_ind_range = aug_text_ind_range
        self.aug_text_prob = aug_text_prob
        self.use_natural_desc = use_natural_desc
        self.use_short_desc = use_short_desc
        self.use_technical_desc = use_technical_desc

        assert not (self.use_natural_desc or self.use_technical_desc), (
            "no description for hand dataset"
        )
        # NOTE: we don't need meta data for now
        # assert self.use_natural_desc or self.use_short_desc or self.use_technical_desc
        # self.motion_paths = self.meta['feature_path'].values
        # if self.use_natural_desc:
        #     self.natural_desc = [self.meta[f'natural_desc_{i}'].values for i in range(1, 4)]
        # if self.use_short_desc:
        #     self.short_desc = self.meta[f'short_description'].values
        # if self.use_technical_desc:
        #     self.technical_desc = self.meta[f'technical_description'].values
        # return

    def __len__(self):
        # return len(self.index)
        return len(self.seqfiles)

    def normalize(self, motion):
        return (motion - self.mean) / self.std

    def __getitem__(self, idx):
        # item = self.index[idx]
        motion_path = os.path.join(self.motion_feature_dir, self.seqfiles[idx])
        motion = np.load(motion_path)
        if self.normalize_motion:
            motion = self.normalize(motion)

        # text_list = []
        # if self.use_natural_desc:
        #     text_list += [(self.natural_desc[i][item], i) for i in range(3)]
        # if self.use_technical_desc:
        #     text_list.append((self.technical_desc[item], 3))
        # if self.use_short_desc:
        #     text_list.append((self.short_desc[item], 4))

        # text, text_sub_ind = text_list[np.random.choice(len(text_list))]
        # if type(text) not in [str, np.str_]:
        #     text = ''
        # if self.augment_text and text_sub_ind != 3 and text != '' and np.random.rand() < self.aug_text_prob: # don't have aug for technical descriptions
        #     try:
        #         text_augment_path = os.path.join(self.text_augment_dir, f'{item:06d}-{text_sub_ind}.txt')
        #         if os.path.exists(text_augment_path):
        #             aug_texts = read_text_augment_file(text_augment_path)
        #             if self.aug_text_ind_range is not None:
        #                 aug_texts = aug_texts[self.aug_text_ind_range[0]: min(self.aug_text_ind_range[1], len(aug_texts))]
        #             text = np.random.choice(aug_texts) # augmented files also include the original text annotation
        #             if text[-1] == '.' and np.random.rand() < 0.5:
        #                 # random drop of period at the end
        #                 text = text[:-1]
        #             if np.random.rand() < 0.5:
        #                 # randomly remove capitalization
        #                 text = text.lower()
        #     except Exception as e:
        #         print(f'Error in text augmentation: {e}')
        #         print(f'item: {item}, text_sub_ind: {text_sub_ind}, {text_augment_path}')
        #         if wandb_run_exists():
        #             wandb.alert(title=f"[{item}-{text_sub_ind}]", text=f"[{item}-{text_sub_ind}] {text_augment_path}\n" + str(e), level=wandb.AlertLevel.ERROR)
        #         # raise

        if self.num_frames == -1:  # no truncation or padding
            m_length = motion.shape[0]
            return motion, m_length

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

        return motion, m_length


class Hand3DV1(KP2DDataset):
    # def __init__(self, datapath='./dataset/humanml_opt.txt',
    #              num_keypoints=14, num_frames=196, mode='train', split="train",
    #              debug=False, rng=None, img_w=1024, img_h=1024, num_views=4, cam_radius=8, cam_elevation=10,
    #              focal_scale=2, synthetic_view_type='even', normalize_type='image_size', bbox_scale=1.4,
    #              use_coco_pelvis=False, tilt_prob=0.0, tilt_std=15, normalize_stats_dir=None, **kwargs):
    #     super().__init__(num_frames, img_w, img_h, num_views, cam_radius, cam_elevation, focal_scale, synthetic_view_type, normalize_type, bbox_scale, tilt_prob, tilt_std, normalize_stats_dir)
    #     self.mode = mode
    #     self.dataset_name = 't2m'
    #     self.dataname = 't2m'
    #     self.debug = debug
    #     self.feat_version = kwargs.get('feat_version', '1')
    #     self.major_version = self.feat_version.split('_')[0]
    #     self.get_coco_keypoints = num_keypoints == 25
    #     abs_base_path = '.'
    #     dataset_opt_path = os.path.join(abs_base_path, datapath)
    #     device = None
    #     opt = get_opt(dataset_opt_path, device)
    #     opt.data_root = './dataset/EgoSMPL3D'
    #     opt.motion_dir = os.path.join(opt.data_root, f'new_joint_vecs_v{self.feat_version}')
    #     opt.smpl_dir = os.path.join(opt.data_root, 'orig_smpl_params')
    #     opt.text_dir = os.path.join(opt.data_root, 'texts')
    #     opt.text_dir = os.path.join(abs_base_path, opt.text_dir)
    #     opt.model_dir = os.path.join(abs_base_path, opt.model_dir)
    #     opt.checkpoints_dir = os.path.join(abs_base_path, opt.checkpoints_dir)
    #     opt.data_root = os.path.join(abs_base_path, opt.data_root)
    #     opt.save_root = os.path.join(abs_base_path, opt.save_root)
    #     opt.meta_dir = './dataset'
    #     self.opt = opt
    #     self.mean = None
    #     self.std = None
    #     self.split = split
    #     self.split_file = os.path.join(opt.data_root, f'{split}.txt')
    #     self.num_keypoints = num_keypoints
    #     self.w_vectorizer = WordVectorizer(os.path.join(abs_base_path, 'glove'), 'our_vab')
    #     # self.t2m_dataset = HumanML3DSubDatasetV1(
    #     #     self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, self.debug, rng=rng, feat_version=self.feat_version, major_version=self.major_version
    #     # )
    #     self.num_actions = 1
    #     self.use_coco_pelvis = use_coco_pelvis
    #     self.wham_regressor = torch.tensor(np.load('data/smpl/J_regressor_wham.npy'))
    #     self.smpl_joints = smpl.all_joints_name
    #     self.coco_joints = ['Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
    #                         'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']
    #     self.smplx_valid_joints = [i for i, j in enumerate(smpl_x.joints_name[:num_keypoints]) if j in self.coco_joints or j in self.smpl_joints]
    #     self.coco_smplx_ind = [smpl_x.joints_name.index(j) for j in self.coco_joints]   # [24, 22, 23, 20, 21, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 6]
    #     breakpoint()
    #     assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, it is probably because your data dir has only texts and no motions. To train and evaluate MDM you should get the FULL data as described in the README file.'

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
        use_natural_desc=False,
        use_short_desc=False,
        use_technical_desc=False,
        split_file_pattern="assets/bones/splits/v1/%s_index.npy",
        stats_folder="assets/bones/stats/v1",
        normalize_motion=True,
        joint_parents_file="assets/mv2d/mano_joint_parents2.p",
        skip_idx_file=None,
        name=None,  # for compatibility with other datasets
        debug=False,
        rng=None,
        img_w=1024,
        img_h=1024,
        num_views=4,
        cam_radius=8,
        cam_elevation=10,
        focal_scale=2,
        synthetic_view_type="even",
        normalize_type="image_size",
        bbox_scale=1.4,
        use_coco_pelvis=False,
        tilt_prob=0.0,
        tilt_std=15,
        normalize_stats_dir=None,
        front_view_joints=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13],
        **kwargs,
    ):
        # self.meta = pd.read_csv(meta_file)
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
            tilt_prob,
            tilt_std,
            normalize_stats_dir,
            joint_parents_file,
            front_view_joints,
        )
        self.split = split
        # self.normalize_motion = normalize_motion
        # if split == 'all':
        #     self.index = np.arange(len(self.meta))
        # else:
        #     self.index = np.load(split_file_pattern % split)
        # # remove data indices with artifacts
        # self.skip_idx = set(np.load(skip_idx_file).tolist()) if skip_idx_file is not None else set()
        # self.index = np.array([i for i in self.index if i not in self.skip_idx])

        # if self.normalize_motion:
        #     self.mean = np.load(os.path.join(stats_folder, 'mean.npy'))
        #     self.std = np.load(os.path.join(stats_folder, 'std.npy'))
        self.num_frames = num_frames
        self.motion_feature_dir = motion_feature_dir
        self.seqfiles = os.listdir(motion_feature_dir)

        self.text_augment_dir = text_augment_dir
        self.augment_text = augment_text
        self.aug_text_ind_range = aug_text_ind_range
        self.aug_text_prob = aug_text_prob
        self.use_natural_desc = use_natural_desc
        self.use_short_desc = use_short_desc
        self.use_technical_desc = use_technical_desc

        self.mano_valid_joints = [i for i, j in enumerate(JOINT_NAMES)]

        print("calculating motion sequence lengths")
        self.total_length = 0
        for seq in self.seqfiles:
            motion_path = os.path.join(self.motion_feature_dir, seq)
            motion = np.load(motion_path)
            motion_length = motion.shape[0]
            if motion_length > self.num_frames:
                chip_num = motion_length - self.num_frames + 1
            else:
                chip_num = 1
            self.total_length += chip_num
        print("total length: ", self.total_length)

        assert not (self.use_natural_desc or self.use_technical_desc), (
            "no description for hand dataset"
        )
        # NOTE: we don't need meta data for now
        # assert self.use_natural_desc or self.use_short_desc or self.use_technical_desc
        # self.motion_paths = self.meta['feature_path'].values
        # if self.use_natural_desc:
        #     self.natural_desc = [self.meta[f'natural_desc_{i}'].values for i in range(1, 4)]
        # if self.use_short_desc:
        #     self.short_desc = self.meta[f'short_description'].values
        # if self.use_technical_desc:
        #     self.technical_desc = self.meta[f'technical_description'].values
        # return
        item = 1
        self.get_motion(item)

    #     breakpoint()

    def get_motion(self, item):
        item = item % len(
            self.seqfiles
        )  # TODO: change this for more uniformly sampling
        motion_path = os.path.join(self.motion_feature_dir, self.seqfiles[item])
        motion = np.load(motion_path)
        lhand_mano_params = motion[:, :51]
        rhand_mano_params = motion[:, 51:102]

        num_joints = 21
        lhand_all_joints = motion[:, 102 : 102 + 21 * 3].reshape(-1, num_joints, 3)
        rhand_all_joints = motion[:, 102 + 21 * 3 : 102 + 21 * 3 * 2].reshape(
            -1, num_joints, 3
        )

        if random.random() > 0.5:
            joints = lhand_all_joints
        else:
            joints = rhand_all_joints

        # seems not necessary
        joints = transform_joint_to_other_db(
            joints.transpose(1, 0, 2), JOINT_NAMES, JOINT_NAMES
        ).transpose(1, 0, 2)

        motion, cam_dict = self.generate_cam_and_2d_motion(joints)
        motion, m_length = self.pad_motion(motion)
        return "", motion, cam_dict, m_length

        # data = self.t2m_dataset.__getitem__(item)
        # text = data[2]
        # smpl_pose = data[7]
        # smpl_trans = data[8]
        # pose = rotation_matrix_to_angle_axis(smpl_pose).reshape(-1, 72)
        # shape = torch.zeros_like(pose[..., :10])
        # output = smpl.layer['neutral'](betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3], transl=smpl_trans)
        # smpl_joint_world = output.joints[:, :len(smpl.all_joints_name)].numpy()
        # smplx_joint_world = transform_joint_to_other_db(smpl_joint_world.transpose(1, 0, 2), smpl.all_joints_name, smpl_x.joints_name[:25]).transpose(1, 0, 2)
        # if self.get_coco_keypoints:
        #     wham_joints = torch.matmul(self.wham_regressor, output.vertices)
        #     coco_joints = wham_joints[:, :17]
        #     smplx_joint_world[:, self.coco_smplx_ind] = coco_joints

        # smplx_joint_world = smplx_joint_world[:, :self.num_keypoints]
        # if self.use_coco_pelvis:
        #     smplx_joint_world[:, 0] = (smplx_joint_world[:, 1] + smplx_joint_world[:, 2]) / 2
        # motion, cam_dict = self.generate_cam_and_2d_motion(smplx_joint_world)
        # motion, m_length = self.pad_motion(motion)
        # return text, motion, cam_dict, m_length

    def __getitem__(self, item):
        text, motion, cam_dict, m_length = self.get_motion(item)
        info = {
            "smpl_valid_joints": self.mano_valid_joints,
            "dataset_name": "humanml3d",
        }
        return text, motion, m_length, cam_dict, info

    # def __len__(self):
    #     return self.t2m_dataset.__len__()

    def __len__(self):
        # return len(self.index)
        return self.total_length

    # def normalize(self, motion):
    #     return (motion - self.mean) / self.std
