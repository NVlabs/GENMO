import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
import glob
import sys
sys.path.append('./')
from torchvision.transforms import transforms
from pycocotools.coco import COCO
from motiondiff.models.common.utils.human_models import smpl_x, smpl
from motiondiff.utils.torch_utils import tensor_to, interp_scipy_ndarray, slerp_joint_rots
from motiondiff.data_pipeline.datasets.kp2d_dataset import KP2DDataset
from motiondiff.models.common.utils.preprocessing import transform_joint_to_other_db
from motiondiff.models.mv2d.mv2d_utils import draw_motion_2d, draw_mv_imgs


sequence_mapping = {
    'train': ['idea400']
}
    


class MotionX():
    def __init__(self, data_dir='dataset/motion-x', split='train', num_keypoints=14, num_frames=196, normalize_type='bbox_frame', bbox_scale=1.4, **kwargs):
        self.num_keypoints = num_keypoints
        self.num_frames = num_frames
        self.normalize_type = normalize_type
        self.bbox_scale = bbox_scale
        self.sequences = sequence_mapping[split]
        self.motion_list = []
        for seq in self.sequences:
            kp_dir = f'{data_dir}/motion/keypoints/{seq}'
            motion_files = glob.glob(f'{kp_dir}/*.json')
            motion_names = [osp.splitext(osp.basename(f))[0] for f in motion_files]
            for motion_name in motion_names:
                self.motion_list.append((seq, motion_name))
        self.joint_names = ['Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
                            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']
        self.joint_parents = [-1, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 11, 12, 13, 14]
        self.smplx_valid_joints = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        # self.smplx_joint_parents = torch.load('assets/mv2d/smplx_joint_parents.p')
        self.smplx_joint_parents = [-1, 0, 0, 1, 2, 3, 0, 0, 6, 7, 8, 9]

    def __len__(self):
        return len(self.motion_list)
    
    def pad_motion(self, motion, conf):
        if motion.shape[0] >= self.num_frames:
            m_length = self.num_frames
            idx = np.random.randint(0, motion.shape[0] - self.num_frames + 1)
            motion = motion[idx:idx + self.num_frames]
            conf_pad = conf[idx:idx + self.num_frames]
        else:
            m_length = motion.shape[0]
            motion = np.concatenate([motion, np.zeros((self.num_frames - motion.shape[0], motion.shape[1]))], axis=0)
            conf_pad = np.concatenate([conf, np.zeros((self.num_frames - conf.shape[0], conf.shape[1]))], axis=0)
        return motion, conf_pad, m_length

    def __getitem__(self, idx):
        seq, motion_name = self.motion_list[idx]
        text_file = f'dataset/motion-x/text/semantic_label/{seq}/{motion_name}.txt'
        text = open(text_file, 'r').read()
        motion_file = f'dataset/motion-x/motion/keypoints/{seq}/{motion_name}.json'
        motion_dict = json.load(open(motion_file, 'r'))
        body_kpts = []
        for frame in motion_dict['annotations']:
            body_kpts.append(np.array(frame['body_kpts']))
        body_kpts_coco = np.stack(body_kpts, axis=0)
        body_kpts_coco = interp_scipy_ndarray(body_kpts_coco, scale=2/3, dim=0)
        # body_kpts_coco_vis = body_kpts_coco[..., :2] - body_kpts_coco[:, :1, :2]
        # body_kpts_coco_vis += np.array([1920/2, 1920/2])
        # draw_motion_2d(torch.tensor(body_kpts_coco_vis[:, None]), 'out/vis/test_motionx.mp4', self.joint_parents, 1920, 1920, fps=30)
        body_kpts_smplx = transform_joint_to_other_db(body_kpts_coco.transpose(1, 0, 2), self.joint_names, smpl_x.joints_name).transpose(1, 0, 2)
        body_kpts_smplx = body_kpts_smplx[:, :self.num_keypoints]
        conf = body_kpts_smplx[:, :, 2]
        if self.normalize_type in {'bbox_frame', 'bbox_seq'}:
            front_view = torch.tensor(body_kpts_smplx[:, self.smplx_valid_joints, :2])     # only use first 14 joints for bbox size
            bbox_min = front_view.min(dim=1)[0]
            bbox_max = front_view.max(dim=1)[0]
            center = (front_view[:, 0] + front_view[:, 1]) * 0.5
            bbox_size = (bbox_max - bbox_min).max(dim=1)[0]
            if self.normalize_type == 'bbox_seq':
                normalize_size = bbox_size.mean() * self.bbox_scale
                motion_2d = (front_view - center[:, None]) / normalize_size * 2
            elif self.normalize_type == 'bbox_frame':
                normalize_size = bbox_size[:, None, None] * self.bbox_scale
                motion_2d = (front_view - center[:, None]) / normalize_size * 2
            
            # motion_2d_vis = (motion_2d + 1) * 0.5 * torch.tensor([1000, 1000]).float()
            # draw_motion_2d(motion_2d_vis[:, None], 'out/vis/test_motionx_bbox.mp4', self.smplx_joint_parents, 1000, 1000, fps=30)
            motion_2d_tmp = motion_2d.numpy()
            motion_2d = np.zeros((motion_2d_tmp.shape[0], self.num_keypoints, 2))
            motion_2d[:, self.smplx_valid_joints] = motion_2d_tmp
        else:
            raise ValueError
            
        motion_2d = motion_2d.reshape(motion_2d.shape[0], -1)
        motion_2d, conf_pad, m_length = self.pad_motion(motion_2d, conf)
        mask = np.zeros((self.num_frames,))
        mask[:m_length] = 1
        return {
            'text': text,
            'motion_2d': motion_2d.astype(np.float32),
            'motion_mask': mask.astype(np.float32),
            'conf': conf_pad.astype(np.float32),
            'lengths': m_length,
            'smpl_valid_joints': np.array(self.smplx_valid_joints),
            'is_2d': True,
            'dataset_name': 'motionx'
        }

if __name__ == '__main__':
    dataset = MotionX()
    for i in range(10):
        batch = dataset[i]
        print(f'data {i}')
