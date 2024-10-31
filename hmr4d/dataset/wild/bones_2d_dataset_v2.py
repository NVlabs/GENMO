import os
import os.path as osp
import numpy as np
import torch
import pickle
import cv2
import json
import copy
import sys
sys.path.append('./')
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import pandas as pd
import pickle


from torchvision.transforms import transforms
from motiondiff.utils.torch_transform import rotation_matrix_to_angle_axis, quaternion_to_angle_axis, quat_mul, angle_axis_to_quaternion, angle_axis_to_rotation_matrix, quaternion_to_rotation_matrix

from hmr4d.dataset.wild.kp2d_dataset_v2 import KP2DDatasetV2
from hmr4d.utils.smplx_utils import make_smplx


class Bones2DDatasetV2(KP2DDatasetV2):
    def __init__(self, datapath='/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_to_smpl/bones_to_smpl_v14.7/smpl',
                 meta_file='/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw_v14/metadata_240527_v014.csv',
                 split_folder='inputs/mv2d/splits/v1', num_keypoints=17, num_frames=120, num_data=None, split="train", shuffle_data_seed=None, debug=False, rng=None, img_w=1024, img_h=1024, num_views=4,
                 cam_radius=8, cam_elevation=0, focal_scale=2, synthetic_view_type='even', normalize_type='image_size', bbox_scale=1.4, use_coco_pelvis=False, 
                 normalize_stats_dir=None, sample_beta=True, cam_aug_cfg={}, use_our_normalization=False, always_start_from_first_frame=False, use_orig_length=False, hand_leg_aug=False,
                 precompute_data_folder='/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/GVHMR/bones2d/v1', device='cpu', **kwargs):
        super().__init__(num_frames, img_w, img_h, num_views, cam_radius, cam_elevation, focal_scale, synthetic_view_type, normalize_type, bbox_scale, normalize_stats_dir, cam_aug_cfg, use_our_normalization, hand_leg_aug, use_orig_length)
        self.datapath = datapath
        meta = pd.read_csv(meta_file)
        bvh_files = meta['move_bvh_path'].values
        self.motion_names = [x[4:].replace('.bvh', '') for x in bvh_files]
        self.all_index = np.load(pjoin(split_folder, f'filtered_smpl_ind.npy'))
        self.split_index = np.load(pjoin(split_folder, f'{split}_index.npy'))
        if shuffle_data_seed is not None:
            rng_state = np.random.RandomState(shuffle_data_seed)
            self.split_index = rng_state.permutation(self.split_index)
        if num_data is not None:
            self.split_index = self.split_index[:num_data]
        
        self.rng_dict = pickle.load(open(pjoin(split_folder, f'rng_dict.pkl'), 'rb'))
        rng_mapping = {x: {} for x in self.all_index}
        for k, rng_arr in self.rng_dict.items():
            for i, val in enumerate(rng_arr):
                rng_mapping[self.all_index[i]][k] = val
        self.rng_mapping = rng_mapping
        
        self.debug = debug
        self.get_coco_keypoints = True
        self.sample_beta = sample_beta
        self.mean = None
        self.std = None
        self.split = split
        self.num_keypoints = num_keypoints
        self.use_coco_pelvis = use_coco_pelvis
        self.always_start_from_first_frame = always_start_from_first_frame
        self.data_files = sorted([f for f in os.listdir(datapath) if f.endswith('.npz')])
        self.coco17_regressor = torch.load("hmr4d/utils/body_model/smpl_coco17_J_regressor.pt").to(device)
        self.coco_joints = ['Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
                            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']
        self.base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        self.base_rot_mat = quaternion_to_rotation_matrix(torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
        self.smpl = make_smplx('smpl').to(device)
        self.precompute_data_folder = precompute_data_folder
        self.precompute = self.precompute_data_folder is not None
        self.device = device
        
    def get_global_rot_augmentation(self, rng):
        """Global coordinate augmentation. Random rotation around y-axis"""
        if rng is not None and 'angle_y' in rng:
            rng_val = rng['angle_y']
        else:
            rng_val = torch.rand(1)
        angle_y = rng_val * 2 * np.pi
        aa = torch.tensor([0.0, angle_y, 0.0]).float().unsqueeze(0)
        rmat = angle_axis_to_rotation_matrix(aa)
        return rmat

    def get_motion(self, item):
        name = self.motion_names[self.split_index[item]]
        rng = self.rng_mapping[self.split_index[item]]
        fname = osp.join(self.datapath, f'{name}.npz')
        data = np.load(osp.join(self.datapath, fname))
        pose = torch.tensor(data['thetas'])
        if not self.use_orig_length and pose.shape[0] > self.num_frames:
            if self.always_start_from_first_frame:
                idx = 0
            else:
                idx = np.random.randint(0, pose.shape[0] - self.num_frames + 1)
            pose = pose[idx:idx + self.num_frames]
        pose = angle_axis_to_rotation_matrix(pose.view(-1, 24, 3))
        rmat = self.get_global_rot_augmentation(rng)
        rmat = self.base_rot_mat @ rmat
        pose[:, 0] = rmat @ pose[:, 0]
        if self.sample_beta:
            if rng is not None and 'shape' in rng:
                shape = torch.tensor(rng['shape']).float()
            else:
                shape = torch.randn(10)
            shape = shape.unsqueeze(0).repeat(pose.shape[0], 1)
        else:
            shape = torch.zeros((pose.shape[0], 10))
        target = {
            'rng': rng,
            'pose': pose,
            'betas': shape,
            'res': torch.tensor([self.img_w, self.img_h]).float(),
            'm_length': pose.shape[0],
        }
        
        output = self.smpl(
            body_pose=target['pose'][:, 1:].to(self.device),
            global_orient=target['pose'][:, :1].to(self.device),
            betas=target['betas'].to(self.device),
            pose2rot=False)
        
        coco_joints = self.coco17_regressor @ output.vertices
        target['kp3d'] = coco_joints
        
        self.get_input(target)
        self.pad_motion(target)
        
        # text = ''
        # motion = target['obs_kp2d']
        # cam_dict = target['cam_dict']
        # info = {
        #     'dataset_name': 'bones2d_v2'
        # }
        # aux_data = {
        #     'kpt3d': target['kp3d'],
        #     'obs_kpt2d': target['obs_kp2d'],
        # }   
        return target

    def __getitem__(self, item):
        return self.get_motion(item)

    def __len__(self):
        return len(self.split_index)


class Bones2DDatasetV2SingleView(Bones2DDatasetV2):
    def __init__(self, num_views=1, **kwargs):
        super().__init__(num_views=num_views, **kwargs)
        
    def get_precompute_data(self, item):
        idx = self.split_index[item]
        data = pickle.load(open(f"{self.precompute_data_folder}/{idx:06d}.pkl", 'rb'))
        length = data['length']
        assert length == data['obs_kp2d'].shape[0]
        # cut or pad
        if length > self.num_frames:
            start = np.random.randint(0, length - self.num_frames + 1)
            end = start + self.num_frames
            for key in {'obs_kp2d', 'mask', 'conf'}:
                data[key] = data[key][start:end]
        elif length < self.num_frames:
            for key in {'obs_kp2d', 'mask', 'conf'}:
                val = data[key]
                data[key] = np.concatenate([val, np.zeros((self.num_frames - val.shape[0],) + val.shape[1:])], axis=0)
        data['mask'] = data['mask'].astype(np.bool8)
        for key in {'obs_kp2d', 'conf'}:
            data[key] = data[key].astype(np.float32)
        data['meta'] = {
            'dataset_id': 'bones2d_v2',
        }
        return data

    def __getitem__(self, item):
        if self.precompute:
            return self.get_precompute_data(item)
        else:
            target = self.get_motion(item)
            m_length = target['m_length']
            if self.use_orig_length:
                mask = np.ones((m_length, ))
            else:
                mask = np.zeros((self.num_frames, ))
                mask[:m_length] = 1
            conf = mask[:, None].repeat(self.num_keypoints, axis=-1)
            return {
                'idx': self.split_index[item],
                'obs_kp2d': target['obs_kp2d'].astype(np.float32),
                'mask': mask.astype(np.bool8),
                'conf': conf,
                'length': m_length,
                'is_2d': True,
                'meta': {
                    'dataset_id': 'bones2d_v2',
                }
            }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--index', type=int, default=0, help='Index of the dataset item to process')
    parser.add_argument('--num_jobs', type=int, default=2, help='Index of the dataset item to process')
    args = parser.parse_args()

    
    cam_aug_cfg = {
        'elevation_mean': 5,
        'elevation_std': 22.5,
        'radius_min': 2,
        'radius_max': 16
    }
    # dataset = Bones2DDatasetV2(debug=True, normalize_type='bbox_frame', num_keypoints=25, use_coco_pelvis=True, use_our_normalization=False, cam_aug_cfg=cam_aug_cfg)
    output_folder = '/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/GVHMR/bones2d/v1'
    os.makedirs(output_folder, exist_ok=True)
    
    """ test """
    dataset = Bones2DDatasetV2SingleView(use_orig_length=True, num_frames=250, device='cuda')
    for i in range(100):
        data = dataset[i]
        print(i, data['idx'])
        print(data['obs_kp2d'].shape)
        print(data['obs_kp2d'][0, 0])
    exit()
    
    """ gen data """
    dataset = Bones2DDatasetV2SingleView(use_orig_length=True, num_frames=800, device='cuda', precompute_data_folder=None)
    start = args.index * len(dataset) // args.num_jobs
    end = ((args.index + 1) * len(dataset) // args.num_jobs) if args.index + 1 < args.num_jobs else len(dataset)
    print(start, end, len(dataset))
    
    for i in range(start, end):
        idx = dataset.split_index[i]
        idx2 = dataset.split_index[min(i+1, len(dataset)-1)]
        fname = f"{output_folder}/{idx:06d}.pkl"
        fname_next = f"{output_folder}/{idx2:06d}.pkl"
        if os.path.exists(fname) and os.path.exists(fname_next):
            print(f"Skipping {i} {idx}")
            continue
        
        data = dataset[i]
        print(i, data['idx'])
        # print(data['obs_kp2d'][0, 0])
        pickle.dump(data, open(f"{output_folder}/{data['idx']:06d}.pkl", 'wb'))
    # import os, sys
    # sys.path.append(os.path.join(os.getcwd()))
    # from motiondiff.utils.config import create_config
    # from omegaconf import OmegaConf
    
    # conf = create_config('gen2d_mv_test_mask_fc_st_2d_fix_w3d_25kp_kungfu_norm')
    # dataset_cfg = conf.train_dataset.copy()
    # add_cfg = dataset_cfg.pop('dataset_kwargs')
    # dataset_cfg.update(add_cfg.get('humanml3d', {}))
    # dataset = Bones2DDatasetV2SingleView(**dataset_cfg, debug=True)