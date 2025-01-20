import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
import sys
sys.path.append('./')
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm


from torchvision.transforms import transforms
from pycocotools.coco import COCO
from motiondiff.models.common.utils.human_models import smpl_x, smpl
from motiondiff.utils.torch_transform import rotation_matrix_to_angle_axis, quaternion_to_angle_axis, quat_mul, angle_axis_to_quaternion

from motiondiff.data_pipeline.humanml.utils.get_opt import get_opt
from motiondiff.data_pipeline.humanml.utils.word_vectorizer import WordVectorizer
from motiondiff.models.common.utils.preprocessing import transform_joint_to_other_db
from motiondiff.data_pipeline.datasets.kp2d_dataset import KP2DDataset
    

class HumanML3DSubDatasetV1(torch.utils.data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, debug=False, rng=None, feat_version='1', major_version='1'):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.rng = rng
        self.feat_version = feat_version
        self.major_version = major_version
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        # split_file = split_file.replace('EgoSMPL3D', 'HumanML3D')   # TODO:

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        if debug:
            if 'train' in split_file:
                id_list = id_list[:10]
            else:
                id_list = id_list[:10]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            # try:
            motion = np.load(pjoin(opt.motion_dir, name + '.npy'))

            smpl_params = torch.load(pjoin(opt.smpl_dir, name + ".npy"))
            smpl_pose = smpl_params["smpl_pose"]
            smpl_trans = smpl_params["smpl_trans"]
            # joints24 = smpl_params["joints24"]
            # joints17 = smpl_params["joints17"]

            if (len(motion)) < min_motion_len or (len(motion) >= 200):
                continue
            text_data = []
            flag = False
            with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        # try:
                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                        n_smpl_pose = smpl_pose[int(f_tag*20) : int(to_tag*20)]
                        n_smpl_trans = smpl_trans[int(f_tag*20) : int(to_tag*20)]
                        # n_joints24 = joints24[int(f_tag*20) : int(to_tag*20)]
                        # n_joints17 = joints17[int(f_tag*20) : int(to_tag*20)]

                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                            continue
                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        while new_name in data_dict:
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name

                        data_dict[new_name] = {
                            "motion": n_motion,
                            "length": len(n_motion),
                            "text": [text_dict],
                            "smpl_pose": n_smpl_pose,
                            "smpl_trans": n_smpl_trans,
                            # "joints24": n_joints24,
                            # "joints17": n_joints17,
                        }
                        new_name_list.append(new_name)
                        length_list.append(len(n_motion))
                        # except:
                        #     print(line_split)
                        #     print(line_split[2], line_split[3], f_tag, to_tag, name)
                            # break

            if flag:
                data_dict[name] = {
                    "motion": motion,
                    "length": len(motion),
                    "text": text_data,
                    "smpl_pose": smpl_pose,
                    "smpl_trans": smpl_trans,
                    # "joints24": joints24,
                    # "joints17": joints17,
                }
                new_name_list.append(name)
                length_list.append(len(motion))
            # except:
            #     pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        rng = self.rng if self.rng is not None else np.random
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = rng.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = rng.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = rng.randint(0, max(len(motion) - m_length, 1))
        # motion = motion[idx:idx+m_length]

        ''' re-canolicalize the motion '''
        motion_th = torch.tensor(motion).float()
        pose_feat = motion_th[:, 9:147]

        smpl_pose = data["smpl_pose"]
        trans = torch.zeros_like(smpl_pose[..., 0, 0, :3])
        # joints24, joints17 = data["joints24"], data["joints17"]     # [T, K, 3]
        orient = smpl_pose[:, 0, :, :]

        local_pose = smpl_pose[idx:idx+m_length, 1:, :, :]
        smpl_pose = smpl_pose[idx:idx+m_length, :, :, :]
        orient = orient[idx:idx+m_length]
        trans = trans[idx:idx+m_length]
        pose_feat = pose_feat[idx:idx+m_length]
        # joints24 = joints24[idx:idx+m_length]
        # joints17 = joints17[idx:idx+m_length]

        motion = None

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            smpl_pose,
            trans,
            # joints17,
            # joints24,
        )


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3DV1(KP2DDataset):
    def __init__(self, datapath='./dataset/humanml_opt.txt', num_keypoints=14, num_frames=196, mode='train', split="train", debug=False, rng=None, img_w=1024, img_h=1024, num_views=4, cam_radius=8, cam_elevation=10, focal_scale=2, synthetic_view_type='even', normalize_type='image_size', bbox_scale=1.4, use_coco_pelvis=False, tilt_prob=0.0, tilt_std=15, normalize_stats_dir=None, sample_beta=False, **kwargs):
        super().__init__(num_frames, img_w, img_h, num_views, cam_radius, cam_elevation, focal_scale, synthetic_view_type, normalize_type, bbox_scale, tilt_prob, tilt_std, normalize_stats_dir)
        self.mode = mode
        self.dataset_name = 't2m'
        self.dataname = 't2m'
        self.debug = debug
        self.feat_version = kwargs.get('feat_version', '1')
        self.major_version = self.feat_version.split('_')[0]
        self.get_coco_keypoints = num_keypoints == 25
        self.sample_beta = sample_beta
        abs_base_path = '.'
        dataset_opt_path = os.path.join(abs_base_path, datapath)
        device = None
        opt = get_opt(dataset_opt_path, device)
        opt.data_root = './dataset/EgoSMPL3D'
        opt.motion_dir = os.path.join(opt.data_root, f'new_joint_vecs_v{self.feat_version}')
        opt.smpl_dir = os.path.join(opt.data_root, 'orig_smpl_params')
        opt.text_dir = os.path.join(opt.data_root, 'texts')
        opt.text_dir = os.path.join(abs_base_path, opt.text_dir)
        opt.model_dir = os.path.join(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = os.path.join(abs_base_path, opt.checkpoints_dir)
        opt.data_root = os.path.join(abs_base_path, opt.data_root)
        opt.save_root = os.path.join(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt
        self.mean = None
        self.std = None
        self.split = split
        self.split_file = os.path.join(opt.data_root, f'{split}.txt')
        self.num_keypoints = num_keypoints
        self.w_vectorizer = WordVectorizer(os.path.join(abs_base_path, 'glove'), 'our_vab')
        self.t2m_dataset = HumanML3DSubDatasetV1(
            self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, self.debug, rng=rng, feat_version=self.feat_version, major_version=self.major_version
        )
        self.num_actions = 1
        self.use_coco_pelvis = use_coco_pelvis
        self.wham_regressor = torch.tensor(np.load('data/smpl/J_regressor_wham.npy'))
        self.smpl_joints = smpl.all_joints_name
        self.coco_joints = ['Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
                            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']
        self.smplx_valid_joints = [i for i, j in enumerate(smpl_x.joints_name[:num_keypoints]) if j in self.coco_joints or j in self.smpl_joints]
        self.coco_smplx_ind = [smpl_x.joints_name.index(j) for j in self.coco_joints]   # [24, 22, 23, 20, 21, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 6]
        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, it is probably because your data dir has only texts and no motions. To train and evaluate MDM you should get the FULL data as described in the README file.'

    def get_motion(self, item):
        data = self.t2m_dataset.__getitem__(item)
        text = data[2]
        smpl_pose = data[7]
        smpl_trans = data[8]
        pose = rotation_matrix_to_angle_axis(smpl_pose).reshape(-1, 72)
        if self.sample_beta:
            shape = torch.randn_like(pose[..., :10])
        else:
            shape = torch.zeros_like(pose[..., :10])
        output = smpl.layer['neutral'](betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3], transl=smpl_trans)
        smpl_joint_world = output.joints[:, :len(smpl.all_joints_name)].numpy()
        smplx_joint_world = transform_joint_to_other_db(smpl_joint_world.transpose(1, 0, 2), smpl.all_joints_name, smpl_x.joints_name[:25]).transpose(1, 0, 2)
        if self.get_coco_keypoints:
            wham_joints = torch.matmul(self.wham_regressor, output.vertices)
            coco_joints = wham_joints[:, :17]
            smplx_joint_world[:, self.coco_smplx_ind] = coco_joints
        
        smplx_joint_world = smplx_joint_world[:, :self.num_keypoints]
        if self.use_coco_pelvis:
            smplx_joint_world[:, 0] = (smplx_joint_world[:, 1] + smplx_joint_world[:, 2]) / 2
        motion, kpt3d, cam_dict = self.generate_cam_and_2d_motion(smplx_joint_world)
        motion, kpt3d, m_length = self.pad_motion(motion, kpt3d)
        return text, motion, kpt3d, cam_dict, m_length
        
    def __getitem__(self, item):
        text, motion, kpt3d, cam_dict, m_length = self.get_motion(item)
        info = {
            'smpl_valid_joints': self.smplx_valid_joints,
            'dataset_name': 'humanml3d'
        }
        aux_data = {
            'kpt3d': kpt3d,
        }
        return text, motion, m_length, cam_dict, aux_data, info

    def __len__(self):
        return self.t2m_dataset.__len__()


class HumanML2DV1(HumanML3DV1):
    def __init__(self, num_views=1, **kwargs):
        super().__init__(num_views=num_views, **kwargs)

    def __getitem__(self, item):
        idx = item % len(self.t2m_dataset)
        text, motion, kpt3d, cam_dict, m_length = self.get_motion(idx)
        motion_2d = motion
        mask = np.zeros((self.num_frames, ))
        mask[:m_length] = 1
        conf = mask[:, None].repeat(self.num_keypoints, axis=-1)
        return {
            'text': text,
            'motion_2d': motion_2d.astype(np.float32),
            'motion_mask': mask.astype(np.float32),
            'conf': conf,
            'lengths': m_length,
            'smpl_valid_joints': np.array(self.smplx_valid_joints),
            'is_2d': True,
            'dataset_name': 'humanml2d'
        }

    def __len__(self):
        return 100000   # self.t2m_dataset.__len__()


if __name__ == '__main__':
    # dataset = HumanML2DV1(debug=True, normalize_type='bbox_frame', num_keypoints=25, use_coco_pelvis=True, tilt_prob=1.0)
    import os, sys
    sys.path.append(os.path.join(os.getcwd()))
    from motiondiff.utils.config import create_config
    from omegaconf import OmegaConf
    
    conf = create_config('gen2d_mv_test_mask_fc_st_2d_fix_w3d_25kp_kungfu_norm')
    dataset_cfg = conf.train_dataset.copy()
    add_cfg = dataset_cfg.pop('dataset_kwargs')
    dataset_cfg.update(add_cfg.get('humanml3d', {}))
    dataset = HumanML3DV1(**dataset_cfg, debug=True)
    # for i in range(10):
    #     batch = dataset[i]
    #     print(f'data {i}')
        
    # compute normalization stats
    count = 0
    mean = 0
    M2 = 0
    # Loop over the dataset and update the mean and std incrementally
    for i in tqdm(range(len(dataset))):
        text, motion, m_length, _, _ = dataset[i]
        motion = motion[:m_length].reshape(m_length * conf.num_views, -1)
        # Update count, mean, and M2 for the new data
        count += motion.shape[0]
        delta = motion - mean
        mean += delta.sum(axis=0) / count
        delta2 = motion - mean
        M2 += np.sum(delta * delta2, axis=0)

    # Finalize the mean and standard deviation
    std = np.sqrt(M2 / count)
    std[np.isnan(std)] = 1e-3
    std[std < 1e-3] = 1e-3
    # Save the computed statistics
    stats_folder = 'stats/gen2d_mv_test_mask_fc_st_2d_fix_w3d_25kp'
    os.makedirs(stats_folder, exist_ok=True)
    np.save(f'{stats_folder}/mean.npy', mean)
    np.save(f'{stats_folder}/std.npy', std)
    print('mean:', mean)
    print('std:', std)
    print(f'{os.getcwd()}/{stats_folder}')
    
