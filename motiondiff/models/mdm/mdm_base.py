import torch
import pytorch_lightning as pl
import numpy as np
import cv2 as cv
import wandb
import time
import pickle
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from .mdm_denoiser import MDMDenoiser
from motiondiff.utils.scheduler import update_scheduled_params
from motiondiff.utils.torch_utils import tensor_to, interp_tensor_with_scipy
from motiondiff.utils.tools import import_type_from_str, wandb_run_exists
from motiondiff.models.model_util import create_gaussian_diffusion
from motiondiff.models.common.cfg_sampler import ClassifierFreeSampleModel
from motiondiff.diffusion.resample import create_named_schedule_sampler
from motiondiff.diffusion.gaussian_diffusion import ModelMeanType
from motiondiff.diffusion.nn import sum_flat
from motiondiff.data_pipeline.tensors import collate
from motiondiff.data_pipeline.humanml.common.quaternion import *
from motiondiff.data_pipeline.humanml.scripts.motion_process import recover_from_ric, recover_root_rot_pos
from motiondiff.utils.torch_transform import quat_apply, normalize
from motiondiff.utils.conversion import humanml_to_smpl, joints_to_smpl
from motiondiff.models.common.smpl import SMPL_BONE_ORDER_NAMES
from collections.abc import Iterable


motion_rep_dims = {
    'full263': 263,
    'position': 67,
    'global_position': 72,
    'global_root_local_joints': 264,
    'global_root': 5,
    'global_root_vel_local_joints': 263
}

motion_rep_root_dims = {
    'full263': 4,
    'position': 4,
    'global_position': 3,
    'global_root_local_joints': 5,
    'global_root': 5,
    'global_root_vel_local_joints': 4
}


""" 
Main Model
"""

class MDMBase(pl.LightningModule):

    def __init__(self, cfg, is_inference=False):
        super().__init__()
        self.cfg = cfg
        self.is_inference = is_inference
        self.model_cfg = cfg.model
        self.motion_rep = cfg.model.get('motion_rep', 'full263')
        self.motion_rep_dim = motion_rep_dims[self.motion_rep]
        self.motion_root_dim = motion_rep_root_dims[self.motion_rep]
        self.motion_localjoints_dim = 63
        self.normalize_global_pos = cfg.model.get('normalize_global_pos', False)
        self.global_pos_z_up = cfg.model.get('global_pos_z_up', True)
        self.transform_root_traj = self.model_cfg.get('transform_root_traj', False)
        self.humanml_root_stats_file  = cfg.model.get('humanml_root_stats_file', 'data/stats/HumanML3D_global_pos_stats.npy')
        self.model_cfg.denoiser.njoints = self.motion_rep_dim
        self.model_cfg.denoiser.normalize_global_pos = self.normalize_global_pos
        self.humanml_mean = torch.tensor(np.load('dataset/HumanML3D/Mean.npy')).float()
        self.humanml_std = torch.tensor(np.load('dataset/HumanML3D/Std.npy')).float()
        humanml_global_mean, humanml_global_std  = np.load(self.humanml_root_stats_file)
        self.humanml_global_mean = torch.tensor(humanml_global_mean).float()
        self.humanml_global_std = torch.tensor(humanml_global_std).float()
        if not is_inference:
            self.load_aug_text_dict()
        self.load_ext_models()
        return
    
    def load_pretrain_checkpoint(self):
        if 'pretrained_checkpoint' in self.model_cfg:
            cp_cfg = self.model_cfg.pretrained_checkpoint
            state_dict = torch.load(cp_cfg.path, map_location='cpu')['state_dict']
            filter_keys = cp_cfg.get('filter_keys', [])
            if len(filter_keys) > 0:
                print(f'Filtering checkpoint keys: {filter_keys}')
                skipped_keys = [k for k in state_dict.keys() if any(key in k for key in filter_keys)]
                print(f'Skipped keys: {skipped_keys}')
                state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in filter_keys)}
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=cp_cfg.get('strict', True))
            if len(missing_keys) > 0:
                print(f'Missing keys: {missing_keys}')
            if len(unexpected_keys) > 0:
                print(f'Unexpected keys: {unexpected_keys}')

    def load_ext_models(self):
        self.ext_models = {}
        em_cfg = self.model_cfg.get('ext_models', {})
        for name, cfg in em_cfg.items():
            em_cfg = import_type_from_str(cfg.config.type)(**cfg.config.args)
            em = import_type_from_str(em_cfg.model.type)(em_cfg, is_inference=True, preload_checkpoint=False)
            checkpoint = torch.load(cfg.checkpoint, map_location='cpu')['state_dict']
            em.load_state_dict(checkpoint)
            em.eval()
            self.ext_models[name] = em

    def to(self, device):
        super().to(device)
        for key in self.ext_models:
            self.ext_models[key].to(device)
        return
    
    def load_aug_text_dict(self):
        self.augment_text = self.model_cfg.get('augment_text', False)
        if self.augment_text:
            assert 'aug_text_file' in self.model_cfg
            self.aug_text_dict = pickle.load(open(self.model_cfg.aug_text_file, 'rb'))

    def init_diffusion(self):
        self.train_diffusion = create_gaussian_diffusion(self.model_cfg.diffusion, training=True)
        self.test_diffusion = create_gaussian_diffusion(self.model_cfg.diffusion, training=False)
        self.schedule_sampler = create_named_schedule_sampler(self.model_cfg.diffusion.schedule_sampler_type, self.train_diffusion)
        self.guided_denoiser = ClassifierFreeSampleModel(self.denoiser)
        return
    
    def get_diffusion_pred_target(self, data, t, noise=None):
        diffusion = self.train_diffusion if self.training else self.test_diffusion

        x_start = data['motion']
        denoiser_kwargs = data['cond']
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = diffusion.q_sample(x_start, t, noise=noise)

        sched_samp_repeat_timesteps_only = self.cfg.train.get('sched_samp_repeat_timesteps_only', False)
        repeat_final_timesteps = self.cfg.model.diffusion.get('repeat_final_timesteps', None)
        if repeat_final_timesteps is None:
            num_repeat_steps = 0
        elif '%' in repeat_final_timesteps:
            num_repeat_steps = int(int(repeat_final_timesteps.replace('%', '')) / 100 * diffusion.num_timesteps)
        else:
            num_repeat_steps = int(repeat_final_timesteps)

        sched_samp_prob_root = self.cfg.train.get('sched_samp_prob_root', 0.0)
        sched_samp_prob_joints = self.cfg.train.get('sched_samp_prob_joints', 0.0)
        if sched_samp_prob_root > 0.0 or sched_samp_prob_joints > 0.0:
            using_gt_root_cond = torch.bernoulli(torch.ones_like(t) * sched_samp_prob_root).bool()
            using_gt_joints_cond = torch.bernoulli(torch.ones_like(t) * sched_samp_prob_joints).bool()
            if sched_samp_repeat_timesteps_only:
                using_gt_root_cond[t >= num_repeat_steps] = 0
                using_gt_joints_cond[t >= num_repeat_steps] = 0
            denoiser_kwargs['using_gt_root_cond'] = using_gt_root_cond
            denoiser_kwargs['using_gt_joints_cond'] = using_gt_joints_cond
            denoiser_kwargs['gt_motion'] = x_start

        data['model_pred'] = self.denoiser(x_t, diffusion._scale_timesteps(t), **denoiser_kwargs)

        if diffusion.model_mean_type == ModelMeanType.PREVIOUS_X:
            data['target'] = diffusion.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0]
        elif diffusion.model_mean_type == ModelMeanType.START_X:
            data['target'] = x_start
        elif diffusion.model_mean_type == ModelMeanType.EPSILON:
            data['target'] = noise
        else:
            raise NotImplementedError

        # learnable variance
        if self.model_cfg.diffusion.get('learnable_variance', False):
            data['vb'] = diffusion.get_vb_term(x_t, x_start, t, data['model_pred'])

        return data

    def on_save_checkpoint(self, checkpoint) -> None:
        if wandb_run_exists():
            checkpoint['wandb_run_id'] = wandb.run.id
        # exclude some keys from the checkpoint
        excluded_keys = self.denoiser.get_excluded_keys()
        all_state_keys = list(checkpoint['state_dict'].keys())
        for key in all_state_keys:
            if any([exc_key in key for exc_key in excluded_keys]):
                del checkpoint['state_dict'][key]
        return
    
    def augment_data_text(self, data):
        new_text_dict = []
        for text in data['cond']['y']['text']:
            if text in self.aug_text_dict:
                new_text = self.aug_text_dict[text][np.random.randint(len(self.aug_text_dict[text]))]
                new_text_dict.append(new_text)
            else:
                new_text_dict.append(text)
        # for old, new in zip(data['cond']['y']['text'], new_text_dict):
        #     if old != new:
        #         print(f'Augmenting text: {old} -> {new}')
        #     else:
        #         print(f'Keeping old text: {old}')
        data['cond']['y']['text'] = new_text_dict

    def generate_motion_mask(self, motion_mask_cfg, motion, lengths, use_mask_type=None, use_unknownt_mask_type=None, return_keyframes=False, mask_cfgs=None, unknownt_mask_cfg=None):
        '''
        If mask_cfgs / unknownt_mask_cfg is given, uses these for configuring individual mask types rather than the base motion_mask_cfg.
        '''
        comp_mask_prob = motion_mask_cfg.get('comp_mask_prob', 1.0)
        mask_comp_type = motion_mask_cfg.get('mask_comp_type', 'exclusive')
        
        mask_probs = None
        mask_type = use_mask_type
        if 'mask_probs' in motion_mask_cfg:
            mask_probs = np.array(motion_mask_cfg.mask_probs) / np.sum(motion_mask_cfg.mask_probs)
            if use_mask_type is not None:
                mask_type = use_mask_type
            else:
                mask_type = np.random.choice(motion_mask_cfg.mask_types, p=mask_probs)
        unknownt_mask_probs = None
        unknownt_mask_type = use_unknownt_mask_type
        if 'unknownt_mask_probs' in motion_mask_cfg:
            unknownt_mask_probs = np.array(motion_mask_cfg.unknownt_mask_probs) / np.sum(motion_mask_cfg.unknownt_mask_probs)
            if use_unknownt_mask_type is not None:
                unknownt_mask_type = use_unknownt_mask_type
            else:
                unknownt_mask_type = np.random.choice(motion_mask_cfg.unknownt_mask_types, p=unknownt_mask_probs)
        
        if not isinstance(mask_type, list):
            mask_type = [mask_type]
        
        motion_mask = torch.zeros_like(motion)
        rm_text_flag = torch.zeros(motion.shape[0], device=motion.device)
        root_dim = self.motion_root_dim
        ljoint_dim = self.motion_localjoints_dim
        all_keyframe_idx = None
        global_motion = None
        global_joint_mask = None
        global_joint_func = None

        selected_keyframe_t = None
        unknownt_observed_motion = None
        unknownt_motion_mask = None


        def get_root_mask_indices(mode):
            if mode in {'root+joints', 'root'}:
                root_mask_ind = slice(0, root_dim)
            elif mode == 'root_xy+rot':
                if self.motion_rep in {'full263', 'position'}:
                    root_mask_ind = slice(0, 3)
                elif self.motion_rep in {'global_root_local_joints', 'global_root'}:
                    root_mask_ind = np.array([0, 2, 3, 4]) # in the original coordinate, y is up, so we use z.
                else:
                    raise NotImplementedError
            elif mode == 'root_xy':
                if self.motion_rep in {'full263', 'position'}:
                    root_mask_ind = slice(1, 3)
                elif self.motion_rep in {'global_root_local_joints', 'global_root'}:
                    root_mask_ind = np.array([0, 2]) # in the original coordinate, y is up, so we use z.
                else:
                    raise NotImplementedError
            elif mode == 'root_pos+joints':
                if self.motion_rep in {'full263', 'position'}:
                    root_mask_ind = slice(1, root_dim)
                elif self.motion_rep in {'global_root_local_joints', 'global_root'}:
                    root_mask_ind = slice(0, 3)
                else:
                    raise NotImplementedError
            elif mode == 'rootheight+joints':
                if self.motion_rep in {'full263', 'position'}:
                    root_mask_ind = slice(3, root_dim)
                elif self.motion_rep in {'global_root_local_joints', 'global_root'}:
                    root_mask_ind = slice(1, 2)
                else:
                    raise NotImplementedError
            elif mode == 'joints':
                root_mask_ind = slice(0, 0)
            else:
                raise NotImplementedError
            return root_mask_ind

        def root_traj(_cfg):
            nonlocal motion_mask, rm_text_flag
            xy_only = _cfg.get('xy_only', False)
            for i in range(motion.shape[0]):
                mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                if self.motion_rep in {'full263', 'position'}:
                    eind = 3 if xy_only else 4
                    motion_mask[i, :eind, ..., :mlen] = 1.0
                elif self.motion_rep in {'global_root_local_joints', 'global_root'}:
                    mode = _cfg.get('mode', 'pos+rot')
                    if mode == 'pos+rot':
                        root_mask_ind = np.arange(5)
                    elif mode == 'pos':
                        root_mask_ind = np.arange(3)
                    elif mode == 'pos_xy':
                        root_mask_ind = np.array([0, 2])
                    elif mode == 'pos_xy+rot':
                        root_mask_ind = np.array([0, 2, 3, 4])  # in the original coordinate, y is up, so we use z.
                    motion_mask[i, root_mask_ind, ..., :mlen] = 1.0
            rm_text_flag = torch.from_numpy(np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])).to(motion.device)
        
        def random_feat_mask(_cfg):
            nonlocal motion_mask, rm_text_flag, all_keyframe_idx
            include_rootheight = _cfg.get('include_rootheight', True)
            obs_feat_prob = _cfg.get('obs_feat_prob', 0.1)
            feat_ind = []
            if include_rootheight:
                feat_ind.append(np.array([1]))
                feat_ind.append(np.arange(root_dim, root_dim + ljoint_dim))
            feat_ind = np.concatenate(feat_ind)
            feat_dim = len(feat_ind)
            feat_mask = torch.from_numpy(np.random.binomial(1, obs_feat_prob, size=(motion.shape[0], feat_dim, 1, motion.shape[-1]))).type_as(motion_mask)
            motion_mask[:, feat_ind, :, :] = feat_mask
            rm_text_flag = torch.from_numpy(np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])).to(motion.device)
        
        def keyframes(_cfg):
            nonlocal motion_mask, rm_text_flag, all_keyframe_idx
            all_keyframe_idx = _cfg.get('keyframe_idx', None)
            sample_keyframes = all_keyframe_idx is None
            if sample_keyframes:
                all_keyframe_idx = []
            mode = _cfg.get('mode', 'root+joints')
            root_mask_ind = get_root_mask_indices(mode)
            for i in range(motion.shape[0]):
                if sample_keyframes:
                    mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                    num_keyframes = np.random.randint(_cfg.num_range[0], min(_cfg.num_range[1], mlen) + 1)
                    keyframe_idx = np.random.choice(mlen, num_keyframes, replace=False)
                    all_keyframe_idx.append(keyframe_idx)
                else:
                    keyframe_idx = all_keyframe_idx[i]
                if not mode in {'root', 'root_xy+rot', 'root_xy'}:
                    motion_mask[i, root_dim:root_dim + ljoint_dim, :, keyframe_idx] = 1.0 # only root + local joint positions
                if isinstance(root_mask_ind, slice):
                    motion_mask[i, root_mask_ind, :, keyframe_idx] = 1.0
                else:
                    for rmi in root_mask_ind:
                        motion_mask[i, rmi, :, keyframe_idx] = 1.0 # root_mask_ind may not be a slice
            rm_text_flag = torch.from_numpy(np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])).to(motion.device)
        
        def local_joints(_cfg):
            nonlocal motion_mask, rm_text_flag, all_keyframe_idx
            mode = _cfg.get('mode', 'joints')
            root_mask_ind = get_root_mask_indices(mode)
            smpl_joint_names = SMPL_BONE_ORDER_NAMES[1:22]
            joint_names = _cfg.get('joint_names',smpl_joint_names)
            if joint_names == 'all':
                joint_names = smpl_joint_names
            joint_indices = [smpl_joint_names.index(joint_name) for joint_name in joint_names]
            all_keyframe_idx = _cfg.get('keyframe_idx', None)
            for i in range(motion.shape[0]):
                if all_keyframe_idx is None:
                    # randomly sample
                    mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                    if _cfg.num_fr_range == 'all':
                        keyframe_idx = np.arange(mlen)
                    else:
                        num_keyframes = np.random.randint(_cfg.num_fr_range[0], min(_cfg.num_fr_range[1], mlen) + 1)
                        keyframe_idx = np.random.choice(mlen, num_keyframes, replace=False)
                else:
                    keyframe_idx = all_keyframe_idx[i]
                if _cfg.get('consistent_joints', False):    # consistent_joints: masked joints are the same for all the frames
                    joint_mask = np.zeros(21, dtype=np.float32)
                    joint_mask[joint_indices] = np.random.binomial(1, _cfg.obs_joint_prob, size=len(joint_indices))
                    motion_mask[i, root_dim:root_dim + ljoint_dim, :, keyframe_idx] = torch.from_numpy(joint_mask).repeat_interleave(3)[:, None, None].to(motion.device)
                    motion_mask[i, root_mask_ind, :, keyframe_idx] = 1.0
                else:
                    for fr in keyframe_idx:
                        joint_mask = np.zeros(21, dtype=np.float32)
                        joint_mask[joint_indices] = np.random.binomial(1, _cfg.obs_joint_prob, size=len(joint_indices))
                        motion_mask[i, root_dim:root_dim + ljoint_dim, :, fr] = torch.from_numpy(joint_mask).repeat_interleave(3).to(motion.device)
                        motion_mask[i, root_mask_ind, :, fr] = 1.0
            rm_text_flag = torch.from_numpy(np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])).to(motion.device)
        
        def global_joints(_cfg):
            nonlocal motion_mask, rm_text_flag, global_motion, global_joint_mask, global_joint_func, all_keyframe_idx

            def get_local_joints(sample, g_motion, g_joint_mask):
                def get_local_joints(in_motion):
                    g_joints = in_motion[:, :, 0].transpose(1, 2).view(sample_root_pos.shape[:2] + (-1, 3))
                    l_joints = g_joints[..., :22, :].clone()
                    l_joints[..., :2] -= sample_root_pos[..., :2]
                    l_joints = qrot(sample_r_rot.expand(l_joints.shape[:-1] + (4,)), l_joints)
                    return l_joints.view(l_joints.shape[:-2] + (-1,))

                sample_global, sample_r_rot = self.get_global_position(sample, return_r_rot=True)
                sample_r_rot = sample_r_rot[:, 0].unsqueeze(2)
                sample_root_pos = sample_global[:, :3, 0].transpose(1, 2).unsqueeze(2)
                g_motion_l_joints = get_local_joints(g_motion)
                sample_l_joints = get_local_joints(sample_global)
                l_joints_diff = g_motion_l_joints - sample_l_joints
                g_motion_l_joints *= g_joint_mask
                l_joints_diff *= g_joint_mask
                return g_motion_l_joints, l_joints_diff

            global_joint_func = get_local_joints
            global_motion = self.get_global_position(motion)
            global_joint_mask = torch.zeros((motion.shape[0], motion.shape[-1], 66)).to(self.device)
            joint_names = _cfg.get('joint_names', SMPL_BONE_ORDER_NAMES[:22])
            if joint_names == 'all':
                joint_names = SMPL_BONE_ORDER_NAMES[:22]
            joint_indices = [SMPL_BONE_ORDER_NAMES.index(joint_name) for joint_name in joint_names]
            all_keyframe_idx = _cfg.get('keyframe_idx', None)
            for i in range(motion.shape[0]):
                if all_keyframe_idx is None:
                    # randomly sample
                    mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                    if _cfg.num_fr_range == 'all':
                        keyframe_idx = np.arange(mlen)
                    elif _cfg.num_fr_range == 'last':
                        keyframe_idx = np.array([mlen - 1])
                    else:
                        num_keyframes = np.random.randint(_cfg.num_fr_range[0], min(_cfg.num_fr_range[1], mlen) + 1)
                        keyframe_idx = np.random.choice(mlen, num_keyframes, replace=False)
                else:
                    keyframe_idx = all_keyframe_idx[i]
                if _cfg.get('consistent_joints', False):    # consistent_joints: masked joints are the same for all the frames
                    joint_mask = np.zeros(22, dtype=np.float32)
                    joint_mask[joint_indices] = np.random.binomial(1, _cfg.obs_joint_prob, size=len(joint_indices))
                    global_joint_mask[i, keyframe_idx, :] = torch.from_numpy(joint_mask).repeat_interleave(3).to(motion.device)
                else:
                    for fr in keyframe_idx:
                        joint_mask = np.zeros(22, dtype=np.float32)
                        joint_mask[joint_indices] = np.random.binomial(1, _cfg.obs_joint_prob, size=len(joint_indices))
                        global_joint_mask[i, fr, :] = torch.from_numpy(joint_mask).repeat_interleave(3).to(motion.device)
            rm_text_flag = torch.from_numpy(np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])).to(motion.device)

        def init_unknownt(_cfg):
            nonlocal all_keyframe_idx
            selected_keyframe_t = []
            unknownt_observed_motion = []
            all_keyframe_idx = _cfg.get('keyframe_idx', None)
            for i in range(motion.shape[0]):
                mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                t = np.random.randint(mlen) if all_keyframe_idx is None else all_keyframe_idx[i][0]
                selected_keyframe_t.append(t)
                unknownt_observed_motion.append(motion[i, :, :, [t]])
            selected_keyframe_t = torch.from_numpy(np.array(selected_keyframe_t)).to(motion.device)
            unknownt_observed_motion = torch.stack(unknownt_observed_motion, dim=0)
            unknownt_motion_mask = torch.zeros_like(unknownt_observed_motion)
            return selected_keyframe_t, unknownt_observed_motion, unknownt_motion_mask
        
        def unknownt_root_traj(_cfg):
            nonlocal selected_keyframe_t, unknownt_observed_motion, unknownt_motion_mask, rm_text_flag
            selected_keyframe_t, unknownt_observed_motion, unknownt_motion_mask = init_unknownt(_cfg)
            xy_only = _cfg.get('xy_only', False)
            if self.motion_rep in {'full263', 'position'}:
                eind = 3 if xy_only else 4
                unknownt_motion_mask[:, :eind] = 1.0
            elif self.motion_rep == 'global_root_local_joints':
                mode = _cfg.get('mode', 'pos+rot')
                if mode == 'pos+rot':
                    root_mask_ind = np.arange(5)
                elif mode == 'pos':
                    root_mask_ind = np.arange(3)
                elif mode == 'pos_xy':
                    root_mask_ind = np.array([0, 2])
                elif mode == 'pos_xy+rot':
                    root_mask_ind = np.array([0, 2, 3, 4])  # in the original coordinate, y is up, so we use z.
                unknownt_motion_mask[:, root_mask_ind] = 1.0
            rm_text_flag = torch.from_numpy(np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])).to(motion.device)
        
        def unknownt_keyframes(_cfg):
            nonlocal selected_keyframe_t, unknownt_observed_motion, unknownt_motion_mask, rm_text_flag
            selected_keyframe_t, unknownt_observed_motion, unknownt_motion_mask = init_unknownt(_cfg)
            mode = _cfg.get('mode', 'root+joints')
            root_mask_ind = get_root_mask_indices(mode)
            if not mode in {'root', 'root_xy+rot', 'root_xy'}:
                unknownt_motion_mask[:, root_dim:root_dim + ljoint_dim] = 1.0 # only root + local joint positions
            unknownt_motion_mask[:, root_mask_ind] = 1.0
            rm_text_flag = torch.from_numpy(np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])).to(motion.device)
        
        def unknownt_local_joints(_cfg):
            nonlocal selected_keyframe_t, unknownt_observed_motion, unknownt_motion_mask, rm_text_flag
            selected_keyframe_t, unknownt_observed_motion, unknownt_motion_mask = init_unknownt(_cfg)
            mode = _cfg.get('mode', 'joints')
            root_mask_ind = get_root_mask_indices(mode)
            smpl_joint_names = SMPL_BONE_ORDER_NAMES[1:22]
            joint_names = _cfg.get('joint_names',smpl_joint_names)
            if joint_names == 'all':
                joint_names = smpl_joint_names
            joint_indices = [smpl_joint_names.index(joint_name) for joint_name in joint_names]
            joint_mask = np.zeros(21, dtype=np.float32)
            joint_mask[joint_indices] = np.random.binomial(1, _cfg.obs_joint_prob, size=len(joint_indices))
            unknownt_motion_mask[:, root_dim:root_dim + ljoint_dim] = torch.from_numpy(joint_mask).repeat_interleave(3)[:, None, None].to(motion.device)
            unknownt_motion_mask[:, root_mask_ind] = 1.0
            rm_text_flag = torch.from_numpy(np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])).to(motion.device)

        enable_mask = np.random.binomial(1, comp_mask_prob)
        if mask_comp_type == 'exclusive':
            enable_unknownt_mask = 1 - enable_mask
        elif mask_comp_type == 'or':
            enable_unknownt_mask = np.random.binomial(1, motion_mask_cfg.comp_unknownt_mask_prob)
        else:
            raise NotImplementedError

        if use_mask_type is not None or enable_mask:
            for mi, cur_mask_type in enumerate(mask_type):
                # motion_mask is updated in place so can aggregate over all types
                if cur_mask_type != 'no_mask':
                    mask_cfg = motion_mask_cfg.get(cur_mask_type, {}) if mask_cfgs is None else mask_cfgs[mi]
                    mask_func = locals()[mask_cfg.get('func', cur_mask_type)]
                    assert mask_func != 'global_joints', "multi-masking does not support global joints right now!"
                    mask_func(mask_cfg)
        
        # only support a single unknown t mask type since model needs to be trained explicitly to take in more than one
        if (use_unknownt_mask_type is not None or enable_unknownt_mask) and unknownt_mask_type != 'no_mask':
            unknownt_mask_cfg = motion_mask_cfg.get(unknownt_mask_type, {}) if unknownt_mask_cfg is None else unknownt_mask_cfg
            unknownt_mask_func = locals()[unknownt_mask_cfg.get('func', unknownt_mask_type)]
            unknownt_mask_func(unknownt_mask_cfg)

        if use_mask_type is not None or use_unknownt_mask_type is not None:
            # remove text conditioning
            rm_text_flag = torch.ones(motion.shape[0], device=motion.device)

        observed_motion = motion_mask * motion
        res = {
            'motion_mask': motion_mask,
            'observed_motion': observed_motion,
            'rm_text_flag': rm_text_flag,
            'global_motion': global_motion,
            'global_joint_mask': global_joint_mask,
            'global_joint_func': global_joint_func,
            'selected_keyframe_t': selected_keyframe_t,
            'unknownt_observed_motion': unknownt_observed_motion,
            'unknownt_motion_mask': unknownt_motion_mask
        }
        if return_keyframes:
            res['all_keyframe_idx'] = all_keyframe_idx
        return res
    
    def transform_global_motion_for_vis(self, global_motion):
        if not self.global_pos_z_up:
            base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=self.device, dtype=global_motion.dtype)
            g_joints = global_motion[:, :, 0].transpose(1, 2).view(global_motion.shape[0], global_motion.shape[-1], -1, 3)
            global_motion = quat_apply(base_rot.expand(g_joints.shape[:-1] + (4,)), g_joints)
            global_motion = global_motion.view(global_motion.shape[:-2] + (-1,)).transpose(1, 2).unsqueeze(2)
        return global_motion

    def get_global_position(self, motion, return_r_rot=False):
        base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=self.device, dtype=motion.dtype)
        if self.motion_rep in {'global_position', 'full263'}:
            # step 1: denormalize
            motion_norm = motion.permute(0, 2, 3, 1) * self.humanml_std.to(self.device) + self.humanml_mean.to(self.device) # [batch, 1, seq_len, nfeat]
            # step 2: local to global conversion
            if return_r_rot:
                global_motion, r_rot = recover_from_ric(motion_norm, 22, return_r_rot=True)
                if self.global_pos_z_up:
                    r_rot = qmul(r_rot, qinv(base_rot).expand_as(r_rot)) # qinv(qmul(base_rot.expand_as(r_rot), qinv(r_rot)))
            else:
                global_motion = recover_from_ric(motion_norm, 22)
            # step 3: add hand joints
            hand_joints = global_motion[..., -2:, :].clone()
            hand_joints[..., 1] += 1e-2
            smpl_body_pos = torch.cat([global_motion, hand_joints], dim=-2)
            # step 4: apply base rotation (y-up to z-up)
            if self.global_pos_z_up:
                global_motion = quat_apply(base_rot.expand(smpl_body_pos.shape[:-1] + (4,)), smpl_body_pos)
            global_motion = global_motion.view(global_motion.shape[:-2] + (-1,)).permute(0, 3, 1, 2)
        elif self.motion_rep == 'global_root_local_joints':
            samples = motion.permute(0, 2, 3, 1)
            root_samp = samples[..., :5]
            if self.normalize_global_pos:
                root_samp = root_samp * self.humanml_global_std.to(self.device) + self.humanml_global_mean.to(self.device)
            r_pos, rot_cos_sin, local_feats = root_samp[..., :3], root_samp[..., 3:5], samples[..., 5:]
            rot_cos_sin = normalize(rot_cos_sin)
            r_rot_quat = torch.cat([rot_cos_sin[..., [0]], torch.zeros_like(rot_cos_sin[..., [0]]), rot_cos_sin[..., [1]], torch.zeros_like(rot_cos_sin[..., [0]])], dim=-1)
            local_feats_norm = local_feats * self.humanml_std[4:].to(self.device) + self.humanml_mean[4:].to(self.device) # [batch, 1, seq_len, nfeat]
            local_feats_norm_pad = torch.cat([torch.zeros_like(local_feats_norm[..., :4]), local_feats_norm], dim=-1)
            global_motion = recover_from_ric(local_feats_norm_pad, 22, r_rot_quat, r_pos)
            global_motion = global_motion.view(global_motion.shape[:-2] + (-1,)).permute(0, 3, 1, 2)
        else:
            raise NotImplementedError

        return (global_motion, r_rot) if return_r_rot else global_motion

    def convert_motion_rep(self, motion):
        if self.motion_rep == 'position':
            motion = motion[:, :67]
        elif self.motion_rep == 'global_position':
            motion = self.get_global_position(motion)
        elif self.motion_rep == 'global_root_local_joints':
            motion_norm = motion.permute(0, 2, 3, 1) * self.humanml_std.to(self.device) + self.humanml_mean.to(self.device) # [batch, 1, seq_len, nfeat]
            r_rot_quat, r_pos, r_rot_ang = recover_root_rot_pos(motion_norm, return_r_rot_ang=True)
            rot_cos_sin = torch.stack([torch.cos(r_rot_ang), torch.sin(r_rot_ang)], dim=-1)
            motion = torch.cat([r_pos.permute(0, 3, 1, 2), rot_cos_sin.permute(0, 3, 1, 2), motion[:, 4:]], dim=1)
            if self.normalize_global_pos:
                motion[:, :5] = (motion[:, :5] - self.humanml_global_mean[None, :, None, None].to(self.device)) / self.humanml_global_std[None, :, None, None].to(self.device)
        elif self.motion_rep == 'global_root_vel_local_joints':
            motion_norm = motion.permute(0, 2, 3, 1) * self.humanml_std.to(self.device) + self.humanml_mean.to(self.device) # [batch, 1, seq_len, nfeat]
            r_rot_quat, r_pos, r_rot_ang = recover_root_rot_pos(motion_norm, return_r_rot_ang=True)
            ang_v = r_rot_ang[:, :, 1:] - r_rot_ang[:, :, :-1]
            ang_v[ang_v > np.pi] -= 2 * np.pi
            ang_v[ang_v < -np.pi] += 2 * np.pi
            ang_v = torch.cat([ang_v, ang_v[:, :, [-1]]], dim=2).unsqueeze(-1)
            pos_v_xy = r_pos[:, :, 1:, [0, 2]] - r_pos[:, :, :-1, [0, 2]]
            pos_v_xy = torch.cat([pos_v_xy, pos_v_xy[:, :, [-1]]], dim=2)
            pos_y = r_pos[:, :, :, [1]]
            motion_root = torch.cat([pos_v_xy, pos_y, ang_v], dim=-1)
            motion = torch.cat([motion_root.permute(0, 3, 1, 2), motion[:, 4:]], dim=1)
            if self.normalize_global_pos:
                motion[:, :4] = (motion[:, :4] - self.humanml_global_mean[None, :, None, None].to(self.device)) / self.humanml_global_std[None, :, None, None].to(self.device)
        elif self.motion_rep == 'global_root':
            motion_norm = motion.permute(0, 2, 3, 1) * self.humanml_std.to(self.device) + self.humanml_mean.to(self.device) # [batch, 1, seq_len, nfeat]
            r_rot_quat, r_pos, r_rot_ang = recover_root_rot_pos(motion_norm, return_r_rot_ang=True)
            rot_cos_sin = torch.stack([torch.cos(r_rot_ang), torch.sin(r_rot_ang)], dim=-1)
            motion = torch.cat([r_pos.permute(0, 3, 1, 2), rot_cos_sin.permute(0, 3, 1, 2)], dim=1)
            if self.normalize_global_pos:
                motion = (motion - self.humanml_global_mean[None, :, None, None].to(self.device)) / self.humanml_global_std[None, :, None, None].to(self.device)
        return motion

    def training_step(self, batch, batch_idx):
        schedule = self.cfg.get('schedule', dict())
        update_scheduled_params(self, schedule, self.global_step)

        data = {}
        motion, cond = batch
        if motion.device != self.device:
            motion, cond = tensor_to([motion, cond], device=self.device)
        motion = self.convert_motion_rep(motion)

        data['motion'], data['cond'] = motion, cond
        data['mask'] = cond['y']['mask']

        if 'motion_mask' in self.model_cfg:
            res = self.generate_motion_mask(self.model_cfg.motion_mask, data['motion'], data['cond']['y']['lengths'])
            for key in ['motion_mask', 'observed_motion', 'rm_text_flag', 'global_motion', 'global_joint_mask', 'global_joint_func', 'unknownt_observed_motion', 'unknownt_motion_mask']:
                data['cond'][key] = res[key]
            if 'selected_keyframe_t' in res:
                data['selected_keyframe_t'] = res['selected_keyframe_t']

        if self.augment_text:
            self.augment_data_text(data)

        t, t_weights = self.schedule_sampler.sample(motion.shape[0], self.device)
        data = self.get_diffusion_pred_target(data, t)
        loss, loss_dict, loss_uw_dict = self.compute_loss(data, t, t_weights)

        self.log('loss/train_all', loss, on_step=True, on_epoch=True, sync_dist=True)
        for key, val in loss_uw_dict.items():
            self.log(f'loss/train_{key}', val, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def compute_loss(self, data, t, t_weights):

        def masked_l2(cfg):
            part = cfg.get('part', 'all')
            if part == 'all':
                ind = None
            elif part == 'root':
                ind = slice(0, self.motion_root_dim)
            elif part == 'body':
                ind = slice(self.motion_root_dim, None)
            a, b = data['model_pred'], data['target']
            if ind is not None:
                a, b = a[:, ind], b[:, ind]
            mask = data['mask']
            loss = (a - b) ** 2
            loss = sum_flat(loss * mask.float())
            n_entries = a.shape[1] * a.shape[2]
            non_zero_elements = sum_flat(mask) * n_entries
            non_zero_elements[non_zero_elements == 0] = 1
            loss = loss / non_zero_elements
            loss = (loss * t_weights).mean()
            return loss, {}
        
        def masked_global_joints_l2(cfg):
            a, b = data['model_pred'], data['target']
            g_joint_mask = data['cond']['global_joint_mask']
            if g_joint_mask is None:
                return torch.tensor(0.0).to(self.device), {}
            
            a_global = self.get_global_position(a)
            b_global = self.get_global_position(b)
            diff = a_global - b_global
            diff = diff[:, :66, 0].transpose(1, 2)
            diff *= g_joint_mask
            loss = diff ** 2
            loss = sum_flat(loss)
            non_zero_elements = sum_flat(g_joint_mask)
            non_zero_elements[non_zero_elements == 0] = 1
            loss = loss / non_zero_elements
            loss = (loss * t_weights).mean()
            return loss, {}
        
        def masked_local_root_l2(cfg):
            mask = data['mask']
            root_motion_pred, root_motion_target = data['model_pred'][:, :5], data['target'][:, :5]     # this is for global_root_local_joints
            root_motion_pred = self.denoiser.convert_root_global_to_local(root_motion_pred)
            root_motion_target = self.denoiser.convert_root_global_to_local(root_motion_target)
            loss = (root_motion_pred - root_motion_target) ** 2
            loss = sum_flat(loss * mask.float())
            n_entries = root_motion_pred.shape[1] * root_motion_pred.shape[2]
            non_zero_elements = sum_flat(mask) * n_entries
            non_zero_elements[non_zero_elements == 0] = 1
            loss = loss / non_zero_elements
            loss = (loss * t_weights).mean()
            return loss, {}
        
        def foot_sliding(cfg):
            fid = [7, 10, 8, 11]
            a, b = data['model_pred'], data['target']
            mask = data['mask']
            a_global = self.get_global_position(a)
            contact_label = data['target'][:, -4:, 0, :]
            contact_label = contact_label * self.humanml_std[-4:, None].to(self.device) + self.humanml_mean[-4:, None].to(self.device)
            contact_label = contact_label.transpose(1, 2)
            foot_m = a_global[:, :66, 0].transpose(1, 2)
            foot_m = foot_m.view(foot_m.shape[:2] + (-1, 3))
            foot_m = foot_m[:, :, fid]
            foot_vel = ((foot_m[:, 1:] - foot_m[:, :-1]) ** 2).sum(dim=-1)
            foot_vel = (foot_vel * contact_label[:, :-1]).sum(dim=-1)
            loss = (foot_vel * mask[:, 0, 0, :-1]).sum(dim=-1)
            non_zero_elements = sum_flat(mask)
            non_zero_elements[non_zero_elements == 0] = 1
            loss = loss / non_zero_elements
            if cfg.get('exp_weighting', False):
                exp_alpha = cfg.get('exp_alpha', 0.01)
                weight = torch.exp(-exp_alpha * t)
                loss *= weight
            loss = (loss * t_weights).mean()
            return loss, {}
        
        def target_frame_l2(cfg):
            if data['cond']['unknownt_motion_mask'] is None:
                return torch.tensor(0.0).to(self.device), {}
            part = cfg.get('part', 'all')
            if part == 'all':
                ind = None
            elif part == 'root':
                ind = slice(0, self.motion_root_dim)
            elif part == 'body':
                ind = slice(self.motion_root_dim, None)
            a, b = data['model_pred'], data['target']
            selected_keyframe_t = data['selected_keyframe_t']
            if ind is not None:
                a, b = a[:, ind], b[:, ind]
            diff = (a - b) ** 2
            diff_target = torch.gather(diff, 3, selected_keyframe_t[:, None, None, None].expand_as(diff[..., [0]])).squeeze(-1)
            loss = diff_target.mean(dim=(1, 2))
            loss = (loss * t_weights).mean()
            return loss, {}

        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        loss_cfg_dict = self.cfg.get('loss', {})
        for loss_name, loss_cfg in loss_cfg_dict.items():
            loss_func = locals()[loss_cfg.get('func', loss_name)]
            loss_unweighted, info = loss_func(loss_cfg)
            skip = info.get('skip', False)
            if skip:
                continue
            loss = loss_unweighted * loss_cfg.get('weight', 1.0)
            monitor_only = loss_cfg.get('monitor_only', False)
            if not monitor_only:
                total_loss += loss
            loss_dict[loss_name] = loss
            loss_unweighted_dict[loss_name] = loss_unweighted

        return total_loss, loss_dict, loss_unweighted_dict

    def configure_optimizers(self):
        optimizer_cfg = self.cfg.train.optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay)
        scheduler_cfg = self.cfg.train.get('scheduler', None) 
        if scheduler_cfg is not None:
            type = scheduler_cfg.pop('type')
            scheduler = import_type_from_str(type)(optimizer, **scheduler_cfg)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def infer_texts_guided(self, texts, num_frames, target_motion, motion_mask=None, observed_motion=None, rm_text_flag=None, 
                            global_motion=None, global_joint_mask=None, global_joint_func=None, 
                            unknownt_motion_mask=None, unknownt_observed_motion=None,
                            guide=None, progress=True):
        diffusion = self.test_diffusion
        batch_size = len(texts)
        _, cond = collate(
            [{'inp': torch.tensor([[0.]]), 'target': 0, 'text': txt, 'tokens': None, 'lengths': num_frames} for txt in texts]
        )
        cond = tensor_to(cond, device=self.device)
        if motion_mask is not None and observed_motion is not None:
            cond['motion_mask'], cond['observed_motion'], cond['rm_text_flag'] = tensor_to([motion_mask, observed_motion, rm_text_flag], device=self.device)
        if global_motion is not None and global_joint_mask is not None:
            cond['global_motion'], cond['global_joint_mask'] = tensor_to([global_motion, global_joint_mask], device=self.device)
            cond['global_joint_func'] = global_joint_func
        if unknownt_motion_mask is not None and unknownt_observed_motion is not None:
            cond['unknownt_motion_mask'], cond['unknownt_observed_motion'] = tensor_to([unknownt_motion_mask, unknownt_observed_motion], device=self.device)

        denoiser = self.guided_denoiser
        cond['y']['scale'] = torch.ones(batch_size, device=self.device) * self.cfg.model.diffusion.guidance_param
        
        diff_sampler = self.cfg.model.diffusion.get('sampler', 'ddim')
        if diff_sampler == 'ddim':
            sample_fn = diffusion.ddim_sample_loop
            kwargs = {'eta': self.cfg.model.diffusion.ddim_eta, 
                      'guide': guide, 
                      'target_motion': target_motion}
        else:
            sample_fn = diffusion.p_sample_loop
            kwargs = {}

        repeat_final_timesteps = self.cfg.model.diffusion.get('repeat_final_timesteps', None)
        if repeat_final_timesteps is not None:
            def model_kwargs_modify_fn(model_kwargs, sample, t, is_final_repeat_timestep):
                if is_final_repeat_timestep:
                    model_kwargs = model_kwargs.copy()
                    model_kwargs['fixed_root_input'] = sample[:, :self.motion_root_dim]
                return model_kwargs
            def update_sample_fn(sample, diffusion_out, t, is_final_repeat_timestep):
                new_sample = diffusion_out['sample']
                if is_final_repeat_timestep:
                    new_sample[:, :self.motion_root_dim] = sample[:, :self.motion_root_dim]
                return new_sample
            kwargs['repeat_final_timesteps'] = repeat_final_timesteps
            kwargs['model_kwargs_modify_fn'] = model_kwargs_modify_fn
            kwargs['update_sample_fn'] = update_sample_fn

        samples = sample_fn(
            denoiser,
            (batch_size, self.denoiser.njoints, self.denoiser.nfeats, num_frames),
            clip_denoised=False,
            model_kwargs=cond,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=progress,
            dump_steps=None,
            noise=None,
            const_noise=False,
            **kwargs
        )
        return samples

    def infer_texts(self, texts, num_frames, motion_mask=None, observed_motion=None, rm_text_flag=None, global_motion=None, global_joint_mask=None, global_joint_func=None, unknownt_motion_mask=None, unknownt_observed_motion=None, progress=True):
        diffusion = self.test_diffusion
        batch_size = len(texts)
        _, cond = collate(
            [{'inp': torch.tensor([[0.]]), 'target': 0, 'text': txt, 'tokens': None, 'lengths': num_frames} for txt in texts]
        )
        cond = tensor_to(cond, device=self.device)
        if motion_mask is not None and observed_motion is not None:
            cond['motion_mask'], cond['observed_motion'], cond['rm_text_flag'] = tensor_to([motion_mask, observed_motion, rm_text_flag], device=self.device)
        if global_motion is not None and global_joint_mask is not None:
            cond['global_motion'], cond['global_joint_mask'] = tensor_to([global_motion, global_joint_mask], device=self.device)
            cond['global_joint_func'] = global_joint_func
        if unknownt_motion_mask is not None and unknownt_observed_motion is not None:
            cond['unknownt_motion_mask'], cond['unknownt_observed_motion'] = tensor_to([unknownt_motion_mask, unknownt_observed_motion], device=self.device)

        denoiser = self.guided_denoiser
        cond['y']['scale'] = torch.ones(batch_size, device=self.device) * self.cfg.model.diffusion.guidance_param
        
        diff_sampler = self.cfg.model.diffusion.get('sampler', 'ddim')
        if diff_sampler == 'ddim':
            sample_fn = diffusion.ddim_sample_loop
            kwargs = {'eta': self.cfg.model.diffusion.ddim_eta}
        else:
            sample_fn = diffusion.p_sample_loop
            kwargs = {}

        repeat_final_timesteps = self.cfg.model.diffusion.get('repeat_final_timesteps', None)
        separate_root_joint_pred = self.model_cfg.get('separate_root_joint_pred', False)
        if repeat_final_timesteps is not None:
            def model_kwargs_modify_fn(model_kwargs, sample, t, is_final_repeat_timestep):
                if is_final_repeat_timestep:
                    model_kwargs = model_kwargs.copy()
                    model_kwargs['fixed_root_input'] = sample[:, :self.motion_root_dim]
                return model_kwargs
            def update_sample_fn(sample, diffusion_out, t, is_final_repeat_timestep, before_repeat_timesteps, sample_start):
                new_sample = diffusion_out['sample']
                if is_final_repeat_timestep:
                    new_sample[:, :self.motion_root_dim] = sample[:, :self.motion_root_dim]
                if before_repeat_timesteps and separate_root_joint_pred:
                    new_sample[:, self.motion_root_dim:] = sample_start[:, self.motion_root_dim:]   # reset joints back to the starting diffusion noise
                return new_sample
            kwargs['repeat_final_timesteps'] = repeat_final_timesteps
            kwargs['model_kwargs_modify_fn'] = model_kwargs_modify_fn
            kwargs['update_sample_fn'] = update_sample_fn

        samples = sample_fn(
            denoiser,
            (batch_size, self.denoiser.njoints, self.denoiser.nfeats, num_frames),
            clip_denoised=False,
            model_kwargs=cond,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=progress,
            dump_steps=None,
            noise=None,
            const_noise=False,
            **kwargs
        )
        return samples
    
    def obtain_full263_motion(self, samples, gt_motion=None):
        if self.motion_rep == 'position':
            motion_pad = gt_motion[:, 67:] if gt_motion is not None else torch.zeros((samples.shape[0], 263 - samples.shape[1], *samples.shape[2:]))
            motion_pad = motion_pad.to(samples.device)
            new_samples = torch.cat([samples, motion_pad], dim=1)
        else:
            new_samples = samples
        return new_samples
    
    def obtain_joints_and_smpl_pose(self, samples, smpl, infer_kwargs=None, interp=True, return_contacts=False):
        samples = self.obtain_full263_motion(samples)
        if self.motion_rep == 'global_position':
            smpl_pose = smpl_trans = None
            joints_pos = samples.squeeze(2).permute(0, 2, 1)     # needs interpolation
            if interp:
                joints_pos = interp_tensor_with_scipy(joints_pos, scale=1.5, dim=1)
            joints_pos = joints_pos.reshape(joints_pos.shape[:-1] + (-1, 3))
        elif self.motion_rep == 'global_root_local_joints':
            smpl_pose = smpl_trans = None
            base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=self.device, dtype=samples.dtype)
            samples = samples.permute(0, 2, 3, 1) # now permulted to: [batch, 1, seq_len, nfeat]
            root_samp = samples[..., :5]
            if self.normalize_global_pos:
                root_samp = root_samp * self.humanml_global_std.to(self.device) + self.humanml_global_mean.to(self.device)
            r_pos, rot_cos_sin, local_feats = root_samp[..., :3], root_samp[..., 3:5], samples[..., 5:]
            rot_cos_sin = normalize(rot_cos_sin)
            r_rot_quat = torch.cat([rot_cos_sin[..., [0]], torch.zeros_like(rot_cos_sin[..., [0]]), rot_cos_sin[..., [1]], torch.zeros_like(rot_cos_sin[..., [0]])], dim=-1)

            local_feats_norm = local_feats * self.humanml_std[4:].to(self.device) + self.humanml_mean[4:].to(self.device) # [batch, 1, seq_len, nfeat]
            local_feats_norm_pad = torch.cat([torch.zeros_like(local_feats_norm[..., :4]), local_feats_norm], dim=-1)
            
            joints_pos = recover_from_ric(local_feats_norm_pad, 22, r_rot_quat, r_pos)[:, 0]
            hand_joints = joints_pos[..., -2:, :].clone()
            hand_joints[..., 1] += 1e-2
            joints_pos = torch.cat([joints_pos, hand_joints], dim=-2)
            if interp:
                joints_pos = interp_tensor_with_scipy(joints_pos, scale=1.5, dim=1)
            smpl_pose, smpl_trans = joints_to_smpl(joints_pos, smpl, self.device)
            joints_pos = quat_apply(base_rot.expand(joints_pos.shape[:-1] + (4,)), joints_pos)
        elif self.motion_rep == 'global_root_vel_local_joints':
            smpl_pose = smpl_trans = None
            base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=self.device, dtype=samples.dtype)
            samples = samples.permute(0, 2, 3, 1)
            root_samp = samples[..., :4]
            if self.normalize_global_pos:
                root_samp = root_samp * self.humanml_global_std.to(self.device) + self.humanml_global_mean.to(self.device)
            pos_v_xy, pos_y, ang_v, local_feats = root_samp[..., :2], root_samp[..., [2]], root_samp[..., [3]], samples[..., 4:]

            r_rot_ang = torch.zeros_like(ang_v).to(self.device)
            r_rot_ang[:, :, 1:] = ang_v[:, :, :-1]
            r_rot_ang = r_rot_ang.cumsum(dim=2)
            r_rot_quat = torch.cat([torch.cos(r_rot_ang), torch.zeros_like(r_rot_ang), torch.sin(r_rot_ang), torch.zeros_like(r_rot_ang)], dim=-1)

            r_pos = torch.zeros(pos_v_xy.shape[:-1] + (3,)).to(self.device)
            r_pos[..., 1:, [0, 2]] = pos_v_xy[..., :-1, :]
            r_pos = torch.cumsum(r_pos, dim=2)
            r_pos[..., [1]] = pos_y

            local_feats_norm = local_feats * self.humanml_std[4:].to(self.device) + self.humanml_mean[4:].to(self.device) # [batch, 1, seq_len, nfeat]
            local_feats_norm_pad = torch.cat([torch.zeros_like(local_feats_norm[..., :4]), local_feats_norm], dim=-1)
            
            joints_pos = recover_from_ric(local_feats_norm_pad, 22, r_rot_quat, r_pos)[:, 0]
            hand_joints = joints_pos[..., -2:, :].clone()
            hand_joints[..., 1] += 1e-2
            joints_pos = torch.cat([joints_pos, hand_joints], dim=-2)
            joints_pos = quat_apply(base_rot.expand(joints_pos.shape[:-1] + (4,)), joints_pos)
            if interp:
                joints_pos = interp_tensor_with_scipy(joints_pos, scale=1.5, dim=1)
        elif self.motion_rep == 'global_root':
            body_model = self.ext_models['body_model']
            if body_model.device != self.device:
                body_model = body_model.to(self.device)
            root_motion_local = self.convert_root_global_to_local(samples)
            observed_motion = torch.zeros((samples.shape[0], body_model.motion_rep_dim, *samples.shape[2:]), device=samples.device)
            observed_motion[:, :root_motion_local.shape[1]] = root_motion_local
            res = body_model.generate_motion_mask(body_model.model_cfg.motion_mask, observed_motion, samples.shape[-1], use_mask_type='root_traj')
            res['rm_text_flag'][:] = 0.0
            body_infer_kwargs = infer_kwargs.copy()
            for key in ['motion_mask', 'observed_motion', 'rm_text_flag']:
                body_infer_kwargs[key] = res[key]
            body_samples = body_model.infer_texts(**body_infer_kwargs)
            joints_pos, smpl_pose, smpl_trans = body_model.obtain_joints_and_smpl_pose(body_samples, smpl, infer_kwargs=body_infer_kwargs, interp=interp)
        else:
            smpl_pose, smpl_trans, joints_pos = humanml_to_smpl(samples, mean=self.humanml_mean.to(self.device), std=self.humanml_std.to(self.device), smpl=smpl)

        if return_contacts and self.motion_rep in {'global_root_local_joints', 'full263'}:
            print(samples.size())
            if not self.motion_rep in {'global_position', 'global_root_local_joints'}:
                samples = samples.permute(0, 2, 3, 1) # now permulted to: [batch, 1, seq_len, nfeat]
            foot_contacts = samples[:, 0, :, -4:] # [batch, seq_len, 4] where 4 is ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")
            print(foot_contacts.size())
            # should be between 0 and 1 unnormalized
            foot_contacts_norm = foot_contacts * self.humanml_std[-4:].to(samples.device) + self.humanml_mean[-4:].to(samples.device)
            if interp:
                foot_contacts_norm = interp_tensor_with_scipy(foot_contacts_norm, scale=1.5, dim=1)
            contacts = foot_contacts_norm > 0.5
        else:
            contacts = None 

        if return_contacts:
            return joints_pos, smpl_pose, smpl_trans, contacts
        else:
            return joints_pos, smpl_pose, smpl_trans

    def validate_loss(self, batch, batch_idx):
        with torch.no_grad():
            training = self.training
            self.train()
            data = {}
            motion, cond = batch
            batch_size = motion.shape[0]
            if motion.device != self.device:
                motion, cond = tensor_to([motion, cond], device=self.device)
            motion = self.convert_motion_rep(motion)

            data['motion'], data['cond'] = motion, cond
            data['mask'] = cond['y']['mask']

            if 'motion_mask' in self.model_cfg:
                res = self.generate_motion_mask(self.model_cfg.motion_mask, data['motion'], data['cond']['y']['lengths'])
                for key in ['motion_mask', 'observed_motion', 'rm_text_flag', 'global_motion', 'global_joint_mask', 'global_joint_func', 'unknownt_observed_motion', 'unknownt_motion_mask']:
                    data['cond'][key] = res[key]
                if 'selected_keyframe_t' in res:
                    data['selected_keyframe_t'] = res['selected_keyframe_t']

            t, t_weights = self.schedule_sampler.sample(motion.shape[0], self.device)
            data = self.get_diffusion_pred_target(data, t)
            loss, loss_dict, loss_uw_dict = self.compute_loss(data, t, t_weights)
            self.train(training)
        return loss, loss_uw_dict, batch_size
    
    def should_validate_batch(self):
        if self.transform_root_traj:
            return False
        if self.motion_rep in ['full263', 'position']:
            return True
        return False

    def convert_root_global_to_local(self, root_motion):
        root_motion = root_motion.permute(0, 2, 3, 1)
        if self.normalize_global_pos:
            root_motion = root_motion * self.humanml_global_std.to(root_motion.device) + self.humanml_global_mean.to(root_motion.device)
        r_pos, rot_cos, rot_sin = root_motion[..., :3], root_motion[..., 3], root_motion[..., 4]
        r_rot_quat = torch.stack([rot_cos, torch.zeros_like(rot_cos), rot_sin, torch.zeros_like(rot_cos)], dim=-1)
        r_pos_y = r_pos[..., [1]].clone()
        r_pos = r_pos[:, :, 1:] - r_pos[:, :, :-1]
        r_pos = torch.cat([r_pos, r_pos[:, :, [-1]]], dim=2)
        r_pos = qrot(r_rot_quat, r_pos)
        r_rot_ang = torch.atan2(rot_sin, rot_cos).unsqueeze(-1)
        ang_v = r_rot_ang[:, :, 1:] - r_rot_ang[:, :, :-1]
        ang_v[ang_v > np.pi] -= 2 * np.pi
        ang_v[ang_v < -np.pi] += 2 * np.pi
        ang_v = torch.cat([ang_v, ang_v[:, :, [-1]]], dim=2)
        local_motion = torch.cat([ang_v, r_pos[..., [0, 2]], r_pos_y], dim=-1)
        local_motion_norm = (local_motion - self.humanml_mean[:4].to(local_motion.device)) / self.humanml_std[:4].to(local_motion.device)
        local_motion_norm = local_motion_norm.permute(0, 3, 1, 2)
        return local_motion_norm
