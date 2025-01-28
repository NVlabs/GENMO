import pickle
import time
from collections.abc import Iterable

import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm

from motiondiff.data_pipeline.humanml.common.quaternion import *
from motiondiff.data_pipeline.tensors import collate
from motiondiff.data_pipeline.utils.ego2global import *
from motiondiff.diffusion.gaussian_diffusion import ModelMeanType
from motiondiff.diffusion.nn import sum_flat
from motiondiff.diffusion.resample import create_named_schedule_sampler
from motiondiff.models.common.cfg_sampler import ClassifierFreeSampleModel
from motiondiff.models.common.smpl import SMPL, SMPL_BONE_ORDER_NAMES
from motiondiff.models.mdm.rotation_conversions import matrix_to_rotation_6d
from motiondiff.models.model_util import create_gaussian_diffusion
from motiondiff.utils.scheduler import update_scheduled_params
from motiondiff.utils.tools import import_type_from_str, wandb_run_exists
from motiondiff.utils.torch_transform import normalize, quat_apply
from motiondiff.utils.torch_utils import interp_tensor_with_scipy, tensor_to

motion_rep_dims = {
    "egosmpl_v1": 147,
    "egosmpl_v2": 147,
    "egosmpl_v3": 147,
    "egosmpl_v1_contact": 151,
    "egosmpl_v2_contact": 151,
    "egosmpl_v3_contact": 151,
    "egosmpl_v4_contact": 202,
    "egosmpl_v5_contact": 248,  # v4 with normed_kpt2d + cam2world
    "egosmpl_v6_contact": 248,  # v4 with normed_kpt2d + cam_vel
    "egosmpl_v8_contact": 257,  # v4 with normed_kpt2d + cam_vel + local_rt
    "egosmpl_v10_contact": 268,  # transl_vel + (24 + 17) * 3 + 23 * 6 + 4
    "egosmpl_v11_contact": 323,  # transl_vel + (24 + 17) * 3 + 23 * 6 + 4
    "egosmpl_v12_contact": 329,  # transl_vel + orient_6d + (24 + 17) * 3 + 23 * 6 + normed_kpt2d + cam_vel + local_rt + 4
    "egosmpl_v13_contact": 320,  # transl_vel + orient_6d + (24 + 17) * 3 + 23 * 6 + normed_kpt2d + cam_vel + 4
}

motion_rep_root_dims = {
    "egosmpl_v1": 9,
    "egosmpl_v2": 9,
    "egosmpl_v3": 9,
    "egosmpl_v1_contact": 9,
    "egosmpl_v2_contact": 9,
    "egosmpl_v3_contact": 9,
    "egosmpl_v4_contact": 9,
    "egosmpl_v5_contact": 9,
    "egosmpl_v6_contact": 9,
    "egosmpl_v8_contact": 9,
    "egosmpl_v10_contact": 3,
    "egosmpl_v11_contact": 3,
    "egosmpl_v12_contact": 9,
    "egosmpl_v13_contact": 9,
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
        self.motion_rep = cfg.model.get("motion_rep", "egosmpl_v1")
        self.rep_version = self.motion_rep.split("egosmpl_")[-1]

        self.motion_rep_dim = motion_rep_dims[self.motion_rep]
        self.motion_root_dim = motion_rep_root_dims[self.motion_rep]
        self.motion_localjoints_dim = 63
        self.normalize_global_pos = cfg.model.get("normalize_global_pos", False)
        self.global_pos_z_up = cfg.model.get("global_pos_z_up", True)
        # self.transform_root_traj = self.model_cfg.get('transform_root_traj', False)
        # self.humanml_root_stats_file  = cfg.model.get('humanml_root_stats_file', 'data/stats/HumanML3D_global_pos_stats.npy')
        self.model_cfg.denoiser.njoints = self.motion_rep_dim
        self.model_cfg.denoiser.normalize_global_pos = self.normalize_global_pos
        self.motion_mean = torch.tensor(
            np.load(f"dataset/EgoSMPL3D/Mean_{self.rep_version}.npy")
        ).float()
        self.motion_std = torch.tensor(
            np.load(f"dataset/EgoSMPL3D/Std_{self.rep_version}.npy")
        ).float()
        self.motion_std[self.motion_std < 0.1] = 0.1
        self.mean_kpt = torch.tensor(
            np.load("dataset/EgoSMPL3D/Mean_normed_kpt2d.npy")
        ).float()
        self.mean_cam = torch.tensor(
            np.load("dataset/EgoSMPL3D/Mean_cam_vel.npy")
        ).float()
        self.std_kpt = torch.tensor(
            np.load("dataset/EgoSMPL3D/Std_normed_kpt2d.npy")
        ).float()
        self.std_cam = torch.tensor(
            np.load("dataset/EgoSMPL3D/Std_cam_vel.npy")
        ).float()
        self.std_local_rt = torch.tensor(
            np.load("dataset/EgoSMPL3D/Std_local_rt.npy")
        ).float()
        self.mean_local_rt = torch.tensor(
            np.load("dataset/EgoSMPL3D/Mean_local_rt.npy")
        ).float()
        self.std_cam[self.std_cam < 0.1] = 0.1

        self.motion2global = eval(f"motion2global_{self.rep_version}")
        self.motion2root = eval(f"motion2root_{self.rep_version}")
        self.smpl2motion = eval(f"smpl2motion_{self.rep_version}")

        if not is_inference:
            self.load_aug_text_dict()
        self.load_ext_models()

        self.smpl = SMPL(
            self.cfg.smpl_model_dir, create_transl=False, gender="neutral"
        ).to(self.device)
        return

    def load_pretrain_checkpoint(self):
        if "pretrained_checkpoint" in self.model_cfg:
            cp_cfg = self.model_cfg.pretrained_checkpoint
            state_dict = torch.load(cp_cfg.path, map_location="cpu")["state_dict"]
            filter_keys = cp_cfg.get("filter_keys", [])
            try_load = cp_cfg.get("try_load", False)
            if len(filter_keys) > 0:
                print(f"Filtering checkpoint keys: {filter_keys}")
                skipped_keys = [
                    k for k in state_dict.keys() if any(key in k for key in filter_keys)
                ]
                print(f"Skipped keys: {skipped_keys}")
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if not any(key in k for key in filter_keys)
                }
            if try_load:
                model_state = self.state_dict()
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if k in model_state and v.size() == model_state[k].size()
                }

            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=cp_cfg.get("strict", True)
            )
            if len(missing_keys) > 0:
                print(f"Missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"Unexpected keys: {unexpected_keys}")

    def load_ext_models(self):
        self.ext_models = {}
        em_cfg = self.model_cfg.get("ext_models", {})
        for name, cfg in em_cfg.items():
            em_cfg = import_type_from_str(cfg.config.type)(**cfg.config.args)
            em = import_type_from_str(em_cfg.model.type)(
                em_cfg, is_inference=True, preload_checkpoint=False
            )
            checkpoint = torch.load(cfg.checkpoint, map_location="cpu")["state_dict"]
            em.load_state_dict(checkpoint)
            em.eval()
            self.ext_models[name] = em

    def to(self, device):
        super().to(device)
        for key in self.ext_models:
            self.ext_models[key].to(device)
        return

    def load_aug_text_dict(self):
        self.augment_text = self.model_cfg.get("augment_text", False)
        if self.augment_text:
            assert "aug_text_file" in self.model_cfg
            self.aug_text_dict = pickle.load(open(self.model_cfg.aug_text_file, "rb"))

    def init_diffusion(self):
        self.train_diffusion = create_gaussian_diffusion(
            self.model_cfg.diffusion, training=True
        )
        self.test_diffusion = create_gaussian_diffusion(
            self.model_cfg.diffusion, training=False
        )
        self.schedule_sampler = create_named_schedule_sampler(
            self.model_cfg.diffusion.schedule_sampler_type, self.train_diffusion
        )
        self.guided_denoiser = ClassifierFreeSampleModel(self.denoiser)
        return

    def get_diffusion_pred_target(self, data, t, noise=None):
        diffusion = self.train_diffusion if self.training else self.test_diffusion

        x_start = data["motion"]
        denoiser_kwargs = data["cond"]
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = diffusion.q_sample(x_start, t, noise=noise)

        # sched_samp_repeat_timesteps_only = self.cfg.train.get('sched_samp_repeat_timesteps_only', False)
        # repeat_final_timesteps = self.cfg.model.diffusion.get('repeat_final_timesteps', None)
        # if repeat_final_timesteps is None:
        #     num_repeat_steps = 0
        # elif '%' in repeat_final_timesteps:
        #     num_repeat_steps = int(int(repeat_final_timesteps.replace('%', '')) / 100 * diffusion.num_timesteps)
        # else:
        #     num_repeat_steps = int(repeat_final_timesteps)

        # sched_samp_prob_root = self.cfg.train.get('sched_samp_prob_root', 0.0)
        # sched_samp_prob_joints = self.cfg.train.get('sched_samp_prob_joints', 0.0)
        # if sched_samp_prob_root > 0.0 or sched_samp_prob_joints > 0.0:
        #     using_gt_root_cond = torch.bernoulli(torch.ones_like(t) * sched_samp_prob_root).bool()
        #     using_gt_joints_cond = torch.bernoulli(torch.ones_like(t) * sched_samp_prob_joints).bool()
        #     if sched_samp_repeat_timesteps_only:
        #         using_gt_root_cond[t >= num_repeat_steps] = 0
        #         using_gt_joints_cond[t >= num_repeat_steps] = 0
        #     denoiser_kwargs['using_gt_root_cond'] = using_gt_root_cond
        #     denoiser_kwargs['using_gt_joints_cond'] = using_gt_joints_cond
        #     denoiser_kwargs['gt_motion'] = x_start
        if "kpt2d" in self.cfg.model.denoiser.cond_mode and False:
            data["model_pred"], out_aux = self.denoiser(
                x_t, diffusion._scale_timesteps(t), return_aux=True, **denoiser_kwargs
            )

            data["pred_local_orient"] = out_aux["pred_local_orient"]
            data["cond_mask_2d"] = out_aux["cond_mask_2d"]
        else:
            data["model_pred"] = self.denoiser(
                x_t, diffusion._scale_timesteps(t), return_aux=False, **denoiser_kwargs
            )

        if diffusion.model_mean_type == ModelMeanType.PREVIOUS_X:
            data["target"] = diffusion.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0]
        elif diffusion.model_mean_type == ModelMeanType.START_X:
            data["target"] = x_start
        elif diffusion.model_mean_type == ModelMeanType.EPSILON:
            data["target"] = noise
        else:
            raise NotImplementedError

        # learnable variance
        if self.model_cfg.diffusion.get("learnable_variance", False):
            data["vb"] = diffusion.get_vb_term(x_t, x_start, t, data["model_pred"])

        return data

    def on_save_checkpoint(self, checkpoint) -> None:
        if wandb_run_exists():
            checkpoint["wandb_run_id"] = wandb.run.id
        # exclude some keys from the checkpoint
        excluded_keys = self.denoiser.get_excluded_keys()
        all_state_keys = list(checkpoint["state_dict"].keys())
        for key in all_state_keys:
            if any([exc_key in key for exc_key in excluded_keys]):
                del checkpoint["state_dict"][key]
        return

    def augment_data_text(self, data):
        new_text_dict = []
        for text in data["cond"]["y"]["text"]:
            if text in self.aug_text_dict:
                new_text = self.aug_text_dict[text][
                    np.random.randint(len(self.aug_text_dict[text]))
                ]
                new_text_dict.append(new_text)
            else:
                new_text_dict.append(text)
        # for old, new in zip(data['cond']['y']['text'], new_text_dict):
        #     if old != new:
        #         print(f'Augmenting text: {old} -> {new}')
        #     else:
        #         print(f'Keeping old text: {old}')
        data["cond"]["y"]["text"] = new_text_dict

    def generate_motion_mask(
        self,
        motion_mask_cfg,
        motion,
        lengths,
        cond,
        use_mask_type=None,
        use_unknownt_mask_type=None,
        return_keyframes=False,
        mask_cfgs=None,
        unknownt_mask_cfg=None,
    ):
        """
        If mask_cfgs / unknownt_mask_cfg is given, uses these for configuring individual mask types rather than the base motion_mask_cfg.
        """
        comp_mask_prob = motion_mask_cfg.get("comp_mask_prob", 1.0)
        mask_comp_type = motion_mask_cfg.get("mask_comp_type", "exclusive")

        mask_probs = None
        mask_type = use_mask_type
        if "mask_probs" in motion_mask_cfg:
            mask_probs = np.array(motion_mask_cfg.mask_probs) / np.sum(
                motion_mask_cfg.mask_probs
            )
            if use_mask_type is not None:
                mask_type = use_mask_type
            else:
                mask_type = np.random.choice(motion_mask_cfg.mask_types, p=mask_probs)
        unknownt_mask_probs = None
        unknownt_mask_type = use_unknownt_mask_type
        if "unknownt_mask_probs" in motion_mask_cfg:
            unknownt_mask_probs = np.array(
                motion_mask_cfg.unknownt_mask_probs
            ) / np.sum(motion_mask_cfg.unknownt_mask_probs)
            if use_unknownt_mask_type is not None:
                unknownt_mask_type = use_unknownt_mask_type
            else:
                unknownt_mask_type = np.random.choice(
                    motion_mask_cfg.unknownt_mask_types, p=unknownt_mask_probs
                )

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
            if mode in {"root+joints", "root"}:
                if "egosmpl" in self.motion_rep:
                    root_mask_ind = slice(0, 9)
                else:
                    raise NotImplementedError
            elif mode == "root_rot":
                if "egosmpl" in self.motion_rep:
                    root_mask_ind = slice(3, 9)
                else:
                    raise NotImplementedError
            elif mode == "root_pos":
                if "egosmpl" in self.motion_rep:
                    root_mask_ind = slice(0, 3)
                else:
                    raise NotImplementedError
            elif mode == "root_pos+pose":
                if "egosmpl" in self.motion_rep:
                    root_mask_ind = np.concatenate((np.arange(0, 3), np.arange(9, 147)))
                else:
                    raise NotImplementedError
            elif mode == "root_rot+pose":
                if "egosmpl" in self.motion_rep:
                    root_mask_ind = slice(3, 147)
                else:
                    raise NotImplementedError
            elif mode == "pose":
                root_mask_ind = slice(9, 147)
            else:
                raise NotImplementedError
            return root_mask_ind

        def root_traj(_cfg):
            nonlocal motion_mask, rm_text_flag
            for i in range(motion.shape[0]):
                mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                if self.motion_rep in {"egosmpl_v1", "egosmpl_v2", "egosmpl_v3"}:
                    mode = _cfg.get("mode", "pos+rot")
                    if mode == "pos+rot":
                        motion_mask[i, :9, ..., :mlen] = 1.0
                    elif mode == "pos":
                        motion_mask[i, :3, ..., :mlen] = 1.0
                    elif mode == "rot":
                        motion_mask[i, 3:9, ..., :mlen] = 1.0
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def random_feat_mask(_cfg):
            nonlocal motion_mask, rm_text_flag, all_keyframe_idx
            obs_feat_prob = _cfg.get("obs_feat_prob", 0.1)
            feat_ind = []
            feat_ind = np.arange(9, 147)
            feat_dim = len(feat_ind)
            feat_mask = torch.from_numpy(
                np.random.binomial(
                    1,
                    obs_feat_prob,
                    size=(motion.shape[0], feat_dim, 1, motion.shape[-1]),
                )
            ).type_as(motion_mask)
            motion_mask[:, feat_ind, :, :] = feat_mask
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def keyframes(_cfg):
            nonlocal motion_mask, rm_text_flag, all_keyframe_idx
            all_keyframe_idx = _cfg.get("keyframe_idx", None)
            sample_keyframes = all_keyframe_idx is None
            if sample_keyframes:
                all_keyframe_idx = []
            mode = _cfg.get("mode", "root+joints")
            root_mask_ind = get_root_mask_indices(mode)
            for i in range(motion.shape[0]):
                if sample_keyframes:
                    mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                    num_keyframes = np.random.randint(
                        _cfg.num_range[0], min(_cfg.num_range[1], mlen) + 1
                    )
                    keyframe_idx = np.random.choice(mlen, num_keyframes, replace=False)
                    all_keyframe_idx.append(keyframe_idx)
                else:
                    keyframe_idx = all_keyframe_idx[i]
                if not mode in {"root", "root_pos+rot", "root_pos"}:
                    motion_mask[i, :3, :, keyframe_idx] = (
                        1.0  # only root + local joint positions
                    )
                    motion_mask[i, 9:-4, :, keyframe_idx] = (
                        1.0  # only root + local joint positions
                    )
                if isinstance(root_mask_ind, slice):
                    motion_mask[i, root_mask_ind, :, keyframe_idx] = 1.0
                else:
                    for rmi in root_mask_ind:
                        motion_mask[i, rmi, :, keyframe_idx] = (
                            1.0  # root_mask_ind may not be a slice
                        )
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def local_joints(_cfg):
            nonlocal motion_mask, rm_text_flag, all_keyframe_idx
            mode = _cfg.get("mode", "joints")
            root_mask_ind = get_root_mask_indices(mode)
            smpl_joint_names = SMPL_BONE_ORDER_NAMES[1:24]
            joint_names = _cfg.get("joint_names", smpl_joint_names)
            if joint_names == "all":
                joint_names = smpl_joint_names
            joint_indices = [
                smpl_joint_names.index(joint_name) for joint_name in joint_names
            ]
            all_keyframe_idx = _cfg.get("keyframe_idx", None)
            for i in range(motion.shape[0]):
                if all_keyframe_idx is None:
                    # randomly sample
                    mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                    if _cfg.num_fr_range == "all":
                        keyframe_idx = np.arange(mlen)
                    else:
                        num_keyframes = np.random.randint(
                            _cfg.num_fr_range[0], min(_cfg.num_fr_range[1], mlen) + 1
                        )
                        keyframe_idx = np.random.choice(
                            mlen, num_keyframes, replace=False
                        )
                else:
                    keyframe_idx = all_keyframe_idx[i]
                if _cfg.get(
                    "consistent_joints", False
                ):  # consistent_joints: masked joints are the same for all the frames
                    joint_mask = np.zeros(23, dtype=np.float32)
                    joint_mask[joint_indices] = np.random.binomial(
                        1, _cfg.obs_joint_prob, size=len(joint_indices)
                    )
                    motion_mask[i, 9:-4, :, keyframe_idx] = (
                        torch.from_numpy(joint_mask)
                        .repeat_interleave(6)[:, None, None]
                        .to(motion.device)
                    )
                    motion_mask[i, root_mask_ind, :, keyframe_idx] = 1.0
                else:
                    for fr in keyframe_idx:
                        joint_mask = np.zeros(23, dtype=np.float32)
                        joint_mask[joint_indices] = np.random.binomial(
                            1, _cfg.obs_joint_prob, size=len(joint_indices)
                        )
                        motion_mask[i, 9:-4, :, fr] = (
                            torch.from_numpy(joint_mask)
                            .repeat_interleave(6)
                            .to(motion.device)
                        )
                        motion_mask[i, root_mask_ind, :, fr] = 1.0
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def init_unknownt(_cfg):
            nonlocal all_keyframe_idx
            selected_keyframe_t = []
            unknownt_observed_motion = []
            all_keyframe_idx = _cfg.get("keyframe_idx", None)
            for i in range(motion.shape[0]):
                mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                t = (
                    np.random.randint(mlen)
                    if all_keyframe_idx is None
                    else all_keyframe_idx[i][0]
                )
                selected_keyframe_t.append(t)
                unknownt_observed_motion.append(motion[i, :, :, [t]])
            selected_keyframe_t = torch.from_numpy(np.array(selected_keyframe_t)).to(
                motion.device
            )
            unknownt_observed_motion = torch.stack(unknownt_observed_motion, dim=0)
            unknownt_motion_mask = torch.zeros_like(unknownt_observed_motion)
            return selected_keyframe_t, unknownt_observed_motion, unknownt_motion_mask

        def unknownt_root_traj(_cfg):
            nonlocal \
                selected_keyframe_t, \
                unknownt_observed_motion, \
                unknownt_motion_mask, \
                rm_text_flag
            selected_keyframe_t, unknownt_observed_motion, unknownt_motion_mask = (
                init_unknownt(_cfg)
            )
            xy_only = _cfg.get("xy_only", False)
            if self.motion_rep in {"full263", "position"}:
                eind = 3 if xy_only else 4
                unknownt_motion_mask[:, :eind] = 1.0
            elif self.motion_rep == "global_root_local_joints":
                mode = _cfg.get("mode", "pos+rot")
                if mode == "pos+rot":
                    root_mask_ind = np.arange(9)
                elif mode == "pos":
                    root_mask_ind = np.arange(3)
                elif mode == "pos_xy":
                    root_mask_ind = np.array([0, 1])
                elif mode == "pos_xy+rot":
                    root_mask_ind = np.array(
                        [0, 1, 3, 4, 5, 6, 7, 8]
                    )  # in the original coordinate, y is up, so we use z.
                unknownt_motion_mask[:, root_mask_ind] = 1.0
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def unknownt_keyframes(_cfg):
            nonlocal \
                selected_keyframe_t, \
                unknownt_observed_motion, \
                unknownt_motion_mask, \
                rm_text_flag
            selected_keyframe_t, unknownt_observed_motion, unknownt_motion_mask = (
                init_unknownt(_cfg)
            )
            mode = _cfg.get("mode", "root+joints")
            root_mask_ind = get_root_mask_indices(mode)
            if not mode in {"root", "root_xy+rot", "root_xy"}:
                unknownt_motion_mask[:, root_dim:-4] = (
                    1.0  # only root + local joint positions
                )
            unknownt_motion_mask[:, root_mask_ind] = 1.0
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def unknownt_local_joints(_cfg):
            nonlocal \
                selected_keyframe_t, \
                unknownt_observed_motion, \
                unknownt_motion_mask, \
                rm_text_flag
            selected_keyframe_t, unknownt_observed_motion, unknownt_motion_mask = (
                init_unknownt(_cfg)
            )
            mode = _cfg.get("mode", "joints")
            root_mask_ind = get_root_mask_indices(mode)
            smpl_joint_names = SMPL_BONE_ORDER_NAMES[1:24]
            joint_names = _cfg.get("joint_names", smpl_joint_names)
            if joint_names == "all":
                joint_names = smpl_joint_names
            joint_indices = [
                smpl_joint_names.index(joint_name) for joint_name in joint_names
            ]
            joint_mask = np.zeros(23, dtype=np.float32)
            joint_mask[joint_indices] = np.random.binomial(
                1, _cfg.obs_joint_prob, size=len(joint_indices)
            )
            unknownt_motion_mask[:, root_dim:-4] = (
                torch.from_numpy(joint_mask)
                .repeat_interleave(3)[:, None, None]
                .to(motion.device)
            )
            unknownt_motion_mask[:, root_mask_ind] = 1.0
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def plucker_kpt(_cfg):
            nonlocal motion_mask, observed_motion, rm_text_flag
            bs, nframes = motion.shape[0], motion.shape[-1]
            kpt_mask = cond["y"]["kpt_mask"].reshape(bs, nframes, 17, 1)  # [B, T, 17]
            plucker_kpt = cond["y"]["plucker_kpt"]  # [B, T, 17, 6]

            plucker_kpt = plucker_kpt * kpt_mask
            plucker_kpt = torch.cat((plucker_kpt, kpt_mask), dim=-1).reshape(
                bs, nframes, 17 * 7
            )
            plucker_kpt = plucker_kpt.permute(0, 2, 1).unsqueeze(2)  # [B, 17 * 7, 1, T]

            observed_motion = plucker_kpt.to(motion.device)
            motion_mask = (
                kpt_mask.repeat(1, 1, 1, 7)
                .reshape(bs, nframes, 17 * 7)
                .permute(0, 2, 1)
                .unsqueeze(2)
                .to(motion.device)
            )
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def v5_kpt2d_cam2world(_cfg):
            nonlocal motion_mask, rm_text_flag
            for i in range(motion.shape[0]):
                mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                # motion_mask[i, root_dim:root_dim + ljoint_dim * 3, 0, :mlen] = 1
                motion_mask[i, 198 : 198 + 37 + 9, 0, :mlen] = 1
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def v8_kpt2d_cam_vel(_cfg):
            nonlocal motion_mask, rm_text_flag
            for i in range(motion.shape[0]):
                mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                # motion_mask[i, root_dim:root_dim + ljoint_dim * 3, 0, :mlen] = 1
                motion_mask[i, 198 : 198 + 37 + 9 + 9, 0, :mlen] = 1
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def v10_kpt2d_cam_vel(_cfg):
            nonlocal motion_mask, rm_text_flag
            for i in range(motion.shape[0]):
                mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                # motion_mask[i, root_dim:root_dim + ljoint_dim * 3, 0, :mlen] = 1
                motion_mask[i, 264 : 264 + 37 + 9, 0, :mlen] = 1
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def v12_kpt2d_cam_vel(_cfg):
            nonlocal motion_mask, rm_text_flag, observed_motion
            kpt_mask_repeat = cond["y"]["kpt_mask_repeat"].transpose(1, 2)  # [B, 34, T]
            if "init_observed_motion" in cond["y"]:
                observed_motion = (
                    cond["y"]["init_observed_motion"].transpose(1, 2).unsqueeze(2)
                )
            else:
                observed_motion = None
            for i in range(motion.shape[0]):
                mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                # motion_mask[i, root_dim:root_dim + ljoint_dim * 3, 0, :mlen] = 1
                motion_mask[i, 270 : 270 + 34, 0, :mlen] = kpt_mask_repeat[i, :, :mlen]
                motion_mask[i, 270 + 34 : 270 + 37 + 9, 0, :mlen] = 1
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def kpt2d_cam_vel(_cfg):
            nonlocal motion_mask, observed_motion, rm_text_flag
            bs, nframes = motion.shape[0], motion.shape[-1]
            kpt_mask = cond["y"]["kpt_mask"].reshape(bs, nframes, 17, 1)  # [B, T, 17]
            normed_kpt2d = cond["y"]["normed_kpt2d"].reshape(
                bs, nframes, 17 * 2 + 3
            )  # [B, T, 17 * 2 + 3]
            cam_vel = cond["y"]["cam_vel"].reshape(bs, nframes, 9)  # [B, T, 9]

            observed_motion = (
                torch.cat((normed_kpt2d, cam_vel), dim=-1)
                .permute(0, 2, 1)
                .unsqueeze(2)
                .to(motion.device)
            )  # [B, 17 * 2 + 3 + 9, 1, T]

            _kpt_mask = (
                kpt_mask.repeat(1, 1, 1, 2)
                .contiguous()
                .reshape(bs, nframes, -1)
                .to(motion.device)
            )
            motion_mask = (
                torch.cat(
                    (_kpt_mask, torch.ones(bs, nframes, 3 + 9, device=motion.device)),
                    dim=-1,
                )
                .permute(0, 2, 1)
                .unsqueeze(2)
            )

            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def kpt2d_cam2world(_cfg):
            nonlocal motion_mask, observed_motion, rm_text_flag
            bs, nframes = motion.shape[0], motion.shape[-1]
            kpt_mask = cond["y"]["kpt_mask"].reshape(bs, nframes, 17, 1)  # [B, T, 17]
            normed_kpt2d = cond["y"]["normed_kpt2d"].reshape(
                bs, nframes, 17 * 2 + 3
            )  # [B, T, 17 * 2 + 3]
            cam2world = cond["y"]["cam2world"].reshape(bs, nframes, 9)  # [B, T, 9]

            observed_motion = (
                torch.cat((normed_kpt2d, cam2world), dim=-1)
                .permute(0, 2, 1)
                .unsqueeze(2)
                .to(motion.device)
            )  # [B, 17 * 2 + 3 + 9, 1, T]

            _kpt_mask = (
                kpt_mask.repeat(1, 1, 1, 2)
                .contiguous()
                .reshape(bs, nframes, -1)
                .to(motion.device)
            )
            motion_mask = (
                torch.cat(
                    (_kpt_mask, torch.ones(bs, nframes, 3 + 9, device=motion.device)),
                    dim=-1,
                )
                .permute(0, 2, 1)
                .unsqueeze(2)
            )

            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def local_smplfeat_camvel(_cfg):
            nonlocal motion_mask, observed_motion, rm_text_flag
            bs, nframes = motion.shape[0], motion.shape[-1]
            local_smplfeat = cond["y"]["local_smplfeat"].reshape(
                bs, nframes, 147
            )  # [B, T, 63]
            cam_vel = cond["y"]["cam_vel"].reshape(bs, nframes, 9)  # [B, T, 9]
            observed_motion = (
                torch.cat((local_smplfeat, cam_vel), dim=-1)
                .permute(0, 2, 1)
                .unsqueeze(2)
                .to(motion.device)
            )  # [B, 63 + 9, 1, T]
            motion_mask = torch.ones(
                bs, 147 + 9, nframes, device=motion.device
            ).unsqueeze(2)
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def local_smplfeat_cam2world(_cfg):
            nonlocal motion_mask, observed_motion, rm_text_flag
            bs, nframes = motion.shape[0], motion.shape[-1]
            local_smplfeat = cond["y"]["local_smplfeat"].reshape(bs, nframes, 147)
            cam2world = cond["y"]["cam2world"].reshape(bs, nframes, 9)  # [B, T, 9]
            observed_motion = (
                torch.cat((local_smplfeat, cam2world), dim=-1)
                .permute(0, 2, 1)
                .unsqueeze(2)
                .to(motion.device)
            )  # [B, 63 + 9, 1, T]
            motion_mask = torch.ones(
                bs, 147 + 9, nframes, device=motion.device
            ).unsqueeze(2)
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def local_smplfeat_camplucker(_cfg):
            nonlocal motion_mask, observed_motion, rm_text_flag
            bs, nframes = motion.shape[0], motion.shape[-1]
            local_smplfeat = cond["y"]["local_smplfeat"].reshape(bs, nframes, 147)
            camplucker = cond["y"]["camplucker"].reshape(
                bs, nframes, 150
            )  # [B, T, 150]
            observed_motion = (
                torch.cat((local_smplfeat, camplucker), dim=-1)
                .permute(0, 2, 1)
                .unsqueeze(2)
                .to(motion.device)
            )  # [B, 147 + 150, 1, T]
            motion_mask = torch.ones(
                bs, 147 + 150, nframes, device=motion.device
            ).unsqueeze(2)
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def global_smplfeat_cam2world(_cfg):
            nonlocal motion_mask, observed_motion, rm_text_flag
            bs, nframes = motion.shape[0], motion.shape[-1]
            global_smplfeat = cond["y"]["global_smplfeat"].reshape(bs, nframes, 147)
            cam2world = cond["y"]["cam2world"].reshape(bs, nframes, 9)  # [B, T, 9]
            observed_motion = (
                torch.cat((global_smplfeat, cam2world), dim=-1)
                .permute(0, 2, 1)
                .unsqueeze(2)
                .to(motion.device)
            )  # [B, 63 + 9, 1, T]
            motion_mask = torch.ones(
                bs, 147 + 9, nframes, device=motion.device
            ).unsqueeze(2)
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def global_kpt3d(_cfg):
            nonlocal motion_mask, observed_motion, rm_text_flag
            bs, nframes = motion.shape[0], motion.shape[-1]
            global_kpt3dfeat = cond["y"]["global_kpt3d"].reshape(bs, nframes, 51)
            # cam2world = cond['y']['cam2world'].reshape(bs, nframes, 9)    # [B, T, 9]
            observed_motion = (
                global_kpt3dfeat.permute(0, 2, 1).unsqueeze(2).to(motion.device)
            )
            # observed_motion = torch.cat((global_kpt3dfeat, cam2world), dim=-1).permute(0, 2, 1).unsqueeze(2).to(motion.device) # [B, 63 + 9, 1, T]
            motion_mask = torch.ones(bs, 51, nframes, device=motion.device).unsqueeze(2)
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def motion_v10(_cfg):
            nonlocal motion_mask, observed_motion, rm_text_flag
            motion_v10 = cond["y"]["motion_v10"].to(motion.device)
            motion_v10 = motion_v10.permute(0, 2, 1).unsqueeze(2).to(motion.device)
            motion_mask = torch.ones_like(motion_v10)
            observed_motion = motion_v10
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        enable_mask = np.random.binomial(1, comp_mask_prob)
        if mask_comp_type == "exclusive":
            enable_unknownt_mask = 1 - enable_mask
        elif mask_comp_type == "or":
            enable_unknownt_mask = np.random.binomial(
                1, motion_mask_cfg.comp_unknownt_mask_prob
            )
        else:
            raise NotImplementedError

        observed_motion = None
        if use_mask_type is not None or enable_mask:
            for mi, cur_mask_type in enumerate(mask_type):
                # motion_mask is updated in place so can aggregate over all types
                if cur_mask_type != "no_mask":
                    mask_cfg = (
                        motion_mask_cfg.get(cur_mask_type, {})
                        if mask_cfgs is None
                        else mask_cfgs[mi]
                    )
                    mask_func = locals()[mask_cfg.get("func", cur_mask_type)]
                    assert mask_func != "global_joints", (
                        "multi-masking does not support global joints right now!"
                    )
                    mask_func(mask_cfg)

        # only support a single unknown t mask type since model needs to be trained explicitly to take in more than one
        if (
            use_unknownt_mask_type is not None or enable_unknownt_mask
        ) and unknownt_mask_type != "no_mask":
            unknownt_mask_cfg = (
                motion_mask_cfg.get(unknownt_mask_type, {})
                if unknownt_mask_cfg is None
                else unknownt_mask_cfg
            )
            unknownt_mask_func = locals()[
                unknownt_mask_cfg.get("func", unknownt_mask_type)
            ]
            unknownt_mask_func(unknownt_mask_cfg)

        if use_mask_type is not None or use_unknownt_mask_type is not None:
            # remove text conditioning
            rm_text_flag = torch.ones(motion.shape[0], device=motion.device)

        if observed_motion is None:
            observed_motion = motion_mask * motion
        else:
            observed_motion = motion_mask * observed_motion
        res = {
            "motion_mask": motion_mask,
            "observed_motion": observed_motion,
            "rm_text_flag": rm_text_flag,
            # 'global_motion': global_motion,
            # 'global_joint_mask': global_joint_mask,
            # 'global_joint_func': global_joint_func,
            # 'selected_keyframe_t': selected_keyframe_t,
            # 'unknownt_observed_motion': unknownt_observed_motion,
            # 'unknownt_motion_mask': unknownt_motion_mask
        }
        if return_keyframes:
            res["all_keyframe_idx"] = all_keyframe_idx
        return res

    def transform_global_motion_for_vis(self, global_motion):
        if not self.global_pos_z_up:
            base_rot = torch.tensor(
                [[0.5, 0.5, 0.5, 0.5]], device=self.device, dtype=global_motion.dtype
            )
            g_joints = (
                global_motion[:, :, 0]
                .transpose(1, 2)
                .view(global_motion.shape[0], global_motion.shape[-1], -1, 3)
            )
            global_motion = quat_apply(
                base_rot.expand(g_joints.shape[:-1] + (4,)), g_joints
            )
            global_motion = (
                global_motion.view(global_motion.shape[:-2] + (-1,))
                .transpose(1, 2)
                .unsqueeze(2)
            )
        return global_motion

    def training_step(self, batch, batch_idx):
        schedule = self.cfg.get("schedule", dict())
        update_scheduled_params(self, schedule, self.global_step)

        data = {}
        motion, cond = batch
        if motion.device != self.device:
            motion, cond = tensor_to([motion, cond], device=self.device)

        data["motion"], data["cond"] = motion, cond
        data["mask"] = cond["y"]["mask"]
        data["local_orient"] = cond["y"]["local_orient"]
        data["valid_mask"] = cond["y"]["valid_mask"]

        data["force_motion_mask"] = cond["y"]["force_motion_mask"]
        data["init_motion_mask"] = cond["y"]["init_motion_mask"]
        data["init_observed_motion"] = (
            cond["y"]["init_observed_motion"].transpose(1, 2).unsqueeze(2)
        )

        if "motion_mask" in self.model_cfg:
            res = self.generate_motion_mask(
                self.model_cfg.motion_mask,
                data["motion"],
                data["cond"]["y"]["lengths"],
                data["cond"],
            )
            for key in ["motion_mask", "observed_motion", "rm_text_flag"]:
                data["cond"][key] = res[key]

        if self.augment_text:
            self.augment_data_text(data)

        t, t_weights = self.schedule_sampler.sample(motion.shape[0], self.device)
        data = self.get_diffusion_pred_target(data, t)
        loss, loss_dict, loss_uw_dict = self.compute_loss(data, t, t_weights)

        self.log("loss/train_all", loss, on_step=True, on_epoch=True, sync_dist=True)
        for key, val in loss_uw_dict.items():
            self.log(
                f"loss/train_{key}", val, on_step=True, on_epoch=True, sync_dist=True
            )
        return loss

    def compute_loss(self, data, t, t_weights):
        def masked_orient(cfg):
            a, b = data["pred_local_orient"], data["local_orient"]
            cond_mask_2d = data["cond_mask_2d"]
            mask = data["mask"][:, 0, 0, :, None]
            mask = mask * cond_mask_2d

            loss = (a - b) ** 2
            loss = sum_flat(loss * mask.float())
            n_entries = a.shape[2]
            non_zero_elements = sum_flat(mask) * n_entries
            non_zero_elements[non_zero_elements == 0] = 1
            loss = loss / non_zero_elements
            loss = (loss * t_weights).mean()
            return loss, {}

        def masked_l2(cfg):
            part = cfg.get("part", "all")
            if part == "all":
                ind = None
            elif part == "root":
                ind = slice(0, self.motion_root_dim)
            elif part == "body":
                ind = slice(self.motion_root_dim, None)
            a, b = data["model_pred"], data["target"]
            valid_mask = data["valid_mask"].transpose(1, 2)[:, :, None, :]
            if ind is not None:
                a, b = a[:, ind], b[:, ind]
            mask = data["mask"] * valid_mask
            loss = (a - b) ** 2
            loss = sum_flat(loss * mask.float())
            n_entries = a.shape[1] * a.shape[2] / (mask.shape[1] * mask.shape[2])
            non_zero_elements = sum_flat(mask) * n_entries
            non_zero_elements[non_zero_elements == 0] = 1
            loss = loss / non_zero_elements
            loss = (loss * t_weights).mean()
            return loss, {}

        def masked_traj(cfg):
            a, b = data["model_pred"], data["target"]
            global_orient_a, smpl_trans_a = self.motion2root(
                a,
                mean=self.motion_mean.to(self.device),
                std=self.motion_std.to(self.device),
            )
            global_orient_b, smpl_trans_b = self.motion2root(
                b,
                mean=self.motion_mean.to(self.device),
                std=self.motion_std.to(self.device),
            )
            global_orient_a = matrix_to_rotation_6d(global_orient_a)
            global_orient_b = matrix_to_rotation_6d(global_orient_b)
            # a_feat = torch.cat([global_orient_a, smpl_trans_a], dim=2)
            # b_feat = torch.cat([global_orient_b, smpl_trans_b], dim=2)

            mask = data["mask"][:, 0, 0, :, None]

            loss_orient = (global_orient_a - global_orient_b) ** 2
            loss_orient = sum_flat(loss_orient * mask.float())
            n_entries = global_orient_a.shape[2]
            non_zero_elements = sum_flat(mask) * n_entries
            non_zero_elements[non_zero_elements == 0] = 1
            loss_orient = loss_orient / non_zero_elements
            loss_orient = (loss_orient * t_weights).mean()

            loss_trans = (smpl_trans_a - smpl_trans_b).abs() ** 2
            loss_trans = sum_flat(loss_trans * mask.float())
            n_entries = smpl_trans_a.shape[2]
            non_zero_elements = sum_flat(mask) * n_entries
            non_zero_elements[non_zero_elements == 0] = 1
            loss_trans = loss_trans / non_zero_elements
            loss_trans = (loss_trans * t_weights).mean()
            loss = loss_orient + loss_trans
            return loss, {}

        def masked_kpt(cfg):
            a, b = data["model_pred"], data["target"]
            mask = data["mask"]
            loss = (a - b) ** 2
            loss = sum_flat(loss * mask.float())
            n_entries = a.shape[1] * a.shape[2]
            non_zero_elements = sum_flat(mask) * n_entries
            non_zero_elements[non_zero_elements == 0] = 1
            loss = loss / non_zero_elements
            return loss, {}

        def masked_local_root_l2(cfg):
            mask = data["mask"]
            root_motion_pred, root_motion_target = (
                data["model_pred"][:, : self.motion_root_dim],
                data["target"][:, : self.motion_root_dim],
            )  # this is for global_root_local_joints
            root_motion_pred = self.denoiser.convert_root_global_to_local(
                root_motion_pred
            )
            root_motion_target = self.denoiser.convert_root_global_to_local(
                root_motion_target
            )
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
            a, b = data["model_pred"], data["target"]
            mask = data["mask"]
            a_global = self.get_global_position(a)
            contact_label = data["target"][:, -4:, 0, :]
            contact_label = contact_label * self.motion_std[-4:, None].to(
                self.device
            ) + self.motion_mean[-4:, None].to(self.device)
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
            if cfg.get("exp_weighting", False):
                exp_alpha = cfg.get("exp_alpha", 0.01)
                weight = torch.exp(-exp_alpha * t)
                loss *= weight
            loss = (loss * t_weights).mean()
            return loss, {}

        def target_frame_l2(cfg):
            if data["cond"]["unknownt_motion_mask"] is None:
                return torch.tensor(0.0).to(self.device), {}
            part = cfg.get("part", "all")
            if part == "all":
                ind = None
            elif part == "root":
                ind = slice(0, self.motion_root_dim)
            elif part == "body":
                ind = slice(self.motion_root_dim, None)
            a, b = data["model_pred"], data["target"]
            selected_keyframe_t = data["selected_keyframe_t"]
            if ind is not None:
                a, b = a[:, ind], b[:, ind]
            diff = (a - b) ** 2
            diff_target = torch.gather(
                diff,
                3,
                selected_keyframe_t[:, None, None, None].expand_as(diff[..., [0]]),
            ).squeeze(-1)
            loss = diff_target.mean(dim=(1, 2))
            loss = (loss * t_weights).mean()
            return loss, {}

        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        loss_cfg_dict = self.cfg.get("loss", {})
        for loss_name, loss_cfg in loss_cfg_dict.items():
            loss_func = locals()[loss_cfg.get("func", loss_name)]
            loss_unweighted, info = loss_func(loss_cfg)
            skip = info.get("skip", False)
            if skip:
                continue
            loss = loss_unweighted * loss_cfg.get("weight", 1.0)
            monitor_only = loss_cfg.get("monitor_only", False)
            if not monitor_only:
                total_loss += loss
            loss_dict[loss_name] = loss
            loss_unweighted_dict[loss_name] = loss_unweighted

        return total_loss, loss_dict, loss_unweighted_dict

    def configure_optimizers(self):
        optimizer_cfg = self.cfg.train.optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=optimizer_cfg.lr,
            weight_decay=optimizer_cfg.weight_decay,
        )
        scheduler_cfg = self.cfg.train.get("scheduler", None)
        if scheduler_cfg is not None:
            type = scheduler_cfg.pop("type")
            lt_kwargs = dict(scheduler_cfg.pop("lt_kwargs", {}))
            scheduler = import_type_from_str(type)(optimizer, **scheduler_cfg)
            lt_kwargs["scheduler"] = scheduler
            return {"optimizer": optimizer, "lr_scheduler": lt_kwargs}
        else:
            return optimizer

    def infer_texts_guided(
        self,
        texts,
        num_frames,
        target_motion,
        motion_mask=None,
        observed_motion=None,
        rm_text_flag=None,
        global_motion=None,
        global_joint_mask=None,
        global_joint_func=None,
        unknownt_motion_mask=None,
        unknownt_observed_motion=None,
        guide=None,
        progress=True,
    ):
        diffusion = self.test_diffusion
        batch_size = len(texts)
        _, cond = collate(
            [
                {
                    "inp": torch.tensor([[0.0]]),
                    "target": 0,
                    "text": txt,
                    "tokens": None,
                    "lengths": num_frames,
                }
                for txt in texts
            ]
        )
        cond = tensor_to(cond, device=self.device)
        if motion_mask is not None and observed_motion is not None:
            cond["motion_mask"], cond["observed_motion"], cond["rm_text_flag"] = (
                tensor_to(
                    [motion_mask, observed_motion, rm_text_flag], device=self.device
                )
            )
        if global_motion is not None and global_joint_mask is not None:
            cond["global_motion"], cond["global_joint_mask"] = tensor_to(
                [global_motion, global_joint_mask], device=self.device
            )
            cond["global_joint_func"] = global_joint_func
        if unknownt_motion_mask is not None and unknownt_observed_motion is not None:
            cond["unknownt_motion_mask"], cond["unknownt_observed_motion"] = tensor_to(
                [unknownt_motion_mask, unknownt_observed_motion], device=self.device
            )

        denoiser = self.guided_denoiser
        cond["y"]["scale"] = (
            torch.ones(batch_size, device=self.device)
            * self.cfg.model.diffusion.guidance_param
        )

        diff_sampler = self.cfg.model.diffusion.get("sampler", "ddim")
        if diff_sampler == "ddim":
            sample_fn = diffusion.ddim_sample_loop
            kwargs = {
                "eta": self.cfg.model.diffusion.ddim_eta,
                "guide": guide,
                "target_motion": target_motion,
            }
        else:
            sample_fn = diffusion.p_sample_loop
            kwargs = {}

        repeat_final_timesteps = self.cfg.model.diffusion.get(
            "repeat_final_timesteps", None
        )
        if repeat_final_timesteps is not None:

            def model_kwargs_modify_fn(
                model_kwargs, sample, t, is_final_repeat_timestep
            ):
                if is_final_repeat_timestep:
                    model_kwargs = model_kwargs.copy()
                    model_kwargs["fixed_root_input"] = sample[:, : self.motion_root_dim]
                return model_kwargs

            def update_sample_fn(sample, diffusion_out, t, is_final_repeat_timestep):
                new_sample = diffusion_out["sample"]
                if is_final_repeat_timestep:
                    new_sample[:, : self.motion_root_dim] = sample[
                        :, : self.motion_root_dim
                    ]
                return new_sample

            kwargs["repeat_final_timesteps"] = repeat_final_timesteps
            kwargs["model_kwargs_modify_fn"] = model_kwargs_modify_fn
            kwargs["update_sample_fn"] = update_sample_fn

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
            **kwargs,
        )
        return samples

    def infer_kpt2d(
        self,
        texts,
        # normed_kpt2d,
        # kpt_mask,
        # cam_angvel,
        # cam_vel,
        # plucker_kpt,
        num_frames,
        cond_y,
        motion_mask=None,
        observed_motion=None,
        rm_text_flag=None,
        global_motion=None,
        global_joint_mask=None,
        global_joint_func=None,
        unknownt_motion_mask=None,
        unknownt_observed_motion=None,
        progress=True,
        zero_noise=True,
        guidance_only=True,
        overwrite_2d=False,
        overwrite_data=None,
    ):
        diffusion = self.test_diffusion
        diffusion.motion2global = self.motion2global
        diffusion.smpl2motion = self.smpl2motion
        diffusion.motion_mean = self.motion_mean.to(self.device)
        diffusion.motion_std = self.motion_std.to(self.device)
        diffusion.smpl = self.smpl
        batch_size = len(texts)
        _, cond = collate(
            [
                {
                    "inp": torch.tensor([[0.0]]),
                    "target": 0,
                    "text": txt,
                    "tokens": None,
                    "lengths": num_frames,
                }
                for txt in texts
            ]
        )
        for k in cond_y:
            cond["y"][k] = cond_y[k]
        cond = tensor_to(cond, device=self.device)

        if motion_mask is not None and observed_motion is not None:
            cond["motion_mask"], cond["observed_motion"], cond["rm_text_flag"] = (
                tensor_to(
                    [motion_mask, observed_motion, rm_text_flag], device=self.device
                )
            )
        if global_motion is not None and global_joint_mask is not None:
            cond["global_motion"], cond["global_joint_mask"] = tensor_to(
                [global_motion, global_joint_mask], device=self.device
            )
            cond["global_joint_func"] = global_joint_func
        if unknownt_motion_mask is not None and unknownt_observed_motion is not None:
            cond["unknownt_motion_mask"], cond["unknownt_observed_motion"] = tensor_to(
                [unknownt_motion_mask, unknownt_observed_motion], device=self.device
            )
        cond["rm_text_flag"] = rm_text_flag

        kpt_mask = cond["y"]["kpt_mask"].reshape(
            batch_size, num_frames, 17, 1
        )  # [B, T, 17]

        motion_mask_cfg = self.model_cfg.get("motion_mask", {})
        if motion_mask_cfg and "plucker_kpt" in motion_mask_cfg["mask_types"]:
            plucker_kpt = cond["y"]["plucker_kpt"]  # [B, T, 17, 6]
            plucker_kpt = plucker_kpt * kpt_mask
            plucker_kpt = torch.cat((plucker_kpt, kpt_mask), dim=-1).reshape(
                batch_size, num_frames, 17 * 7
            )
            plucker_kpt = plucker_kpt.permute(0, 2, 1).unsqueeze(2)  # [B, 17 * 7, 1, T]

            cond["observed_motion"] = plucker_kpt
            cond["motion_mask"] = (
                kpt_mask.repeat(1, 1, 1, 7)
                .reshape(batch_size, num_frames, 17 * 7)
                .permute(0, 2, 1)
                .unsqueeze(2)
            )
        elif motion_mask_cfg and "kpt2d_cam_vel" in motion_mask_cfg["mask_types"]:
            kpt_mask = kpt_mask.reshape(batch_size, num_frames, 17, 1).to(
                self.device
            )  # [B, T, 17]
            normed_kpt2d = (
                cond["y"]["normed_kpt2d"]
                .reshape(batch_size, num_frames, 17 * 2 + 3)
                .to(self.device)
            )  # [B, T, 17 * 2 + 3]
            cam_vel = (
                cond["y"]["cam_vel"].reshape(batch_size, num_frames, 9).to(self.device)
            )  # [B, T, 9]
            normed_kpt2d = normed_kpt2d.reshape(batch_size, num_frames, 17 * 2 + 3).to(
                self.device
            )  # [B, T, 17 * 2 + 3]
            cam_vel = cam_vel.reshape(batch_size, num_frames, 9).to(
                self.device
            )  # [B, T, 3]

            cond["observed_motion"] = (
                torch.cat((normed_kpt2d, cam_vel), dim=-1).permute(0, 2, 1).unsqueeze(2)
            )  # [B, 17 * 2 + 3 + 6 + 3, 1, T]
            _kpt_mask = (
                kpt_mask.repeat(1, 1, 1, 2)
                .contiguous()
                .reshape(batch_size, num_frames, -1)
            )
            cond["motion_mask"] = (
                torch.cat(
                    (
                        _kpt_mask,
                        torch.ones(batch_size, num_frames, 3 + 9, device=self.device),
                    ),
                    dim=-1,
                )
                .permute(0, 2, 1)
                .unsqueeze(2)
            )

        denoiser = self.guided_denoiser
        cond["guidance_only"] = guidance_only
        cond["y"]["scale"] = (
            torch.ones(batch_size, device=self.device)
            * self.cfg.model.diffusion.guidance_param
        )

        diff_sampler = self.cfg.model.diffusion.get("sampler", "ddim")
        if diff_sampler == "ddim":
            sample_fn = diffusion.ddim_sample_loop
            kwargs = {"eta": self.cfg.model.diffusion.ddim_eta}
        else:
            sample_fn = diffusion.p_sample_loop
            kwargs = {}

        repeat_final_timesteps = self.cfg.model.diffusion.get(
            "repeat_final_timesteps", None
        )
        separate_root_joint_pred = self.model_cfg.get("separate_root_joint_pred", False)
        if repeat_final_timesteps is not None:

            def model_kwargs_modify_fn(
                model_kwargs, sample, t, is_final_repeat_timestep
            ):
                if is_final_repeat_timestep:
                    model_kwargs = model_kwargs.copy()
                    model_kwargs["fixed_root_input"] = sample[:, : self.motion_root_dim]
                return model_kwargs

            def update_sample_fn(
                sample,
                diffusion_out,
                t,
                is_final_repeat_timestep,
                before_repeat_timesteps,
                sample_start,
            ):
                new_sample = diffusion_out["sample"]
                if is_final_repeat_timestep:
                    new_sample[:, : self.motion_root_dim] = sample[
                        :, : self.motion_root_dim
                    ]
                if before_repeat_timesteps and separate_root_joint_pred:
                    new_sample[:, self.motion_root_dim :] = sample_start[
                        :, self.motion_root_dim :
                    ]  # reset joints back to the starting diffusion noise
                return new_sample

            kwargs["repeat_final_timesteps"] = repeat_final_timesteps
            kwargs["model_kwargs_modify_fn"] = model_kwargs_modify_fn
            kwargs["update_sample_fn"] = update_sample_fn

        kwargs["overwrite_2d"] = overwrite_2d
        kwargs["overwrite_data"] = overwrite_data

        if zero_noise:
            noise = torch.zeros(
                batch_size, self.denoiser.njoints, self.denoiser.nfeats, num_frames
            ).to(self.device)
        else:
            noise = None
        samples = sample_fn(
            denoiser,
            (batch_size, self.denoiser.njoints, self.denoiser.nfeats, num_frames),
            clip_denoised=False,
            model_kwargs=cond,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=progress,
            dump_steps=None,
            noise=noise,
            const_noise=False,
            **kwargs,
        )
        return samples

    def infer_local3d(
        self,
        texts,
        num_frames,
        cond_y,
        motion_mask=None,
        observed_motion=None,
        rm_text_flag=None,
        global_motion=None,
        global_joint_mask=None,
        global_joint_func=None,
        unknownt_motion_mask=None,
        unknownt_observed_motion=None,
        progress=True,
        zero_noise=True,
        guidance_only=True,
        guide_2d=None,
    ):
        diffusion = self.test_diffusion
        diffusion.motion2global = self.motion2global
        diffusion.smpl2motion = self.smpl2motion
        diffusion.motion_mean = self.motion_mean.to(self.device)
        diffusion.motion_std = self.motion_std.to(self.device)
        diffusion.smpl = self.smpl

        batch_size = len(texts)
        _, cond = collate(
            [
                {
                    "inp": torch.tensor([[0.0]]),
                    "target": 0,
                    "text": txt,
                    "tokens": None,
                    "lengths": num_frames,
                }
                for txt in texts
            ]
        )
        for k in cond_y:
            cond["y"][k] = cond_y[k]
        cond = tensor_to(cond, device=self.device)

        if motion_mask is not None and observed_motion is not None:
            cond["motion_mask"], cond["observed_motion"], cond["rm_text_flag"] = (
                tensor_to(
                    [motion_mask, observed_motion, rm_text_flag], device=self.device
                )
            )
        if global_motion is not None and global_joint_mask is not None:
            cond["global_motion"], cond["global_joint_mask"] = tensor_to(
                [global_motion, global_joint_mask], device=self.device
            )
            cond["global_joint_func"] = global_joint_func
        if unknownt_motion_mask is not None and unknownt_observed_motion is not None:
            cond["unknownt_motion_mask"], cond["unknownt_observed_motion"] = tensor_to(
                [unknownt_motion_mask, unknownt_observed_motion], device=self.device
            )
        cond["rm_text_flag"] = rm_text_flag
        cond["guidance_only"] = guidance_only

        denoiser = self.guided_denoiser
        cond["y"]["scale"] = (
            torch.ones(batch_size, device=self.device)
            * self.cfg.model.diffusion.guidance_param
        )

        diff_sampler = self.cfg.model.diffusion.get("sampler", "ddim")
        if diff_sampler == "ddim":
            sample_fn = diffusion.ddim_sample_loop
            kwargs = {"eta": self.cfg.model.diffusion.ddim_eta}
        else:
            sample_fn = diffusion.p_sample_loop
            kwargs = {}

        repeat_final_timesteps = self.cfg.model.diffusion.get(
            "repeat_final_timesteps", None
        )
        separate_root_joint_pred = self.model_cfg.get("separate_root_joint_pred", False)
        if repeat_final_timesteps is not None:

            def model_kwargs_modify_fn(
                model_kwargs, sample, t, is_final_repeat_timestep
            ):
                if is_final_repeat_timestep:
                    model_kwargs = model_kwargs.copy()
                    model_kwargs["fixed_root_input"] = sample[:, : self.motion_root_dim]
                return model_kwargs

            def update_sample_fn(
                sample,
                diffusion_out,
                t,
                is_final_repeat_timestep,
                before_repeat_timesteps,
                sample_start,
            ):
                new_sample = diffusion_out["sample"]
                if is_final_repeat_timestep:
                    new_sample[:, : self.motion_root_dim] = sample[
                        :, : self.motion_root_dim
                    ]
                if before_repeat_timesteps and separate_root_joint_pred:
                    new_sample[:, self.motion_root_dim :] = sample_start[
                        :, self.motion_root_dim :
                    ]  # reset joints back to the starting diffusion noise
                return new_sample

            kwargs["repeat_final_timesteps"] = repeat_final_timesteps
            kwargs["model_kwargs_modify_fn"] = model_kwargs_modify_fn
            kwargs["update_sample_fn"] = update_sample_fn

        if guide_2d is not None:
            kwargs["guide_2d"] = guide_2d

        if zero_noise:
            noise = torch.zeros(
                batch_size, self.denoiser.njoints, self.denoiser.nfeats, num_frames
            ).to(self.device)
        else:
            noise = None
        samples = sample_fn(
            denoiser,
            (batch_size, self.denoiser.njoints, self.denoiser.nfeats, num_frames),
            clip_denoised=False,
            model_kwargs=cond,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=progress,
            dump_steps=None,
            noise=noise,
            const_noise=False,
            **kwargs,
        )
        return samples

    def infer_local3d_sds(
        self,
        x0,
        texts,
        num_frames,
        cond_y,
        sds_weight_type="alphas",
        motion_mask=None,
        observed_motion=None,
        rm_text_flag=None,
        global_motion=None,
        global_joint_mask=None,
        global_joint_func=None,
        unknownt_motion_mask=None,
        unknownt_observed_motion=None,
        progress=True,
        opt_steps=500,
        zero_noise=True,
        guidance_only=True,
        guide_2d=None,
    ):
        diffusion = self.test_diffusion
        diffusion.motion2global = self.motion2global
        diffusion.smpl2motion = self.smpl2motion
        diffusion.motion_mean = self.motion_mean.to(self.device)
        diffusion.motion_std = self.motion_std.to(self.device)
        diffusion.smpl = self.smpl

        batch_size = len(texts)
        _, cond = collate(
            [
                {
                    "inp": torch.tensor([[0.0]]),
                    "target": 0,
                    "text": txt,
                    "tokens": None,
                    "lengths": num_frames,
                }
                for txt in texts
            ]
        )
        for k in cond_y:
            cond["y"][k] = cond_y[k]
        cond = tensor_to(cond, device=self.device)

        if motion_mask is not None and observed_motion is not None:
            cond["motion_mask"], cond["observed_motion"], cond["rm_text_flag"] = (
                tensor_to(
                    [motion_mask, observed_motion, rm_text_flag], device=self.device
                )
            )
        if global_motion is not None and global_joint_mask is not None:
            cond["global_motion"], cond["global_joint_mask"] = tensor_to(
                [global_motion, global_joint_mask], device=self.device
            )
            cond["global_joint_func"] = global_joint_func
        if unknownt_motion_mask is not None and unknownt_observed_motion is not None:
            cond["unknownt_motion_mask"], cond["unknownt_observed_motion"] = tensor_to(
                [unknownt_motion_mask, unknownt_observed_motion], device=self.device
            )
        cond["rm_text_flag"] = rm_text_flag
        cond["guidance_only"] = guidance_only

        denoiser = self.guided_denoiser
        cond["y"]["scale"] = (
            torch.ones(batch_size, device=self.device)
            * self.cfg.model.diffusion.guidance_param
        )

        sample_fn = diffusion.ddim_sds_loop
        kwargs = {"eta": self.cfg.model.diffusion.ddim_eta}

        if guide_2d is not None:
            kwargs["guide_2d"] = guide_2d

        if zero_noise:
            noise = torch.zeros(
                batch_size, self.denoiser.njoints, self.denoiser.nfeats, num_frames
            ).to(self.device)
        else:
            noise = None
        samples = sample_fn(
            denoiser,
            x0,
            (batch_size, self.denoiser.njoints, self.denoiser.nfeats, num_frames),
            sds_weight_type=sds_weight_type,
            clip_denoised=False,
            model_kwargs=cond,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=progress,
            dump_steps=None,
            opt_steps=opt_steps,
            noise=noise,
            const_noise=False,
            **kwargs,
        )
        return samples

    def infer_texts(
        self,
        texts,
        num_frames,
        motion_mask=None,
        observed_motion=None,
        rm_text_flag=None,
        rm_kpt_flag=None,
        global_motion=None,
        global_joint_mask=None,
        global_joint_func=None,
        unknownt_motion_mask=None,
        unknownt_observed_motion=None,
        progress=True,
    ):
        diffusion = self.test_diffusion
        batch_size = len(texts)
        _, cond = collate(
            [
                {
                    "inp": torch.tensor([[0.0]]),
                    "target": 0,
                    "text": txt,
                    "tokens": None,
                    "lengths": num_frames,
                }
                for txt in texts
            ]
        )
        cond = tensor_to(cond, device=self.device)
        if motion_mask is not None and observed_motion is not None:
            cond["motion_mask"], cond["observed_motion"], cond["rm_text_flag"] = (
                tensor_to(
                    [motion_mask, observed_motion, rm_text_flag], device=self.device
                )
            )
        if global_motion is not None and global_joint_mask is not None:
            cond["global_motion"], cond["global_joint_mask"] = tensor_to(
                [global_motion, global_joint_mask], device=self.device
            )
            cond["global_joint_func"] = global_joint_func
        if unknownt_motion_mask is not None and unknownt_observed_motion is not None:
            cond["unknownt_motion_mask"], cond["unknownt_observed_motion"] = tensor_to(
                [unknownt_motion_mask, unknownt_observed_motion], device=self.device
            )
        cond["rm_kpt_flag"] = rm_kpt_flag

        denoiser = self.guided_denoiser
        cond["y"]["scale"] = (
            torch.ones(batch_size, device=self.device)
            * self.cfg.model.diffusion.guidance_param
        )

        diff_sampler = self.cfg.model.diffusion.get("sampler", "ddim")
        if diff_sampler == "ddim":
            sample_fn = diffusion.ddim_sample_loop
            kwargs = {"eta": self.cfg.model.diffusion.ddim_eta}
        else:
            sample_fn = diffusion.p_sample_loop
            kwargs = {}

        repeat_final_timesteps = self.cfg.model.diffusion.get(
            "repeat_final_timesteps", None
        )
        separate_root_joint_pred = self.model_cfg.get("separate_root_joint_pred", False)
        if repeat_final_timesteps is not None:

            def model_kwargs_modify_fn(
                model_kwargs, sample, t, is_final_repeat_timestep
            ):
                if is_final_repeat_timestep:
                    model_kwargs = model_kwargs.copy()
                    model_kwargs["fixed_root_input"] = sample[:, : self.motion_root_dim]
                return model_kwargs

            def update_sample_fn(
                sample,
                diffusion_out,
                t,
                is_final_repeat_timestep,
                before_repeat_timesteps,
                sample_start,
            ):
                new_sample = diffusion_out["sample"]
                if is_final_repeat_timestep:
                    new_sample[:, : self.motion_root_dim] = sample[
                        :, : self.motion_root_dim
                    ]
                if before_repeat_timesteps and separate_root_joint_pred:
                    new_sample[:, self.motion_root_dim :] = sample_start[
                        :, self.motion_root_dim :
                    ]  # reset joints back to the starting diffusion noise
                return new_sample

            kwargs["repeat_final_timesteps"] = repeat_final_timesteps
            kwargs["model_kwargs_modify_fn"] = model_kwargs_modify_fn
            kwargs["update_sample_fn"] = update_sample_fn

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
            **kwargs,
        )
        return samples

    def obtain_full263_motion(self, samples, gt_motion=None):
        return samples

    def obtain_joints_and_smpl_pose(
        self, samples, smpl, infer_kwargs=None, interp=True, return_contacts=False
    ):
        samples = self.obtain_full263_motion(samples)

        smpl_pose, smpl_trans, joints_pos = self.motion2global(
            samples,
            mean=self.motion_mean.to(self.device),
            std=self.motion_std.to(self.device),
            smpl=smpl,
            mean_kpt=self.mean_kpt.to(self.device),
            std_kpt=self.std_kpt.to(self.device),
            mean_cam=self.mean_cam.to(self.device),
            std_cam=self.std_cam.to(self.device),
            mean_local_rt=self.mean_local_rt.to(self.device),
            std_local_rt=self.std_local_rt.to(self.device),
        )

        if return_contacts and self.motion_rep in {
            "egosmpl_v1_contact",
            "egosmpl_v2_contact",
            "egosmpl_v3_contact",
            "egosmpl_v4_contact",
        }:
            # print(samples.size())
            # if not self.motion_rep in {'global_position', 'global_root_local_joints'}:
            #     samples = samples.permute(0, 2, 3, 1) # now permulted to: [batch, 1, seq_len, nfeat]
            # motion_raw: [batch, nfeat, 1, seq_len]
            # foot_contacts = samples[:, 0, :, -4:] # [batch, seq_len, 4] where 4 is ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")
            foot_contacts = samples[:, -4:, 0, :].permute(
                0, 2, 1
            )  # [batch, seq_len, 4] where 4 is ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")

            # print(foot_contacts.size())
            # should be between 0 and 1 unnormalized
            foot_contacts_norm = foot_contacts * self.motion_std[-4:].to(
                samples.device
            ) + self.motion_mean[-4:].to(samples.device)
            # if interp:
            #     foot_contacts_norm = interp_tensor_with_scipy(foot_contacts_norm, scale=1.5, dim=1)
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

            data["motion"], data["cond"] = motion, cond
            data["mask"] = cond["y"]["mask"]
            data["local_orient"] = cond["y"]["local_orient"]
            data["valid_mask"] = cond["y"]["valid_mask"]

            data["force_motion_mask"] = cond["y"]["force_motion_mask"]
            data["init_motion_mask"] = cond["y"]["init_motion_mask"]

            if "motion_mask" in self.model_cfg:
                res = self.generate_motion_mask(
                    self.model_cfg.motion_mask,
                    data["motion"],
                    data["cond"]["y"]["lengths"],
                    data["cond"],
                )
                # for key in ['motion_mask', 'observed_motion', 'rm_text_flag', 'global_motion', 'global_joint_mask', 'global_joint_func', 'unknownt_observed_motion', 'unknownt_motion_mask']:
                for key in ["motion_mask", "observed_motion", "rm_text_flag"]:
                    data["cond"][key] = res[key]
                if "selected_keyframe_t" in res:
                    data["selected_keyframe_t"] = res["selected_keyframe_t"]

            t, t_weights = self.schedule_sampler.sample(motion.shape[0], self.device)
            data = self.get_diffusion_pred_target(data, t)
            loss, loss_dict, loss_uw_dict = self.compute_loss(data, t, t_weights)
            self.train(training)
        return loss, loss_uw_dict, batch_size

    def should_validate_batch(self):
        if self.transform_root_traj:
            return False
        if "egosmpl" in self.motion_rep:
            return True
        return False
