import os
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

from motiondiff.data_pipeline.bones.motion_process import (
    recover_from_ric,
    recover_from_ric_with_joint_rot,
    recover_root_rot_pos,
)
from motiondiff.data_pipeline.humanml.common.quaternion import *
from motiondiff.data_pipeline.tensors import collate
from motiondiff.diffusion.gaussian_diffusion import ModelMeanType
from motiondiff.diffusion.nn import sum_flat
from motiondiff.diffusion.resample import create_named_schedule_sampler
from motiondiff.models.common.bones import BONES_BONE_ORDER_NAMES
from motiondiff.models.common.cfg_sampler import ClassifierFreeSampleModel
from motiondiff.models.model_util import create_gaussian_diffusion
from motiondiff.utils.scheduler import update_scheduled_params
from motiondiff.utils.tools import import_type_from_str, wandb_run_exists
from motiondiff.utils.torch_transform import normalize, quat_apply, quat_mul
from motiondiff.utils.torch_utils import (
    interp_tensor_with_scipy,
    slerp_joint_rots,
    tensor_to,
)

motion_rep_dims = {
    "orig": 347,
    "orig_rot": 353,
    "global_root_local_joints": 348,
    "global_root_local_joints_root_rot": 354,
}

motion_rep_root_dims = {
    "orig": 4,
    "orig_rot": 4,
    "global_root_local_joints": 5,
    "global_root_local_joints_root_rot": 5,
}


"""
Main Model
"""


class MDMBase(pl.LightningModule):
    def __init__(self, cfg, is_inference=False):
        super().__init__()
        self.cfg = cfg
        self.is_inference = is_inference
        self.num_joints = cfg.get("num_joints", 29)
        self.model_cfg = cfg.model
        self.motion_rep = cfg.model.get("motion_rep", "orig")
        self.motion_rep_dim = motion_rep_dims[self.motion_rep]
        self.motion_root_dim = motion_rep_root_dims[self.motion_rep]
        self.motion_localjoints_dim = (self.num_joints - 1) * 3  # excludes root
        self.normalize_global_pos = cfg.model.get("normalize_global_pos", False)
        self.global_pos_z_up = cfg.model.get("global_pos_z_up", True)
        self.model_cfg.denoiser.njoints = self.motion_rep_dim
        self.model_cfg.denoiser.normalize_global_pos = self.normalize_global_pos
        self.motion_mean = torch.tensor(
            np.load(f"{self.model_cfg.motion_stats_folder}/mean.npy")
        ).float()
        self.motion_std = torch.tensor(
            np.load(f"{self.model_cfg.motion_stats_folder}/std.npy")
        ).float()
        if self.normalize_global_pos:
            motion_root_stats_dir = cfg.model.get(
                "motion_root_stats_dir", "assets/bones/global_stats/v1/frames_196"
            )
            print(f"Normalizing global root with stats from {motion_root_stats_dir}...")
            self.motion_global_mean = torch.tensor(
                np.load(os.path.join(motion_root_stats_dir, "mean.npy"))
            ).float()
            self.motion_global_std = torch.tensor(
                np.load(os.path.join(motion_root_stats_dir, "std.npy"))
            ).float()
        if self.motion_rep in {"orig_rot", "global_root_local_joints_root_rot"}:
            self.neutral_joints = torch.load("assets/bones/skeleton/joints.p")
            self.joint_parents = torch.load("assets/bones/skeleton/parents.p")
        self.load_ext_models()
        return

    def load_pretrain_checkpoint(self):
        if "pretrained_checkpoint" in self.model_cfg:
            cp_cfg = self.model_cfg.pretrained_checkpoint
            state_dict = torch.load(cp_cfg.path, map_location="cpu")["state_dict"]
            filter_keys = cp_cfg.get("filter_keys", [])
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

        data["model_pred"] = self.denoiser(
            x_t, diffusion._scale_timesteps(t), **denoiser_kwargs
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

    def generate_motion_mask(
        self,
        motion_mask_cfg,
        motion,
        lengths,
        use_mask_type=None,
        return_keyframes=False,
        mask_cfgs=None,
    ):
        """
        If mask_cfgs is given, uses these for configuring individual mask types rather than the base motion_mask_cfg.
        """
        comp_mask_prob = motion_mask_cfg.get("comp_mask_prob", 1.0)

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

        if not isinstance(mask_type, list):
            mask_type = [mask_type]

        motion_mask = torch.zeros_like(motion)
        rm_text_flag = torch.zeros(motion.shape[0], device=motion.device)
        root_dim = self.motion_root_dim
        ljoint_dim = self.motion_localjoints_dim
        all_keyframe_idx = None

        def get_root_mask_indices(mode):
            if mode in {"root+joints", "root"}:
                root_mask_ind = slice(0, root_dim)
            elif mode == "root_xy+rot":
                if self.motion_rep in {"orig", "orig_rot"}:
                    root_mask_ind = slice(0, 3)
                elif self.motion_rep in {
                    "global_root_local_joints",
                    "global_root_local_joints_root_rot",
                }:
                    root_mask_ind = np.array(
                        [0, 2, 3, 4]
                    )  # in the original coordinate, y is up, so we use z.
                else:
                    raise NotImplementedError
            elif mode == "root_xy":
                if self.motion_rep in {"orig", "orig_rot"}:
                    root_mask_ind = slice(1, 3)
                elif self.motion_rep in {
                    "global_root_local_joints",
                    "global_root_local_joints_root_rot",
                }:
                    root_mask_ind = np.array(
                        [0, 2]
                    )  # in the original coordinate, y is up, so we use z.
                else:
                    raise NotImplementedError
            elif mode == "root_pos+joints":
                if self.motion_rep in {"orig", "orig_rot"}:
                    root_mask_ind = slice(1, root_dim)
                elif self.motion_rep in {
                    "global_root_local_joints",
                    "global_root_local_joints_root_rot",
                }:
                    root_mask_ind = slice(0, 3)
                else:
                    raise NotImplementedError
            elif mode == "rootheight+joints":
                if self.motion_rep in {"orig", "orig_rot"}:
                    root_mask_ind = slice(3, root_dim)
                elif self.motion_rep in {
                    "global_root_local_joints",
                    "global_root_local_joints_root_rot",
                }:
                    root_mask_ind = slice(1, 2)  # y
                else:
                    raise NotImplementedError
            elif mode == "joints":
                root_mask_ind = slice(0, 0)
            else:
                raise NotImplementedError
            return root_mask_ind

        def root_traj(_cfg):
            nonlocal motion_mask, rm_text_flag
            root_mask_ind = None
            mode = _cfg.get("mode", "pos+rot")
            if self.motion_rep in {"orig", "orig_rot"}:
                if mode == "pos+rot":
                    root_mask_ind = np.arange(root_dim)
                elif mode == "pos":
                    root_mask_ind = np.arange([1, 2, 3])
                elif mode == "pos_xy":
                    root_mask_ind = np.array([1, 2])
                elif mode == "pos_xy+rot":
                    root_mask_ind = np.arange(root_dim - 1)  # leave out root height
                else:
                    raise NotImplementedError
            elif self.motion_rep in {
                "global_root_local_joints",
                "global_root_local_joints_root_rot",
            }:
                if mode == "pos+rot":
                    root_mask_ind = np.arange(5)
                elif mode == "pos":
                    root_mask_ind = np.arange(3)
                elif mode == "pos_xy":
                    root_mask_ind = np.array([0, 2])
                elif mode == "pos_xy+rot":
                    root_mask_ind = np.array(
                        [0, 2, 3, 4]
                    )  # in the original coordinate, y is up, so we use z.
                else:
                    raise NotImplementedError

            for i in range(motion.shape[0]):
                mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                motion_mask[i, root_mask_ind, ..., :mlen] = 1.0
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
                    if _cfg.num_range == "bounds":
                        # just use first and last frame
                        keyframe_idx = np.array([0, mlen - 1])
                    else:
                        num_keyframes = np.random.randint(
                            _cfg.num_range[0], min(_cfg.num_range[1], mlen) + 1
                        )
                        keyframe_idx = np.random.choice(
                            mlen, num_keyframes, replace=False
                        )
                    all_keyframe_idx.append(keyframe_idx)
                else:
                    keyframe_idx = all_keyframe_idx[i]
                if not mode in {"root", "root_xy+rot", "root_xy"}:
                    motion_mask[
                        i, root_dim : root_dim + ljoint_dim, :, keyframe_idx
                    ] = 1.0  # only root + local joint positions
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
            bones_joint_names = BONES_BONE_ORDER_NAMES[1:]  # leave off Hips (root)
            num_body_joints = len(bones_joint_names)
            joint_names = _cfg.get("joint_names", bones_joint_names)
            if joint_names == "all":
                joint_names = bones_joint_names
            joint_indices = [
                bones_joint_names.index(joint_name) for joint_name in joint_names
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
                    joint_mask = np.zeros(num_body_joints, dtype=np.float32)
                    joint_mask[joint_indices] = np.random.binomial(
                        1, _cfg.obs_joint_prob, size=len(joint_indices)
                    )
                    motion_mask[
                        i, root_dim : root_dim + ljoint_dim, :, keyframe_idx
                    ] = (
                        torch.from_numpy(joint_mask)
                        .repeat_interleave(3)[:, None, None]
                        .to(motion.device)
                    )
                    motion_mask[i, root_mask_ind, :, keyframe_idx] = (
                        1.0  # root_mask_ind is always a slice
                    )
                else:
                    for fr in keyframe_idx:
                        joint_mask = np.zeros(num_body_joints, dtype=np.float32)
                        joint_mask[joint_indices] = np.random.binomial(
                            1, _cfg.obs_joint_prob, size=len(joint_indices)
                        )
                        motion_mask[i, root_dim : root_dim + ljoint_dim, :, fr] = (
                            torch.from_numpy(joint_mask)
                            .repeat_interleave(3)
                            .to(motion.device)
                        )
                        motion_mask[i, root_mask_ind, :, fr] = 1.0
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        enable_mask = np.random.binomial(1, comp_mask_prob)

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

        if use_mask_type is not None:
            # remove text conditioning by default is a specific mask is requested
            rm_text_flag = torch.ones(motion.shape[0], device=motion.device)

        observed_motion = motion_mask * motion
        res = {
            "motion_mask": motion_mask,
            "observed_motion": observed_motion,
            "rm_text_flag": rm_text_flag,
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

    def convert_motion_rep(self, motion):
        if self.motion_rep in {"orig", "orig_rot"}:
            motion = motion
        elif self.motion_rep in {
            "global_root_local_joints",
            "global_root_local_joints_root_rot",
        }:
            motion_norm = motion.permute(0, 2, 3, 1) * self.motion_std.to(
                self.device
            ) + self.motion_mean.to(self.device)  # [batch, 1, seq_len, nfeat]
            _, r_pos, r_rot_ang = recover_root_rot_pos(
                motion_norm, return_r_rot_ang=True
            )
            rot_cos_sin = torch.stack(
                [torch.cos(r_rot_ang), torch.sin(r_rot_ang)], dim=-1
            )
            motion = torch.cat(
                [
                    r_pos.permute(0, 3, 1, 2),
                    rot_cos_sin.permute(0, 3, 1, 2),
                    motion[:, 4:],
                ],
                dim=1,
            )
            if self.normalize_global_pos:
                motion[:, :5] = (
                    motion[:, :5]
                    - self.motion_global_mean[None, :, None, None].to(self.device)
                ) / self.motion_global_std[None, :, None, None].to(self.device)
        else:
            raise ValueError(f"Unknown motion representation: {self.motion_rep}")
        return motion

    def training_step(self, batch, batch_idx):
        schedule = self.cfg.get("schedule", dict())
        update_scheduled_params(self, schedule, self.global_step)

        data = {}
        motion, cond = batch
        if motion.device != self.device:
            motion, cond = tensor_to([motion, cond], device=self.device)
        motion = self.convert_motion_rep(motion)

        data["motion"], data["cond"] = motion, cond
        data["mask"] = cond["y"]["mask"]

        if "motion_mask" in self.model_cfg:
            res = self.generate_motion_mask(
                self.model_cfg.motion_mask, data["motion"], data["cond"]["y"]["lengths"]
            )
            for key in ["motion_mask", "observed_motion", "rm_text_flag"]:
                data["cond"][key] = res[key]

        t, t_weights = self.schedule_sampler.sample(motion.shape[0], self.device)
        data = self.get_diffusion_pred_target(data, t)
        loss, loss_dict, loss_uw_dict = self.compute_loss(data, t, t_weights)

        use_twopass = self.cfg.train.get("use_twopass_loss", False)
        if use_twopass:
            # take model output as input
            new_xstart = data["model_pred"].detach()
            data["motion"] = new_xstart

            data = self.get_diffusion_pred_target(data, t)
            data["target"] = motion  # still want to hit the actual GT
            new_loss, new_loss_dict, new_loss_uw_dict = self.compute_loss(
                data, t, t_weights
            )

            twopass_weight = self.cfg.train.get("twopass_weight", 1.0)
            loss = loss + new_loss * twopass_weight
            loss_dict.update({k + "_twopass": v for k, v in new_loss_dict.items()})
            loss_uw_dict.update(
                {k + "_twopass": v for k, v in new_loss_uw_dict.items()}
            )

        self.log("loss/train_all", loss, on_step=True, on_epoch=True, sync_dist=True)
        for key, val in loss_uw_dict.items():
            self.log(
                f"loss/train_{key}", val, on_step=True, on_epoch=True, sync_dist=True
            )
        return loss

    def compute_loss(self, data, t, t_weights):
        def masked_l2(cfg):
            part = cfg.get("part", "all")
            if part == "all":
                ind = None
            elif part == "root":
                ind = slice(0, self.motion_root_dim)
            elif part == "body":
                ind = slice(self.motion_root_dim, None)
            a, b = data["model_pred"], data["target"]
            if ind is not None:
                a, b = a[:, ind], b[:, ind]
            mask = data["mask"]
            loss = (a - b) ** 2
            loss = sum_flat(loss * mask.float())
            n_entries = a.shape[1] * a.shape[2]
            non_zero_elements = sum_flat(mask) * n_entries
            non_zero_elements[non_zero_elements == 0] = 1
            loss = loss / non_zero_elements
            loss = (loss * t_weights).mean()
            return loss, {}

        def masked_l2_sum(cfg):
            part = cfg.get("part", "all")
            if part == "all":
                ind = None
            elif part == "root":
                ind = slice(0, self.motion_root_dim)
            elif part == "body":
                ind = slice(self.motion_root_dim, None)
            a, b = data["model_pred"], data["target"]
            if ind is not None:
                a, b = a[:, ind], b[:, ind]
            mask = data["mask"]
            loss = (a - b) ** 2
            loss = sum_flat(loss * mask.float())
            loss = (loss * t_weights).sum()
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
        if rm_text_flag is not None:
            cond["rm_text_flag"] = tensor_to(rm_text_flag, device=self.device)
        if motion_mask is not None and observed_motion is not None:
            cond["motion_mask"], cond["observed_motion"] = tensor_to(
                [motion_mask, observed_motion], device=self.device
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

    def infer_texts(
        self,
        texts,
        num_frames,
        motion_mask=None,
        observed_motion=None,
        rm_text_flag=None,
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
        if rm_text_flag is not None:
            cond["rm_text_flag"] = tensor_to(rm_text_flag, device=self.device)
        if motion_mask is not None and observed_motion is not None:
            cond["motion_mask"], cond["observed_motion"] = tensor_to(
                [motion_mask, observed_motion], device=self.device
            )

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

    def obtain_full_motion_rep(self, samples, gt_motion=None):
        return samples

    def obtain_joints(
        self, samples, interp=True, return_contacts=False, return_joint_rot=False
    ):
        base_rot = torch.tensor(
            [[0.5, 0.5, 0.5, 0.5]], device=samples.device, dtype=samples.dtype
        )
        if self.motion_rep in {
            "global_root_local_joints",
            "global_root_local_joints_root_rot",
        }:
            samples_perm = samples.permute(
                0, 2, 3, 1
            )  # now permulted to: [batch, 1, seq_len, nfeat]
            root_samp = samples_perm[..., :5]
            if self.normalize_global_pos:
                root_samp = root_samp * self.motion_global_std.to(
                    self.device
                ) + self.motion_global_mean.to(self.device)
            r_pos, rot_cos_sin, local_feats = (
                root_samp[..., :3],
                root_samp[..., 3:5],
                samples_perm[..., 5:],
            )
            rot_cos_sin = normalize(rot_cos_sin)
            r_rot_ang = torch.atan2(rot_cos_sin[..., [1]], rot_cos_sin[..., [0]])
            r_rot_quat = torch.cat(
                [
                    torch.cos(r_rot_ang / 2),
                    torch.zeros_like(rot_cos_sin[..., [0]]),
                    torch.sin(r_rot_ang / 2),
                    torch.zeros_like(rot_cos_sin[..., [0]]),
                ],
                dim=-1,
            )

            local_feats_norm = local_feats * self.motion_std[4:].to(
                self.device
            ) + self.motion_mean[4:].to(self.device)  # [batch, 1, seq_len, nfeat]
            local_feats_norm_pad = torch.cat(
                [torch.zeros_like(local_feats_norm[..., :4]), local_feats_norm], dim=-1
            )

            if self.motion_rep == "global_root_local_joints":
                joints_pos = recover_from_ric(
                    local_feats_norm_pad, self.num_joints, r_rot_quat, r_pos
                )[:, 0]
            elif self.motion_rep == "global_root_local_joints_root_rot":
                joints_pos = recover_from_ric_with_joint_rot(
                    local_feats_norm_pad,
                    self.neutral_joints.to(samples_perm),
                    self.joint_parents.to(samples_perm.device),
                    r_rot_quat=r_rot_quat,
                    r_pos=r_pos,
                )[:, 0]

                joints_pos, joints_rot = recover_from_ric_with_joint_rot(
                    local_feats_norm_pad,
                    self.neutral_joints.to(samples_perm),
                    self.joint_parents.to(samples_perm.device),
                    r_rot_quat=r_rot_quat,
                    r_pos=r_pos,
                    return_joint_rot=True,
                )
                joints_pos = joints_pos[:, 0]
            if interp:
                joints_pos = interp_tensor_with_scipy(joints_pos, scale=1.5, dim=1)
            joints_pos = quat_apply(
                base_rot.expand(joints_pos.shape[:-1] + (4,)), joints_pos
            )
            if return_joint_rot:
                if interp:
                    joints_rot = slerp_joint_rots(joints_rot, scale=1.5)
                root_rot_quat = joints_rot[:, :, 0:1]
                root_rot_quat = quat_mul(
                    base_rot[:, None, None].expand_as(root_rot_quat), root_rot_quat
                )
                joints_rot = torch.cat([root_rot_quat, joints_rot[:, :, 1:]], dim=2)
        elif self.motion_rep == "orig":
            samples_perm = samples.permute(0, 2, 3, 1)[
                :, 0
            ]  # now permuted to: [batch, 1, seq_len, nfeat]
            samples_perm = samples_perm * self.motion_std.to(
                samples_perm.device
            ) + self.motion_mean.to(samples_perm.device)
            joints_pos = recover_from_ric(samples_perm, self.num_joints)
            if interp:
                joints_pos = interp_tensor_with_scipy(joints_pos, scale=1.5, dim=1)
            joints_pos = quat_apply(
                base_rot.expand(joints_pos.shape[:-1] + (4,)), joints_pos
            )
        elif self.motion_rep == "orig_rot":
            samples_perm = samples.permute(0, 2, 3, 1)[:, 0]  # [batch, seq_len, nfeat]
            samples_perm = samples_perm * self.motion_std.to(
                samples_perm.device
            ) + self.motion_mean.to(samples_perm.device)
            joints_pos, joints_rot = recover_from_ric_with_joint_rot(
                samples_perm,
                self.neutral_joints.to(samples_perm),
                self.joint_parents.to(samples_perm.device),
                return_joint_rot=True,
            )
            if interp:
                joints_pos = interp_tensor_with_scipy(joints_pos, scale=1.5, dim=1)
            joints_pos = quat_apply(
                base_rot.expand(joints_pos.shape[:-1] + (4,)), joints_pos
            )
            if return_joint_rot:
                if interp:
                    joints_rot = slerp_joint_rots(joints_rot, scale=1.5)
                root_rot_quat = joints_rot[:, :, 0:1]
                root_rot_quat = quat_mul(
                    base_rot[:, None, None].expand_as(root_rot_quat), root_rot_quat
                )
                joints_rot = torch.cat([root_rot_quat, joints_rot[:, :, 1:]], dim=2)
        else:
            raise ValueError(f"Unknown motion representation: {self.motion_rep}")

        output = [joints_pos]

        if return_contacts:
            samples_perm = samples.permute(0, 2, 3, 1)
            foot_contacts = samples_perm[
                :, 0, :, -4:
            ]  # [batch, seq_len, 4] where 4 is ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")
            # should be between 0 and 1 unnormalized
            foot_contacts_norm = foot_contacts * self.motion_std[-4:].to(
                samples_perm.device
            ) + self.motion_mean[-4:].to(samples_perm.device)
            if interp:
                foot_contacts_norm = interp_tensor_with_scipy(
                    foot_contacts_norm, scale=1.5, dim=1
                )
            contacts = foot_contacts_norm > 0.5
            output += [contacts]

        if return_joint_rot:
            output += [joints_rot]

        if len(output) > 1:
            return tuple(output)
        else:
            return output[0]

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

            data["motion"], data["cond"] = motion, cond
            data["mask"] = cond["y"]["mask"]

            if "motion_mask" in self.model_cfg:
                res = self.generate_motion_mask(
                    self.model_cfg.motion_mask,
                    data["motion"],
                    data["cond"]["y"]["lengths"],
                )
                for key in ["motion_mask", "observed_motion", "rm_text_flag"]:
                    data["cond"][key] = res[key]

            t, t_weights = self.schedule_sampler.sample(motion.shape[0], self.device)
            data = self.get_diffusion_pred_target(data, t)
            loss, loss_dict, loss_uw_dict = self.compute_loss(data, t, t_weights)
            self.train(training)
        return loss, loss_uw_dict, batch_size
