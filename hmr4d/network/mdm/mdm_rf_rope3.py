import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import numpy as np
import random
from hmr4d.utils.net_utils import length_to_mask
from hmr4d.network.base_arch.transformer.layer import zero_module
from motiondiff.models.model_util import create_gaussian_diffusion
from motiondiff.models.common.cfg_sampler import ClassifierFreeSampleModel
from motiondiff.diffusion.resample import create_named_schedule_sampler
from hmr4d.utils.geo.hmr_cam import (
    compute_bbox_info_bedlam,
    compute_transl_full_cam,
    get_a_pred_cam,
    project_to_bi01,
)
from hmr4d.utils.geo.hmr_global import (
    rollout_local_transl_vel,
    get_static_joint_mask,
    get_tgtcoord_rootparam,
)
from skimage.util.shape import view_as_windows
from timm.models.vision_transformer import Mlp


def chunk_dict_batch(dic, chunk):
    new_dict = {}
    for key, value in dic.items():
        if isinstance(value, dict):
            new_dict[key] = chunk_dict_batch(value, chunk)
        elif isinstance(value, torch.Tensor):
            if key == 'length':
                new_dict[key] = value
            else:
                new_dict[key] = value[:, chunk]
        else:
            raise ValueError(f"Unsupported type: {type(value)}")
    return new_dict


def cat_dict_batch(dic):
    new_dict = {}
    for key, value in dic.items():
        if isinstance(value, dict):
            new_dict[key] = cat_dict_batch(value)
        elif isinstance(value, list):
            new_dict[key] = torch.cat(value, dim=0)
        else:
            raise ValueError(f"Unsupported type: {type(value)}")
    return new_dict


def append_dict_batch(dic1, dic2):
    for key, value in dic2.items():
        if isinstance(value, dict):
            if key not in dic1:
                dic1[key] = {}
            append_dict_batch(dic1[key], value)
        else:
            if key not in dic1:
                dic1[key] = []
            dic1[key].append(value)
    return dic1


def import_type_from_str(s):
    module_name, type_name = s.rsplit(".", 1)
    module = importlib.import_module(module_name)
    type_to_import = getattr(module, type_name)
    return type_to_import



class MDMBase(nn.Module):
    def __init__(
        self,
        model_cfg,
        motion_rep_dim,
        sde,
        max_len=120,
        # condition
        cliffcam_dim=3,
        cam_angvel_dim=6,
        cam_t_vel_dim=3,
        imgseq_dim=1024,
        # intermediate
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        # output
        pred_cam_dim=3,
        static_conf_dim=6,
        # training
        dropout=0.1,
        # other
        avgbeta=True,
        zero_unknown_motion=0.0,
        pred_2dkpt=True,
        args=None,
        **kwargs,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.args = args
        self.sde = sde
        self.pred_mode = sde.pred_mode

        self.motion_rep_dim = motion_rep_dim
        self.max_len = max_len

        # condition
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.cam_t_vel_dim = cam_t_vel_dim
        self.imgseq_dim = imgseq_dim
        self.s_pred_ind = 0

        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.zero_unknown_motion = zero_unknown_motion
        self.pred_2dkpt = pred_2dkpt

        assert 'obs' in self.args.in_attr, "obs (kp2d) must be in in_attr"
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=self.latent_dim * 2, out_features=self.latent_dim, drop=dropout
        )
        latent_dim = self.latent_dim
        dropout = self.dropout
        if "bbx" in self.args.in_attr:
            self.cliffcam_embedder = nn.Sequential(
                nn.Linear(self.cliffcam_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
            self.learned_cliffcam_params = nn.Parameter(torch.randn(latent_dim), requires_grad=True)

        if "cam_angvel" in self.args.in_attr:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
            self.learned_cam_angvel_params = nn.Parameter(torch.randn(latent_dim), requires_grad=True)

        if "cam_t_vel" in self.args.in_attr:
            self.cam_t_vel_embedder = nn.Sequential(
                nn.Linear(self.cam_t_vel_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
            )
            self.learned_cam_t_vel_params = nn.Parameter(torch.randn(latent_dim), requires_grad=True)

        if "imgfeat" in self.args.in_attr:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
            )

        self.load_ext_models()

        return

    def load_pretrain_checkpoint(self):
        if 'pretrained_checkpoint' in self.model_cfg:
            cp_cfg = self.model_cfg.pretrained_checkpoint
            state_dict = torch.load(cp_cfg.path, map_location='cpu')['state_dict']
            filter_keys = cp_cfg.get('filter_keys', [])
            try_load = cp_cfg.get('try_load', False)
            if len(filter_keys) > 0:
                print(f'Filtering checkpoint keys: {filter_keys}')
                skipped_keys = [k for k in state_dict.keys() if any(key in k for key in filter_keys)]
                print(f'Skipped keys: {skipped_keys}')
                state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in filter_keys)}
            if try_load:
                model_state = self.state_dict()
                state_dict = {k: v for k, v in state_dict.items()
                            if k in model_state and v.size() == model_state[k].size()}

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=cp_cfg.get('strict', True))
            # if len(missing_keys) > 0:
            #     print(f'Missing keys: {missing_keys}')
            # if len(unexpected_keys) > 0:
            #     print(f'Unexpected keys: {unexpected_keys}')


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

    def init_diffusion(self):
        self.train_diffusion = create_gaussian_diffusion(self.model_cfg.diffusion, training=True)
        self.test_diffusion = create_gaussian_diffusion(self.model_cfg.diffusion, training=False)
        self.schedule_sampler = create_named_schedule_sampler(self.model_cfg.diffusion.schedule_sampler_type, self.train_diffusion)
        # self.guided_denoiser = ClassifierFreeSampleModel(self.denoiser)
        return

    def generate_motion_rep(
        self, batch, target_x, static_gt
    ):
        f_condition = batch["f_condition"]
        f_condition_valid_mask = batch['f_condition_valid_mask']

        length = batch["length"]
        assert 'obs' in f_condition
        if 'obs' in f_condition:
            obs = f_condition["obs"]
            B, L, J, C = obs.shape
            assert J == 17 and C == 3

            # Main token from observation (2D pose)
            obs = obs.clone()
            visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)

            if 'obs' in f_condition_valid_mask:
                valid_mask = f_condition_valid_mask["obs"]  # (B, L)
                visible_mask[~valid_mask] = False
            obs[~visible_mask[..., 0]] = 0  # set low-conf to all zeros

            # f_obs = obs[..., :2]  # (B, L, J, 32)
            f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
            f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask  # (B, L, J, 32)
            f_obs = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, J*32) -> (B, L, C)
            f_cond = f_obs

        if 'f_cliffcam' in f_condition:
            f_cliffcam = f_condition["f_cliffcam"]  # (B, L, 3)
            f_cliffcam = self.cliffcam_embedder(f_cliffcam)
            if 'f_cliffcam' in f_condition_valid_mask:
                valid_mask = f_condition_valid_mask["f_cliffcam"]  # (B, L)
                f_cliffcam = f_cliffcam * valid_mask[..., None] + self.learned_cliffcam_params.repeat(B, L, 1) * ~valid_mask[..., None]

            f_cond = f_cond + f_cliffcam
        if "f_cam_angvel" in f_condition:
            f_cam_angvel = f_condition["f_cam_angvel"]  # (B, L, 6)
            f_cam_angvel = self.cam_angvel_embedder(f_cam_angvel)
            if 'f_cam_angvel' in f_condition_valid_mask:
                valid_mask = f_condition_valid_mask["f_cam_angvel"]  # (B, L)
                f_cam_angvel = f_cam_angvel * valid_mask[..., None] + self.learned_cam_angvel_params.repeat(B, L, 1) * ~valid_mask[..., None]
            f_cond = f_cond + f_cam_angvel
        if "f_cam_t_vel" in f_condition:
            f_cam_t_vel = f_condition["f_cam_t_vel"]  # (B, L, 3)
            f_cam_t_vel = self.cam_t_vel_embedder(f_cam_t_vel)
            if 'f_cam_t_vel' in f_condition_valid_mask:
                valid_mask = f_condition_valid_mask["f_cam_t_vel"]  # (B, L)
                f_cam_t_vel = f_cam_t_vel * valid_mask[..., None] + self.learned_cam_t_vel_params.repeat(B, L, 1) * ~valid_mask[..., None]
            f_cond = f_cond + f_cam_t_vel
        if 'f_imgseq' in f_condition:
            f_imgseq = f_condition["f_imgseq"]  # (B, L, C)
            f_imgseq = self.imgseq_embedder(f_imgseq)
            if 'f_imgseq' in f_condition_valid_mask:
                valid_mask = f_condition_valid_mask["f_imgseq"]  # (B, L)
                f_imgseq = f_imgseq * valid_mask[..., None]
            f_cond = f_cond + f_imgseq

        # f_cam = self.cam_angvel_embedder(f_cam_angvel)
        # clean_f_obs = clean_obs[..., :2]
        # clean_f_obs = clean_f_obs.reshape(B, L, -1)

        vis_mask = length_to_mask(length, L)  # (B, L)
        pmask = ~vis_mask  # (B, L)

        # 6 + 151
        motion = torch.cat([static_gt, target_x], dim=-1)
        clean_motion = torch.cat([static_gt, target_x], dim=-1)
        motion_mask = torch.zeros_like(motion)
        motion_mask = motion_mask * vis_mask[..., None]

        motion = motion * vis_mask[..., None]
        clean_motion = clean_motion * vis_mask[..., None]

        self.denoiser.s_pred_ind = self.s_pred_ind
        # if 'f_condition_valid_mask' in batch:
        #     f_condition_valid_mask = batch['f_condition_valid_mask']
        #     # j2d_visible_mask = batch['j2d_visible_mask'][..., None].to(motion)
        #     # j2d_visible_mask = j2d_visible_mask.repeat(1, 1, 1, 2).reshape(B, L, 34)
        #     # motion_mask[:, :, :17 * 2] *= (f_condition_valid_mask['obs'][..., None].to(motion) * j2d_visible_mask)
        #     # motion_mask[:, :, 17 * 2:17 * 2 + 3] *= f_condition_valid_mask['f_cliffcam'][..., None].to(motion)
        #     motion_mask[:, :, :6] *= f_condition_valid_mask['f_cam_angvel'][..., None].to(motion)
        return f_cond, motion, motion_mask, clean_motion

    def get_diffusion_pred_target(self, batch, target_x, static_gt, mode):
        diffusion = self.train_diffusion if self.training else self.test_diffusion

        outputs = dict()
        f_condition = batch["f_condition"]
        clean_f_condition = batch["clean_f_condition"]
        f_condition_valid_mask = batch["f_condition_valid_mask"]
        length = batch["length"]

        f_cond, motion, motion_mask, clean_motion = self.generate_motion_rep(
            batch, target_x, static_gt
        )
        B, L, _ = motion.shape
        # Setup length and make padding mask
        vis_mask = length_to_mask(length, L)  # (B, L)
        if self.args.get("vis_masking_f_cond", True):
            f_cond = f_cond * vis_mask[..., None]

        denoiser_kwargs = {
            "y": {
                "text": [""] * B,
                "f_cond": f_cond.clone(),
                "mask": vis_mask.clone(),
                "length": length.clone(),
            },
            "motion_mask": motion_mask.clone(),
            "observed_motion": motion.clone(),
        }


        valid_mask = batch["mask"]["valid"]

        if mode == 'regression':
            t = torch.rand(motion.shape[0], device=motion.device) * (self.sde.T - self.sde.sampling_eps) + self.sde.sampling_eps
            t = t * 0.0
            t_weights = torch.ones(B).to(motion.device)
            t_expand = t.view(-1, 1, 1)
            zt = motion * motion_mask
            noise = torch.zeros_like(zt)

        elif mode == 'rf':
            t = torch.rand(motion.shape[0], device=motion.device) * (self.sde.T - self.sde.sampling_eps) + self.sde.sampling_eps
            t_weights = torch.ones(B).to(motion.device)
            pred_x_start_regression = batch['regression_outputs']['model_output']['pred_x_start'].detach()
            if self.args.get('diffusion_all_regression_outputs', False):
                x_start = pred_x_start_regression
            else:
                inpaint_mask = torch.ones_like(pred_x_start_regression)
                inpaint_mask = inpaint_mask * valid_mask[:, :, None]
                inpaint_mask = inpaint_mask * vis_mask[:, :, None]
                x_start = clean_motion.clone() * inpaint_mask + pred_x_start_regression * (1 - inpaint_mask)
                x_start = x_start * valid_mask[:, :, None]
            noise = torch.randn_like(x_start)
            t_expand = t.view(-1, 1, 1)
            zt = t_expand * x_start + (1 - t_expand) * noise
            target_drift = x_start - noise
        
        denoise_out = self.denoiser(
            zt, (t * 999).long(), return_aux=False, clip_cam=False, **denoiser_kwargs
        )
        target_x_start = clean_motion

        if self.pred_mode == "drift":
            pred_x_start_drift = denoise_out["pred_x_start"]
            pred_x_start = zt + pred_x_start_drift * (1 - t_expand)
        elif self.pred_mode == "z1":
            pred_x_start_drift = denoise_out["pred_x_start"]
            # pred_x_start = z0 + pred_x_start_drift
            pred_x_start = zt + pred_x_start_drift * (1 - t_expand)
        elif self.pred_mode == "z1_z0":
            pred_x_start_drift = denoise_out["pred_x_start"]
            pred_x_start = noise + pred_x_start_drift

        static_conf_logits = pred_x_start[:, :, self.s_pred_ind : self.s_pred_ind + 6]
        assert pred_x_start.shape[-1] == self.s_pred_ind + 6 + 151
        sample = pred_x_start[:, :, self.s_pred_ind + 6:]

        valid_loss_mask = torch.ones_like(pred_x_start)
        valid_loss_mask[batch["mask"]["spv_incam_only"], :, -9:] *= 0
        valid_loss_mask[batch["mask"]["spv_incam_only"], :, self.s_pred_ind :self.s_pred_ind + 6] *= 0
        valid_loss_mask = valid_loss_mask * valid_mask[:, :, None]
        
        output = {
            "pred_x_start": pred_x_start,
            "target_x_start": target_x_start,
            "valid_loss_mask": valid_loss_mask,
            "pred_x": sample,
            "static_conf_logits": static_conf_logits,
            "t_weights": t_weights,
        }
        for x in self.args.out_attr:
            if x == 'pred_cam':
                pred_cam = denoise_out[x]
                torch.clamp_min_(pred_cam[..., 0], 0.25)
                output[x] = pred_cam
            else:
                output[x] = denoise_out[x]

        return output

    def forward_train(self, inputs, train=False, postproc=False, static_cam=False, mode=None):
        assert self.training, "forward_train should only be called during training"
        outputs = dict()
        length = inputs["length"]  # (B,) effective length of each sample

        # assert self.endecoder.clip_std, "endecoder should be clip_std"
        target_x = self.endecoder.encode(inputs)  # (B, L, C)

        # vel_thr = args.static_conf.vel_thr
        # assert vel_thr > 0
        vel_thr = 0.15
        joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
        gt_w_j3d = self.endecoder.fk_v2(**inputs["smpl_params_w"])  # (B, L, J=22, 3)
        static_gt = get_static_joint_mask(gt_w_j3d, vel_thr=vel_thr, repeat_last=True)  # (B, L, J)
        static_gt = static_gt[:, :, joint_ids].float()  # (B, L, J')

        output = self.get_diffusion_pred_target(
            batch=inputs,
            target_x=target_x,
            static_gt=static_gt,
            mode=mode
        )
        return output

    def forward_train_2d(self, inputs, mode):
        assert self.training, "forward_train_2d should only be called during training"
        diffusion = self.train_diffusion if self.training else self.test_diffusion

        outputs = dict()
        length = inputs["length"]  # (B,) effective length of each sample
        L = self.max_len
        C = self.motion_rep_dim
        B = length.shape[0]
        target_x = torch.zeros(B, L, 151).to(inputs['f_condition']['obs'].device)
        static_gt = torch.zeros(B, L, 6).to(inputs['f_condition']['obs'].device)

        f_cond, motion, _, _ = self.generate_motion_rep(
            inputs, target_x, static_gt
        )
        # Setup length and make padding mask
        vis_mask = length_to_mask(length, L)  # (B, L)
        if self.args.get("vis_masking_f_cond", True):
            f_cond = f_cond * vis_mask[..., None]

        denoiser_kwargs = {
            "y": {
                "text": [""] * B,
                "f_cond": f_cond.clone(),
                "mask": vis_mask.clone(),
                "length": length.clone(),
            }
        }
        # self.denoiser.s_pred_ind = self.s_pred_ind
        if mode == 'regression':
            t = torch.rand(motion.shape[0], device=motion.device) * (self.sde.T - self.sde.sampling_eps) + self.sde.sampling_eps
            t = t * 0.0
            t_weights = torch.ones(B).to(motion.device)
            t_expand = t.view(-1, 1, 1)
            zt = torch.rand_like(motion)
            noise = torch.zeros_like(zt)
        elif mode == 'rf':
            t = torch.rand(motion.shape[0], device=motion.device) * (self.sde.T - self.sde.sampling_eps) + self.sde.sampling_eps
            t_weights = torch.ones(B).to(motion.device)
            pred_x_start_regression = inputs['regression_outputs']['2d_model_output']['pred_x_start'].detach()
            x_start = pred_x_start_regression
            noise = torch.randn_like(x_start)
            t_expand = t.view(-1, 1, 1)
            zt = t_expand * x_start + (1 - t_expand) * noise
            target_drift = x_start - noise
        denoise_out = self.denoiser(
            zt, (t * 999).long(), return_aux=False, clip_cam=False, **denoiser_kwargs
        )

        if self.pred_mode == "drift":
            pred_x_start_drift = denoise_out["pred_x_start"]
            pred_x_start = zt + pred_x_start_drift * (1 - t_expand)
        elif self.pred_mode == "z1":
            pred_x_start_drift = denoise_out["pred_x_start"]
            # pred_x_start = z0 + pred_x_start_drift
            pred_x_start = zt + pred_x_start_drift * (1 - t_expand)
        elif self.pred_mode == "z1_z0":
            pred_x_start_drift = denoise_out["pred_x_start"]
            pred_x_start = noise + pred_x_start_drift

        static_conf_logits = pred_x_start[:, :, self.s_pred_ind : self.s_pred_ind + 6]
        assert pred_x_start.shape[-1] == self.s_pred_ind + 6 + 151
        sample = pred_x_start[:, :, self.s_pred_ind + 6:]

        output = {
            "pred_x_start": pred_x_start,
            "pred_x": sample,
            "static_conf_logits": static_conf_logits,
            "t_weights": t_weights,
        }
        for x in self.args.out_attr:
            if x == "pred_cam":
                pred_cam = denoise_out[x]
                torch.clamp_min_(pred_cam[..., 0], 0.25)
                output[x] = pred_cam
            else:
                output[x] = denoise_out[x]
        return output

    def forward_test(self, inputs, train=False, postproc=False, static_cam=False, progress=False, mode=None):
        assert not self.training, "forward_test should only be called during inference"
        diffusion = self.test_diffusion
        denoiser = self.denoiser
        length = inputs["length"]  # (B,) effective length of each sample
        regression_only = self.args.get('regression_only', False)

        f_condition = inputs["f_condition"]
        cliff_cam = f_condition["f_cliffcam"]
        B, L = cliff_cam.shape[:2]

        target_x = torch.zeros(B, L, 151).to(cliff_cam)
        static_gt = torch.zeros(B, L, 6).to(cliff_cam)
        f_cond, motion, motion_mask, clean_motion = self.generate_motion_rep(
            inputs, target_x, static_gt
        )

        vis_mask = length_to_mask(length, L)  # (B, L)
        if self.args.get("vis_masking_f_cond", True):
            f_cond = f_cond * vis_mask[..., None]

        zt = torch.randn_like(motion)
        zt = torch.zeros_like(motion)
        sample_N = self.sde.sample_N
        dt = (1.0 - self.sde.sampling_eps) / sample_N
        if "pred_cam" in self.args.out_attr:
            pred_cam_zt = 0
        if "pred_cam_t_vel" in self.args.out_attr:
            pred_cam_t_vel_z1 = 0
        if 'cam_scale' in self.args.out_attr:
            pred_cam_scale_z1 = 0

        for i in range(sample_N):
            num_t = i / sample_N * (self.sde.T - self.sde.sampling_eps) + self.sde.sampling_eps
            vec_t = torch.ones(B, device=motion.device) * num_t


            denoiser_kwargs = {
                "y": {
                    "text": [""] * B,
                    "f_cond": f_cond.clone(),
                    "mask": vis_mask.clone(),
                    "length": length.clone(),
                },
                "motion_mask": motion_mask,
                "observed_motion": motion,
                # "encoder": self.endecoder,
                # "inputs": inputs,
            }

            denoise_out = self.denoiser(
                zt, (vec_t * 999).long(), return_aux=False, clip_cam=False, **denoiser_kwargs
            )
            if self.pred_mode == "drift":
                drift = denoise_out["pred_x_start"]

                # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability
                zt = zt + drift * dt

                if "pred_cam" in self.args.out_attr:
                    drift_cam = denoise_out["pred_cam"]
                    pred_cam_zt = pred_cam_zt + drift_cam * dt
                if "cam_t_vel" in self.args.out_attr:
                    drift_cam_t_vel = denoise_out["pred_cam_t_vel"]
                    pred_cam_t_vel_z1 = pred_cam_t_vel_z1 + drift_cam_t_vel * dt
                if 'cam_scale' in self.args.out_attr:
                    drift_cam_scale = denoise_out["pred_cam_scale"]
                    pred_cam_scale_z1 = pred_cam_scale_z1 + drift_cam_scale * dt

            else:
                drift = denoise_out["pred_x_start"]
                # drift = (pred_z1 - zt) / (1 - vec_t[:, None, None])

                # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability
                old_zt = zt.clone()
                zt = zt + drift * dt
                # zt = torch.clamp(zt, -20, 20)

                if torch.isnan(zt).any():
                    import ipdb; ipdb.set_trace()

                if "pred_cam" in self.args.out_attr:
                    pred_cam = denoise_out["pred_cam"]
                    torch.clamp_min_(pred_cam[..., 0], 0.25)
                    drift_cam = (pred_cam - pred_cam_zt) / (1 - vec_t[:, None])
                    pred_cam_zt = pred_cam_zt + drift_cam * dt

                if "cam_t_vel" in self.args.out_attr:
                    pred_cam_t_vel = denoise_out["pred_cam_t_vel"]
                    drift_cam_t_vel = (pred_cam_t_vel - pred_cam_t_vel_z1) / (1 - vec_t[:, None])
                    pred_cam_t_vel_z1 = pred_cam_t_vel_z1 + drift_cam_t_vel * dt
                if 'cam_scale' in self.args.out_attr:
                    pred_cam_scale = denoise_out["pred_cam_scale"]
                    drift_cam_scale = (pred_cam_scale - pred_cam_scale_z1) / (1 - vec_t[:, None])
                    pred_cam_scale_z1 = pred_cam_scale_z1 + drift_cam_scale * dt

        samples = zt
        pred_cam_z1 = pred_cam_zt
        torch.clamp_min_(pred_cam_z1[..., 0], 0.25)

        static_conf_logits = samples[:, :, self.s_pred_ind:self.s_pred_ind + 6]
        sample = samples[:, :, self.s_pred_ind + 6:]

        output = {
            "pred_x": sample,
            "static_conf_logits": static_conf_logits,
        }
        if 'pred_cam' in self.args.out_attr:
            output["pred_cam"] = pred_cam_z1
        if 'cam_t_vel' in self.args.out_attr:
            output["pred_cam_t_vel"] = pred_cam_t_vel_z1
        if 'cam_scale' in self.args.out_attr:
            output["pred_cam_scale"] = pred_cam_scale_z1
        return output

    def forward(self, inputs, train=False, postproc=False, static_cam=False, mode=None):
        if train:
            return self.forward_train(inputs, train, postproc, static_cam, mode=mode)
        else:
            return self.forward_test(inputs, train, postproc, static_cam, mode=mode)


class MDMRF(MDMBase):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.denoiser = import_type_from_str(self.model_cfg.denoiser.type)(
            pl_module=self, **self.model_cfg.denoiser
        )
        self.init_diffusion()
        if self.model_cfg.get("preload_checkpoint", True):
            self.load_pretrain_checkpoint()
        return
