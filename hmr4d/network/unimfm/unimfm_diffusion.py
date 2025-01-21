import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import numpy as np
import random
from copy import deepcopy
from hmr4d.utils.net_utils import length_to_mask
from hmr4d.network.base_arch.transformer.layer import zero_module
from motiondiff.models.model_util import create_gaussian_diffusion
from .unimfm_cfg_sampler import ClassifierFreeSampleModel
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



def import_type_from_str(s):
    module_name, type_name = s.rsplit(".", 1)
    module = importlib.import_module(module_name)
    type_to_import = getattr(module, type_name)
    return type_to_import



class UNIMFMDiffusion(nn.Module):
    def __init__(
        self,
        model_cfg,
        max_len=120,
        # condition
        cliffcam_dim=3,
        cam_angvel_dim=6,
        cam_t_vel_dim=3,
        imgseq_dim=1024,
        latent_dim=512,
        dropout=0.1,
        args=None,
        use_cond_exists_as_input=False,
        cond_merge_strategy="add",
        cond_exists_dim=512,
        **kwargs,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.args = args
        self.max_len = max_len

        # condition
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.cam_t_vel_dim = cam_t_vel_dim
        self.imgseq_dim = imgseq_dim
        self.s_pred_ind = 0

        # intermediate
        self.latent_dim = latent_dim
        self.dropout = dropout
        
        self.regression_input_type = self.args.get('regression_input_type', 'zero')
        self.use_cond_exists_as_input = use_cond_exists_as_input
        self.cond_merge_strategy = cond_merge_strategy
        self.cond_exists_dim = cond_exists_dim

        assert 'obs' in self.args.in_attr, "obs (kp2d) must be in in_attr"
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32, hidden_features=self.latent_dim * 2, out_features=self.latent_dim, drop=dropout
        )
        latent_dim = self.latent_dim
        dropout = self.dropout
        if "f_cliffcam" in self.args.in_attr:
            self.cliffcam_embedder = nn.Sequential(
                nn.Linear(self.cliffcam_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
            self.learned_cliffcam_params = nn.Parameter(torch.randn(latent_dim), requires_grad=True)

        if "f_cam_angvel" in self.args.in_attr:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
            self.learned_cam_angvel_params = nn.Parameter(torch.randn(latent_dim), requires_grad=True)

        if "f_cam_t_vel" in self.args.in_attr:
            self.cam_t_vel_embedder = nn.Sequential(
                nn.Linear(self.cam_t_vel_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
            )
            self.learned_cam_t_vel_params = nn.Parameter(torch.randn(latent_dim), requires_grad=True)

        if "f_imgseq" in self.args.in_attr:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
            )
            
        add_dim = 0
        if self.use_cond_exists_as_input:
            if cond_merge_strategy == 'add':
                self.cond_exists_embedder = nn.ModuleDict()
                for k in self.args.in_attr:
                    self.cond_exists_embedder[k] = nn.Sequential(
                        nn.Linear(latent_dim + 1, latent_dim),
                        nn.SiLU(),
                        zero_module(nn.Linear(latent_dim, latent_dim)),
                    )
            elif cond_merge_strategy == 'concat':
                add_dim = cond_exists_dim
                
        if cond_merge_strategy == 'concat':
            # self.cond_merger = nn.Linear(len(self.args.in_attr) * (latent_dim + add_dim), latent_dim)
            self.cond_merger = nn.Sequential(
                nn.Linear(len(self.args.in_attr) * (latent_dim + add_dim), latent_dim),
                nn.SiLU(),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )

        self.denoiser = import_type_from_str(self.model_cfg.denoiser.type)(
            pl_module=self, **self.model_cfg.denoiser
        )
        self.init_diffusion()
        return

    def init_diffusion(self):
        self.train_diffusion = create_gaussian_diffusion(self.model_cfg.diffusion, training=True)
        self.test_diffusion = create_gaussian_diffusion(self.model_cfg.diffusion, training=False)
        text_only_diffusion = deepcopy(self.model_cfg.diffusion)
        text_only_diffusion.test_timestep_respacing = self.model_cfg.diffusion.get('text_only_test_timestep_respacing', '50')
        print(f"Text only test timestep respacing: {text_only_diffusion.test_timestep_respacing}")
        self.test_text_only_diffusion = create_gaussian_diffusion(text_only_diffusion, training=False)
        self.schedule_sampler = create_named_schedule_sampler(self.model_cfg.diffusion.schedule_sampler_type, self.train_diffusion)
        return

    def generate_motion_rep(
        self, batch, target_x, static_gt
    ):
        f_condition = batch["f_condition"]
        f_condition_exists = batch["f_condition_exists"] 

        length = batch["length"]
        assert 'obs' in f_condition
        f_cond_dict = {}
        if 'obs' in f_condition:
            obs = f_condition["obs"]
            B, L, J, C = obs.shape
            assert J == 17 and C == 3
            obs = obs.clone()
            visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
            obs[~visible_mask[..., 0]] = 0  # set low-conf to all zeros
            f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
            f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask  # (B, L, J, 32)
            f_obs = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, J*32) -> (B, L, C)
            f_cond_dict["obs"] = f_obs

        if 'f_cliffcam' in f_condition:
            f_cliffcam = f_condition["f_cliffcam"]  # (B, L, 3)
            f_cliffcam = self.cliffcam_embedder(f_cliffcam)
            f_cond_dict["f_cliffcam"] = f_cliffcam
        if "f_cam_angvel" in f_condition:
            f_cam_angvel = f_condition["f_cam_angvel"]  # (B, L, 6)
            f_cam_angvel = self.cam_angvel_embedder(f_cam_angvel)
            f_cond_dict["f_cam_angvel"] = f_cam_angvel
        if "f_cam_t_vel" in f_condition:
            f_cam_t_vel = f_condition["f_cam_t_vel"]  # (B, L, 3)
            f_cam_t_vel = self.cam_t_vel_embedder(f_cam_t_vel)
            f_cond_dict["f_cam_t_vel"] = f_cam_t_vel
        if 'f_imgseq' in f_condition:
            f_imgseq = f_condition["f_imgseq"]  # (B, L, C)
            f_imgseq = self.imgseq_embedder(f_imgseq)
            f_cond_dict["f_imgseq"] = f_imgseq
            
        if self.cond_merge_strategy == 'add':
            if self.use_cond_exists_as_input:
                for k in f_cond_dict:
                    f_cond_dict[k] = torch.cat([f_cond_dict[k], f_condition_exists[k][:, :, None].float()], dim=-1)
                    f_cond_dict[k] = self.cond_exists_embedder[k](f_cond_dict[k])
            f_cond = sum(f_cond_dict.values())
        elif self.cond_merge_strategy == 'concat':
            f_cond = torch.cat(list(f_cond_dict.values()), dim=-1)
            f_cond_exists = torch.cat([f_condition_exists[k][:, :, None].float().repeat(1, 1, self.cond_exists_dim) for k in f_cond_dict], dim=-1)
            if self.use_cond_exists_as_input:
                f_cond = torch.cat([f_cond, f_cond_exists], dim=-1)
            f_cond = self.cond_merger(f_cond)

        vis_mask = length_to_mask(length, L)  # (B, L)

        motion = torch.cat([static_gt, target_x], dim=-1)
        motion = motion * vis_mask[..., None]
        return f_cond, motion

    def get_diffusion_pred_target(self, batch, mode):
        diffusion = self.train_diffusion if self.training else self.test_diffusion
        length = batch["length"]
        
        target_x = batch['target_x']
        static_gt = batch['static_gt']

        f_cond, motion = self.generate_motion_rep(
            batch, target_x, static_gt
        )
        B, L, _ = motion.shape
        vis_mask = length_to_mask(length, L)  # (B, L)
        valid_mask = batch["mask"]["valid"]
        assert (vis_mask == valid_mask).all()

        denoiser_kwargs = {
            "y": {
                "text": batch.get("caption", [""] * B),
                "f_cond": f_cond,
                "mask": vis_mask,
                "length": length,
            },
        }
        if 'encoded_text' in batch:
            denoiser_kwargs['y']['encoded_text'] = batch['encoded_text']
        
        if mode == 'regression':
            t = (torch.ones(B) * 999).long().to(motion.device)
            t_weights = torch.ones(B).to(motion.device)
            if self.regression_input_type == 'zero':
                x_t = torch.zeros_like(motion)
            elif self.regression_input_type == 'normal':
                x_t = torch.randn_like(motion)
            else:
                raise ValueError(f"Unsupported regression_input_type: {self.regression_input_type}")
        elif mode == 'diffusion':
            t, t_weights = self.schedule_sampler.sample(motion.shape[0], motion.device)
            if 'regression_outputs' in batch:
                pred_x_start_regression = batch['regression_outputs']['model_output']['pred_x_start'].detach()
            else:
                pred_x_start_regression = torch.zeros_like(motion)
            x_start_reg = pred_x_start_regression
            if self.args.get('inpaint_x_start_gt', True):
                inpaint_mask = torch.ones_like(pred_x_start_regression)
                inpaint_mask = inpaint_mask * valid_mask[:, :, None]
                x_start_gt = motion.clone() * inpaint_mask + pred_x_start_regression * (1 - inpaint_mask)
            else:
                x_start_gt = motion.clone()
            regression_mask = (torch.rand(B).to(motion.device) < self.args.use_regression_outputs_prob).float()
            if 'text_only' in batch and self.args.get('use_gt_for_text_only', True):
                regression_mask[batch['text_only']] = 0
            x_start = x_start_reg * regression_mask[:, None, None] + x_start_gt * (1 - regression_mask[:, None, None])
            noise = torch.randn_like(x_start)
            x_t = self.train_diffusion.q_sample(x_start.clone(), t, noise=noise)
        
        denoise_out = self.denoiser(
            x_t, diffusion._scale_timesteps(t), return_aux=False, **denoiser_kwargs
        )

        target_x_start = motion
        pred_x_start = denoise_out["pred_x_start"]
        static_conf_logits = denoise_out["static_conf_logits"]
        sample = denoise_out["pred_x"]
        
        output = {
            "pred_x_start": pred_x_start,
            "target_x_start": target_x_start,
            "pred_x": sample,
            "static_conf_logits": static_conf_logits,
            "t_weights": t_weights,
        }
        for x in self.args.out_attr:
            output[x] = denoise_out[x]
        
        return output

    def forward_train(self, inputs, train=False, postproc=False, static_cam=False, mode=None):
        assert self.training, "forward_train should only be called during training"

        output = self.get_diffusion_pred_target(
            batch=inputs,
            mode=mode
        )
        return output

    def forward_train_2d(self, inputs, mode):
        assert self.training, "forward_train_2d should only be called during training"
        diffusion = self.train_diffusion

        length = inputs["length"]  # (B,) effective length of each sample
        
        obs = inputs["f_condition"]["obs"]
        B, L = obs.shape[:2]

        mdim = self.endecoder.get_motion_dim()
        target_x = torch.zeros(B, L, mdim).to(obs)
        static_gt = torch.zeros(B, L, 6).to(obs)

        f_cond, motion = self.generate_motion_rep(
            inputs, target_x, static_gt
        )
        # Setup length and make padding mask
        vis_mask = length_to_mask(length, L)  # (B, L)
        
        denoiser_kwargs = {
            "y": {
                "text": inputs.get("caption", [""] * B),
                "f_cond": f_cond,
                "mask": vis_mask,
                "length": length,
            }
        }
        if 'encoded_text' in inputs:
            denoiser_kwargs['y']['encoded_text'] = inputs['encoded_text']
        
        if mode == 'regression':
            t = (torch.ones(B) * 999).long().to(motion.device)
            t_weights = torch.ones(B).to(motion.device)
            if self.regression_input_type == 'zero':
                x_t = torch.zeros_like(motion)
            elif self.regression_input_type == 'normal':
                x_t = torch.randn_like(motion)
            else:
                raise ValueError(f"Unsupported regression_input_type: {self.regression_input_type}")
        elif mode == 'diffusion':
            t, t_weights = self.schedule_sampler.sample(motion.shape[0], motion.device)
            pred_x_start_regression = inputs['regression_outputs']['2d_model_output']['pred_x_start'].detach()
            x_start = pred_x_start_regression
            noise = torch.randn_like(x_start)
            x_t = self.train_diffusion.q_sample(x_start.clone(), t, noise=noise)
            
        denoise_out = self.denoiser(
            x_t, diffusion._scale_timesteps(t), return_aux=False, **denoiser_kwargs
        )
        
        pred_x_start = denoise_out["pred_x_start"]

        static_conf_logits = denoise_out["static_conf_logits"]
        sample = denoise_out["pred_x"]

        output = {
            "pred_x_start": pred_x_start,
            "pred_x": sample,
            "static_conf_logits": static_conf_logits,
            "t_weights": t_weights,
        }
        for x in self.args.out_attr:
            output[x] = denoise_out[x]
        return output

    def forward_test(self, inputs, train=False, postproc=False, static_cam=False, progress=False, mode=None):
        assert not self.training, "forward_test should only be called during inference"
        eval_text_only = inputs.get('eval_text_only', False)
        diffusion = self.test_text_only_diffusion if eval_text_only else self.test_diffusion
        denoiser = self.denoiser
        length = inputs["length"]  # (B,) effective length of each sample
        regression_only = self.args.get('regression_only', False) and not eval_text_only
        if self.args.get('force_regression_only', False):
            regression_only = True

        f_condition = inputs["f_condition"]
        
        test_motion_len = self.args.get("test_motion_len", None)
        if test_motion_len is not None and test_motion_len > self.max_len:
            L = test_motion_len
            length = torch.ones_like(length) * L
            f_condition_exists = inputs["f_condition_exists"]
            for k in f_condition:
                f_condition[k] = torch.cat([f_condition[k], f_condition[k][:, [-1]].repeat_interleave(L-f_condition[k].shape[1], dim=1)], dim=1)
                f_condition_exists[k] = torch.cat([f_condition_exists[k], f_condition_exists[k][:, [-1]].repeat_interleave(L-f_condition_exists[k].shape[1], dim=1)], dim=1)
            for k in ['bbx_xys', 'K_fullimg', 'cam_angvel']:
                inputs[k] = torch.cat([inputs[k], inputs[k][:, [-1]].repeat_interleave(L-inputs[k].shape[1], dim=1)], dim=1)
            
        obs = f_condition["obs"]
        B, L = obs.shape[:2]

        mdim = self.endecoder.get_motion_dim()
        target_x = torch.zeros(B, L, mdim).to(obs)
        static_gt = torch.zeros(B, L, 6).to(obs)
        f_cond, motion = self.generate_motion_rep(
            inputs, target_x, static_gt
        )

        vis_mask = length_to_mask(length, L)  # (B, L)
        
        cond = {
            "y": {
                "text": inputs.get("caption", [""] * B),
                "f_cond": f_cond,
                "mask": vis_mask,
                "length": length,
            },
        }
        if 'encoded_text' in inputs:
            cond['y']['encoded_text'] = inputs['encoded_text']
                
        if regression_only:
            t, t_weights = self.schedule_sampler.sample(motion.shape[0], motion.device)
            t = (torch.ones_like(t) * 999).long()
            t_weights = torch.ones_like(t_weights)
            x_t = torch.zeros_like(motion)

            denoise_out = denoiser(
                x_t, self.train_diffusion._scale_timesteps(t), return_aux=False, **cond
            )
            sample = denoise_out["pred_x"]
            pred_cam = denoise_out["pred_cam"]
            static_conf_logits = denoise_out["static_conf_logits"]

        else:
            if self.args.get("use_cfg_sampler_for_text", False) and eval_text_only:
                denoiser = ClassifierFreeSampleModel(denoiser)
                cond["y"]["scale"] = self.model_cfg.diffusion.guidance_param
            diff_sampler = self.model_cfg.diffusion.get("sampler", "ddim")
            if diff_sampler == "ddim":
                sample_fn = diffusion.ddim_sample_loop_with_aux
                kwargs = {"eta": self.model_cfg.diffusion.ddim_eta}
            else:
                raise NotImplementedError(f"Sampler {diff_sampler} not implemented")
            
            if self.args.get('force_zero_noise', False):
                noise = torch.zeros_like(motion)
            else:    
                if eval_text_only:
                    noise = torch.randn_like(motion)
                else:
                    noise = torch.zeros_like(motion)
            
            if self.args.get("return_mid", False):
                kwargs['return_mid'] = True

            samples_out = sample_fn(
                denoiser,
                motion.shape,
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
            if "pred_cam" in self.args.out_attr:
                pred_cam = samples_out["pred_cam"]
            if "cam_t_vel" in self.args.out_attr:
                pred_cam_t_vel = samples_out["pred_cam_t_vel"]
            if 'cam_scale' in self.args.out_attr:
                pred_cam_scale = samples_out["pred_cam_scale"]

            static_conf_logits = samples_out["static_conf_logits"]
            sample = samples_out["pred_x"]
            # sample = inputs['gt']
            # static_conf_logits = inputs['static_gt']

        output = {
            "pred_x": sample,
            "static_conf_logits": static_conf_logits,
        }

        if self.args.get("return_mid", False):
            output['intermediate_pred_x'] = [sample_i['pred_x'] for sample_i in samples_out['intermediates']]

        if 'pred_cam' in self.args.out_attr:
            output["pred_cam"] = pred_cam
        if 'cam_t_vel' in self.args.out_attr:
            output["pred_cam_t_vel"] = pred_cam_t_vel
        if 'cam_scale' in self.args.out_attr:
            output["pred_cam_scale"] = pred_cam_scale
        return output

    def forward(self, inputs, train=False, postproc=False, static_cam=False, mode=None):
        if train:
            return self.forward_train(inputs, train, postproc, static_cam, mode=mode)
        else:
            return self.forward_test(inputs, train, postproc, static_cam, mode=mode)
