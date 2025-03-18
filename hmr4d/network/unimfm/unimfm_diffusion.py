import importlib
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from skimage.util.shape import view_as_windows
from timm.models.vision_transformer import Mlp

from hmr4d.network.base_arch.transformer.layer import zero_module
from hmr4d.utils.net_utils import length_to_mask
from motiondiff.diffusion.resample import create_named_schedule_sampler
from motiondiff.models.model_util import create_gaussian_diffusion

from .unimfm_cfg_sampler import ClassifierFreeSampleModel
from .utils import encode_text_batch, load_and_freeze_llm


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
        observed_motion_3d_dim=151,
        humanoid_obs_dim=358,
        latent_dim=512,
        dropout=0.1,
        args=None,
        use_cond_exists_as_input=False,
        cond_merge_strategy="add",
        cond_exists_dim=512,
        img_process_modules=None,
        img_process_modules_enable_grad={},
        multi_text_module_cfg={},
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
        self.observed_motion_3d_dim = observed_motion_3d_dim
        self.humanoid_obs_dim = humanoid_obs_dim
        self.s_pred_ind = 0

        # intermediate
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.regression_input_type = self.args.get("regression_input_type", "zero")
        self.use_cond_exists_as_input = use_cond_exists_as_input
        self.cond_merge_strategy = cond_merge_strategy
        self.cond_exists_dim = cond_exists_dim

        if img_process_modules is not None:
            self.img_process_modules = nn.ModuleDict(instantiate(img_process_modules))
        else:
            self.img_process_modules = {}
        self.img_process_modules_enable_grad = img_process_modules_enable_grad

        if "obs" in self.args.in_attr:
            self.learned_pos_linear = nn.Linear(2, 32)
            self.learned_pos_params = nn.Parameter(
                torch.randn(17, 32), requires_grad=True
            )
            self.embed_noisyobs = Mlp(
                17 * 32,
                hidden_features=self.latent_dim * 2,
                out_features=self.latent_dim,
                drop=dropout,
            )
        if "f_cliffcam" in self.args.in_attr:
            self.cliffcam_embedder = nn.Sequential(
                nn.Linear(self.cliffcam_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
            self.learned_cliffcam_params = nn.Parameter(
                torch.randn(latent_dim), requires_grad=True
            )

        if "f_cam_angvel" in self.args.in_attr:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
            self.learned_cam_angvel_params = nn.Parameter(
                torch.randn(latent_dim), requires_grad=True
            )

        if "f_cam_t_vel" in self.args.in_attr:
            self.cam_t_vel_embedder = nn.Sequential(
                nn.Linear(self.cam_t_vel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
            self.learned_cam_t_vel_params = nn.Parameter(
                torch.randn(latent_dim), requires_grad=True
            )

        if "f_imgseq" in self.args.in_attr:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
            )

        if "observed_motion_3d" in self.args.in_attr:
            self.observed_motion_3d_embedder = nn.Sequential(
                nn.Linear(self.observed_motion_3d_dim * 2, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )

        if "humanoid_obs" in self.args.in_attr:
            self.humanoid_obs_embedder = nn.Sequential(
                nn.Linear(self.humanoid_obs_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if "multi_text_embed" in self.args.in_attr:
            text_embed_dim = multi_text_module_cfg.get("text_embed_dim", 1024)
            self.multi_text_embedder = nn.Linear(text_embed_dim, latent_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,  # Input dimension
                nhead=multi_text_module_cfg.get(
                    "nhead", 8
                ),  # Number of attention heads
                dim_feedforward=multi_text_module_cfg.get("dim_feedforward", 2048),
                dropout=dropout,
                batch_first=True,
            )
            self.multi_text_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=multi_text_module_cfg.get("num_layers", 3)
            )

        self.add_features_dim = {
            "humanoid_contact_force": 90,
        }
        self.not_add_features = [
            "obs",
            "f_cliffcam",
            "f_cam_angvel",
            "f_cam_t_vel",
            "f_imgseq",
            "observed_motion_3d",
            "humanoid_obs",
            "humanoid_rgb_obs",
            "multi_text_embed",
        ]
        self.add_feature_embedders = nn.ModuleDict()
        for k in self.args.in_attr:
            if k in self.not_add_features:
                continue
            module = nn.Sequential(
                nn.Linear(self.add_features_dim[k], latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
            self.add_feature_embedders[k] = module

        add_dim = 0
        self.no_exist_keys = ["observed_motion_3d", "multi_text_embed"]
        if self.use_cond_exists_as_input:
            if cond_merge_strategy == "add":
                self.cond_exists_embedder = nn.ModuleDict()
                for k in self.args.in_attr:
                    if k not in self.no_exist_keys:
                        self.cond_exists_embedder[k] = nn.Sequential(
                            nn.Linear(latent_dim + 1, latent_dim),
                            nn.SiLU(),
                            zero_module(nn.Linear(latent_dim, latent_dim)),
                        )
            elif cond_merge_strategy == "concat":
                add_dim = cond_exists_dim

        if cond_merge_strategy == "concat":
            # self.cond_merger = nn.Linear(len(self.args.in_attr) * (latent_dim + add_dim), latent_dim)
            in_dim = len(self.args.in_attr) * (latent_dim + add_dim)
            for x in self.no_exist_keys:
                if x in self.args.in_attr:
                    in_dim -= add_dim
            self.cond_merger = nn.Sequential(
                nn.Linear(in_dim, latent_dim),
                nn.SiLU(),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )

        self.denoiser = instantiate(self.model_cfg.denoiser)
        self.init_diffusion()
        self.text_encoder, self.tokenizer = None, None
        return

    def init_diffusion(self):
        self.train_diffusion = create_gaussian_diffusion(
            self.model_cfg.diffusion, training=True
        )
        self.test_diffusion = create_gaussian_diffusion(
            self.model_cfg.diffusion, training=False
        )
        text_only_diffusion = deepcopy(self.model_cfg.diffusion)
        text_only_diffusion.test_timestep_respacing = self.model_cfg.diffusion.get(
            "text_only_test_timestep_respacing", "50"
        )
        print(
            f"Text only test timestep respacing: {text_only_diffusion.test_timestep_respacing}"
        )
        self.test_text_only_diffusion = create_gaussian_diffusion(
            text_only_diffusion, training=False
        )
        self.schedule_sampler = create_named_schedule_sampler(
            self.model_cfg.diffusion.schedule_sampler_type, self.train_diffusion
        )
        return

    def generate_motion_rep(self, inputs, target_x, first_k_frames=None):
        f_condition = inputs["f_condition"]
        f_condition_exists = inputs["f_condition_exists"]
        length = inputs["length"]
        end_fr = first_k_frames if first_k_frames is not None else None
        if first_k_frames is not None:
            length = length.clamp(max=first_k_frames)

        f_cond_dict = {}
        if "obs" in f_condition:
            obs = f_condition["obs"][:, :end_fr]
            B, L, J, C = obs.shape
            assert J == 17 and C == 3
            obs = obs.clone()
            visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
            obs[~visible_mask[..., 0]] = 0  # set low-conf to all zeros
            f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
            f_obs = (
                f_obs * visible_mask
                + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
            )  # (B, L, J, 32)
            f_obs = self.embed_noisyobs(
                f_obs.view(B, L, -1)
            )  # (B, L, J*32) -> (B, L, C)
            f_cond_dict["obs"] = f_obs

        if "f_cliffcam" in f_condition:
            f_cliffcam = f_condition["f_cliffcam"][:, :end_fr]  # (B, L, 3)
            f_cliffcam = self.cliffcam_embedder(f_cliffcam)
            f_cond_dict["f_cliffcam"] = f_cliffcam
        if "f_cam_angvel" in f_condition:
            f_cam_angvel = f_condition["f_cam_angvel"][:, :end_fr]  # (B, L, 6)
            f_cam_angvel = self.cam_angvel_embedder(f_cam_angvel)
            f_cond_dict["f_cam_angvel"] = f_cam_angvel
        if "f_cam_t_vel" in f_condition:
            f_cam_t_vel = f_condition["f_cam_t_vel"][:, :end_fr]  # (B, L, 3)
            f_cam_t_vel = self.cam_t_vel_embedder(f_cam_t_vel)
            f_cond_dict["f_cam_t_vel"] = f_cam_t_vel
        if "f_imgseq" in f_condition:
            f_imgseq = f_condition["f_imgseq"][:, :end_fr]  # (B, L, C)
            f_imgseq = self.imgseq_embedder(f_imgseq)
            f_cond_dict["f_imgseq"] = f_imgseq
        if "observed_motion_3d" in f_condition:
            motion_mask_3d = inputs.get(
                "motion_mask_3d", torch.zeros_like(f_condition["observed_motion_3d"])
            )[:, :end_fr]
            f_observed_motion_3d = torch.cat(
                [f_condition["observed_motion_3d"][:, :end_fr], motion_mask_3d], dim=-1
            )
            f_observed_motion_3d = self.observed_motion_3d_embedder(
                f_observed_motion_3d
            )
            f_cond_dict["observed_motion_3d"] = f_observed_motion_3d

        if "humanoid_obs" in f_condition:
            f_humanoid_obs = f_condition["humanoid_obs"][
                :, :end_fr
            ]  # (B, L, humanoid_obs_dim)
            f_humanoid_obs = self.humanoid_obs_embedder(f_humanoid_obs)
            f_cond_dict["humanoid_obs"] = f_humanoid_obs

        for k in self.args.in_attr:
            if k in self.not_add_features:
                continue
            f_cond_dict[k] = self.add_feature_embedders[k](f_condition[k][:, :end_fr])

        for k in self.img_process_modules:
            enable_grad = self.img_process_modules_enable_grad.get(k, False)
            with torch.set_grad_enabled(enable_grad):
                B = f_condition[k].shape[0]
                img_batched = f_condition[k][:, :end_fr].reshape(
                    -1, *f_condition[k].shape[2:]
                )
                # Visualize batched images using PIL
                # from PIL import Image
                # import os
                # os.makedirs('out/vis/batched_imgs', exist_ok=True)
                # for i in range(img_batched.shape[0]):  # Save up to 10 images
                #     img = img_batched[i].detach().cpu()
                #     img = img.permute(1, 2, 0)
                #     img = (img * 255).numpy().astype('uint8')
                #     img = Image.fromarray(img)
                #     img.save(f'out/vis/batched_imgs/batch_{i}.png')
                f_cond_dict[k] = self.img_process_modules[k](img_batched)
                f_cond_dict[k] = f_cond_dict[k].view(B, -1, *f_cond_dict[k].shape[1:])

        if "multi_text_embed" in f_condition:
            x_text = f_condition["multi_text_embed"]
            x_text = x_text.reshape(-1, *x_text.shape[2:])
            x_text = self.multi_text_embedder(x_text)
            x_text = self.multi_text_transformer(x_text)
            x_text = x_text.mean(dim=1)
            x_text = x_text.reshape(B, -1, *x_text.shape[1:])
            text_label_id = inputs["text_label_ids"][:, :end_fr]
            batch_indices = torch.arange(B, device=x_text.device)[:, None]  # [B, 1]
            batch_indices = batch_indices.expand(-1, text_label_id.size(1))  # [B, F]
            text_features = x_text[batch_indices, text_label_id]
            f_cond_dict["multi_text_embed"] = text_features

        if self.cond_merge_strategy == "add":
            if self.use_cond_exists_as_input:
                for k in f_cond_dict:
                    if k not in self.no_exist_keys:
                        f_cond_dict[k] = torch.cat(
                            [
                                f_cond_dict[k],
                                f_condition_exists[k][:, :end_fr, None].float(),
                            ],
                            dim=-1,
                        )
                        f_cond_dict[k] = self.cond_exists_embedder[k](f_cond_dict[k])
            f_cond = sum(f_cond_dict.values())
        elif self.cond_merge_strategy == "concat":
            f_cond = torch.cat(list(f_cond_dict.values()), dim=-1)
            f_cond_exists = torch.cat(
                [
                    f_condition_exists[k][:, :end_fr, None]
                    .float()
                    .repeat(1, 1, self.cond_exists_dim)
                    for k in f_cond_dict
                    if k not in self.no_exist_keys
                ],
                dim=-1,
            )
            if self.use_cond_exists_as_input:
                f_cond = torch.cat([f_cond, f_cond_exists], dim=-1)
            f_cond = self.cond_merger(f_cond)

        vis_mask = length_to_mask(length, f_cond.shape[1])[:, :end_fr]  # (B, L)

        motion = target_x * vis_mask[..., None]
        motion = motion[:, :end_fr]
        return f_cond, motion

    def forward_train(
        self, inputs, train=False, postproc=False, static_cam=False, mode=None
    ):
        assert self.training, "forward_train should only be called during training"
        diffusion = self.train_diffusion if self.training else self.test_diffusion
        length = inputs["length"]
        target_x = inputs["target_x"]

        f_cond, motion = self.generate_motion_rep(inputs, target_x)
        B, L, _ = motion.shape
        vis_mask = length_to_mask(length, L)  # (B, L)
        valid_mask = inputs["mask"]["valid"]
        assert (vis_mask == valid_mask).all()

        denoiser_kwargs = {
            "y": {
                "text": inputs.get("caption", [""] * B),
                "f_cond": f_cond,
                "mask": vis_mask,
                "length": length,
            },
            "inputs": inputs,
        }
        if "encoded_text" in inputs:
            denoiser_kwargs["y"]["encoded_text"] = inputs["encoded_text"]
        if "observed_motion_3d" in inputs:
            denoiser_kwargs["observed_motion_3d"] = inputs["observed_motion_3d"]
            denoiser_kwargs["motion_mask_3d"] = inputs["motion_mask_3d"]
            denoiser_kwargs["rm_text_flag"] = inputs["rm_text_flag"]

        if mode == "regression":
            t = (
                (torch.ones(B) * (diffusion.original_num_steps - 1))
                .long()
                .to(motion.device)
            )
            t_weights = torch.ones(B).to(motion.device)
            if self.regression_input_type == "zero":
                x_t = torch.zeros_like(motion)
            elif self.regression_input_type == "normal":
                x_t = torch.randn_like(motion)
            else:
                raise ValueError(
                    f"Unsupported regression_input_type: {self.regression_input_type}"
                )
        elif mode == "diffusion":
            t, t_weights = self.schedule_sampler.sample(motion.shape[0], motion.device)
            if "regression_outputs" in inputs:
                pred_x_start_regression = inputs["regression_outputs"]["model_output"][
                    "pred_x_start"
                ].detach()
            else:
                pred_x_start_regression = torch.zeros_like(motion)
            x_start_reg = pred_x_start_regression
            if self.args.get("inpaint_x_start_gt", True):
                inpaint_mask = torch.ones_like(pred_x_start_regression)
                inpaint_mask = inpaint_mask * valid_mask[:, :, None]
                x_start_gt = motion.clone() * inpaint_mask + pred_x_start_regression * (
                    1 - inpaint_mask
                )
            else:
                x_start_gt = motion.clone()
            regression_mask = (
                torch.rand(B).to(motion.device) < self.args.use_regression_outputs_prob
            ).float()
            if "text_only" in inputs and self.args.get("use_gt_for_text_only", True):
                regression_mask[inputs["text_only"]] = 0
            x_start = x_start_reg * regression_mask[:, None, None] + x_start_gt * (
                1 - regression_mask[:, None, None]
            )
            noise = torch.randn_like(x_start)
            x_t = self.train_diffusion.q_sample(x_start.clone(), t, noise=noise)

        denoise_out = self.denoiser(
            x_t, diffusion._scale_timesteps(t), return_aux=False, **denoiser_kwargs
        )

        output = {
            "target_x_start": motion,
            "t_weights": t_weights,
        }
        output.update(denoise_out)
        for x in self.args.out_attr:
            assert x in output, f"Output {x} not found in denoise_out"

        return output

    def forward_train_2d(self, inputs, mode):
        assert self.training, "forward_train_2d should only be called during training"
        diffusion = self.train_diffusion

        length = inputs["length"]  # (B,) effective length of each sample

        obs = inputs["f_condition"]["obs"]
        B, L = obs.shape[:2]

        mdim = self.endecoder.get_motion_dim()
        target_x = torch.zeros(B, L, mdim).to(obs)

        f_cond, motion = self.generate_motion_rep(inputs, target_x)
        # Setup length and make padding mask
        vis_mask = length_to_mask(length, L)  # (B, L)

        denoiser_kwargs = {
            "y": {
                "text": inputs.get("caption", [""] * B),
                "f_cond": f_cond,
                "mask": vis_mask,
                "length": length,
            },
            "inputs": inputs,
        }
        if "encoded_text" in inputs:
            denoiser_kwargs["y"]["encoded_text"] = inputs["encoded_text"]
        if "observed_motion_3d" in inputs:
            denoiser_kwargs["observed_motion_3d"] = inputs["observed_motion_3d"]
            denoiser_kwargs["motion_mask_3d"] = inputs["motion_mask_3d"]
            denoiser_kwargs["rm_text_flag"] = inputs["rm_text_flag"]

        if mode == "regression":
            t = (
                (torch.ones(B) * (diffusion.original_num_steps - 1))
                .long()
                .to(motion.device)
            )
            t_weights = torch.ones(B).to(motion.device)
            if self.regression_input_type == "zero":
                x_t = torch.zeros_like(motion)
            elif self.regression_input_type == "normal":
                x_t = torch.randn_like(motion)
            else:
                raise ValueError(
                    f"Unsupported regression_input_type: {self.regression_input_type}"
                )
        elif mode == "diffusion":
            t, t_weights = self.schedule_sampler.sample(motion.shape[0], motion.device)
            pred_x_start_regression = inputs["regression_outputs"]["2d_model_output"][
                "pred_x_start"
            ].detach()
            x_start = pred_x_start_regression
            noise = torch.randn_like(x_start)
            x_t = self.train_diffusion.q_sample(x_start.clone(), t, noise=noise)

        denoise_out = self.denoiser(
            x_t, diffusion._scale_timesteps(t), return_aux=False, **denoiser_kwargs
        )

        output = {
            "t_weights": t_weights,
        }
        output.update(denoise_out)
        for x in self.args.out_attr:
            assert x in output, f"Output {x} not found in denoise_out"
        return output

    def forward_test(
        self,
        inputs,
        train=False,
        postproc=False,
        static_cam=False,
        progress=False,
        mode=None,
    ):
        assert not self.training, "forward_test should only be called during inference"
        eval_text_only = inputs.get("eval_text_only", False)
        diffusion = (
            self.test_text_only_diffusion if eval_text_only else self.test_diffusion
        )
        denoiser = self.denoiser
        length = (
            torch.ones_like(inputs["length"]) * self.max_len
        )  # use max length for test
        regression_only = self.args.get("regression_only", False) and not eval_text_only
        if self.args.get("force_regression_only", False):
            regression_only = True

        f_condition = inputs["f_condition"]

        test_motion_len = self.args.get("test_motion_len", None)
        if test_motion_len is not None and test_motion_len > self.max_len:
            L = test_motion_len
            length = torch.ones_like(length) * L
            f_condition_exists = inputs["f_condition_exists"]
            for k in f_condition:
                f_condition[k] = torch.cat(
                    [
                        f_condition[k],
                        f_condition[k][:, [-1]].repeat_interleave(
                            L - f_condition[k].shape[1], dim=1
                        ),
                    ],
                    dim=1,
                )
                f_condition_exists[k] = torch.cat(
                    [
                        f_condition_exists[k],
                        f_condition_exists[k][:, [-1]].repeat_interleave(
                            L - f_condition_exists[k].shape[1], dim=1
                        ),
                    ],
                    dim=1,
                )
            for k in ["bbx_xys", "K_fullimg", "cam_angvel"]:
                inputs[k] = torch.cat(
                    [
                        inputs[k],
                        inputs[k][:, [-1]].repeat_interleave(
                            L - inputs[k].shape[1], dim=1
                        ),
                    ],
                    dim=1,
                )

        obs = f_condition["obs"]
        B, L = obs.shape[:2]

        mdim = self.endecoder.get_motion_dim()
        target_x = torch.zeros(B, L, mdim).to(obs)

        f_cond, motion = self.generate_motion_rep(inputs, target_x)

        vis_mask = length_to_mask(length, L)  # (B, L)

        denoiser_kwargs = {
            "y": {
                "text": inputs.get("caption", [""] * B),
                "f_cond": f_cond,
                "mask": vis_mask,
                "length": length,
            },
            "inputs": inputs,
        }
        if "encoded_text" in inputs:
            denoiser_kwargs["y"]["encoded_text"] = inputs["encoded_text"]
        if "meta" in inputs and "multi_text_data" in inputs["meta"][0]:
            denoiser_kwargs["y"]["multi_text_data"] = inputs["meta"][0][
                "multi_text_data"
            ]
        if "observed_motion_3d" in inputs:
            denoiser_kwargs["observed_motion_3d"] = inputs["observed_motion_3d"]
            denoiser_kwargs["motion_mask_3d"] = inputs["motion_mask_3d"]
            denoiser_kwargs["rm_text_flag"] = inputs.get("rm_text_flag", None)

        if regression_only:
            t, t_weights = self.schedule_sampler.sample(motion.shape[0], motion.device)
            t = (torch.ones_like(t) * diffusion.original_num_steps).long()
            t_weights = torch.ones_like(t_weights)
            x_t = torch.zeros_like(motion)

            denoise_out = denoiser(
                x_t,
                self.train_diffusion._scale_timesteps(t),
                return_aux=False,
                **denoiser_kwargs,
            )
        else:
            if self.args.get("use_cfg_sampler_for_text", False) and eval_text_only:
                denoiser = ClassifierFreeSampleModel(denoiser)
                denoiser_kwargs["y"]["scale"] = self.model_cfg.diffusion.guidance_param
            diff_sampler = self.model_cfg.diffusion.get("sampler", "ddim")
            if diff_sampler == "ddim":
                sample_fn = diffusion.ddim_sample_loop_with_aux
                kwargs = {"eta": self.model_cfg.diffusion.ddim_eta}
            else:
                raise NotImplementedError(f"Sampler {diff_sampler} not implemented")

            if self.args.get("force_zero_noise", False):
                noise = torch.zeros_like(motion)
            else:
                if eval_text_only:
                    noise = torch.randn_like(motion)
                else:
                    noise = torch.zeros_like(motion)

            if self.args.get("return_mid", False):
                kwargs["return_mid"] = True

            denoise_out = sample_fn(
                denoiser,
                motion.shape,
                clip_denoised=False,
                model_kwargs=denoiser_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=progress,
                dump_steps=None,
                noise=noise,
                const_noise=False,
                **kwargs,
            )

        output = denoise_out.copy()

        if self.args.get("return_mid", False):
            output["intermediate_pred_x"] = [
                sample_i["pred_x"] for sample_i in denoise_out["intermediates"]
            ]

        for x in self.args.out_attr:
            assert x in output, f"Output {x} not found in denoise_out"
        return output

    def forward_test_humanoid(
        self,
        inputs,
        train=False,
        postproc=False,
        static_cam=False,
        progress=False,
        mode=None,
        test_mode=None,
        normalizer_stats=None,
    ):
        assert not self.training, "forward_test should only be called during inference"
        eval_text_only = inputs.get("eval_text_only", False)
        diffusion = (
            self.test_text_only_diffusion if eval_text_only else self.test_diffusion
        )
        denoiser = self.denoiser
        length = inputs["length"]  # (B,) effective length of each sample
        regression_only = self.args.get("regression_only", False) and not eval_text_only
        if self.args.get("force_regression_only", False):
            regression_only = True
        if (
            self.args.get("use_cfg_sampler_for_text", False)
            and eval_text_only
            and not regression_only
        ):
            denoiser = ClassifierFreeSampleModel(denoiser)

        f_condition = inputs["f_condition"]
        f_condition_exists = inputs["f_condition_exists"]
        humanoid_obs = f_condition["humanoid_obs"]
        B, L = humanoid_obs.shape[:2]
        mdim = self.endecoder.get_motion_dim()
        humanoid = inputs["humanoid"]
        obs_keys = [
            x
            for x in self.args.in_attr
            if x in ["humanoid_obs", "humanoid_rgb_obs", "humanoid_contact_force"]
        ]
        cur_obs = {k: None for k in obs_keys}
        outputs_all = []
        decode_dict_all = []

        humanoid_use_text = self.args.get("humanoid_use_text", False)
        humanoid_use_multi_text = self.args.get("humanoid_use_multi_text", False)
        if humanoid_use_text or humanoid_use_multi_text:
            if self.text_encoder is None:
                self.text_encoder, self.tokenizer = load_and_freeze_llm("t5-3b")
                self.text_encoder.eval()
            self.text_encoder.to(humanoid_obs.device)
        last_text_label = None
        encoded_humanoid_text = None
        if humanoid_use_multi_text:
            cur_multi_text_embed = torch.zeros_like(f_condition["multi_text_embed"])
            cur_text_label_ids = torch.zeros_like(inputs["text_label_ids"])
            last_text_label_ids = None

        def policy(obs_dict, extras):
            nonlocal \
                cur_obs, \
                encoded_humanoid_text, \
                last_text_label, \
                cur_multi_text_embed, \
                last_text_label_ids, \
                cur_text_label_ids
            for k in obs_keys:
                if k == "humanoid_obs":
                    if self.args.get("humanoid_obs_inc_task", False):
                        obs_k = torch.cat(
                            [
                                obs_dict["self_obs"]["policy"],
                                obs_dict["task_obs"]["policy"],
                            ],
                            dim=1,
                        )
                    else:
                        obs_k = obs_dict["self_obs"]["policy"]
                elif k == "humanoid_rgb_obs":
                    obs_rgb_raw = (
                        extras["rgb_raw_new"].permute(0, 3, 1, 2).contiguous() / 255.0
                    )
                    obs_rgb_obs = obs_dict["task_obs"]["rgb"]
                    if self.args.get("humanoid_use_rgb_raw", False):
                        obs_k = obs_rgb_raw
                    else:
                        obs_k = obs_rgb_obs
                    # obs_k = obs_rgb_obs
                    # diff = obs_rgb_raw - obs_rgb_obs
                    # print('rgb diff', diff.max(), diff.min(), diff.mean())
                    # Save both raw and processed RGB images
                    # if True:
                    #     from PIL import Image
                    #     import os
                    #     num_fr = cur_obs["humanoid_obs"].shape[1]
                    #     # Save rgb_raw (CHW -> HWC format)
                    #     rgb_raw = obs_rgb_raw[0].permute(1, 2, 0).cpu().numpy()
                    #     rgb_raw = (rgb_raw * 255).astype(np.uint8)
                    #     img_raw = Image.fromarray(rgb_raw)
                    #     fname = f'out/vis/rgb_raw/step{num_fr}.png'
                    #     os.makedirs(os.path.dirname(fname), exist_ok=True)
                    #     img_raw.save(fname)

                    #     # Save rgb_obs (CHW -> HWC format)
                    #     rgb_obs = obs_rgb_obs[0].permute(1, 2, 0).cpu().numpy()
                    #     rgb_obs = (rgb_obs * 255).astype(np.uint8)
                    #     img_obs = Image.fromarray(rgb_obs)
                    #     fname = f'out/vis/rgb_obs/step{num_fr}.png'
                    #     os.makedirs(os.path.dirname(fname), exist_ok=True)
                    #     img_obs.save(fname)
                elif k == "humanoid_contact_force":
                    obs_k = obs_dict["task_obs"]["contact_forces"]
                if k in normalizer_stats:
                    obs_k = (
                        obs_k - normalizer_stats[k]["mean"].to(obs_k)
                    ) / normalizer_stats[k]["std"].to(obs_k)
                obs_k = obs_k.unsqueeze(1)
                if cur_obs[k] is None:
                    cur_obs[k] = obs_k
                else:
                    cur_obs[k] = torch.cat([cur_obs[k], obs_k], dim=1)

            num_fr = cur_obs["humanoid_obs"].shape[1]
            target_x = torch.zeros(B, num_fr, mdim).to(obs_k.device)
            for k in cur_obs:
                f_condition[k][:, :num_fr, :] = cur_obs[k]
                f_condition_exists[k][:, :num_fr] = True

            text_label = extras.get("text_label", None)
            if humanoid_use_multi_text:
                # print('before',text_label.tolist(), last_text_label_ids.tolist() if last_text_label_ids is not None else None, cur_text_label_ids[:, :num_fr].tolist())
                print(
                    text_label.tolist(),
                    last_text_label_ids.tolist()
                    if last_text_label_ids is not None
                    else None,
                )
                if not (
                    last_text_label is not None
                    and (last_text_label == text_label).all()
                ):
                    text_embed = encode_text_batch(
                        text_label,
                        self.text_encoder,
                        self.tokenizer,
                        device=humanoid_obs.device,
                    )
                    if last_text_label is not None:
                        ind = (last_text_label != text_label).nonzero()[0]
                        cur_multi_text_embed[ind, last_text_label_ids[ind] + 1] = (
                            text_embed[ind, :]
                        )
                        new_text_label_ids = last_text_label_ids.clone()
                        new_text_label_ids[ind] += 1
                    else:
                        cur_multi_text_embed[:, 0] = text_embed
                        new_text_label_ids = (
                            torch.zeros(
                                B,
                            )
                            .long()
                            .to(humanoid_obs.device)
                        )
                else:
                    new_text_label_ids = last_text_label_ids
                cur_text_label_ids[:, num_fr - 1] = new_text_label_ids
                last_text_label = text_label
                last_text_label_ids = new_text_label_ids
                f_condition["multi_text_embed"][:] = cur_multi_text_embed
                inputs["text_label_ids"][:] = cur_text_label_ids
                # print('after', text_label.tolist(), last_text_label_ids.tolist() if last_text_label_ids is not None else None, cur_text_label_ids[:, :num_fr].tolist())

            elif humanoid_use_text:
                print(text_label.tolist())
                if (
                    encoded_humanoid_text is None
                    or last_text_label is None
                    or last_text_label != text_label
                ):
                    encoded_humanoid_text = encode_text_batch(
                        text_label.tolist(),
                        self.text_encoder,
                        self.tokenizer,
                        device=humanoid_obs.device,
                    )
                    last_text_label = text_label

            f_cond, motion = self.generate_motion_rep(
                inputs, target_x, first_k_frames=num_fr
            )
            length_new = length.clamp(max=num_fr)

            vis_mask = length_to_mask(length_new, num_fr)  # (B, L)

            denoiser_kwargs = {
                "y": {
                    "text": inputs.get("caption", [""] * B),
                    "f_cond": f_cond,
                    "mask": vis_mask,
                    "length": length_new,
                },
                "inputs": inputs,
            }
            if "encoded_text" in inputs:
                denoiser_kwargs["y"]["encoded_text"] = inputs["encoded_text"]
            if encoded_humanoid_text is not None:
                denoiser_kwargs["y"]["encoded_text"] = encoded_humanoid_text

            if regression_only:
                t, t_weights = self.schedule_sampler.sample(
                    motion.shape[0], motion.device
                )
                t = (torch.ones_like(t) * diffusion.original_num_steps).long()
                t_weights = torch.ones_like(t_weights)
                x_t = torch.zeros_like(motion)

                denoise_out = denoiser(
                    x_t,
                    self.train_diffusion._scale_timesteps(t),
                    return_aux=False,
                    **denoiser_kwargs,
                )
            else:
                if self.args.get("use_cfg_sampler_for_text", False) and eval_text_only:
                    denoiser_kwargs["y"]["scale"] = (
                        self.model_cfg.diffusion.guidance_param
                    )
                diff_sampler = self.model_cfg.diffusion.get("sampler", "ddim")
                if diff_sampler == "ddim":
                    sample_fn = diffusion.ddim_sample_loop_with_aux
                    kwargs = {"eta": self.model_cfg.diffusion.ddim_eta}
                else:
                    raise NotImplementedError(f"Sampler {diff_sampler} not implemented")

                if self.args.get("force_zero_noise", False):
                    noise = torch.zeros_like(motion)
                else:
                    if eval_text_only:
                        noise = torch.randn_like(motion)
                    else:
                        noise = torch.zeros_like(motion)

                if self.args.get("return_mid", False):
                    kwargs["return_mid"] = True

                denoise_out = sample_fn(
                    denoiser,
                    motion.shape,
                    clip_denoised=False,
                    model_kwargs=denoiser_kwargs,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=progress,
                    dump_steps=None,
                    noise=noise,
                    const_noise=False,
                    **kwargs,
                )

            output = denoise_out.copy()
            decode_dict = self.endecoder.decode(output["pred_x"])
            if "humanoid_z_action" in decode_dict:
                action = decode_dict["humanoid_z_action"]
            else:
                action = decode_dict["humanoid_clean_action"]
            outputs_all.append(output)
            decode_dict_all.append(decode_dict)
            return action[:, -1, :]

        if test_mode == "humanoid_vla":
            # all_motion_names = humanoid.env.env.get_motion_names()
            # ref_motions = np.random.choice(all_motion_names, size=B, replace=False).tolist()
            ref_motions = None
        else:
            ref_motions = [x["mid"] for x in inputs["meta"]]
        humanoid.run_genmo_player(policy=policy, ref_motions=ref_motions, max_steps=L)

        output = outputs_all[-1]

        if self.args.get("return_mid", False):
            output["intermediate_pred_x"] = [
                sample_i["pred_x"] for sample_i in output["intermediates"]
            ]

        for x in self.args.out_attr:
            assert x in output, f"Output {x} not found in denoise_out"
        return output

    def forward(
        self,
        inputs,
        train=False,
        postproc=False,
        static_cam=False,
        mode=None,
        test_mode=None,
        normalizer_stats=None,
    ):
        if train:
            return self.forward_train(inputs, train, postproc, static_cam, mode=mode)
        else:
            if "humanoid" in test_mode:
                return self.forward_test_humanoid(
                    inputs,
                    train,
                    postproc,
                    static_cam,
                    mode=mode,
                    test_mode=test_mode,
                    normalizer_stats=normalizer_stats,
                )
            else:
                return self.forward_test(inputs, train, postproc, static_cam, mode=mode)
