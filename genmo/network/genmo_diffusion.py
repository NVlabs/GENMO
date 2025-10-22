from copy import deepcopy

import torch
import torch.nn as nn
from hydra.utils import instantiate

from genmo.diffusion_utils.model_util import create_gaussian_diffusion
from genmo.diffusion_utils.resample import create_named_schedule_sampler
from genmo.utils.net_utils import length_to_mask

from .genmo_cfg_sampler import ClassifierFreeSampleModel


class GENMODiffusion(nn.Module):
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
        encoded_music_dim=438,
        encoded_audio_dim=128,
        latent_dim=512,
        dropout=0.1,
        args=None,
        cond_merge_strategy="add",
        cond_exists_dim=512,
        music_mask_prob=0.1,
        img_process_modules=None,
        img_process_modules_enable_grad={},
        multi_text_module_cfg={},
        **kwargs,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.args = args
        self.max_len = max_len

        self.regression_input_type = self.args.get("regression_input_type", "zero")

        self.denoiser = instantiate(self.model_cfg.denoiser)
        self.init_diffusion()
        self.text_encoder, self.tokenizer = None, None

    def init_diffusion(self):
        self.train_diffusion = create_gaussian_diffusion(
            self.model_cfg.diffusion, training=True
        )
        self.test_diffusion = create_gaussian_diffusion(
            self.model_cfg.diffusion, training=False
        )
        gen_only_diffusion = deepcopy(self.model_cfg.diffusion)
        gen_only_diffusion.test_timestep_respacing = self.model_cfg.diffusion.get(
            "gen_only_test_timestep_respacing", "50"
        )
        print(
            f"Gen only test timestep respacing: {gen_only_diffusion.test_timestep_respacing}"
        )
        self.test_gen_only_diffusion = create_gaussian_diffusion(
            gen_only_diffusion, training=False
        )
        self.schedule_sampler = create_named_schedule_sampler(
            self.model_cfg.diffusion.schedule_sampler_type, self.train_diffusion
        )
        return

    def forward_train(self, inputs, mode):
        assert self.training, "forward_train should only be called during training"
        diffusion = self.train_diffusion if self.training else self.test_diffusion
        length = inputs["length"]
        # target_x = inputs["target_x"]
        motion = inputs["motion"]
        f_cond = inputs["f_cond"]
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
            x_start = motion
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
                raise ValueError("No regression outputs found")
                # pred_x_start_regression = torch.zeros_like(motion)
            x_start_reg = pred_x_start_regression.clone()
            x_start = motion.clone()
            x_start[inputs["mask"]["2d_only"]] = x_start_reg[inputs["mask"]["2d_only"]]
            # regression_mask = (
            #     torch.rand(B).to(motion.device) < self.args.use_regression_outputs_prob
            # ).float()
            # if "gen_only" in inputs and self.args.get("use_gt_for_gen_only", True):
            #     regression_mask[inputs["gen_only"]] = 0
            # x_start = x_start_reg * regression_mask[:, None, None] + x_start_gt * (
            #     1 - regression_mask[:, None, None]
            # )
            noise = torch.randn_like(x_start)
            x_t = self.train_diffusion.q_sample(x_start.clone(), t, noise=noise)

        denoise_out = self.denoiser(
            x_t, diffusion._scale_timesteps(t), return_aux=False, **denoiser_kwargs
        )

        output = {
            "target_x_start": x_start,
            "t_weights": t_weights,
        }
        output.update(denoise_out)
        for x in self.args.out_attr:
            assert x in output, f"Output {x} not found in denoise_out"

        return output

    def forward_test(self, inputs, progress=False):
        assert not self.training, "forward_test should only be called during inference"
        diffusion = self.test_gen_only_diffusion

        denoiser = self.denoiser
        length = inputs["length"]
        B, L = inputs["B"], inputs["L"]

        motion = inputs["motion"]
        f_cond, f_uncond = inputs["f_cond"], inputs["f_uncond"]

        vis_mask = length_to_mask(length, L)  # (B, L)

        denoiser_kwargs = {
            "y": {
                "text": inputs.get("caption", [""] * B),
                "f_cond": f_cond,
                "f_uncond": f_uncond,
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

        if self.args.get("use_cfg_sampler_for_gen", False):
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
        elif self.args.get("force_rand_noise", False):
            noise = torch.randn_like(motion)
        else:
            noise = torch.randn_like(motion)

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
            return self.forward_train(inputs, mode=mode)
        else:
            return self.forward_test(inputs)
