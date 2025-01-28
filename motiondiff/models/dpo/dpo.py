import numpy as np
import torch

from motiondiff.diffusion.gaussian_diffusion import ModelMeanType
from motiondiff.diffusion.nn import sum_flat
from motiondiff.models.mdm.mdm import MDM
from motiondiff.utils.config import create_config
from motiondiff.utils.scheduler import update_scheduled_params
from motiondiff.utils.tools import (
    find_last_version,
    get_checkpoint_path,
    import_type_from_str,
    load_ema_weights_from_checkpoint,
)
from motiondiff.utils.torch_utils import tensor_to


class DPO(MDM):
    def __init__(self, cfg, is_inference=False, preload_checkpoint=True):
        super().__init__(cfg, is_inference, preload_checkpoint)
        if not is_inference:
            self.load_ref_model(cfg.ref_model)
        return

    def load_ref_model(self, ref_model_specs):
        ref_cfg = create_config(ref_model_specs.cfg, tmp=False, training=True)
        self.ext_models["ref_model"] = import_type_from_str(ref_cfg.model.type)(
            ref_cfg, is_inference=True, preload_checkpoint=False
        )
        version = find_last_version(ref_cfg.cfg_dir, cp=ref_model_specs.cp)
        checkpoint_dir = f"{ref_cfg.cfg_dir}/version_{version}/checkpoints"
        model_cp, cp_name = get_checkpoint_path(
            checkpoint_dir, ref_model_specs.cp, return_name=True
        )
        print(f"loading reference model from checkpoint {model_cp}")
        checkpoint = torch.load(model_cp, map_location="cpu")
        self.ext_models["ref_model"].load_state_dict(checkpoint["state_dict"])
        self.ext_models["ref_model"].train()

    def augment_data_text(self, data):
        for key in ["cond_w", "cond_l"]:
            new_text_dict = []
            for text in data[key]["y"]["text"]:
                if text in self.aug_text_dict:
                    new_text = self.aug_text_dict[text][
                        np.random.randint(len(self.aug_text_dict[text]))
                    ]
                    new_text_dict.append(new_text)
                else:
                    new_text_dict.append(text)
            data[key]["y"]["text"] = new_text_dict

    def training_step(self, batch, batch_idx):
        schedule = self.cfg.get("schedule", dict())
        update_scheduled_params(self, schedule, self.global_step)

        data = {}
        motion_w, cond_w, motion_l, cond_l = batch
        if motion_w.device != self.device:
            motion_w, cond_w, motion_l, cond_l = tensor_to(
                [motion_w, cond_w, motion_l, cond_l], device=self.device
            )
        motion_w = self.convert_motion_rep(motion_w)
        motion_l = self.convert_motion_rep(motion_l)

        data["motion_w"], data["cond_w"] = motion_w, cond_w
        data["motion_l"], data["cond_l"] = motion_l, cond_l
        data["mask_w"] = cond_w["y"]["mask"]
        data["mask_l"] = cond_l["y"]["mask"]

        if self.augment_text:
            self.augment_data_text(data)

        t, t_weights = self.schedule_sampler.sample(motion_w.shape[0], self.device)
        data = self.get_diffusion_pred_target(data, t)
        loss, loss_dict, loss_uw_dict = self.compute_loss(data, t, t_weights)

        self.log("loss/train_all", loss, on_step=True, on_epoch=True, sync_dist=True)
        for key, val in loss_uw_dict.items():
            self.log(
                f"loss/train_{key}", val, on_step=True, on_epoch=True, sync_dist=True
            )
        return loss

    def get_diffusion_pred_target(self, data, t, noise=None):
        diffusion = self.train_diffusion if self.training else self.test_diffusion

        x_w = data["motion_w"]
        x_l = data["motion_l"]
        if noise is None:
            noise = torch.randn_like(x_w)
        x_t_w = diffusion.q_sample(x_w, t, noise=noise)
        x_t_l = diffusion.q_sample(x_l, t, noise=noise)

        data["model_pred_w"] = self.denoiser(
            x_t_w, diffusion._scale_timesteps(t), **data["cond_w"]
        )
        data["model_pred_l"] = self.denoiser(
            x_t_l, diffusion._scale_timesteps(t), **data["cond_l"]
        )

        with torch.no_grad():
            data["ref_pred_w"] = self.ext_models["ref_model"].denoiser(
                x_t_w, diffusion._scale_timesteps(t), **data["cond_w"]
            )
            data["ref_pred_l"] = self.ext_models["ref_model"].denoiser(
                x_t_l, diffusion._scale_timesteps(t), **data["cond_l"]
            )

        if diffusion.model_mean_type == ModelMeanType.PREVIOUS_X:
            data["target_w"] = diffusion.q_posterior_mean_variance(
                x_start=x_w, x_t=x_t_w, t=t
            )[0]
            data["target_l"] = diffusion.q_posterior_mean_variance(
                x_start=x_l, x_t=x_t_l, t=t
            )[0]
        elif diffusion.model_mean_type == ModelMeanType.START_X:
            data["target_w"] = x_w
            data["target_l"] = x_l
        elif diffusion.model_mean_type == ModelMeanType.EPSILON:
            data["target_w"] = data["target_l"] = noise
        else:
            raise NotImplementedError

        return data

    def compute_loss(self, data, t, t_weights):
        def dpo_loss(cfg):
            beta = cfg.get("beta", 1000.0)
            use_t_weights = cfg.get("use_t_weights", False)

            def comp_err(x_pred, x_target, mask):
                loss = (x_pred - x_target) ** 2
                loss = sum_flat(loss * mask.float())
                n_entries = x_pred.shape[1] * x_pred.shape[2]
                non_zero_elements = sum_flat(mask) * n_entries
                non_zero_elements[non_zero_elements == 0] = 1
                loss = loss / non_zero_elements
                return loss

            model_w_err = comp_err(
                data["model_pred_w"], data["target_w"], data["cond_w"]["y"]["mask"]
            )
            model_l_err = comp_err(
                data["model_pred_l"], data["target_l"], data["cond_l"]["y"]["mask"]
            )
            ref_w_err = comp_err(
                data["ref_pred_w"], data["target_w"], data["cond_w"]["y"]["mask"]
            )
            ref_l_err = comp_err(
                data["ref_pred_l"], data["target_l"], data["cond_l"]["y"]["mask"]
            )
            w_diff = model_w_err - ref_w_err
            l_diff = model_l_err - ref_l_err
            print(w_diff.mean().item(), l_diff.mean().item())
            inside_term = -beta * (w_diff - l_diff)
            loss = -torch.log(torch.sigmoid(inside_term))
            if use_t_weights:
                loss *= t_weights
            loss = loss.mean()
            print(loss)
            return loss, {}

        def dpo_loss_v2(cfg):
            beta = cfg.get("beta", 1000.0)
            use_t_weights = cfg.get("use_t_weights", False)

            def comp_err(x_pred, x_target, mask):
                loss = (x_pred - x_target) ** 2
                loss = sum_flat(loss * mask.float())
                n_entries = x_pred.shape[1] * x_pred.shape[2]
                non_zero_elements = sum_flat(mask) * n_entries
                non_zero_elements[non_zero_elements == 0] = 1
                loss = loss / non_zero_elements
                return loss

            model_ref_w_err = comp_err(
                data["model_pred_w"], data["ref_pred_w"], data["cond_w"]["y"]["mask"]
            )
            model_ref_l_err = comp_err(
                data["model_pred_l"], data["ref_pred_l"], data["cond_l"]["y"]["mask"]
            )
            print(model_ref_w_err.mean().item(), model_ref_l_err.mean().item())
            inside_term = -beta * (model_ref_w_err - model_ref_l_err)
            loss = -torch.log(torch.sigmoid(inside_term))
            if use_t_weights:
                loss *= t_weights
            loss = loss.mean()
            print(loss)
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
