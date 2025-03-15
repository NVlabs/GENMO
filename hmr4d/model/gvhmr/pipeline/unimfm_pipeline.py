import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from torch.cuda.amp import autocast

from hmr4d.model.gvhmr.utils import stats_compose
from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
from hmr4d.model.gvhmr.utils.mv2d_utils import coco_joint_parents, draw_motion_2d
from hmr4d.model.gvhmr.utils.postprocess import (
    pp_floor_height,
    pp_static_joint,
    pp_static_joint_cam,
    process_ik,
)
from hmr4d.utils.geo.hmr_cam import (
    compute_bbox_info_bedlam,
    compute_transl_full_cam,
    convert_bbx_xys_to_lurb,
    cvt_to_bi01_p2d,
    get_a_pred_cam,
    get_bbx_xys,
    normalize_kp2d,
    perspective_projection,
    project_to_bi01,
)
from hmr4d.utils.geo.hmr_global import (
    estimate_camscale,
    get_static_joint_mask,
    get_tgtcoord_rootparam,
    rollout_local_transl_vel,
)
from hmr4d.utils.net_utils import gaussian_smooth
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.wis3d_utils import add_motion_as_lines, make_wis3d
from motiondiff.models.mdm.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


class Pipeline(nn.Module):
    def __init__(self, args, args_denoiser3d, **kwargs):
        super().__init__()
        self.args = args
        self.weights = args.weights  # loss weights

        # Networks
        self.denoiser3d = instantiate(args_denoiser3d, _recursive_=False)
        # Log.info(self.denoiser3d)

        # Normalizer
        self.endecoder: EnDecoder = instantiate(args.endecoder_opt, _recursive_=False)
        if self.args.normalize_cam_angvel:
            cam_angvel_stats = stats_compose.cam_angvel["manual"]
            self.register_buffer(
                "cam_angvel_mean",
                torch.tensor(cam_angvel_stats["mean"]),
                persistent=False,
            )
            self.register_buffer(
                "cam_angvel_std",
                torch.tensor(cam_angvel_stats["std"]),
                persistent=False,
            )

        # Load normalizer stats
        self.normalizer_stats = {}
        if "norm_attr_stats" in args:
            for key, stats_path in args.norm_attr_stats.items():
                self.normalizer_stats[key] = torch.load(
                    stats_path, map_location="cpu", weights_only=False
                )

        self.denoiser3d.endecoder = self.endecoder

        self.f_condition_dim = {
            "obs": (17, 3),
            "f_cliffcam": (3,),
            "f_cam_angvel": (6,),
            "f_imgseq": (1024,),
            "observed_motion_3d": (151,),
            "humanoid_obs": (self.args.get("humanoid_obs_dim", 358),),
            "humanoid_rgb_obs": self.args.get("humanoid_rgb_obs_dim", (4, 100, 100)),
            "humanoid_contact_force": (90,),
        }

    def normalize_attr(self, x, key):
        """Normalize input tensor using stored statistics"""
        mean = self.normalizer_stats[key]["mean"].to(x)
        std = self.normalizer_stats[key]["std"].to(x)
        return (x - mean) / std

    # ========== Training ========== #

    def prepare_inputs(self, inputs, train):
        B, L = inputs["B"], inputs["L"]

        if "bbx_xys" in inputs and (
            "use_cliffcam" not in inputs or inputs["use_cliffcam"].any()
        ):
            inputs["f_cliffcam"] = compute_bbox_info_bedlam(
                inputs["bbx_xys"], inputs["K_fullimg"]
            )  # (B, L, 3)
            if "use_cliffcam" in inputs:
                inputs["f_cliffcam"][~inputs["use_cliffcam"]] = 0
        if "cam_angvel" in inputs:
            f_cam_angvel = inputs["cam_angvel"]
            if self.args.normalize_cam_angvel:
                f_cam_angvel = (
                    f_cam_angvel - self.cam_angvel_mean
                ) / self.cam_angvel_std
            inputs["f_cam_angvel"] = f_cam_angvel

        f_condition = dict()
        for k in self.args.in_attr:
            if k in inputs:
                f_condition[k] = inputs[k]
                if k in self.normalizer_stats:
                    f_condition[k] = self.normalize_attr(f_condition[k], k)
            else:
                f_condition[k] = torch.zeros(B, L, *self.f_condition_dim[k]).to(
                    inputs["device"]
                )
        for k in self.args.mask_out_attr:
            f_condition[k] = torch.zeros_like(f_condition[k])
            if k in inputs["f_condition_exists"]:
                inputs["f_condition_exists"][k][:] = False
        f_condition_mask = inputs.get("f_condition_mask", {})
        for k in f_condition_mask:
            if k in f_condition:
                mask = (
                    f_condition_mask[k][:, None, None, None]
                    if k in ["obs"]
                    else f_condition_mask[k][:, None, None]
                )
                f_condition[k] = f_condition[k] * (1 - mask.float())

        if inputs.get("text_mask", None) is not None:
            inputs["encoded_text"] = inputs["encoded_text"] * (
                1 - inputs["text_mask"][:, None, None].float()
            )

        if train and not self.args.get("disable_random_null_condition", False):
            skip_keys = self.args.get("skip_keys_for_null_condition", ["humanoid_obs"])
            f_condition = randomly_set_null_condition(
                f_condition, 0.1, skip_keys=skip_keys
            )

        eval_text_only = inputs.get("eval_text_only", False)
        if eval_text_only:
            for k in f_condition.keys():
                f_condition[k] = torch.zeros_like(f_condition[k])
                if k in inputs["f_condition_exists"]:
                    inputs["f_condition_exists"][k][:] = False

        inputs["f_condition"] = f_condition
        return inputs

    def forward(
        self,
        inputs,
        train=False,
        postproc=False,
        static_cam=False,
        global_step=0,
        mode=None,
        test_mode=None,
    ):
        outputs = dict()
        inputs = self.prepare_inputs(inputs, train)

        # Forward & output
        model_output = self.denoiser3d(
            inputs,
            train=train,
            postproc=postproc,
            static_cam=static_cam,
            mode=mode,
            test_mode=test_mode,
            normalizer_stats=self.normalizer_stats,
        )  # pred_x, pred_cam, static_conf_logits
        decode_dict = self.endecoder.decode(model_output["pred_x"])  # (B, L, C) -> dict
        outputs.update({"model_output": model_output, "decode_dict": decode_dict})
        if "intermediate_pred_x" in model_output:
            int_decode_dict = [
                self.endecoder.decode(pred_x)
                for pred_x in model_output["intermediate_pred_x"]
            ]
            outputs["intermediate_decode_dict"] = int_decode_dict

        # Post-processing
        if "gvhmr" in self.endecoder.feature_arr:
            outputs["pred_smpl_params_incam"] = {
                "body_pose": decode_dict["body_pose"],  # (B, L, 63)
                "betas": decode_dict["betas"],  # (B, L, 10)
                "global_orient": decode_dict["global_orient"],  # (B, L, 3)
                "transl": compute_transl_full_cam(
                    model_output["pred_cam"], inputs["bbx_xys"], inputs["K_fullimg"]
                ),
            }
            if "intermediate_decode_dict" in outputs:
                outputs["intermediate_pred_smpl_params_incam"] = [
                    {
                        "body_pose": int_decode_dict["body_pose"],
                        "betas": int_decode_dict["betas"],
                        "global_orient": int_decode_dict["global_orient"],
                        "transl": compute_transl_full_cam(
                            model_output["pred_cam"],
                            inputs["bbx_xys"],
                            inputs["K_fullimg"],
                        ),
                    }
                    for int_decode_dict in outputs["intermediate_decode_dict"]
                ]

        if not train:
            # if eval_text_only:
            #     inputs["cam_angvel"] = torch.zeros(decode_dict["global_orient_gv"].shape[:2] + (6,), device=decode_dict["global_orient_gv"].device)
            if "gvhmr" in self.endecoder.feature_arr:
                if self.args.get("infer_version", 2) == 2:
                    pred_smpl_params_global = (
                        get_smpl_params_w_Rt_v2(  # This function has for-loop
                            global_orient_gv=decode_dict["global_orient_gv"],
                            local_transl_vel=decode_dict["local_transl_vel"],
                            global_orient_c=decode_dict["global_orient"],
                            cam_angvel=inputs["cam_angvel"],
                        )
                    )
                    outputs["pred_smpl_params_global"] = {
                        "body_pose": decode_dict["body_pose"],
                        "betas": decode_dict["betas"],
                        **pred_smpl_params_global,
                    }
                    if "intermediate_decode_dict" in outputs:
                        intermediate_pred_smpl_params_global = []
                        for int_decode_dict in outputs["intermediate_decode_dict"]:
                            pred_smpl_params_global = get_smpl_params_w_Rt_v2(
                                global_orient_gv=int_decode_dict["global_orient_gv"],
                                local_transl_vel=int_decode_dict["local_transl_vel"],
                                global_orient_c=int_decode_dict["global_orient"],
                                cam_angvel=inputs["cam_angvel"],
                            )
                            pred_smpl_params_global = {
                                "body_pose": int_decode_dict["body_pose"],
                                "betas": int_decode_dict["betas"],
                                **pred_smpl_params_global,
                            }
                            intermediate_pred_smpl_params_global.append(
                                pred_smpl_params_global
                            )
                        outputs["intermediate_pred_smpl_params_global"] = (
                            intermediate_pred_smpl_params_global
                        )
                elif self.args.get("infer_version", 2) == 3:
                    if "vimo_smpl_params" in inputs:
                        vimo_smpl_params = inputs["vimo_smpl_params"]
                        transl_c = vimo_smpl_params["pred_trans_c"].squeeze(2)
                        # transl_c = outputs["pred_smpl_params_incam"]["transl"]
                        slam_scales = inputs["scales"]
                        mean_scale = inputs["mean_scale"]
                        std_slam_scales = slam_scales.std(dim=-1, keepdim=True)
                        conf_scales = (slam_scales - mean_scale).abs() < std_slam_scales
                    else:
                        transl_c = outputs["pred_smpl_params_incam"]["transl"]
                        conf_scales = None

                    if "cam_angvel" in self.args.out_attr:
                        cam_angvel = model_output["cam_angvel"]
                    else:
                        cam_angvel = inputs["cam_angvel"]

                    if "cam_t_vel" in self.args.out_attr:
                        cam_tvel = model_output["pred_cam_t_vel"]
                    else:
                        cam_tvel = inputs["cam_tvel"]

                    if "cam_scale" in self.args.out_attr:
                        cam_scale = model_output["pred_cam_scale"]
                        cam_tvel = inputs["cam_tvel"] * cam_scale

                    (
                        pred_smpl_params_global,
                        pred_T_w2c,
                        pred_smpl_params_body_pose,
                        static_conf_logits,
                    ) = get_smpl_params_w_Rt_v3(  # This function has for-loop
                        endecoder=self.endecoder,
                        global_orient_gv=decode_dict["global_orient_gv"],
                        local_transl_vel=decode_dict["local_transl_vel"],
                        local_transl_c=transl_c,
                        global_orient_c=decode_dict["global_orient"],
                        offset=decode_dict["offset"],
                        cam_angvel=cam_angvel,
                        cam_tvel=cam_tvel,
                        slam_R_w2c=inputs["R_w2c"],
                        static_conf_logits=model_output["static_conf_logits"],
                        conf_scales=conf_scales,
                        betas=decode_dict["betas"],
                        body_pose=decode_dict["body_pose"],
                        slam_scales=slam_scales,
                        mean_scale=mean_scale,
                    )
                    outputs["pred_T_w2c"] = pred_T_w2c
                    outputs["pred_smpl_params_global"] = {
                        # "body_pose": decode_dict["body_pose"],
                        "body_pose": pred_smpl_params_body_pose,
                        "betas": decode_dict["betas"],
                        **pred_smpl_params_global,
                    }
                    model_output["static_conf_logits"] = static_conf_logits
                    decode_dict["body_pose"] = pred_smpl_params_body_pose
                    outputs["pred_smpl_params_global"]["body_pose"] = (
                        pred_smpl_params_body_pose
                    )
                    outputs["pred_smpl_params_incam"]["body_pose"] = (
                        pred_smpl_params_body_pose
                    )
            elif "body_pose" in decode_dict:
                outputs["pred_smpl_params_global"] = {
                    "body_pose": decode_dict["body_pose"],
                    "betas": decode_dict["betas"],
                    "global_orient": decode_dict["global_orient_w"],
                    "transl": decode_dict["transl_w"],
                }
            if "static_conf_logits" in model_output:
                outputs["static_conf_logits"] = model_output["static_conf_logits"]

            if (
                postproc
                and "gvhmr" in self.endecoder.feature_arr
                and self.args.get("infer_version", 2) != 3
            ):  # apply post-processing
                if static_cam:  # extra post-processing to utilize static camera prior
                    outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_cam(
                        outputs, self.endecoder
                    )
                else:
                    outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(
                        outputs, self.endecoder
                    )

                if "intermediate_pred_smpl_params_global" in outputs:
                    for i in range(
                        len(outputs["intermediate_pred_smpl_params_global"])
                    ):
                        outputs["intermediate_pred_smpl_params_global"][i]["transl"] = (
                            pp_floor_height(
                                outputs["intermediate_pred_smpl_params_global"][i],
                                self.endecoder,
                            )
                        )
                body_pose = process_ik(outputs, self.endecoder)
                decode_dict["body_pose"] = body_pose
                outputs["pred_smpl_params_global"]["body_pose"] = body_pose
                outputs["pred_smpl_params_incam"]["body_pose"] = body_pose

            return outputs

        # ========== Compute Loss ========== #
        total_loss = 0
        mask = inputs["mask"]["valid"]  # (B, L)
        text_only = inputs.get("text_only", None)

        # 1. Simple loss: MSE
        if self.weights.get("simple", 1.0) > 0.0:
            pred_x = model_output["pred_x"][..., :151]  # (B, L, C)
            target_x = inputs["target_x"][..., :151]  # (B, L, C)
            simple_loss = F.mse_loss(pred_x, target_x, reduction="none")
            if (
                text_only is not None
                and self.weights.get("text_only_no_reg_loss", False)
                and mode == "regression"
            ):
                simple_loss[text_only] = 0
            mask_simple = (
                mask[:, :, None].expand(-1, -1, pred_x.size(2)).clone()
            )  # (B, L, C)
            if "gvhmr" in self.endecoder.feature_arr:
                mask_simple[inputs["mask"]["spv_incam_only"], :, 142:] = (
                    False  # 3dpw training
                )
            if self.weights.get("simple_loss_local_only", False):
                mask_simple[..., 142:] = False
            simple_loss = (simple_loss * mask_simple).mean()
            total_loss += simple_loss * self.weights.get("simple", 1.0)
            outputs["simple_loss"] = simple_loss

        # 2. Extra loss
        if "gvhmr" in self.endecoder.feature_arr:
            extra_funcs = [
                compute_extra_incam_loss,
                compute_extra_global_loss,
            ]
            for extra_func in extra_funcs:
                extra_loss, extra_loss_dict = extra_func(inputs, outputs, self, mode)
                total_loss += extra_loss
                outputs.update(extra_loss_dict)

        humanoid_loss, humanoid_loss_dict = compute_humanoid_loss(
            inputs, outputs, self, mode
        )
        total_loss += humanoid_loss
        outputs.update(humanoid_loss_dict)

        outputs["loss"] = total_loss
        return outputs

    def forward_2d(
        self,
        inputs,
        train=False,
        postproc=False,
        static_cam=False,
        global_step=0,
        mode=None,
    ):
        outputs = dict()
        inputs = self.prepare_inputs(inputs, train)

        if train:
            model_output = self.denoiser3d.forward_train_2d(
                inputs, mode
            )  # pred_x, pred_cam, static_conf_logits
        else:
            model_output = self.denoiser3d(
                inputs, train=train, postproc=postproc, static_cam=static_cam, mode=mode
            )  # pred_x, pred_cam, static_conf_logits

        if "decode_dict" in model_output:
            decode_dict = model_output.pop("decode_dict")
        else:
            decode_dict = self.endecoder.decode(
                model_output["pred_x"]
            )  # (B, L, C) -> dict
        outputs.update({"2d_model_output": model_output, "2d_decode_dict": decode_dict})

        # Post-processing
        if "gvhmr" in self.endecoder.feature_arr:
            outputs["2d_pred_smpl_params_incam"] = {
                "body_pose": decode_dict["body_pose"],  # (B, L, 63)
                "betas": decode_dict["betas"],  # (B, L, 10)
                "global_orient": decode_dict["global_orient"],  # (B, L, 3)
                "transl": compute_transl_full_cam(
                    model_output["pred_cam"], inputs["bbx_xys"], inputs["K_fullimg"]
                ),
            }

        if not train:
            if "gvhmr" in self.endecoder.feature_arr:
                pred_smpl_params_global = (
                    get_smpl_params_w_Rt_v2(  # This function has for-loop
                        global_orient_gv=decode_dict["global_orient_gv"],
                        local_transl_vel=decode_dict["local_transl_vel"],
                        global_orient_c=decode_dict["global_orient"],
                        cam_angvel=inputs["cam_angvel"],
                    )
                )
                outputs["2d_pred_smpl_params_global"] = {
                    "body_pose": decode_dict["body_pose"],
                    "betas": decode_dict["betas"],
                    **pred_smpl_params_global,
                }
            else:
                outputs["2d_pred_smpl_params_global"] = {
                    "body_pose": decode_dict["body_pose"],
                    "betas": decode_dict["betas"],
                    "global_orient": decode_dict["global_orient_w"],
                    "transl": decode_dict["transl_w"],
                }
            outputs["static_conf_logits"] = model_output["static_conf_logits"]

            if (
                postproc and "gvhmr" in self.endecoder.feature_arr
            ):  # apply post-processing
                if static_cam:  # extra post-processing to utilize static camera prior
                    outputs["2d_pred_smpl_params_global"]["transl"] = (
                        pp_static_joint_cam(
                            outputs,
                            self.endecoder,
                            smpl_key="2d_pred_smpl_params_global",
                        )
                    )
                else:
                    outputs["2d_pred_smpl_params_global"]["transl"] = pp_static_joint(
                        outputs, self.endecoder, smpl_key="2d_pred_smpl_params_global"
                    )
                body_pose = process_ik(
                    outputs, self.endecoder, smpl_key="2d_pred_smpl_params_global"
                )
                decode_dict["body_pose"] = body_pose
                outputs["2d_pred_smpl_params_global"]["body_pose"] = body_pose
                outputs["2d_pred_smpl_params_incam"]["body_pose"] = body_pose

            return outputs

        # ========== Compute Loss ========== #
        total_loss = 0
        mask = inputs["mask"]
        text_only_losses = self.weights.get("text_only_losses", "all")
        text_only = inputs.get("text_only", None)

        # 1. Simple loss: MSE
        if self.weights.get("simple_diffusion2d", 0.0) > 0.0 and mode == "diffusion":
            pred_x = model_output["pred_x"]  # (B, L, C)
            target_x = inputs["regression_outputs"]["2d_model_output"]["pred_x"]
            simple_loss = F.mse_loss(pred_x, target_x, reduction="none")
            mask_simple = (
                mask[:, :, None].expand(-1, -1, pred_x.size(2)).clone()
            )  # (B, L, C)
            if self.weights.get("simple_loss_diffusion2d_local_only", False):
                mask_simple[..., 142:] = False
            simple_loss = (simple_loss * mask_simple).mean()
            total_loss += simple_loss * self.weights.simple_diffusion2d
            outputs["simple_loss"] = simple_loss

        # 2. Extra loss
        model_output = outputs["2d_model_output"]
        endecoder = self.endecoder
        # mask_reproj = ~inputs["mask"]["spv_incam_only"]  # do not supervise reproj for 3DPW

        if "gvhmr" in self.endecoder.feature_arr:
            if self.weights.get("j2d_train2d", 0.0) > 0.0 and not (
                mode == "regression" and self.weights.train2d_skip_regression
            ):
                pred_c_j17 = endecoder.smplx_model(
                    **outputs["2d_pred_smpl_params_incam"]
                )[1]
                conf_2d = inputs["conf"]
                pred_c_j17[conf_2d < 0.1] = 0.0
                # prevent divide 0 or small value to overflow(fp16)
                reproj_z_thr = 0.3
                pred_c_j3d_z0_mask = pred_c_j17[..., 2].abs() <= reproj_z_thr
                pred_c_j17[pred_c_j3d_z0_mask] = reproj_z_thr

                # project and normalize
                pred_j2d_01 = project_to_bi01(
                    pred_c_j17, inputs["bbx_xys"], inputs["K_fullimg"]
                )
                pred_j2d_01[conf_2d < 0.1] = 0.0

                if self.weights.get("proj_gt_j2d_to_bi01", False):
                    bbx_lurb = convert_bbx_xys_to_lurb(inputs["bbx_xys"])
                    gt_c_j17_2d = cvt_to_bi01_p2d(inputs["obs_kp2d_raw"], bbx_lurb)
                    gt_c_j17_2d[~inputs["mask"]] = 0
                else:
                    gt_c_j17_2d = inputs["orig_obs"][..., :2]

                j2d_loss = F.mse_loss(pred_j2d_01, gt_c_j17_2d, reduction="none")
                if (
                    text_only is not None
                    and text_only_losses != "all"
                    and "j2d_train2d" not in text_only_losses
                ):
                    j2d_loss[text_only] = 0
                j2d_loss = (
                    j2d_loss * mask[..., None, None] * conf_2d[..., None]
                ).mean()

                total_loss += j2d_loss * self.weights.j2d_train2d
                outputs["j2d_loss_2d"] = j2d_loss

            if self.weights.get("norm_j2d", 0.0) > 0.0 and not (
                mode == "regression" and self.weights.train2d_skip_regression
            ):
                pred_c_j17 = endecoder.smplx_model(
                    **outputs["2d_pred_smpl_params_incam"]
                )[1]
                conf_2d = inputs["conf"]
                pred_c_j17[conf_2d < 0.1] = 0.0
                # prevent divide 0 or small value to overflow(fp16)
                reproj_z_thr = 0.3
                pred_c_j3d_z0_mask = pred_c_j17[..., 2].abs() <= reproj_z_thr
                pred_c_j17[pred_c_j3d_z0_mask] = reproj_z_thr

                kp2d = perspective_projection(pred_c_j17, inputs["K_fullimg"])
                bbx = get_bbx_xys(kp2d, do_augment=False)
                kp2d_norm = normalize_kp2d(kp2d, bbx)
                kp2d_gt = inputs["orig_obs"][..., :2]

                norm_j2d_loss = F.mse_loss(kp2d_norm, kp2d_gt, reduction="none")
                if (
                    text_only is not None
                    and text_only_losses != "all"
                    and "norm_j2d" not in text_only_losses
                ):
                    norm_j2d_loss[text_only] = 0
                norm_j2d_loss = (
                    norm_j2d_loss * mask[..., None, None] * conf_2d[..., None]
                ).mean()

                total_loss += norm_j2d_loss * self.weights.norm_j2d
                outputs["norm_j2d_loss_2d"] = norm_j2d_loss

        outputs["loss_2d"] = total_loss
        return outputs


def randomly_set_null_condition(f_condition, uncond_prob=0.1, skip_keys=[]):
    """Conditions are in shape (B, L, *)"""
    keys = list(f_condition.keys())
    for k in keys:
        if k in skip_keys:
            continue
        if f_condition[k] is None or type(f_condition[k]) == bool:
            continue
        f_condition[k] = f_condition[k].clone()
        mask = torch.rand(f_condition[k].shape[:2]) < uncond_prob
        f_condition[k][mask] = 0.0
    return f_condition


def compute_humanoid_loss(inputs, outputs, ppl, mode):
    """Compute humanoid-specific losses

    Args:
        inputs (dict): Input data dictionary containing:
            - mask: Masking information
            - text_only: Optional flag for text-only inputs
        outputs (dict): Model outputs dictionary containing:
            - model_output: Raw model outputs
            - decode_dict: Decoded model outputs
        ppl (Pipeline): Pipeline object containing:
            - weights: Loss weights configuration
            - endecoder: Encoder/decoder object
            - args: Pipeline arguments
        mode (str): Current training mode

    Returns:
        dict: Dictionary containing computed losses
    """
    weights = ppl.weights
    # text_only_losses = weights.get("text_only_losses", "all")
    # text_only = inputs.get("text_only", None)

    # if weights.get("text_only_no_reg_loss", False) and mode == "regression":
    #     text_only_losses = []

    humanoid_loss_dict = {}
    humanoid_loss = 0
    mask = inputs["mask"]["humanoid"]  # (B, L)

    # Add humanoid-specific loss computations here
    # Example:
    # if weights.humanoid_loss > 0:
    #     humanoid_loss = compute_specific_loss()
    #     if text_only is not None and text_only_losses != "all" and "humanoid_loss" not in text_only_losses:
    #         humanoid_loss[text_only] = 0
    #     humanoid_loss = (humanoid_loss * mask).mean()
    #     extra_loss += humanoid_loss * weights.humanoid_loss
    #     extra_loss_dict["humanoid_loss"] = humanoid_loss

    # add clean action loss
    if weights.get("humanoid_clean_action", 0.0) > 0.0:
        gt_humanoid_clean_action = inputs["humanoid_clean_action"]
        if "humanoid_clean_action" in outputs["model_output"]:
            humanoid_clean_action = outputs["model_output"]["humanoid_clean_action"]
        else:
            humanoid_clean_action = outputs["decode_dict"]["humanoid_clean_action"]
        humanoid_clean_action_loss = F.mse_loss(
            humanoid_clean_action, gt_humanoid_clean_action, reduction="none"
        )
        humanoid_clean_action_loss = (
            humanoid_clean_action_loss * mask[..., None]
        ).mean()
        humanoid_loss += humanoid_clean_action_loss * weights.humanoid_clean_action
        humanoid_loss_dict["humanoid_clean_action_loss"] = humanoid_clean_action_loss

    if weights.get("humanoid_z_action", 0.0) > 0.0:
        gt_humanoid_z_action = inputs["humanoid_z_action"]
        humanoid_z_action = outputs["decode_dict"]["humanoid_z_action"]
        humanoid_z_action_loss = F.mse_loss(
            humanoid_z_action, gt_humanoid_z_action, reduction="none"
        )
        humanoid_z_action_loss = (humanoid_z_action_loss * mask[..., None]).mean()
        humanoid_loss += humanoid_z_action_loss * weights.humanoid_z_action
        humanoid_loss_dict["humanoid_z_action_loss"] = humanoid_z_action_loss

    return humanoid_loss, humanoid_loss_dict


def compute_extra_incam_loss(inputs, outputs, ppl, mode):
    model_output = outputs["model_output"]
    decode_dict = outputs["decode_dict"]
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args

    text_only_losses = weights.get("text_only_losses", "all")
    text_only = inputs.get("text_only", None)
    if weights.get("text_only_no_reg_loss", False) and mode == "regression":
        text_only_losses = []

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"]  # effective length mask
    mask_reproj = ~inputs["mask"]["spv_incam_only"]  # do not supervise reproj for 3DPW

    # Incam FK
    # prediction
    pred_c_j3d = endecoder.fk_v2(**outputs["pred_smpl_params_incam"])
    pred_cr_j3d = pred_c_j3d - pred_c_j3d[:, :, :1]  # (B, L, J, 3)

    # gt
    gt_c_j3d = endecoder.fk_v2(**inputs["smpl_params_c"])  # (B, L, J, 3)
    gt_cr_j3d = gt_c_j3d - gt_c_j3d[:, :, :1]  # (B, L, J, 3)

    # Root aligned C-MPJPE Loss
    if weights.cr_j3d > 0.0:
        cr_j3d_loss = F.mse_loss(pred_cr_j3d, gt_cr_j3d, reduction="none")
        if (
            text_only is not None
            and text_only_losses != "all"
            and "cr_j3d" not in text_only_losses
        ):
            cr_j3d_loss[text_only] = 0
        cr_j3d_loss = (cr_j3d_loss * mask[..., None, None]).mean()
        extra_loss += cr_j3d_loss * weights.cr_j3d
        extra_loss_dict["cr_j3d_loss"] = cr_j3d_loss

    # Reprojection (to align with image)
    if weights.transl_c > 0.0:
        # pred_transl = decode_dict["transl"]  # (B, L, 3)
        # gt_transl = inputs["smpl_params_c"]["transl"]
        # transl_c_loss = F.l1_loss(pred_transl, gt_transl, reduction="none")
        # transl_c_loss = (transl_c_loss * mask[..., None]).mean()

        # Instead of supervising transl, we convert gt to pred_cam (prevent divide 0)
        pred_cam = model_output["pred_cam"]  # (B, L, 3)
        gt_transl = inputs["smpl_params_c"]["transl"]  # (B, L, 3)
        gt_pred_cam = get_a_pred_cam(
            gt_transl, inputs["bbx_xys"], inputs["K_fullimg"]
        )  # (B, L, 3)
        gt_pred_cam[gt_pred_cam.isinf()] = -1  # this will be handled by valid_mask
        # (compute_transl_full_cam(gt_pred_cam, inputs["bbx_xys"], inputs["K_fullimg"]) - gt_transl).abs().max()

        # Skip gts that are not good during random construction
        gt_j3d_z_min = inputs["gt_j3d"][..., 2].min(dim=-1)[0]
        valid_mask = (
            (gt_j3d_z_min > 0.3)
            * (gt_pred_cam[..., 0] > 0.3)
            * (gt_pred_cam[..., 0] < 5.0)
            * (gt_pred_cam[..., 1] > -3.0)
            * (gt_pred_cam[..., 1] < 3.0)
            * (gt_pred_cam[..., 2] > -3.0)
            * (gt_pred_cam[..., 2] < 3.0)
            * (inputs["bbx_xys"][..., 2] > 0)
        )[..., None]
        transl_c_loss = F.mse_loss(pred_cam, gt_pred_cam, reduction="none")
        if (
            text_only is not None
            and text_only_losses != "all"
            and "transl_c" not in text_only_losses
        ):
            transl_c_loss[text_only] = 0
        transl_c_loss = (transl_c_loss * mask[..., None] * valid_mask).mean()

        extra_loss_dict["transl_c_loss"] = transl_c_loss
        extra_loss += transl_c_loss * weights.transl_c

    if weights.j2d > 0.0:
        # prevent divide 0 or small value to overflow(fp16)
        reproj_z_thr = 0.3
        pred_c_j3d_z0_mask = pred_c_j3d[..., 2].abs() <= reproj_z_thr
        pred_c_j3d[pred_c_j3d_z0_mask] = reproj_z_thr
        gt_c_j3d_z0_mask = gt_c_j3d[..., 2].abs() <= reproj_z_thr
        gt_c_j3d[gt_c_j3d_z0_mask] = reproj_z_thr

        pred_j2d_01 = project_to_bi01(
            pred_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"]
        )
        gt_j2d_01 = project_to_bi01(
            gt_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"]
        )  # (B, L, J, 2)

        valid_mask = (
            (gt_c_j3d[..., 2] > reproj_z_thr)
            * (pred_c_j3d[..., 2] > reproj_z_thr)  # Be safe
            * (gt_j2d_01[..., 0] > 0.0)
            * (gt_j2d_01[..., 0] < 1.0)
            * (gt_j2d_01[..., 1] > 0.0)
            * (gt_j2d_01[..., 1] < 1.0)
        )[..., None]
        valid_mask[~mask_reproj] = False  # Do not supervise on 3dpw
        j2d_loss = F.mse_loss(pred_j2d_01, gt_j2d_01, reduction="none")
        if (
            text_only is not None
            and text_only_losses != "all"
            and "j2d" not in text_only_losses
        ):
            j2d_loss[text_only] = 0
        j2d_loss = (j2d_loss * mask[..., None, None] * valid_mask).mean()

        extra_loss += j2d_loss * weights.j2d
        extra_loss_dict["j2d_loss"] = j2d_loss

    if weights.cr_verts > 0:
        # SMPL forward
        pred_c_verts437, pred_c_j17 = endecoder.smplx_model(
            **outputs["pred_smpl_params_incam"]
        )
        root_ = pred_c_j17[:, :, [11, 12], :].mean(-2, keepdim=True)
        pred_cr_verts437 = pred_c_verts437 - root_

        gt_cr_verts437 = inputs["gt_cr_verts437"]  # (B, L, 437, 3)
        cr_vert_loss = F.mse_loss(pred_cr_verts437, gt_cr_verts437, reduction="none")
        if (
            text_only is not None
            and text_only_losses != "all"
            and "cr_verts" not in text_only_losses
        ):
            cr_vert_loss[text_only] = 0
        cr_vert_loss = (cr_vert_loss * mask[:, :, None, None]).mean()
        extra_loss += cr_vert_loss * weights.cr_verts
        extra_loss_dict["cr_vert_loss"] = cr_vert_loss

    if weights.verts2d > 0:
        gt_c_verts437 = inputs["gt_c_verts437"]  # (B, L, 437, 3)

        # prevent divide 0 or small value to overflow(fp16)
        reproj_z_thr = 0.3
        pred_c_verts437_z0_mask = pred_c_verts437[..., 2].abs() <= reproj_z_thr
        pred_c_verts437[pred_c_verts437_z0_mask] = reproj_z_thr
        gt_c_verts437_z0_mask = gt_c_verts437[..., 2].abs() <= reproj_z_thr
        gt_c_verts437[gt_c_verts437_z0_mask] = reproj_z_thr

        pred_verts2d_01 = project_to_bi01(
            pred_c_verts437, inputs["bbx_xys"], inputs["K_fullimg"]
        )
        gt_verts2d_01 = project_to_bi01(
            gt_c_verts437, inputs["bbx_xys"], inputs["K_fullimg"]
        )  # (B, L, 437, 2)

        valid_mask = (
            (gt_c_verts437[..., 2] > reproj_z_thr)
            * (pred_c_verts437[..., 2] > reproj_z_thr)  # Be safe
            * (gt_verts2d_01[..., 0] > 0.0)
            * (gt_verts2d_01[..., 0] < 1.0)
            * (gt_verts2d_01[..., 1] > 0.0)
            * (gt_verts2d_01[..., 1] < 1.0)
        )[..., None]
        valid_mask[~mask_reproj] = False  # Do not supervise on 3dpw
        verts2d_loss = F.mse_loss(pred_verts2d_01, gt_verts2d_01, reduction="none")
        if (
            text_only is not None
            and text_only_losses != "all"
            and "verts2d" not in text_only_losses
        ):
            verts2d_loss[text_only] = 0
        verts2d_loss = (verts2d_loss * mask[..., None, None] * valid_mask).mean()

        extra_loss += verts2d_loss * weights.verts2d
        extra_loss_dict["verts2d_loss"] = verts2d_loss

    return extra_loss, extra_loss_dict


def compute_extra_global_loss(inputs, outputs, ppl, mode):
    decode_dict = outputs["decode_dict"]
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"].clone()  # (B, L)
    mask[inputs["mask"]["spv_incam_only"]] = False

    text_only_losses = weights.get("text_only_losses", "all")
    text_only = inputs.get("text_only", None)
    if weights.get("text_only_no_reg_loss", False) and mode == "regression":
        text_only_losses = []

    if weights.transl_w > 0:
        # compute pred_transl_w by rollout
        gt_transl_w = inputs["smpl_params_w"]["transl"]
        gt_global_orient_w = inputs["smpl_params_w"]["global_orient"]
        local_transl_vel = decode_dict["local_transl_vel"]
        pred_transl_w = rollout_local_transl_vel(
            local_transl_vel, gt_global_orient_w, gt_transl_w[:, [0]]
        )

        trans_w_loss = F.l1_loss(pred_transl_w, gt_transl_w, reduction="none")
        if (
            text_only is not None
            and text_only_losses != "all"
            and "transl_w" not in text_only_losses
        ):
            trans_w_loss[text_only] = 0
        trans_w_loss = (trans_w_loss * mask[..., None]).mean()
        extra_loss += trans_w_loss * weights.transl_w
        extra_loss_dict["transl_w_loss"] = trans_w_loss

    # Static-Conf loss
    if weights.static_conf_bce > 0:
        # Compute gt by thresholding velocity
        vel_thr = args.static_conf.vel_thr
        assert vel_thr > 0
        joint_ids = [
            7,
            10,
            8,
            11,
            20,
            21,
        ]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
        gt_w_j3d = endecoder.fk_v2(**inputs["smpl_params_w"])  # (B, L, J=22, 3)
        static_gt = get_static_joint_mask(
            gt_w_j3d, vel_thr=vel_thr, repeat_last=True
        )  # (B, L, J)
        static_gt = static_gt[:, :, joint_ids].float()  # (B, L, J')
        pred_static_conf_logits = outputs["model_output"]["static_conf_logits"]

        static_conf_loss = F.binary_cross_entropy_with_logits(
            pred_static_conf_logits, static_gt, reduction="none"
        )
        if (
            text_only is not None
            and text_only_losses != "all"
            and "static_conf_bce" not in text_only_losses
        ):
            static_conf_loss[text_only] = 0
        static_conf_loss = (static_conf_loss * mask[..., None]).mean()
        extra_loss += static_conf_loss * weights.static_conf_bce
        extra_loss_dict["static_conf_loss"] = static_conf_loss

    return extra_loss, extra_loss_dict


@autocast(enabled=False)
def get_smpl_params_w_Rt_v2(
    global_orient_gv,
    local_transl_vel,
    global_orient_c,
    cam_angvel,
):
    """Get global R,t in GV0(ay)
    Args:
        cam_angvel: (B, L, 6), defined as R @ R_{w2c}^{t} = R_{w2c}^{t+1}
    """

    # Get R_ct_to_c0 from cam_angvel
    def as_identity(R):
        is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
        R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
        return R

    B = cam_angvel.shape[0]
    R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3)
    R_t_to_tp1 = as_identity(R_t_to_tp1)

    # Get R_c2gv
    R_gv = axis_angle_to_matrix(global_orient_gv)  # (B, L, 3, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (B, L, 3, 3)

    # Camera view direction in GV coordinate: Rc2gv @ [0,0,1]
    R_c2gv = R_gv @ R_c.mT
    view_axis_gv = R_c2gv[
        :, :, :, 2
    ]  # (B, L, 3)  Rc2gv is estimated, so the x-axis is not accurate, i.e. != 0

    # Rotate axis use camera relative rotation
    R_cnext2gv = R_c2gv @ R_t_to_tp1.mT
    view_axis_gv_next = R_cnext2gv[..., 2]

    vec1_xyz = view_axis_gv.clone()
    vec1_xyz[..., 1] = 0
    vec1_xyz = F.normalize(vec1_xyz, dim=-1)
    vec2_xyz = view_axis_gv_next.clone()
    vec2_xyz[..., 1] = 0
    vec2_xyz = F.normalize(vec2_xyz, dim=-1)

    aa_tp1_to_t = vec2_xyz.cross(vec1_xyz, dim=-1)
    aa_tp1_to_t_angle = torch.acos(
        torch.clamp((vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True), -1.0, 1.0)
    )
    aa_tp1_to_t = F.normalize(aa_tp1_to_t, dim=-1) * aa_tp1_to_t_angle

    aa_tp1_to_t = gaussian_smooth(aa_tp1_to_t, dim=-2)  # Smooth
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT  # (B, L, 3)

    # Get R_t_to_0
    R_t_to_0 = [torch.eye(3)[None].expand(B, -1, -1).to(R_t_to_tp1)]
    for i in range(1, R_t_to_tp1.shape[1]):
        R_t_to_0.append(R_t_to_0[-1] @ R_tp1_to_t[:, i])
    R_t_to_0 = torch.stack(R_t_to_0, dim=1)  # (B, L, 3, 3)
    R_t_to_0 = as_identity(R_t_to_0)

    global_orient = matrix_to_axis_angle(R_t_to_0 @ R_gv)

    # Rollout to global transl
    # Start from transl0, in gv0 -> flip y-axis of gv0
    transl = rollout_local_transl_vel(local_transl_vel, global_orient)
    global_orient, transl, _ = get_tgtcoord_rootparam(
        global_orient, transl, tsf="any->ay"
    )

    smpl_params_w_Rt = {"global_orient": global_orient, "transl": transl}
    return smpl_params_w_Rt


def gaussian_probability(x, mean, std_dev):
    """Calculate the Gaussian probability of x given mean and standard deviation."""
    exponent = torch.exp(-((x - mean) ** 2) / (2 * std_dev**2))
    return (1 / (math.sqrt(2 * math.pi) * std_dev)) * exponent


@autocast(enabled=False)
def get_smpl_params_w_Rt_v3(
    endecoder,
    global_orient_gv,
    local_transl_vel,
    local_transl_c,
    global_orient_c,
    offset,
    cam_angvel,
    cam_tvel,
    slam_R_w2c,
    static_conf_logits,
    conf_scales,
    **kwargs,
):
    """Get global R,t in GV0(ay)
    Args:
        cam_angvel: (B, L, 6), defined as R @ R_{w2c}^{t} = R_{w2c}^{t+1}
        cam_tvel: (B, L, 3)
    """

    # Get R_ct_to_c0 from cam_angvel
    def as_identity(R):
        is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
        R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
        return R

    B = cam_angvel.shape[0]
    device = cam_angvel.device
    R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3)
    R_t_to_tp1 = as_identity(R_t_to_tp1)

    # Get R_c2gv
    R_gv = axis_angle_to_matrix(global_orient_gv)  # (B, L, 3, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (B, L, 3, 3)

    # Camera view direction in GV coordinate: Rc2gv @ [0,0,1]
    R_c2gv = R_gv @ R_c.mT
    view_axis_gv = R_c2gv[
        :, :, :, 2
    ]  # (B, L, 3)  Rc2gv is estimated, so the x-axis is not accurate, i.e. != 0

    # Rotate axis use camera relative rotation
    R_cnext2gv = R_c2gv @ R_t_to_tp1.mT
    view_axis_gv_next = R_cnext2gv[..., 2]

    vec1_xyz = view_axis_gv.clone()
    vec1_xyz[..., 1] = 0
    vec1_xyz = F.normalize(vec1_xyz, dim=-1)
    vec2_xyz = view_axis_gv_next.clone()
    vec2_xyz[..., 1] = 0
    vec2_xyz = F.normalize(vec2_xyz, dim=-1)

    aa_tp1_to_t = vec2_xyz.cross(vec1_xyz, dim=-1)
    aa_tp1_to_t_angle = torch.acos(
        torch.clamp((vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True), -1.0, 1.0)
    )
    aa_tp1_to_t = F.normalize(aa_tp1_to_t, dim=-1) * aa_tp1_to_t_angle

    aa_tp1_to_t = gaussian_smooth(aa_tp1_to_t, dim=-2)  # Smooth
    R_tp1_to_t = axis_angle_to_matrix(aa_tp1_to_t).mT  # (B, L, 3)

    # Get R_t_to_0
    R_t_to_0 = [torch.eye(3)[None].expand(B, -1, -1).to(R_t_to_tp1)]
    for i in range(1, R_t_to_tp1.shape[1]):
        R_t_to_0.append(R_t_to_0[-1] @ R_tp1_to_t[:, i - 1])
    R_t_to_0 = torch.stack(R_t_to_0, dim=1)  # (B, L, 3, 3)
    R_t_to_0 = as_identity(R_t_to_0)

    R_w = R_t_to_0 @ R_gv
    global_orient = matrix_to_axis_angle(R_w)

    # Rollout to global transl
    # Start from transl0, in gv0 -> flip y-axis of gv0
    transl = rollout_local_transl_vel(local_transl_vel, global_orient)
    global_orient, transl, _ = get_tgtcoord_rootparam(
        global_orient, transl, tsf="any->ay"
    )

    body_pose = kwargs["body_pose"][0]
    if body_pose.shape[1] < 69:
        bs = body_pose.shape[0]
        pad_size = 69 - body_pose.shape[1]
        body_pose = torch.cat(
            (body_pose, torch.zeros(bs, pad_size).to(body_pose)), dim=1
        )

    if False:
        from motiondiff.utils.vis_scenepic import ScenepicVisualizer

        sp_visualizer = ScenepicVisualizer(
            "inputs/checkpoints/body_models/smpl", device=device
        )
        smpl = sp_visualizer.smpl_dict["neutral"]

        smpl_out = smpl(
            betas=kwargs["betas"][0],
            body_pose=body_pose,
            global_orient=global_orient[0],
            transl=transl[0],
            orig_joints=True,
        )
        joints = smpl_out.joints
        ground = joints.reshape(-1, 3)[:, 1].min()
        transl[:, :, 1] = transl[:, :, 1] - ground.reshape(B, 1)
    assert B == 1, "only support batch 1 inference"

    # Get slam_R_w2c
    slam_R_w2c = slam_R_w2c
    # cam_angvel_mat = rotation_6d_to_matrix(cam_angvel)  # (B, L, 3, 3)
    # slam_R_w2c = [torch.eye(3)[None].expand(B, -1, -1).to(cam_angvel)]

    # for i in range(1, cam_angvel.shape[1]):
    #     slam_R_w2c.append(cam_angvel_mat[:, i - 1] @ slam_R_w2c[-1].mT)
    # slam_R_w2c = torch.stack(slam_R_w2c, dim=1)  # (B, L, 3, 3)
    # slam_R_w2c = as_identity(slam_R_w2c)

    R_c_full = R_c.clone()
    slam_R_w2c_full = slam_R_w2c.clone()
    offset_full = offset.clone()
    global_orient_list = []
    transl_list = []
    scale_hmr_c2w_list = []
    R_w_hmr = axis_angle_to_matrix(global_orient).clone()
    t_w_hmr = transl.clone()

    motion_f_hmr = {
        "global_orient": matrix_to_axis_angle(R_w_hmr),
        "body_pose": body_pose[None][..., :63],
        "betas": kwargs["betas"],
        "transl": t_w_hmr,
    }
    out_hmr = {
        "pred_smpl_params_global": motion_f_hmr,
        "static_conf_logits": static_conf_logits,
    }
    t_w_hmr = pp_static_joint(out_hmr, endecoder)
    transl = t_w_hmr.clone()

    slam_scales = kwargs["slam_scales"]
    mean_scale = kwargs["mean_scale"]

    for bid in range(B):
        L = transl.shape[1]
        offset = offset_full[bid]

        R_w_bak = axis_angle_to_matrix(global_orient).clone()[bid]
        t_w_bak = transl.clone()[bid]
        R_w = axis_angle_to_matrix(global_orient)[bid]
        t_w = transl[bid]
        # set start point
        cam_t0 = cam_tvel[bid, :1, :].clone().detach().zero_()
        cam_t = torch.cat([cam_t0, cam_tvel[bid, :-1, :]], dim=-2)
        # rollout from start point
        slam_t_w2c = torch.cumsum(cam_t, dim=-2)
        R_c = R_c_full[bid]
        t_c = local_transl_c[bid]
        R_w2c = R_c @ R_w.mT
        # align with the norm t_w2c
        R0 = R_w2c[:1]
        R_w2c = R_w2c @ R0.mT
        R_w2c = slam_R_w2c_full[bid]
        # norm_t_w = torch.einsum("fij,fj->fi", R0, t_w)
        # t_w2c_hmr = t_c + offset - torch.einsum("fij,fj->fi", R_w2c, norm_t_w + offset)
        # t_c2w_hmr = (-R_w2c.mT @ t_w2c_hmr[..., None])[..., 0]
        # t_c2w = (-R_w2c.mT @ t_w2c[..., None])[..., 0]

        init_T_w2c = torch.eye(4)[None].repeat(L, 1, 1).to(device)
        norm_T_c2w = torch.eye(4)[None].repeat(L, 1, 1).to(device)
        init_T_w2c[:, :3, :3] = R_w2c
        init_T_w2c[:, :3, 3] = slam_t_w2c

        # align the first frame
        init_T_c2w = init_T_w2c.inverse()
        R_c2w = as_identity(init_T_c2w[:, :3, :3])
        t_c2w = init_T_c2w[:, :3, 3]
        R0_c2w = R_c2w[:1]
        t0_c2w = t_c2w[:1]
        norm_R_c2w = R0_c2w.mT @ R_c2w
        norm_t_c2w = (R0_c2w.mT @ (t_c2w - t0_c2w)[..., None])[..., 0]
        norm_T_c2w[:, :3, :3] = norm_R_c2w
        norm_T_c2w[:, :3, 3] = norm_t_c2w
        norm_T_w2c = norm_T_c2w.inverse()
        norm_T_w2c[:, :3, :3] = as_identity(norm_T_w2c[:, :3, :3])
        norm_T_w2c[:, 3, :3] = 0

        # R0_w2c = R_w2c[:1]
        # t0_w2c = t_w2c[:1]
        # init_R_w2c = R_w2c @ R0_w2c.mT
        # init_t_w2c = t_w2c - t0_w2c
        # init_T_w2c[:, :3, :3] = init_R_w2c
        # init_T_w2c[:, :3, 3] = init_t_w2c

        smpl_param_c = {
            "global_orient": R_c,
            "transl": t_c,
        }
        smpl_param_w = {
            "global_orient": R_w,
            "transl": t_w,
        }
        est_T_w2c, scale_hmr_c2w = estimate_camscale(
            smpl_param_c,
            smpl_param_w,
            norm_T_w2c,
            offset,
            slam_scales[bid],
            mean_scale[bid],
        )

        est_R_w2c = est_T_w2c[:, :3, :3]
        est_t_w2c = est_T_w2c[:, :3, 3]

        est_t_w = (est_R_w2c.mT @ (t_c + offset - est_t_w2c)[..., None])[
            ..., 0
        ] - offset

        global_orient_list.append(matrix_to_axis_angle(R_w))
        transl_list.append(est_t_w)
        scale_hmr_c2w_list.append(scale_hmr_c2w)

    R_w = torch.stack(global_orient_list)
    t_w = torch.stack(transl_list)
    scale_hmr_c2w_list = torch.stack(scale_hmr_c2w_list)

    # update transl with contacts
    motion_f_slam = {
        "global_orient": R_w,
        "body_pose": body_pose[None][..., :63],
        "betas": kwargs["betas"],
        "transl": t_w,
    }
    motion_f_hmr = {
        "global_orient": matrix_to_axis_angle(R_w_hmr),
        "body_pose": body_pose[None][..., :63],
        "betas": kwargs["betas"],
        "transl": t_w_hmr,
    }

    if True:
        pred_w_j3d_slam = endecoder.fk_v2(**motion_f_slam)
        pred_w_j3d_hmr = endecoder.fk_v2(**motion_f_hmr)

        post_w_transl_slam = motion_f_slam["transl"]
        post_w_transl_hmr = motion_f_hmr["transl"]
        # process contact info
        static_conf_logits = static_conf_logits[:, :-1].clone()
        static_label_ = static_conf_logits > 0  # (B, L-1, J) # avoid non-contact frame
        static_conf_logits = static_conf_logits.float() - (
            ~static_label_ * 1e6
        )  # fp16 cannot go through softmax
        is_static = static_label_.sum(dim=-1) > 0  # (B, L-1)

        L = pred_w_j3d_hmr.shape[1]
        joint_ids = [
            7,
            10,
            8,
            11,
            20,
            21,
        ]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
        pred_j3d_static_slam = pred_w_j3d_slam.clone()[:, :, joint_ids]  # (B, L, J, 3)
        pred_j3d_static_hmr = pred_w_j3d_hmr.clone()[:, :, joint_ids]  # (B, L, J, 3)

        pred_j_disp_slam = (
            pred_j3d_static_slam[:, 1:] - pred_j3d_static_slam[:, :-1]
        )  # (B, L-1, J, 3)
        pred_j_disp_hmr = (
            pred_j3d_static_hmr[:, 1:] - pred_j3d_static_hmr[:, :-1]
        )  # (B, L-1, J, 3)

        conf_contact = static_conf_logits[..., None].softmax(dim=-2)
        # if "skate" in kwargs["meta"][0]["vid"]:
        if True:
            # conf_contact = static_conf_logits[..., None].softmax(dim=-2)
            slam_scales = kwargs["slam_scales"]
            mean_scale = kwargs["mean_scale"]
            std_slam_scales = slam_scales.std(dim=-1, keepdim=True)
            scale_hmr_c2w = scale_hmr_c2w_list[bid]

            scale_thrd = 5
            invalid_hmr = (scale_hmr_c2w > scale_thrd) | (
                scale_hmr_c2w < (1 / scale_thrd)
            )
            invalid_hmr = torch.cat([invalid_hmr, invalid_hmr[-1:]], dim=-1)
            # assume gaussian distribution
            conf_scales_ = gaussian_probability(
                slam_scales, mean_scale, torch.ones_like(std_slam_scales)
            )
            conf_scales_[conf_scales_ > 1.0] = 1.0
            conf_scales_[conf_scales_ < 0.3] = 0.0
            conf_scales_[invalid_hmr[None]] = 1.0

            conf_contact = conf_contact * (1 - conf_scales_[:, :-1])[..., None, None]
            # vid = kwargs["meta"][0]["vid"].replace('/', '_')
            # if 'P8' in vid:
            #     import ipdb; ipdb.set_trace()
            # conf_contact = conf_contact * invalid_hmr.reshape(1, -1, 1, 1)[:, :-1].float()

        # conf_contact[conf_contact < 0.5] = 0.
        # conf_contact = 0
        pred_disp_slam = pred_j_disp_slam * conf_contact  # (B, L-1, J, 3)
        pred_disp_slam = pred_disp_slam * is_static[..., None, None]  # (B, L-1, J, 3)
        pred_disp_slam = pred_disp_slam.sum(-2)  # (B, L-1, 3)

        pred_disp_hmr = pred_j_disp_hmr * static_conf_logits[..., None].softmax(
            dim=-2
        )  # (B, L-1, J, 3)
        # pred_disp_hmr = pred_j_disp_hmr * conf_contact  # (B, L-1, J, 3)
        pred_disp_hmr = pred_disp_hmr * is_static[..., None, None]  # (B, L-1, J, 3)
        pred_disp_hmr = pred_disp_hmr.sum(-2)  # (B, L-1, 3)

        pred_w_disp_slam = (
            motion_f_slam["transl"][:, 1:] - motion_f_slam["transl"][:, :-1]
        )  # (B, L-1, 3)
        pred_w_disp_slam_new = pred_w_disp_slam - pred_disp_slam
        post_w_transl_slam = torch.cumsum(
            torch.cat([motion_f_slam["transl"][:, :1], pred_w_disp_slam_new], dim=1),
            dim=1,
        )
        # post_w_transl_slam = motion_f_slam['transl'].clone()
        # post_w_transl_slam[..., 0] = gaussian_smooth(post_w_transl_slam[..., 0], dim=-1)
        # post_w_transl_slam[..., 2] = gaussian_smooth(post_w_transl_slam[..., 2], dim=-1)

        pred_w_disp_hmr = (
            motion_f_hmr["transl"][:, 1:] - motion_f_hmr["transl"][:, :-1]
        )  # (B, L-1, 3)
        pred_w_disp_hmr_new = pred_w_disp_hmr - pred_disp_hmr
        post_w_transl_hmr = torch.cumsum(
            torch.cat([motion_f_hmr["transl"][:, :1], pred_w_disp_hmr_new], dim=1),
            dim=1,
        )
        # post_w_transl_hmr[..., 0] = gaussian_smooth(post_w_transl_hmr[..., 0], dim=-1)
        # post_w_transl_hmr[..., 2] = gaussian_smooth(post_w_transl_hmr[..., 2], dim=-1)

        motion_f_slam["transl"] = post_w_transl_slam
        motion_f_hmr["transl"] = post_w_transl_hmr

        # slam_out = {"pred_smpl_params_global": motion_f_slam}
        # body_pose_slam = process_ik(slam_out, endecoder, conf_contact[..., 0].float())
        # motion_f_slam["body_pose"] = body_pose_slam

        hmr_out = {"pred_smpl_params_global": motion_f_hmr}
        # body_pose_hmr = process_ik(hmr_out, endecoder, static_conf_logits.sigmoid().float())
        # motion_f_hmr["body_pose"] = body_pose_hmr

        pred_w_j3d_slam = endecoder.fk_v2(**motion_f_slam)
        pred_w_j3d_hmr = endecoder.fk_v2(**motion_f_hmr)

        # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
        ground_y = (
            pred_w_j3d_slam[..., 1].flatten(-2).min(dim=-1)[0]
        )  # (B,)  Minimum y value
        post_w_transl_slam[..., 1] -= ground_y
        motion_f_slam["transl"] = post_w_transl_slam

        post_w_transl_slam, use_slam, static_conf_logits = merge_slam_w_hmr(
            pred_w_j3d_slam,
            pred_w_j3d_hmr,
            post_w_transl_slam,
            post_w_transl_hmr,
            static_conf_logits,
            static_label_,
            conf_scales,
        )
        motion_f_slam["transl"] = post_w_transl_slam

        body_pose = motion_f_slam["body_pose"].clone()
        # final_out = {"pred_smpl_params_global": motion_f_slam}
        # body_pose_ik = process_ik(final_out, endecoder, static_conf_logits.sigmoid().float())
        # use_slam = torch.cat([use_slam, use_slam[:, -1:]], dim=1)
        # use_slam = use_slam.repeat(1, 1, 63)
        # body_pose[~use_slam] = body_pose_ik[~use_slam]
        motion_f_slam["body_pose"] = body_pose

    for bid in range(B):
        if False:  # visualization
            from motiondiff.utils.vis_scenepic import ScenepicVisualizer

            sp_visualizer = ScenepicVisualizer(
                "inputs/checkpoints/body_models/smpl", device=device
            )
            smpl = sp_visualizer.smpl_dict["neutral"]

            # smpl_out = smpl(
            #     betas=kwargs["betas"][0],
            #     body_pose=body_pose,
            #     global_orient=global_orient[0],
            #     transl=transl[0],
            #     orig_joints=True,
            # )
            # joints = smpl_out.joints
            # ground = joints.reshape(-1, 3)[:, 1].min()
            # transl[:, :, 1] = transl[:, :, 1] - ground.reshape(B, 1)

            t_w = motion_f_slam["transl"]
            t_w_hmr = motion_f_hmr["transl"]
            R_w = motion_f_slam["global_orient"]
            R_w_hmr = motion_f_hmr["global_orient"]
            vis_mat = (
                torch.tensor(((1, 0, 0), (0, 0, -1), (0, 1, 0))).to(device).float()
            )

            smpl_out = smpl(
                betas=kwargs["betas"][bid],
                body_pose=body_pose,
                global_orient=R_w[bid],
                transl=t_w[bid],
                orig_joints=True,
            )
            joints = smpl_out.joints
            joints_vis = (vis_mat[None] @ joints.mT).mT
            est_T_w2c_vis = est_T_w2c.clone()
            est_T_w2c_vis[:, :3, :3] = est_T_w2c[:, :3, :3] @ vis_mat[None].mT

            res = {
                "text": "",
                "joints_pos": joints_vis.detach(),
                "T_w2c": est_T_w2c_vis.detach(),
                "vis_all_cam": False,
            }
            # compute gvhmr output
            pred_R_w2c = R_c @ axis_angle_to_matrix(R_w_hmr[bid]).mT
            pred_t_w2c = (
                t_c
                + offset
                - torch.einsum("fij,fj->fi", pred_R_w2c, t_w_hmr[bid] + offset)
            )
            pred_T_w2c = torch.eye(4)[None].repeat(L, 1, 1).to(device)
            pred_T_w2c[:, :3, :3] = pred_R_w2c
            pred_T_w2c[:, :3, 3] = pred_t_w2c

            smpl_out = smpl(
                betas=kwargs["betas"][bid],
                body_pose=body_pose,
                global_orient=R_w_hmr[bid],
                transl=t_w_hmr[bid],
                orig_joints=True,
            )
            joints = smpl_out.joints
            joints_vis = (vis_mat[None] @ joints.mT).mT
            pred_T_w2c_vis = pred_T_w2c.clone()
            pred_T_w2c_vis[:, :3, :3] = pred_T_w2c[:, :3, :3] @ vis_mat[None].mT
            res_orig = {
                "text": "",
                "joints_pos": joints_vis.detach(),
                "T_w2c": pred_T_w2c_vis.detach(),
                "vis_all_cam": False,
            }
            res = {
                "gvhmr": res_orig,
                "slam": res,
            }
            dataset = kwargs["meta"][0]["dataset_id"]
            if "flip" in kwargs["meta"][0]:
                dataset = dataset + "_flip"
            vid = kwargs["meta"][0]["vid"].replace("/", "_")
            html_path = f"tmp/{dataset}/{vid}.html"
            import os

            os.makedirs(f"tmp/{dataset}", exist_ok=True)
            sp_visualizer.vis_smpl_scene(res, html_path, window_size=(600, 600))

            est_t_c2w = (-pred_R_w2c.mT @ pred_t_w2c[..., None])[..., 0]
            aligned_R_w2c = est_T_w2c[:, :3, :3]
            slam_t_w2c = est_T_w2c[:, :3, 3]
            slam_t_c2w = (-aligned_R_w2c.mT @ slam_t_w2c[..., None])[..., 0]

    est_t_w2c_new = (
        t_c
        + offset
        - (est_R_w2c @ (motion_f_slam["transl"].float() + offset)[..., None])[..., 0]
    )
    est_T_w2c[:, :3, 3] = est_t_w2c_new

    smpl_params_w_Rt = {
        "global_orient": motion_f_slam["global_orient"],
        "transl": motion_f_slam["transl"],
    }

    return smpl_params_w_Rt, est_T_w2c, motion_f_slam["body_pose"], static_conf_logits


def merge_slam_w_hmr(
    w_j3d_slam,
    w_j3d_hmr,
    t_w_slam,
    t_w_hmr,
    static_conf_logits,
    static_label_,
    conf_scales,
):
    is_static = static_label_.sum(dim=-1) > 0  # (B, L-1)
    joint_ids = [
        7,
        10,
        8,
        11,
        20,
        21,
    ]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]

    j_static_slam = w_j3d_slam.clone()[:, :, joint_ids]
    j_static_hmr = w_j3d_hmr.clone()[:, :, joint_ids]
    j_disp_slam = j_static_slam[:, 1:] - j_static_slam[:, :-1]  # Corrected slicing
    j_disp_hmr = j_static_hmr[:, 1:] - j_static_hmr[:, :-1]  # Corrected slicing

    disp_slam = j_disp_slam * static_conf_logits[..., None].softmax(
        dim=-2
    )  # (B, L-1, J, 3)
    disp_slam = disp_slam * is_static[..., None, None]  # (B, L-1, J, 3)
    norm_slam = disp_slam.norm(dim=-1, keepdim=True).sum(-2)  # (B, L-1, 1)
    disp_slam = disp_slam.sum(-2)  # (B, L-1, 3)

    disp_hmr = j_disp_hmr * static_conf_logits[..., None].softmax(
        dim=-2
    )  # (B, L-1, J, 3)
    disp_hmr = disp_hmr * is_static[..., None, None]  # (B, L-1, J, 3)
    norm_hmr = disp_hmr.norm(dim=-1, keepdim=True).sum(-2)  # (B, L-1, 1)
    disp_hmr = disp_hmr.sum(-2)  # (B, L-1, 3)

    use_slam_norm = (norm_slam < norm_hmr).repeat(1, 1, 3)
    use_slam = (conf_scales[:, :-1, None] > 0.3).repeat(1, 1, 3)
    # use_slam = use_slam_norm | use_slam

    vel_slam = t_w_slam[:, 1:] - t_w_slam[:, :-1]
    vel_hmr = t_w_hmr[:, 1:] - t_w_hmr[:, :-1]
    use_hmr_vel = (vel_slam - vel_hmr).norm(dim=-1, keepdim=True) < 0.01
    use_slam = use_slam | ~use_hmr_vel

    floating_mask = (use_slam[:, :, :1].clone() & (norm_slam > 0.1)).repeat(1, 1, 6)
    static_conf_logits[floating_mask] = -1e6
    new_disp = disp_slam.clone()
    new_disp[~use_slam] = disp_hmr[~use_slam]

    w_disp_slam = t_w_slam[:, 1:] - t_w_slam[:, :-1]
    w_disp_hmr = t_w_hmr[:, 1:] - t_w_hmr[:, :-1]
    w_disp_slam[~use_slam] = w_disp_hmr[~use_slam].to(w_disp_slam.dtype)
    w_disp_slam_new = w_disp_slam - disp_hmr
    post_w_transl_hmr = torch.cumsum(
        torch.cat([t_w_slam[:, :1], w_disp_slam_new], dim=1), dim=1
    )
    post_w_transl_hmr[..., 0] = gaussian_smooth(
        post_w_transl_hmr[..., 0], sigma=5, dim=-1
    )
    post_w_transl_hmr[..., 1] = gaussian_smooth(
        post_w_transl_hmr[..., 1], sigma=5, dim=-1
    )
    post_w_transl_hmr[..., 2] = gaussian_smooth(
        post_w_transl_hmr[..., 2], sigma=5, dim=-1
    )

    use_slam = use_slam[..., [0]]
    return post_w_transl_hmr, use_slam, static_conf_logits
