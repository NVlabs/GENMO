import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from einops import einsum, rearrange, repeat
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from hmr4d.utils.net_utils import gaussian_smooth

from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
from hmr4d.model.gvhmr.utils.postprocess import (
    pp_static_joint,
    process_ik,
    pp_floor_height,
    pp_static_joint_cam,
)
from hmr4d.model.gvhmr.utils import stats_compose

from motiondiff.models.mdm.rotation_conversions import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from hmr4d.utils.geo.hmr_cam import compute_bbox_info_bedlam, compute_transl_full_cam, get_a_pred_cam, project_to_bi01, perspective_projection, get_bbx_xys, normalize_kp2d, convert_bbx_xys_to_lurb, cvt_to_bi01_p2d
from hmr4d.utils.geo.hmr_global import (
    rollout_local_transl_vel,
    get_static_joint_mask,
    get_tgtcoord_rootparam,
)
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.model.gvhmr.utils.mv2d_utils import draw_motion_2d, coco_joint_parents


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
            self.register_buffer("cam_angvel_mean", torch.tensor(cam_angvel_stats["mean"]), persistent=False)
            self.register_buffer("cam_angvel_std", torch.tensor(cam_angvel_stats["std"]), persistent=False)
            
        self.denoiser3d.endecoder = self.endecoder
        
        self.f_condition_dim = {
            "obs": 3,
            "f_cliffcam": 3,
            "f_cam_angvel": 6,
            "f_imgseq": 1024,
        }
        
    # ========== Training ========== #

    def forward(self, inputs, train=False, postproc=False, static_cam=False, global_step=0, mode=None):
        outputs = dict()
        B, L = inputs["obs"].shape[:2]

        # *. Conditions
        inputs['f_cliffcam'] = compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
        f_cam_angvel = inputs["cam_angvel"]
        if self.args.normalize_cam_angvel:
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std
        inputs["f_cam_angvel"] = f_cam_angvel
        
        f_condition = dict()
        for k in self.args.in_attr:
            if k in inputs:
                f_condition[k] = inputs[k]
            else:
                f_condition[k] = torch.zeros(B, L, self.f_condition_dim[k]).to(inputs["obs"])
        for k in self.args.mask_out_attr:
            f_condition[k] = torch.zeros_like(f_condition[k])
        f_condition_mask = inputs.get("f_condition_mask", {})
        for k in f_condition_mask:
            if k in f_condition:
                mask = f_condition_mask[k][:, None, None, None] if k in ['obs'] else f_condition_mask[k][:, None, None]
                f_condition[k] = f_condition[k] * (1 - mask.float())
        # for k, v in f_condition.items():
        #     print(k, v[1].norm())
        
        if inputs.get('text_mask', None) is not None:
            inputs['encoded_text'] = inputs['encoded_text'] * (1 - inputs['text_mask'][:, None, None].float())
           
        if train:
            f_condition = randomly_set_null_condition(f_condition, 0.1)

        eval_text_only = inputs.get('eval_text_only', False)
        if eval_text_only:
            for k in f_condition.keys():
                f_condition[k] = torch.zeros_like(f_condition[k])
        
        inputs["f_condition"] = f_condition
        # Forward & output
        model_output = self.denoiser3d(inputs, train=train, postproc=postproc, static_cam=static_cam, mode=mode)  # pred_x, pred_cam, static_conf_logits
        decode_dict = self.endecoder.decode(model_output["pred_x"])  # (B, L, C) -> dict
        outputs.update({"model_output": model_output, "decode_dict": decode_dict})
        if 'intermediate_pred_x' in model_output:
            int_decode_dict = [self.endecoder.decode(pred_x) for pred_x in model_output['intermediate_pred_x']]
            outputs['intermediate_decode_dict'] = int_decode_dict

        # Post-processing
        if self.endecoder.encode_type == 'gvhmr':
            outputs["pred_smpl_params_incam"] = {
                "body_pose": decode_dict["body_pose"],  # (B, L, 63)
                "betas": decode_dict["betas"],  # (B, L, 10)
                "global_orient": decode_dict["global_orient"],  # (B, L, 3)
                "transl": compute_transl_full_cam(model_output["pred_cam"], inputs["bbx_xys"], inputs["K_fullimg"]),
            }
            if 'intermediate_decode_dict' in outputs:
                outputs["intermediate_pred_smpl_params_incam"] = [
                    {
                        "body_pose": int_decode_dict["body_pose"],
                        "betas": int_decode_dict["betas"],
                        "global_orient": int_decode_dict["global_orient"],
                        "transl": compute_transl_full_cam(
                            model_output["pred_cam"], inputs["bbx_xys"], inputs["K_fullimg"]
                        ),
                    }
                    for int_decode_dict in outputs["intermediate_decode_dict"]
                ]
        
        if not train:
            # if eval_text_only:
            #     inputs["cam_angvel"] = torch.zeros(decode_dict["global_orient_gv"].shape[:2] + (6,), device=decode_dict["global_orient_gv"].device)
            if self.endecoder.encode_type == 'gvhmr':
                pred_smpl_params_global = get_smpl_params_w_Rt_v2(  # This function has for-loop
                    global_orient_gv=decode_dict["global_orient_gv"],
                    local_transl_vel=decode_dict["local_transl_vel"],
                    global_orient_c=decode_dict["global_orient"],
                    cam_angvel=inputs["cam_angvel"],
                )
                outputs["pred_smpl_params_global"] = {
                    "body_pose": decode_dict["body_pose"],
                    "betas": decode_dict["betas"],
                    **pred_smpl_params_global,
                }
                if 'intermediate_decode_dict' in outputs:
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
                        intermediate_pred_smpl_params_global.append(pred_smpl_params_global)
                    outputs["intermediate_pred_smpl_params_global"] = intermediate_pred_smpl_params_global
            else:
                outputs["pred_smpl_params_global"] = {
                    "body_pose": decode_dict["body_pose"],
                    "betas": decode_dict["betas"],
                    "global_orient": decode_dict["global_orient_w"],
                    "transl": decode_dict["transl_w"],
                }
            outputs["static_conf_logits"] = model_output["static_conf_logits"]

            if postproc and self.endecoder.encode_type == 'gvhmr':  # apply post-processing
                if static_cam:  # extra post-processing to utilize static camera prior
                    outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_cam(outputs, self.endecoder)
                else:
                    outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(outputs, self.endecoder)

                if "intermediate_pred_smpl_params_global" in outputs:
                    for i in range(len(outputs["intermediate_pred_smpl_params_global"])):
                        outputs["intermediate_pred_smpl_params_global"][i]["transl"] = pp_floor_height(
                            outputs["intermediate_pred_smpl_params_global"][i],
                            self.endecoder
                        )
                body_pose = process_ik(outputs, self.endecoder)
                decode_dict["body_pose"] = body_pose
                outputs["pred_smpl_params_global"]["body_pose"] = body_pose
                outputs["pred_smpl_params_incam"]["body_pose"] = body_pose

            return outputs

        # ========== Compute Loss ========== #
        total_loss = 0
        mask = inputs["mask"]["valid"]  # (B, L)
        text_only = inputs.get('text_only', None)
        
        # 1. Simple loss: MSE
        pred_x = model_output["pred_x"]  # (B, L, C)
        target_x = inputs["target_x"]  # (B, L, C)
        simple_loss = F.mse_loss(pred_x, target_x, reduction="none")
        if text_only is not None and self.weights.get('text_only_no_reg_loss', False) and mode == 'regression':
            simple_loss[text_only] = 0
        mask_simple = mask[:, :, None].expand(-1, -1, pred_x.size(2)).clone()  # (B, L, C)
        if self.endecoder.encode_type == 'gvhmr':
            mask_simple[inputs["mask"]["spv_incam_only"], :, 142:] = False  # 3dpw training
        if self.weights.get('simple_loss_local_only', False):
            mask_simple[..., 142:] = False
        simple_loss = (simple_loss * mask_simple).mean()
        total_loss += simple_loss * self.weights.get("simple", 1.0)
        outputs["simple_loss"] = simple_loss

        # 2. Extra loss
        if self.endecoder.encode_type == 'gvhmr':
            extra_funcs = [
                compute_extra_incam_loss,
                compute_extra_global_loss,
            ]
            for extra_func in extra_funcs:
                extra_loss, extra_loss_dict = extra_func(inputs, outputs, self, mode)
                total_loss += extra_loss
                outputs.update(extra_loss_dict)

        outputs["loss"] = total_loss
        return outputs
    
    def forward_2d(self, inputs, train=False, global_step=0, mode=None):
        outputs = dict()
        device = inputs["obs"].device
        B, L = inputs["obs"].shape[:2]
        
        if inputs['use_cliffcam'].any():
            inputs['f_cliffcam'] = compute_bbox_info_bedlam(inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
            inputs['f_cliffcam'][~inputs['use_cliffcam']] = 0
        if 'cam_angvel' in inputs:
            f_cam_angvel = inputs["cam_angvel"]
            if self.args.normalize_cam_angvel:
                f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std
            inputs["f_cam_angvel"] = f_cam_angvel

        # input view generation
        f_condition = dict()
        for k in self.args.in_attr:
            if k in inputs:
                f_condition[k] = inputs[k]
            else:
                f_condition[k] = torch.zeros(B, L, self.f_condition_dim[k]).to(inputs["obs"])
        for k in self.args.mask_out_attr:
            f_condition[k] = torch.zeros_like(f_condition[k])
        f_condition_mask = inputs.get("f_condition_mask", {})
        for k in f_condition_mask:
            if k in f_condition:
                mask = f_condition_mask[k][:, None, None, None] if k in ['obs'] else f_condition_mask[k][:, None, None]
                f_condition[k] = f_condition[k] * (1 - mask.float())
                
        if inputs.get('text_mask', None) is not None:
            inputs['encoded_text'] = inputs['encoded_text'] * (1 - inputs['text_mask'][:, None, None].float())
           
        if train:
            f_condition = randomly_set_null_condition(f_condition, 0.1)

        inputs["f_condition"] = f_condition
        model_output = self.denoiser3d.forward_train_2d(inputs, mode)  # pred_x, pred_cam, static_conf_logits
        if 'decode_dict' in model_output:
            decode_dict = model_output.pop("decode_dict")
        else:
            decode_dict = self.endecoder.decode(model_output["pred_x"])  # (B, L, C) -> dict
        outputs.update({"2d_model_output": model_output, "2d_decode_dict": decode_dict})

        # Post-processing
        if self.endecoder.encode_type == 'gvhmr':
            outputs["2d_pred_smpl_params_incam"] = {
                "body_pose": decode_dict["body_pose"],  # (B, L, 63)
                "betas": decode_dict["betas"],  # (B, L, 10)
                "global_orient": decode_dict["global_orient"],  # (B, L, 3)
                "transl": compute_transl_full_cam(model_output["pred_cam"], inputs["bbx_xys"], inputs["K_fullimg"]),
            }

        # ========== Compute Loss ========== #
        total_loss = 0
        mask = inputs["mask"]
        text_only_losses = self.weights.get('text_only_losses', 'all')
        text_only = inputs.get('text_only', None)
        
        # 1. Simple loss: MSE
        if self.weights.get('simple_diffusion2d', 0.0) > 0.0 and mode == 'diffusion':
            pred_x = model_output["pred_x"]  # (B, L, C)
            target_x = inputs['regression_outputs']['2d_model_output']['pred_x']
            simple_loss = F.mse_loss(pred_x, target_x, reduction="none")
            mask_simple = mask[:, :, None].expand(-1, -1, pred_x.size(2)).clone()  # (B, L, C)
            if self.weights.get('simple_loss_diffusion2d_local_only', False):
                mask_simple[..., 142:] = False
            simple_loss = (simple_loss * mask_simple).mean()
            total_loss += simple_loss * self.weights.simple_diffusion2d
            outputs["simple_loss"] = simple_loss
        else:
            outputs["simple_loss"] = 0.0

        # 2. Extra loss
        model_output = outputs["2d_model_output"]
        endecoder = self.endecoder
        # mask_reproj = ~inputs["mask"]["spv_incam_only"]  # do not supervise reproj for 3DPW

        if self.endecoder.encode_type == 'gvhmr':
            if self.weights.get('j2d_train2d', 0.0) > 0.0 and not (mode == 'regression' and self.weights.train2d_skip_regression):
                pred_c_j17 = endecoder.smplx_model(**outputs["2d_pred_smpl_params_incam"])[1]
                conf_2d = inputs['conf']
                pred_c_j17[conf_2d < 0.1] = 0.0
                # prevent divide 0 or small value to overflow(fp16)
                reproj_z_thr = 0.3
                pred_c_j3d_z0_mask = pred_c_j17[..., 2].abs() <= reproj_z_thr
                pred_c_j17[pred_c_j3d_z0_mask] = reproj_z_thr

                # project and normalize
                pred_j2d_01 = project_to_bi01(pred_c_j17, inputs["bbx_xys"], inputs["K_fullimg"])
                pred_j2d_01[conf_2d < 0.1] = 0.0
                
                if self.weights.get('proj_gt_j2d_to_bi01', False):
                    bbx_lurb = convert_bbx_xys_to_lurb(inputs["bbx_xys"])
                    gt_c_j17_2d = cvt_to_bi01_p2d(inputs['obs_kp2d_raw'], bbx_lurb)
                    gt_c_j17_2d[~inputs["mask"]] = 0
                else:
                    gt_c_j17_2d = inputs["orig_obs"][..., :2]
                    
                j2d_loss = F.mse_loss(pred_j2d_01, gt_c_j17_2d, reduction="none")
                if text_only is not None and text_only_losses != 'all' and 'j2d_train2d' not in text_only_losses:
                    j2d_loss[text_only] = 0
                j2d_loss = (j2d_loss * mask[..., None, None] * conf_2d[..., None]).mean()

                total_loss += j2d_loss * self.weights.j2d_train2d
                outputs["j2d_loss_2d"] = j2d_loss
                
            if self.weights.get('norm_j2d', 0.0) > 0.0 and not (mode == 'regression' and self.weights.train2d_skip_regression):
                pred_c_j17 = endecoder.smplx_model(**outputs["2d_pred_smpl_params_incam"])[1]
                conf_2d = inputs['conf']
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
                if text_only is not None and text_only_losses != 'all' and 'norm_j2d' not in text_only_losses:
                    norm_j2d_loss[text_only] = 0
                norm_j2d_loss = (norm_j2d_loss * mask[..., None, None] * conf_2d[..., None]).mean()

                total_loss += norm_j2d_loss * self.weights.norm_j2d
                outputs["norm_j2d_loss_2d"] = norm_j2d_loss

        outputs["loss_2d"] = total_loss
        return outputs
    

def randomly_set_null_condition(f_condition, uncond_prob=0.1):
    """Conditions are in shape (B, L, *)"""
    keys = list(f_condition.keys())
    for k in keys:
        if f_condition[k] is None or type(f_condition[k]) == bool:
            continue
        f_condition[k] = f_condition[k].clone()
        mask = torch.rand(f_condition[k].shape[:2]) < uncond_prob
        f_condition[k][mask] = 0.0
    return f_condition


def compute_extra_incam_loss(inputs, outputs, ppl, mode):
    model_output = outputs["model_output"]
    decode_dict = outputs["decode_dict"]
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args
    
    text_only_losses = weights.get('text_only_losses', 'all')
    text_only = inputs.get('text_only', None)
    if weights.get('text_only_no_reg_loss', False) and mode == 'regression':
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
        if text_only is not None and text_only_losses != 'all' and 'cr_j3d' not in text_only_losses:
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
        gt_pred_cam = get_a_pred_cam(gt_transl, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
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
        if text_only is not None and text_only_losses != 'all' and 'transl_c' not in text_only_losses:
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

        pred_j2d_01 = project_to_bi01(pred_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"])
        gt_j2d_01 = project_to_bi01(gt_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, J, 2)

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
        if text_only is not None and text_only_losses != 'all' and 'j2d' not in text_only_losses:
            j2d_loss[text_only] = 0
        j2d_loss = (j2d_loss * mask[..., None, None] * valid_mask).mean()

        extra_loss += j2d_loss * weights.j2d
        extra_loss_dict["j2d_loss"] = j2d_loss

    if weights.cr_verts > 0:
        # SMPL forward
        pred_c_verts437, pred_c_j17 = endecoder.smplx_model(**outputs["pred_smpl_params_incam"])
        root_ = pred_c_j17[:, :, [11, 12], :].mean(-2, keepdim=True)
        pred_cr_verts437 = pred_c_verts437 - root_

        gt_cr_verts437 = inputs["gt_cr_verts437"]  # (B, L, 437, 3)
        cr_vert_loss = F.mse_loss(pred_cr_verts437, gt_cr_verts437, reduction="none")
        if text_only is not None and text_only_losses != 'all' and 'cr_verts' not in text_only_losses:
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

        pred_verts2d_01 = project_to_bi01(pred_c_verts437, inputs["bbx_xys"], inputs["K_fullimg"])
        gt_verts2d_01 = project_to_bi01(gt_c_verts437, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 437, 2)

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
        if text_only is not None and text_only_losses != 'all' and 'verts2d' not in text_only_losses:
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
    
    text_only_losses = weights.get('text_only_losses', 'all')
    text_only = inputs.get('text_only', None)
    if weights.get('text_only_no_reg_loss', False) and mode == 'regression':
        text_only_losses = []

    if weights.transl_w > 0:
        # compute pred_transl_w by rollout
        gt_transl_w = inputs["smpl_params_w"]["transl"]
        gt_global_orient_w = inputs["smpl_params_w"]["global_orient"]
        local_transl_vel = decode_dict["local_transl_vel"]
        pred_transl_w = rollout_local_transl_vel(local_transl_vel, gt_global_orient_w, gt_transl_w[:, [0]])

        trans_w_loss = F.l1_loss(pred_transl_w, gt_transl_w, reduction="none")
        if text_only is not None and text_only_losses != 'all' and 'transl_w' not in text_only_losses:
            trans_w_loss[text_only] = 0
        trans_w_loss = (trans_w_loss * mask[..., None]).mean()
        extra_loss += trans_w_loss * weights.transl_w
        extra_loss_dict["transl_w_loss"] = trans_w_loss

    # Static-Conf loss
    if weights.static_conf_bce > 0:
        # Compute gt by thresholding velocity
        vel_thr = args.static_conf.vel_thr
        assert vel_thr > 0
        joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
        gt_w_j3d = endecoder.fk_v2(**inputs["smpl_params_w"])  # (B, L, J=22, 3)
        static_gt = get_static_joint_mask(gt_w_j3d, vel_thr=vel_thr, repeat_last=True)  # (B, L, J)
        static_gt = static_gt[:, :, joint_ids].float()  # (B, L, J')
        pred_static_conf_logits = outputs["model_output"]["static_conf_logits"]

        static_conf_loss = F.binary_cross_entropy_with_logits(pred_static_conf_logits, static_gt, reduction="none")
        if text_only is not None and text_only_losses != 'all' and 'static_conf_bce' not in text_only_losses:
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
    view_axis_gv = R_c2gv[:, :, :, 2]  # (B, L, 3)  Rc2gv is estimated, so the x-axis is not accurate, i.e. != 0

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
    aa_tp1_to_t_angle = torch.acos(torch.clamp((vec1_xyz * vec2_xyz).sum(dim=-1, keepdim=True), -1.0, 1.0))
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
    global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf="any->ay")

    smpl_params_w_Rt = {"global_orient": global_orient, "transl": transl}
    return smpl_params_w_Rt
