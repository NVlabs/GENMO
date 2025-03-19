import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from einops import einsum, rearrange
from hydra.utils import instantiate
from transformers import T5EncoderModel, T5Tokenizer

from hmr4d.configs import MainStore, builds
from hmr4d.model.gvhmr.utils.postprocess import (
    pp_static_joint,
    pp_static_joint_cam,
    process_ik,
)
from hmr4d.utils.geo.augment_noisy_pose import (
    get_invisible_legs_mask,
    get_visible_mask,
    get_wham_aug_kp3d,
    randomly_modify_hands_legs,
    randomly_occlude_lower_half,
)
from hmr4d.utils.geo.flip_utils import avg_smplx_aa, flip_smplx_params
from hmr4d.utils.geo.hmr_cam import (
    get_bbx_xys,
    normalize_kp2d,
    perspective_projection,
    safely_render_x3d_K,
)
from hmr4d.utils.geo.hmr_global import get_static_joint_mask, rollout_vel
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import save_video
from hmr4d.utils.vis.cv2_utils import draw_bbx_xys_on_image_batch
from hmr4d.utils.wis3d_utils import add_motion_as_lines, make_wis3d
from motiondiff.diffusion.resample import create_named_schedule_sampler
from motiondiff.models.common.cfg_sampler import ClassifierFreeSampleModel
from motiondiff.models.model_util import create_gaussian_diffusion
from motiondiff.utils.tools import Timer
from motiondiff.utils.torch_transform import (
    angle_axis_to_rotation_matrix,
    inverse_transform,
    make_transform,
    transform_trans,
)
from motiondiff.utils.torch_utils import interp_tensor_with_scipy

from .utils.mv2d_utils import (
    coco_joint_parents,
    draw_motion_2d,
    generate_cam,
    project_keypoints,
    recon_from_2d,
)


class UNIMFM_demo(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        model_cfg,
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.model_cfg = model_cfg

        # SMPLX
        self.smplx = make_smplx("supermotion_v437coco17")
        self.max_text_len = 50
        if "text_encoder" in model_cfg:
            self.use_text_encoder = True
            if model_cfg.text_encoder.get("load_llm", False):
                llm_version = model_cfg.text_encoder.llm_version
                self.max_text_len = model_cfg.text_encoder.max_text_len
                text_encoder, self.tokenizer = self.load_and_freeze_llm(llm_version)
                self.text_encoder = [text_encoder.cuda()]
            else:
                self.text_encoder = self.tokenizer = None
        else:
            self.use_text_encoder = False

    def load_and_freeze_llm(self, llm_version):
        tokenizer = T5Tokenizer.from_pretrained(llm_version)
        model = T5EncoderModel.from_pretrained(llm_version)
        # Freeze llm weights
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model, tokenizer

    def init_condition_exists(self, batch):
        B, L = batch["obs"].shape[:2]
        f_condition_exists = dict()
        # import ipdb; ipdb.set_trace()
        if self.model_cfg.get("perframe_condition_exists", False) or True:
            f_condition_exists["obs"] = batch["obs"].view(B, L, -1).norm(dim=-1) > 1e-4
            f_condition_exists["f_cliffcam"] = f_condition_exists["obs"].clone()
            f_condition_exists["f_imgseq"] = (
                batch["f_imgseq"].view(B, L, -1).norm(dim=-1) > 1e-4
            )
            if "cam_angvel" in batch:
                f_condition_exists["f_cam_angvel"] = (
                    batch["cam_angvel"].view(B, L, -1).norm(dim=-1) > 1e-4
                )
            else:
                f_condition_exists["f_cam_angvel"] = (
                    torch.zeros(B, L).bool().to(batch["obs"].device)
                )
            if "encoded_music" in batch:
                f_condition_exists["encoded_music"] = (
                    batch["encoded_music"].view(B, L, -1).norm(dim=-1) > 1e-4
                )
            else:
                f_condition_exists["encoded_music"] = (
                    torch.zeros(B, L).bool().to(batch["obs"].device)
                )
            if "encoded_audio" in batch:
                f_condition_exists["encoded_audio"] = (
                    batch["encoded_audio"].view(B, L, -1).norm(dim=-1) > 1e-4
                )
            else:
                f_condition_exists["encoded_audio"] = (
                    torch.zeros(B, L).bool().to(batch["obs"].device)
                )
        else:
            f_condition_exists["obs"] = (
                (batch["obs"].view(B, -1).norm(dim=-1) > 1e-4)
                .unsqueeze(-1)
                .repeat(1, L)
            )
            f_condition_exists["f_cliffcam"] = f_condition_exists["obs"].clone()
            f_condition_exists["f_imgseq"] = (
                (batch["f_imgseq"].view(B, -1).norm(dim=-1) > 1e-4)
                .unsqueeze(-1)
                .repeat(1, L)
            )
            if "cam_angvel" in batch:
                f_condition_exists["f_cam_angvel"] = (
                    (batch["cam_angvel"].view(B, -1).norm(dim=-1) > 1e-4)
                    .unsqueeze(-1)
                    .repeat(1, L)
                )
            else:
                f_condition_exists["f_cam_angvel"] = (
                    torch.zeros(B, L).bool().to(batch["obs"].device)
                )
            if "encoded_music" in batch:
                f_condition_exists["encoded_music"] = (
                    (batch["encoded_music"].view(B, -1).norm(dim=-1) > 1e-4)
                    .unsqueeze(-1)
                    .repeat(1, L)
                )
            else:
                f_condition_exists["encoded_music"] = (
                    torch.zeros(B, L).bool().to(batch["obs"].device)
                )
            if "encoded_audio" in batch:
                f_condition_exists["encoded_audio"] = (
                    (batch["encoded_audio"].view(B, -1).norm(dim=-1) > 1e-4)
                    .unsqueeze(-1)
                    .repeat(1, L)
                )
            else:
                f_condition_exists["encoded_audio"] = (
                    torch.zeros(B, L).bool().to(batch["obs"].device)
                )
        batch["f_condition_exists"] = f_condition_exists

    def create_condition_mask(self, batch, cond_mask_cfg, mode):
        device = batch["obs"].device
        reuse_regression_mask = cond_mask_cfg.get("reuse_regression_mask", True)
        regression_no_img_mask = cond_mask_cfg.get("regression_no_img_mask", False)
        mask_text_prob = cond_mask_cfg.get("mask_text_prob", {}).get(mode, 0.0)
        mask_img_prob = cond_mask_cfg.get("mask_img_prob", 0.0)
        mask_cam_prob = cond_mask_cfg.get("mask_cam_prob", 0.0)
        mask_f_imgseq_prob = cond_mask_cfg.get("mask_f_imgseq_prob", 0.0)
        self.init_condition_exists(batch)

        if mask_text_prob > 0:
            mask_text = (torch.rand(batch["B"]) < mask_text_prob).to(device)
            batch["text_mask"] = mask_text
        else:
            batch["text_mask"] = None
        if batch.get("text_mask", None) is not None:
            batch["has_text"][batch["text_mask"]] = False

        if (
            reuse_regression_mask
            and mode == "diffusion"
            and "regression_outputs" in batch
        ):
            batch["f_condition_mask"] = batch["regression_outputs"]["f_condition_mask"]
            batch["gen_only"] = batch["regression_outputs"]["gen_only"]
        else:
            f_condition_mask = dict()
            has_text = batch["has_text"]
            has_audio = batch["has_audio"]
            has_music = batch["has_music"]
            if regression_no_img_mask and mode == "regression":
                mask_img_prob = 0
                mask_f_imgseq_prob = 0
            if mask_img_prob > 0:
                mask_img = (has_text | has_audio | has_music) & (
                    torch.rand(batch["B"]) < mask_img_prob
                ).to(device)
                for k in ["obs", "f_cliffcam", "f_imgseq"]:
                    f_condition_mask[k] = mask_img  # 1: not used, 0: used
                batch["gen_only"] = mask_img
            if mask_cam_prob > 0:
                mask_cam = (has_text | has_music | has_audio) & (
                    torch.rand(batch["B"]) < mask_cam_prob
                ).to(device)
                for k in ["f_cam_angvel"]:
                    f_condition_mask[k] = mask_cam
            if mask_f_imgseq_prob > 0:
                mask_f_imgseq = (torch.rand(batch["B"]) < mask_f_imgseq_prob).to(device)
                if "f_imgseq" in f_condition_mask:
                    f_condition_mask["f_imgseq"] = (
                        f_condition_mask["f_imgseq"] | mask_f_imgseq
                    )
                else:
                    f_condition_mask["f_imgseq"] = mask_f_imgseq
            for k in f_condition_mask.keys():
                batch["f_condition_exists"][k][f_condition_mask[k]] = False
            batch["f_condition_mask"] = f_condition_mask

    def prepare_3d_batch(self, batch):
        if "text_embed" in batch:
            batch["encoded_text"] = batch["text_embed"].cuda()
        elif self.use_text_encoder:
            batch["encoded_text"] = self.encode_text(
                batch["caption"], batch["has_text"]
            )
        if "music_embed" in batch:
            batch["encoded_music"] = batch["music_embed"].cuda()
        else:
            batch["encoded_music"] = torch.zeros(
                batch["obs"].shape[:2] + (self.pipeline.args.encoded_music_dim,),
                device=batch["obs"].device,
            )
        if "audio_array" in batch:
            encoded_audio = (
                self.audio_encoder(batch["audio_array"].cuda().unsqueeze(1))
                .transpose(1, 2)
                .contiguous()
            )
            has_audio = batch["has_audio"]
            encoded_audio[~has_audio] = 0
            batch["encoded_audio"] = encoded_audio
        else:
            bs, T = batch["obs"].shape[:2]
            batch["encoded_audio"] = torch.zeros(bs, T, 128, device=batch["obs"].device)

        batch["target_x"] = self.endecoder.encode(batch)  # (B, L, C)
        batch["static_gt"] = self.endecoder.get_static_gt(
            batch, self.pipeline.args.static_conf.vel_thr
        )  # (B, L, 6)

        B, F = batch["smpl_params_c"]["body_pose"].shape[:2]

        # Create augmented noisy-obs : gt_j3d(coco17)
        with torch.no_grad():
            gt_verts437, gt_j3d = self.smplx(**batch["smpl_params_c"])
            root_ = gt_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
            batch["gt_j3d"] = gt_j3d
            batch["gt_cr_coco17"] = gt_j3d - root_
            batch["gt_c_verts437"] = gt_verts437
            batch["gt_cr_verts437"] = gt_verts437 - root_

        # bbx_xys
        i_x2d = safely_render_x3d_K(gt_verts437, batch["K_fullimg"], thr=0.3)
        bbx_xys = get_bbx_xys(i_x2d, do_augment=True)
        if False:  # trust image bbx_xys seems better
            batch["bbx_xys"] = bbx_xys
        else:
            mask_bbx_xys = batch["mask"]["bbx_xys"]
            batch["bbx_xys"][~mask_bbx_xys] = bbx_xys[~mask_bbx_xys].to(
                batch["bbx_xys"]
            )

        # noisy_j3d -> project to i_j2d -> compute a bbx -> normalized kp2d [-1, 1]

        noisy_j3d = gt_j3d + get_wham_aug_kp3d(gt_j3d.shape[:2])
        if True:
            noisy_j3d = randomly_modify_hands_legs(noisy_j3d)
        obs_i_j2d = perspective_projection(
            noisy_j3d, batch["K_fullimg"]
        )  # (B, L, J, 2)
        j2d_visible_mask = get_visible_mask(gt_j3d.shape[:2]).cuda()  # (B, L, J)
        j2d_visible_mask[noisy_j3d[..., 2] < 0.3] = (
            False  # Set close-to-image-plane points as invisible
        )
        if True:  # Set both legs as invisible for a period
            legs_invisible_mask = get_invisible_legs_mask(
                gt_j3d.shape[:2]
            ).cuda()  # (B, L, J)
            j2d_visible_mask[legs_invisible_mask] = False
        if "mask_cfg" in self.model_cfg:
            mask = self.generate_mask(
                self.model_cfg.mask_cfg, j2d_visible_mask, batch["length"]
            )
            j2d_visible_mask = j2d_visible_mask & mask
        if "body_mask_cfg" in self.model_cfg:
            mask = self.generate_mask(
                self.model_cfg.body_mask_cfg, j2d_visible_mask, batch["length"]
            )
            j2d_visible_mask = j2d_visible_mask & mask
        if self.model_cfg.get("mask_occluded_imgfeats", False) and "f_imgseq" in batch:
            occluded_img_mask = (~j2d_visible_mask).all(dim=-1)
            batch["f_imgseq"][occluded_img_mask] = 0

        obs_kp2d = torch.cat(
            [obs_i_j2d, j2d_visible_mask[:, :, :, None].float()], dim=-1
        )  # (B, L, J, 3)
        obs = normalize_kp2d(obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
        # vis_ind = 0
        # mv2d_norm = obs.unsqueeze(2)
        # mv2d_norm = torch.cat([mv2d_norm, (mv2d_norm[..., [11], :] + mv2d_norm[..., [12], :]) * 0.5], dim=-2)
        # draw_motion_2d((mv2d_norm[vis_ind, ..., :2].cpu() + 1.0) * 500, f"out/debug_vis/mask_infill.mp4", coco_joint_parents, 1000, 1000, fps=30, mask=mv2d_norm[vis_ind, ..., 2].cpu())

        obs[~j2d_visible_mask] = 0  # if not visible, set to (0,0,0)
        batch["obs"] = obs
        batch["j2d_visible_mask"] = j2d_visible_mask

        if True:  # Use some detected vitpose (presave data)
            prob = 0.5
            mask_real_vitpose = (torch.rand(B).to(obs_kp2d) < prob) * batch["mask"][
                "vitpose"
            ]
            batch["obs"][mask_real_vitpose] = normalize_kp2d(
                batch["kp2d"], batch["bbx_xys"]
            )[mask_real_vitpose]

        # Set untrusted frames to False
        batch["obs"][~batch["mask"]["valid"]] = 0
        return batch

    def train_3d_step(self, batch, batch_idx, mode):
        batch = batch.copy()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                # print(k, v.shape)
                batch[k] = v.detach().clone()

        cond_mask_cfg = self.model_cfg.get("condition_mask", {})
        self.create_condition_mask(batch, cond_mask_cfg, mode)

        # Forward and get loss
        outputs = self.pipeline.forward(
            batch, train=True, global_step=self.trainer.global_step, mode=mode
        )
        outputs["batch_size"] = batch["B"]
        outputs["f_condition_mask"] = batch["f_condition_mask"]
        outputs["gen_only"] = batch.get("gen_only", None)
        outputs[f"{mode}_condition"] = {
            "f_condition_mask": batch["f_condition_mask"],
            "gen_only": batch.get("gen_only", None),
        }
        return outputs

    def prepare_2d_batch(self, batch):
        if "text_embed" in batch:
            batch["encoded_text"] = batch["text_embed"].cuda()
        elif self.use_text_encoder:
            batch["encoded_text"] = self.encode_text(
                batch["caption"], batch["has_text"]
            )

        batch["obs_kp2d_raw"] = batch["obs_kp2d"].squeeze(2).clone()
        obs_kp2d = batch["obs_kp2d"].squeeze(2)
        conf = batch["conf"]
        aug_bbox = self.model_cfg.get("train_2d_aug_bbox", False)
        batch["bbx_xys"] = get_bbx_xys(obs_kp2d, do_augment=aug_bbox)

        orig_obs_kp2d = obs_kp2d.clone()
        orig_obs_kp2d = torch.cat(
            [orig_obs_kp2d, conf[:, :, :, None].float()], dim=-1
        )  # (B, L, J, 3)
        batch["orig_obs"] = normalize_kp2d(
            orig_obs_kp2d, batch["bbx_xys"]
        )  # (B, L, J, 3)
        batch["orig_obs"][~batch["mask"]] = 0

        noisy_2d_obs = self.model_cfg.get("noisy_2d_obs", False)
        if noisy_2d_obs:
            aug = get_wham_aug_kp3d(obs_kp2d.shape[:2])[..., :2]
            f = torch.tensor([1024.0, 1024.0]).to(aug) / 4.0
            aug *= f * self.model_cfg.kp2d_noise_scale
            obs_kp2d += aug
            obs_kp2d = randomly_modify_hands_legs(obs_kp2d)
            j2d_visible_mask = get_visible_mask(obs_kp2d.shape[:2]).cuda()  # (B, L, J)
            legs_invisible_mask = get_invisible_legs_mask(
                obs_kp2d.shape[:2]
            ).cuda()  # (B, L, J)
            j2d_visible_mask[legs_invisible_mask] = False
            j2d_visible_mask *= conf > 0.5
        else:
            j2d_visible_mask = conf > 0.5
        if "mask_cfg" in self.model_cfg:
            mask = self.generate_mask(
                self.model_cfg.mask_cfg, j2d_visible_mask, batch["length"]
            )
            j2d_visible_mask = j2d_visible_mask & mask
        if "body_mask_cfg" in self.model_cfg:
            mask = self.generate_mask(
                self.model_cfg.body_mask_cfg, j2d_visible_mask, batch["length"]
            )
            j2d_visible_mask = j2d_visible_mask & mask
        if self.model_cfg.get("mask_occluded_imgfeats", False) and "f_imgseq" in batch:
            occluded_img_mask = (~j2d_visible_mask).all(dim=-1)
            batch["f_imgseq"][occluded_img_mask] = 0
        obs_kp2d = torch.cat(
            [obs_kp2d, j2d_visible_mask[:, :, :, None].float()], dim=-1
        )  # (B, L, J, 3)
        obs = normalize_kp2d(obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
        if self.model_cfg.get("train2d_mask_invis_obs", False):
            obs[~j2d_visible_mask] = 0  # if not visible, set to (0,0,0)
        obs[~batch["mask"]] = 0
        batch["obs"] = obs
        # vis_ind = 0
        # mv2d_norm = obs.unsqueeze(2)
        # mv2d_norm = torch.cat([mv2d_norm, (mv2d_norm[..., [11], :] + mv2d_norm[..., [12], :]) * 0.5], dim=-2)
        # draw_motion_2d((mv2d_norm[vis_ind, ..., :2].cpu() + 1.0) * 500, f"out/debug_vis/motionx_test_noisy.mp4", coco_joint_parents, 1000, 1000, fps=30, mask=mv2d_norm[vis_ind, ..., 2].cpu())
        # mv2d_norm = batch["orig_obs"].unsqueeze(2)
        # mv2d_norm = torch.cat([mv2d_norm, (mv2d_norm[..., [11], :] + mv2d_norm[..., [12], :]) * 0.5], dim=-2)
        # draw_motion_2d((mv2d_norm[vis_ind, ..., :2].cpu() + 1.0) * 500, f"out/debug_vis/motionx_test_obs.mp4", coco_joint_parents, 1000, 1000, fps=30, mask=mv2d_norm[vis_ind, ..., 2].cpu())
        return batch

    def generate_mask(self, mask_cfg, orig_mask, length):
        _cfg = mask_cfg
        mask = torch.ones_like(orig_mask)
        drop_prob = _cfg.get("drop_prob", 0.0)
        if drop_prob <= 0:
            return mask
        max_num_drops = _cfg.get("max_num_drops", 1)
        min_drop_nframes = _cfg.get("min_drop_nframes", 1)
        max_drop_nframes = _cfg.get("max_drop_nframes", 30)
        joint_drop_prob = _cfg.get("joint_drop_prob", 0.0)
        for i in range(orig_mask.shape[0]):
            mlen = length[i].item()
            if np.random.rand() < drop_prob:
                num_drops = np.random.randint(1, max_num_drops + 1)
                for _ in range(num_drops):
                    drop_len = np.random.randint(
                        min_drop_nframes, min(max_drop_nframes, mlen) + 1
                    )
                    drop_start = np.random.randint(0, max(mlen - drop_len, 1))
                    if joint_drop_prob > 0:
                        drop_joints = np.random.rand(17) < joint_drop_prob
                        mask[i, drop_start : drop_start + drop_len, drop_joints] = False
                    else:
                        mask[i, drop_start : drop_start + drop_len] = False
                    # print(f"Drop {i} {drop_start} {drop_len}")
        if joint_drop_prob > 0:
            COCO17_TREE = [
                [5, 6],
                0,
                0,
                1,
                2,
                -1,
                -1,
                5,
                6,
                7,
                8,
                -1,
                -1,
                11,
                12,
                13,
                14,
                15,
                15,
                15,
                16,
                16,
                16,
            ]
            for child in range(17):
                parent = COCO17_TREE[child]
                if parent == -1:
                    continue
                if isinstance(parent, list):
                    mask[..., child] *= mask[..., parent[0]] * mask[..., parent[1]]
                else:
                    mask[..., child] *= mask[..., parent]
        return mask

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if "is_2d" in batch and batch["is_2d"][0]:
            return self.validation_2d(batch, batch_idx, dataloader_idx)
        else:
            return self.validation_3d(batch, batch_idx, dataloader_idx)

    def validation_2d(self, batch, batch_idx, dataloader_idx=0):
        do_postproc = self.trainer.state.stage == "test"  # Only apply postproc in test
        if self.infer_version == 3:
            do_postproc = False
        B = batch["obs_kp2d"].shape[0]
        obs_kp2d = batch["obs_kp2d"].squeeze(2)
        conf = batch["conf"]
        batch["bbx_xys"] = get_bbx_xys(obs_kp2d, do_augment=False)

        orig_obs_kp2d = obs_kp2d.clone()
        orig_obs_kp2d = torch.cat(
            [orig_obs_kp2d, conf[:, :, :, None].float()], dim=-1
        )  # (B, L, J, 3)
        batch["orig_obs"] = normalize_kp2d(
            orig_obs_kp2d, batch["bbx_xys"]
        )  # (B, L, J, 3)
        batch["orig_obs"][~batch["mask"]] = 0

        noisy_2d_obs = self.model_cfg.get("test_with_noisy_2d_obs", False)
        test_2d_with_mask = self.model_cfg.get("test_2d_with_mask", True)
        if noisy_2d_obs:
            aug = get_wham_aug_kp3d(obs_kp2d.shape[:2])[..., :2]
            f = torch.tensor([1024.0, 1024.0]).to(aug) / 4.0
            aug *= f * self.model_cfg.kp2d_noise_scale
            obs_kp2d += aug
            obs_kp2d = randomly_modify_hands_legs(obs_kp2d)
            j2d_visible_mask = get_visible_mask(obs_kp2d.shape[:2]).cuda()  # (B, L, J)
            legs_invisible_mask = get_invisible_legs_mask(
                obs_kp2d.shape[:2]
            ).cuda()  # (B, L, J)
            j2d_visible_mask[legs_invisible_mask] = False
            j2d_visible_mask *= conf > 0.5
        else:
            j2d_visible_mask = conf > 0.5
        if test_2d_with_mask and "mask_cfg" in self.model_cfg:
            mask = self.generate_mask(
                self.model_cfg.mask_cfg, j2d_visible_mask, batch["length"]
            )
            j2d_visible_mask = j2d_visible_mask & mask
        obs_kp2d = torch.cat(
            [obs_kp2d, j2d_visible_mask[:, :, :, None].float()], dim=-1
        )  # (B, L, J, 3)
        obs = normalize_kp2d(obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
        obs[~batch["mask"]] = 0
        batch["obs"] = obs
        batch["eval_gen_only"] = batch["meta"][0].get("eval_gen_only", False)
        if "text_embed" in batch:
            batch["encoded_text"] = batch["text_embed"].cuda()
        elif self.use_text_encoder:
            batch["encoded_text"] = self.encode_text(
                batch["caption"], batch["has_text"]
            )
        if "cam_angvel" not in batch:
            batch["cam_angvel"] = torch.zeros(
                batch["obs"].shape[:2] + (6,), device=batch["obs"].device
            )
        self.init_condition_exists(batch)
        # Forward and get loss
        if self.infer_mode == "regression":
            outputs = self.pipeline.forward_2d(
                batch,
                train=False,
                postproc=do_postproc,
                global_step=self.trainer.global_step,
            )
        else:
            outputs = self.infer_diffusion(batch)
        outputs["2d_pred_smpl_params_global"] = {
            k: v[0] for k, v in outputs["2d_pred_smpl_params_global"].items()
        }
        if "2d_pred_smpl_params_incam" in outputs:
            outputs["2d_pred_smpl_params_incam"] = {
                k: v[0] for k, v in outputs["2d_pred_smpl_params_incam"].items()
            }
        outputs["eval_gen_only"] = batch["eval_gen_only"]
        outputs["batch"] = batch
        outputs["vis_2d"] = self.model_cfg.get("vis_2d", False)
        return outputs

    @torch.no_grad()
    def predict(self, data, static_cam=False):
        # ROPE inference
        eval_gen_only = data["meta"][0].get("eval_gen_only", False)
        batch = {
            "length": data["length"][None],
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
            "bbx_xys": data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "cam_tvel": data["cam_tvel"][None],
            "R_w2c": data["R_w2c"][None],
            "f_imgseq": data["f_imgseq"][None],
        }
        batch = {k: v.cuda() for k, v in batch.items()}
        if "meta" in data:
            batch["meta"] = data["meta"]
        else:
            batch["meta"] = None
        if "eval_gen_only_mask" in data:
            batch["eval_gen_only_mask"] = data["eval_gen_only_mask"][None].cuda()
        batch["eval_gen_only"] = eval_gen_only
        if "vimo_smpl_params" in data:
            batch["vimo_smpl_params"] = {
                k: v[None].cuda() for k, v in data["vimo_smpl_params"].items()
            }
            batch["scales"] = data["scales"][None].cuda()
            batch["mean_scale"] = torch.tensor(data["mean_scale"])[None].cuda()

        if "text_embed" in batch:
            batch["encoded_text"] = batch["text_embed"].cuda()
        else:
            if "caption" in data:
                batch["caption"] = [data["caption"]]
            else:
                batch["caption"] = [""]
            batch["has_text"] = torch.tensor([True])
            batch["encoded_text"] = self.encode_text(
                batch["caption"], batch["has_text"]
            )

        if "music_embed" in batch:
            batch["encoded_music"] = batch["music_embed"].cuda()
        else:
            batch["encoded_music"] = torch.zeros(
                batch["obs"].shape[:2] + (self.pipeline.args.encoded_music_dim,),
                device=batch["obs"].device,
            )
        if "multi_text_data" in batch["meta"][0]:
            if "text_embed" not in batch["meta"][0]["multi_text_data"]:
                multi_text_data = batch["meta"][0]["multi_text_data"]
                num_text = len(multi_text_data["caption"])
                text_embed = self.encode_text(
                    multi_text_data["caption"], torch.tensor([True] * num_text)
                )
                batch["meta"][0]["multi_text_data"]["text_embed"] = text_embed

        self.init_condition_exists(batch)

        if self.pipeline.args.infer_version == 3:
            postproc = False
        else:
            postproc = True
        outputs = self.pipeline.forward(
            batch, train=False, postproc=postproc, static_cam=static_cam
        )

        pred = {
            "smpl_params_global": {
                k: v[0] for k, v in outputs["pred_smpl_params_global"].items()
            },
            "smpl_params_incam": {
                k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()
            },
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,  # intermediate outputs
        }
        return pred

    def encode_text(self, raw_text, has_text=None):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                max_text_len = self.max_text_len

                encoded = self.tokenizer.batch_encode_plus(
                    raw_text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_text_len,
                    truncation=True,
                )
                # We expect all the processing is done in GPU.
                input_ids = encoded.input_ids.to(device)
                attn_mask = encoded.attention_mask.to(device)

                with torch.no_grad():
                    output = self.text_encoder[0](
                        input_ids=input_ids, attention_mask=attn_mask
                    )
                    encoded_text = output.last_hidden_state.detach()

                encoded_text = encoded_text[:, :max_text_len]
                attn_mask = attn_mask[:, :max_text_len]
                encoded_text *= attn_mask.unsqueeze(-1)
                # for bnum in range(encoded_text.shape[0]):
                #     nvalid_elem = attn_mask[bnum].sum().item()
                #     encoded_text[bnum][nvalid_elem:] = 0
        if has_text is not None:
            no_text = ~has_text
            encoded_text[no_text] = 0
        return encoded_text

    def validation_3d(self, batch, batch_idx, dataloader_idx=0):
        # Options & Check
        try:
            stage = self.trainer.state.stage
            global_step = self.trainer.global_step
        except:
            stage = "test"
            global_step = 0
        do_postproc = stage == "test"  # Only apply postproc in test
        do_flip_test = "flip_test" in batch
        do_postproc_not_flip_test = (
            do_postproc and not do_flip_test
        )  # later pp when flip_test
        assert batch["B"] == 1, "Only support batch size 1 in evalution."
        infer_version = self.pipeline.args.get("infer_version", 2)

        if infer_version == 3:
            do_postproc = False
        # ROPE inference
        obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])

        #################### Debug validate 2d ####################
        # device = batch["obs_kp2d"].device
        # B, L = batch["obs_kp2d"].shape[:2]
        # obs = torch.zeros(B, L, 17, 3).to(device)
        # batch["bbx_xys"] = torch.zeros(B, L, 3).to(device)
        # batch["cam_angvel"] = torch.zeros(B, L, 6).to(device)
        # batch["cam_tvel"] = torch.zeros(B, L, 3).to(device)
        # batch["R_w2c"] = torch.zeros(B, L, 3, 3).to(device)
        #################### Debug ################################

        if "mask" in batch:
            mask = batch["mask"]
            if isinstance(mask, dict):
                mask = mask["valid"]
            obs[0, ~mask[0]] = 0

        eval_gen_only = batch["meta"][0].get("eval_gen_only", False)
        batch_ = {
            "length": batch["length"],
            "obs": obs,
            "bbx_xys": batch["bbx_xys"],
            "K_fullimg": batch["K_fullimg"],
            "cam_angvel": batch["cam_angvel"],
            "f_imgseq": batch["f_imgseq"],
            "eval_gen_only": eval_gen_only,
            "meta": batch["meta"],
        }
        if infer_version == 3:
            batch_["R_w2c"] = batch["R_w2c"]
            batch_["cam_tvel"] = batch["cam_tvel"]
            if "vimo_smpl_params" in batch:
                batch_["vimo_smpl_params"] = batch["vimo_smpl_params"]
                batch_["scales"] = batch["scales"]
                batch_["mean_scale"] = batch["mean_scale"]

        # batch_['gt'] = self.endecoder.encode(batch)
        # batch_['static_gt'] = self.endecoder.get_static_gt(batch, self.pipeline.args.static_conf.vel_thr)

        if "text_embed" in batch:
            batch_["encoded_text"] = batch["text_embed"].cuda()
        elif self.use_text_encoder:
            batch_["encoded_text"] = self.encode_text(
                batch["caption"], batch["has_text"]
            )
        self.init_condition_exists(batch_)
        outputs = self.pipeline.forward(
            batch_,
            train=False,
            postproc=do_postproc_not_flip_test,
            global_step=global_step,
        )
        outputs["pred_smpl_params_global"] = {
            k: v[0] for k, v in outputs["pred_smpl_params_global"].items()
        }
        if "pred_smpl_params_incam" in outputs:
            outputs["pred_smpl_params_incam"] = {
                k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()
            }
        outputs["eval_gen_only"] = eval_gen_only

        if do_flip_test:
            flip_test = batch["flip_test"]
            obs = normalize_kp2d(flip_test["kp2d"], flip_test["bbx_xys"])
            if "mask" in batch:
                mask = batch["mask"]
                if isinstance(mask, dict):
                    mask = mask["valid"]
                obs[0, ~mask[0]] = 0

            batch_ = {
                "length": batch["length"],
                "obs": obs,
                "bbx_xys": flip_test["bbx_xys"],
                "K_fullimg": batch["K_fullimg"],
                "cam_angvel": flip_test["cam_angvel"],
                "f_imgseq": flip_test["f_imgseq"],
                "meta": batch["meta"],
            }
            if infer_version == 3:
                batch_["R_w2c"] = flip_test["R_w2c"]
                batch_["cam_tvel"] = flip_test["cam_tvel"]
                if "vimo_smpl_params" in flip_test:
                    batch_["vimo_smpl_params"] = flip_test["vimo_smpl_params"]
                    batch_["scales"] = flip_test["scales"]
                    batch_["mean_scale"] = flip_test["mean_scale"]
            if "text_embed" in batch:
                batch_["encoded_text"] = batch["text_embed"].cuda()
            elif self.use_text_encoder:
                batch_["encoded_text"] = self.encode_text(
                    batch["caption"], batch["has_text"]
                )
            self.init_condition_exists(batch_)
            flipped_outputs = self.pipeline.forward(
                batch_, train=False, global_step=global_step
            )

            # First update incam results
            flipped_outputs["pred_smpl_params_incam"] = {
                k: v[0] for k, v in flipped_outputs["pred_smpl_params_incam"].items()
            }
            smpl_params1 = outputs["pred_smpl_params_incam"]
            smpl_params2 = flip_smplx_params(flipped_outputs["pred_smpl_params_incam"])

            smpl_params_avg = smpl_params1.copy()
            smpl_params_avg["betas"] = (
                smpl_params1["betas"] + smpl_params2["betas"]
            ) / 2
            smpl_params_avg["body_pose"] = avg_smplx_aa(
                smpl_params1["body_pose"], smpl_params2["body_pose"]
            )
            smpl_params_avg["global_orient"] = avg_smplx_aa(
                smpl_params1["global_orient"], smpl_params2["global_orient"]
            )
            outputs["pred_smpl_params_incam"] = smpl_params_avg

            # Then update global results
            outputs["pred_smpl_params_global"]["betas"] = smpl_params_avg["betas"]
            outputs["pred_smpl_params_global"]["body_pose"] = smpl_params_avg[
                "body_pose"
            ]

            # Finally, apply postprocess
            if do_postproc:
                # temporarily recover the original batch-dim
                outputs["pred_smpl_params_global"] = {
                    k: v[None] for k, v in outputs["pred_smpl_params_global"].items()
                }
                outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(
                    outputs, self.pipeline.endecoder
                )
                body_pose = process_ik(outputs, self.pipeline.endecoder)
                outputs["pred_smpl_params_global"] = {
                    k: v[0] for k, v in outputs["pred_smpl_params_global"].items()
                }

                outputs["pred_smpl_params_global"]["body_pose"] = body_pose[0]
                # outputs["pred_smpl_params_incam"]["body_pose"] = body_pose[0]

        return outputs

    def configure_optimizers(self):
        params = []
        for k, v in self.pipeline.named_parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = self.optimizer(params=params)

        if self.scheduler_cfg is None or self.scheduler_cfg["scheduler"] is None:
            return optimizer

        scheduler_cfg = dict(self.scheduler_cfg)
        scheduler_cfg["scheduler"] = instantiate(
            scheduler_cfg["scheduler"], optimizer=optimizer
        )
        return [optimizer], [scheduler_cfg]

    # ============== Utils ================= #
    def on_save_checkpoint(self, checkpoint) -> None:
        for ig_keys in self.ignored_weights_prefix:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    # Log.info(f"Remove key `{ig_keys}' from checkpoint.")
                    checkpoint["state_dict"].pop(k)

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer] Loading ckpt: {ckpt_path}")

        ckpt = torch.load(ckpt_path, "cpu")
        state_dict = ckpt["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            ignored_when_saving = any(
                k.startswith(ig_keys) for ig_keys in self.ignored_weights_prefix
            )
            if not ignored_when_saving:
                real_missing.append(k)

        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")
        return ckpt


unimfm = builds(
    UNIMFM_demo,
    pipeline="${pipeline}",
    model_cfg="${model_cfg}",
)
MainStore.store(name="unimfm_demo", node=unimfm, group="model/gvhmr")
