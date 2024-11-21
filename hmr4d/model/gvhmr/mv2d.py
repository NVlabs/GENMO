from typing import Any, Dict
import numpy as np
from pathlib import Path
import torch
import pytorch_lightning as pl
import os
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from einops import rearrange, einsum
from hmr4d.configs import MainStore, builds

from hmr4d.utils.geo_transform import compute_T_ayfz2ay, apply_T_on_points
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo.augment_noisy_pose import (
    get_wham_aug_kp3d,
    get_visible_mask,
    get_invisible_legs_mask,
    randomly_occlude_lower_half,
    randomly_modify_hands_legs,
)
from hmr4d.utils.geo.hmr_cam import perspective_projection, normalize_kp2d, safely_render_x3d_K, get_bbx_xys

from hmr4d.utils.video_io_utils import save_video
from hmr4d.utils.vis.cv2_utils import draw_bbx_xys_on_image_batch
from hmr4d.utils.geo.flip_utils import flip_smplx_params, avg_smplx_aa
from hmr4d.model.gvhmr.utils.postprocess import pp_static_joint, pp_static_joint_cam, process_ik
from motiondiff.utils.torch_transform import angle_axis_to_rotation_matrix, make_transform, transform_trans, inverse_transform
from motiondiff.utils.tools import Timer
from motiondiff.models.model_util import create_gaussian_diffusion
from motiondiff.models.common.cfg_sampler import ClassifierFreeSampleModel
from motiondiff.diffusion.resample import create_named_schedule_sampler
from .utils.mv2d_utils import draw_motion_2d, coco_joint_parents




class MV2D(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        optimizer=None,
        scheduler_cfg=None,
        model_cfg=None,
        ignored_weights_prefix=["smplx", "pipeline.endecoder"],
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        self.model_cfg = model_cfg
        self.num_views = model_cfg.num_views
        self.scheduler_cfg = scheduler_cfg
        self.validate_2d_in_3d = model_cfg.get("validate_2d_in_3d", False)
        self.enable_test_time_opt = model_cfg.get("enable_test_time_opt", False)
        self.train_3d_modes = model_cfg.get("train_3d_modes", ["regression"])
        self.train_2d_modes = model_cfg.get("train_2d_modes", ["regression"])
        self.infer_mode = model_cfg.get("infer_mode", "regression")
        if isinstance(self.train_3d_modes, str):
            self.train_3d_modes = [self.train_3d_modes]
        if isinstance(self.train_2d_modes, str):
            self.train_2d_modes = [self.train_2d_modes]

        # Options
        self.ignored_weights_prefix = ignored_weights_prefix

        # The test step is the same as validation
        self.test_step = self.predict_step = self.validation_step
        self.timing = os.environ.get("DEBUG_TIMING", "FALSE") == "TRUE"

        # SMPLX
        self.smplx = make_smplx("supermotion_v437coco17")
        self.init_diffusion()
        
    def init_diffusion(self):
        self.train_diffusion = create_gaussian_diffusion(self.model_cfg.diffusion, training=True)
        self.test_diffusion = create_gaussian_diffusion(self.model_cfg.diffusion, training=False)
        self.schedule_sampler = create_named_schedule_sampler(self.model_cfg.diffusion.schedule_sampler_type, self.train_diffusion)
        return
        
    def obtain_mv2d(self, batch, gt_j3d):
        gt_j3d = gt_j3d.float()
        h36m_index = torch.tensor([meta["data_name"] == "h36m" for meta in batch["meta"]])
        rot_angle_base = torch.zeros(gt_j3d.shape[0], 3).float()
        rot_angle_base[h36m_index, 2] = 0.5 * np.pi
        rot_angle_base[~h36m_index, 1] = 0.5 * np.pi
        
        with torch.autocast(device_type='cuda', enabled=False):
            mv2d = []
            T_c2w = inverse_transform(batch["T_w2c"])
            gt_j3d_world = apply_T_on_points(gt_j3d, T_c2w)
            gt_root_world = gt_j3d_world[:, :, 0]
            T_c2w[..., :3, -1] -= gt_root_world
            for i in range(self.num_views):
                y_rot = angle_axis_to_rotation_matrix(rot_angle_base * i).unsqueeze(1)
                T_y_rot = make_transform(y_rot, torch.zeros(3).to(T_c2w))
                T_c2w_new = T_y_rot @ T_c2w
                T_c2w_new[..., :3, -1] += gt_root_world
                T_w2c_new = inverse_transform(T_c2w_new)
                gt_j3d_local = apply_T_on_points(gt_j3d_world, T_w2c_new) 
                new_j2d = perspective_projection(gt_j3d_local, batch["K_fullimg"])
                mv2d.append(new_j2d)
        mv2d = torch.stack(mv2d, dim=2)
        # mv2d = torch.cat([mv2d, (mv2d[..., [11], :] + mv2d[..., [12], :]) * 0.5], dim=-2)
        # vis_id = 0
        # draw_motion_2d(mv2d[vis_id].cpu(), f"out/debug_vis/{batch['meta'][vis_id]['data_name']}.mp4", coco_joint_parents, 1000, 1000, fps=30)
        return mv2d
    
    def training_step(self, batch, batch_idx):
        if not ('3d' in batch or '2d' in batch):
            if 'is_2d' in batch and batch['is_2d'][0]:
                batch = {'2d': batch}
            else:
                batch = {'3d': batch}
                
        def append_mode_to_loss(outputs, mode):
            for k in list(outputs.keys()):
                if "_loss" in k or k in {"loss", "loss_2d"}:
                    outputs[f'Loss_{mode}/{k}'] = outputs.pop(k)
            return outputs
            
        outputs = {'loss': 0}
        if '3d' in batch:
            with Timer("train_3d_step", enabled=self.timing):
                for mode in self.train_3d_modes:
                    outputs_3d = self.train_3d_step(batch['3d'], batch_idx, mode=mode)
                    outputs['loss'] += outputs_3d['loss']
                    append_mode_to_loss(outputs_3d, mode)
                    outputs.update(outputs_3d)
                    
        
        start_2d_training_steps = self.model_cfg.get("start_2d_training_steps", 0)
        if '2d' in batch and self.trainer.global_step >= start_2d_training_steps:
            with Timer("train_2d_step", enabled=self.timing):
                for mode in self.train_2d_modes:
                    outputs_2d = self.train_2d_step(batch['2d'], batch_idx, mode=mode)
                    outputs['loss'] += outputs_2d['loss_2d']
                    append_mode_to_loss(outputs_2d, mode)
                    outputs.update(outputs_2d)

        # Log
        log_kwargs = {
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "sync_dist": True,
            "batch_size": outputs["batch_size"],
        }
        self.log("train/loss", outputs["loss"], **log_kwargs)
        for k, v in outputs.items():
            if "_loss" in k:
                self.log(f"{k}", v, **log_kwargs)
                
        return outputs

    def train_3d_step(self, batch, batch_idx, mode):
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
            batch["bbx_xys"][~mask_bbx_xys] = bbx_xys[~mask_bbx_xys]
        if False:  # visualize bbx_xys from an iPhone view
            render_w, render_h = 120, 160  # iphone main-lens 24mm 3:4
            ratio = render_w / 1528
            offset = torch.tensor([764 - 500, 1019 - 500]).to(i_x2d)
            i_x2d_render = (i_x2d + offset).clone()
            i_x2d_render = (i_x2d_render * ratio).long().clone()
            torch.clamp_(i_x2d_render[..., 0], 0, render_w - 1)
            torch.clamp_(i_x2d_render[..., 1], 0, render_h - 1)
            bbx_xys_render = bbx_xys.clone()
            bbx_xys_render[..., :2] += offset
            bbx_xys_render *= ratio

            output_dir = Path("outputs/simulated_bbx_xys")
            output_dir.mkdir(parents=True, exist_ok=True)
            video_list = []
            for bid in range(B):
                images = torch.zeros(F, render_h, render_w, 3, device=i_x2d.device)
                for fid in range(F):
                    images[fid, i_x2d_render[bid, fid, :, 1], i_x2d_render[bid, fid, :, 0]] = 255

                images = draw_bbx_xys_on_image_batch(bbx_xys_render[bid].cpu().numpy(), images.cpu().numpy())
                images = np.stack(images).astype("uint8")  # (L, H, W, 3)
                images[:, 0, :] = np.array([255, 255, 255])
                images[:, -1, :] = np.array([255, 255, 255])
                images[:, :, 0] = np.array([255, 255, 255])
                images[:, :, -1] = np.array([255, 255, 255])
                video_list.append(images)

            # stack videos
            video_output = []
            for i in range(0, len(video_list), 4):
                if i + 4 <= len(video_list):
                    video_output.append(np.concatenate(video_list[i : i + 4], axis=2))
            video_output = np.concatenate(video_output, axis=1)
            save_video(video_output, output_dir / f"{batch_idx}.mp4", fps=30, quality=5)

        # noisy_j3d -> project to i_j2d -> compute a bbx -> normalized kp2d [-1, 1]
        
        noisy_j3d = gt_j3d + get_wham_aug_kp3d(gt_j3d.shape[:2])
        if True:
            noisy_j3d = randomly_modify_hands_legs(noisy_j3d)
        obs_i_j2d = perspective_projection(noisy_j3d, batch["K_fullimg"])  # (B, L, J, 2)
        j2d_visible_mask = get_visible_mask(gt_j3d.shape[:2]).cuda()  # (B, L, J)
        j2d_visible_mask[noisy_j3d[..., 2] < 0.3] = False  # Set close-to-image-plane points as invisible
        if True:  # Set both legs as invisible for a period
            legs_invisible_mask = get_invisible_legs_mask(gt_j3d.shape[:2]).cuda()  # (B, L, J)
            j2d_visible_mask[legs_invisible_mask] = False
        obs_kp2d = torch.cat([obs_i_j2d, j2d_visible_mask[:, :, :, None].float()], dim=-1)  # (B, L, J, 3)
        obs = normalize_kp2d(obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
        obs[~j2d_visible_mask] = 0  # if not visible, set to (0,0,0)
        batch["obs"] = obs
        batch["j2d_visible_mask"] = j2d_visible_mask

        mv2d = self.obtain_mv2d(batch, gt_j3d)
        T_w2c = batch["T_w2c"]
        mv2d = torch.cat([mv2d, torch.ones_like(mv2d[..., :1])], dim=-1)
        mv2d_bbox = []
        mv2d_norm = []
        for i in range(self.num_views):
            bbox = get_bbx_xys(mv2d[:, :, i], do_augment=False)
            mv2d_bbox.append(bbox)
            mv2d_norm.append(normalize_kp2d(mv2d[:, :, i], bbox))
        batch['mv2d_bbox'] = mv2d_bbox = torch.stack(mv2d_bbox, dim=2)
        batch['mv2d_norm'] = mv2d_norm = torch.stack(mv2d_norm, dim=2)
        # cam parameters
        batch['cam_elevations'] = torch.arcsin(-T_w2c[:, :, 2, 1])
        cam_tilt = np.pi - torch.atan2(T_w2c[:, :, 0, 1], T_w2c[:, :, 1, 1])
        cam_tilt[cam_tilt > np.pi] -= 2 * np.pi
        cam_tilt[cam_tilt < -np.pi] += 2 * np.pi
        batch['cam_tilt'] = cam_tilt
        batch['cam_param_valid'] = torch.tensor([meta["data_name"] != "h36m" for meta in batch["meta"]])
        
        # vis_ind = 0
        # mv2d_norm = torch.cat([mv2d_norm, (mv2d_norm[..., [11], :] + mv2d_norm[..., [12], :]) * 0.5], dim=-2)
        # draw_motion_2d((mv2d_norm[vis_ind, ..., :2].cpu() + 1.0) * 500, f"out/debug_vis/{batch['meta'][vis_ind]['data_name']}_new.mp4", coco_joint_parents, 1000, 1000, fps=30)
        # mv2d_norm[:, :, 0, :17] = obs
        # mv2d_norm[:, :, :, [17]] = (mv2d_norm[..., [11], :] + mv2d_norm[..., [12], :]) * 0.5
        # draw_motion_2d((mv2d_norm[vis_ind, ..., :2].cpu() + 1.0) * 500, f"out/debug_vis/{batch['meta'][vis_ind]['data_name']}_obs.mp4", coco_joint_parents, 1000, 1000, fps=30)
        
        if True:  # Use some detected vitpose (presave data)
            prob = 0.5
            mask_real_vitpose = (torch.rand(B).to(obs_kp2d) < prob) * batch["mask"]["vitpose"]
            batch["obs"][mask_real_vitpose] = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])[mask_real_vitpose]

        # Set untrusted frames to False
        batch["obs"][~batch["mask"]["valid"]] = 0
        batch["mv2d_bbox"][~batch["mask"]["valid"]] = 0
        batch["mv2d_norm"][~batch["mask"]["valid"]] = 0

        if False:  # wis3d
            wis3d = make_wis3d(name="debug-aug-kp3d")
            add_motion_as_lines(gt_j3d[0], wis3d, name="gt_j3d", skeleton_type="coco17")
            add_motion_as_lines(noisy_j3d[0], wis3d, name="noisy_j3d", skeleton_type="coco17")

        # f_imgseq: apply random aug on offline extracted features
        # f_imgseq = batch["f_imgseq"] + torch.randn_like(batch["f_imgseq"]) * 0.1
        # f_imgseq[~batch["mask"]["f_imgseq"]] = 0
        # batch["f_imgseq"] = f_imgseq.clone()

        # Forward and get loss
        outputs = self.pipeline.forward(batch, train=True, global_step=self.trainer.global_step)
        outputs['batch_size'] = B

        return outputs
    
    def train_2d_step(self, batch, batch_idx, mode):
        B = batch["obs_kp2d"].shape[0]
        obs_kp2d = batch['obs_kp2d'].squeeze(2)
        conf = batch['conf']
        batch["bbx_xys"] = get_bbx_xys(obs_kp2d, do_augment=False)
        
        orig_obs_kp2d = obs_kp2d.clone()
        orig_obs_kp2d = torch.cat([orig_obs_kp2d, conf[:, :, :, None].float()], dim=-1)  # (B, L, J, 3)
        batch["orig_obs"] = normalize_kp2d(orig_obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
        batch["orig_obs"][~batch["mask"]] = 0
        
        noisy_2d_obs = self.model_cfg.get("noisy_2d_obs", False)
        if noisy_2d_obs:
            aug = get_wham_aug_kp3d(obs_kp2d.shape[:2])[..., :2]
            f = torch.tensor([1024., 1024.]).to(aug) / 4.
            aug *= f * self.model_cfg.kp2d_noise_scale
            obs_kp2d += aug
            obs_kp2d = randomly_modify_hands_legs(obs_kp2d)
            j2d_visible_mask = get_visible_mask(obs_kp2d.shape[:2]).cuda()  # (B, L, J)
            legs_invisible_mask = get_invisible_legs_mask(obs_kp2d.shape[:2]).cuda()  # (B, L, J)
            j2d_visible_mask[legs_invisible_mask] = False
            j2d_visible_mask *= (conf > 0.5)
        else:
            j2d_visible_mask = conf > 0.5
        if 'mask_cfg' in self.model_cfg:
            mask = self.generate_mask(self.model_cfg.mask_cfg, j2d_visible_mask, batch["length"])
            j2d_visible_mask = j2d_visible_mask & mask
        
        if mode == 'sv-diffusion':  #single view diffusion
            diffusion = self.train_diffusion
            t, t_weights = self.schedule_sampler.sample(B, obs_kp2d.device)
            scaled_t = diffusion._scale_timesteps(t)
            x_start = batch["orig_obs"][..., :2]
            noise = torch.randn_like(x_start)
            x_t = diffusion.q_sample(x_start, t, noise=noise)
            x_t = torch.cat([x_t, batch["orig_obs"][..., [-1]]], dim=-1)
            batch['obs_x_t'] = x_t
            batch["obs_x_t"][~batch["mask"]] = 0
            batch['scaled_t'] = scaled_t
            outputs = self.pipeline.forward_singleview_diffusion(batch, train=True, global_step=self.trainer.global_step)
            # vis_ind = 0
            # mv2d_norm = batch["orig_obs"].unsqueeze(2)
            # mv2d_norm = torch.cat([mv2d_norm, (mv2d_norm[..., [11], :] + mv2d_norm[..., [12], :]) * 0.5], dim=-2)
            # draw_motion_2d((mv2d_norm[vis_ind, ..., :2].cpu() + 1.0) * 500, f"out/debug_vis/2d_test_obs.mp4", coco_joint_parents, 1000, 1000, fps=30, mask=mv2d_norm[vis_ind, ..., 2].cpu())
        elif mode in {'regression'}:
            obs_kp2d = torch.cat([obs_kp2d, j2d_visible_mask[:, :, :, None].float()], dim=-1)  # (B, L, J, 3)
            obs = normalize_kp2d(obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
            obs[~batch["mask"]] = 0
            batch["obs"] = obs
            
            # vis_ind = 0
            # mv2d_norm = obs.unsqueeze(2)
            # mv2d_norm = torch.cat([mv2d_norm, (mv2d_norm[..., [11], :] + mv2d_norm[..., [12], :]) * 0.5], dim=-2)
            # draw_motion_2d((mv2d_norm[vis_ind, ..., :2].cpu() + 1.0) * 500, f"out/debug_vis/2d_test_noisy.mp4", coco_joint_parents, 1000, 1000, fps=30, mask=mv2d_norm[vis_ind, ..., 2].cpu())
            # mv2d_norm = batch["orig_obs"].unsqueeze(2)
            # mv2d_norm = torch.cat([mv2d_norm, (mv2d_norm[..., [11], :] + mv2d_norm[..., [12], :]) * 0.5], dim=-2)
            # draw_motion_2d((mv2d_norm[vis_ind, ..., :2].cpu() + 1.0) * 500, f"out/debug_vis/2d_test_obs.mp4", coco_joint_parents, 1000, 1000, fps=30, mask=mv2d_norm[vis_ind, ..., 2].cpu())

            # Forward and get loss
            outputs = self.pipeline.forward_2d(batch, train=True, global_step=self.trainer.global_step)

        return outputs
    
    def generate_mask(self, mask_cfg, orig_mask, length):
        _cfg = mask_cfg
        mask = torch.ones_like(orig_mask)
        drop_prob = _cfg.get('drop_prob', 0.0)
        if drop_prob <= 0:
            return mask
        max_num_drops = _cfg.get('max_num_drops', 1)
        min_drop_nframes = _cfg.get('min_drop_nframes', 1)
        max_drop_nframes = _cfg.get('max_drop_nframes', 30)
        for i in range(orig_mask.shape[0]):
            mlen = length[i].item()
            if np.random.rand() < drop_prob:
                num_drops = np.random.randint(1, max_num_drops + 1)
                for _ in range(num_drops):
                    drop_len = np.random.randint(min_drop_nframes, min(max_drop_nframes, mlen) + 1)
                    drop_start = np.random.randint(0, max(mlen - drop_len, 1))
                    mask[i, drop_start:drop_start+drop_len] = False
                    # print(f"Drop {i} {drop_start} {drop_len}")
        return mask
    
    def infer_diffusion(self, batch):
        obs_shape = batch["obs"].shape
        
        cond = {
            "length": batch["length"],
            "reshape": True
        }
        
        diffusion = self.test_diffusion
        diff_sampler = self.model_cfg.diffusion.get("sampler", "ddim")
        if diff_sampler == "ddim":
            sample_fn = diffusion.ddim_sample_loop
            kwargs = {"eta": self.model_cfg.diffusion.ddim_eta}
        else:
            sample_fn = diffusion.p_sample_loop
            kwargs = {}
        samples = sample_fn(
            self.pipeline.denoiser3d.get_denoiser(),
            (obs_shape[0], obs_shape[2]*2, 1, obs_shape[1]),
            clip_denoised=False,
            model_kwargs=cond,
            noise=None,
            progress=True,
            device=batch["obs"].device,
            **kwargs,
        )
        samples = samples.reshape(samples.shape[0], -1, 2, samples.shape[-1]).permute(0, 3, 1, 2)
        outputs = {
            'diffusion': {
                'kp2d': samples,
            }
        }
        return outputs


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if 'is_2d' in batch and batch['is_2d'][0]:
            return self.validation_2d(batch, batch_idx, dataloader_idx)
        else:
            return self.validation_3d(batch, batch_idx, dataloader_idx)
        
    def validation_2d(self, batch, batch_idx, dataloader_idx=0):
        B = batch["obs_kp2d"].shape[0]
        obs_kp2d = batch['obs_kp2d'].squeeze(2)
        conf = batch['conf']
        batch["bbx_xys"] = get_bbx_xys(obs_kp2d, do_augment=False)
        
        orig_obs_kp2d = obs_kp2d.clone()
        orig_obs_kp2d = torch.cat([orig_obs_kp2d, conf[:, :, :, None].float()], dim=-1)  # (B, L, J, 3)
        batch["orig_obs"] = normalize_kp2d(orig_obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
        batch["orig_obs"][~batch["mask"]] = 0
        
        noisy_2d_obs = self.model_cfg.get("test_with_noisy_2d_obs", False)
        test_2d_with_mask = self.model_cfg.get("test_2d_with_mask", True)
        if noisy_2d_obs:
            aug = get_wham_aug_kp3d(obs_kp2d.shape[:2])[..., :2]
            f = torch.tensor([1024., 1024.]).to(aug) / 4.
            aug *= f * self.model_cfg.kp2d_noise_scale
            obs_kp2d += aug
            obs_kp2d = randomly_modify_hands_legs(obs_kp2d)
            j2d_visible_mask = get_visible_mask(obs_kp2d.shape[:2]).cuda()  # (B, L, J)
            legs_invisible_mask = get_invisible_legs_mask(obs_kp2d.shape[:2]).cuda()  # (B, L, J)
            j2d_visible_mask[legs_invisible_mask] = False
            j2d_visible_mask *= (conf > 0.5)
        else:
            j2d_visible_mask = conf > 0.5
        if test_2d_with_mask and 'mask_cfg' in self.model_cfg:
            mask = self.generate_mask(self.model_cfg.mask_cfg, j2d_visible_mask, batch["length"])
            j2d_visible_mask = j2d_visible_mask & mask
        obs_kp2d = torch.cat([obs_kp2d, j2d_visible_mask[:, :, :, None].float()], dim=-1)  # (B, L, J, 3)
        obs = normalize_kp2d(obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
        obs[~batch["mask"]] = 0
        batch["obs"] = obs
        
        # vis_ind = 0
        # mv2d_norm = obs.unsqueeze(2)
        # mv2d_norm = torch.cat([mv2d_norm, (mv2d_norm[..., [11], :] + mv2d_norm[..., [12], :]) * 0.5], dim=-2)
        # draw_motion_2d((mv2d_norm[vis_ind, ..., :2].cpu() + 1.0) * 500, f"out/debug_vis/2d_test_noisy.mp4", coco_joint_parents, 1000, 1000, fps=30, mask=mv2d_norm[vis_ind, ..., 2].cpu())
        # mv2d_norm = batch["orig_obs"].unsqueeze(2)
        # mv2d_norm = torch.cat([mv2d_norm, (mv2d_norm[..., [11], :] + mv2d_norm[..., [12], :]) * 0.5], dim=-2)
        # draw_motion_2d((mv2d_norm[vis_ind, ..., :2].cpu() + 1.0) * 500, f"out/debug_vis/2d_test_obs.mp4", coco_joint_parents, 1000, 1000, fps=30, mask=mv2d_norm[vis_ind, ..., 2].cpu())

        # Forward and get loss
        if self.infer_mode == 'regression':
            outputs = self.pipeline.forward_2d(batch, train=False, global_step=self.trainer.global_step)
        else:
            outputs = self.infer_diffusion(batch)
        outputs["batch"] = batch
        outputs['vis_2d'] = self.model_cfg.get("vis_2d", False)
        return outputs
        
    def validation_3d(self, batch, batch_idx, dataloader_idx=0):
        # Options & Check
        do_postproc = self.trainer.state.stage == "test"  # Only apply postproc in test
        do_flip_test = "flip_test" in batch
        do_postproc_not_flip_test = do_postproc and not do_flip_test  # later pp when flip_test
        assert batch["B"] == 1, "Only support batch size 1 in evalution."

        # ROPE inference
        obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
        if "mask" in batch:
            obs[0, ~batch["mask"][0]] = 0

        if self.validate_2d_in_3d:
            batch_2d = {
                'obs_kp2d': batch['kp2d'][..., :2],
                'conf': batch['kp2d'][..., 2],
                'mask': batch['mask'],
                'length': batch['length']
            }
            
            if self.enable_test_time_opt:
                init_state_dict = {k: v.detach().clone() for k, v in self.pipeline.state_dict().items()}
                with torch.enable_grad():
                    self.train()
                    optimizer = torch.optim.AdamW(params=self.pipeline.parameters(), lr=1e-5)
                    for _ in range(50):
                        outputs_2d = self.validation_2d(batch_2d, batch_idx, dataloader_idx)
                        loss = outputs_2d['loss_2d']
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        print(loss)
                self.eval()
            else:
                outputs_2d = self.validation_2d(batch_2d, batch_idx, dataloader_idx)
            
        batch_ = {
            "length": batch["length"],
            "obs": obs,
            "bbx_xys": batch["bbx_xys"],
            "K_fullimg": batch["K_fullimg"],
            "cam_angvel": batch["cam_angvel"],
            "f_imgseq": batch["f_imgseq"],
        }
        outputs = self.pipeline.forward(batch_, train=False, postproc=do_postproc_not_flip_test, global_step=self.trainer.global_step)
        outputs["pred_smpl_params_global"] = {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()}
        outputs["pred_smpl_params_incam"] = {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()}

        if do_flip_test:
            flip_test = batch["flip_test"]
            obs = normalize_kp2d(flip_test["kp2d"], flip_test["bbx_xys"])
            if "mask" in batch:
                obs[0, ~batch["mask"][0]] = 0

            batch_ = {
                "length": batch["length"],
                "obs": obs,
                "bbx_xys": flip_test["bbx_xys"],
                "K_fullimg": batch["K_fullimg"],
                "cam_angvel": flip_test["cam_angvel"],
                "f_imgseq": flip_test["f_imgseq"],
            }
            flipped_outputs = self.pipeline.forward(batch_, train=False, global_step=self.trainer.global_step)

            # First update incam results
            flipped_outputs["pred_smpl_params_incam"] = {
                k: v[0] for k, v in flipped_outputs["pred_smpl_params_incam"].items()
            }
            smpl_params1 = outputs["pred_smpl_params_incam"]
            smpl_params2 = flip_smplx_params(flipped_outputs["pred_smpl_params_incam"])

            smpl_params_avg = smpl_params1.copy()
            smpl_params_avg["betas"] = (smpl_params1["betas"] + smpl_params2["betas"]) / 2
            smpl_params_avg["body_pose"] = avg_smplx_aa(smpl_params1["body_pose"], smpl_params2["body_pose"])
            smpl_params_avg["global_orient"] = avg_smplx_aa(
                smpl_params1["global_orient"], smpl_params2["global_orient"]
            )
            outputs["pred_smpl_params_incam"] = smpl_params_avg

            # Then update global results
            outputs["pred_smpl_params_global"]["betas"] = smpl_params_avg["betas"]
            outputs["pred_smpl_params_global"]["body_pose"] = smpl_params_avg["body_pose"]

            # Finally, apply postprocess
            if do_postproc:
                # temporarily recover the original batch-dim
                outputs["pred_smpl_params_global"] = {k: v[None] for k, v in outputs["pred_smpl_params_global"].items()}
                outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(outputs, self.pipeline.endecoder)
                body_pose = process_ik(outputs, self.pipeline.endecoder)
                outputs["pred_smpl_params_global"] = {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()}

                outputs["pred_smpl_params_global"]["body_pose"] = body_pose[0]
                # outputs["pred_smpl_params_incam"]["body_pose"] = body_pose[0]
        
        if self.validate_2d_in_3d:
            outputs["outputs_2d"] = outputs_2d
            if self.enable_test_time_opt:
                self.pipeline.load_state_dict(init_state_dict)

        if False:  # wis3d
            wis3d = make_wis3d(name="debug-rich-cap")
            smplx_model = make_smplx("rich-smplx", gender="neutral").cuda()
            gender = batch["gender"][0]
            T_w2ay = batch["T_w2ay"][0]

            # Prediction
            # add_motion_as_lines(outputs_window["pred_ayfz_motion"][bid], wis3d, name="pred_ayfz_motion")

            smplx_out = smplx_model(**pred_smpl_params_global)
            for i in range(len(smplx_out.vertices)):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(smplx_out.vertices[i], smplx_model.bm.faces, name=f"pred-smplx-global")

            # GT (w)
            smplx_models = {
                "male": make_smplx("rich-smplx", gender="male").cuda(),
                "female": make_smplx("rich-smplx", gender="female").cuda(),
            }
            gt_smpl_params = {k: v[0, windows[0]] for k, v in batch["gt_smpl_params"].items()}
            gt_smplx_out = smplx_models[gender](**gt_smpl_params)

            # GT (ayfz)
            smplx_verts_ay = apply_T_on_points(gt_smplx_out.vertices, T_w2ay)
            smplx_joints_ay = apply_T_on_points(gt_smplx_out.joints, T_w2ay)
            T_ay2ayfz = compute_T_ayfz2ay(smplx_joints_ay[:1], inverse=True)[0]  # (4, 4)
            smplx_verts_ayfz = apply_T_on_points(smplx_verts_ay, T_ay2ayfz)  # (F, 22, 3)

            for i in range(len(smplx_verts_ayfz)):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(smplx_verts_ayfz[i], smplx_models[gender].bm.faces, name=f"gt-smplx-ayfz")

            breakpoint()

        if False:  # o3d
            prog_keys = [
                "pred_smpl_progress",
                "pred_localjoints_progress",
                "pred_incam_localjoints_progress",
            ]
            for k in prog_keys:
                if k in outputs_window:
                    seq_out = torch.cat(
                        [v[:, :l] for v, l in zip(outputs_window[k], length)], dim=1
                    )  # (B, P, L, J, 3) -> (P, L, J, 3) -> (P, CL, J, 3)
                    outputs[k] = seq_out[None]

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
        scheduler_cfg["scheduler"] = instantiate(scheduler_cfg["scheduler"], optimizer=optimizer)
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
            ignored_when_saving = any(k.startswith(ig_keys) for ig_keys in self.ignored_weights_prefix)
            if not ignored_when_saving:
                real_missing.append(k)

        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")
        return ckpt


mv2d = builds(
    MV2D,
    pipeline="${pipeline}",
    optimizer="${optimizer}",
    scheduler_cfg="${scheduler_cfg}",
    model_cfg="${model_cfg}",
    populate_full_signature=True,  # Adds all the arguments to the signature
)
MainStore.store(name="mv2d", node=mv2d, group="model/gvhmr")
