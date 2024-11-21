import torch
import pytorch_lightning as pl
import numpy as np
import wandb
import os
import shutil
import torch.distributed as dist
from pathlib import Path
from einops import einsum, rearrange

from hmr4d.configs import MainStore, builds
from hmr4d.utils.pylogger import Log
from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.eval.eval_utils import compute_camcoord_metrics, as_np_array
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.cv2_utils import cv2, draw_bbx_xys_on_image_batch, draw_coco17_skeleton_batch
from hmr4d.utils.video_io_utils import read_video_np, get_video_lwh, save_video
from hmr4d.utils.geo_transform import apply_T_on_points
from hmr4d.utils.seq_utils import rearrange_by_mask
from hmr4d.model.gvhmr.utils.mv2d_utils import draw_motion_2d, coco_joint_parents, draw_mv_imgs, images_to_video


class Vis2D(pl.Callback):
    def __init__(self):
        super().__init__()
        # vid->result
        self.metric_aggregator = {
            # "pa_mpjpe": {},
            # "mpjpe": {},
            # "pve": {},
            # "accel": {},
        }

        # SMPLX and SMPL
        self.smplx = make_smplx("supermotion_EVAL3DPW")
        self.smpl = {"male": make_smplx("smpl", gender="male"), "female": make_smplx("smpl", gender="female")}
        self.J_regressor = torch.load("hmr4d/utils/body_model/smpl_3dpw14_J_regressor_sparse.pt").to_dense()
        self.J_regressor24 = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")
        self.faces_smplx = self.smplx.faces
        self.faces_smpl = self.smpl["male"].faces
        self.img_h = self.img_w = 256

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end
        
    def log_2d(self, trainer, pl_module, batch_idx, results):
        mv2d = results['mv2d']
        input_view_id = results.get('input_view_id', None)
        for ind in range(mv2d.shape[0]):
            vid_file = f'out/video/vis_2d/b{batch_idx:03d}_i{ind:02d}.mp4'
            os.makedirs(os.path.dirname(vid_file), exist_ok=True)
            frame_dir = os.path.splitext(vid_file)[0] + '_frames'
            os.makedirs(frame_dir, exist_ok=True)
            # create blank image
            for t in range(mv2d.shape[1]):
                mv_imgs = []
                if 'obs' in results:
                    obs_img = draw_mv_imgs(results['obs'][ind, t], coco_joint_parents, self.img_w, self.img_h, add_coco_root=True, unnormalize=True, highlight_view=input_view_id)
                    mv_imgs.append(obs_img)
                if 'mv2d_proj' in results:
                    mv2d_proj = draw_mv_imgs(results['mv2d_proj'][ind, t], coco_joint_parents, self.img_w, self.img_h, add_coco_root=True, unnormalize=True)
                    mv_imgs.append(mv2d_proj)
                mv2d_img = draw_mv_imgs(mv2d[ind, t].cpu(), coco_joint_parents, self.img_w, self.img_h, add_coco_root=True, unnormalize=True)
                mv_imgs.append(mv2d_img)
                if 'mv2d_shuffle' in results:
                    mv2d_shuffle_img = draw_mv_imgs(results['mv2d_shuffle'][ind, t], coco_joint_parents, self.img_w, self.img_h, add_coco_root=True, unnormalize=True, highlight_view=input_view_id)
                    mv_imgs.append(mv2d_shuffle_img)
                if 'mv2d_sv' in results:
                    mv2d_sv_img = draw_mv_imgs(results['mv2d_sv'][ind, t], coco_joint_parents, self.img_w, self.img_h, add_coco_root=True, unnormalize=True, highlight_view=input_view_id)
                    mv_imgs.append(mv2d_sv_img)
                mv_imgs = np.concatenate(mv_imgs, axis=0)

                if trainer.global_rank == 0:
                    cv2.imwrite(f'{frame_dir}/{t:06d}.jpg', mv_imgs[..., ::-1])

            if trainer.global_rank == 0:
                images_to_video(frame_dir, vid_file, fps=30, verbose=False)
                shutil.rmtree(frame_dir, ignore_errors=True)
            
                if isinstance(wandb.run, wandb.sdk.wandb_run.Run):
                    pl_module.logger.log_metrics({f'b{batch_idx:03d}_i{ind:02d}': wandb.Video(vid_file)})
                    # wandb.log({f'vis_2d/{ind:04d}': wandb.Video(vid_file)}, step=trainer.global_step if trainer is not None else 0)
        return

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if 'is_2d' not in batch or not batch['is_2d'][0]:
            return
        
        if '2d_model_output' in outputs:
            input_view_id = outputs['2d_model_output']['input_view_id']
            obs = outputs["batch"]["obs"]
            orig_obs = outputs["batch"]["orig_obs"]
            obs = torch.stack([obs, torch.zeros_like(obs), torch.zeros_like(obs), torch.zeros_like(obs)], dim=2)
            obs[:, :, input_view_id] = orig_obs
            results = {
                'obs': obs,
                'mv2d': outputs['2d_model_output']['mv2d'],
                'mv2d_shuffle': outputs['2d_model_output']['mv2d_shuffle'],
                'mv2d_sv': outputs['2d_model_output_sv']['mv2d'],
                'input_view_id': input_view_id,
                # 'mv2d_proj': outputs['2d_model_output']['mv2d_proj'],
            }
            self.log_2d(trainer, pl_module, batch_idx, results)
        elif 'diffusion' in outputs:
            results = {
                'mv2d': outputs['diffusion']['kp2d'].unsqueeze(2).repeat(1, 1, 4, 1, 1),
            }
            self.log_2d(trainer, pl_module, batch_idx, results)
            
        
        if dist.is_initialized():
            dist.barrier()

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        return
        """Without logger"""
        local_rank, world_size = trainer.local_rank, trainer.world_size
        monitor_metric = "pa_mpjpe"

        # Reduce metric_aggregator across all processes
        metric_keys = list(self.metric_aggregator.keys())
        with torch.inference_mode(False):  # allow in-place operation of all_gather
            metric_aggregator_gathered = all_gather(self.metric_aggregator)  # list of dict
        for metric_key in metric_keys:
            for d in metric_aggregator_gathered:
                self.metric_aggregator[metric_key].update(d[metric_key])

        if False:  # debug to make sure the all_gather is correct
            print(f"[RANK {local_rank}/{world_size}]: {self.metric_aggregator[monitor_metric].keys()}")

        total = len(self.metric_aggregator[monitor_metric])
        Log.info(f"{total} sequences evaluated in {self.__class__.__name__}")
        if total == 0:
            return

        # print monitored metric per sequence
        mm_per_seq = {k: v.mean() for k, v in self.metric_aggregator[monitor_metric].items()}
        if len(mm_per_seq) > 0:
            sorted_mm_per_seq = sorted(mm_per_seq.items(), key=lambda x: x[1], reverse=True)
            n_worst = 5 if trainer.state.stage == "validate" else len(sorted_mm_per_seq)
            if local_rank == 0:
                Log.info(
                    f"monitored metric {monitor_metric} per sequence\n"
                    + "\n".join([f"{m:5.1f} : {s}" for s, m in sorted_mm_per_seq[:n_worst]])
                    + "\n------"
                )

        # average over all batches
        metrics_avg = {k: np.concatenate(list(v.values())).mean() for k, v in self.metric_aggregator.items()}
        if local_rank == 0:
            Log.info(f"[Metrics] 3DPW:\n" + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items()) + "\n------")

        # save to logger if available
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics({f"val_metric_3DPW/{k}": v}, step=cur_epoch)

        # reset
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}


node_vis2d = builds(Vis2D)
MainStore.store(name="vis_2d", node=node_vis2d, group="callbacks", package="callbacks.vis_2d")
