import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.configs import MainStore, builds

from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.pylogger import Log

from hmr4d.utils.eval.eval_utils import (
    compute_camcoord_metrics,
    compute_global_metrics,
    compute_camcoord_perjoint_metrics,
    as_np_array,
)
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.smplx_utils import make_smplx
from einops import einsum, rearrange

from motiondiff.models.mdm.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines, get_colors_by_conf
# from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from hmr4d.utils.geo.hmr_cam import estimate_focal_length
from hmr4d.utils.video_io_utils import read_video_np, save_video, get_writer
import imageio
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2
import hydra
import os

from smplx.joint_names import JOINT_NAMES
from hmr4d.utils.net_utils import repeat_to_max_len, gaussian_smooth
from hmr4d.utils.geo.hmr_global import rollout_vel, get_static_joint_mask
from hmr4d.model.gvhmr.utils.vis_utils import visualize_smpl_scene


class VisText(pl.Callback):
    def __init__(self, vis_every_n_val=10, save_feats=False, save_dir=None, endecoder=None):
        super().__init__()
        self.vis_every_n_val = vis_every_n_val
        self.num_val = 0
        self.save_feats = save_feats
        self.save_dir = save_dir
        if endecoder is not None:
            self.endecoder = hydra.utils.instantiate(endecoder).cuda()
        # vid->result

        # SMPL
        self.smplx_model = {
            "male": make_smplx("supermotion_smpl24_male"),
            "female": make_smplx("supermotion_smpl24_female"),
            "neutral": make_smplx("supermotion_smpl24"),
        }
        self.J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")
        self.faces_smpl = make_smplx("smpl").faces
        self.faces_smplx = self.smplx_model["neutral"].faces

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end
        
        # Only validation record the metrics with logger
        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id not in ['humanml3d']:
            return

        # Move to cuda if not
        for g in ["male", "female", "neutral"]:
            self.smplx_model[g] = self.smplx_model[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        text = batch['caption'][0]
        vid = text.replace(' ', '_').replace('.', '_').replace(',', '_')
        seq_length = batch["length"][0].item()
        gender = 'neutral'

        # Groundtruth (world, cam)
        target_w_params = {k: v[0] for k, v in batch["smpl_params_w"].items()}
        target_w_j3d = self.smplx_model[gender](**target_w_params)
        offset = batch["smpl_params_w"]["transl"][0, :, None] - target_w_j3d[:, [0]]
        target_w_j3d = target_w_j3d + offset
        # target_w_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in target_w_output.vertices])
        # target_w_j3d = torch.matmul(self.J_regressor, target_w_verts)

        # 2. ay
        pred_smpl_params_global = outputs["pred_smpl_params_global"]
        pred_ay_j3d = self.smplx_model["neutral"](**pred_smpl_params_global)
        # pred_ay_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        # pred_ay_j3d = einsum(self.J_regressor, pred_ay_verts, "j v, l v i -> l j i")
        
        if self.save_feats:
            encoder_inputs = {
                'smpl_params_w': {k: v.unsqueeze(0) for k, v in outputs["pred_smpl_params_global"].items()},
            }
            feats = self.endecoder.encode_humanml3d(encoder_inputs)
            self.feats_arr.append(feats)
        else:
            # Visualize
            if trainer.global_rank == 0 and self.num_val % self.vis_every_n_val == 0:
                wandb_dict = visualize_smpl_scene('vis_text_global', batch_idx, vid, pred_ay_j3d, target_w_j3d, transform_mode='global')
                self.wandb_html_dict.update(wandb_dict)
        return


    def on_predict_epoch_start(self, trainer, pl_module):
        self.wandb_html_dict = {}
        if self.save_feats:
            self.feats_arr = []
            print(f"start generating text-to-motion features which will be saved at {self.save_dir}")
        

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        self.num_val += 1
        if len(self.wandb_html_dict) > 0:
            pl_module.logger.log_metrics(self.wandb_html_dict)
        if self.save_feats:
            feats_arr = torch.cat(self.feats_arr, dim=0).cpu()
            os.makedirs(self.save_dir, exist_ok=True)
            torch.save(feats_arr, self.save_dir + '/feats.pt')
            print(f"text-to-motion features saved at {self.save_dir}")
