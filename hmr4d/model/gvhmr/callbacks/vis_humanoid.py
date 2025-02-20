import os
from pathlib import Path

import cv2
import hydra
import imageio
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from pytorch_lightning.utilities import rank_zero_only
from smplx.joint_names import JOINT_NAMES

from hmr4d.configs import MainStore, builds
from hmr4d.model.gvhmr.utils.vis_utils import (
    visualize_intermediate_smpl_scene,
    visualize_intermediate_smplmesh_scene_img,
    visualize_smpl_scene,
)
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import get_writer, read_video_np, save_video


class VisHumanoid(pl.Callback):
    def __init__(
        self,
        vis_every_n_val=1,
        save_dir=None,
    ):
        super().__init__()
        self.vis_every_n_val = vis_every_n_val
        self.num_val = 0
        self.save_dir = save_dir

        # SMPL models
        self.smplx_model = {
            "male": make_smplx("supermotion_smpl24_male"),
            "female": make_smplx("supermotion_smpl24_female"),
            "neutral": make_smplx("supermotion_smpl24"),
        }
        self.smplx = make_smplx("supermotion")
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")
        self.J_regressor = torch.load(
            "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt"
        )
        self.faces_smpl = make_smplx("smpl").faces
        self.faces_smplx = self.smplx_model["neutral"].faces

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = (
            self.on_predict_batch_end
        )
        self.on_test_epoch_end = self.on_validation_epoch_end = (
            self.on_predict_epoch_end
        )
        self.on_test_epoch_start = self.on_validation_epoch_start = (
            self.on_predict_epoch_start
        )

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Process each batch during prediction"""
        mode = batch["meta"][0].get("mode", None)
        if mode != "humanoid":
            return

        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id not in ["humanml3d"]:  # Only process humanoid dataset
            return

        # # Move models to GPU if needed
        # for g in ["male", "female", "neutral"]:
        #     self.smplx_model[g] = self.smplx_model[g].cuda()
        # self.smplx = self.smplx.cuda()
        # self.J_regressor = self.J_regressor.cuda()
        # self.smplx2smpl = self.smplx2smpl.cuda()

        # # Get sequence info
        # seq_id = batch["meta"][0].get("seq_id", "unknown")
        # seq_length = batch["length"][0].item()
        # gender = "neutral"

        # # Process predictions
        # pred_smpl_params_global = outputs["pred_smpl_params_global"]
        # pred_ay_j3d = self.smplx_model["neutral"](**pred_smpl_params_global)

        # # Get ground truth if available
        # target_w_j3d = None
        # if "smpl_params_w" in batch:
        #     target_w_params = {k: v[0] for k, v in batch["smpl_params_w"].items()}
        #     target_w_j3d = self.smplx_model[gender](**target_w_params)
        #     offset = batch["smpl_params_w"]["transl"][0, :, None] - target_w_j3d[:, [0]]
        #     target_w_j3d = target_w_j3d + offset

        # # Visualize
        # if trainer.global_rank == 0 and self.num_val % self.vis_every_n_val == 0:
        #     wandb_dict = visualize_smpl_scene(
        #         f"vis_humanoid_global",
        #         batch_idx,
        #         seq_id,
        #         pred_ay_j3d,
        #         target_w_j3d,
        #         transform_mode="global",
        #     )
        #     self.wandb_html_dict.update(wandb_dict)

    def on_predict_epoch_start(self, trainer, pl_module):
        """Initialize visualization dictionary at start of epoch"""
        self.wandb_html_dict = {}

    def on_predict_epoch_end(self, trainer, pl_module):
        """Log visualizations at end of epoch"""
        self.num_val += 1
        if len(self.wandb_html_dict) > 0:
            pl_module.logger.log_metrics(self.wandb_html_dict)
