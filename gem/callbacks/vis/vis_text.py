# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytorch_lightning as pl
import torch

from gem.utils.pylogger import Log
from gem.utils.smplx_utils import make_smplx

_BODY_MODEL_DIR = "inputs/checkpoints/body_models"


class VisText(pl.Callback):
    """Visualization callback for text-to-motion datasets (humanml3d, motion-x++2d).

    Logs predicted joint positions during validation/test. Visualization hooks are
    stripped; extend this class to add rendering if needed.
    """

    def __init__(self, vis_every_n_val=1):
        super().__init__()
        self.vis_every_n_val = vis_every_n_val
        self.num_val = 0

        # SMPL models
        self.smplx_model = {
            "male": make_smplx("supermotion_smpl24_male"),
            "female": make_smplx("supermotion_smpl24_female"),
            "neutral": make_smplx("supermotion_smpl24"),
        }
        self.smplx = make_smplx("supermotion")
        self.J_regressor = torch.load(f"{_BODY_MODEL_DIR}/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load(f"{_BODY_MODEL_DIR}/smplx2smpl_sparse.pt")

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end
        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        mode = batch["meta"][0].get("mode", None)
        if mode != "default":
            return
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id not in ["humanml3d", "motion-x++2d"]:
            return

        # Move to cuda if not
        for g in ["male", "female", "neutral"]:
            self.smplx_model[g] = self.smplx_model[g].cuda()
        self.smplx = self.smplx.cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        text = batch["caption"][0]
        seq_length = batch["length"][0].item()

        # Groundtruth (world)
        if dataset_id == "humanml3d":
            target_w_params = {k: v[0] for k, v in batch["smpl_params_w"].items()}
            target_w_j3d = self.smplx_model["neutral"](**target_w_params)
            offset = batch["smpl_params_w"]["transl"][0, :, None] - target_w_j3d[:, [0]]
            target_w_j3d = target_w_j3d + offset
        else:
            target_w_j3d = None

        if trainer.global_rank == 0 and self.num_val % self.vis_every_n_val == 0:
            Log.info(
                f"[VisText] batch {batch_idx} | dataset={dataset_id} | "
                f"text='{text[:60]}' | seq_len={seq_length}"
            )

    def on_predict_epoch_start(self, trainer, pl_module):
        pass

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        self.num_val += 1
