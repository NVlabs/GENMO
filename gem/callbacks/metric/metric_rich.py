# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytorch_lightning as pl
import torch
from einops import einsum

from gem.utils.eval_utils import (
    as_np_array,
    compute_camcoord_metrics,
    compute_global_metrics,
)
from gem.utils.gather import all_gather
from gem.utils.geo_transform import apply_T_on_points
from gem.utils.pylogger import Log
from gem.utils.smplx_utils import make_smplx

_BODY_MODEL_DIR = "inputs/checkpoints/body_models"


class MetricRICH(pl.Callback):
    """Metric callback for RICH (and RICH-OCC) dataset.

    Computes both camera-coordinate and global metrics.
    """

    def __init__(self, occ=False):
        super().__init__()
        self.occ = occ
        # vid->result
        self.metric_aggregator = {
            "pa_mpjpe": {},
            "mpjpe": {},
            "pve": {},
            "accel": {},
            "wa2_mpjpe": {},
            "waa_mpjpe": {},
            "rte": {},
            "jitter": {},
            "fs": {},
        }

        # SMPL-X for prediction and GT FK
        self.smplx_model = {
            "male": make_smplx("supermotion"),
            "female": make_smplx("supermotion"),
            "neutral": make_smplx("supermotion"),
        }
        self.J_regressor = torch.load(f"{_BODY_MODEL_DIR}/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load(f"{_BODY_MODEL_DIR}/smplx2smpl_sparse.pt")
        self.faces_smpl = make_smplx("smpl").faces

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end
        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if self.occ and dataset_id != "RICH-OCC":
            return
        elif (not self.occ) and dataset_id != "RICH":
            return

        # Move to cuda if not
        for g in ["male", "female", "neutral"]:
            self.smplx_model[g] = self.smplx_model[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        vid = batch["meta"][0]["vid"]
        gender = batch["gender"][0]
        T_w2ay = batch["T_w2ay"][0]
        T_w2c = batch["T_w2c"][0]

        # Groundtruth (world, cam)
        target_w_params = {k: v[0] for k, v in batch["gt_smpl_params"].items()}
        target_w_output = self.smplx_model[gender](**target_w_params)
        target_w_verts = torch.stack(
            [torch.matmul(self.smplx2smpl, v_) for v_ in target_w_output.vertices]
        )
        target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
        target_c_j3d = torch.matmul(self.J_regressor, target_c_verts)
        # ay for global comparison
        target_ay_verts = apply_T_on_points(target_w_verts, T_w2ay)
        target_ay_j3d = torch.matmul(self.J_regressor, target_ay_verts)

        # + Prediction -> Metric
        # 1. cam
        pred_smpl_params_incam = outputs["pred_body_params_incam"]
        smpl_out = self.smplx_model["neutral"](**pred_smpl_params_incam)
        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")

        # 2. ay (global)
        pred_smpl_params_global = outputs["pred_body_params_global"]
        smpl_out = self.smplx_model["neutral"](**pred_smpl_params_global)
        pred_ay_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        pred_ay_j3d = einsum(self.J_regressor, pred_ay_verts, "j v, l v i -> l j i")

        # Camera-coord metrics
        batch_eval = {
            "pred_j3d": pred_c_j3d,
            "target_j3d": target_c_j3d,
            "pred_verts": pred_c_verts,
            "target_verts": target_c_verts,
        }
        camcoord_metrics = compute_camcoord_metrics(batch_eval)
        for k in camcoord_metrics:
            self.metric_aggregator[k][vid] = as_np_array(camcoord_metrics[k])

        # Global metrics
        batch_eval = {
            "pred_j3d_glob": pred_ay_j3d,
            "target_j3d_glob": target_ay_j3d,
            "pred_verts_glob": pred_ay_verts,
            "target_verts_glob": target_ay_verts,
        }
        global_metrics = compute_global_metrics(batch_eval)
        for k in global_metrics:
            self.metric_aggregator[k][vid] = as_np_array(global_metrics[k])

    def on_predict_epoch_start(self, trainer, pl_module):
        pass

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        local_rank = trainer.local_rank
        monitor_metric = "mpjpe"

        # Reduce metric_aggregator across all processes
        metric_keys = list(self.metric_aggregator.keys())
        with torch.inference_mode(False):
            metric_aggregator_gathered = all_gather(self.metric_aggregator)
        for metric_key in metric_keys:
            for d in metric_aggregator_gathered:
                self.metric_aggregator[metric_key].update(d[metric_key])

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
        metrics_avg = {
            k: np.concatenate(list(v.values())).mean() for k, v in self.metric_aggregator.items()
        }
        if local_rank == 0:
            Log.info(
                f"[Metrics] RICH{'_OCC' if self.occ else ''}:\n"
                + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        # save to logger if available
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics(
                    {f"val_metric_RICH{'-OCC' if self.occ else ''}/{k}": v},
                    step=cur_epoch,
                )

        # reset
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}
