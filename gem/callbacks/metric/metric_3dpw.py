# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytorch_lightning as pl
import torch
from einops import einsum

from gem.utils.eval_utils import as_np_array, compute_camcoord_metrics
from gem.utils.gather import all_gather
from gem.utils.geo_transform import apply_T_on_points
from gem.utils.pylogger import Log
from gem.utils.smplx_utils import make_smplx

_BODY_MODEL_DIR = "inputs/checkpoints/body_models"


class Metric3DPW(pl.Callback):
    def __init__(self):
        super().__init__()
        # vid->result
        self.metric_aggregator = {
            "pa_mpjpe": {},
            "mpjpe": {},
            "pve": {},
            "accel": {},
        }

        # SMPLX (flat_hand_mean for 3DPW) and SMPL
        self.smplx = make_smplx("supermotion_EVAL3DPW")
        self.smpl = {
            "male": make_smplx("smpl", gender="male"),
            "female": make_smplx("smpl", gender="female"),
        }
        self.J_regressor = torch.load(
            f"{_BODY_MODEL_DIR}/smpl_3dpw14_J_regressor_sparse.pt"
        ).to_dense()
        self.J_regressor24 = torch.load(f"{_BODY_MODEL_DIR}/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load(f"{_BODY_MODEL_DIR}/smplx2smpl_sparse.pt")
        self.faces_smpl = self.smpl["male"].faces

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end
        self.on_test_epoch_start = self.on_validation_epoch_start = self.on_predict_epoch_start

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id != "3DPW":
            return

        # Move to cuda if not
        self.smplx = self.smplx.cuda()
        for g in ["male", "female"]:
            self.smpl[g] = self.smpl[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.J_regressor24 = self.J_regressor24.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        vid = batch["meta"][0]["vid"]
        gender = batch["gender"][0]
        T_w2c = batch["gt_T_w2c"][0]
        mask = batch["mask"]["valid"][0]

        # Groundtruth (cam)
        target_w_params = {k: v[0] for k, v in batch["smpl_params"].items()}
        target_w_output = self.smpl[gender](**target_w_params)
        target_w_verts = target_w_output.vertices
        target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
        target_c_j3d = torch.matmul(self.J_regressor, target_c_verts)

        # + Prediction -> Metric
        smpl_out = self.smplx(**outputs["pred_body_params_incam"])
        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")
        del smpl_out

        # Metric of current sequence
        batch_eval = {
            "pred_j3d": pred_c_j3d,
            "target_j3d": target_c_j3d,
            "pred_verts": pred_c_verts,
            "target_verts": target_c_verts,
        }
        camcoord_metrics = compute_camcoord_metrics(batch_eval, mask=mask, pelvis_idxs=[2, 3])
        for k in camcoord_metrics:
            self.metric_aggregator[k][vid] = as_np_array(camcoord_metrics[k])

    def on_predict_epoch_start(self, trainer, pl_module):
        pass

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        local_rank = trainer.local_rank
        monitor_metric = "pa_mpjpe"

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
                "[Metrics] 3DPW:\n"
                + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        # save to logger if available
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics({f"val_metric_3DPW/{k}": v}, step=cur_epoch)

        # reset
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}
