import os

import hydra
import numpy as np
import pytorch_lightning as pl
import scipy
import torch
from aist_plusplus.features.kinetic import extract_kinetic_features
from aist_plusplus.features.manual import extract_manual_features

from genmo.utils.eval_utils import as_np_array, compute_music_metrics
from genmo.utils.gather import all_gather
from genmo.utils.pylogger import Log
from third_party.GVHMR.hmr4d.utils.smplx_utils import make_smplx


class MetricMusic(pl.Callback):
    def __init__(
        self,
        vis_every_n_val=10,
        save_feats=False,
        save_dir=None,
        dataset_part_ind=-1,
        endecoder=None,
        trial_ind=0,
        text_len=120,
    ):
        super().__init__()
        self.vis_every_n_val = vis_every_n_val
        self.num_val = 0
        self.save_feats = save_feats
        self.trial_ind = trial_ind
        self.save_dir = save_dir
        self.text_len = text_len
        self.dataset_part_ind = dataset_part_ind
        if endecoder is not None:
            self.endecoder = hydra.utils.instantiate(endecoder).cuda()
        # vid->result
        self.metric_aggregator = {
            "PFC": {},
            "BAS": {},
            # "FID_k": {},
            # "FID_m": {},
            # "Dist_k": {},
            # "Dist_m": {},
        }
        self.prediction_aggregator = {}

        # SMPL
        self.smplx_model = {
            "male": make_smplx("supermotion_smpl24"),
            "female": make_smplx("supermotion_smpl24"),
            "neutral": make_smplx("supermotion_smpl24"),
        }
        self.smplx = make_smplx("supermotion")
        self.smplx2smpl = torch.load(
            "inputs/checkpoints/body_models/smplx2smpl_sparse.pt"
        )

        self.J_regressor = torch.load(
            "inputs/checkpoints/body_models/smpl_neutral_J_regressor.pt"
        )
        self.smplx2smpl = torch.load(
            "inputs/checkpoints/body_models/smplx2smpl_sparse.pt"
        )
        self.faces_smpl = make_smplx("smpl").faces
        self.faces_smplx = self.smplx_model["neutral"].faces

        # The metrics are calculated similarly for val/test/predict
        self.on_test_batch_end = self.on_validation_batch_end = (
            self.on_predict_batch_end
        )

        # Only validation record the metrics with logger
        self.on_test_epoch_end = self.on_validation_epoch_end = (
            self.on_predict_epoch_end
        )

        # Only validation record the metrics with logger
        self.on_test_epoch_start = self.on_validation_epoch_start = (
            self.on_predict_epoch_start
        )

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1
        data_name = batch["meta"][0]["data_name"]
        vid = batch["meta"][0]["vid"]

        if data_name not in ["aist++"]:
            return

        # Move to cuda if not
        for g in ["male", "female", "neutral"]:
            self.smplx_model[g] = self.smplx_model[g].cuda()
        self.smplx = self.smplx.cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        # print(batch_idx)
        # os.makedirs('out/motions', exist_ok=True)
        # torch.save(outputs, f'out/motions/outputs_{batch_idx}.pt')
        # if 'multi_text_data' in batch['meta'][0]:
        #     multi_text_data = batch['meta'][0]['multi_text_data']
        #     for i in range(len(multi_text_data['vid'])):
        #         print(multi_text_data['vid'][i], multi_text_data['caption'][i])
        music_beats = batch["music_beats"][0].cpu().numpy()

        seq_length = batch["length"][0].item()
        gender = "neutral"

        # Groundtruth (world, cam)
        if data_name == "aist++":
            target_w_params = {k: v[0] for k, v in batch["smpl_params_w"].items()}
            target_w_params["betas"] = torch.zeros_like(target_w_params["betas"])
            target_w_j3d = self.smplx_model[gender](**target_w_params)
            # offset = batch["smpl_params_w"]["transl"][0, :, None] - target_w_j3d[:, [0]]
            # target_w_j3d = target_w_j3d + offset
            # target_w_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in target_w_output.vertices])
            # target_w_j3d = torch.matmul(self.J_regressor, target_w_verts)

        # 2. ay
        pred_smpl_params_global = outputs["pred_smpl_params_global"]
        pred_smpl_params_global["betas"] = torch.zeros_like(
            pred_smpl_params_global["betas"]
        )
        pred_ay_j3d = self.smplx_model["neutral"](**pred_smpl_params_global)
        # pred_smplx = self.smplx(**pred_smpl_params_global)
        # pred_ay_verts = torch.stack(
        #     [torch.matmul(self.smplx2smpl, v_) for v_ in pred_smplx.vertices]
        # )

        # Metrics of current sequence
        batch_eval = {
            "pred_j3d_glob": pred_ay_j3d,
            "target_j3d_glob": target_w_j3d,
            "music_beats": music_beats,
            # "pred_verts_glob": pred_ay_verts,
            # "target_verts_glob": target_w_verts,
        }
        music_metrics = compute_music_metrics(batch_eval)
        for k in music_metrics:
            self.metric_aggregator[k][vid] = as_np_array(music_metrics[k])

        self.prediction_aggregator[vid] = {
            "pred_j3d_glob": pred_ay_j3d.cpu().numpy(),
            "target_j3d_glob": target_w_j3d.cpu().numpy(),
            # "pred_feats_k": pred_feats_k,
            # "pred_feats_m": pred_feats_m,
            # "pred_feats_k_full": pred_feats_k_full,
            # "pred_feats_m_full": pred_feats_m_full,
        }

        self.eval_type = "full"
        if self.eval_type == "chunk":
            stride = 15
            chunk_size = 150
            pred_feats_k = []
            pred_feats_m = []
            for i in range(0, pred_ay_j3d.shape[0], stride):
                pred_j3d_glob_chunk = pred_ay_j3d[i : i + chunk_size].cpu().numpy()
                offset = pred_j3d_glob_chunk[:1, :1, :]
                pred_j3d_glob_chunk = pred_j3d_glob_chunk - offset
                pred_feat_k_chunk = extract_kinetic_features(pred_j3d_glob_chunk.copy())
                pred_feat_m_chunk = extract_manual_features(pred_j3d_glob_chunk.copy())
                pred_feats_k.append(pred_feat_k_chunk)
                pred_feats_m.append(pred_feat_m_chunk)
            pred_feats_k = np.stack(pred_feats_k, axis=0)
            pred_feats_m = np.stack(pred_feats_m, axis=0)
            self.prediction_aggregator[vid]["pred_feats_k"] = pred_feats_k
            self.prediction_aggregator[vid]["pred_feats_m"] = pred_feats_m
        elif self.eval_type == "full":
            pred_ay_j3d_full = pred_ay_j3d.cpu().numpy()
            offset = pred_ay_j3d_full[:1, :1, :]
            pred_ay_j3d_full = pred_ay_j3d_full - offset
            pred_feats_k_full = extract_kinetic_features(pred_ay_j3d_full.copy())
            pred_feats_m_full = extract_manual_features(pred_ay_j3d_full.copy())
            self.prediction_aggregator[vid]["pred_feats_k_full"] = pred_feats_k_full
            self.prediction_aggregator[vid]["pred_feats_m_full"] = pred_feats_m_full
        return

    def on_predict_epoch_start(self, trainer, pl_module):
        self.wandb_html_dict = {}
        if self.save_feats:
            self.feats_arr = []
            self.text_arr = []
            print(
                f"start generating text-to-motion features which will be saved at {self.save_dir}\n"
            )
            print(
                "#### saving a dump feature first to check the correctness of the path #### \n"
            )
            dump_feats = torch.randn(1, 1, 1, 1)
            os.makedirs(self.save_dir, exist_ok=True)
            torch.save(dump_feats, self.save_dir + "/dump.pt")
            print(f"dump feature saved to {self.save_dir}/dump.pt\n")

    # ================== Epoch Summary  ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        self.num_val += 1
        if len(self.wandb_html_dict) > 0:
            pl_module.logger.log_metrics(self.wandb_html_dict)

        """Without logger"""
        local_rank, world_size = trainer.local_rank, trainer.world_size
        monitor_metric = "PFC"

        if self.eval_type == "chunk":
            pred_feats_k = []
            pred_feats_m = []
            for vid, pred in self.prediction_aggregator.items():
                pred_feats_k.append(pred["pred_feats_k"])
                pred_feats_m.append(pred["pred_feats_m"])
                # pred_feats_k_full.append(pred["pred_feats_k_full"])
                # pred_feats_m_full.append(pred["pred_feats_m_full"])
            pred_feats_k = np.concatenate(pred_feats_k, axis=0)
            pred_feats_m = np.concatenate(pred_feats_m, axis=0)
            # pred_feats_k_full = np.stack(pred_feats_k_full, axis=0)
            # pred_feats_m_full = np.stack(pred_feats_m_full, axis=0)

            gt_feats_k = torch.load("inputs/AIST++/gt_feats_k_c150.pt")
            gt_feats_m = torch.load("inputs/AIST++/gt_feats_m_c150.pt")
            # gt_feats_k_full = torch.load("inputs/AIST++/gt_feats_k_full_30fps.pt")
            # gt_feats_m_full = torch.load("inputs/AIST++/gt_feats_m_full_30fps.pt")

            gt_feats_k, pred_feats_k = normalize(gt_feats_k, pred_feats_k)
            gt_feats_m, pred_feats_m = normalize(gt_feats_m, pred_feats_m)
            # gt_feats_k_full, pred_feats_k_full = normalize(gt_feats_k_full, pred_feats_k_full)
            # gt_feats_m_full, pred_feats_m_full = normalize(gt_feats_m_full, pred_feats_m_full)

            # # compute fid
            fid_k = calc_fid(pred_feats_k, gt_feats_k)
            fid_m = calc_fid(pred_feats_m, gt_feats_m)
            # diversity_k = calc_diversity(pred_feats_k)
            # diversity_m = calc_diversity(pred_feats_m)
            # # diversity_k_gt = calc_diversity(gt_feats_k)
            # # diversity_m_gt = calc_diversity(gt_feats_m)
            div_k = calculate_avg_distance(pred_feats_k)
            div_m = calculate_avg_distance(pred_feats_m)
            # div_k_gt = calculate_avg_distance(gt_feats_k_full)
            # div_m_gt = calculate_avg_distance(gt_feats_m_full)
        elif self.eval_type == "full":
            pred_feats_k_full = []
            pred_feats_m_full = []
            for vid, pred in self.prediction_aggregator.items():
                pred_feats_k_full.append(pred["pred_feats_k_full"])
                pred_feats_m_full.append(pred["pred_feats_m_full"])
            pred_feats_k_full = np.stack(pred_feats_k_full, axis=0)
            pred_feats_m_full = np.stack(pred_feats_m_full, axis=0)

            gt_feats_k_full = torch.load("inputs/AIST++/gt_feats_k_full_all.pt")
            gt_feats_m_full = torch.load("inputs/AIST++/gt_feats_m_full_all.pt")

            normed_gt_feats_k_full, normed_pred_feats_k_full = normalize(
                gt_feats_k_full, pred_feats_k_full
            )
            normed_gt_feats_m_full, normed_pred_feats_m_full = normalize(
                gt_feats_m_full, pred_feats_m_full
            )

            # # compute fid
            fid_k = calc_fid(normed_pred_feats_k_full, normed_gt_feats_k_full)
            fid_m = calc_fid(normed_pred_feats_m_full, normed_gt_feats_m_full)
            # diversity_k = calc_diversity(pred_feats_k)
            # diversity_m = calc_diversity(pred_feats_m)
            # # diversity_k_gt = calc_diversity(gt_feats_k)
            # # diversity_m_gt = calc_diversity(gt_feats_m)
            div_k = calculate_avg_distance(normed_pred_feats_k_full)
            div_m = calculate_avg_distance(normed_pred_feats_m_full)
            # div_k_gt = calculate_avg_distance(gt_feats_k_full)
            # div_m_gt = calculate_avg_distance(gt_feats_m_full)

        # Reduce metric_aggregator across all processes
        metric_keys = list(self.metric_aggregator.keys())
        with torch.inference_mode(False):  # allow in-place operation of all_gather
            metric_aggregator_gathered = all_gather(
                self.metric_aggregator
            )  # list of dict
        for metric_key in metric_keys:
            for d in metric_aggregator_gathered:
                self.metric_aggregator[metric_key].update(d[metric_key])

        total = len(self.metric_aggregator[monitor_metric])
        Log.info(f"{total} sequences evaluated in {self.__class__.__name__}")
        if total == 0:
            return

        # print monitored metric per sequence
        mm_per_seq = {
            k: v.mean() for k, v in self.metric_aggregator[monitor_metric].items()
        }
        if len(mm_per_seq) > 0:
            sorted_mm_per_seq = sorted(
                mm_per_seq.items(), key=lambda x: x[1], reverse=True
            )
            n_worst = 5 if trainer.state.stage == "validate" else len(sorted_mm_per_seq)
            if local_rank == 0:
                Log.info(
                    f"monitored metric {monitor_metric} per sequence\n"
                    + "\n".join(
                        [f"{m:5.4f} : {s}" for s, m in sorted_mm_per_seq[:n_worst]]
                    )
                    + "\n------"
                )

        # average over all batches
        metrics_avg = {}
        for k, v in self.metric_aggregator.items():
            try:
                metrics_avg[k] = np.concatenate(list(v.values())).mean()
            except Exception:
                metrics_avg[k] = np.stack(list(v.values())).mean()

        metrics_avg["fid_k"] = fid_k
        metrics_avg["fid_m"] = fid_m
        metrics_avg["div_k"] = div_k
        metrics_avg["div_m"] = div_m
        if local_rank == 0:
            Log.info(
                "[Metrics] AIST++:\n"
                + "\n".join(f"{k}: {v:.4f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        # save to logger if available
        if pl_module.logger is not None:
            cur_epoch = pl_module.current_epoch
            for k, v in metrics_avg.items():
                pl_module.logger.log_metrics(
                    {f"val_metric_AIST++/{k}": v},
                    step=cur_epoch,
                )

        # reset
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}
        self.prediction_aggregator = {}

        if self.save_feats:
            feats_arr = torch.cat(self.feats_arr, dim=0).cpu()
            results = {
                "feats": feats_arr,
                "text": self.text_arr,
            }
            os.makedirs(self.save_dir, exist_ok=True)
            if self.dataset_part_ind >= 0:
                fname = (
                    self.save_dir
                    + f"/new_feats_part{self.dataset_part_ind}_len{self.text_len}_{self.trial_ind}.pt"
                )
            else:
                fname = (
                    self.save_dir + f"/new_feats_len{self.text_len}_{self.trial_ind}.pt"
                )
            torch.save(results, fname)
            os.chmod(fname, 0o755)
            print(f"text-to-motion features saved to {fname}")


def normalize(feat1, feat2):
    mean = feat1.mean(axis=0)
    std = feat1.std(axis=0)
    std[std == 0] = 1
    return (feat1 - mean) / std, (feat2 - mean) / std


def calc_fid(feats_gen, feats_gt):
    mu_gen = np.mean(feats_gen, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)

    mu_gt = np.mean(feats_gt, axis=0)
    sigma_gt = np.cov(feats_gt, rowvar=False)

    mu1, mu2, sigma1, sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calc_diversity(feats):
    n, c = feats.shape

    diff = np.array([feats] * n) - feats.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n - 1)


def calculate_avg_distance(feats, mean=None, std=None):
    n = feats.shape[0]
    # normalize the scale
    if mean is None:
        mean = np.mean(feats, axis=0)
    if std is None:
        std = np.std(feats, axis=0)
        std[std == 0] = 1
    feats = (feats - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feats[i] - feats[j])
    dist /= (n * n - n) / 2
    return dist
