import torch
import pytorch_lightning as pl
import time
import numpy as np
import hydra
import os

from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.model.gvhmr.utils.vis_utils import visualize_smpl_scene, visualize_intermediate_smpl_scene
from motiondiff.models.mdm.rotation_conversions import axis_angle_to_quaternion
from hmr4d.utils.geo.quaternion import qinv_np, qrot_np
from hmr4d.utils.eval.eval_utils import batch_compute_similarity_transform_torch


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def get_y_heading_q(q):
    q_new = q.clone()
    q_new[..., 1] = 0
    q_new[..., 3] = 0
    q_new = normalize(q_new)
    return q_new

def canonicalize(posed_joints, root_rot, z_up=True):
    root_quat = axis_angle_to_quaternion(root_rot)
    heading_quat = get_y_heading_q(root_quat)
    heading_quat = heading_quat.unsqueeze(1).repeat(1, posed_joints.shape[1], 1)
    heading_quat = heading_quat.numpy()
    heading_quat_inv = qinv_np(heading_quat)
    init_heading_quat_inv = np.repeat(heading_quat_inv[[0]], heading_quat_inv.shape[0], axis=0)
    
    posed_joints = posed_joints.numpy()

    '''Put on Floor'''
    floor_height = posed_joints.min(axis=0).min(axis=0)[1]
    posed_joints[:, :, 1] -= floor_height
    # verts[:, :, 1] -= floor_height

    '''XZ at origin'''
    root_pos_init = posed_joints[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    posed_joints = posed_joints - root_pose_init_xz
    # verts = verts - root_pose_init_xz

    '''All initially face Z+'''
    posed_joints = qrot_np(init_heading_quat_inv, posed_joints)
    # verts = qrot_np(init_heading_quat_inv, verts)

    posed_joints = torch.Tensor(posed_joints)
    return posed_joints


class MetricInpainting(pl.Callback):
    def __init__(self, vis_every_n_val=10, save_feats=False, save_dir=None, endecoder=None, vis=True):
        super().__init__()
        self.vis_every_n_val = vis_every_n_val
        self.num_val = 0
        self.save_feats = save_feats
        self.save_dir = save_dir
        if endecoder is not None:
            self.endecoder = hydra.utils.instantiate(endecoder).cuda()
        self.vis = vis
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

        self.test_start_time = int(time.time())

        self.metrics = {'mpjpe_one': [], 'pa_mpjpe_one': [], 'mpjpe_min': [], 'pa_mpjpe_min': []}

    # ================== Batch-based Computation  ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """The behaviour is the same for val/test/predict"""
        assert batch["B"] == 1

        # Move to cuda if not
        for g in ["male", "female", "neutral"]:
            self.smplx_model[g] = self.smplx_model[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        text = batch['caption'][0]
        vid = text.replace(' ', '_').replace('.', '_').replace(',', '_')
        vid = vid[:100]
        seq_length = batch["length"][0].item()
        gender = 'neutral'

        # Groundtruth (world, cam)
        target_w_params = {k: v[0] for k, v in batch["smpl_params_w"].items()}
        target_w_j3d = self.smplx_model[gender](**target_w_params)
        offset = batch["smpl_params_w"]["transl"][0, :, None] - target_w_j3d[:, [0]]
        target_w_j3d = target_w_j3d + offset
        # Canonicalize the motion so that it always starts facing Z+
        target_j3d_can = canonicalize(target_w_j3d.cpu(), target_w_params['global_orient'].cpu()).cuda()

        # 2. ay
        mpjpe_best, pa_mpjpe_best, pred_j3d_can_best = 1e9, 1e9, None
        for res_idx, out in enumerate(outputs):
            pred_smpl_params_global = out["pred_smpl_params_global"]
            pred_ay_j3d = self.smplx_model["neutral"](**pred_smpl_params_global)
            pred_j3d_can = canonicalize(pred_ay_j3d.cpu(), pred_smpl_params_global['global_orient'].cpu()).cuda()
            # Hack: aligns the root position at the first frame with GT
            pred_j3d_can[:, :] = pred_j3d_can[:, :] + target_j3d_can[[0], [0]] - pred_j3d_can[[0], [0]]

            # Compute metrics
            if trainer.model.model_cfg.inpainting_3d.mode.startswith('body_pose_root_rot_keyframe'):
                keyframes = batch["keyframes"]
                pred_kf_j3d_relative = pred_j3d_can[keyframes[1:]] - pred_j3d_can[keyframes[1:], :1]
                target_kf_j3d_relative = target_j3d_can[keyframes[1:]] - target_j3d_can[keyframes[1:], :1]
                mpjpe = (pred_kf_j3d_relative - target_kf_j3d_relative).norm(dim=-1).mean() * 1000
                pred_kf_j3d_relative_pa = batch_compute_similarity_transform_torch(pred_kf_j3d_relative, target_kf_j3d_relative)
                pa_mpjpe = (pred_kf_j3d_relative_pa - target_kf_j3d_relative).norm(dim=-1).mean() * 1000

                if mpjpe.item() < mpjpe_best:
                    mpjpe_best = mpjpe.item()
                    pa_mpjpe_best = pa_mpjpe.item()
                    pred_j3d_can_best = pred_j3d_can

        self.metrics['mpjpe_one'] += [mpjpe.item()]
        self.metrics['pa_mpjpe_one'] += [pa_mpjpe.item()]
        self.metrics['mpjpe_min'] += [mpjpe_best]
        self.metrics['pa_mpjpe_min'] += [pa_mpjpe_best]
        
        if self.save_feats:
            encoder_inputs = {
                'smpl_params_w': {k: v.unsqueeze(0) for k, v in outputs["pred_smpl_params_global"].items()},
            }
            feats = self.endecoder.encode_humanml3d(encoder_inputs)
            self.feats_arr.append(feats)
        
        if self.vis:
            # Visualize
            if trainer.global_rank == 0 and self.num_val % self.vis_every_n_val == 0:
                wandb_dict = visualize_smpl_scene(f'vis_inpainting_3d_{trainer.model.model_cfg.inpainting_3d.mode}_{self.test_start_time}', batch_idx, f'{res_idx:02d}-{vid}', pred_j3d_can_best, target_j3d_can, transform_mode='global')
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
            torch.save(feats_arr, self.save_dir / 'feats.pt')
            print(f"text-to-motion features saved at {self.save_dir}")
        
        for metric_name in sorted(self.metrics):
            metric_avg = np.average(self.metrics[metric_name])
            print(f"Metric [{metric_name}]: {metric_avg:.05f}")
            pl_module.logger.log_metrics({f"metrics/{metric_name}": metric_avg}, step=pl_module.current_epoch)
