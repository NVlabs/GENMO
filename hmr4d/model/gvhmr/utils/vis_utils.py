import os
import torch
import numpy as np
import wandb
from motiondiff.models.mdm.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
from motiondiff.utils.vis_scenepic import ScenepicVisualizer
from motiondiff.utils.tools import wandb_run_exists


sp_visualizer = ScenepicVisualizer("inputs/smpl_data", device='cuda')


def visualize_smpl_scene(vis_type, index, vid, j3d, gt_j3d, logger, transform_mode=None):
    if transform_mode == 'global':
        global_rot = axis_angle_to_matrix(torch.tensor([np.pi / 2, 0, 0])).cuda()
        j3d = (global_rot @ j3d.transpose(1, 2)).transpose(1, 2)
        gt_j3d = (global_rot @ gt_j3d.transpose(1, 2)).transpose(1, 2)
    elif transform_mode == 'local':
        global_rot = axis_angle_to_matrix(torch.tensor([-np.pi / 2, 0, 0])).cuda()
        j3d = (global_rot @ j3d.transpose(1, 2)).transpose(1, 2)
        gt_j3d = (global_rot @ gt_j3d.transpose(1, 2)).transpose(1, 2)
        j3d[..., 2] += 0.8
        gt_j3d[..., 2] += 0.8
    smpl_seq = {
        'pred': {
            'joints_pos': j3d,
        },
        'gt': {
            'joints_pos': gt_j3d,
        },
    }

    vid_ = vid.replace("/", "_")
    fname = f'{index:03d}-{vid_}'
    html_file = f"out/{vis_type}/{fname}.html"
    os.makedirs(os.path.dirname(html_file), exist_ok=True)
    sp_visualizer.vis_smpl_scene(smpl_seq, html_file)
    
    if wandb_run_exists():
        logger.log_metrics({f'{vis_type}/{fname}': wandb.Html(html_file)})

# Example usage withi