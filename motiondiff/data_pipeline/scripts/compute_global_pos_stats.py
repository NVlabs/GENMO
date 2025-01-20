import os
import sys
sys.path.append('./')
import argparse
import glob
import torch
import lmdb
import numpy as np
from tqdm import tqdm

from motiondiff.utils.tools import import_type_from_str
from motiondiff.utils.config import create_config
from motiondiff.data_pipeline.get_data import get_dataset_loader, get_dataset
from motiondiff.utils.torch_utils import tensor_to


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out_file', default='data/stats/HumanML3D_global_pos_vel_stats.npy')
parser.add_argument('-c', '--cfg', default='mdm_t5_enc_cat_len50_aug_amp2_b256_twostage_v2_vel')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('-bs', '--batch_size', type=int, default=256)
args = parser.parse_args()


torch.set_grad_enabled(False)
cfg = create_config(args.cfg, tmp=True, training=False)
model = import_type_from_str(cfg.model.type)(cfg)
model.cuda()
device = torch.device('cuda')

dataloader = get_dataset_loader(name='humanml', batch_size=args.batch_size, num_frames=None, split='train', hml_mode='train', debug=args.debug, drop_last=False)


num_motion = 0
global_motion_arr = []

for batch in dataloader:
    motion, cond = batch
    motion, cond = tensor_to([motion, cond], device=device)

    global_motion = model.convert_motion_rep(motion)
    global_motion = global_motion[:, :, 0].permute(0, 2, 1)[..., :model.motion_root_dim].reshape(-1, model.motion_root_dim)
    mask = cond['y']['mask'].view(-1)
    global_motion = global_motion[mask]
    global_motion_arr.append(global_motion.cpu().numpy())

    num_motion += motion.shape[0]
    print(global_motion.shape, motion.shape, num_motion)

global_motion_arr = np.concatenate(global_motion_arr, axis=0)
global_motion_mean = np.mean(global_motion_arr, axis=0)
global_motion_std = np.std(global_motion_arr, axis=0)
np.save(args.out_file, (global_motion_mean, global_motion_std))
print('num_motion:', num_motion)