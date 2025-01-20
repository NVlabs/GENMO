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
from motiondiff.data_pipeline.get_data import get_dataset_loader
from motiondiff.utils.torch_utils import tensor_to


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out_dir', default='assets/bones/global_stats/v5')
parser.add_argument('-c', '--cfg', default='bones_twostage_root_rot_v5')
parser.add_argument('-bs', '--batch_size', type=int, default=256)
parser.add_argument('-n', '--num_frames', type=int, default=196)
args = parser.parse_args()

out_dir = os.path.join(args.out_dir, f'frames_{args.num_frames}')
os.makedirs(out_dir, exist_ok=True)

torch.set_grad_enabled(False)
cfg = create_config(args.cfg, tmp=True, training=False)
model = import_type_from_str(cfg.model.type)(cfg)
model.cuda()
device = torch.device('cuda')

model.normalize_global_pos = False # don't have the stats yet to do this
# for global root, we only care what values it takes on over the max_len we're predicting (196 frames)
# the downside here is the data loader is returning random 196 frame crops, so this is not really the stats
#       over the whole dataset. And if we run this multiple times we'll get slightly different stats.
#       But since there is so much data, the difference is very small.
cfg.train_dataset.num_frames = args.num_frames 

dataloader = get_dataset_loader(name='bones', batch_size=args.batch_size, split='train', hml_mode='train', drop_last=False, data_cfg=cfg.train_dataset,
                                num_frames=None) # num_frames is not actually used here, only the one in the cfg

# Initialize variables to compute the mean and std incrementally
count = 0
mean = 0
M2 = 0
# dist_list = []
for batch in tqdm(dataloader):
    motion, cond = batch
    motion, cond = tensor_to([motion, cond], device=device)

    global_motion = model.convert_motion_rep(motion) # returns motion in metric global since normalize_global_pos = False
    pad_mask = cond['y']['mask']

    ### debug
    # root_pos = global_motion[:,[0,2]] # [batch, 3, 1, 196]
    # mlen = cond['y']['lengths']
    # last_root_pos = torch.gather(root_pos, 3, mlen[:,None,None,None].expand(root_pos.shape[:3] + (1,)) - 1)[:,:,0,0]
    # dist = torch.norm(last_root_pos, dim=-1)
    # dist_list = dist_list + dist.cpu().numpy().tolist()

    # inlier_mask = dist < 10.0
    # global_motion = global_motion[inlier_mask] # throw away outliers
    # pad_mask = pad_mask[inlier_mask]
    # print(global_motion.size())
    # if global_motion.size(0) == 0:
    #     continue
    ######

    global_motion = global_motion[:, :, 0].permute(0, 2, 1)[..., :model.motion_root_dim].reshape(-1, model.motion_root_dim)
    # ignore padding
    mask = pad_mask.view(-1)
    global_motion = global_motion[mask] # num_valid_timesteps x 5

    # Update count, mean, and M2 for the new data
    count += global_motion.shape[0]
    delta = global_motion - mean
    mean += delta.sum(dim=0, keepdim=True) / count
    delta2 = global_motion - mean
    M2 += torch.sum(delta * delta2, dim=0, keepdim=True)

# import matplotlib.pyplot as plt

# # Create histogram
# plt.hist(dist_list, bins=50, color='blue', edgecolor='black')

# # Add labels and title
# plt.xlabel('Final Dist')
# plt.ylabel('Frequency')

# plt.savefig(os.path.join(out_dir, 'dist_distrib.png'))

# # Show the plot
# plt.show()

# Finalize the mean and standard deviation
std = torch.sqrt(M2 / count)
std[torch.isnan(std)] = 1e-3
std[std < 1e-3] = 1e-3
# Save the computed statistics
np.save(os.path.join(out_dir, 'mean.npy'), mean[0].cpu().numpy())
np.save(os.path.join(out_dir, 'std.npy'), std[0].cpu().numpy())
print('mean:', mean)
print('std:', std)