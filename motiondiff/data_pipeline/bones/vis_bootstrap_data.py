import os
import sys

sys.path.append("./")
import argparse
import glob

import lmdb
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from motiondiff.data_pipeline.get_data import get_dataset_loader
from motiondiff.utils.config import create_config
from motiondiff.utils.tools import import_type_from_str
from motiondiff.utils.torch_utils import tensor_to

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out_dir", default="out/bootstrap_datasets_vis/")
parser.add_argument(
    "-c", "--cfg", default="mask_bones_twostage_root_rot_v4_body_key_fine_tune_mix"
)
parser.add_argument("-bs", "--batch_size", type=int, default=256)
parser.add_argument("-n", "--num_frames", type=int, default=196)
args = parser.parse_args()

out_dir = os.path.join(args.out_dir, f"frames_{args.num_frames}")
os.makedirs(out_dir, exist_ok=True)

torch.set_grad_enabled(False)
cfg = create_config(args.cfg, tmp=True, training=False)
model = import_type_from_str(cfg.model.type)(cfg)
model.cuda()
device = torch.device("cuda")

assert model.motion_rep == "global_root_local_joints_root_rot"
model.normalize_global_pos = False  # want to return unnormalized
# for global root, we only care what values it takes on over the max_len we're predicting (196 frames)
# the downside here is the data loader is returning random 196 frame crops, so this is not really the stats
#       over the whole dataset. And if we run this multiple times we'll get slightly different stats.
#       But since there is so much data, the difference is very small.
cfg.train_dataset.num_frames = args.num_frames

print(cfg.train_dataset["mix_paths"])
print(cfg.train_dataset["sample_ratio"])
num_datasets = len(cfg.train_dataset["mix_paths"]) + 1  # also the base bones dataset

fig, axes = plt.subplots(nrows=1, ncols=num_datasets)
fig.set_size_inches(18.0, 6.0)
max_val = 17

for di in range(num_datasets):
    cfg.train_dataset["sample_ratio"] = [1e-8, 1e-8, 1e-8]
    cfg.train_dataset["sample_ratio"][di] = 1.0
    print(cfg.train_dataset["sample_ratio"])
    dataloader = get_dataset_loader(
        name=cfg.train_dataset.name,
        batch_size=args.batch_size,
        split="train",
        hml_mode="train",
        drop_last=False,
        data_cfg=cfg.train_dataset,
        num_frames=None,
    )  # num_frames is not actually used here, only the one in the cfg
    print(len(dataloader))

    end_pos_arr = []
    bi = 0
    for batch in tqdm(dataloader):
        motion, cond = batch
        motion, cond = tensor_to([motion, cond], device=device)

        global_motion = model.convert_motion_rep(
            motion
        ).permute(
            (0, 2, 3, 1)
        )  # returns root motion in metric global since normalize_global_pos = False (the rest is normalized)
        pad_mask = cond["y"]["mask"]
        gt_data_lengths = cond["y"]["lengths"] - 1

        # print(global_motion.shape) # [batch, 1, num_steps, featsize]

        # get final root 2D location
        root_motion = global_motion[:, 0, :, [0, 2]]  # [batch, num_steps, 2]

        cur_idx = gt_data_lengths[:, None, None].expand((root_motion.shape[0], 1, 2))
        final_root_pos = torch.gather(root_motion, dim=1, index=cur_idx)[:, 0]

        end_pos_arr.append(final_root_pos)

        # if bi > 100:
        #     break
        bi += 1

    end_pos_arr = torch.cat(end_pos_arr, dim=0)
    print(end_pos_arr.shape)

    end_pos_arr = end_pos_arr.cpu().numpy()

    title = "Base" if di == 0 else cfg.train_dataset["mix_paths"][di - 1].split("/")[-1]
    axes[di].set_title(title)
    axes[di].hist2d(
        end_pos_arr[:, 0],
        end_pos_arr[:, 1],
        bins=100,
        range=[[-max_val, max_val], [-max_val, max_val]],
        cmap="plasma",
        norm=mcolors.PowerNorm(0.1),
    )
    axes[di].set_xlim(-max_val, max_val)
    axes[di].set_ylim(-max_val, max_val)
    axes[di].set_aspect("equal", adjustable="box")

fig.tight_layout()
plt.show()
