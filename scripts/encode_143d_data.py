import os
import sys

import torch
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import numpy as np
from scipy import linalg

from hmr4d.datamodule.mocap_trainX_testY import collate_fn
from hmr4d.dataset.pure_motion.humanml3d import Humanml3dDataset
from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
from motiondiff.utils.torch_utils import tensor_to

split = "test"
humanml3d_path = "inputs/HumanML3D_SMPL/hmr4d_support"
humanml3d_file = humanml3d_path + "/humanml3d_smplhpose_{}.pth".format(split)
save_path = "outputs/humanml3d_feats_gt"
os.makedirs(save_path, exist_ok=True)

######### Part 1: Compute the statistics of the text-to-motion GT features #########
motion_files = torch.load(humanml3d_file)

# dict_keys(['pose', 'trans', 'beta', 'gender', 'text_data', 'motion_id', 'length'])
dataset = Humanml3dDataset(cam_augmentation="v11", split=split)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=collate_fn
)
encoder = EnDecoder(stats_name="DEFAULT_01", encode_type="humanml3d").cuda()

feats_arr = []
for data in tqdm(dataloader):
    breakpoint()
    data = tensor_to(data, "cuda")
    # x = encoder.encode(data)
    feats = encoder.encode_humanml3d(data)
    feats_arr.append(feats)

save_feats = torch.cat(feats_arr, dim=0).cpu()
torch.save(save_feats, os.path.join(save_path, "feats_{}.pt".format(split)))
print(f"text-to-motion GT features saved at {save_path}")


########## End Part 1 ##########

########## Part 2: Train the VAE for FID computation ##########
