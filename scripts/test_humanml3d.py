import os

import torch

split = "train"
fname = f"inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose_{split}.pth"
motion_data = torch.load(fname)

for vid, data in motion_data.items():
    print(vid)
    global_orient = data["pose"][:, :3]
    body_pose = data["pose"][:, 3:]
    betas = data["beta"]
    trans = data["trans"]
    print(global_orient.shape, body_pose.shape, betas.shape)
    break
