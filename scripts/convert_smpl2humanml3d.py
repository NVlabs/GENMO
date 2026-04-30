# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.12.13

# from smplx.lbs import lbs
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Vassilis Choutas <https://vchoutas.github.io/>
#
from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from motiondiff.utils.humanml3d_tools import (
    BodyModel,
    c2c,
    process_file,
    recover_from_ric,
)

male_bm_path = "./body_models/smplh/male/model.npz"
male_dmpl_path = "./body_models/dmpls/male/model.npz"

female_bm_path = "./body_models/smplh/female/model.npz"
female_dmpl_path = "./body_models/dmpls/female/model.npz"

neutral_bm_path = "./body_models/smplh/neutral/model.npz"
neutral_dmpl_path = "./body_models/dmpls/neutral/model.npz"

num_betas = 10  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

male_bm = BodyModel(
    bm_fname=male_bm_path,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=male_dmpl_path,
).to("cuda")
female_bm = BodyModel(
    bm_fname=female_bm_path,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=female_dmpl_path,
).to("cuda")
neutral_bm = BodyModel(
    bm_fname=neutral_bm_path,
    num_betas=num_betas,
    num_dmpls=num_dmpls,
    dmpl_fname=neutral_dmpl_path,
).to("cuda")
faces = c2c(male_bm.f)

# GT_vectors_path = 'outputs/humanml3d_feats_gt/feats_test.pt'
# Pred_vectors_path = 'outputs/mocap_mixed_v1/unimfm/unimfm_test_st_g8/version_0/text_feats/feats.pt'

# smpl_format_data = 'outputs/humanml3d_feats_gt/feats_test_smpl.pt'
# save_path = 'outputs/humanml3d_feats_gt/feats_test_humanml3d_format.npy'

smpl_format_data = (
    "outputs/mocap_mixed_v1/unimfm/unimfm_test_st_g8/version_0/text_feats/feats_smpl.pt"
)
save_path = "outputs/mocap_mixed_v1/unimfm/unimfm_test_st_g8/version_0/text_feats/feats_humanml3d_format.npy"

bdata = torch.load(smpl_format_data)

global_orient = bdata["global_orient_w"].to("cuda")
betas = bdata["betas"].to("cuda")
body_pose = bdata["body_pose"].to("cuda")
transl = bdata["transl_w"].to("cuda")

total_num, L = global_orient.shape[:2]

all_humanml3d_data = []

for i in tqdm(range(total_num)):
    body_parms = {
        "root_orient": global_orient[i],
        "pose_body": body_pose[i],
        "pose_hand": torch.zeros((L, 90)).to("cuda"),
        "trans": transl[i],
        "betas": betas[i],
    }
    joints_num = 22
    with torch.no_grad():
        body = male_bm(**body_parms)
        pose_seq_np = body.Jtr.detach().cpu().numpy()

    joints_data = pose_seq_np[:, :joints_num]
    data, ground_positions, positions, l_velocity = process_file(joints_data, 0.002)
    rec_ric_data = recover_from_ric(
        torch.from_numpy(data).unsqueeze(0).float(), joints_num
    )
    all_humanml3d_data.append(data[None, ...])

all_humanml3d_data = np.concatenate(all_humanml3d_data, axis=0)
np.save(save_path, all_humanml3d_data)
