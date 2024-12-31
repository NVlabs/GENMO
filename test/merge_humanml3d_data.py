import codecs as cs
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from hmr4d.utils.smplx_utils import make_smplx
from motiondiff.models.mdm.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from motiondiff.utils.vis_scenepic import ScenepicVisualizer

device = 'cuda:0'
data_file = "/home/jiefengl/git/HumanML3D/HumanML3D/train.txt"
text_dir = "/home/jiefengl/git/HumanML3D/HumanML3D/texts/"
motion_dir = "/home/jiefengl/git/HumanML3D/pose_smplh/"
sp_visualizer = ScenepicVisualizer("/home/jiefengl/git/physdiff_megm/data/smpl_data", device=device)
trans_matrix = torch.tensor(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)
smpl_dict = {
    'male': make_smplx(type="supermotion_smpl24_male"),
    'female': make_smplx(type="supermotion_smpl24_female"),
    'neutral': make_smplx(type="supermotion_smpl24"),
}

with open(data_file, "r") as f:
    lines = f.readlines()

db = {}

new_name_list = []
length_list = []
miss_cnt = 0
for line in tqdm(lines):
    motion_id = line.strip()
    if not os.path.exists(os.path.join(motion_dir, f"{motion_id}.pt")):
        miss_cnt += 1
        continue

    text_file = os.path.join(text_dir, f"{motion_id}.txt")
    motion_file = os.path.join(motion_dir, f"{motion_id}.pt")

    motion_data = torch.load(motion_file)
    # text_data = open(text_file, "r").read()

    gender = str(motion_data["gender"])
    if gender.startswith("b'"):
        gender = gender[2:-1]
    assert gender in smpl_dict
    # smpl_layer = smpl_dict[gender]

    # swap y and z
    trans = motion_data["trans_n"]
    trans = trans @ trans_matrix
    global_orient = motion_data["root_orient_n"]
    global_orient_mat = axis_angle_to_matrix(global_orient)
    global_orient_mat = torch.einsum("ij,bnjk->bnik", [trans_matrix, global_orient_mat])
    global_orient = matrix_to_axis_angle(global_orient_mat).reshape(-1, 3)
    body_pose = motion_data["pose_body"].reshape(-1, 21 * 3)
    pose = torch.cat([global_orient, body_pose], dim=1)
    betas = motion_data["betas"][0, :10]

    name = motion_id
    flag = False
    text_data = []

    with cs.open(text_file) as f:
        for caption_line in f.readlines():
            text_dict = {}
            line_split = caption_line.split('#')
            caption = line_split[0]
            tokens = line_split[1:]
            f_tag = float(line_split[2])
            to_tag = float(line_split[3])
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag

            text_dict['caption'] = caption
            text_dict['tokens'] = tokens
            if f_tag == 0.0 and to_tag == 0.0:
                flag = True
                text_data.append(text_dict)
            else:
                # try:
                n_pose = pose[int(f_tag*20) : int(to_tag*20)]
                n_trans = trans[int(f_tag*20) : int(to_tag*20)]

                if (len(n_pose)) < 25 or (len(n_pose) >= 200):
                    continue
                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                while new_name in db:
                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                db[new_name] = {
                    "pose": n_pose,
                    "trans": n_trans,
                    "beta": betas,
                    "gender": gender,
                    "text_data": [text_dict],
                    "motion_id": motion_id,
                    "length": len(n_pose),
                }
                new_name_list.append(new_name)
                length_list.append(len(n_pose))

        if flag:
            data = {
                "pose": pose,
                "trans": trans,
                "beta": betas,
                "gender": gender,
                "text_data": text_data,
                "motion_id": motion_id,
                "length": len(pose),
            }

            db[name] = data

print(f"missed {miss_cnt} humanact12 motions")
print(f"total {len(db)} samples from {len(lines)} motions")
torch.save(db, "humanml3d_smplhpose.pth")
