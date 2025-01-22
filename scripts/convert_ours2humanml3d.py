from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys 
import os 
from tqdm import tqdm
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from hmr4d.model.gvhmr.utils.endecoder import EnDecoder

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys 
import os 
from tqdm import tqdm
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from motiondiff.utils.humanml3d_tools import BodyModel, c2c, process_file, recover_from_ric

male_bm_path = './body_models/smplh/male/model.npz'
male_dmpl_path = './body_models/dmpls/male/model.npz'

female_bm_path = './body_models/smplh/female/model.npz'
female_dmpl_path = './body_models/dmpls/female/model.npz'

neutral_bm_path = './body_models/smplh/neutral/model.npz'
neutral_dmpl_path = './body_models/dmpls/neutral/model.npz'

num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to('cuda')
female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path).to('cuda')
neutral_bm = BodyModel(bm_fname=neutral_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=neutral_dmpl_path).to('cuda')
faces = c2c(male_bm.f)

encoder = EnDecoder(stats_name='DEFAULT_01', encode_type='humanml3d').cuda()

# data_path = 'outputs/humanml3d_feats_gt/feats_test.pt'
data_path = (
    "/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/"
    "workspaces/motiondiff/motiondiff_results/yey/gvhmr/mocap_mixed_v1/unimfm/unimfm_est_st_norm_di_lg_g8/"
    "version_0/text_feats_ts10_humanml3d/"
)

for seed in [0, 1, 2, 3, 4]:
    save_data_name = 'feats_seed{}_263d.pt'.format(seed)
    # np.save(data_path + save_data_name, all_humanml3d_data)
    final_save_path = data_path + save_data_name
    final_save_path = final_save_path.replace('yey', 'jinkunc')
    print('Processing to save at: ', final_save_path)
    data1 = 'feats_part0_len196_{}.pt'.format(seed)
    data2 = 'feats_part1_len196_{}.pt'.format(seed)
    raw_data1 = torch.load(data_path + data1)['feats'].cuda()
    raw_data2 = torch.load(data_path + data2)['feats'].cuda()

    raw_text1 = torch.load(data_path + data1)['text']
    raw_text2 = torch.load(data_path + data2)['text']

    raw_data = torch.cat([raw_data1, raw_data2], dim=0)

    bdata = encoder.decode_humanml3d(raw_data)

    global_orient = bdata['global_orient_w'].to('cuda')
    betas = bdata['betas'].to('cuda')
    body_pose = bdata['body_pose'].to('cuda')
    transl = bdata['transl_w'].to('cuda')

    total_num, L = global_orient.shape[:2]

    all_humanml3d_data = []

    for i in tqdm(range(total_num)):
        body_parms = {
            'root_orient': global_orient[i],
            'pose_body': body_pose[i],
            'pose_hand': torch.zeros((L, 90)).to('cuda'),
            'trans': transl[i],
            'betas': betas[i],
        }
        joints_num = 22
        with torch.no_grad():
            body = male_bm(**body_parms)
            pose_seq_np = body.Jtr.detach().cpu().numpy()   
            
        joints_data = pose_seq_np[:, :joints_num]
        data, ground_positions, positions, l_velocity = process_file(joints_data, 0.002)
        rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
        all_humanml3d_data.append(data[None,...])

    all_humanml3d_data = np.concatenate(all_humanml3d_data, axis=0)

    final_feats = {}
    final_feats['feats'] = torch.from_numpy(all_humanml3d_data).float().cuda()
    final_feats['text'] = raw_text1 + raw_text2
    final_save_dir = os.path.dirname(final_save_path)
    os.makedirs(final_save_dir, exist_ok=True)
    torch.save(final_feats, final_save_path)
    print('Saved to: ', final_save_path)
