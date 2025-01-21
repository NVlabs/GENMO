"""
    For the feature meaning of the 263-D HumanML3D data, please refer to:
    https://github.com/EricGuo5513/HumanML3D/issues/83
"""
import torch 
import sys 
import os 
from tqdm import tqdm
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from hmr4d.utils.pylogger import Log
from hmr4d.utils.geo.hmr_global import get_local_transl_vel, rollout_local_transl_vel, get_static_joint_mask
from hmr4d.utils.geo.quaternion import qinv_np, quaternion_to_cont6d, cont6d_to_matrix
from hmr4d.utils.smplx_utils import make_smplx
import numpy as np
from motiondiff.utils.torch_transform import angle_axis_to_quaternion, quaternion_to_angle_axis, quat_mul, quat_conjugate, get_y_heading_q, quat_apply
from motiondiff.models.common.smpl import SMPL
from motiondiff.models.mdm.rotation_conversions import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from motiondiff.utils.conversion import humanml_to_smpl
from hmr4d.model.gvhmr.utils.endecoder import EnDecoder

encoder = EnDecoder(stats_name='DEFAULT_01', encode_type='humanml3d').cuda()

smpl_data_path = 'inputs/smpl_data'
humanml3d_raw_data = 'outputs/mdm_gt_motions.npy'
save_path = 'outputs/converted_mdm_gt_motions.npy'
humanml3d_raw_path = '/home/jinkunc/fsw_datasets/HumanML3D'

# mean = np.load(os.path.join(humanml3d_raw_path, 'Mean.npy'))
# std = np.load(os.path.join(humanml3d_raw_path, 'Std.npy'))

mean = np.load('outputs/humanml3d_mean.npy')
std = np.load('outputs/humanml3d_std.npy')

smpl = SMPL(smpl_data_path)

"""
    root_rot_velocity (B, seq_len, 1)
    root_linear_velocity (B, seq_len,2)
    root_y (B, seq_len, 1)
    ric_data (B, seq_len, (joint_num - 1)*3)
    rot_data (B, seq_len, (joint_num - 1)*6)
    local_velocity (B, seq_len,joint_num * 3)
    foot contact (B, seq_len, 4)
"""
# joint_num = 22
raw_motions = np.load(humanml3d_raw_data)
# root_rot_velocity = raw_motions[:, :, 0:1]
# root_linear_velocity = raw_motions[:, :, 1:3]
# root_y = raw_motions[:, :, 3:4]
# ric_data = raw_motions[:, :, 4:4 + (joint_num - 1) * 3]
# rot_data = raw_motions[:, :, 4 + (joint_num - 1) * 3:4 + (joint_num - 1) * 9]
# local_velocity = raw_motions[:, :, 4 + (joint_num - 1) * 9:4 + (joint_num - 1) * 9 + joint_num * 3]
# foot_contact = raw_motions[:, :, 4 + (joint_num - 1) * 9 + joint_num * 3:]

raw_motions = torch.tensor(raw_motions).cuda()
mean = torch.tensor(mean).cuda()
std = torch.tensor(std).cuda()
smpl = smpl.cuda()

batch_size = 64
total_num  = raw_motions.shape[0]
num_batches = total_num // batch_size + 1
feats_arr = [] 

for i in tqdm(range(num_batches)):
    if batch_size * i == total_num:
        break 
    else:
        raw_motions_batch = raw_motions[i * batch_size: i * batch_size + batch_size]
    batch_size, seq_len, _ = raw_motions_batch.shape
    raw_motions_batch = raw_motions_batch.view(batch_size * seq_len, 1, -1)
    raw_motions_batch = raw_motions_batch.permute(0, 2, 1).unsqueeze(2) # [batch, nfeat, 1, seq_len]
    
    # real_motions_batch = raw_motions_batch * std[None, ..., None, None] + mean[None, ..., None, None]
    converted_batch = humanml_to_smpl(raw_motions_batch.float(), mean.float(), std.float(), smpl)
    smpl_pose, smpl_trans, joints_3d = converted_batch
    # HumanML3D removes the last two joints from SMPL (left and right hands)
    # see: https://github.com/EricGuo5513/HumanML3D/issues/67
    smpl_pose = smpl_pose.view(batch_size, seq_len, -1, 3)
    body_pose = smpl_pose[:, :, 1:22].view(batch_size, seq_len, -1)
    global_orient = smpl_pose[:, :, 0].view(batch_size, seq_len, -1)
    betas = torch.zeros((batch_size, seq_len, 10)).cuda()
    smpl_trans = smpl_trans.view(batch_size, seq_len, -1)

    inputs = dict()
    inputs["smpl_params_w"] = dict()
    inputs["smpl_params_w"]["body_pose"] = body_pose 
    inputs["smpl_params_w"]["betas"] = betas
    inputs["smpl_params_w"]["global_orient"] = global_orient 
    inputs["smpl_params_w"]['transl'] = smpl_trans

    feats = encoder.encode_humanml3d(inputs)
    feats_arr.append(feats)
    

save_feats = torch.cat(feats_arr, dim=0).cpu()
torch.save(save_feats, save_path)
print(f"text-to-motion eatures saved at {save_path}")
