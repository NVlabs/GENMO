import torch
from torch.cuda.amp import autocast

import gem.utils.matrix as matrix
from gem.network.endecoder import EnDecoder
from gem.utils.ik.ccd_ik import CCD_IK
from gem.utils.net_utils import gaussian_smooth
from gem.utils.rotation_conversions import (
    matrix_to_axis_angle,
)

# SMPL joint IDs: [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
SMPL_STATIC_JOINT_IDS = [7, 10, 8, 11, 20, 21]

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

SMPL_LEFT_LEG_CHAIN = [0, 1, 4, 7, 10]
SMPL_RIGHT_LEG_CHAIN = [0, 2, 5, 8, 11]
SMPL_LEFT_HAND_CHAIN = [9, 13, 16, 18, 20]
SMPL_RIGHT_HAND_CHAIN = [9, 14, 17, 19, 21]

SMPL_LEFT_FOOT_ID = [7, 10]
SMPL_RIGHT_FOOT_ID = [8, 11]
SMPL_LEFT_WRIST_ID = [20]
SMPL_RIGHT_WRIST_ID = [21]


@autocast(enabled=False)
def pp_static_joint(outputs, endecoder: EnDecoder, smpl_key="pred_body_params_global"):
    # Global FK
    pred_w_j3d = endecoder.fk_v2(**outputs[smpl_key])
    joint_ids = SMPL_STATIC_JOINT_IDS

    pred_j3d_static = pred_w_j3d.clone()[:, :, joint_ids]  # (B, L, J, 3)

    # Update overall movement with static info, and make displacement ~[0,0,0]
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J, 3)

    static_conf_logits = outputs["static_conf_logits"][:, :-1].clone()
    static_label_ = static_conf_logits > 0  # (B, L-1, J) # avoid non-contact frame
    static_conf_logits = static_conf_logits.float() - (
        ~static_label_ * 1e6
    )  # fp16 cannot go through softmax
    is_static = static_label_.sum(dim=-1) > 0  # (B, L-1)

    pred_disp = pred_j_disp * static_conf_logits[..., None].softmax(dim=-2)  # (B, L-1, J, 3)
    pred_disp = pred_disp * is_static[..., None, None]  # (B, L-1, J, 3)
    pred_disp = pred_disp.sum(-2)  # (B, L-1, 3)

    # Overwrite results (vectorized)
    pred_w_transl = outputs[smpl_key]["transl"].clone()  # (B, L, 3)
    pred_w_disp = pred_w_transl[:, 1:] - pred_w_transl[:, :-1]  # (B, L-1, 3)
    pred_w_disp_new = pred_w_disp - pred_disp
    post_w_transl = torch.cumsum(torch.cat([pred_w_transl[:, :1], pred_w_disp_new], dim=1), dim=1)
    post_w_transl[..., 0] = gaussian_smooth(post_w_transl[..., 0], dim=-1)
    post_w_transl[..., 2] = gaussian_smooth(post_w_transl[..., 2], dim=-1)

    # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
    post_w_j3d = pred_w_j3d - pred_w_transl.unsqueeze(-2) + post_w_transl.unsqueeze(-2)
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,)  Minimum y value
    post_w_transl[..., 1] -= ground_y

    return post_w_transl


@autocast(enabled=False)
def process_ik(outputs, endecoder, static_conf=None, smpl_key="pred_body_params_global"):
    if static_conf is None:
        static_conf = outputs["static_conf_logits"].sigmoid()  # (B, L, J)
    post_w_j3d, local_mat, post_w_mat = endecoder.fk_v2(**outputs[smpl_key], get_intermediate=True)

    joint_ids = SMPL_STATIC_JOINT_IDS
    parents = SMPL_PARENTS

    post_target_j3d = post_w_j3d.clone()
    for i in range(1, post_w_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = post_w_j3d[:, i, joint_ids]
        c_prev = static_conf[:, i - 1, :, None]
        post_target_j3d[:, i, joint_ids] = prev * c_prev + this * (1 - c_prev)

    # IK
    global_rot = matrix.get_rotation(post_w_mat)

    def ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        IK_solver = CCD_IK(
            local_mat,
            parents,
            target_ind,
            target_pos,
            target_rot,
            kinematic_chain=chain,
            max_iter=2,
        )

        chain_local_mat = IK_solver.solve()
        chain_rotmat = matrix.get_rotation(chain_local_mat)
        local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]  # (B, L, J, 3, 3)
        return local_mat

    local_mat = ik(
        local_mat,
        post_target_j3d[:, :, SMPL_LEFT_FOOT_ID],
        global_rot[:, :, SMPL_LEFT_FOOT_ID],
        [3],
        SMPL_LEFT_LEG_CHAIN,
    )
    local_mat = ik(
        local_mat,
        post_target_j3d[:, :, SMPL_RIGHT_FOOT_ID],
        global_rot[:, :, SMPL_RIGHT_FOOT_ID],
        [3],
        SMPL_RIGHT_LEG_CHAIN,
    )
    local_mat = ik(
        local_mat,
        post_target_j3d[:, :, SMPL_LEFT_WRIST_ID],
        global_rot[:, :, SMPL_LEFT_WRIST_ID],
        [4],
        SMPL_LEFT_HAND_CHAIN,
    )
    local_mat = ik(
        local_mat,
        post_target_j3d[:, :, SMPL_RIGHT_WRIST_ID],
        global_rot[:, :, SMPL_RIGHT_WRIST_ID],
        [4],
        SMPL_RIGHT_HAND_CHAIN,
    )

    body_pose = matrix_to_axis_angle(matrix.get_rotation(local_mat[:, :, 1:]))  # (B, L, J-1, 3, 3)
    body_pose = body_pose.flatten(2)  # (B, L, (J-1)*3)

    return body_pose
