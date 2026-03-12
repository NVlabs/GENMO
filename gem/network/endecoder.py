# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

from gem.utils import matrix
from gem.utils.motion_utils import get_local_transl_vel, get_static_joint_mask
from gem.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from gem.utils.smplx_utils import make_smplx

from . import stats_compose


class EnDecoder(nn.Module):
    def __init__(
        self,
        stats_name="DEFAULT_01",
        encode_type="gvhmr",
        feature_arr=None,
        stats_arr=None,
        noise_pose_k=10,
        clip_std=False,
        feat_dim=None,
    ):
        super().__init__()

        if encode_type in ["gvhmr"]:
            feature_arr = [encode_type]
            stats_arr = [stats_name]

        # Define feature dimensions as a class attribute
        self.FEATURE_DIMS = {
            "gvhmr": 151,
        }
        if feat_dim is not None:
            self.FEATURE_DIMS[encode_type] = feat_dim

        # Store stats for each feature type
        self.stats_dict = {}

        for feature, stats_name in zip(feature_arr, stats_arr):
            stats = getattr(stats_compose, stats_name)
            mean = torch.tensor(stats["mean"]).float()
            std = torch.tensor(stats["std"]).float()

            feature_dim = self.FEATURE_DIMS[feature]
            if stats_name != "DEFAULT_01":
                assert mean.shape[-1] == feature_dim
                assert std.shape[-1] == feature_dim

            if clip_std:
                std[std < 1] = 1

            self.stats_dict[feature] = {"mean": mean, "std": std}

        # Store feature configuration
        self.feature_arr = feature_arr
        self.stats_arr = stats_arr
        self.clip_std = clip_std

        # option
        self.noise_pose_k = noise_pose_k
        self.encode_type = encode_type
        self.obs_indices_dict = None

        if encode_type == "gvhmr":
            self.smplx_model = make_smplx("supermotion_v437coco17")
            self.num_J = 22
            parents = self.smplx_model.parents[: self.num_J]
            self.register_buffer("parents_tensor", parents, False)
            self.parents = parents.tolist()

    def normalize(self, x, feature_type):
        """Normalize input using stats for specific feature type"""
        stats = self.stats_dict[feature_type]
        return (x - stats["mean"].to(x)) / stats["std"].to(x)

    def denormalize(self, x_norm, feature_type):
        """Denormalize input using stats for specific feature type"""
        stats = self.stats_dict[feature_type]
        return x_norm * stats["std"].to(x_norm) + stats["mean"].to(x_norm)

    def get_static_gt(self, inputs, vel_thr):
        joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
        gt_w_j3d = self.fk_v2(**inputs["smpl_params_w"])  # (B, L, J, 3)
        static_gt = get_static_joint_mask(gt_w_j3d, vel_thr=vel_thr, repeat_last=True)
        static_gt = static_gt[:, :, joint_ids].float()
        return static_gt

    def build_obs_indices_dict(self):
        """
        Initialize observation index mapping for decode-time use.
        This mirrors the legacy behavior where eval/demo could decode without
        a preceding encode() call.
        """
        for feature in self.feature_arr:
            if feature == "gvhmr":
                self.obs_indices_dict = {
                    "body_pose": (0, 126),
                    "betas": (126, 136),
                    "global_orient": (136, 142),
                    "global_orient_gv": (142, 148),
                    "local_transl_vel": (148, 151),
                }

    def encode(self, inputs):
        """Composite encoder that combines multiple feature types"""
        encoded_features = []

        for feature in self.feature_arr:
            if feature == "gvhmr":
                encoded = self.encode_gvhmr(inputs)
            encoded_features.append(encoded)

        return torch.cat(encoded_features, dim=-1)

    def encode_gvhmr(self, inputs):
        self.obs_indices_dict = {
            "body_pose": (0, 126),
            "betas": (126, 136),
            "global_orient": (136, 142),
            "global_orient_gv": (142, 148),
            "local_transl_vel": (148, 151),
        }
        B, L = inputs["smpl_params_c"]["body_pose"].shape[:2]
        smpl_params_c = inputs["smpl_params_c"]
        body_pose = smpl_params_c["body_pose"].reshape(B, L, 21, 3)
        body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).flatten(-2)
        betas = smpl_params_c["betas"]
        global_orient_R = axis_angle_to_matrix(smpl_params_c["global_orient"])
        global_orient_r6d = matrix_to_rotation_6d(global_orient_R)
        R_c2gv = inputs["R_c2gv"]
        global_orient_gv_r6d = matrix_to_rotation_6d(R_c2gv @ global_orient_R)
        smpl_params_w = inputs["smpl_params_w"]
        local_transl_vel = get_local_transl_vel(
            smpl_params_w["transl"], smpl_params_w["global_orient"]
        )
        x = torch.cat(
            [body_pose_r6d, betas, global_orient_r6d, global_orient_gv_r6d, local_transl_vel],
            dim=-1,
        )
        return self.normalize(x, "gvhmr")

    def decode(self, x_norm):
        """Composite decoder that handles multiple feature types"""
        current_idx = 0
        decoded_outputs = {}

        for feature in self.feature_arr:
            feature_size = self.FEATURE_DIMS[feature]
            feature_norm = x_norm[..., current_idx : current_idx + feature_size]

            if feature == "gvhmr":
                decoded = self.decode_gvhmr(feature_norm)

            decoded_outputs.update(decoded)
            current_idx += feature_size

        return decoded_outputs

    def decode_gvhmr(self, x_norm):
        B, L, C = x_norm.shape
        x = self.denormalize(x_norm, "gvhmr")
        body_pose_r6d = x[:, :, :126]
        betas = x[:, :, 126:136]
        global_orient_r6d = x[:, :, 136:142]
        global_orient_gv_r6d = x[:, :, 142:148]
        local_transl_vel = x[:, :, 148:151]
        body_pose = matrix_to_axis_angle(
            rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6))
        ).flatten(-2)
        global_orient_c = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_r6d))
        global_orient_gv = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_gv_r6d))
        offset = self.smplx_model.get_skeleton(betas)[:, :, 0]
        return {
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": global_orient_c,
            "global_orient_gv": global_orient_gv,
            "local_transl_vel": local_transl_vel,
            "offset": offset,
        }

    def fk_v2(
        self,
        body_pose,
        betas=None,
        global_orient=None,
        transl=None,
        get_intermediate=False,
        **kwargs,
    ):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
            transl: (B, L, 3)
            get_intermediate: if True, also return local transform matrices and FK matrices
        Returns:
            joints: (B, L, 22, 3)
            (optional) mat: (B, L, 22, 4, 4) local transform matrices
            (optional) fk_mat: (B, L, 22, 4, 4) global FK matrices
        """
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)
        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)
        if transl is not None:
            local_skeleton[..., 0, :] += transl
        mat = matrix.get_TRS(rotmat, local_skeleton)
        fk_mat = matrix.forward_kinematics(mat, self.parents)
        joints = matrix.get_position(fk_mat)
        if not get_intermediate:
            return joints
        else:
            return joints, mat, fk_mat

    def get_motion_dim(self):
        """Calculate total dimension based on enabled features"""
        return sum(self.FEATURE_DIMS[feature] for feature in self.feature_arr)

    def get_obs_indices(self, obs):
        return self.obs_indices_dict[obs]
