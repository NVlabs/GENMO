# This code is based on git@github.com:zju3dv/GVHMR.git

import numpy as np
import torch
import torch.nn as nn

import genmo.utils.matrix as matrix
from genmo.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from genmo.utils.torch_transform import (
    angle_axis_to_quaternion,
    get_y_heading_q,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quaternion_to_angle_axis,
)
from third_party.GVHMR.hmr4d.utils.geo.augment_noisy_pose import gaussian_augment
from third_party.GVHMR.hmr4d.utils.geo.hmr_global import (
    get_local_transl_vel,
    get_static_joint_mask,
    rollout_local_transl_vel,
)
from third_party.GVHMR.hmr4d.utils.smplx_utils import make_smplx

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
    ):
        super().__init__()

        if encode_type in ["gvhmr", "humanml3d"]:
            feature_arr = [encode_type]
            stats_arr = [stats_name]

        # Define feature dimensions as a class attribute
        self.FEATURE_DIMS = {
            "gvhmr": 151,
            "humanml3d": 143,
        }

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
                std = torch.clamp(std, 0.1, 1)

            self.stats_dict[feature] = {"mean": mean, "std": std}

        # Store feature configuration
        self.feature_arr = feature_arr
        self.stats_arr = stats_arr
        self.clip_std = clip_std

        # option
        self.noise_pose_k = noise_pose_k
        self.encode_type = encode_type
        self.obs_indices_dict = None

        # smpl
        self.smplx_model = make_smplx("supermotion_v437coco17")
        parents = self.smplx_model.parents[:22]
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

    def get_noisyobs(self, data, return_type="r6d"):
        """
        Noisy observation contains local pose with noise
        Args:
            data (dict):
                body_pose: (B, L, J*3) or (B, L, J, 3)
        Returns:
            noisy_bosy_pose: (B, L, J, 6) or (B, L, J, 3) or (B, L, 3, 3) depends on return_type
        """
        body_pose = data["body_pose"]  # (B, L, 63)
        B, L, _ = body_pose.shape
        body_pose = body_pose.reshape(B, L, -1, 3)

        # (B, L, J, C)
        return_mapping = {"R": 0, "r6d": 1, "aa": 2}
        return_id = return_mapping[return_type]
        noisy_bosy_pose = gaussian_augment(body_pose, self.noise_pose_k, to_R=True)[
            return_id
        ]
        return noisy_bosy_pose

    def normalize_body_pose_r6d(self, body_pose_r6d):
        """body_pose_r6d: (B, L, {J*6}/{J, 6}) ->  (B, L, J*6)"""
        B, L = body_pose_r6d.shape[:2]
        body_pose_r6d = body_pose_r6d.reshape(B, L, -1)
        if (
            self.stats_dict[self.encode_type]["mean"].shape[-1] == 1
        ):  # no mean, std provided
            return body_pose_r6d
        body_pose_r6d = (
            body_pose_r6d - self.stats_dict["gvhmr"]["mean"]
        ) / self.stats_dict["gvhmr"]["std"]  # (B, L, C)
        return body_pose_r6d

    def fk_v2(
        self, body_pose, betas, global_orient=None, transl=None, get_intermediate=False
    ):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)

        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat(
            [skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2
        )

        if transl is not None:
            local_skeleton[..., 0, :] += transl  # B, L, 22, 3

        mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
        fk_mat = matrix.forward_kinematics(mat, self.parents)  # B, L, 22, 4, 4
        joints = matrix.get_position(fk_mat)  # B, L, 22, 3
        if not get_intermediate:
            return joints
        else:
            return joints, mat, fk_mat

    def get_local_pos(self, betas):
        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat(
            [skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2
        )
        return local_skeleton

    def get_static_gt(self, inputs, vel_thr):
        joint_ids = [
            7,
            10,
            8,
            11,
            20,
            21,
        ]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
        gt_w_j3d = self.fk_v2(**inputs["smpl_params_w"])  # (B, L, J=22, 3)
        static_gt = get_static_joint_mask(
            gt_w_j3d, vel_thr=vel_thr, repeat_last=True
        )  # (B, L, J)
        static_gt = static_gt[:, :, joint_ids].float()  # (B, L, J')
        return static_gt

    def encode(self, inputs):
        """Composite encoder that combines multiple feature types"""
        encoded_features = []

        for feature in self.feature_arr:
            if feature == "gvhmr":
                encoded = self.encode_gvhmr(inputs)
            elif feature == "humanml3d":
                encoded = self.encode_humanml3d(inputs)
            encoded_features.append(encoded)

        # Concatenate all encoded features
        return torch.cat(encoded_features, dim=-1)

    def encode_humanml3d(self, inputs):
        """
        definition: {
                body_pose_r6d,  # (B, L, (J-1)*6) -> 0:126
                betas, # (B, L, 10) -> 126:136
                root_data,  # (B, L, 10) -> 136:143
            }
        """
        self.obs_indices_dict = {
            "body_pose": torch.arange(126),
            "betas": torch.arange(126, 136),
            "root_data": torch.arange(136, 143),
        }
        B, L = inputs["smpl_params_w"]["body_pose"].shape[:2]
        # cam
        smpl_params_w = inputs["smpl_params_w"]
        body_pose = smpl_params_w["body_pose"].reshape(B, L, 21, 3)
        body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).flatten(
            -2
        )
        betas = smpl_params_w["betas"]
        global_orient = smpl_params_w["global_orient"]
        trans = smpl_params_w["transl"].clone()

        root_quat = angle_axis_to_quaternion(global_orient)
        heading_quat = get_y_heading_q(root_quat)
        heading_quat_inv = quat_conjugate(heading_quat)
        root_quat_wo_heading = quat_mul(heading_quat_inv, root_quat)
        # root_quat_wo_heading = quaternion_to_cont6d(root_quat_wo_heading)
        root_quat_wo_heading = quaternion_to_angle_axis(root_quat_wo_heading)

        init_heading_quat_inv = heading_quat_inv[:, [0]].repeat(1, L, 1)

        """XZ at origin"""
        root_y = trans[..., [1]]
        root_pos_init = trans[:, [0]]
        root_pose_init_xz = root_pos_init * torch.tensor([1, 0, 1]).to(root_pos_init)
        trans = trans - root_pose_init_xz

        """All initially face Z+"""
        trans = quat_apply(init_heading_quat_inv, trans)
        heading_quat = quat_mul(
            heading_quat, init_heading_quat_inv
        )  # normalize heading coordiante, so the heading is 0 for the first frame
        heading_quat_inv = quat_conjugate(heading_quat)

        """Root Linear Velocity"""
        # (seq_len - 1, 3)
        velocity = trans[:, 1:] - trans[:, :-1]
        velocity = torch.cat([velocity, velocity[:, [-1]]], axis=1)
        #     print(r_rot.shape, velocity.shape)
        velocity = quat_apply(heading_quat_inv, velocity)
        l_velocity = velocity[..., [0, 2]]
        """Root Angular Velocity"""
        # (seq_len - 1, 4)
        r_angles = torch.arctan2(heading_quat[..., 2:3], heading_quat[..., :1]) * 2
        r_velocity = r_angles[:, 1:] - r_angles[:, :-1]
        r_velocity[r_velocity > np.pi] -= 2 * np.pi
        r_velocity[r_velocity < -np.pi] += 2 * np.pi
        r_velocity = torch.cat([r_velocity, r_velocity[:, [-1]]], axis=1)

        root_data = torch.cat(
            [r_velocity, l_velocity, root_y, root_quat_wo_heading], axis=-1
        )
        # 126 + 10 + 7 = 143d
        x = torch.cat([body_pose_r6d, betas, root_data], dim=-1)
        return self.normalize(x, "humanml3d")

    def encode_gvhmr(self, inputs):
        """
        definition: {
                body_pose_r6d,  # (B, L, (J-1)*6) -> 0:126
                betas, # (B, L, 10) -> 126:136
                global_orient_r6d,  # (B, L, 6) -> 136:142  incam
                global_orient_gv_r6d: # (B, L, 6) -> 142:148  gv
                local_transl_vel,  # (B, L, 3) -> 148:151, smpl-coord
            }
        """
        self.obs_indices_dict = {
            "body_pose": torch.arange(126),
            "betas": torch.arange(126, 136),
            "global_orient": torch.arange(136, 142),
            "global_orient_gv": torch.arange(142, 148),
            "local_transl_vel": torch.arange(148, 151),
        }

        B, L = inputs["smpl_params_c"]["body_pose"].shape[:2]
        # cam
        smpl_params_c = inputs["smpl_params_c"]
        body_pose = smpl_params_c["body_pose"].reshape(B, L, 21, 3)
        body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).flatten(
            -2
        )
        betas = smpl_params_c["betas"]
        global_orient_R = axis_angle_to_matrix(smpl_params_c["global_orient"])
        global_orient_r6d = matrix_to_rotation_6d(global_orient_R)

        # global
        R_c2gv = inputs["R_c2gv"]  # (B, L, 3, 3)
        global_orient_gv_r6d = matrix_to_rotation_6d(R_c2gv @ global_orient_R)

        # local_transl_vel
        smpl_params_w = inputs["smpl_params_w"]
        local_transl_vel = get_local_transl_vel(
            smpl_params_w["transl"], smpl_params_w["global_orient"]
        )
        if False:  # debug
            transl_recover = rollout_local_transl_vel(
                local_transl_vel,
                smpl_params_w["global_orient"],
                smpl_params_w["transl"][:, [0]],
            )
            print((transl_recover - smpl_params_w["transl"]).abs().max())

        # returns
        x = torch.cat(
            [
                body_pose_r6d,
                betas,
                global_orient_r6d,
                global_orient_gv_r6d,
                local_transl_vel,
            ],
            dim=-1,
        )
        return self.normalize(x, "gvhmr")

    def encode_translw(self, inputs):
        """
        definition: {
                body_pose_r6d,  # (B, L, (J-1)*6) -> 0:126
                betas, # (B, L, 10) -> 126:136
                global_orient_r6d,  # (B, L, 6) -> 136:142  incam
                global_orient_gv_r6d: # (B, L, 6) -> 142:148  gv
                local_transl_vel,  # (B, L, 3) -> 148:151, smpl-coord
            }
        """
        # local_transl_vel
        smpl_params_w = inputs["smpl_params_w"]
        local_transl_vel = get_local_transl_vel(
            smpl_params_w["transl"], smpl_params_w["global_orient"]
        )

        # returns
        x = local_transl_vel
        x_norm = (x - self.stats_dict["gvhmr"]["mean"][-3:]) / self.stats_dict["gvhmr"][
            "std"
        ][-3:]
        return x_norm

    def decode_translw(self, x_norm):
        return (
            x_norm * self.stats_dict["gvhmr"]["std"][-3:]
            + self.stats_dict["gvhmr"]["mean"][-3:]
        )

    def decode(self, x_norm):
        """Composite decoder that handles multiple feature types"""
        current_idx = 0
        decoded_outputs = {}

        for feature in self.feature_arr:
            feature_size = self.FEATURE_DIMS[feature]
            feature_norm = x_norm[..., current_idx : current_idx + feature_size]

            if feature == "gvhmr":
                decoded = self.decode_gvhmr(feature_norm)
            elif feature == "humanml3d":
                decoded = self.decode_humanml3d(feature_norm)

            decoded_outputs.update(decoded)
            current_idx += feature_size

        return decoded_outputs

    def decode_humanml3d(self, x_norm):
        """x_norm: (B, L, C)"""
        B, L, C = x_norm.shape
        x = self.denormalize(x_norm, "humanml3d")

        body_pose_r6d = x[:, :, :126]
        betas = x[:, :, 126:136]
        root_data = x[:, :, 136:143]

        body_pose = matrix_to_axis_angle(
            rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6))
        )
        body_pose = body_pose.flatten(-2)
        offset = self.smplx_model.get_skeleton(betas)[:, :, 0]

        rot_vel = root_data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(root_data.device)
        """Get Y-axis rotation from rotation velocity"""
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
        r_rot_quat = torch.zeros(root_data.shape[:-1] + (4,)).to(root_data)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang / 2)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang / 2)

        r_pos = torch.zeros(root_data.shape[:-1] + (3,)).to(root_data)
        r_pos[..., 1:, [0, 2]] = root_data[..., :-1, 1:3]
        """Add Y-axis rotation to root position"""
        r_pos = quat_apply(r_rot_quat, r_pos)
        r_pos = torch.cumsum(r_pos, dim=-2)
        r_pos[..., 1] = root_data[..., 3]
        # return r_rot_quat, r_pos, r_rot_ang
        # root_rot_wo_heading = rotation_6d_to_matrix(root_data[..., 4:])
        # root_rot_wo_heading = cont6d_to_matrix(root_data[..., 4:])
        # root_rot = quaternion_to_matrix(r_rot_quat) @ root_rot_wo_heading
        # global_orient_w = matrix_to_axis_angle(root_rot)
        global_orient_w = quaternion_to_angle_axis(
            quat_mul(r_rot_quat, angle_axis_to_quaternion(root_data[..., 4:]))
        )

        output = {
            "body_pose": body_pose,
            "betas": betas,
            "global_orient_w": global_orient_w,
            "transl_w": r_pos,
            "offset": offset,
        }

        return output

    def decode_gvhmr(self, x_norm):
        """x_norm: (B, L, C)"""
        B, L, C = x_norm.shape
        x = self.denormalize(x_norm, "gvhmr")

        body_pose_r6d = x[:, :, :126]
        betas = x[:, :, 126:136]
        global_orient_r6d = x[:, :, 136:142]
        global_orient_gv_r6d = x[:, :, 142:148]
        local_transl_vel = x[:, :, 148:151]

        body_pose = matrix_to_axis_angle(
            rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6))
        )
        body_pose = body_pose.flatten(-2)
        global_orient_c = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_r6d))
        global_orient_gv = matrix_to_axis_angle(
            rotation_6d_to_matrix(global_orient_gv_r6d)
        )

        offset = self.smplx_model.get_skeleton(betas)[:, :, 0]
        output = {
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": global_orient_c,
            "global_orient_gv": global_orient_gv,
            "local_transl_vel": local_transl_vel,
            "offset": offset,
        }

        return output

    def get_motion_dim(self):
        """Calculate total dimension based on enabled features"""
        return sum(self.FEATURE_DIMS[feature] for feature in self.feature_arr)

    def get_obs_indices(self, obs):
        return self.obs_indices_dict[obs]
