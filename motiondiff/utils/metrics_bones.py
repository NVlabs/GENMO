import numpy as np
import torch

from motiondiff.data_pipeline.bones.motion_process import (
    recover_from_ric,
    recover_from_ric_264,
    recover_from_ric_with_joint_rot,
)
from motiondiff.models.common.bones import BONES_BONE_ORDER_NAMES

# which motion reps are supported
MOTION_REPS = {"global_root_local_joints", "global_root_local_joints_root_rot"}
# which metrics are supported
METRICS = {"foot_skate"}

NUM_JOINTS = len(BONES_BONE_ORDER_NAMES)


class Metrics:
    def __init__(
        self,
        motion_mean,
        motion_std,
        motion_global_mean,
        motion_global_std,
        normalize_global_pos=True,
        pred_samples_rep="global_root_local_joints",
        metrics=["foot_skate"],
        neutral_joints=None,
        joint_parents=None,
    ):
        self.pred_samples_rep = pred_samples_rep
        assert self.pred_samples_rep in MOTION_REPS
        self.metrics = metrics
        for met in self.metrics:
            assert met in METRICS
        self.neutral_joints = neutral_joints
        self.joint_parents = joint_parents
        if self.pred_samples_rep == "global_root_local_joints_root_rot":
            assert self.neutral_joints is not None
            assert self.joint_parents is not None
            self.neutral_joints = self.neutral_joints.to(motion_mean.device)
            self.joint_parents = self.joint_parents.to(motion_mean.device)

        self.mean = motion_mean
        self.std = motion_std
        self.normalize_global_pos = normalize_global_pos
        self.global_mean = motion_global_mean
        self.global_std = motion_global_std

    def preprocess(self, samples):
        """
        - samples: [batch, 1, nframes, nfeat]
        """

        # unnormalize
        if self.pred_samples_rep == "orig":
            samples = samples * self.std + self.mean
        elif self.pred_samples_rep in {
            "global_root_local_joints",
            "global_root_local_joints_root_rot",
        }:
            # first 5d are 3d root pos and 2d root heading
            root, local_pose = samples[..., :5], samples[..., 5:]
            if self.normalize_global_pos:
                root = root * self.global_std + self.global_mean
            local_pose = local_pose * self.std[4:] + self.mean[4:]
            samples = torch.cat([root, local_pose], dim=-1)

        # compute joint and rotation information
        if self.pred_samples_rep in {
            "global_root_local_joints",
            "global_root_local_joints_root_rot",
        }:
            pred_r_pos, rot_cos_sin = samples[..., :3], samples[..., 3:5]
            r_rot_ang = torch.atan2(rot_cos_sin[..., [1]], rot_cos_sin[..., [0]])
            pred_r_rot_quat = torch.cat(
                [
                    torch.cos(r_rot_ang / 2),
                    torch.zeros_like(rot_cos_sin[..., [0]]),
                    torch.sin(r_rot_ang / 2),
                    torch.zeros_like(rot_cos_sin[..., [0]]),
                ],
                dim=-1,
            )
            if self.pred_samples_rep == "global_root_local_joints_root_rot":
                global_joint_pos, global_rot_quat = recover_from_ric_with_joint_rot(
                    samples,
                    self.neutral_joints,
                    self.joint_parents,
                    root_dim=5,
                    r_rot_quat=pred_r_rot_quat,
                    r_pos=pred_r_pos,
                    return_r_rot=True,
                )
            else:
                global_joint_pos, global_rot_quat = recover_from_ric_264(
                    samples, NUM_JOINTS, r_rot_quat=pred_r_rot_quat, r_pos=pred_r_pos
                )
        elif self.pred_samples_rep == "orig":
            global_joint_pos, global_rot_quat = recover_from_ric(
                samples, NUM_JOINTS, return_r_rot=True
            )

        return global_joint_pos, global_rot_quat

    def compute(self, pred_samples=None, global_joint_pos=None):
        """
        Compute the metrics specified by self.metrics.

        - pred_samples : [batch, nfeat, 1, nframes] full output from the model (in self.pred_samples_rep format)
        - global_joint_pos : [batch, nframes, NUM_JOINTS, 3] Optionally, global joint positions, unnormalized. If this is provided
                                                      along with pred_samples, will be overwritten by global joints computed
                                                      from pred_samples
        """
        global_rot_quat = None
        if pred_samples is not None:
            pred_samples = pred_samples.permute(
                0, 2, 3, 1
            )  # [batch, 1, nframes, nfeat]
            global_joint_pos, global_rot_quat = self.preprocess(pred_samples)
            global_joint_pos = global_joint_pos[:, 0]  # [batch, nframes, NUM_JOINTS, 3]
            global_rot_quat = global_rot_quat[:, 0]  # [batch, nframes, 4]

        eval_out = dict()
        # each metrics returns [batch]
        for met_name in self.metrics:
            if met_name == "foot_skate":
                fs_out = foot_skate(global_joint_pos)
                eval_out.update(fs_out)

        # TODO more metrics like FID and text alignment with TMR

        return eval_out


def foot_skate(global_joint_pos, height_thresh=0.05):
    """
    When toe joint is near the floor, compute velocity of the toes and ankles.
    - global_joint_pos : [batch, nframes, NUM_JOINTS, 3]
    """
    foot_joint_names = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]
    fid = [BONES_BONE_ORDER_NAMES.index(joint_name) for joint_name in foot_joint_names]
    feet_pos = global_joint_pos[:, :, fid]
    toe_pos = feet_pos[:, :, [1, 3]]

    # [batch, nframes, 2]
    toe_on_floor = toe_pos[..., 1] < height_thresh  # y-up
    toe_on_floor = torch.concatenate(
        [
            toe_on_floor[:, :, 0:1],
            toe_on_floor[:, :, 0:1],
            toe_on_floor[:, :, 1:2],
            toe_on_floor[:, :, 1:2],
        ],
        dim=-1,
    )[:, :-1]
    feet_vel = torch.norm(
        feet_pos[:, 1:] - feet_pos[:, :-1], dim=-1
    )  # [batch, nframes-1, 4]
    contact_feet_vel = feet_vel * toe_on_floor
    mean_vel = torch.sum(contact_feet_vel, (1, 2)) / (
        torch.sum(toe_on_floor, (1, 2)) + 1e-6
    )
    return {"foot_skate": mean_vel}


# TODO foot floating/penetration metrics, since skating will not detect if the foot is not near the floor
