import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel:

    def __init__(self, model):
        self.model = model  # model is the actual model to run
        # pointers to inner model
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode

    def __call__(self, x, timesteps, y=None, motion_mask=None, observed_motion=None, rm_text_flag=None, rm_kpt_flag=None,
                 guidance_only=False,
                 global_motion=None, global_joint_mask=None, global_joint_func=None, 
                 unknownt_motion_mask=None, unknownt_observed_motion=None, **kwargs):
        cond_mode = self.model.cond_mode
        assert set(cond_mode.split(",")).issubset(
            {
                "text",
                "action",
                "kpt2d+cam_angvel",
                "cam_vel",
                "cam2world",
                "camplucker",
                "local_smplfeat",
                "global_smplfeat",
                "plucker_kpt",
                "kpt2d_cam_vel",
                "kpt2d_cam2world",
                "global_kpt3d",
                "motion_v10",
            }
        ), cond_mode

        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True

        out = self.model(
            x, timesteps, y, motion_mask=motion_mask, observed_motion=observed_motion, rm_text_flag=rm_text_flag, 
            rm_kpt_flag=rm_kpt_flag,
            global_motion=global_motion, global_joint_mask=global_joint_mask, global_joint_func=global_joint_func, 
            unknownt_motion_mask=unknownt_motion_mask, unknownt_observed_motion=unknownt_observed_motion, **kwargs)
        if guidance_only:
            return out
        out_uncond = self.model(
            x, timesteps, y_uncond, motion_mask=motion_mask, observed_motion=observed_motion, 
            global_motion=global_motion, global_joint_mask=global_joint_mask, global_joint_func=global_joint_func, 
            unknownt_motion_mask=unknownt_motion_mask, unknownt_observed_motion=unknownt_observed_motion, **kwargs)
        return out_uncond + (y['scale'][0].view(-1, 1, 1, 1) * (out - out_uncond))

    def parameters(self):
        return self.model.parameters()
    
    def named_parameters(self):
        return self.model.named_parameters()