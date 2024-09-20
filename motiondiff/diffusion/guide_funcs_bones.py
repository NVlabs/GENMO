
from abc import ABC, abstractmethod

import torch
import numpy as np

from motiondiff.models.common.bones import BONES_BONE_ORDER_NAMES

NUM_JOINTS = len(BONES_BONE_ORDER_NAMES)
NUM_BODY_JOINTS = NUM_JOINTS-1
ROOT_NAME = 'Hips'

class GuideFunc(ABC):
    '''
    Abstract class for a test-time guidance function.
    '''
    guide_type = 'abstract'

    def __init__(self, weight):
        '''
        weight will always be passed in.
        Can do any initial setup here.
        '''
        self.weight = weight
    
    @abstractmethod
    def loss(self, pred, tgt):
        '''
        Compute the loss value used for guidance.
        Should return a dict of losses and a separate dict of corresponding loss weights.
        When implemented, this should only take in what's necessary from pred and tgt (e.g. local joint positions or global positions), 
            not the entire feature vector.
        '''
        loss_dict, weight_dict = None, None
        return loss_dict, weight_dict

    @abstractmethod
    def eval(self, pred, tgt):
        '''
        Evaluate how well the guidance was followed given a final prediction result and the target.
        When implemented, this should only take in what's necessary from pred and tgt (e.g. local joint positions or global positions), 
            not the entire feature vector.
        '''
        pass

#
# Base guidance functions
#

class FootSkate(GuideFunc):
    '''
    Encourage no foot skating with velocity penalty on feet.
    '''
    guide_type = 'foot_skate'

    def __init__(self, weight, exp_weighting=False, exp_alpha=0.01):
        self.weight = weight
        self.exp_weighting = exp_weighting
        self.exp_alpha = exp_alpha

        foot_joint_names = ['LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase']
        self.foot_joint_idx = [BONES_BONE_ORDER_NAMES.index(joint_name) for joint_name in foot_joint_names]

    def compute_foot_vel(self, global_pred_pose):
        # compute velocities of global foot joints
        feet_pos = global_pred_pose[:,:,:,self.foot_joint_idx] # [B, 1, nframes, 4, 3]
        foot_vel = torch.sum((feet_pos[:,:,1:] - feet_pos[:,:,:-1])**2, dim=-1) # [B, 1, nframes-1, 4]
        return foot_vel

    def loss(self, global_pred_pose, foot_contacts, t):
        '''
        - global_pred_pose : global joint positions [batch, 1, nframes, NUM_JOINTS, 3]
        - foot_contacts : predicted foot contacts [batch, 1, nframes, 4] where 4 is ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")
        - t : current diffusion timestep
        '''
        # velocities should be near 0 when confident of foot contact
        foot_vel = self.compute_foot_vel(global_pred_pose)
        loss = foot_vel * foot_contacts[:,:,:-1] # [B, 1, nframes-1, 4]

        if self.exp_weighting:
            # more important later in denoising (near step 0)
            weight = torch.exp(-self.exp_alpha * t[:,None,None,None])
            loss *= weight

        return {'foot_skate' : loss}, {'foot_skate' : self.weight}

    def eval(self, global_pred_pose, foot_contacts):
        '''
        - global_pred_pose : global joint positions [batch, 1, nframes, NUM_JOINTS, 3]
        - foot_contacts : predicted foot contacts [batch, 1, nframes, 4] where 4 is ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")
        - t : current diffusion timestep
        '''
        foot_vel = self.compute_foot_vel(global_pred_pose)
        # vel is sum of squares already
        foot_vel = torch.sqrt(foot_vel) # [B, 1, nframes-1, 4]
        # contacts are direct output of model, threshold to get binary
        foot_contacts = foot_contacts[:,:,:-1] > 0.5 # [batch, 1, nframes-1, 4]

        vel_err = foot_vel * foot_contacts
        vel_err = torch.sum(vel_err, (1,2,3)) / torch.sum(foot_contacts, (1,2,3)) # mean over contacting frames

        return {'foot_skate_vel_err' : vel_err}
    
class KeyframeBase(GuideFunc):
    '''
    Abstract base class for a guide based on specific keyframe timings.
    '''
    def __init__(self, weight, keyframe_inds='all', keyframe_ratio=True):
        self.weight = weight
        self.keyframe_inds_in = keyframe_inds
        self.keyframe_ratio = keyframe_ratio
        self.keyframe_inds = None
    
    def init_keyframe_inds(self, batch_size, pred_nframes, tgt_data_len, device):
        '''
        Compute which frames are the desired keyframes for each element of the batch, potentially based on the tgt data sequence length.
        - batch_size : of data that loss will be computed on
        - pred_nframes : seq len of data that loss will be computed on
        - tgt_data_len : seq len for each sequence in the GT data that creates the tgt for which loss will be computed on
        '''
        self.tgt_data_len = tgt_data_len
        self.keyframe_inds = self.keyframe_inds_in
        if self.keyframe_inds != 'all':
            self.keyframe_inds = torch.tensor([self.keyframe_inds]).expand((batch_size, -1)).to(device)
            if self.keyframe_ratio:
                data_len = torch.ones((batch_size, 1))*pred_nframes if tgt_data_len is None else tgt_data_len.unsqueeze(1)
                self.keyframe_inds = self.keyframe_inds * (data_len.to(device) - 1)
            self.keyframe_inds = self.keyframe_inds.round().to(torch.long) # [B, num_keys]

class GlobalJointPos(KeyframeBase):
    '''
    Encourage global joint positions to be at a specific global location at a specific timestep.
    '''
    guide_type = 'global_joint_pos'

    def __init__(self, weight, keyframe_inds='all', keyframe_ratio=True, joint_names=None, use_root_2d=False):
        super().__init__(weight, keyframe_inds, keyframe_ratio)
        self.use_root_2d = use_root_2d

        if joint_names == 'all':
            self.joint_names = set(BONES_BONE_ORDER_NAMES)
        else:
            self.joint_names = set(joint_names)

        self.global_joint_inds = [BONES_BONE_ORDER_NAMES.index(joint_name) for joint_name in self.joint_names]

    def init_keyframe_inds(self, batch_size, pred_nframes, tgt_data_len, device):
        super().init_keyframe_inds(batch_size, pred_nframes, tgt_data_len, device)
        if self.keyframe_inds != 'all':
            nkeys = self.keyframe_inds.size(1)
            self.keyframe_inds = self.keyframe_inds[:,None,:,None, None].expand(batch_size, 1, nkeys, NUM_JOINTS, 3) # expand to same size as poses 

    def compute_keyframe_diff(self, global_pred_pose, global_target_pose):
        # compare all global joint positions at each desired keyframe
        if self.keyframe_inds != 'all': 
            global_pred_keyframes = torch.gather(global_pred_pose, 2, self.keyframe_inds)
            global_tgt_keyframes = torch.gather(global_target_pose, 2, self.keyframe_inds)
        else:
            # will compute error of full sequence and then chop off what we don't need
            global_pred_keyframes = global_pred_pose
            global_tgt_keyframes = global_target_pose

        if self.use_root_2d and ROOT_NAME in self.joint_names:
            # zero out y (up) dim
            root_idx = BONES_BONE_ORDER_NAMES.index(ROOT_NAME)
            global_pred_keyframes[..., root_idx, 1] = 0.0
            global_tgt_keyframes[..., root_idx, 1] = 0.0

        # diff calculation
        global_pose_diff = global_pred_keyframes[:,:,:,self.global_joint_inds] - global_tgt_keyframes[:,:,:,self.global_joint_inds]
        return global_pose_diff

    def loss(self, global_pred_pose, global_target_pose):
        '''
        - global_pred_pose : global joint positions [batch, 1, nframes, NUM_JOINTS, 3]
        - global_target_pose : global joint positions [batch, 1, nframes, NUM_JOINTS, 3]
        '''
        assert self.keyframe_inds is not None, 'Must call init_keyframe_inds before computing loss!'
        global_pose_diff = self.compute_keyframe_diff(global_pred_pose, global_target_pose)   
        global_loss = torch.mean(global_pose_diff**2, dim=-1) # [B, 1, num_keys, num_joints]

        if self.keyframe_inds == 'all':
            # extract only the valid frames based on GT sequence length
            global_loss = torch.cat([global_loss[bi, :, :self.tgt_data_len[bi]].flatten() for bi in range(global_pred_pose.size(0))])

        return {'global_joint_pos' : global_loss}, {'global_joint_pos' : self.weight}

    def eval(self, global_pred_pose, global_target_pose):
        assert self.keyframe_inds is not None, 'Must call init_keyframe_inds before computing eval!'
        global_pose_diff = self.compute_keyframe_diff(global_pred_pose, global_target_pose)   
        global_joints_err = torch.norm(global_pose_diff, dim=-1)

        if self.keyframe_inds == 'all':
            global_joints_err = torch.stack([torch.mean(global_joints_err[bi, :, :self.tgt_data_len[bi]]) for bi in range(global_pred_pose.size(0))])
        else:
            global_joints_err = torch.mean(global_joints_err, (1, 2, 3))

        return {'global_joint_pos_err' : global_joints_err}

class GlobalRootOrient(KeyframeBase):
    '''
    Encourage root joint to be at a specific orientation at a specific timestep.
    '''
    guide_type = 'global_root_orient'

    def __init__(self, weight, keyframe_inds='all', keyframe_ratio=True):
        super().__init__(weight, keyframe_inds, keyframe_ratio)

    def init_keyframe_inds(self, batch_size, pred_nframes, tgt_data_len, device):
        super().init_keyframe_inds(batch_size, pred_nframes, tgt_data_len, device)
        if self.keyframe_inds != 'all':
            self.keyframe_inds = self.keyframe_inds[:,None,:,None].expand(batch_size, 1, self.keyframe_inds.size(1), 4) # expand to same size as rot quats

    def compute_root_orient_diff(self, global_pred_rot_quat, global_target_rot_quat):
        if self.keyframe_inds != 'all':
            pred_rot_keyframe = torch.gather(global_pred_rot_quat, 2, self.keyframe_inds)
            tgt_rot_keyframe = torch.gather(global_target_rot_quat, 2, self.keyframe_inds)
        else:
            pred_rot_keyframe = global_pred_rot_quat
            tgt_rot_keyframe = global_target_rot_quat

        global_rot_diff = (pred_rot_keyframe - tgt_rot_keyframe)
        return global_rot_diff

    def loss(self, global_pred_rot_quat, global_target_rot_quat):
        '''
        - global_pred_rot_quat : global rotation quaternion of root [batch, 1, nframes, 4]
        - global_target_rot_quat :  global rotation quaternion of root [batch, 1, nframes, 4]
        '''
        assert self.keyframe_inds is not None, 'Must call init_keyframe_inds before computing loss!'
        global_rot_diff = self.compute_root_orient_diff(global_pred_rot_quat, global_target_rot_quat)
        global_rot_loss = torch.mean(global_rot_diff**2, dim=-1, keepdim=True) # [B, 1, num_keys, 1]

        if self.keyframe_inds == 'all':
            # extract only the valid frames based on GT sequence length
            global_rot_loss = torch.cat([global_rot_loss[bi, :, :self.tgt_data_len[bi]].flatten() for bi in range(global_pred_rot_quat.size(0))])

        return {'global_root_orient' : global_rot_loss}, {'global_root_orient' : self.weight}

    def eval(self, global_pred_rot_quat, global_target_rot_quat):
        assert self.keyframe_inds is not None, 'Must call init_keyframe_inds before computing eval!'
        global_rot_diff = self.compute_root_orient_diff(global_pred_rot_quat, global_target_rot_quat)
        # only care about the cos and sin components
        global_rot_diff = global_rot_diff[:,:,:,[0,2]] # [batch, 1, num_keys, 2]
        err_vec = torch.ones_like(global_rot_diff)
        err_vec[:,:,:,1] = 0.0
        err_vec = err_vec + global_rot_diff
        ang_err = torch.abs(torch.rad2deg(torch.atan2(err_vec[:,:,:,1], err_vec[:,:,:,0])))
        if self.keyframe_inds == 'all':
            ang_err = torch.stack([torch.mean(ang_err[bi, :, :self.tgt_data_len[bi]]) for bi in range(global_pred_rot_quat.size(0))])
        else:
            ang_err = torch.mean(ang_err, (1,2))
        
        return {'global_root_ang_err' : ang_err}


class LocalJointPos(KeyframeBase):
    '''
    Encourage local joint positions (i.e., in the frame of the root) to be at a specific location at a specific timestep.
    '''
    guide_type = 'local_joint_pos'

    def __init__(self, weight, keyframe_inds='all', keyframe_ratio=True, joint_names=None):
        super().__init__(weight, keyframe_inds, keyframe_ratio)
        if joint_names == 'all':
            self.joint_names = set(BONES_BONE_ORDER_NAMES[1:]) # leave out root
        else:
            self.joint_names = set(joint_names)
        assert ROOT_NAME not in self.joint_names, 'Cannot apply a local loss to the root (it is at the origin)'

        self.local_joint_inds = [BONES_BONE_ORDER_NAMES.index(joint_name)-1 for joint_name in self.joint_names] # the root is not considered here so subtract 1
        self.local_joint_inds = np.concatenate([np.arange(idx*3, idx*3 + 3) for idx in self.local_joint_inds])

    def init_keyframe_inds(self, batch_size, pred_nframes, tgt_data_len, device):
        super().init_keyframe_inds(batch_size, pred_nframes, tgt_data_len, device)
        self.og_keyframe_inds = self.keyframe_inds # save for computing local_in_global eval later
        if self.keyframe_inds != 'all':
            nkeys = self.keyframe_inds.size(1)
            self.keyframe_inds = self.keyframe_inds[:,None,:,None].expand(batch_size, 1, nkeys, NUM_BODY_JOINTS*3) # expand to same size as local poses

    def compute_keyframe_diff(self, local_pred_pose, local_target_pose):
        # pull out only keyframe poses
        if self.keyframe_inds != 'all':
            # compare all global joint positions at each desired keyframe
            local_pred_keyframes = torch.gather(local_pred_pose, 2, self.keyframe_inds)  # [batch, 1, nkeys, NUM_BODY_JOINTS*3]
            local_tgt_keyframes = torch.gather(local_target_pose, 2, self.keyframe_inds) # [batch, 1, nkeys, NUM_BODY_JOINTS*3]
        else:
            # use all timesteps for now and will chop off invalid frames (past GT length) later
            local_pred_keyframes = local_pred_pose
            local_tgt_keyframes = local_target_pose

        # pull out desired joint positions
        local_pose_diff = local_pred_keyframes[:,:,:,self.local_joint_inds] - local_tgt_keyframes[:,:,:,self.local_joint_inds] # [B, 1, nkeys, njoint_feats]

        return local_pose_diff

    def loss(self, local_pred_pose, local_target_pose):
        '''
        - local_pred_pose : local joint positions excluding the root [batch, 1, nframes, NUM_BODY_JOINTS*3]
        - local_target_pose : local joint positions excluding the root [batch, 1, nframes, NUM_BODY_JOINTS*3]
        '''
        assert self.keyframe_inds is not None, 'Must call init_keyframe_inds before computing loss!'
        local_pose_diff = self.compute_keyframe_diff(local_pred_pose, local_target_pose)
        local_loss = local_pose_diff**2

        if self.keyframe_inds == 'all':
            # extract only the valid frames based on GT sequence length
            local_loss = torch.cat([local_loss[bi, :, :self.tgt_data_len[bi]].flatten() for bi in range(local_pred_pose.size(0))])

        return {'local_joint_pos' : local_loss}, {'local_joint_pos' : self.weight}

    def eval(self, local_pred_pose=None, local_target_pose=None, 
                   global_pred_pose=None, global_target_pose=None):
        '''
        If local_pred_pose and local_target_pose given as input, computes joint position error in local frame.

        If global_pred_pose and global_target_pose are additionally given as input, will also compute the error 
        of the global joint positions (i.e. local_in_global error).
        '''
        assert self.keyframe_inds is not None, 'Must call init_keyframe_inds before computing eval!'
        eval_out = dict()
        if local_pred_pose is not None and local_target_pose is not None:
            # local_pose_diff is [batch, 1, nkeys, njoint_feats]
            local_pose_diff = self.compute_keyframe_diff(local_pred_pose, local_target_pose)
            njoints = NUM_BODY_JOINTS if self.joint_names == 'all' else len(self.joint_names)
            local_pose_diff = local_pose_diff.reshape((local_pred_pose.size(0), 1, -1, njoints, 3)) # [batch, 1, nkeys, njoints, 3]
            local_joints_err = torch.norm(local_pose_diff, dim=-1) # [batch, 1, num_keys, num_joints]
            if self.keyframe_inds == 'all':
                local_joints_err = torch.stack([torch.mean(local_joints_err[bi, :, :self.tgt_data_len[bi]]) for bi in range(local_pred_pose.size(0))])
            else:
                local_joints_err = torch.mean(local_joints_err, (1,2,3)) # [batch]

            eval_out['local_joint_pos'] = local_joints_err

        # local_in_global error if glob info given
        if global_pred_pose is not None and global_target_pose is not None:
            # global joint positions [batch, 1, nframes, NUM_JOINTS, 3] 
            global_keyframe_inds = self.og_keyframe_inds
            if global_keyframe_inds != 'all':
                nkeys = global_keyframe_inds.size(1)
                global_keyframe_inds = global_keyframe_inds[:,None,:,None, None].expand(global_pred_pose.size(0), 1, nkeys, NUM_JOINTS, 3) # expand to same size as global poses
            
            # compare all global joint positions at each desired keyframe
            if global_keyframe_inds != 'all': 
                global_pred_keyframes = torch.gather(global_pred_pose, 2, global_keyframe_inds)
                global_tgt_keyframes = torch.gather(global_target_pose, 2, global_keyframe_inds)
            else:
                # will compute error of full sequence and then chop off what we don't need
                global_pred_keyframes = global_pred_pose
                global_tgt_keyframes = global_target_pose

            # diff calculation
            global_joint_inds = [BONES_BONE_ORDER_NAMES.index(joint_name) for joint_name in self.joint_names]
            local_in_glob_diff = global_pred_keyframes[:,:,:,global_joint_inds] - global_tgt_keyframes[:,:,:,global_joint_inds]

            # local_in_glob_diff is  [B, 1, nkeys, njoints, 3]
            local_in_glob_joints_err = torch.norm(local_in_glob_diff, dim=-1)
            if global_keyframe_inds == 'all':
                local_in_glob_joints_err = torch.stack([torch.mean(local_in_glob_joints_err[bi, :, :self.tgt_data_len[bi]]) for bi in range(global_pred_pose.size(0))])
            else:
                local_in_glob_joints_err = torch.mean(local_in_glob_joints_err, (1, 2, 3))

            eval_out['local_in_global_joint_pos_err'] = local_in_glob_joints_err            

        return eval_out
    

class LocalJointPosFromRot(LocalJointPos):
    '''
    Encourage local joint positions (i.e., in the frame of the root) to be at a specific location at a specific timestep.
    This assumes joint pos inputs are a result of FK. It is simply a convenience wrapper around LocalJointPos which is the exact same
    '''
    guide_type = 'local_joint_pos_from_rot'

    def __init__(self, weight, keyframe_inds='all', keyframe_ratio=True, joint_names=None):
        super().__init__(weight, keyframe_inds, keyframe_ratio, joint_names)

#
# Combo guidance functions - use several base guidance functions in tandem
#
    

# TODO additional combo functions like root trajectory. Basically any guidance that would have a single mask associated with it.

class Keyframes(KeyframeBase):
    '''
    A single guidance function that combines 3 different guidance at the same time:
        global joint position, local joint position, global root orientation. 
    '''
    guide_type = 'keyframes'

    def __init__(self, weight, keyframe_inds='all', unknown_time=False, global_joint_names=None, 
                        use_root_orient=True, use_root_2d=False, local_joint_names=None, exp_upweight=False, 
                        exp_start_t=500, exp_alpha=0.01, keyframe_ratio=True):
        super().__init__(weight, keyframe_inds, keyframe_ratio)
        self.unknown_time = unknown_time
        self.global_joint_names = global_joint_names
        self.use_root_orient = use_root_orient
        self.use_root_2d = use_root_2d
        self.local_joint_names = local_joint_names
        self.exp_upweight = exp_upweight
        self.exp_start_t = exp_start_t
        self.exp_alpha = exp_alpha

        if self.unknown_time:
            raise NotImplementedError('Unknown time not currently supported!')
        if self.exp_upweight:
            raise NotImplementedError('Exponential up-weighting not currently supported!')

        # init other guides
        # TODO we downweight global root weights here to be more similar to old implementation so can re-use configs
        #       -- this is a hack, eventually should remove and add separate loss weights for root pos/orients instead
        global_pos_weight = self.weight[0] * (len(self.global_joint_names) / (len(self.global_joint_names)+1)) if self.use_root_orient else self.weight[0]
        self.global_pos_guide = GlobalJointPos(global_pos_weight, self.keyframe_inds_in, self.keyframe_ratio, self.global_joint_names, self.use_root_2d)
        if self.use_root_orient:
            global_rrot_weight = self.weight[0] / (len(self.global_joint_names)+1)
            self.global_rrot_guide = GlobalRootOrient(global_rrot_weight, self.keyframe_inds_in, self.keyframe_ratio)
        else:
            self.global_rrot_guide = None
        if self.local_joint_names is not None:
            self.local_pos_guide = LocalJointPos(self.weight[1], self.keyframe_inds_in, self.keyframe_ratio, self.local_joint_names) 
        else:
            self.local_pos_guide = None

    def init_keyframe_inds(self, batch_size, pred_nframes, tgt_data_len, device):
        super().init_keyframe_inds(batch_size, pred_nframes, tgt_data_len, device)
        self.global_pos_guide.init_keyframe_inds(batch_size, pred_nframes, tgt_data_len, device)
        if self.global_rrot_guide is not None:
            self.global_rrot_guide.init_keyframe_inds(batch_size, pred_nframes, tgt_data_len, device)
        if self.local_pos_guide is not None:
            self.local_pos_guide.init_keyframe_inds(batch_size, pred_nframes, tgt_data_len, device)

    def loss(self, global_pred_pose, global_target_pose, global_pred_rot_quat, global_target_rot_quat, local_pred_pose, local_target_pose):
        losses, weights = self.global_pos_guide.loss(global_pred_pose, global_target_pose)
        if self.global_rrot_guide is not None:
            global_root_rot_loss, global_root_rot_weight = self.global_rrot_guide.loss(global_pred_rot_quat, global_target_rot_quat)
            losses.update(global_root_rot_loss)
            weights.update(global_root_rot_weight)
        if self.local_pos_guide is not None:
            local_pos_loss, local_pos_weight = self.local_pos_guide.loss(local_pred_pose, local_target_pose)
            losses.update(local_pos_loss)
            weights.update(local_pos_weight)

        return losses, weights

    def eval(self, global_pred_pose, global_target_pose, 
                    global_pred_rot_quat=None, global_target_rot_quat=None,
                    local_pred_pose=None, local_target_pose=None):
        # root pos
        eval_dict = self.global_pos_guide.eval(global_pred_pose, global_target_pose)
        # root rotation
        if self.global_rrot_guide is not None and global_pred_rot_quat is not None and global_target_rot_quat is not None:
            eval_dict.update(self.global_rrot_guide.eval(global_pred_rot_quat, global_target_rot_quat))
        # local joint pos
        if self.local_pos_guide is not None:
            eval_dict.update(self.local_pos_guide.eval(local_pred_pose, local_target_pose, global_pred_pose, global_target_pose))

        return eval_dict


GUIDE_CLS = [FootSkate, Keyframes, GlobalJointPos, GlobalRootOrient, LocalJointPos, LocalJointPosFromRot]
GUIDE_OBJECTIVES = {guide_cls.guide_type : guide_cls for guide_cls in GUIDE_CLS}
