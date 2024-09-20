import torch
import torch.nn.functional as F
import numpy as np
from motiondiff.utils.torch_utils import tensor_to
from motiondiff.data_pipeline.humanml.scripts.motion_process import (
    recover_root_rot_pos_264,
    recover_from_ric_264,
)
from motiondiff.models.common.smpl import SMPL_BONE_ORDER_NAMES

GUIDE_OBJECTIVES = {'keyframes', 'foot_skate'}
GUIDE_TYPES = {'soft'}
GRAD_THRESH_OPTIONS = {None, 'none', 'var', 'clip_norm', 'clip_value'}
PRED_THRESH_OPTIONS = {None, 'none', 'static', 'dynamic'}

SOFTMIN = False
SOFTMIN_TEMP = 1.0

DEBUG_PLOT_MIN_DIST_IDX = False

MOTION_REPS = {'full263', 'global_root_local_joints', 'global_root'}

class GuideFunctions():
    def __init__(self, cfg, model, diff_steps, tgt_data_len=None, pred_motion_rep='full263', tgt_motion_rep='full263'):
        '''
        Diffusion guidance.
        '''
        self.cfg = cfg
        self.diff_steps = diff_steps 
        self.tgt_data_len = tgt_data_len
        self.pred_motion_rep = pred_motion_rep
        assert self.pred_motion_rep in MOTION_REPS
        self.tgt_motion_rep = tgt_motion_rep
        assert self.pred_motion_rep in MOTION_REPS

        self.mean = model.humanml_mean.to(model.device)
        self.std = model.humanml_std.to(model.device)
        self.normalize_global_pos = model.normalize_global_pos
        self.global_mean = model.humanml_global_mean.to(model.device)
        self.global_std = model.humanml_global_std.to(model.device)
        self.model = model
        self.need_update_mask = False

        self.thresh_grad = cfg.get('thresh_grad', 'none')
        assert self.thresh_grad in GRAD_THRESH_OPTIONS, f"thresh_grad must be one of {GRAD_THRESH_OPTIONS}"
        self.thresh_grad_param = cfg.get('thresh_grad_param', None)

        self.thresh_pred = cfg.get('thresh_pred', 'none')
        assert self.thresh_pred in PRED_THRESH_OPTIONS, f"thresh_pred must be one of {PRED_THRESH_OPTIONS}"
        self.thresh_pred_param = cfg.get('thresh_pred_param', None)

        self.guide_type = self.cfg.type
        assert self.guide_type in GUIDE_TYPES
        # the rest are lists containing all guidance objectives
        guide_dict = self.cfg.objectives
        self.guide_names = [k for k in guide_dict.keys()]
        self.obj = [guide['objective'] for _, guide in guide_dict.items()]
        for guide_obj in self.obj:
            assert guide_obj in GUIDE_OBJECTIVES
        # the model masks to use for each objective
        self.mask = [guide.get('mask', None) for _, guide in guide_dict.items()]
        # the guidance weight for each objectives
        self.alpha = [[float(guide['alpha'])] if isinstance(guide['alpha'], float) or isinstance(guide['alpha'], int) \
                        else list(guide['alpha']) for _, guide in guide_dict.items()]
        # any additional params for each guidance function
        self.guide_kwargs = [guide['kwargs'] if 'kwargs' in guide else None for _, guide in guide_dict.items()]

        self.guidance_losses = {
            'keyframes' : self.unified_keyframes_loss,
            'foot_skate' : self.foot_skate_loss,
        }
        self.eval_funcs = {
            'keyframes' : self.unified_keyframes_eval,
            'foot_skate' : self.foot_skate_eval,
        }
        self.loss_names = {
            'keyframes' : ('global_keyframes', 'local_keyframes'),
            'foot_skate' : ('foot_skate',),
        }

        # NOTE: just for debugging
        self.unknown_min_dist_tracker = []

    #
    # Main guidance functions
    #

    def update_pred(self, pred, grad, var):
        '''
        Update pred based on guidance gradient while accounting for 
        any gradient clipping or prediction thresholding.
        '''
        # modifying gradient
        if self.thresh_grad == 'var':
            grad = grad * var
        elif self.thresh_grad == 'clip_norm':
            max_norm = self.thresh_grad_param
            grad_norm = torch.linalg.vector_norm(grad, ord=2, dim=(1,2,3))
            # print(grad_norm)
            clip_coef = max_norm / (grad_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            grad = grad * clip_coef_clamped[:,None,None,None]
        elif self.thresh_grad == 'clip_value':
            clip_val = self.thresh_grad_param
            grad = torch.clamp(grad, min=-clip_val, max=clip_val)

        # update prediction
        pred = pred - grad

        # threshold prediction
        # implemented based on imagen: https://arxiv.org/pdf/2205.11487.pdf
        if self.thresh_pred == 'static':
            clip_val = self.thresh_pred_param
            pred = torch.clamp(pred, min=-clip_val, max=clip_val)
        elif self.thresh_pred == 'dynamic':
            B = pred.size(0)
            perc, clip_val = self.thresh_pred_param
            s = torch.quantile(torch.abs(pred.reshape((B, -1))), perc, dim=1)
            # print(s)
            # for those motions with quantile over thresh, clamp and then scale so they are clip_val at most
            s = s[:,None,None,None].expand_as(pred)
            pred = torch.where(s > clip_val, 
                               torch.clamp(pred, min=-s, max=s) / ((1. / clip_val) * s + 1e-6), 
                               pred)

        return pred

    def guide(self, model_in, pred, target, model_variance, t, model_kwargs):
        # pred / target: [batch, nfeat, 1, seq_len]

        # don't use any guidance on final step
        cur_t = torch.max(t).item()
        is_last_step = cur_t == 0
        if is_last_step:
            return pred

        # compute loss
        with torch.enable_grad():
            # now permulted to: [batch, 1, seq_len, nfeat]
            pred_p = self.preprocess(pred.permute(0,2,3,1), motion_rep=self.pred_motion_rep)
            target_p = self.preprocess(target.permute(0,2,3,1), motion_rep=self.tgt_motion_rep)
            # create a continous representation for angle 
            pred_c = self.convert_to_264(pred_p, motion_rep=self.pred_motion_rep)
            target_c = self.convert_to_264(target_p, motion_rep=self.tgt_motion_rep)

            loss_tot = 0.0
            for cur_obj, cur_name, cur_alphas, cur_kwargs in zip(self.obj, self.guide_names, self.alpha, self.guide_kwargs):
                # every loss function assumes 264 dim rep in un-normalized space
                base_kwargs = {'pred' : pred_c, 'target' : target_c, 't' : t}
                guide_func = self.guidance_losses[cur_obj]
                if cur_kwargs is not None:
                    base_kwargs.update(cur_kwargs)
                losses = guide_func(**base_kwargs) # returns list of losses
                
                assert len(losses) == len(cur_alphas), "Number of alphas given for %s must match the number of losses returned" % (cur_obj)

                for loss, loss_name, loss_alpha in zip(losses, self.loss_names[cur_obj], cur_alphas):
                    if loss is None:
                        continue
                    loss_tot = loss_tot + loss.mean() * loss_alpha
                    print('%s-%s = %0.6f' % (cur_name, loss_name, loss.mean().item()))

            # compute grad and update prediction
            grad_outs = [loss_tot]
            grad_ins = [model_in]
            all_grad = torch.autograd.grad(grad_outs, grad_ins)
            grad = all_grad[0]
            soft_pred = self.update_pred(pred, grad, model_variance)

        return soft_pred

    def evaluate(self, pred, target):
        '''
        Evaluates how well guidance is followed
        - pred : output from the model
        - target : the gt motion from which guidance objectives are derived
        '''
        # now permuted to: [batch, 1, seq_len, nfeat]
        pred_p = self.preprocess(pred.permute(0,2,3,1), motion_rep=self.pred_motion_rep)
        target_p = self.preprocess(target.permute(0,2,3,1), motion_rep=self.tgt_motion_rep)
        # create a continous representation for angle 
        pred_c = self.convert_to_264(pred_p, motion_rep=self.pred_motion_rep)
        target_c = self.convert_to_264(target_p, motion_rep=self.tgt_motion_rep)

        eval_res = dict()
        for cur_obj, cur_name, cur_kwargs in zip(self.obj, self.guide_names, self.guide_kwargs):
            if cur_obj not in self.eval_funcs:
                print(f"{cur_obj}'s eval function is not implemented. skipping")
                eval_res.update({cur_obj + '_missing_eval': -1 * torch.ones([pred_p.size(0)])})
                continue
            base_kwargs = {'pred' : pred_c, 'target' : target_c, 't' : torch.zeros((pred_p.size(0)))}
            if cur_kwargs is not None:
                base_kwargs.update(cur_kwargs)
            eval_out = self.eval_funcs[cur_obj](**base_kwargs)
            eval_out = {f'{cur_name}-{k}' : v for k, v in eval_out.items()}
            eval_res.update(eval_out)

        return eval_res
        
    #
    # Handling model conditioning (masking) functions
    #

    def init_model_mask(self, gt_motion, gt_data_lengths, mask_motion_lengths):
        '''
        Create mask to condition model at start of denoising.
        - gt_motion: ground truth motion
        - gt_data_lengths: the lengths of GT sequences used to compute the keyframe indices
        - mask_motion_lengths: used directly as input to model.generate_motion_mask, this shouldn't be greater than the generated sequence length
        '''
        # known t masks
        mask_type_list = []
        mask_cfg_list = []
        # unknown t mask
        unknownt_mask_type = unknownt_mask_cfg = None
        # all keyframes for all guidance objectives, returned to user
        all_keyframe_idx_list = [] 

        # loop through to create configs and save info for mask generation
        for cur_obj, cur_mask, cur_kwargs in zip(self.obj, self.mask, self.guide_kwargs):
            # compute keyframe indices where relevant
            all_keyframe_idx = None
            if cur_obj == 'keyframes':
                all_keyframe_idx = []
                for i in range(gt_motion.shape[0]):
                    # each batch index has different keyframe inds based on motion length.
                    if cur_kwargs.keyframe_inds == 'all':
                        keyframe_idx = np.arange(gt_data_lengths[i].item()).astype(int).tolist()
                    else:
                        keyframe_idx = np.array(cur_kwargs.keyframe_inds) * (gt_data_lengths[i].item() - 1)
                        keyframe_idx = keyframe_idx.round().astype(int).tolist()
                    all_keyframe_idx.append(keyframe_idx)
            all_keyframe_idx_list.append(all_keyframe_idx)

            # build mask config based on guidance hyperparams
            if cur_mask is not None and cur_mask != 'no_mask':
                cur_mask_cfg = self.model.model_cfg.motion_mask.get(cur_mask, {})
                if cur_mask in {'local_joints_ee_sf', 'local_joints_ee_af', 'unknownt_local_joints'}:
                    # end-effector keyframes or full trajectory
                    assert cur_obj == 'keyframes'
                    cur_mask_cfg.mode = 'root+joints'
                    cur_mask_cfg.joint_names = cur_kwargs['local_joint_names']
                    cur_mask_cfg.obs_joint_prob = 1.0 # joint observations are never dropped
                    if cur_mask in {'local_joints_ee_sf', 'local_joints_ee_af'}:
                        cur_mask_cfg.consistent_joints = True # masked joints are the same over time
                elif cur_mask in {'root_traj_xy', 'unknownt_root_traj_xy'}:
                    # root 2D full trajectory
                    assert cur_obj == 'keyframes'
                    assert cur_kwargs['use_root_2d'], "guidance and mask are inconsistent wrt 2D or 3D root constraint"
                    # make sure we're only using rotation if desired
                    cur_mask_cfg.mode = 'pos_xy+rot' if cur_kwargs['use_root_orient'] else 'pos_xy'
                elif cur_mask in {'root_traj', 'unknownt_root_traj'}:
                    # root 3d full trajectory
                    assert cur_obj in {'keyframes', 'root_params'}
                    if cur_obj == 'keyframes':
                        assert not cur_kwargs['use_root_2d'], "guidance and mask are inconsistent wrt 2D or 3D root constraint"
                        cur_mask_cfg.mode = 'pos+rot' if cur_kwargs['use_root_orient'] else 'pos'
                    else:
                        cur_mask_cfg.mode = 'pos+rot'
                elif cur_mask in {'keyframes', 'unknownt_keyframes'}:
                    # full-body keyframes
                    assert cur_obj == 'keyframes'
                    cur_mask_cfg.mode = 'root+joints'
                elif cur_mask == 'keyframes_root2d':
                    # 2D root keyframes
                    assert cur_obj == 'keyframes'
                    assert cur_kwargs['use_root_2d'], "guidance and mask are inconsistent wrt 2D or 3D root constraint"
                    cur_mask_cfg.mode = 'root_xy'
                else:
                    NotImplementedError()

                # add keyframe inds to config
                if cur_obj == 'keyframes':
                    cur_mask_cfg['keyframe_idx'] = all_keyframe_idx

                # save relevant info
                if cur_mask.split('_')[0] == 'unknownt':
                    assert cur_kwargs.unknown_time, 'mask and guidance are inconsistent wrt known or unknown time'
                    if unknownt_mask_type is None:
                        unknownt_mask_type = cur_mask
                        unknownt_mask_cfg = cur_mask_cfg
                    else:
                        NotImplementedError("More than one unknownt mask specified, currently only suppport one")
                else:
                    mask_type_list.append(cur_mask)
                    mask_cfg_list.append(cur_mask_cfg)

        # print(mask_type_list)
        # print(mask_cfg_list)
        # print(unknownt_mask_type)
        # print(unknownt_mask_cfg)

        if len(mask_type_list) == 0:
            mask_type_list = ['no_mask']
            mask_cfg_list = None
        if unknownt_mask_type is None:
            unknownt_mask_type = 'no_mask'

        self.mask_in = {
            'observed_motion' : self.model.convert_motion_rep(gt_motion.clone()),
            'mask_motion_lengths' :  mask_motion_lengths,
            'use_mask_type' : mask_type_list,
            'mask_cfgs' : mask_cfg_list,
            'use_unknownt_mask_type' : unknownt_mask_type,
            'unknownt_mask_cfg' : unknownt_mask_cfg,
        }

        res = {k : None for k in ['motion_mask', 'observed_motion', 'rm_text_flag', 'global_motion', 'global_joint_mask', 'global_joint_func', 'selected_keyframe_t', 'unknownt_observed_motion', 'unknownt_motion_mask']}
        # generate the initial mask
        res = self.generate_model_mask(self.diff_steps)

        return res, all_keyframe_idx_list
    
    def generate_model_mask(self, t):
        '''
        If the model mask needs to be updated at this step, regenerate and return.
        Else return None.

        - t : int, denoising timestep from 1000 -> 0
        '''
        res = self.model.generate_motion_mask(self.model.model_cfg.motion_mask, self.mask_in['observed_motion'], self.mask_in['mask_motion_lengths'], 
                                                use_mask_type=self.mask_in['use_mask_type'],
                                                mask_cfgs=self.mask_in['mask_cfgs'],
                                                use_unknownt_mask_type=self.mask_in['use_unknownt_mask_type'], 
                                                unknownt_mask_cfg=self.mask_in['unknownt_mask_cfg']
                                              )
        return res
    
    #
    # Motion processing functions
    #

    def convert_to_264(self, x, motion_rep='full263'):
        bs, _, s, n = x.shape
        if motion_rep in {'global_root', 'global_root_local_joints'}:
            return x
        elif motion_rep == 'full263':
            out = torch.zeros(bs, 1, s, n+1, device=x.device)
            out [..., 0] = torch.sin(x[...,0])
            out [..., 1] = torch.cos(x[...,0])
            out [..., 2:] = x[...,1:]
        return out

    def preprocess(self, x, motion_rep='full263'):
        if motion_rep == 'full263':
            out = x * self.std + self.mean # x: [batch, 1, seq_len, nfeat]
        elif motion_rep in {'global_root', 'global_root_local_joints'}:
            # only need to unnormalize local pose part
            # first 5d are 3d root pos and 2d root heading
            root, local_pose = x[..., :5], x[..., 5:]
            if self.normalize_global_pos:
                root = root * self.global_std + self.global_mean
            if motion_rep == 'global_root_local_joints':
                local_pose = local_pose * self.std[4:] + self.mean[4:]
                out = torch.cat([root, local_pose], dim=-1)
            elif motion_rep == 'global_root':
                out = root
        return out
    
    def local_to_global_264(self, pred=None, target=None, root_only=False):
        '''
        Convert from local pose rep to global with the 264 dim rep.
        '''
        out_list = []

        in_list = [pred, target]
        rep_list = [self.pred_motion_rep, self.tgt_motion_rep]
        for x, motion_rep in zip(in_list, rep_list):
            if x is not None:
                pred_r_rot_quat = pred_r_pos = None
                if motion_rep in {'global_root', 'global_root_local_joints'}:
                    pred_r_pos, rot_cos_sin = x[..., :3], x[..., 3:5]
                    pred_r_rot_quat = torch.cat([rot_cos_sin[..., [0]], torch.zeros_like(rot_cos_sin[..., [0]]), rot_cos_sin[..., [1]], torch.zeros_like(rot_cos_sin[..., [0]])], dim=-1)
                # [B, 1, nframes, 22, 3]
                if root_only and motion_rep in {'global_root', 'global_root_local_joints'}:
                    global_pred_pose, global_pred_rot_quat = pred_r_pos, pred_r_rot_quat
                elif root_only:
                    global_pred_rot_quat, global_pred_pose = recover_root_rot_pos_264(x)
                elif not root_only and motion_rep == 'global_root':
                    # NOTE: this is a hack since position is expected to include 22 joints
                    #        if actually using the global_root model, should make this nicer
                    # [batch, 1, seq_len, 22, 3]
                    global_pred_pose = pred_r_pos.unsqueeze(-2).expand(-1, -1, -1, 22, -1)
                    global_pred_rot_quat = pred_r_rot_quat
                else:
                    global_pred_pose, global_pred_rot_quat = recover_from_ric_264(x, 22, r_rot_quat=pred_r_rot_quat, r_pos=pred_r_pos)        
                out_list.append((global_pred_pose, global_pred_rot_quat))

        # if target is not None:
        #     if root_only:
        #         global_target_rot_quat, global_target_pose = recover_root_rot_pos_264(target)
        #     else:
        #         global_target_pose, global_target_rot_quat = recover_from_ric_264(target, 22)
        #     out_list.append((global_target_pose, global_target_rot_quat))

        if len(out_list) > 1:
            return tuple(out_list)
        else:
            return out_list[0]
        
    #
    # Guidance losses
    #
    
    def unified_keyframes_loss(self, pred, target, t, keyframe_inds='all', unknown_time=False,
                                     global_joint_names=None, use_root_orient=True, use_root_2d=False, 
                                     local_joint_names=None, exp_upweight=False, exp_start_t=500, exp_alpha=0.01,
                                     return_raw=False, min_dist_idx=None, keyframe_ratio=True):
        '''
        Loss on joint positions (both global and local) + root orientation (if applicable).

        - pred: the model output motion
        - target: GT motion from which keyframes are taken
        - t: the denoising step
        - keyframe_inds: 'all' will use every frame, or can feed a list of ratios e.g., [0.8] will use a keyframe 80% of the way through each target sequence
        - unknown_time: if True, the time of the keyframe is considered unknown and the loss uses a min dist heuristic based on the union of local and global joints
        - global_joint_names: the joints to apply the global position loss function to or 'all'. If None, does not apply a global loss
        - use_root_orient: If global_joint_names contains 'Pelvis', can apply an optional orientation loss as well.
        - use_root_2d: if true, apply the global position loss for 'Pelvis' in 2D (projected to floor)
        - local_joint_names: the joints to apply the local position loss function to or 'all'. if None, does not apply a local loss.
        - min_dist_idx: the frame idx for the unknown time distance heuristic [B, 1, nkeys, 1], if not specifed will be computed automatically
        - return_raw: If true, returns additional information
        '''
        # pred is: [batch, 1, seq_len, nfeat]

        use_global_loss = global_joint_names is not None and len(global_joint_names) > 0
        use_local_loss = local_joint_names is not None and len(local_joint_names) > 0
        assert use_global_loss or use_local_loss, "must pass either global or local joint names to keyframes loss"

        if not use_global_loss:
            global_joint_names = set()
        elif global_joint_names == 'all':
            global_joint_names = set(SMPL_BONE_ORDER_NAMES[:22]) # no hands
        else:
            global_joint_names = set(global_joint_names)
        if not use_local_loss:
            local_joint_names = set()
        elif local_joint_names == 'all':
            local_joint_names = set(SMPL_BONE_ORDER_NAMES[1:22]) # no hands and leave out root
        else:
            local_joint_names = set(local_joint_names)
        all_joint_names = global_joint_names.union(local_joint_names)

        B, _, nframes, nfeat = pred.size()
        nkeys = None
        if keyframe_inds != 'all':
            # have specific timesteps to hit, otherwise assume want to match full trajectory
            keyframe_inds = torch.tensor([keyframe_inds]).expand((B, -1)).to(pred.device)
            if keyframe_ratio:
                data_len = torch.ones((B, 1))*nframes if self.tgt_data_len is None else self.tgt_data_len.unsqueeze(1)
                keyframe_inds = keyframe_inds * (data_len.to(pred.device) - 1)
            keyframe_inds = keyframe_inds.round().to(torch.long) # [B, num_keys]
            nkeys = keyframe_inds.size(1)

        # local poses
        local_pred_pose = pred
        local_target_pose = target

        # recover global poses if needed
        global_pred_pose = global_pred_rot_quat = None
        global_target_pose = global_target_rot_quat = None
        if use_global_loss or unknown_time: # will always need global for unknown time to compute min dist keyframe
            global_pred, global_target = self.local_to_global_264(pred=local_pred_pose, target=local_target_pose)
            global_pred_pose, global_pred_rot_quat = global_pred
            global_target_pose, global_target_rot_quat = global_target

        # 
        # Global loss
        #
        global_pred_keyframes = global_tgt_keyframes = None
        if use_global_loss or unknown_time:
            if keyframe_inds != 'all':
                global_pos_keyframe_inds = keyframe_inds[:,None,:,None, None].expand(B, 1, nkeys, 22, 3) # expand to same size as poses
                # compare all global joint positions at each desired keyframe
                global_pred_keyframes = global_pred_pose if unknown_time else torch.gather(global_pred_pose, 2, global_pos_keyframe_inds)
                global_tgt_keyframes = torch.gather(global_target_pose, 2, global_pos_keyframe_inds)
            else:
                # will compute error of full sequence and then chop off what we don't need
                global_pred_keyframes = global_pred_pose
                global_tgt_keyframes = global_target_pose

        if use_root_2d and 'Pelvis' in global_joint_names:
            # zero out y (up) dim
            root_idx = SMPL_BONE_ORDER_NAMES.index('Pelvis')
            global_pred_keyframes[..., root_idx, 1] = 0.0
            global_tgt_keyframes[..., root_idx, 1] = 0.0

        # Compute the minimum distance keyframe idx based on global positions of ALL specified joints (global+local) if we don't know the keyframe time already
        all_joint_inds = [SMPL_BONE_ORDER_NAMES.index(joint_name) for joint_name in all_joint_names]
        global_joint_inds = [SMPL_BONE_ORDER_NAMES.index(joint_name) for joint_name in global_joint_names]
        if unknown_time:
            if keyframe_inds == 'all':
                # in this case, each batch index will have a different number of "keyframes" (i.e. sequence length) which gets annoying to deal with
                raise NotImplementedError('unknown time only supports sparse keyframes')
            # pred is [B, 1, nframes, 22, 3]
            # tgt is  [B, 1, nkeys, 22, 3]
            global_pred_keyframes = global_pred_keyframes[:,:,None].expand(B, 1, nkeys, nframes, 22, global_pred_keyframes.size(-1)) # [B, 1, nkeys, nframes, 22, 3 or 2]
            global_tgt_keyframes = global_tgt_keyframes[:,:,:,None] # [B, 1, nkeys, 1, 22, 3]

            if min_dist_idx is None:
                # use ALL joint inds here so that min distance calculation considers both global and local constraint joints
                unknown_time_pose_diff = global_pred_keyframes[:,:,:,:,all_joint_inds]  - global_tgt_keyframes[:,:,:,:,all_joint_inds] # [B, 1, nkeys, nframes, njoints, 3]
                # unknown_time_pose_diff = global_pred_keyframes[:,:,:,:,global_joint_inds]  - global_tgt_keyframes[:,:,:,:,global_joint_inds] # DEBUG: use root dist only
                pose_dist = torch.norm(unknown_time_pose_diff, dim=-1) # [B, 1, nkeys, nframes, njoints]
                # want minimum pose over all joints we care about
                pose_dist = torch.sum(pose_dist, dim=-1) # [B, 1, nkeys, nframes]
                if SOFTMIN:
                    raise NotImplementedError('unified keyframes only supports hardmin')
                else:
                    # hardmin
                    min_dist_idx = torch.argmin(pose_dist, dim=-1, keepdim=True) # [B, 1, nkeys, 1]

        # global joint positions loss calculation
        global_loss = global_pose_diff = None
        if (use_global_loss or unknown_time) and self.pred_motion_rep == 'global_root':
            print('WARNING: with global_root rep, global keyframe loss will only be on pelvis location')
        if use_global_loss and unknown_time:
            global_pose_diff = global_pred_keyframes[:,:,:,:,global_joint_inds]  - global_tgt_keyframes[:,:,:,:,global_joint_inds] # [B, 1, nkeys, nframes, njoints, 3]
            njoints = len(global_joint_inds)
            pos_min_dist_idx = min_dist_idx[..., None, None].expand(B, 1, nkeys, 1, njoints, global_pred_keyframes.size(-1)) # [B, 1, nkeys, 1, njoints, 3 or 2]
            global_pose_loss = torch.gather(global_pose_diff**2, 3, pos_min_dist_idx)[:,:,:,0] # [B, 1, nkeys, njoints, 3 or 2]
            global_loss = torch.mean(global_pose_loss, dim=-1) # [B, 1, nkeys, njoints]
        elif use_global_loss:
            global_pose_diff = global_pred_keyframes[:,:,:,global_joint_inds] - global_tgt_keyframes[:,:,:,global_joint_inds]
            global_pose_loss = global_pose_diff**2
            global_loss = torch.mean(global_pose_loss, dim=-1) # [B, 1, num_keys, num_joints]

        # global root orientation loss calculation
        global_rot_diff = None
        if use_global_loss and use_root_orient and 'Pelvis' in global_joint_names:
            if keyframe_inds != 'all':
                global_rot_keyframe_inds = keyframe_inds[:,None,:,None].expand(B, 1, keyframe_inds.size(1), 4) # expand to same size as rot quats
                pred_rot_keyframe = global_pred_rot_quat if unknown_time else torch.gather(global_pred_rot_quat, 2, global_rot_keyframe_inds)
                tgt_rot_keyframe = torch.gather(global_target_rot_quat, 2, global_rot_keyframe_inds)
            else:
                pred_rot_keyframe = global_pred_rot_quat
                tgt_rot_keyframe = global_target_rot_quat

            if unknown_time:
                # pred_rot_keyframe is [B, 1, nframes, 4]
                # tgt_rot_keyframe is [B, 1, nkeys, 4]
                pred_rot_keyframe = pred_rot_keyframe[:,:,None].expand(B, 1, nkeys, nframes, 4) # [B, 1, nkeys, nframes, 4]
                tgt_rot_keyframe = tgt_rot_keyframe[:,:,:,None] # [B, 1, nkeys, 1, 4]
                # get the weighting based on previously computed position distance over desired joints
                global_rot_diff = pred_rot_keyframe - tgt_rot_keyframe # [B, 1, nkeys, nframes, 4]
                if SOFTMIN:
                    raise NotImplementedError('unified keyframes only supports hardmin')
                else:
                    # hardmin
                    rot_min_dist_idx = min_dist_idx.unsqueeze(-1).expand(B, 1, nkeys, 1, 4) # [B, 1, nkeys, 1, 4]
                    global_rot_loss = torch.gather(global_rot_diff**2, 3, rot_min_dist_idx)[:,:,:,0] # [B, 1, nkeys, 4]
                global_rot_loss = torch.mean(global_rot_loss, dim=-1, keepdim=True) # [B, 1, nkeys, 1]
            else:
                global_rot_diff = (pred_rot_keyframe - tgt_rot_keyframe)
                global_rot_loss = global_rot_diff**2
                global_rot_loss = torch.mean(global_rot_loss, dim=-1, keepdim=True) # [B, 1, num_keys, 1]

            global_loss = torch.cat([global_loss, global_rot_loss], dim=-1) # [B, 1, num_keys, num_joints+1]

            if exp_upweight:
                # more important earlier in denoising
                # # exponential
                # weight = torch.exp(-exp_alpha * (self.diff_steps - t[:,None,None]))
                # weight = torch.where(t[:,None,None] > exp_cutoff_t, weight, 0.0)
                # # linear
                # weight = (exp_start_t - t[:,None,None]) / exp_start_t
                # weight = torch.where(weight > 0, weight, 0.0)
                # step
                weight = (exp_start_t - t[:,None,None,None]) / exp_start_t
                weight = torch.where(weight > 0, 1.0, 0.0)

                global_loss *= weight

        if use_global_loss and keyframe_inds == 'all':
            # extract only the valid frames based on GT sequence length
            global_loss = torch.cat([global_loss[bi, :, :self.tgt_data_len[bi]].flatten() for bi in range(B)])
        
        #
        # Local loss
        #
        local_loss = local_pose_diff = None
        if use_local_loss:
            assert 'Pelvis' not in local_joint_names, 'Cannot apply a local loss to the root (it is at the origin)'
            # pull out only keyframe poses
            local_pred_keyframes = local_tgt_keyframes = None
            if keyframe_inds != 'all':
                local_keyframe_inds = keyframe_inds[:,None,:,None].expand(B, 1, keyframe_inds.size(1), nfeat) # expand to same size as poses
                # compare all global joint positions at each desired keyframe
                local_pred_keyframes = local_pred_pose if unknown_time else torch.gather(local_pred_pose, 2, local_keyframe_inds)  # [batch, 1, nkeys, nfeat]
                local_tgt_keyframes = torch.gather(local_target_pose, 2, local_keyframe_inds) # [batch, 1, nkeys, nfeat]
            else:
                # use all timesteps for now and will chop off invalid frames (past GT length) later
                local_pred_keyframes = local_pred_pose
                local_tgt_keyframes = local_target_pose


            # pull out desired joint positions
            local_joint_inds = [SMPL_BONE_ORDER_NAMES.index(joint_name)-1 for joint_name in local_joint_names] # the root is not considered here so subtract 1
            local_joint_inds = np.concatenate([np.arange(5 + idx*3, 5 + idx*3 + 3) for idx in local_joint_inds])
            if unknown_time:
                if keyframe_inds == 'all':
                    # in this case each batch index will have a different number of "keyframes" (i.e. sequence length) which gets annoying to deal with
                    raise NotImplementedError('unknown time only supports sparse keyframes')
                
                # pred_keyframes is [batch, 1, nframes, nfeat]
                # tgt_keyframes is [batch, 1, nkeys, nfeat]
                local_pred_keyframes = local_pred_keyframes[:,:,None].expand(B, 1, nkeys, nframes, nfeat)
                local_tgt_keyframes = local_tgt_keyframes[:,:,:,None] # [batch, 1, nkeys, 1, nfeat]
                local_pose_diff = local_pred_keyframes[..., local_joint_inds] - local_tgt_keyframes[..., local_joint_inds] # [batch, 1, nkeys, nframes, njoint_feats]

                local_min_dist_idx = min_dist_idx.unsqueeze(-1).expand(B, 1, nkeys, 1, local_pose_diff.size(-1)) # [B, 1, nkeys, 1, njoint_feats]
                local_loss = torch.gather(local_pose_diff**2, 3, local_min_dist_idx)[:,:,:,0] # [B, 1, nkeys, njoint_feats]
            else:
                local_pose_diff = local_pred_keyframes[:,:,:,local_joint_inds] - local_tgt_keyframes[:,:,:,local_joint_inds] # [B, 1, nkeys, njoint_feats]
                local_loss = local_pose_diff**2

            if exp_upweight:
                # more important earlier in denoising
                # # exponential
                # weight = torch.exp(-exp_alpha * (self.diff_steps - t[:,None,None]))
                # weight = torch.where(t[:,None,None] > exp_cutoff_t, weight, 0.0)
                # # linear
                # weight = (exp_start_t - t[:,None,None]) / exp_start_t
                # weight = torch.where(weight > 0, weight, 0.0)
                # step
                weight = (exp_start_t - t[:,None,None,None]) / exp_start_t
                weight = torch.where(weight > 0, 1.0, 0.0)

                local_loss *= weight

            if keyframe_inds == 'all':
                # extract only the valid frames based on GT sequence length
                local_loss = torch.cat([local_loss[bi, :, :self.tgt_data_len[bi]].flatten() for bi in range(B)])

        # print(local_loss.size())
        if DEBUG_PLOT_MIN_DIST_IDX and min_dist_idx is not None:
            # only show first keyframe for now
            cur_min_dist_frac = min_dist_idx[:,0,0,0].cpu().numpy() / float(nframes)
            # TODO if there is > 1 unknownt constraint they will both be writing to the tracker here.
            self.unknown_min_dist_tracker.append(cur_min_dist_frac)

        if return_raw:
            return global_loss, local_loss, global_pose_diff, global_rot_diff, local_pose_diff, min_dist_idx
        else:
            return global_loss, local_loss

    def unified_keyframes_eval(self, pred, target, t, keyframe_inds='all', unknown_time=False,
                                     global_joint_names=None, use_root_orient=True, use_root_2d=False, 
                                     local_joint_names=None, exp_upweight=False, exp_start_t=500, exp_alpha=0.01,
                                     keyframe_ratio=True):
        loss_out = self.unified_keyframes_loss(pred, target, t, keyframe_inds, unknown_time, global_joint_names, use_root_orient, use_root_2d, local_joint_names, 
                                                False, exp_start_t, exp_alpha, return_raw=True, keyframe_ratio=keyframe_ratio)
        global_loss, local_loss, global_pose_diff, global_rot_diff, local_pose_diff, min_dist_idx = loss_out

        # min_dist_idx is [B, 1, nkeys, 1]

        eval_out = dict()

        # TODO this is kinda dumb we have to do all the gather stuff with the min idx here, could just do this in the loss func

        # global errors
        if global_loss is not None:
            # global_pose_diff is  [B, 1, nkeys, nframes, njoints, 3] if unkown time
            #           else       [B, 1, nkeys, njoints, 3]
            global_joints_err = torch.norm(global_pose_diff, dim=-1)
            if unknown_time:
                # global_joints_err is [B, 1, nkeys, nframes, njoints]
                B, _, nkeys, _, njoints_global = global_joints_err.size()
                # compute loss at the timestep that the loss determines was optimal/minimal
                global_joints_min_idx = min_dist_idx[..., None].expand(B, 1, nkeys, 1, njoints_global)
                global_joints_err = torch.gather(global_joints_err, 3, global_joints_min_idx)[:,:,:,0] # [B, 1, nkeys, njoints]

            if keyframe_inds == 'all':
                global_joints_err = torch.stack([torch.mean(global_joints_err[bi, :, :self.tgt_data_len[bi]]) for bi in range(pred.size(0))])
            else:
                global_joints_err = torch.mean(global_joints_err, (1, 2, 3))

            eval_out['keyframes_global_joint_err'] = global_joints_err

            if global_rot_diff is not None:
                # rot_diff is quaternion difference, 
                #       but only idx 0 and 2 of quats are filled in: cos(theta) and sin(theta) of heading theta
                if unknown_time:
                    # rot diff is [B, 1, nkeys, nframes, 4]
                    # take the angle at the min root dist
                    B, _, nkeys, _, D = global_rot_diff.size()
                    global_rot_min_idx = min_dist_idx[..., None].expand(B, 1, nkeys, 1, D)
                    global_rot_diff = torch.gather(global_rot_diff, 3, global_rot_min_idx)[:,:,:,0]
                
                global_rot_diff = global_rot_diff[:,:,:,[0,2]] # [batch, 1, num_keys, 2]
                err_vec = torch.ones_like(global_rot_diff)
                err_vec[:,:,:,1] = 0.0
                err_vec = err_vec + global_rot_diff
                ang_err = torch.abs(torch.rad2deg(torch.atan2(err_vec[:,:,:,1], err_vec[:,:,:,0])))
                if keyframe_inds == 'all':
                    ang_err = torch.stack([torch.mean(ang_err[bi, :, :self.tgt_data_len[bi]]) for bi in range(pred.size(0))])
                else:
                    ang_err = torch.mean(ang_err, (1,2))
                eval_out['keyframes_global_root_ang_err'] = ang_err

        # local errors
        if local_loss is not None:
            if unknown_time:
                #  only care about error at closest frame
                # local_pose_diff is [batch, 1, nkeys, nframes, njoint_feats]
                B, _, nkeys, _, njoint_feats = local_pose_diff.size()
                local_min_idx = min_dist_idx[..., None].expand(B, 1, nkeys, 1, njoint_feats)
                local_pose_diff = torch.gather(local_pose_diff, 3, local_min_idx)[:,:,:,0]

            # local_pose_diff is [batch, 1, nkeys, njoint_feats]
            njoints = 21 if local_joint_names == 'all' else len(local_joint_names)
            local_pose_diff = local_pose_diff.reshape((pred.size(0), 1, -1, njoints, 3)) # [batch, 1, nkeys, njoints, 3]
            local_joints_err = torch.norm(local_pose_diff, dim=-1) # [batch, 1, num_keys, num_joints]
            if keyframe_inds == 'all':
                local_joints_err = torch.stack([torch.mean(local_joints_err[bi, :, :self.tgt_data_len[bi]]) for bi in range(pred.size(0))])
            else:
                local_joints_err = torch.mean(local_joints_err, (1,2,3)) # batch

            eval_out['keyframes_local_joint_err'] = local_joints_err

            # Also measure error of local constraints in GLOBAL space (i.e. how much does error compound with global+local)
            #       do this with modified call to keyframe loss using local joints as global and same min_dist_idx if unknown time
            local_in_glob_loss_out = self.unified_keyframes_loss(pred, target, t, keyframe_inds, unknown_time, 
                                                                    global_joint_names=local_joint_names,
                                                                    use_root_orient=False, 
                                                                    use_root_2d=False, 
                                                                    local_joint_names=None, 
                                                                    exp_upweight=False, exp_start_t=exp_start_t, exp_alpha=exp_alpha,
                                                                    return_raw=True,
                                                                    min_dist_idx=min_dist_idx,
                                                                    keyframe_ratio=keyframe_ratio)
            _, _, local_in_glob_diff, _, _, _ = local_in_glob_loss_out

            # global_pose_diff is  [B, 1, nkeys, nframes, njoints, 3] if unknown time
            #           else       [B, 1, nkeys, njoints, 3]
            local_in_glob_joints_err = torch.norm(local_in_glob_diff, dim=-1)
            if unknown_time:
                # global_joints_err is [B, 1, nkeys, nframes, njoints]
                B, _, nkeys, _, njoints_local = local_in_glob_joints_err.size()
                # compute loss at the timestep that the loss determines was optimal/minimal
                global_joints_min_idx = min_dist_idx[..., None].expand(B, 1, nkeys, 1, njoints_local)
                local_in_glob_joints_err = torch.gather(local_in_glob_joints_err, 3, global_joints_min_idx)[:,:,:,0] # [B, 1, nkeys, njoints]

            if keyframe_inds == 'all':
                local_in_glob_joints_err = torch.stack([torch.mean(local_in_glob_joints_err[bi, :, :self.tgt_data_len[bi]]) for bi in range(pred.size(0))])
            else:
                local_in_glob_joints_err = torch.mean(local_in_glob_joints_err, (1, 2, 3))

            eval_out['local_in_global_joint_err'] = local_in_glob_joints_err


        if min_dist_idx is not None:
            # only save if unknownt
            eval_out['unknown_keyframe_inds'] = min_dist_idx[:,0,:,0]
            # this assumes only tracks a single unknownt constraint
            if len(self.unknown_min_dist_tracker) > 0:
                eval_out['unknown_keyframe_ind_track'] = np.stack(self.unknown_min_dist_tracker, axis=0)
                self.unknown_min_dist_tracker = [] # reset
            
        return eval_out
    
    def foot_skate_loss(self, pred, target, t, exp_weighting=False, exp_alpha=0.01, return_raw=False):
        # get foot contacts
        foot_contacts = pred[..., -4:] # [batch, 1, nframes, 4] where 4 is ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")

        # compute velocities of global foot joints
        foot_joint_names = ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]
        foot_joint_idx = [SMPL_BONE_ORDER_NAMES.index(joint_name) for joint_name in foot_joint_names]

        global_pred_pose, _  = self.local_to_global_264(pred=pred)

        feet_pos = global_pred_pose[:,:,:,foot_joint_idx] # [B, 1, nframes, 4, 3]
        foot_vel = torch.sum((feet_pos[:,:,1:] - feet_pos[:,:,:-1])**2, dim=-1) # [B, 1, nframes-1, 4]

        # velocities should be near 0 when confident of foot contact
        loss = foot_vel * foot_contacts[:,:,:-1] # [B, 1, nframes-1, 4]

        if exp_weighting:
            # more important later in denoising (near step 0)
            weight = torch.exp(-exp_alpha * t[:,None,None,None])
            loss *= weight

        if return_raw:
            return loss, foot_vel, foot_contacts
        else:
            return (loss,)

    def foot_skate_eval(self, pred, target, t, exp_weighting, exp_alpha):
        _, foot_vel, foot_contacts = self.foot_skate_loss(pred, target, t, False, exp_alpha, return_raw=True)
        # vel is sum of squares already
        foot_vel = torch.sqrt(foot_vel) # [B, 1, nframes-1, 4]
        # contacts are direct output of model, threshold to get binary
        foot_contacts = foot_contacts[:,:,:-1] > 0.5 # [batch, 1, nframes-1, 4]

        vel_err = foot_vel * foot_contacts
        vel_err = torch.sum(vel_err, (1,2,3)) / torch.sum(foot_contacts, (1,2,3)) # mean over contacting frames

        return {'foot_skate_vel_err' : vel_err}