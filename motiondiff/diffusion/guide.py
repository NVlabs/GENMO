import numpy as np
import torch
import torch.nn.functional as F

import motiondiff.diffusion.guide_funcs as guide_funcs
from motiondiff.data_pipeline.humanml.scripts.motion_process import (
    recover_from_ric_264,
    recover_root_rot_pos_264,
)
from motiondiff.diffusion.guide_funcs import GUIDE_OBJECTIVES
from motiondiff.models.common.smpl import SMPL_BONE_ORDER_NAMES
from motiondiff.utils.torch_utils import tensor_to

GUIDE_TYPES = {"soft"}
GRAD_THRESH_OPTIONS = {None, "none", "var", "clip_norm", "clip_value"}
PRED_THRESH_OPTIONS = {None, "none", "static", "dynamic"}

MOTION_REPS = {"full263", "global_root_local_joints", "global_root"}


class Guide:
    def __init__(
        self,
        cfg,
        model,
        diff_steps,
        batch_size,
        pred_nframes,
        tgt_data_len=None,
        pred_motion_rep="full263",
        tgt_motion_rep="full263",
    ):
        """
        Diffusion guidance.
        """
        self.cfg = cfg
        self.diff_steps = diff_steps
        self.batch_size = batch_size
        self.pred_nframes = pred_nframes  # nframes in generated sequences
        self.tgt_data_len = tgt_data_len  # nframes in each of the ground truth target sequences (before padding)
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

        self.thresh_grad = cfg.get("thresh_grad", "none")
        assert self.thresh_grad in GRAD_THRESH_OPTIONS, (
            f"thresh_grad must be one of {GRAD_THRESH_OPTIONS}"
        )
        self.thresh_grad_param = cfg.get("thresh_grad_param", None)

        self.thresh_pred = cfg.get("thresh_pred", "none")
        assert self.thresh_pred in PRED_THRESH_OPTIONS, (
            f"thresh_pred must be one of {PRED_THRESH_OPTIONS}"
        )
        self.thresh_pred_param = cfg.get("thresh_pred_param", None)

        self.guide_type = self.cfg.type
        assert self.guide_type in GUIDE_TYPES

        # create objectives
        guide_dict = self.cfg.objectives
        self.guides = {
            k: GUIDE_OBJECTIVES[guide["objective"]](
                weight=guide["alpha"], **guide["kwargs"]
            )
            if "kwargs" in guide
            else GUIDE_OBJECTIVES[guide["objective"]](weight=guide["alpha"])
            for k, guide in guide_dict.items()
        }
        # for objectives based on keyframes, initialize with batch_size / num frames info
        for cur_guide in self.guides.values():
            if cur_guide.guide_type in {
                "global_joint_pos",
                "global_root_orient",
                "local_joint_pos",
                "keyframes",
            }:
                cur_guide.init_keyframe_inds(
                    self.batch_size, self.pred_nframes, self.tgt_data_len, model.device
                )

        print(self.guides)

        # the model masks to use for each objective
        self.mask = [guide.get("mask", None) for _, guide in guide_dict.items()]

    #
    # Main guidance functions
    #

    def update_pred(self, pred, grad, var):
        """
        Update pred based on guidance gradient while accounting for
        any gradient clipping or prediction thresholding.
        """
        # modifying gradient
        if self.thresh_grad == "var":
            grad = grad * var
        elif self.thresh_grad == "clip_norm":
            max_norm = self.thresh_grad_param
            grad_norm = torch.linalg.vector_norm(grad, ord=2, dim=(1, 2, 3))
            # print(grad_norm)
            clip_coef = max_norm / (grad_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            grad = grad * clip_coef_clamped[:, None, None, None]
        elif self.thresh_grad == "clip_value":
            clip_val = self.thresh_grad_param
            grad = torch.clamp(grad, min=-clip_val, max=clip_val)

        # update prediction
        pred = pred - grad

        # threshold prediction
        # implemented based on imagen: https://arxiv.org/pdf/2205.11487.pdf
        if self.thresh_pred == "static":
            clip_val = self.thresh_pred_param
            pred = torch.clamp(pred, min=-clip_val, max=clip_val)
        elif self.thresh_pred == "dynamic":
            B = pred.size(0)
            perc, clip_val = self.thresh_pred_param
            s = torch.quantile(torch.abs(pred.reshape((B, -1))), perc, dim=1)
            # print(s)
            # for those motions with quantile over thresh, clamp and then scale so they are clip_val at most
            s = s[:, None, None, None].expand_as(pred)
            pred = torch.where(
                s > clip_val,
                torch.clamp(pred, min=-s, max=s) / ((1.0 / clip_val) * s + 1e-6),
                pred,
            )

        return pred

    def guide(self, model_in, pred, target, model_variance, t):
        # pred / target: [batch, nfeat, 1, seq_len]

        # don't use any guidance on final step
        cur_t = torch.max(t).item()
        is_last_step = cur_t == 0
        if is_last_step:
            return pred

        # compute losses
        with torch.enable_grad():
            # now permulted to: [batch, 1, seq_len, nfeat]
            # unnormalizes the inputs
            pred_p = self.preprocess(
                pred.permute(0, 2, 3, 1), motion_rep=self.pred_motion_rep
            )
            target_p = self.preprocess(
                target.permute(0, 2, 3, 1), motion_rep=self.tgt_motion_rep
            )
            # create a continous representation for angle TODO: not sure this is necessary
            pred_c = self.convert_to_264(pred_p, motion_rep=self.pred_motion_rep)
            target_c = self.convert_to_264(target_p, motion_rep=self.tgt_motion_rep)

            # pull out possible inputs to loss functions
            global_pred, global_target = self.local_to_global_264(
                pred=pred_c, target=target_c
            )
            global_pred_pose, global_pred_rot_quat = global_pred
            global_target_pose, global_target_rot_quat = global_target

            start_local = 5
            end_local = start_local + 21 * 3  # 21 joints with root excluded
            local_pred_pose = pred_c[..., start_local:end_local]
            local_target_pose = target_c[..., start_local:end_local]
            pred_contacts = pred_c[..., -4:]

            loss_tot = 0.0
            for cur_name, cur_guide in self.guides.items():
                if cur_guide.guide_type == guide_funcs.FootSkate.guide_type:
                    loss_out = cur_guide.loss(global_pred_pose, pred_contacts, t)
                elif cur_guide.guide_type == guide_funcs.Keyframes.guide_type:
                    loss_out = cur_guide.loss(
                        global_pred_pose,
                        global_target_pose,
                        global_pred_rot_quat,
                        global_target_rot_quat,
                        local_pred_pose,
                        local_target_pose,
                    )
                elif cur_guide.guide_type == guide_funcs.GlobalJointPos.guide_type:
                    loss_out = cur_guide.loss(global_pred_pose, global_target_pose)
                elif cur_guide.guide_type == guide_funcs.GlobalRootOrient.guide_type:
                    loss_out = cur_guide.loss(
                        global_pred_rot_quat, global_target_rot_quat
                    )
                elif cur_guide.guide_type == guide_funcs.LocalJointPos.guide_type:
                    loss_out = cur_guide.loss(local_pred_pose, local_target_pose)
                else:
                    raise NotImplementedError(
                        "Guide type %s has not been implemented!"
                        % (cur_guide.guide_type)
                    )

                loss_dict, weight_dict = loss_out
                for loss_name, loss in loss_dict.items():
                    loss_mean = loss.mean()
                    loss_tot = loss_tot + loss_mean * weight_dict[loss_name]
                    print("%s-%s = %0.6f" % (cur_name, loss_name, loss_mean.item()))

            # compute grad and update prediction
            grad_outs = [loss_tot]
            grad_ins = [model_in]
            all_grad = torch.autograd.grad(grad_outs, grad_ins)
            grad = all_grad[0]
            soft_pred = self.update_pred(pred, grad, model_variance)

        return soft_pred

    def evaluate(
        self, target, pred_full_rep=None, global_pred_pose=None, pred_contacts=None
    ):
        """
        Evaluates how well guidance is followed
        - target : the gt motion from which guidance objectives are derived (in self.tgt_motion_rep)
        - pred_full_rep : [batch, nfeat, 1, nframes] full output from the model (in self.pred_motion_rep format)
        Alternatively, can evaluate just the predicted global joint positions (optionally with foot contact)
        - global_pred_pose : [batch, nframes, 22, 3] global joint positions, unnormalized
        - pred_contacts: [batch, 4, 1, nframes] foot contact probabilities (NORMALIZED outputs from model)
        """

        # always need target
        target_p = self.preprocess(
            target.permute(0, 2, 3, 1), motion_rep=self.tgt_motion_rep
        )
        target_c = self.convert_to_264(target_p, motion_rep=self.tgt_motion_rep)
        global_target_pose, global_target_rot_quat = self.local_to_global_264(
            target=target_c
        )
        start_local = 5
        end_local = start_local + 21 * 3  # 21 joints with root excluded
        local_target_pose = target_c[..., start_local:end_local]

        if pred_full_rep is not None:
            pred_p = self.preprocess(
                pred_full_rep.permute(0, 2, 3, 1), motion_rep=self.pred_motion_rep
            )
            pred_c = self.convert_to_264(pred_p, motion_rep=self.pred_motion_rep)
            global_pred_pose, global_pred_rot_quat = self.local_to_global_264(
                pred=pred_c
            )
            local_pred_pose = pred_c[..., start_local:end_local]
            pred_contacts = pred_c[..., -4:]
        elif global_pred_pose is not None:
            global_pred_pose = global_pred_pose.unsqueeze(
                1
            )  # [batch, 1, nframes, 22, 3]
            pred_contacts = pred_contacts.permute(0, 2, 3, 1)  # [batch, 1, nframes, 4]
            pred_contacts = (
                pred_contacts * self.std[-4:] + self.mean[-4:]
            )  # unnormalize
            # don't have these, will have to skip metrics that involve them
            global_pred_rot_quat = None
            local_pred_pose = None
        else:
            print(
                "Couldn't evaluate guidance! Must pass in either the predicted full rep or joint positions."
            )
            return dict()

        eval_res = dict()
        for cur_name, cur_guide in self.guides.items():
            if cur_guide.guide_type == guide_funcs.FootSkate.guide_type:
                eval_out = cur_guide.eval(global_pred_pose, pred_contacts)
            elif cur_guide.guide_type == guide_funcs.Keyframes.guide_type:
                # if global_pred_rot_quat and local_pred_pose are unavailable, they will be ignored
                eval_out = cur_guide.eval(
                    global_pred_pose,
                    global_target_pose,
                    global_pred_rot_quat,
                    global_target_rot_quat,
                    local_pred_pose,
                    local_target_pose,
                )
            elif cur_guide.guide_type == guide_funcs.GlobalJointPos.guide_type:
                eval_out = cur_guide.eval(global_pred_pose, global_target_pose)
            elif cur_guide.guide_type == guide_funcs.GlobalRootOrient.guide_type:
                if global_pred_rot_quat is None:
                    continue
                eval_out = cur_guide.eval(global_pred_rot_quat, global_target_rot_quat)
            elif cur_guide.guide_type == guide_funcs.LocalJointPos.guide_type:
                # if local_pred_pose are unavailable, they will be ignored
                eval_out = cur_guide.eval(
                    local_pred_pose,
                    local_target_pose,
                    global_pred_pose,
                    global_target_pose,
                )
            else:
                raise NotImplementedError(
                    "Guide type %s has not been implemented!" % (cur_guide.guide_type)
                )

            eval_out = {f"{cur_name}-{k}": v for k, v in eval_out.items()}
            eval_res.update(eval_out)

        return eval_res

    #
    # Handling model conditioning (masking) functions
    #

    def init_model_mask(self, gt_motion, gt_data_lengths, mask_motion_lengths):
        """
        Create mask to condition model at start of denoising.
        - gt_motion: ground truth motion
        - gt_data_lengths: the lengths of GT sequences used to compute the keyframe indices
        - mask_motion_lengths: used directly as input to model.generate_motion_mask, this shouldn't be greater than the generated sequence length
        """
        # known t masks
        mask_type_list = []
        mask_cfg_list = []
        # unknown t mask
        unknownt_mask_type = unknownt_mask_cfg = None
        # all keyframes for all guidance objectives, returned to user
        all_keyframe_idx_list = []

        # loop through to create configs and save info for mask generation
        for cur_guide, cur_mask in zip(self.guides.values(), self.mask):
            # compute keyframe indices where relevant
            all_keyframe_idx = None
            if isinstance(cur_guide, guide_funcs.KeyframeBase):
                all_keyframe_idx = []
                for i in range(gt_motion.shape[0]):
                    # each batch index has different keyframe inds based on motion length.
                    if cur_guide.keyframe_inds_in == "all":
                        keyframe_idx = (
                            np.arange(gt_data_lengths[i].item()).astype(int).tolist()
                        )
                    else:
                        keyframe_idx = np.array(cur_guide.keyframe_inds_in) * (
                            gt_data_lengths[i].item() - 1
                        )
                        keyframe_idx = keyframe_idx.round().astype(int).tolist()
                    all_keyframe_idx.append(keyframe_idx)
            all_keyframe_idx_list.append(all_keyframe_idx)

            # build mask config based on guidance hyperparams
            if cur_mask is not None and cur_mask != "no_mask":
                cur_mask_cfg = self.model.model_cfg.motion_mask.get(cur_mask, {})
                if cur_mask in {"local_joints_ee_sf", "local_joints_ee_af"}:
                    # end-effector keyframes or full trajectory
                    assert cur_guide.guide_type in {"keyframes", "local_joint_pos"}
                    cur_mask_cfg.mode = "root+joints"
                    if cur_guide.guide_type == "keyframes":
                        cur_mask_cfg.joint_names = cur_guide.local_joint_names
                    elif cur_guide.guide_type == "local_joint_pos":
                        cur_mask_cfg.joint_names = cur_guide.joint_names
                    cur_mask_cfg.obs_joint_prob = (
                        1.0  # joint observations are never dropped
                    )
                    if cur_mask in {"local_joints_ee_sf", "local_joints_ee_af"}:
                        cur_mask_cfg.consistent_joints = (
                            True  # masked joints are the same over time
                        )
                elif cur_mask in {"root_traj_xy"}:
                    # root 2D full trajectory
                    assert cur_guide.guide_type in {
                        "keyframes",
                        "global_joint_pos",
                        "global_root_orient",
                    }
                    if cur_guide.guide_type != "global_root_orient":
                        assert cur_guide.use_root_2d, (
                            "guidance and mask are inconsistent wrt 2D or 3D root constraint"
                        )
                    if cur_guide.guide_type == "keyframes":
                        # make sure we're only using rotation if desired
                        cur_mask_cfg.mode = (
                            "pos_xy+rot" if cur_guide.use_root_orient else "pos_xy"
                        )
                    else:
                        cur_mask_cfg.mode = (
                            "pos_xy+rot"
                            if cur_guide.guide_type == "global_root_orient"
                            else "pos_xy"
                        )
                elif cur_mask in {"root_traj"}:
                    # root 3d full trajectory
                    assert cur_guide.guide_type in {
                        "keyframes",
                        "global_joint_pos",
                        "global_root_orient",
                    }
                    if cur_guide.guide_type != "global_root_orient":
                        assert not cur_guide.use_root_2d, (
                            "guidance and mask are inconsistent wrt 2D or 3D root constraint"
                        )
                    if cur_guide.guide_type == "keyframes":
                        cur_mask_cfg.mode = (
                            "pos+rot" if cur_guide.use_root_orient else "pos"
                        )
                elif cur_mask in {"keyframes"}:
                    # full-body keyframes
                    assert cur_guide.guide_type in {
                        "keyframes",
                        "global_joint_pos",
                        "global_root_orient",
                        "local_joint_pos",
                    }
                    if cur_guide.guide_type == "local_joint_pos":
                        cur_mask_cfg.model = "joints"
                    else:
                        cur_mask_cfg.mode = "root+joints"
                elif cur_mask == "keyframes_root2d":
                    # 2D root keyframes
                    assert cur_guide.guide_type in {"keyframes", "global_joint_pos"}
                    assert cur_guide.use_root_2d, (
                        "guidance and mask are inconsistent wrt 2D or 3D root constraint"
                    )
                    cur_mask_cfg.mode = "root_xy"
                else:
                    NotImplementedError()

                # add keyframe inds to config
                if isinstance(cur_guide, guide_funcs.KeyframeBase):
                    cur_mask_cfg["keyframe_idx"] = all_keyframe_idx

                # save relevant info
                if cur_mask.split("_")[0] == "unknownt":
                    assert cur_guide.unknown_time, (
                        "mask and guidance are inconsistent wrt known or unknown time"
                    )
                    if unknownt_mask_type is None:
                        unknownt_mask_type = cur_mask
                        unknownt_mask_cfg = cur_mask_cfg
                    else:
                        NotImplementedError(
                            "More than one unknownt mask specified, currently only suppport one"
                        )
                else:
                    mask_type_list.append(cur_mask)
                    mask_cfg_list.append(cur_mask_cfg)

        # print(mask_type_list)
        # print(mask_cfg_list)
        # print(unknownt_mask_type)
        # print(unknownt_mask_cfg)

        if len(mask_type_list) == 0:
            mask_type_list = ["no_mask"]
            mask_cfg_list = None
        if unknownt_mask_type is None:
            unknownt_mask_type = "no_mask"

        self.mask_in = {
            "observed_motion": self.model.convert_motion_rep(gt_motion.clone()),
            "mask_motion_lengths": mask_motion_lengths,
            "use_mask_type": mask_type_list,
            "mask_cfgs": mask_cfg_list,
            "use_unknownt_mask_type": unknownt_mask_type,
            "unknownt_mask_cfg": unknownt_mask_cfg,
        }

        # generate the initial mask
        res = self.model.generate_motion_mask(
            self.model.model_cfg.motion_mask,
            self.mask_in["observed_motion"],
            self.mask_in["mask_motion_lengths"],
            use_mask_type=self.mask_in["use_mask_type"],
            mask_cfgs=self.mask_in["mask_cfgs"],
            use_unknownt_mask_type=self.mask_in["use_unknownt_mask_type"],
            unknownt_mask_cfg=self.mask_in["unknownt_mask_cfg"],
        )

        return res, all_keyframe_idx_list

    #
    # Motion processing functions
    #

    def convert_to_264(self, x, motion_rep="full263"):
        bs, _, s, n = x.shape
        if motion_rep in {"global_root", "global_root_local_joints"}:
            return x
        elif motion_rep == "full263":
            out = torch.zeros(bs, 1, s, n + 1, device=x.device)
            out[..., 0] = torch.sin(x[..., 0])
            out[..., 1] = torch.cos(x[..., 0])
            out[..., 2:] = x[..., 1:]
        return out

    def preprocess(self, x, motion_rep="full263"):
        if motion_rep == "full263":
            out = x * self.std + self.mean  # x: [batch, 1, seq_len, nfeat]
        elif motion_rep in {"global_root", "global_root_local_joints"}:
            # only need to unnormalize local pose part
            # first 5d are 3d root pos and 2d root heading
            root, local_pose = x[..., :5], x[..., 5:]
            if self.normalize_global_pos:
                root = root * self.global_std + self.global_mean
            if motion_rep == "global_root_local_joints":
                local_pose = local_pose * self.std[4:] + self.mean[4:]
                out = torch.cat([root, local_pose], dim=-1)
            elif motion_rep == "global_root":
                out = root
        return out

    def local_to_global_264(self, pred=None, target=None, root_only=False):
        """
        Convert from local pose rep to global with the 264 dim rep.
        """
        out_list = []

        in_list = [pred, target]
        rep_list = [self.pred_motion_rep, self.tgt_motion_rep]
        for x, motion_rep in zip(in_list, rep_list):
            if x is not None:
                pred_r_rot_quat = pred_r_pos = None
                if motion_rep in {"global_root", "global_root_local_joints"}:
                    pred_r_pos, rot_cos_sin = x[..., :3], x[..., 3:5]
                    pred_r_rot_quat = torch.cat(
                        [
                            rot_cos_sin[..., [0]],
                            torch.zeros_like(rot_cos_sin[..., [0]]),
                            rot_cos_sin[..., [1]],
                            torch.zeros_like(rot_cos_sin[..., [0]]),
                        ],
                        dim=-1,
                    )
                # [B, 1, nframes, 22, 3]
                if root_only and motion_rep in {
                    "global_root",
                    "global_root_local_joints",
                }:
                    global_pred_pose, global_pred_rot_quat = pred_r_pos, pred_r_rot_quat
                elif root_only:
                    global_pred_rot_quat, global_pred_pose = recover_root_rot_pos_264(x)
                elif not root_only and motion_rep == "global_root":
                    # NOTE: this is a hack since position is expected to include 22 joints
                    #        if actually using the global_root model, should make this nicer
                    # [batch, 1, seq_len, 22, 3]
                    global_pred_pose = pred_r_pos.unsqueeze(-2).expand(
                        -1, -1, -1, 22, -1
                    )
                    global_pred_rot_quat = pred_r_rot_quat
                else:
                    global_pred_pose, global_pred_rot_quat = recover_from_ric_264(
                        x, 22, r_rot_quat=pred_r_rot_quat, r_pos=pred_r_pos
                    )
                out_list.append((global_pred_pose, global_pred_rot_quat))

        if len(out_list) > 1:
            return tuple(out_list)
        else:
            return out_list[0]
