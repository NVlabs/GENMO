import numpy as np
import torch

from motiondiff.data_pipeline.humanml.utils.word_vectorizer import WordVectorizer
from motiondiff.models.common.smpl import SMPL_BONE_ORDER_NAMES
from motiondiff.models.mdm.mdm_base import MDMBase
from motiondiff.utils.scheduler import update_scheduled_params
from motiondiff.utils.tools import import_type_from_str
from motiondiff.utils.torch_utils import interp_tensor_with_scipy, tensor_to

"""
Main Model
"""


class MDMUnknownt(MDMBase):
    def __init__(self, cfg, is_inference=False, preload_checkpoint=True):
        super().__init__(cfg, is_inference)
        self.denoiser = import_type_from_str(self.model_cfg.denoiser.type)(
            pl_module=self, **self.model_cfg.denoiser
        )
        self.init_diffusion()
        if preload_checkpoint:
            self.load_pretrain_checkpoint()
        return

    def generate_motion_mask(
        self,
        motion_mask_cfg,
        motion,
        lengths,
        use_mask_type=None,
        return_keyframes=False,
        all_keyframe_idx=None,
    ):
        if use_mask_type is not None:
            mask_type = use_mask_type
        else:
            mask_type = np.random.choice(
                motion_mask_cfg.mask_types, p=motion_mask_cfg.mask_probs
            )

        sample_keyframes = all_keyframe_idx is None
        selected_keyframe_t = []
        observed_motion = []
        for i in range(motion.shape[0]):
            if sample_keyframes:
                mlen = lengths if isinstance(lengths, int) else lengths[i].item()
                t = np.random.randint(
                    mlen
                )  # random step from 0 to mlen for this sequence
            else:
                t = all_keyframe_idx[i]
                if not isinstance(t, int):
                    assert len(t) == 1, (
                        "unknown t trained model only supports single keyframe"
                    )
                    t = t[0]
            selected_keyframe_t.append(t)
            # observed motion is now just a single target frame, and it's not multiplied by mask below
            #        NOTE: this includes ALL dimensions for now but is masked later to only be root + joint pos
            observed_motion.append(motion[i, :, :, [t]])
        selected_keyframe_t = torch.from_numpy(np.array(selected_keyframe_t)).to(
            motion.device
        )
        observed_motion = torch.stack(observed_motion, dim=0)
        # so the mask here is really just determining which dimensions from the single selected keyframe that
        #   we want to constrain, nothing to do with timesteps
        motion_mask = torch.zeros_like(observed_motion)
        rm_text_flag = torch.zeros(motion.shape[0], device=motion.device)
        root_dim = self.motion_root_dim
        ljoint_dim = self.motion_localjoints_dim
        global_motion = None
        global_joint_mask = None
        global_joint_func = None

        def get_root_mask_indices(mode):
            if mode == "root+joints":
                root_mask_ind = slice(0, root_dim)
            elif mode == "root_pos+joints":
                if self.motion_rep in {"full263", "position"}:
                    root_mask_ind = slice(1, root_dim)
                elif self.motion_rep == "global_root_local_joints":
                    root_mask_ind = slice(0, 3)
                else:
                    raise NotImplementedError
            elif mode == "rootheight+joints":
                if self.motion_rep in {"full263", "position"}:
                    root_mask_ind = slice(3, root_dim)
                elif self.motion_rep == "global_root_local_joints":
                    root_mask_ind = slice(1, 2)
                else:
                    raise NotImplementedError
            elif mode == "joints":
                root_mask_ind = slice(0, 0)
            else:
                raise NotImplementedError
            return root_mask_ind

        def root_traj(_cfg):
            nonlocal motion_mask, rm_text_flag
            xy_only = _cfg.get("xy_only", False)
            if self.motion_rep in {"full263", "position"}:
                eind = 3 if xy_only else 4
                motion_mask[:, :eind] = 1.0
            elif self.motion_rep == "global_root_local_joints":
                mode = _cfg.get("mode", "pos+rot")
                if mode == "pos+rot":
                    root_mask_ind = np.arange(5)
                elif mode == "pos":
                    root_mask_ind = np.arange(3)
                elif mode == "pos_xy":
                    root_mask_ind = np.array([0, 2])
                elif mode == "pos_xy+rot":
                    root_mask_ind = np.array(
                        [0, 2, 3, 4]
                    )  # in the original coordinate, y is up, so we use z.
                motion_mask[:, root_mask_ind] = 1.0
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def keyframes(_cfg):
            nonlocal motion_mask, rm_text_flag
            mode = _cfg.get("mode", "root+joints")
            root_mask_ind = get_root_mask_indices(mode)
            motion_mask[:, root_dim : root_dim + ljoint_dim] = (
                1.0  # only root + local joint positions
            )
            motion_mask[:, root_mask_ind] = 1.0
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        def local_joints(_cfg):
            nonlocal motion_mask, rm_text_flag
            mode = _cfg.get("mode", "joints")
            root_mask_ind = get_root_mask_indices(mode)
            smpl_joint_names = SMPL_BONE_ORDER_NAMES[1:22]
            joint_names = _cfg.get("joint_names", smpl_joint_names)
            if joint_names == "all":
                joint_names = smpl_joint_names
            joint_indices = [
                smpl_joint_names.index(joint_name) for joint_name in joint_names
            ]
            joint_mask = np.zeros(21, dtype=np.float32)
            joint_mask[joint_indices] = np.random.binomial(
                1, _cfg.obs_joint_prob, size=len(joint_indices)
            )
            motion_mask[:, root_dim : root_dim + ljoint_dim] = (
                torch.from_numpy(joint_mask)
                .repeat_interleave(3)[:, None, None]
                .to(motion.device)
            )
            motion_mask[:, root_mask_ind] = 1.0
            rm_text_flag = torch.from_numpy(
                np.random.binomial(1, _cfg.mask_text_prob, size=motion.shape[0])
            ).to(motion.device)

        if mask_type != "no_mask":
            mask_cfg = motion_mask_cfg.get(mask_type, {})
            mask_func = locals()[mask_cfg.get("func", mask_type)]
            mask_func(mask_cfg)

        if use_mask_type is not None:
            rm_text_flag = torch.ones(motion.shape[0], device=motion.device)

        res = {
            "motion_mask": motion_mask,
            "observed_motion": observed_motion,
            "rm_text_flag": rm_text_flag,
            "global_motion": global_motion,
            "global_joint_mask": global_joint_mask,
            "global_joint_func": global_joint_func,
            "selected_keyframe_t": selected_keyframe_t,
        }
        if return_keyframes:
            res["all_keyframe_idx"] = all_keyframe_idx
        return res
