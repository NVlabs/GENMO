"""Body model factory for GEM.

Replaces the GVHMR hmr4d.utils.smplx_utils.make_smplx dependency.
"""

import torch.nn as nn


def make_smplx(type="smpl", **kwargs):
    """Create a body model.

    Supported types:
        "smpl"                   — SMPL neutral body model
        "supermotion"            — SMPL-X neutral model (full wrapper)
        "supermotion_v437coco17" — SmplxLiteV437Coco17: outputs 437 verts + COCO17 joints
        "supermotion_smpl24"     — SmplxLiteSmplN24: outputs SMPL-neutral 24 joints
        "supermotion_smpl24_male"   — SmplxLiteSmplN24(gender="male"): outputs SMPL-male 24 joints
        "supermotion_smpl24_female" — SmplxLiteSmplN24(gender="female"): outputs SMPL-female 24 joints
    """
    _BODY_MODEL_PATH = "inputs/checkpoints/body_models"

    _kwargs_disable = {
        "create_body_pose": False,
        "create_betas": False,
        "create_global_orient": False,
        "create_transl": False,
    }

    if type == "smpl":
        bm_kwargs = {
            "model_path": _BODY_MODEL_PATH,
            "model_type": "smpl",
            "gender": "neutral",
            "num_betas": 10,
        }
        bm_kwargs.update(_kwargs_disable)
        bm_kwargs.update(kwargs)
        return _BodyModelSMPL(**bm_kwargs)

    elif type == "supermotion":
        bm_kwargs = {
            "model_path": _BODY_MODEL_PATH,
            "model_type": "smplx",
            "gender": "neutral",
            "num_pca_comps": 12,
            "flat_hand_mean": False,
            "create_body_pose": False,
            "create_betas": False,
            "create_global_orient": False,
            "create_transl": False,
            "create_left_hand_pose": False,
            "create_right_hand_pose": False,
        }
        bm_kwargs.update(kwargs)
        return _BodyModelSMPLX(**bm_kwargs)

    elif type == "supermotion_EVAL3DPW":
        bm_kwargs = {
            "model_path": _BODY_MODEL_PATH,
            "model_type": "smplx",
            "gender": "neutral",
            "num_pca_comps": 12,
            "flat_hand_mean": True,
            "create_body_pose": False,
            "create_betas": False,
            "create_global_orient": False,
            "create_transl": False,
            "create_left_hand_pose": False,
            "create_right_hand_pose": False,
        }
        bm_kwargs.update(kwargs)
        return _BodyModelSMPLX(**bm_kwargs)

    elif type == "supermotion_v437coco17":
        from gem.utils.body_model.smplx_lite import SmplxLiteV437Coco17

        return SmplxLiteV437Coco17()

    elif type == "supermotion_smpl24":
        from gem.utils.body_model.smplx_lite import SmplxLiteSmplN24

        return SmplxLiteSmplN24()

    elif type == "supermotion_smpl24_male":
        from gem.utils.body_model.smplx_lite import SmplxLiteSmplN24

        return SmplxLiteSmplN24(gender="male")

    elif type == "supermotion_smpl24_female":
        from gem.utils.body_model.smplx_lite import SmplxLiteSmplN24

        return SmplxLiteSmplN24(gender="female")

    else:
        raise ValueError(f"[smplx_utils] Unknown body model type: {type}")


class _BodyModelSMPL(nn.Module):
    """Thin wrapper around smplx.SMPL for batch inference."""

    def __init__(self, **kwargs):
        super().__init__()
        import smplx

        self.bm = smplx.create(**kwargs)
        self.faces = self.bm.faces

    def forward(self, betas=None, global_orient=None, transl=None, body_pose=None, **kwargs):
        import torch

        device = self.bm.shapedirs.device
        dtype = self.bm.shapedirs.dtype
        batch_size = 1
        for v in [betas, global_orient, transl, body_pose]:
            if v is not None:
                batch_size = max(batch_size, len(v))

        if global_orient is None:
            global_orient = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if body_pose is None:
            body_pose = torch.zeros(
                [batch_size, 3 * self.bm.NUM_BODY_JOINTS], dtype=dtype, device=device
            )
        if betas is None:
            betas = torch.zeros([batch_size, self.bm.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        return self.bm(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            **kwargs,
        )

    def cuda(self):
        self.bm = self.bm.cuda()
        return self

    def to(self, *args, **kwargs):
        self.bm = self.bm.to(*args, **kwargs)
        return self


class _BodyModelSMPLX(nn.Module):
    """Thin wrapper around smplx.SMPLX for batch inference."""

    def __init__(self, **kwargs):
        super().__init__()
        import smplx

        self.bm = smplx.create(**kwargs)
        self.faces = self.bm.faces

    def get_skeleton(self, betas):
        """Compute skeleton joint positions from shape parameters.

        betas: (*, 10) -> skeleton: (*, J, 3) where J=55 for SMPLX
        """
        import torch

        J_template = self.bm.J_regressor @ self.bm.v_template  # (J, 3)
        J_shapedirs = torch.einsum(
            "jv, vcd -> jcd", self.bm.J_regressor, self.bm.shapedirs
        )  # (J, 3, num_betas)
        return J_template + torch.einsum("...d, jcd -> ...jc", betas, J_shapedirs)

    def forward(
        self,
        betas=None,
        global_orient=None,
        transl=None,
        body_pose=None,
        left_hand_pose=None,
        right_hand_pose=None,
        **kwargs,
    ):
        import torch

        device = self.bm.shapedirs.device
        dtype = self.bm.shapedirs.dtype
        batch_size = 1
        for v in [betas, global_orient, transl, body_pose]:
            if v is not None:
                batch_size = max(batch_size, len(v))

        if global_orient is None:
            global_orient = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if body_pose is None:
            body_pose = torch.zeros(
                [batch_size, 3 * self.bm.NUM_BODY_JOINTS], dtype=dtype, device=device
            )
        if betas is None:
            betas = torch.zeros([batch_size, self.bm.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        hand_pose_dim = self.bm.num_pca_comps if self.bm.use_pca else 3 * self.bm.NUM_HAND_JOINTS
        if left_hand_pose is None:
            left_hand_pose = torch.zeros([batch_size, hand_pose_dim], dtype=dtype, device=device)
        if right_hand_pose is None:
            right_hand_pose = torch.zeros([batch_size, hand_pose_dim], dtype=dtype, device=device)
        jaw_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        leye_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        reye_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        expression = torch.zeros(
            [batch_size, self.bm.num_expression_coeffs], dtype=dtype, device=device
        )

        return self.bm(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expression,
            **kwargs,
        )

    def cuda(self):
        self.bm = self.bm.cuda()
        return self

    def to(self, *args, **kwargs):
        self.bm = self.bm.to(*args, **kwargs)
        return self
