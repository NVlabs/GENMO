# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the GEM-SMPL denoiser (with condition composition) to ONNX.

Wraps the GEM Lightning module's embedding layers + the core ``NetworkEncoderRoPE``
into a single ONNX module so the demo doesn't need PyTorch for the heavy
forward pass. Mirrors ``gem/tools/export/export_denoiser_onnx.py`` in the
sister gem repo, adapted for SMPL: 17 COCO keypoints, 151-D motion output,
HMR2 image features (1024-D).

Usage::

    cd /path/to/gem-smpl
    python tools/export/export_denoiser_onnx.py \
        --ckpt gem_smpl_s03.ckpt \
        --output inputs/onnx/gem_smpl_denoiser.onnx

    # No-image-feature variant (image features replaced with zeros + learned bias)
    python tools/export/export_denoiser_onnx.py \
        --ckpt gem_smpl_s03.ckpt --no-imgfeat
"""
# ruff: noqa: E402, I001
import argparse
import functools
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch
import torch.nn as nn

torch.load = functools.partial(torch.load, weights_only=False)

from gem.utils.cam_utils import compute_bbox_info_bedlam


class GEMSMPLDenoiserWrapper(nn.Module):
    """Wraps condition embedding + denoiser into a single exportable module.

    Inputs (all on the same device):
        obs:           (B, L, 17, 3)   raw 2D keypoints [x, y, confidence]
        bbx_xys:       (B, L, 3)       bounding box [cx, cy, size]
        K_fullimg:     (B, L, 3, 3)    camera intrinsics
        f_imgseq:      (B, L, 1024)    HMR2 image features
        f_cam_angvel:  (B, L, 6)       camera angular velocity (6D rotation)

    Outputs:
        pred_x:   (B, L, 151)  motion prediction (gvhmr feature layout)
        pred_cam: (B, L, 3)    weak-perspective camera prediction
    """

    def __init__(self, gem_model: nn.Module):
        super().__init__()
        self.gem = gem_model.eval()

        # The wrapper hardcodes the regression-only inference convention from
        # GEMDiffusion.forward_test (gem/network/gem_diffusion.py:216-233):
        # `xt = zeros, t = num_steps - 1`. For dual-mode checkpoints we force
        # regression-only behavior at export time — the diffusion sampler is
        # not exported. This is the same shortcut used by gem repo's exporter.
        if not getattr(gem_model.pipeline.denoiser3d, "regression_only", False):
            print(
                "[Export] WARNING: checkpoint was trained in dual mode "
                "(regression_only=False); forcing regression-only inference for export. "
                "Only the regression path of the model is reproduced in ONNX."
            )
            gem_model.pipeline.denoiser3d.regression_only = True

        self.denoiser = gem_model.pipeline.denoiser3d.denoiser

        try:
            diff = gem_model.pipeline.denoiser3d.train_diffusion
            self.last_t = int(diff.original_num_steps - 1)
        except Exception:
            self.last_t = 999

        self.learned_pos_linear = gem_model.learned_pos_linear
        self.learned_pos_params = gem_model.learned_pos_params
        self.embed_noisyobs = gem_model.embed_noisyobs
        self.cliffcam_embedder = gem_model.cliffcam_embedder
        self.imgseq_embedder = gem_model.imgseq_embedder
        self.cam_angvel_embedder = gem_model.cam_angvel_embedder

        self.use_cond_exists = gem_model.model_cfg.use_cond_exists_as_input
        if self.use_cond_exists and hasattr(gem_model, "cond_exists_embedder"):
            self.cond_exists_embedder = gem_model.cond_exists_embedder

        self.normalize_cam_angvel = gem_model.model_cfg.normalize_cam_angvel
        if self.normalize_cam_angvel:
            self.register_buffer("cam_angvel_mean", gem_model.cam_angvel_mean.clone())
            self.register_buffer("cam_angvel_std", gem_model.cam_angvel_std.clone())

        self.in_attr = list(gem_model.pipeline.args.in_attr)
        self.latent_dim = gem_model.latent_dim
        self.obs_num_joints = gem_model.obs_num_joints  # 17 for SMPL

        self.endecoder = gem_model.endecoder
        if self.endecoder.obs_indices_dict is None:
            self.endecoder.build_obs_indices_dict()
        self.motion_dim = int(self.endecoder.get_motion_dim())
        self._sample_indices_dict = self.endecoder.obs_indices_dict

    def _normalize_kp2d(self, obs_kp2d, bbx_xys):
        """ONNX-friendly kp2d normalization (matches gem.utils.geo_transform.normalize_kp2d)."""
        obs_xy = obs_kp2d[..., :2]
        center = bbx_xys[..., :2]
        scale = bbx_xys[..., [2]].clamp(min=1e-2)

        xy_max = center + scale / 2
        xy_min = center - scale / 2
        invisible_mask = (
            (obs_xy[..., 0] < xy_min[..., None, 0])
            | (obs_xy[..., 0] > xy_max[..., None, 0])
            | (obs_xy[..., 1] < xy_min[..., None, 1])
            | (obs_xy[..., 1] > xy_max[..., None, 1])
        )
        normalized_xy = 2 * (obs_xy - center.unsqueeze(-2)) / scale.unsqueeze(-2)
        obs_conf = obs_kp2d[..., 2] * (~invisible_mask).float()
        return torch.cat([normalized_xy, obs_conf.unsqueeze(-1)], dim=-1)

    def _embed_with_exists(self, key, embedded, exists_mask):
        """Apply cond_exists_embedder if configured for this key."""
        if (
            self.use_cond_exists
            and hasattr(self, "cond_exists_embedder")
            and key in self.cond_exists_embedder
        ):
            embedded = torch.cat([embedded, exists_mask.float()], dim=-1)
            embedded = self.cond_exists_embedder[key](embedded)
        return embedded

    def _embed_obs(self, obs_normed):
        """Compute the noisy-obs embedding (no cond_exists for obs)."""
        B, L = obs_normed.shape[:2]
        vis_mask = obs_normed[..., 2] > 0.5  # (B, L, J)
        obs_c = obs_normed * vis_mask.unsqueeze(-1).float()
        f_obs = self.learned_pos_linear(obs_c[..., :2])  # (B, L, J, 32)
        vis_mask_4d = vis_mask.unsqueeze(-1).float()
        f_obs = f_obs * vis_mask_4d + self.learned_pos_params * (1.0 - vis_mask_4d)
        f_obs = self.embed_noisyobs(f_obs.reshape(B, L, -1))  # (B, L, latent_dim)
        return f_obs

    def _build_f_cond(self, obs_normed, f_cliffcam_raw, f_imgseq, f_cam_angvel, imgseq_mode="full"):
        """Compose f_cond = sum over in_attr of embedded conditions.

        ``imgseq_mode`` is "full" (use f_imgseq) or "absent" (zero embedding,
        zero exists-flag — bakes in the no-image learned bias).
        """
        B, L = obs_normed.shape[:2]
        device = obs_normed.device
        ones_bl = torch.ones(B, L, 1, device=device)
        zeros_bl = torch.zeros(B, L, 1, device=device)

        f_cond = torch.zeros(B, L, self.latent_dim, device=device)

        for k in self.in_attr:
            if k == "obs":
                f_cond = f_cond + self._embed_obs(obs_normed)

            elif k == "f_cliffcam":
                emb = self.cliffcam_embedder(f_cliffcam_raw)
                emb = self._embed_with_exists(k, emb, ones_bl)
                f_cond = f_cond + emb

            elif k == "f_imgseq":
                emb = self.imgseq_embedder(f_imgseq)
                if imgseq_mode == "absent":
                    emb = emb * 0.0
                    emb = self._embed_with_exists(k, emb, zeros_bl)
                else:
                    emb = self._embed_with_exists(k, emb, ones_bl)
                f_cond = f_cond + emb

            elif k == "f_cam_angvel":
                emb = self.cam_angvel_embedder(f_cam_angvel)
                emb = self._embed_with_exists(k, emb, ones_bl)
                f_cond = f_cond + emb

        return f_cond

    def _run_denoiser(self, f_cond):
        B, L = f_cond.shape[:2]
        device = f_cond.device

        xt_dim = self.denoiser.add_cond_linear.in_features - self.latent_dim
        xt = torch.zeros(B, L, xt_dim, device=device)
        timesteps = torch.full((B,), self.last_t, dtype=torch.long, device=device)

        y = {
            "f_cond": f_cond,
            "length": torch.full((B,), L, dtype=torch.long, device=device),
        }
        if getattr(self.denoiser, "encode_text", False):
            text_dim = int(getattr(self.denoiser, "encoded_text_dim", 1024))
            y["encoded_text"] = torch.zeros(B, L, text_dim, device=device)

        out = self.denoiser(
            xt, timesteps, y=y, inputs={},
            sample_indices_dict=self._sample_indices_dict,
        )
        pred_x = out["pred_x"]
        pred_cam = out.get("pred_cam", None)
        if pred_cam is None:
            pred_cam = torch.zeros(B, L, 3, device=device)
        return pred_x, pred_cam

    def forward(self, obs, bbx_xys, K_fullimg, f_imgseq, f_cam_angvel):
        obs_normed = self._normalize_kp2d(obs, bbx_xys)
        f_cliffcam_raw = compute_bbox_info_bedlam(bbx_xys, K_fullimg)
        if self.normalize_cam_angvel:
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std

        f_cond = self._build_f_cond(
            obs_normed, f_cliffcam_raw, f_imgseq, f_cam_angvel, imgseq_mode="full",
        )
        return self._run_denoiser(f_cond)


class GEMSMPLDenoiserNoImgWrapper(GEMSMPLDenoiserWrapper):
    """Variant that bakes in has_img_mask=False (no HMR2 features needed at runtime).

    Inputs are identical to the full wrapper; ``f_imgseq`` is consumed by the
    graph but its contribution is zeroed and the cond_exists path is fed
    a zero exists-flag. This reproduces the training-time augmentation with
    ``mask_img_prob`` so the published checkpoint is calibrated for it.
    """

    def forward(self, obs, bbx_xys, K_fullimg, f_imgseq, f_cam_angvel):
        obs_normed = self._normalize_kp2d(obs, bbx_xys)
        f_cliffcam_raw = compute_bbox_info_bedlam(bbx_xys, K_fullimg)
        if self.normalize_cam_angvel:
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std

        f_cond = self._build_f_cond(
            obs_normed, f_cliffcam_raw, f_imgseq, f_cam_angvel, imgseq_mode="absent",
        )
        return self._run_denoiser(f_cond)


def parse_args():
    parser = argparse.ArgumentParser(description="Export GEM-SMPL denoiser to ONNX")
    parser.add_argument("--ckpt", type=str, default=None, help="GEM-SMPL Lightning checkpoint")
    parser.add_argument("--exp", type=str, default="gem_smpl_regression",
                        help="Hydra experiment config (gem_smpl_regression or gem_smpl)")
    parser.add_argument("--output", type=str, default="inputs/onnx/gem_smpl_denoiser.onnx")
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--no-imgfeat", dest="no_imgfeat", action="store_true",
                        help="Bake in has_img_mask=False; output renamed to *_no_imgfeat.onnx")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    cfg_dir = str(PROJECT_ROOT / "configs")
    overrides = [
        f"exp={args.exp}",
        "video_name=export_dummy",
        "use_wandb=false",
        "task=test",
    ]
    if args.ckpt is not None:
        overrides.append(f"ckpt_path={args.ckpt}")

    import builtins
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", builtins.eval, replace=True)

    from hydra import compose, initialize_config_dir
    import hydra as _hydra

    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="demo", overrides=overrides)

    print(f"[Export] Instantiating model from exp={args.exp} ...")
    model = _hydra.utils.instantiate(cfg.model, _recursive_=False)
    ckpt_path = args.ckpt or cfg.ckpt_path
    if ckpt_path is None:
        from gem.utils.hf_utils import download_checkpoint
        ckpt_path = download_checkpoint()
    model.load_pretrained_model(ckpt_path)
    model = model.eval().cuda()

    if args.no_imgfeat:
        wrapper = GEMSMPLDenoiserNoImgWrapper(model).eval().cuda()
        print("[Export] Using no-imgfeat wrapper (has_img_mask=False baked in)")
    else:
        wrapper = GEMSMPLDenoiserWrapper(model).eval().cuda()

    B, L = 1, args.seq_len
    device = torch.device("cuda")
    obs = torch.randn(B, L, model.obs_num_joints, 3, device=device)
    bbx_xys = torch.ones(B, L, 3, device=device) * 100.0
    K_fullimg = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1).contiguous()
    K_fullimg[..., 0, 0] = 500.0
    K_fullimg[..., 1, 1] = 500.0
    K_fullimg[..., 0, 2] = 320.0
    K_fullimg[..., 1, 2] = 240.0
    f_imgseq = torch.randn(B, L, 1024, device=device)
    f_cam_angvel = torch.randn(B, L, 6, device=device)

    dummy_args = (obs, bbx_xys, K_fullimg, f_imgseq, f_cam_angvel)
    input_names = ["obs", "bbx_xys", "K_fullimg", "f_imgseq", "f_cam_angvel"]
    dynamic_axes = {n: {0: "B", 1: "L"} for n in input_names + ["pred_x", "pred_cam"]}

    pred_x, pred_cam = wrapper(*dummy_args)
    print(f"[Export] Forward pass OK: pred_x={tuple(pred_x.shape)}, pred_cam={tuple(pred_cam.shape)}")

    output_path = args.output
    if args.no_imgfeat and output_path == "inputs/onnx/gem_smpl_denoiser.onnx":
        output_path = "inputs/onnx/gem_smpl_denoiser_no_imgfeat.onnx"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_args,
        output_path,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["pred_x", "pred_cam"],
        dynamic_axes=dynamic_axes,
    )
    print(f"[Export] Wrote {output_path}")

    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(
            output_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        ort_inputs = {n: dummy_args[i].cpu().numpy() for i, n in enumerate(input_names)}
        ort_out = sess.run(None, ort_inputs)
        print(f"[Export] ONNX validation OK: pred_x={ort_out[0].shape}, pred_cam={ort_out[1].shape}")

        max_diff_x = np.abs(pred_x.cpu().numpy() - ort_out[0]).max()
        max_diff_cam = np.abs(pred_cam.cpu().numpy() - ort_out[1]).max()
        print(f"[Export] Max diff: pred_x={max_diff_x:.6f}, pred_cam={max_diff_cam:.6f}")
    except ImportError:
        print("[Export] onnxruntime not installed; skipping validation")


if __name__ == "__main__":
    main()
