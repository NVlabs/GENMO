import os
import time

from hydra.utils import instantiate
import numpy as np
import pytorch_lightning as pl
from timm.models.vision_transformer import Mlp
import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer

from genmo.network.base_arch.transformer.layer import BasicBlock, zero_module
from genmo.utils.geo_transform import get_bbx_xys, normalize_kp2d
from genmo.utils.net_utils import length_to_mask
from genmo.utils.pylogger import Log
from genmo.utils.tools import Timer
from third_party.GVHMR.hmr4d.model.gvhmr.utils import stats_compose
from third_party.GVHMR.hmr4d.model.gvhmr.utils.postprocess import (
    pp_static_joint,
    process_ik,
)
from third_party.GVHMR.hmr4d.utils.geo.augment_noisy_pose import (
    get_invisible_legs_mask,
    get_visible_mask,
    get_wham_aug_kp3d,
    randomly_modify_hands_legs,
)
from third_party.GVHMR.hmr4d.utils.geo.flip_utils import avg_smplx_aa, flip_smplx_params
from third_party.GVHMR.hmr4d.utils.geo.hmr_cam import (
    compute_bbox_info_bedlam,
    perspective_projection,
    safely_render_x3d_K,
)
from third_party.GVHMR.hmr4d.utils.smplx_utils import make_smplx

reproj_z_thr = 0.3


class GENMO(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        optimizer=None,
        scheduler=None,
        model_cfg=None,
        ignored_weights_prefix=["smplx", "pipeline.endecoder"],
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.endecoder = self.pipeline.endecoder
        self.optimizer = instantiate(optimizer, _partial_=True)
        self.model_cfg = model_cfg
        self.scheduler = scheduler
        self.enable_test_time_opt = model_cfg.get("enable_test_time_opt", False)
        self.train_modes = model_cfg.get("train_modes", [])
        if isinstance(self.train_modes, str):
            self.train_modes = [self.train_modes]

        # Options
        self.ignored_weights_prefix = ignored_weights_prefix

        # The test step is the same as validation
        self.test_step = self.predict_step = self.validation_step
        self.timing = os.environ.get("DEBUG_TIMING", "FALSE") == "TRUE"

        # SMPLX
        self.smplx = make_smplx("supermotion_v437coco17")

        if "text_encoder" in model_cfg:
            self.max_text_len = model_cfg.text_encoder.max_text_len

            self.use_text_encoder = True
            if model_cfg.text_encoder.get("load_llm", False):
                llm_version = model_cfg.text_encoder.llm_version
                self.max_text_len = model_cfg.text_encoder.max_text_len
                text_encoder, self.tokenizer = self.load_and_freeze_llm(llm_version)
                self.text_encoder = [text_encoder.cuda()]
            else:
                self.text_encoder = self.tokenizer = None
        else:
            self.use_text_encoder = False

        self.f_condition_dim = {
            "obs": (17, 3),
            "f_cliffcam": (3,),
            "f_cam_angvel": (6,),
            "f_cam_t_vel": (3,),
            "f_imgseq": (1024,),
            # "encoded_music": 438,
            "encoded_music": (self.pipeline.args.encoded_music_dim,),
            "encoded_audio": (128,),
            "observed_motion_3d": (151,),
        }

        self.not_add_features = [
            "obs",
            "f_cliffcam",
            "f_cam_angvel",
            "f_cam_t_vel",
            "f_imgseq",
            "observed_motion_3d",
            "multi_text_embed",
            "encoded_music",
            "encoded_audio",
        ]

        dropout = self.pipeline.args_denoiser3d.get("dropout", 0.1)
        latent_dim = self.pipeline.args_denoiser3d.get("latent_dim", 512)
        self.latent_dim = latent_dim
        if "obs" in self.pipeline.args.in_attr:
            self.learned_pos_linear = nn.Linear(2, 32)
            self.learned_pos_params = nn.Parameter(
                torch.randn(17, 32), requires_grad=True
            )
            self.embed_noisyobs = Mlp(
                17 * 32,
                hidden_features=latent_dim * 2,
                out_features=latent_dim,
                drop=dropout,
            )

        if "f_cliffcam" in self.pipeline.args.in_attr:
            self.cliffcam_embedder = nn.Sequential(
                nn.Linear(self.f_condition_dim["f_cliffcam"][0], latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )

        if "f_imgseq" in self.pipeline.args.in_attr:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.f_condition_dim["f_imgseq"][0]),
                zero_module(nn.Linear(self.f_condition_dim["f_imgseq"][0], latent_dim)),
            )

        if "f_cam_angvel" in self.pipeline.args.in_attr:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.f_condition_dim["f_cam_angvel"][0], latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )

        if "f_cam_t_vel" in self.pipeline.args.in_attr:
            self.cam_t_vel_embedder = nn.Sequential(
                nn.Linear(self.f_condition_dim["f_cam_t_vel"][0], latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )

        if "encoded_music" in self.pipeline.args.in_attr:
            self.music_embedder = Mlp(
                self.f_condition_dim["encoded_music"][0],
                hidden_features=latent_dim * 2,
                out_features=latent_dim,
                drop=dropout,
            )
            self.music_mask_prob = model_cfg.music_mask_prob

        if "encoded_audio" in self.pipeline.args.in_attr:
            self.audio_encoder = torch.nn.Sequential(
                BasicBlock(1, 32, 15, 5),
                BasicBlock(32, 32, 15, 6),
                BasicBlock(32, 32, 15, 1),
                BasicBlock(32, 64, 15, 5),
                BasicBlock(64, 64, 15, 1),
                BasicBlock(64, 128, 15, 4),
            )
            self.audio_embedder = nn.Sequential(
                nn.LayerNorm(self.f_condition_dim["encoded_audio"][0]),
                zero_module(
                    nn.Linear(self.f_condition_dim["encoded_audio"][0], latent_dim)
                ),
            )

            self.audio_mask_prob = model_cfg.audio_mask_prob

        if "multi_text_embed" in self.pipeline.args.in_attr:
            multi_text_module_cfg = model_cfg.get("multi_text_module_cfg", {})
            text_embed_dim = multi_text_module_cfg.get("text_embed_dim", 1024)
            self.multi_text_embedder = nn.Linear(text_embed_dim, latent_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,  # Input dimension
                nhead=multi_text_module_cfg.get(
                    "nhead", 8
                ),  # Number of attention heads
                dim_feedforward=multi_text_module_cfg.get("dim_feedforward", 2048),
                dropout=dropout,
                batch_first=True,
            )
            self.multi_text_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=multi_text_module_cfg.get("num_layers", 3)
            )

        self.condition_source = {
            "image": ["f_imgseq"],
            "2d": ["obs", "f_cliffcam"],
            "camera": ["f_cam_angvel", "f_cam_tvel"],
            "audio": ["encoded_audio"],
            "music": ["encoded_music"],
        }

        if self.model_cfg.normalize_cam_angvel:
            cam_angvel_stats = stats_compose.cam_angvel["manual"]
            self.register_buffer(
                "cam_angvel_mean",
                torch.tensor(cam_angvel_stats["mean"]),
                persistent=False,
            )
            self.register_buffer(
                "cam_angvel_std",
                torch.tensor(cam_angvel_stats["std"]),
                persistent=False,
            )

        # Load normalizer stats
        self.normalizer_stats = {}
        if "norm_attr_stats" in self.model_cfg:
            for key, stats_path in self.model_cfg.norm_attr_stats.items():
                self.normalizer_stats[key] = torch.load(
                    stats_path, map_location="cpu", weights_only=False
                )

        self.no_exist_keys = ["obs", "observed_motion_3d", "multi_text_embed"]
        # self.no_exist_keys = ["observed_motion_3d", "multi_text_embed"]
        if self.model_cfg.use_cond_exists_as_input:
            if self.model_cfg.cond_merge_strategy == "add":
                self.cond_exists_embedder = nn.ModuleDict()
                for k in self.pipeline.args.in_attr:
                    if k not in self.no_exist_keys:
                        self.cond_exists_embedder[k] = nn.Sequential(
                            nn.Linear(latent_dim + 1, latent_dim),
                            nn.SiLU(),
                            zero_module(nn.Linear(latent_dim, latent_dim)),
                        )
            elif self.model_cfg.cond_merge_strategy == "concat":
                raise NotImplementedError("Concat is not implemented")

    def normalize_attr(self, x, key):
        """Normalize input tensor using stored statistics"""
        mean = self.normalizer_stats[key]["mean"].to(x)
        std = self.normalizer_stats[key]["std"].to(x)
        return (x - mean) / std

    def load_and_freeze_llm(self, llm_version):
        tokenizer = T5Tokenizer.from_pretrained(llm_version)
        model = T5EncoderModel.from_pretrained(llm_version)
        # Freeze llm weights
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model, tokenizer

    def encode_text(self, raw_text, has_text=None):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                max_text_len = self.max_text_len

                encoded = self.tokenizer.batch_encode_plus(
                    raw_text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_text_len,
                    truncation=True,
                )
                # We expect all the processing is done in GPU.
                input_ids = encoded.input_ids.to(device)
                attn_mask = encoded.attention_mask.to(device)

                with torch.no_grad():
                    output = self.text_encoder[0](
                        input_ids=input_ids, attention_mask=attn_mask
                    )
                    encoded_text = output.last_hidden_state.detach()

                encoded_text = encoded_text[:, :max_text_len]
                attn_mask = attn_mask[:, :max_text_len]
                encoded_text *= attn_mask.unsqueeze(-1)
                # for bnum in range(encoded_text.shape[0]):
                #     nvalid_elem = attn_mask[bnum].sum().item()
                #     encoded_text[bnum][nvalid_elem:] = 0
        if has_text is not None:
            no_text = ~has_text
            encoded_text[no_text] = 0
        return encoded_text

    def generate_mask(self, mask_cfg, orig_mask, length):
        _cfg = mask_cfg
        mask = torch.ones_like(orig_mask)
        drop_prob = _cfg.get("drop_prob", 0.0)
        if drop_prob <= 0:
            return mask
        max_num_drops = _cfg.get("max_num_drops", 1)
        min_drop_nframes = _cfg.get("min_drop_nframes", 1)
        max_drop_nframes = _cfg.get("max_drop_nframes", 30)
        joint_drop_prob = _cfg.get("joint_drop_prob", 0.0)
        for i in range(orig_mask.shape[0]):
            mlen = length[i].item()
            if np.random.rand() < drop_prob:
                num_drops = np.random.randint(1, max_num_drops + 1)
                for _ in range(num_drops):
                    drop_len = np.random.randint(
                        min_drop_nframes, min(max_drop_nframes, mlen) + 1
                    )
                    drop_start = np.random.randint(0, max(mlen - drop_len, 1))
                    if joint_drop_prob > 0:
                        drop_joints = np.random.rand(17) < joint_drop_prob
                        mask[i, drop_start : drop_start + drop_len, drop_joints] = False
                    else:
                        mask[i, drop_start : drop_start + drop_len] = False
                    # print(f"Drop {i} {drop_start} {drop_len}")
        if joint_drop_prob > 0:
            COCO17_TREE = [
                [5, 6],
                0,
                0,
                1,
                2,
                -1,
                -1,
                5,
                6,
                7,
                8,
                -1,
                -1,
                11,
                12,
                13,
                14,
                15,
                15,
                15,
                16,
                16,
                16,
            ]
            for child in range(17):
                parent = COCO17_TREE[child]
                if parent == -1:
                    continue
                if isinstance(parent, list):
                    mask[..., child] *= mask[..., parent[0]] * mask[..., parent[1]]
                else:
                    mask[..., child] *= mask[..., parent]
        return mask

    def training_step(self, batch, batch_idx):
        def append_mode_to_loss(outputs, mode, suffix=""):
            if suffix != "":
                suffix = f"_{suffix}"
            for k in list(outputs.keys()):
                if "_loss" in k or k in {"loss"}:
                    outputs[f"Loss_{mode}{suffix}/{k}"] = outputs.pop(k)
            return outputs

        outputs = {"loss": 0}

        with Timer("train_step", enabled=self.timing):
            for mode in self.train_modes:
                self.prepare_batch(batch, mode)  # set "obs" (2d keypoints)
                outputs_mode = self.train_step(batch, batch_idx, mode=mode)
                outputs["loss"] += outputs_mode["loss"]
                append_mode_to_loss(outputs_mode, mode)
                outputs.update(outputs_mode)
                if mode == "regression" and "diffusion" in self.train_modes:
                    batch["regression_outputs"] = outputs_mode.copy()
                # batch[f"{mode}_condition"] = outputs_mode[f"{mode}_condition"]

        # Log
        log_kwargs = {
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "sync_dist": True,
            "batch_size": outputs["batch_size"],
        }
        self.log("train/loss", outputs["loss"], **log_kwargs)
        for k, v in outputs.items():
            if "_loss" in k:
                self.log(f"{k}", v, **log_kwargs)

        return outputs

    def prepare_batch(self, batch, mode):
        target_x = self.endecoder.encode(batch)  # (B, L, C)
        if mode == "diffusion":
            target_x[batch["mask"]["2d_only"]] = batch["regression_outputs"][
                "model_output"
            ]["pred_x_start"][batch["mask"]["2d_only"]]
        else:
            target_x[batch["mask"]["2d_only"]] = 0
        valid_mask = batch["mask"]["valid"]
        target_x_mask = torch.ones_like(target_x).bool()
        target_x_mask[batch["mask"]["2d_only"]] = False
        target_x_mask[batch["mask"]["spv_incam_only"], :, 142:] = False
        target_x_mask = target_x_mask & valid_mask[:, :, None]

        batch["target_x"] = target_x
        batch["target_x_mask"] = target_x_mask

        batch["device"] = batch["target_x"].device
        batch["B"], batch["L"] = B, L = batch["target_x"].shape[:2]

        if "text_embed" in batch:
            batch["encoded_text"] = batch["text_embed"].cuda()
        elif self.use_text_encoder:
            batch["encoded_text"] = self.encode_text(
                batch["caption"], batch["has_text"]
            )

        # Create augmented noisy-obs : gt_j3d(coco17)
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                gt_verts437, gt_j3d = self.smplx(**batch["smpl_params_c"])
                root_ = gt_j3d[:, :, [11, 12], :].mean(-2, keepdim=True)
                batch["gt_j3d"] = gt_j3d
                batch["gt_cr_coco17"] = gt_j3d - root_
                batch["gt_c_verts437"] = gt_verts437
                batch["gt_cr_verts437"] = gt_verts437 - root_

            # compute bbx_xys from GT Vertices
            i_x2d = safely_render_x3d_K(gt_verts437, batch["K_fullimg"], thr=0.3)
        det_kp2d = batch["kp2d"]
        assert det_kp2d.ndim == 4 and det_kp2d.shape[-1] == 3, (
            f"det_kp2d.shape: {det_kp2d.shape}"
        )
        det_kp2d_conf = det_kp2d[..., 2]
        batch["det_kp2d_conf"] = det_kp2d_conf
        det_kp2d = det_kp2d[..., :2]
        det_kp2d_mask = det_kp2d_conf > 0.5
        bbx_xys = get_bbx_xys(i_x2d, do_augment=True)
        bbx_xys_detected = get_bbx_xys(det_kp2d, det_kp2d_mask, do_augment=True)
        bbx_xys[batch["mask"]["2d_only"]] = bbx_xys_detected[batch["mask"]["2d_only"]]

        # TODO: Need to compare
        if False:  # trust image bbx_xys seems better
            batch["bbx_xys"] = bbx_xys
        else:
            mask_bbx_xys = batch["mask"]["bbx_xys"]
            batch["bbx_xys"][~mask_bbx_xys] = bbx_xys[~mask_bbx_xys].to(
                batch["bbx_xys"]
            )

        with torch.cuda.amp.autocast(enabled=False):
            gt_kp2d = perspective_projection(gt_j3d, batch["K_fullimg"])  # (B, L, J, 2)
            gt_kp2d[batch["mask"]["2d_only"]] = det_kp2d[batch["mask"]["2d_only"]]
            gt_kp2d_normed = normalize_kp2d(gt_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
            valid_mask_j17 = (
                (gt_j3d[..., 2] > reproj_z_thr)
                * (gt_kp2d_normed[..., 0] > 0.0)
                * (gt_kp2d_normed[..., 0] < 1.0)
                * (gt_kp2d_normed[..., 1] > 0.0)
                * (gt_kp2d_normed[..., 1] < 1.0)
            )[..., None]

            batch["gt_kp2d_normed"] = gt_kp2d_normed
            batch["valid_mask_j17"] = valid_mask_j17

            noisy_j3d = gt_j3d + get_wham_aug_kp3d(gt_j3d.shape[:2])
            obs_i_j2d = perspective_projection(
                noisy_j3d, batch["K_fullimg"]
            )  # (B, L, J, 2)
            noisy_det_j2d = det_kp2d.clone()
            aug = get_wham_aug_kp3d(noisy_det_j2d.shape[:2])[..., :2]
            f = torch.tensor([1024.0, 1024.0]).to(aug) / 4.0
            aug *= f * self.model_cfg.kp2d_noise_scale
            noisy_det_j2d = noisy_det_j2d + aug

        # Use some detected vitpose (presave data)
        prob = 0.5
        mask_real_vitpose = (torch.rand(batch["B"]).to(obs_i_j2d) < prob) * batch[
            "mask"
        ]["vitpose"]
        mask_real_vitpose = mask_real_vitpose | batch["mask"]["2d_only"]

        assert batch["mask"]["2d_only"].sum() == 0, batch["mask"]["2d_only"].sum()

        obs_i_j2d[mask_real_vitpose] = noisy_det_j2d[mask_real_vitpose]

        obs_i_j2d = randomly_modify_hands_legs(obs_i_j2d)
        j2d_visible_mask = get_visible_mask(gt_j3d.shape[:2]).cuda()  # (B, L, J)
        j2d_visible_mask = j2d_visible_mask & batch["mask"]["has_2d_mask"][:, :, None]

        j2d_visible_mask[mask_real_vitpose] *= det_kp2d_mask[mask_real_vitpose]
        close_mask = (noisy_j3d[..., 2] < 0.3) & (~mask_real_vitpose)[:, None, None]
        j2d_visible_mask[close_mask] = (
            False  # Set close-to-image-plane points as invisible
        )

        legs_invisible_mask = get_invisible_legs_mask(
            gt_j3d.shape[:2]
        ).cuda()  # (B, L, J)
        j2d_visible_mask[legs_invisible_mask] = False
        if "mask_cfg" in self.model_cfg:
            mask = self.generate_mask(
                self.model_cfg.mask_cfg, j2d_visible_mask, batch["length"]
            )
            j2d_visible_mask = j2d_visible_mask & mask
        if "body_mask_cfg" in self.model_cfg:
            mask = self.generate_mask(
                self.model_cfg.body_mask_cfg, j2d_visible_mask, batch["length"]
            )
            j2d_visible_mask = j2d_visible_mask & mask

        occluded_img_mask = j2d_visible_mask.sum(dim=-1) <= 3
        f_cliffcam = compute_bbox_info_bedlam(
            batch["bbx_xys"], batch["K_fullimg"]
        )  # (B, L, 3)
        f_cliffcam[occluded_img_mask] = 0
        batch["f_cliffcam"] = f_cliffcam
        condition_mask = dict()
        condition_mask["has_img_mask"] = batch["mask"]["has_img_mask"] & (
            ~occluded_img_mask
        )
        condition_mask["has_2d_mask"] = batch["mask"]["has_2d_mask"] & (
            ~occluded_img_mask
        )
        condition_mask["has_cam_mask"] = batch["mask"]["has_cam_mask"].clone()
        condition_mask["has_audio_mask"] = batch["mask"]["has_audio_mask"].clone()
        condition_mask["has_music_mask"] = batch["mask"]["has_music_mask"].clone()
        batch["condition_mask"] = condition_mask

        obs_kp2d = torch.cat(
            [obs_i_j2d, j2d_visible_mask[:, :, :, None].float()], dim=-1
        )  # (B, L, J, 3)
        obs = normalize_kp2d(obs_kp2d, batch["bbx_xys"])  # (B, L, J, 3)
        obs[~j2d_visible_mask] = 0  # if not visible, set to (0,0,0)
        j2d_visible_mask[~batch["mask"]["valid"]] = False
        batch["obs"] = obs
        condition_mask["j2d_visible_mask"] = j2d_visible_mask
        batch["obs"][~batch["mask"]["valid"]] = 0

        batch["static_gt"] = self.endecoder.get_static_gt(
            batch, self.pipeline.args.static_conf.vel_thr
        )  # (B, L, 6)
        batch["static_gt_mask"] = ~batch["mask"]["invalid_contact"]

        f_cam_angvel = batch["cam_angvel"]
        if self.model_cfg.normalize_cam_angvel:
            f_cam_angvel = (f_cam_angvel - self.cam_angvel_mean) / self.cam_angvel_std
        batch["f_cam_angvel"] = f_cam_angvel

        for k in self.normalizer_stats:
            if k in batch:
                batch[k] = self.normalize_attr(batch[k], k)

    def create_condition_mask(
        self, batch, cond_mask_cfg, mode, train, first_k_frames=None
    ):
        B, L = batch["B"], batch["L"]
        device = batch["device"]

        has_text = batch["has_text"]
        condition_mask = batch["condition_mask"]
        has_img_mask = condition_mask["has_img_mask"].clone()
        has_2d_mask = condition_mask["has_2d_mask"].clone()
        has_cam_mask = condition_mask["has_cam_mask"].clone()
        has_audio_mask = condition_mask["has_audio_mask"].clone()
        has_music_mask = condition_mask["has_music_mask"].clone()
        j2d_visible_mask = condition_mask["j2d_visible_mask"].clone()

        if train:
            regression_no_img_mask = cond_mask_cfg.get("regression_no_img_mask", False)
            mask_text_prob = cond_mask_cfg.get("mask_text_prob", {}).get(mode, 0.0)
            mask_img_prob = cond_mask_cfg.get("mask_img_prob", 0.0)
            mask_cam_prob = cond_mask_cfg.get("mask_cam_prob", 0.0)
            mask_f_imgseq_prob = cond_mask_cfg.get("mask_f_imgseq_prob", 0.0)

            if mask_text_prob > 0:
                mask_text = (torch.rand(batch["B"]) < mask_text_prob).to(device)
                batch["text_mask"] = mask_text
            else:
                batch["text_mask"] = None
            if batch.get("text_mask", None) is not None:
                batch["has_text"][batch["text_mask"]] = False

            if regression_no_img_mask and mode == "regression":
                mask_img_prob = 0
                mask_f_imgseq_prob = 0
                has_2d_mask[~batch["mask"]["2d_only"]] = True

            if mask_img_prob > 0:
                mask_img = (has_text[:, None] | has_audio_mask | has_music_mask) & (
                    torch.rand(batch["B"]) < mask_img_prob
                ).to(device)[:, None]
                has_img_mask = has_img_mask & ~mask_img
                has_2d_mask = has_2d_mask & ~mask_img
                j2d_visible_mask = j2d_visible_mask & ~mask_img[..., None]

            if mask_cam_prob > 0:
                mask_cam = (has_text[:, None] | has_music_mask | has_audio_mask) & (
                    torch.rand(batch["B"]) < mask_cam_prob
                ).to(device)[:, None]
                has_cam_mask = has_cam_mask & ~mask_cam

            has_music_mask = (
                has_music_mask
                & (torch.rand((B,), device=device) > self.music_mask_prob)[:, None]
            )
            has_audio_mask = (
                has_audio_mask
                & (torch.rand((B,), device=device) > self.audio_mask_prob)[:, None]
            )

        j2d_visible_mask = j2d_visible_mask & has_2d_mask[:, :, None]
        has_2d_mask = j2d_visible_mask.sum(dim=-1) > 3

        f_condition_exists = dict()
        # f_condition = dict()
        for k in self.condition_source["image"]:
            f_condition_exists[k] = has_img_mask.clone()
        for k in self.condition_source["2d"]:
            if k == "obs":
                f_condition_exists[k] = j2d_visible_mask.clone()
            else:
                f_condition_exists[k] = has_2d_mask.clone()
        for k in self.condition_source["camera"]:
            f_condition_exists[k] = has_cam_mask.clone()
        for k in self.condition_source["audio"]:
            f_condition_exists[k] = has_audio_mask.clone()
        for k in self.condition_source["music"]:
            f_condition_exists[k] = has_music_mask.clone()

        if train and mask_f_imgseq_prob > 0:
            mask_f_imgseq = (torch.rand(batch["B"]) < mask_f_imgseq_prob).to(device)
            f_condition_exists["f_imgseq"] = f_condition_exists["f_imgseq"] & (
                ~mask_f_imgseq
            )

        # randomly set null condition
        skip_keys = self.pipeline.args.get("skip_keys_for_null_condition", [])
        uncond_prob = self.pipeline.args.get("uncond_prob", 0.1)
        if train and not self.pipeline.args.get("disable_random_null_condition", False):
            for k in self.pipeline.args.in_attr:
                if k in skip_keys:
                    continue
                mask = torch.rand(f_condition_exists[k].shape[:2]) < uncond_prob
                f_condition_exists[k][mask] = False

        f_cond_dict = {}
        f_uncond_dict = {}
        f_uncond_exists = {k: f_condition_exists[k].clone() for k in f_condition_exists}

        length = batch["length"]
        end_fr = first_k_frames if first_k_frames is not None else None
        if first_k_frames is not None:
            length = length.clamp(max=first_k_frames)
        for k in self.pipeline.args.in_attr:
            if k == "obs":
                obs = batch["obs"][:, :end_fr]
                B, L, J, C = obs.shape
                assert J == 17 and C == 3
                obs = obs.clone()
                obs = obs * j2d_visible_mask[:, :, :, None]
                visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
                obs[~visible_mask[..., 0]] = 0  # set low-conf to all zeros
                f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
                f_obs = (
                    f_obs * visible_mask
                    + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
                )  # (B, L, J, 32)
                f_obs = self.embed_noisyobs(
                    f_obs.view(B, L, -1)
                )  # (B, L, J*32) -> (B, L, C)
                f_cond_dict["obs"] = f_obs
                f_uncond_dict["obs"] = f_obs
            elif k == "f_cliffcam":
                f_cliffcam = batch["f_cliffcam"][:, :end_fr]  # (B, L, 3)
                f_cliffcam = self.cliffcam_embedder(f_cliffcam)
                mask = f_condition_exists[k][:, :, None]
                f_cond_dict["f_cliffcam"] = f_cliffcam * mask.float()
                f_uncond_dict["f_cliffcam"] = f_cliffcam * mask.float()
            elif k == "f_cam_angvel":
                f_cam_angvel = batch["f_cam_angvel"][:, :end_fr]  # (B, L, 6)
                f_cam_angvel = self.cam_angvel_embedder(f_cam_angvel)
                mask = f_condition_exists[k][:, :, None]
                f_cond_dict["f_cam_angvel"] = f_cam_angvel * mask.float()
                f_uncond_dict["f_cam_angvel"] = f_cam_angvel * mask.float()
            elif k == "f_cam_t_vel":
                f_cam_t_vel = batch["f_cam_t_vel"][:, :end_fr]  # (B, L, 3)
                f_cam_t_vel = self.cam_t_vel_embedder(f_cam_t_vel)
                mask = f_condition_exists[k][:, :, None]
                f_cond_dict["f_cam_t_vel"] = f_cam_t_vel * mask.float()
                f_uncond_dict["f_cam_t_vel"] = f_cam_t_vel * mask.float()
            elif k == "f_imgseq":
                f_imgseq = batch["f_imgseq"][:, :end_fr]  # (B, L, C)
                f_imgseq = self.imgseq_embedder(f_imgseq)
                mask = f_condition_exists[k][:, :, None]
                f_cond_dict["f_imgseq"] = f_imgseq * mask.float()
                f_uncond_dict["f_imgseq"] = f_imgseq * mask.float()
            elif k == "encoded_music":
                if "music_embed" in batch:
                    f_encoded_music = batch["music_embed"][:, :end_fr]  # (B, L, C)
                    f_encoded_music = self.music_embedder(f_encoded_music)
                    mask = f_condition_exists[k][:, :, None]
                    f_cond_dict["encoded_music"] = f_encoded_music * mask.float()
                else:
                    f_cond_dict["encoded_music"] = torch.zeros(
                        B, L, self.latent_dim
                    ).to(batch["device"])
                f_uncond_dict["encoded_music"] = torch.zeros(B, L, self.latent_dim).to(
                    batch["device"]
                )
                f_uncond_exists["encoded_music"] = torch.zeros_like(
                    f_condition_exists["encoded_music"]
                )
            elif k == "encoded_audio":
                if "audio_array" in batch:
                    encoded_audio = (
                        self.audio_encoder(batch["audio_array"].cuda().unsqueeze(1))
                        .transpose(1, 2)
                        .contiguous()
                    )[:, :end_fr]
                    mask = f_condition_exists[k][:, :, None]
                    encoded_audio = self.audio_embedder(encoded_audio)
                    f_cond_dict["encoded_audio"] = encoded_audio * mask.float()
                else:
                    f_cond_dict["encoded_audio"] = torch.zeros(
                        B, L, self.latent_dim
                    ).to(batch["device"])
                f_uncond_dict["encoded_audio"] = torch.zeros(B, L, self.latent_dim).to(
                    batch["device"]
                )
                f_uncond_exists["encoded_audio"] = torch.zeros_like(
                    f_condition_exists["encoded_audio"]
                )
            elif k == "observed_motion_3d":
                motion_mask_3d = batch.get(
                    "motion_mask_3d",
                    torch.zeros_like(batch["observed_motion_3d"]),
                )[:, :end_fr]
                f_observed_motion_3d = torch.cat(
                    [batch["observed_motion_3d"][:, :end_fr], motion_mask_3d],
                    dim=-1,
                )
                f_observed_motion_3d = self.observed_motion_3d_embedder(
                    f_observed_motion_3d
                )
                f_cond_dict["observed_motion_3d"] = f_observed_motion_3d
                f_uncond_dict["observed_motion_3d"] = torch.zeros_like(
                    f_observed_motion_3d
                )
            else:
                assert False, f"Unknown condition key: {k}"

            if k not in self.not_add_features:
                f_cond_dict[k] = self.add_feature_embedders[k](batch[k][:, :end_fr])

            if self.model_cfg.use_cond_exists_as_input:
                if k not in self.no_exist_keys:
                    if k == "obs":
                        exist_mask = f_condition_exists[k][:, :end_fr]
                        exist_mask = exist_mask.sum(dim=-1, keepdim=True) > 0
                        uncond_exist_mask = f_uncond_exists[k][:, :end_fr]
                        uncond_exist_mask = (
                            uncond_exist_mask.sum(dim=-1, keepdim=True) > 0
                        )
                    else:
                        exist_mask = f_condition_exists[k][:, :end_fr, None]
                        uncond_exist_mask = f_uncond_exists[k][:, :end_fr, None]
                    f_cond_dict[k] = torch.cat(
                        [
                            f_cond_dict[k],
                            exist_mask.float(),
                        ],
                        dim=-1,
                    )
                    f_cond_dict[k] = self.cond_exists_embedder[k](f_cond_dict[k])
                    f_uncond_dict[k] = torch.cat(
                        [f_uncond_dict[k], uncond_exist_mask.float()],
                        dim=-1,
                    )
                    f_uncond_dict[k] = self.cond_exists_embedder[k](f_uncond_dict[k])

        f_cond = sum(f_cond_dict.values())
        f_uncond = sum(f_uncond_dict.values())
        batch["f_cond"] = f_cond
        batch["f_uncond"] = f_uncond

        if batch.get("text_mask", None) is not None:
            batch["encoded_text"] = batch["encoded_text"] * (
                1 - batch["text_mask"][:, None, None].float()
            )
        vis_mask = length_to_mask(length, f_cond.shape[1])[:, :end_fr]  # (B, L)
        motion = batch["target_x"] * vis_mask[..., None]
        batch["motion"] = motion[:, :end_fr]

        return batch

    def train_step(self, batch, batch_idx, mode):
        batch = batch.copy()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.detach().clone()

        cond_mask_cfg = self.model_cfg.get("condition_mask", {})
        batch = self.create_condition_mask(batch, cond_mask_cfg, mode, train=True)

        # Forward and get loss
        outputs = self.pipeline.forward(
            batch,
            train=True,
            global_step=self.trainer.global_step,
            mode=mode,
            normalizer_stats=self.normalizer_stats,
        )
        outputs["batch_size"] = batch["B"]
        return outputs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        test_mode = batch["meta"][0].get("mode", "default")
        return self.validation(batch, test_mode, batch_idx, dataloader_idx)

    def validation(self, batch, test_mode, batch_idx, dataloader_idx=0):
        # Options & Check
        try:
            stage = self.trainer.state.stage
            global_step = self.trainer.global_step
        except Exception:
            stage = "test"
            global_step = 0
        do_postproc = stage == "test"  # Only apply postproc in test
        do_flip_test = "flip_test" in batch
        do_postproc_not_flip_test = (
            do_postproc and not do_flip_test
        )  # later pp when flip_test

        # ROPE inference
        obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
        B, L = obs.shape[:2]

        if "mask" in batch:
            mask = batch["mask"]
            if isinstance(mask, dict):
                mask = mask["valid"]
            obs[0, ~mask[0]] = 0

        test_mode = batch["meta"][0].get("mode", "default")
        batch_ = {
            "length": batch["length"],
            "obs": obs,
            "bbx_xys": batch["bbx_xys"],
            "K_fullimg": batch["K_fullimg"],
            "cam_angvel": batch["cam_angvel"].clone(),
            "f_cam_angvel": batch["cam_angvel"].clone(),
            "f_imgseq": batch["f_imgseq"],
            "caption": batch.get("caption", [""] * B),
            "has_text": batch.get("has_text", torch.zeros(B).to(obs.device).bool()),
            # "eval_gen_only": eval_gen_only,
            "mode": test_mode,
            "meta": batch["meta"],
            "B": batch["B"],
            "L": obs.shape[1],
            "device": obs.device,
            "target_x": torch.zeros(B, L, self.endecoder.get_motion_dim()).to(
                obs.device
            ),
            "mask": batch["mask"],
        }
        if "music_embed" in batch:
            batch_["music_embed"] = batch["music_embed"]
        if "audio_array" in batch:
            batch_["audio_array"] = batch["audio_array"]
        det_kp2d = batch["kp2d"]
        det_kp2d_conf = det_kp2d[..., 2]
        j2d_visible_mask = det_kp2d_conf > 0.5
        f_cliffcam = compute_bbox_info_bedlam(
            batch_["bbx_xys"], batch_["K_fullimg"]
        )  # (B, L, 3)
        batch_["f_cliffcam"] = f_cliffcam

        condition_mask = dict()
        condition_mask["has_img_mask"] = batch["mask"]["has_img_mask"]
        condition_mask["has_2d_mask"] = batch["mask"]["has_2d_mask"]
        condition_mask["has_cam_mask"] = batch["mask"]["has_cam_mask"].clone()
        condition_mask["has_audio_mask"] = batch["mask"]["has_audio_mask"].clone()
        condition_mask["has_music_mask"] = batch["mask"]["has_music_mask"].clone()
        condition_mask["j2d_visible_mask"] = j2d_visible_mask
        batch_["condition_mask"] = condition_mask

        if self.model_cfg.normalize_cam_angvel:
            batch_["f_cam_angvel"] = (
                batch_["f_cam_angvel"] - self.cam_angvel_mean
            ) / self.cam_angvel_std

        if "text_embed" in batch:
            batch_["encoded_text"] = batch["text_embed"].cuda()
        elif self.use_text_encoder:
            batch_["encoded_text"] = self.encode_text(
                batch["caption"], batch["has_text"]
            )

        if test_mode == "infilling":
            batch["target_x"] = self.endecoder.encode(batch)  # (B, L, C)
            rng = np.random.RandomState(
                batch["meta"][0].get("eval_seed", 7) + batch_idx
            )
            assert "motion_3d_mask_cfg" in self.model_cfg
            all_mask_types = [
                x
                for x in self.model_cfg.motion_3d_mask_cfg.mask_types
                if x != "no_mask"
            ]
            use_mask_type = all_mask_types[batch_idx % len(all_mask_types)]
            mask_res = self.generate_motion_3d_mask(
                self.model_cfg.motion_3d_mask_cfg,
                batch["target_x"],
                batch["length"],
                rng=rng,
                use_mask_type=use_mask_type,
            )
            batch_.update(mask_res)

        if "inpainting_3d" in self.model_cfg:
            batch_["observed_motion_3d"] = self.endecoder.encode(batch)
            motion_mask_3d = torch.zeros_like(batch_["observed_motion_3d"]).cuda()
            L = batch["length"][0]
            keyframes = [i for i in range(L)]
            if self.model_cfg["inpainting_3d"]["mode"] == "body_pose_dense":
                motion_mask_3d[:, :, : 126 + 10] = 1
            elif self.model_cfg["inpainting_3d"]["mode"] == "body_pose_root_rot_dense":
                motion_mask_3d[:, :, : 126 + 10 + 12] = 1
            elif (
                self.model_cfg["inpainting_3d"]["mode"]
                == "body_pose_root_rot_keyframe2"
            ):
                # keyframes = [0, L-1] # start and fix end
                keyframes = [
                    0,
                    np.random.choice(keyframes[L // 2 :], 1)[0],
                ]  # start and random end
                motion_mask_3d[:, keyframes, : 126 + 10 + 12] = 1
            elif (
                self.model_cfg["inpainting_3d"]["mode"]
                == "body_pose_root_rot_keyframe5"
            ):
                keyframes = [int((L - 1) * i / 4) for i in range(5)]
                motion_mask_3d[:, keyframes, : 126 + 10 + 12] = 1
            elif self.model_cfg["inpainting_3d"]["mode"] == "root_rot_vel_dense":
                motion_mask_3d[:, :, 126:] = 1
            else:
                raise ValueError(
                    f"Unknown inpainting mode [{self.model_cfg['inpainting_3d']['mode']}]"
                )
            batch_["motion_mask_3d"] = motion_mask_3d
            batch["keyframes"] = keyframes

        for k in self.normalizer_stats:
            if k in batch_:
                batch_[k] = self.normalize_attr(batch_[k], k)

        batch_ = self.create_condition_mask(
            batch_, cond_mask_cfg=None, mode=None, train=False
        )

        outputs = self.pipeline.forward(
            batch_,
            train=False,
            postproc=do_postproc_not_flip_test,
            global_step=global_step,
            test_mode=test_mode,
        )
        if "pred_smpl_params_global" in outputs:
            outputs["pred_smpl_params_global"] = {
                k: v[0] for k, v in outputs["pred_smpl_params_global"].items()
            }
        if "pred_smpl_params_incam" in outputs:
            outputs["pred_smpl_params_incam"] = {
                k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()
            }

        if test_mode == "infilling":
            outputs.update(mask_res)

        if do_flip_test:
            flip_test = batch["flip_test"]
            obs = normalize_kp2d(flip_test["kp2d"], flip_test["bbx_xys"])
            if "mask" in batch:
                mask = batch["mask"]
                if isinstance(mask, dict):
                    mask = mask["valid"]
                obs[0, ~mask[0]] = 0

            batch_ = {
                "length": batch["length"],
                "obs": obs,
                "bbx_xys": flip_test["bbx_xys"],
                "K_fullimg": batch["K_fullimg"],
                "cam_angvel": flip_test["cam_angvel"].clone(),
                "f_cam_angvel": flip_test["cam_angvel"].clone(),
                "f_imgseq": flip_test["f_imgseq"],
                "caption": flip_test.get("caption", [""] * B),
                "has_text": flip_test.get(
                    "has_text", torch.zeros(B).to(obs.device).bool()
                ),
                "meta": batch["meta"],
                "B": batch["B"],
                "L": obs.shape[1],
                "device": obs.device,
                "target_x": torch.zeros(B, L, self.endecoder.get_motion_dim()).to(
                    obs.device
                ),
                "mask": batch["mask"],
            }

            det_kp2d = flip_test["kp2d"]
            det_kp2d_conf = det_kp2d[..., 2]
            j2d_visible_mask = det_kp2d_conf > 0.5

            f_cliffcam = compute_bbox_info_bedlam(
                batch_["bbx_xys"], batch_["K_fullimg"]
            )  # (B, L, 3)
            batch_["f_cliffcam"] = f_cliffcam

            condition_mask = dict()
            condition_mask["has_img_mask"] = batch["mask"]["has_img_mask"]
            condition_mask["has_2d_mask"] = batch["mask"]["has_2d_mask"]
            condition_mask["has_cam_mask"] = batch["mask"]["has_cam_mask"].clone()
            condition_mask["has_audio_mask"] = batch["mask"]["has_audio_mask"].clone()
            condition_mask["has_music_mask"] = batch["mask"]["has_music_mask"].clone()
            condition_mask["j2d_visible_mask"] = j2d_visible_mask
            batch_["condition_mask"] = condition_mask

            if self.model_cfg.normalize_cam_angvel:
                batch_["f_cam_angvel"] = (
                    batch_["f_cam_angvel"] - self.cam_angvel_mean
                ) / self.cam_angvel_std
            for k in self.normalizer_stats:
                if k in batch_:
                    batch_[k] = self.normalize_attr(batch_[k], k)

            if "text_embed" in batch:
                batch_["encoded_text"] = batch["text_embed"].cuda()
            elif self.use_text_encoder:
                batch_["encoded_text"] = self.encode_text(
                    batch["caption"], batch["has_text"]
                )
            batch_ = self.create_condition_mask(
                batch_, cond_mask_cfg=None, mode=None, train=False
            )

            flipped_outputs = self.pipeline.forward(
                batch_, train=False, global_step=global_step, test_mode=test_mode
            )
            # First update incam results
            flipped_outputs["pred_smpl_params_incam"] = {
                k: v[0] for k, v in flipped_outputs["pred_smpl_params_incam"].items()
            }
            smpl_params1 = outputs["pred_smpl_params_incam"]
            smpl_params2 = flip_smplx_params(flipped_outputs["pred_smpl_params_incam"])

            smpl_params_avg = smpl_params1.copy()
            smpl_params_avg["betas"] = (
                smpl_params1["betas"] + smpl_params2["betas"]
            ) / 2
            smpl_params_avg["body_pose"] = avg_smplx_aa(
                smpl_params1["body_pose"], smpl_params2["body_pose"]
            )
            smpl_params_avg["global_orient"] = avg_smplx_aa(
                smpl_params1["global_orient"], smpl_params2["global_orient"]
            )
            outputs["pred_smpl_params_incam"] = smpl_params_avg

            # Then update global results
            outputs["pred_smpl_params_global"]["betas"] = smpl_params_avg["betas"]
            outputs["pred_smpl_params_global"]["body_pose"] = smpl_params_avg[
                "body_pose"
            ]

            # Finally, apply postprocess
            if do_postproc:
                # temporarily recover the original batch-dim
                outputs["pred_smpl_params_global"] = {
                    k: v[None] for k, v in outputs["pred_smpl_params_global"].items()
                }
                outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(
                    outputs, self.pipeline.endecoder
                )
                body_pose = process_ik(outputs, self.pipeline.endecoder)
                outputs["pred_smpl_params_global"] = {
                    k: v[0] for k, v in outputs["pred_smpl_params_global"].items()
                }

                outputs["pred_smpl_params_global"]["body_pose"] = body_pose[0]
                # outputs["pred_smpl_params_incam"]["body_pose"] = body_pose[0]

        return outputs

    @torch.no_grad()
    def predict(self, data, static_cam=False, postproc=True):
        now = time.time()
        # ROPE inference
        test_mode = data["meta"][0].get("mode", "default")
        batch = {
            "length": data["length"][None].cuda(),
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None].cuda(),
            "bbx_xys": data["bbx_xys"][None].cuda(),
            "K_fullimg": data["K_fullimg"][None].cuda(),
            "cam_angvel": data["cam_angvel"][None].cuda(),
            "f_cam_angvel": data["cam_angvel"][None].cuda(),
            "cam_tvel": data["cam_tvel"][None].cuda(),
            "R_w2c": data["R_w2c"][None].cuda(),
            "f_imgseq": data["f_imgseq"][None].cuda(),
            # "text_embed": data["text_embed"][None].cuda(),
            "has_text": data["has_text"].cuda(),
            "B": 1,
            "L": data["f_imgseq"].shape[0],
            "mode": test_mode,
            "target_x": torch.zeros(
                1, data["f_imgseq"].shape[0], self.endecoder.get_motion_dim()
            ).cuda(),
        }
        if "music_embed" in data:
            batch["music_embed"] = data["music_embed"][None].cuda()
        if "audio_array" in data:
            batch["audio_array"] = data["audio_array"][None].cuda()

        if "fast_rollout" in data:
            batch["fast_rollout"] = data["fast_rollout"]
        batch["device"] = batch["f_imgseq"].device

        if "meta" in data:
            batch["meta"] = data["meta"]
        else:
            batch["meta"] = None

        if "text_embed" in batch:
            batch["encoded_text"] = batch["text_embed"].cuda()
        else:
            if "caption" in data:
                batch["caption"] = [data["caption"]]
            else:
                batch["caption"] = [""]
            batch["has_text"] = torch.tensor([True])
            batch["encoded_text"] = self.encode_text(
                batch["caption"], batch["has_text"]
            )

        batch["f_cliffcam"] = compute_bbox_info_bedlam(
            batch["bbx_xys"], batch["K_fullimg"]
        ).cuda()

        condition_mask = dict()
        condition_mask["has_img_mask"] = data["mask"]["has_img_mask"][None].cuda()
        condition_mask["has_2d_mask"] = data["mask"]["has_2d_mask"][None].cuda()
        condition_mask["has_cam_mask"] = (
            data["mask"]["has_cam_mask"][None].cuda().clone()
        )
        condition_mask["has_audio_mask"] = (
            data["mask"]["has_audio_mask"][None].cuda().clone()
        )
        condition_mask["has_music_mask"] = (
            data["mask"]["has_music_mask"][None].cuda().clone()
        )
        kp2d_conf = data["kp2d"][..., 2][None].cuda()
        condition_mask["j2d_visible_mask"] = kp2d_conf > 0.7
        batch["condition_mask"] = condition_mask

        if self.model_cfg.normalize_cam_angvel:
            batch["f_cam_angvel"] = (
                batch["f_cam_angvel"] - self.cam_angvel_mean
            ) / self.cam_angvel_std
        for k in self.normalizer_stats:
            if k in batch:
                batch[k] = self.normalize_attr(batch[k], k)

        if "multi_text_data" in batch["meta"][0]:
            if "text_embed" not in batch["meta"][0]["multi_text_data"]:
                multi_text_data = batch["meta"][0]["multi_text_data"]
                num_text = len(multi_text_data["caption"])
                text_embed = self.encode_text(
                    multi_text_data["caption"], torch.tensor([True] * num_text)
                )
                batch["meta"][0]["multi_text_data"]["text_embed"] = text_embed

        batch = self.create_condition_mask(
            batch, cond_mask_cfg=None, mode=None, train=False
        )

        if self.pipeline.args.infer_version == 3:
            postproc = False
        else:
            postproc = postproc
        print(f"Preproc taken: {time.time() - now}")
        now = time.time()
        outputs = self.pipeline.forward(
            batch,
            train=False,
            postproc=postproc,
            static_cam=static_cam,
            test_mode=test_mode,
        )

        pred = {
            "smpl_params_global": {
                k: v[0] for k, v in outputs["pred_smpl_params_global"].items()
            },
            "smpl_params_incam": {
                k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()
            },
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,  # intermediate outputs
        }
        print(f"Demo taken: {time.time() - now}")
        return pred

    def configure_optimizers(self):
        params = []
        for k, v in self.named_parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = self.optimizer(params=params)

        if self.scheduler is None or self.scheduler["scheduler"] is None:
            return optimizer

        scheduler = dict(self.scheduler)
        scheduler["scheduler"] = instantiate(
            scheduler["scheduler"], optimizer=optimizer
        )
        return [optimizer], [scheduler]

    # ============== Utils ================= #
    def on_save_checkpoint(self, checkpoint) -> None:
        for ig_keys in self.ignored_weights_prefix:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    # Log.info(f"Remove key `{ig_keys}' from checkpoint.")
                    checkpoint["state_dict"].pop(k)

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer] Loading ckpt: {ckpt_path}")

        ckpt = torch.load(ckpt_path, "cpu")
        state_dict = ckpt["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            ignored_when_saving = any(
                k.startswith(ig_keys) for ig_keys in self.ignored_weights_prefix
            )
            if not ignored_when_saving:
                real_missing.append(k)

        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")
        return ckpt
