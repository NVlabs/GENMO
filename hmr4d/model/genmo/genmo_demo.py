import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from timm.models.vision_transformer import Mlp
from transformers import T5EncoderModel, T5Tokenizer

from hmr4d.configs import MainStore, builds
from hmr4d.model.gvhmr.utils import stats_compose
from hmr4d.network.base_arch.transformer.layer import BasicBlock, zero_module
from hmr4d.utils.geo.hmr_cam import (
    compute_bbox_info_bedlam,
    normalize_kp2d,
)
from hmr4d.utils.net_utils import length_to_mask
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx


class GENMO_demo(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        optimizer=None,
        scheduler_cfg=None,
        model_cfg=None,
        ignored_weights_prefix=["smplx", "pipeline.endecoder"],
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.endecoder = self.pipeline.endecoder
        self.optimizer = instantiate(optimizer)
        self.model_cfg = model_cfg
        self.scheduler_cfg = scheduler_cfg

        # Options
        self.ignored_weights_prefix = ignored_weights_prefix

        # The test step is the same as validation
        self.test_step = self.predict_step = self.validation_step
        self.timing = os.environ.get("DEBUG_TIMING", "FALSE") == "TRUE"

        # SMPLX
        self.smplxcoco17 = make_smplx("supermotion_v437coco17")
        self.smplxsmpl24 = make_smplx("supermotion_smpl24")

        if "text_encoder" in model_cfg:
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
            "humanoid_obs": (self.pipeline.args.get("humanoid_obs_dim", 358),),
            "humanoid_rgb_obs": self.pipeline.args.get(
                "humanoid_rgb_obs_dim", (4, 100, 100)
            ),
            "humanoid_contact_force": (90,),
        }

        self.not_add_features = [
            "obs",
            "f_cliffcam",
            "f_cam_angvel",
            "f_cam_t_vel",
            "f_imgseq",
            "observed_motion_3d",
            "humanoid_obs",
            "humanoid_rgb_obs",
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

        if "humanoid_obs" in self.pipeline.args.in_attr:
            self.humanoid_obs_embedder = nn.Sequential(
                nn.Linear(self.humanoid_obs_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
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

        self.humanoid = None

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

    def load_and_freeze_llm(self, llm_version):
        tokenizer = T5Tokenizer.from_pretrained(llm_version)
        model = T5EncoderModel.from_pretrained(llm_version)
        # Freeze llm weights
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model, tokenizer

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
            import ipdb

            ipdb.set_trace()
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
        skip_keys = self.pipeline.args.get(
            "skip_keys_for_null_condition", ["humanoid_obs"]
        )
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
            elif k == "humanoid_obs":
                f_humanoid_obs = batch["humanoid_obs"][
                    :, :end_fr
                ]  # (B, L, humanoid_obs_dim)
                f_humanoid_obs = self.humanoid_obs_embedder(f_humanoid_obs)
                mask = f_condition_exists[k]
                f_cond_dict["humanoid_obs"] = f_humanoid_obs * mask.float()
                f_uncond_dict["humanoid_obs"] = f_humanoid_obs * mask.float()
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

    @torch.no_grad()
    def predict(self, data, static_cam=False):
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

        batch["device"] = batch["f_imgseq"].device

        if "meta" in data:
            batch["meta"] = data["meta"]
        else:
            batch["meta"] = None
        if "vimo_smpl_params" in data:
            batch["vimo_smpl_params"] = {
                k: v[None].cuda() for k, v in data["vimo_smpl_params"].items()
            }
            batch["scales"] = data["scales"][None].cuda()
            batch["mean_scale"] = torch.tensor(data["mean_scale"])[None].cuda()

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
            postproc = True
        outputs = self.pipeline.forward(
            batch,
            train=False,
            postproc=postproc,
            static_cam=static_cam,
            test_mode=test_mode,
        )

        _, pred_coco17_joints_global = self.smplxcoco17(**outputs["pred_smpl_params_global"])
        _, pred_coco17_joints_incam = self.smplxcoco17(**outputs["pred_smpl_params_incam"])
        pred_smpl24_joints_global = self.smplxsmpl24(**outputs["pred_smpl_params_global"])
        pred_smpl24_joints_incam, _ = self.smplxsmpl24(**outputs["pred_smpl_params_incam"])
        pred = {
            "smpl_params_global": {
                k: v[0] for k, v in outputs["pred_smpl_params_global"].items()
            },
            "smpl_params_incam": {
                k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()
            },
            "K_fullimg": data["K_fullimg"],
            "coco17_joints_global": pred_coco17_joints_global[0],
            "coco17_joints_incam": pred_coco17_joints_incam[0],
            "smpl24_joints_global": pred_smpl24_joints_global[0],
            "smpl24_joints_incam": pred_smpl24_joints_incam[0],
            "net_outputs": outputs,  # intermediate outputs
        }
        return pred

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

    def configure_optimizers(self):
        params = []
        for k, v in self.named_parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = self.optimizer(params=params)

        if self.scheduler_cfg is None or self.scheduler_cfg["scheduler"] is None:
            return optimizer

        scheduler_cfg = dict(self.scheduler_cfg)
        scheduler_cfg["scheduler"] = instantiate(
            scheduler_cfg["scheduler"], optimizer=optimizer
        )
        return [optimizer], [scheduler_cfg]

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


genmo = builds(
    GENMO_demo,
    pipeline="${pipeline}",
    model_cfg="${model_cfg}",
)
MainStore.store(name="genmo_demo", node=genmo, group="model/genmo")
