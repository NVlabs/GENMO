import atexit

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer

from motiondiff.data_pipeline.humanml.common.quaternion import *
from motiondiff.models.common.transformer import generate_ar_mask
from motiondiff.utils.tools import are_arrays_equal
from motiondiff.utils.torch_utils import move_module_dict_to_device

from .modules import *
from .rotation2xyz import Rotation2xyz


class MDMDenoiser(nn.Module):
    def __init__(
        self,
        pl_module,
        modeltype,
        njoints,
        nfeats,
        num_actions,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        ablation=None,
        activation="gelu",
        padding_mask=False,
        legacy=False,
        data_rep="rot",
        dataset="amass",
        llm_dim=512,
        max_text_len=20,
        arch="trans_enc",
        emb_trans_dec=False,
        add_x_to_memory=False,
        autoregressive=False,
        llm_type="t5",
        llm_version="t5-small",
        text_enc_mode="mean",
        pretrained_checkpoint=None,
        use_precomp_text_embed=False,
        llm_embed_lmdb_path=None,
        use_motion_mask=False,
        obs_motion_constraints="hard",
        use_obs_diff=False,
        use_unknownt_motion_mask=False,
        use_unknownt_obs_diff=False,
        use_global_constraints=False,
        global_feat_type=["g_motion_l_joints", "l_joints_diff", "global_joint_mask"],
        normalize_global_pos=False,
        pred_root_with_joints=False,
        use_foot_skate_feats=False,
        **kargs,
    ):
        super().__init__()

        self.ext_models = dict()

        self.ext_models["pl_module"] = pl_module
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.padding_mask = padding_mask

        self.ablation = ablation
        self.activation = activation
        self.llm_dim = llm_dim
        self.max_text_len = max_text_len
        self.text_enc_mode = text_enc_mode
        self.action_emb = kargs.get("action_emb", None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get("normalize_encoder_output", False)

        self.cond_mode = kargs.get("cond_mode", "no_cond")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == "gru" else 0
        self.use_motion_mask = use_motion_mask
        self.use_global_constraints = use_global_constraints
        self.global_feat_type = global_feat_type
        self.use_foot_skate_feats = use_foot_skate_feats

        self.use_unknownt_motion_mask = use_unknownt_motion_mask
        self.use_unknownt_obs_diff = use_unknownt_obs_diff

        self.motion_root_dim = self.ext_models["pl_module"].motion_root_dim
        self.motion_localjoints_dim = self.ext_models[
            "pl_module"
        ].motion_localjoints_dim
        self.motion_root_joints_dim = (
            self.motion_root_dim + self.motion_localjoints_dim
        )  # 67 : previously, may need to change to run old models
        input_dim = (
            (self.input_feats * 2 if use_motion_mask else self.input_feats)
            + self.gru_emb_dim
            - (self.motion_root_dim - 4)
        )
        if use_obs_diff:
            input_dim += self.motion_root_joints_dim
        if use_global_constraints:
            self.global_feat_dims = 66 * len(global_feat_type)
            input_dim += self.global_feat_dims
        if use_foot_skate_feats:
            input_dim += 12
        if self.use_unknownt_motion_mask:
            input_dim += self.input_feats * 2
        if self.use_unknownt_obs_diff:
            input_dim += self.motion_root_joints_dim
        self.input_process = InputProcess(self.data_rep, input_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        self.add_x_to_memory = add_x_to_memory
        self.autoregressive = autoregressive
        self.last_text = None
        self.last_encoded_text = None
        self.use_precomp_text_embed = use_precomp_text_embed
        self.llm_embed_lmdb_path = llm_embed_lmdb_path
        if self.use_precomp_text_embed:
            self.env = lmdb.open(self.llm_embed_lmdb_path, readonly=True)
            self.txn = self.env.begin()
            atexit.register(self.cleanup)
        self.obs_motion_constraints = obs_motion_constraints
        self.use_obs_diff = use_obs_diff
        self.normalize_global_pos = self.ext_models["pl_module"].normalize_global_pos
        self.pred_root_with_joints = pred_root_with_joints

        self.humanml_mean = pl_module.humanml_mean
        self.humanml_std = pl_module.humanml_std
        self.humanml_global_mean = pl_module.humanml_global_mean
        self.humanml_global_std = pl_module.humanml_global_std

        # root trajectory models
        self.root_input_feats = (
            (self.input_feats * 2 if use_motion_mask else self.input_feats)
            if self.pred_root_with_joints
            else self.motion_root_dim
        )
        if use_obs_diff:
            self.root_input_feats += self.motion_root_joints_dim
        if self.use_foot_skate_feats:
            self.root_input_feats += 12
        if self.use_unknownt_motion_mask:
            self.root_input_feats += self.input_feats * 2
        if self.use_unknownt_obs_diff:
            self.root_input_feats += self.motion_root_joints_dim
        self.root_output_feats = self.motion_root_dim
        self.root_input_process = InputProcess(
            "root", self.root_input_feats, self.latent_dim
        )
        root_seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )

        self.root_seqTransEncoder = nn.TransformerEncoder(
            root_seqTransEncoderLayer, num_layers=self.num_layers
        )
        self.root_output_process = OutputProcess(
            self.data_rep,
            self.root_output_feats,
            self.latent_dim,
            self.root_output_feats,
            1,
        )

        if self.arch == "trans_enc":
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
            )

            self.seqTransEncoder = nn.TransformerEncoder(
                seqTransEncoderLayer, num_layers=self.num_layers
            )
        elif self.arch == "trans_dec":
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=activation,
            )
            self.seqTransDecoder = nn.TransformerDecoder(
                seqTransDecoderLayer, num_layers=self.num_layers
            )
        elif self.arch == "gru":
            print("GRU init")
            self.gru = nn.GRU(
                self.latent_dim,
                self.latent_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
        else:
            raise ValueError(
                "Please choose correct architecture [trans_enc, trans_dec, gru]"
            )

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        if self.cond_mode != "no_cond":
            if "text" in self.cond_mode:
                self.embed_text = nn.Linear(self.llm_dim, self.latent_dim)
                print("EMBED TEXT")
                print(f"Loading {llm_type}-{llm_version}...")
                self.llm_type = llm_type
                self.llm_version = llm_version
                self.ext_models["llm"], self.tokenizer = self.load_and_freeze_llm(
                    llm_version
                )
                # text = ["translate English to German: The house is wonderful."]
                # encoded_text = self.encode_text(text)
            if "action" in self.cond_mode:
                if self.action_emb == "scalar":
                    self.embed_action = EmbedActionScalar(
                        in_features=1,
                        out_features=self.latent_dim,
                        activation=self.activation,
                    )
                elif self.action_emb == "tensor":
                    self.embed_action = EmbedActionTensor(
                        self.num_actions, self.latent_dim
                    )
                else:
                    raise Exception(f"Unknown action embedding {self.action_emb}.")
                print("EMBED ACTION")

        self.output_process = OutputProcess(
            self.data_rep,
            self.input_feats - self.root_output_feats,
            self.latent_dim,
            self.njoints - self.root_output_feats,
            self.nfeats,
        )

        self.rot2xyz = Rotation2xyz(dataset=self.dataset)
        if pretrained_checkpoint is not None:
            state_dict = torch.load(pretrained_checkpoint, map_location="cpu")
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]
            self.load_model_wo_llm(state_dict)

    def cleanup(self):
        if self.use_precomp_text_embed:
            self.env.close()

    def load_model_wo_llm(self, state_dict):
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith("llm_model.") for k in missing_keys])

    def parameters_wo_llm(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("llm_model.")
        ]

    def load_and_freeze_llm(self, llm_version):
        tokenizer = T5Tokenizer.from_pretrained(llm_version)
        model = T5EncoderModel.from_pretrained(llm_version)
        # Freeze llm weights
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model, tokenizer

    def mask_cond(self, cond, force_mask=False, rm_text_flag=None):
        bs, t, d = cond.shape
        if rm_text_flag is not None:
            return cond * (1.0 - rm_text_flag.view((bs,) + (1,) * len(cond.shape[1:])))
        elif force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                (bs,) + (1,) * len(cond.shape[1:])
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        if are_arrays_equal(raw_text, self.last_text):
            return self.last_encoded_text

        try:
            if self.use_precomp_text_embed:
                encoded_text = []
                for text in raw_text:
                    data = self.txn.get(text.encode())
                    if data is None:
                        raise Exception(f"No data for text {text}")
                    embed = np.frombuffer(data, dtype=np.float32).reshape(
                        -1, self.llm_dim
                    )
                    encoded_text.append(embed)
                encoded_text = torch.tensor(encoded_text, device=device)
                return encoded_text
        except:
            pass

        with torch.cuda.amp.autocast(enabled=False):
            move_module_dict_to_device(self.ext_models, device)
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
                output = self.ext_models["llm"](
                    input_ids=input_ids, attention_mask=attn_mask
                )
                encoded_text = output.last_hidden_state.detach()

            encoded_text = encoded_text[:, :max_text_len]
            attn_mask = attn_mask[:, :max_text_len]
            encoded_text *= attn_mask.unsqueeze(-1)
            # for bnum in range(encoded_text.shape[0]):
            #     nvalid_elem = attn_mask[bnum].sum().item()
            #     encoded_text[bnum][nvalid_elem:] = 0

        self.last_text = raw_text
        self.last_encoded_text = encoded_text
        return encoded_text

    def convert_root_global_to_local(self, root_motion):
        root_motion = root_motion.permute(0, 2, 3, 1)
        if self.normalize_global_pos:
            root_motion = root_motion * self.humanml_global_std.to(
                root_motion.device
            ) + self.humanml_global_mean.to(root_motion.device)
        if self.ext_models["pl_module"].motion_rep == "global_root_vel_local_joints":
            pos_v_xy, pos_y, ang_v = (
                root_motion[..., :2],
                root_motion[..., [2]],
                root_motion[..., [3]],
            )
            r_rot_ang = torch.zeros_like(ang_v)
            r_rot_ang[:, :, 1:] = ang_v[:, :, :-1]
            r_rot_ang = r_rot_ang.cumsum(dim=2).to(ang_v.dtype)
            r_rot_quat = torch.cat(
                [
                    torch.cos(r_rot_ang),
                    torch.zeros_like(r_rot_ang),
                    torch.sin(r_rot_ang),
                    torch.zeros_like(r_rot_ang),
                ],
                dim=-1,
            )
            r_pos = torch.zeros(
                pos_v_xy.shape[:-1] + (3,),
                dtype=pos_v_xy.dtype,
                device=root_motion.device,
            )
            r_pos[..., 1:, [0, 2]] = pos_v_xy[..., :-1, :]
            r_pos = torch.cumsum(r_pos, dim=2).to(pos_v_xy.dtype)
            r_pos[..., [1]] = pos_y
        else:
            r_pos, rot_cos, rot_sin = (
                root_motion[..., :3],
                root_motion[..., 3],
                root_motion[..., 4],
            )
            r_rot_ang = torch.atan2(rot_sin, rot_cos).unsqueeze(-1)
            r_rot_quat = torch.stack(
                [
                    rot_cos,
                    torch.zeros_like(rot_cos),
                    rot_sin,
                    torch.zeros_like(rot_cos),
                ],
                dim=-1,
            )
        r_pos_y = r_pos[..., [1]].clone()
        r_pos = r_pos[:, :, 1:] - r_pos[:, :, :-1]
        r_pos = torch.cat([r_pos, r_pos[:, :, [-1]]], dim=2)
        r_pos = qrot(r_rot_quat, r_pos)
        ang_v = r_rot_ang[:, :, 1:] - r_rot_ang[:, :, :-1]
        ang_v[ang_v > np.pi] -= 2 * np.pi
        ang_v[ang_v < -np.pi] += 2 * np.pi
        ang_v = torch.cat([ang_v, ang_v[:, :, [-1]]], dim=2)
        local_motion = torch.cat([ang_v, r_pos[..., [0, 2]], r_pos_y], dim=-1)
        local_motion_norm = (
            local_motion - self.humanml_mean[:4].to(local_motion.device)
        ) / self.humanml_std[:4].to(local_motion.device)
        local_motion_norm = local_motion_norm.permute(0, 3, 1, 2)
        return local_motion_norm

    def forward_root(self, x, emb, emb_text, y, x_gt, using_gt_joints_cond):
        xr = x[:, : self.motion_root_dim]
        if self.pred_root_with_joints:
            if using_gt_joints_cond is None:
                xr = x
            else:
                xr_gt = torch.cat([xr, x_gt[:, self.motion_root_dim :]], axis=1)
                xr = torch.where(using_gt_joints_cond[:, None, None, None], xr_gt, x)
        xr = self.root_input_process(xr)

        if self.arch == "trans_enc":
            # adding the timestep embed
            pose_start_ind = 1
            xseq = torch.cat((emb, xr), axis=0)  # [seqlen+1, bs, d]
            if self.text_enc_mode == "seq_concat":
                xseq = torch.cat((emb_text.transpose(0, 1), xseq), axis=0)
                pose_start_ind += emb_text.shape[1]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            src_mask = None
            src_key_padding_mask = None
            if self.padding_mask:
                src_key_padding_mask = torch.zeros(
                    (xseq.shape[1], xseq.shape[0]), dtype=torch.bool, device=xseq.device
                )
                src_key_padding_mask[:, pose_start_ind:] = ~y["mask"][:, 0, 0]
            if self.autoregressive:
                src_mask = generate_ar_mask(
                    xseq.shape[0],
                    xseq.shape[0],
                    tgt_start_dim=pose_start_ind,
                    src_start_dim=pose_start_ind,
                ).to(xr.device)
            output = self.root_seqTransEncoder(
                xseq, mask=src_mask, src_key_padding_mask=src_key_padding_mask
            )[pose_start_ind:]

        xr = self.root_output_process(output)
        return xr

    def get_foot_skate_feats(self, x):
        fid = [7, 10, 8, 11]
        x_global = self.ext_models["pl_module"].get_global_position(x)
        foot_m = x_global[:, :66, 0].transpose(1, 2)
        foot_m = foot_m.view(foot_m.shape[:2] + (-1, 3))
        foot_m = foot_m[:, :, fid]
        foot_vel = foot_m[:, 1:] - foot_m[:, :-1]
        foot_vel = torch.cat([foot_vel, foot_vel[:, [-1]]], dim=1)
        foot_vel = (
            foot_vel.view(foot_vel.shape[:-2] + (-1,)).transpose(1, 2).unsqueeze(2)
        )
        return foot_vel

    def forward(
        self,
        x,
        timesteps,
        y=None,
        motion_mask=None,
        observed_motion=None,
        rm_text_flag=None,
        global_motion=None,
        global_joint_mask=None,
        global_joint_func=None,
        unknownt_motion_mask=None,
        unknownt_observed_motion=None,
        gt_motion=None,
        using_gt_root_cond=None,
        using_gt_joints_cond=None,
        fixed_root_input=None,
    ):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        assert (y is not None) == (self.cond_mode != "no_cond"), (
            "must specify y if and only if the model is class-conditional"
        )
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        x_orig = x

        if global_joint_func is not None:
            g_motion_l_joints, l_joints_diff = global_joint_func(
                x, global_motion, global_joint_mask
            )

        if "text" in self.cond_mode:
            enc_text = self.encode_text(
                y["text"]
            )  # enc_text: [B, max_text_len, llm_dim], emb: [1, B, d]
            force_mask = y.get("uncond", False)
            emb_text = self.embed_text(
                self.mask_cond(
                    enc_text, force_mask=force_mask, rm_text_flag=rm_text_flag
                )
            )
            if self.text_enc_mode == "mean":
                emb_text = emb_text.mean(dim=1)
                emb += emb_text
        if "action" in self.cond_mode:
            if not (
                y["action"] == -1
            ).any():  # FIXME - a hack so we can use already trained models
                action_emb = self.embed_action(y["action"])
                emb += self.mask_cond(action_emb)

        if self.use_motion_mask:
            if motion_mask is not None:
                assert observed_motion is not None
                x = x * (1 - motion_mask) + observed_motion * motion_mask
                if self.use_obs_diff:
                    motion_diff = ((observed_motion - x_orig) * motion_mask)[
                        :, : self.motion_root_joints_dim
                    ]
                    x = torch.cat([x, motion_mask, motion_diff], axis=1)
                else:
                    x = torch.cat([x, motion_mask], axis=1)
            else:
                if self.use_obs_diff:
                    x = torch.cat(
                        [
                            x,
                            torch.zeros_like(x),
                            torch.zeros_like(x[:, : self.motion_root_joints_dim]),
                        ],
                        axis=1,
                    )
                else:
                    x = torch.cat([x, torch.zeros_like(x)], axis=1)
        if self.use_unknownt_motion_mask:
            if unknownt_motion_mask is not None:
                len_mask = y["mask"]
                unknownt_motion_mask = unknownt_motion_mask.repeat(1, 1, 1, x.shape[-1])
                unknownt_motion_mask = unknownt_motion_mask * len_mask
                unknownt_observed_motion = unknownt_observed_motion.repeat(
                    1, 1, 1, x.shape[-1]
                )
                unknownt_observed_motion = (
                    unknownt_observed_motion * unknownt_motion_mask
                )
                if self.use_unknownt_obs_diff:
                    unknownt_motion_diff = (
                        (unknownt_observed_motion - x_orig) * unknownt_motion_mask
                    )[:, : self.motion_root_joints_dim]
                    x = torch.cat(
                        [
                            x,
                            unknownt_observed_motion,
                            unknownt_motion_mask,
                            unknownt_motion_diff,
                        ],
                        axis=1,
                    )
                else:
                    x = torch.cat(
                        [x, unknownt_observed_motion, unknownt_motion_mask], axis=1
                    )
            else:
                if self.use_unknownt_obs_diff:
                    x = torch.cat(
                        [
                            x,
                            torch.zeros_like(x_orig),
                            torch.zeros_like(x_orig),
                            torch.zeros_like(x_orig[:, : self.motion_root_joints_dim]),
                        ],
                        axis=1,
                    )
                else:
                    x = torch.cat(
                        [x, torch.zeros_like(x_orig), torch.zeros_like(x_orig)], axis=1
                    )
        if self.use_global_constraints:
            if global_motion is not None:
                local_vars = locals()
                global_feat = torch.cat(
                    [local_vars[feat] for feat in self.global_feat_type], dim=-1
                )
                x = torch.cat([x, global_feat.transpose(1, 2).unsqueeze(2)], axis=1)
            else:
                x = torch.cat(
                    [
                        x,
                        torch.zeros(
                            (x.shape[0], self.global_feat_dims, *x.shape[2:]),
                            device=x.device,
                            dtype=x.dtype,
                        ),
                    ],
                    axis=1,
                )

        if self.use_foot_skate_feats:
            foot_skate_feats = self.get_foot_skate_feats(x)
            x = torch.cat([x, foot_skate_feats], axis=1)

        if gt_motion is not None:
            x_gt = torch.cat([gt_motion.clone(), x[:, gt_motion.shape[1] :]], axis=1)
        else:
            x_gt = (
                x.clone()
            )  # use current x as gt, for example, keep the root motion fixed as gt

        # predict root motion first
        if fixed_root_input is not None:
            root_motion_pred = fixed_root_input.clone()
        else:
            root_motion_pred = self.forward_root(
                x, emb, emb_text, y, x_gt, using_gt_joints_cond
            )

        with torch.no_grad():
            if using_gt_root_cond is None:
                root_motion_for_joints = root_motion_pred
            else:
                root_motion_for_joints = torch.where(
                    using_gt_root_cond[:, None, None, None],
                    x_gt[:, : self.motion_root_dim],
                    root_motion_pred,
                )
            root_motion_local = self.convert_root_global_to_local(
                root_motion_for_joints
            )

        if self.use_foot_skate_feats:
            x_after_root_pred = torch.cat(
                [root_motion_for_joints, x_orig[:, self.motion_root_dim :]], axis=1
            )
            foot_skate_feats = self.get_foot_skate_feats(x_after_root_pred)
            x_new = torch.cat(
                [
                    root_motion_local.detach(),
                    x[:, self.motion_root_dim : -foot_skate_feats.shape[1]],
                    foot_skate_feats,
                ],
                axis=1,
            )
        else:
            x_new = torch.cat(
                [root_motion_local.detach(), x[:, self.motion_root_dim :]], axis=1
            )

        x = self.input_process(x_new)

        if self.arch == "trans_enc":
            # adding the timestep embed
            pose_start_ind = 1
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            if self.text_enc_mode == "seq_concat":
                xseq = torch.cat((emb_text.transpose(0, 1), xseq), axis=0)
                pose_start_ind += emb_text.shape[1]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            src_mask = None
            src_key_padding_mask = None
            if self.padding_mask:
                src_key_padding_mask = torch.zeros(
                    (xseq.shape[1], xseq.shape[0]), dtype=torch.bool, device=xseq.device
                )
                src_key_padding_mask[:, pose_start_ind:] = ~y["mask"][:, 0, 0]
            if self.autoregressive:
                src_mask = generate_ar_mask(
                    xseq.shape[0],
                    xseq.shape[0],
                    tgt_start_dim=pose_start_ind,
                    src_start_dim=pose_start_ind,
                ).to(x.device)
            output = self.seqTransEncoder(
                xseq, mask=src_mask, src_key_padding_mask=src_key_padding_mask
            )[pose_start_ind:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == "trans_dec":
            xseq = x
            pose_start_ind = 0
            memory = emb
            if self.text_enc_mode == "seq_concat":
                memory = [emb_text.transpose(0, 1), emb]
                if self.add_x_to_memory:
                    memory.append(x)
                memory = torch.cat(memory, axis=0)
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
                pose_start_ind += 1
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            memory = self.sequence_pos_encoder(memory)  # [seqlen+1, bs, d]
            tgt_mask = memory_mask = None
            if self.autoregressive:
                tgt_mask = generate_ar_mask(
                    xseq.shape[0],
                    xseq.shape[0],
                    tgt_start_dim=pose_start_ind,
                    src_start_dim=pose_start_ind,
                ).to(x.device)
                if self.add_x_to_memory:
                    memory_mask = generate_ar_mask(
                        xseq.shape[0],
                        memory.shape[0],
                        tgt_start_dim=pose_start_ind,
                        src_start_dim=emb_text.shape[1] + 1,
                    ).to(x.device)

            output = self.seqTransDecoder(
                tgt=xseq, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask
            )[pose_start_ind:]  # [seqlen, bs, d] # FIXME - maybe add a causal mask

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]

        output = torch.cat([root_motion_pred, output], axis=1)

        if self.use_motion_mask and motion_mask is not None:
            if self.obs_motion_constraints == "hard":
                output = output * (1 - motion_mask) + observed_motion * motion_mask
        return output

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.rot2xyz.smpl_model.to(*args, **kwargs)

    def eval(self, *args, **kwargs):
        super().eval(*args, **kwargs)
        self.rot2xyz.smpl_model.eval(*args, **kwargs)

    def get_excluded_keys(self):
        return ["llm_model"]
