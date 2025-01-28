import atexit

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer

from motiondiff.data_pipeline.humanml.common.quaternion import *
from motiondiff.models.common.llama import LLAMADecoder
from motiondiff.models.common.moe import MoETransformerEncoderLayer
from motiondiff.models.common.transformer import generate_ar_mask
from motiondiff.models.mdm.modules import *
from motiondiff.utils.tools import are_arrays_equal
from motiondiff.utils.torch_utils import move_module_dict_to_device


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
        data_rep="hml_vec",
        dataset="amass",
        llm_dim=512,
        max_text_len=50,
        arch="trans_enc",
        use_moe=False,
        moe_cfg=None,
        llama_cfg=None,
        emb_trans_dec=False,
        add_x_to_memory=False,
        llm_type="t5",
        llm_version="t5-small",
        text_enc_mode="seq_concat",
        pretrained_checkpoint=None,
        use_motion_mask=False,
        obs_motion_constraints="soft",
        pred_root_with_joints=True,
        sep_root_body=False,
        project_output_to_mask=False,
        project_root_to_mask=False,
        **kargs,
    ):
        super().__init__()

        self.ext_models = dict()

        self.ext_models["pl_module"] = pl_module
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

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get("normalize_encoder_output", False)

        self.cond_mode = kargs.get("cond_mode", "no_cond")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.arch = arch
        self.use_moe = use_moe
        self.moe_cfg = moe_cfg
        self.llama_cfg = llama_cfg
        self.gru_emb_dim = self.latent_dim if self.arch == "gru" else 0
        self.use_motion_mask = use_motion_mask
        self.project_output_to_mask = project_output_to_mask
        self.project_root_to_mask = project_root_to_mask
        if self.project_output_to_mask or self.project_root_to_mask:
            assert self.use_motion_mask, "Must be using motion mask to project"

        self.motion_root_dim = self.ext_models["pl_module"].motion_root_dim
        self.motion_localjoints_dim = self.ext_models[
            "pl_module"
        ].motion_localjoints_dim
        self.motion_root_joints_dim = self.motion_root_dim + self.motion_localjoints_dim
        input_dim = (
            (self.input_feats * 2 if use_motion_mask else self.input_feats)
            + self.gru_emb_dim
            - (self.motion_root_dim - 4)
        )
        self.input_process = InputProcess(self.data_rep, input_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        self.add_x_to_memory = add_x_to_memory

        self.last_text = None
        self.last_encoded_text = None
        self.obs_motion_constraints = obs_motion_constraints
        self.normalize_global_pos = self.ext_models["pl_module"].normalize_global_pos
        self.pred_root_with_joints = pred_root_with_joints

        self.motion_mean = pl_module.motion_mean
        self.motion_std = pl_module.motion_std
        self.motion_global_mean = pl_module.motion_global_mean
        self.motion_global_std = pl_module.motion_global_std

        # root trajectory models
        self.root_input_feats = (
            (self.input_feats * 2 if use_motion_mask else self.input_feats)
            if self.pred_root_with_joints
            else self.motion_root_dim
        )
        self.root_output_feats = self.motion_root_dim
        self.root_input_process = InputProcess(
            "root", self.root_input_feats, self.latent_dim
        )
        if self.arch == "trans_enc":
            if self.use_moe:
                root_seqTransEncoderLayer = MoETransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_size,
                    dropout=self.dropout,
                    activation=self.activation,
                    moe_cfg=self.moe_cfg,
                )
            else:
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
        elif self.arch == "trans_dec":
            if self.use_moe:
                root_seqTransDecoderLayer = MoETransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_size,
                    dropout=self.dropout,
                    activation=self.activation,
                    moe_cfg=self.moe_cfg,
                )
            else:
                root_seqTransDecoderLayer = nn.TransformerDecoderLayer(
                    d_model=self.latent_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_size,
                    dropout=self.dropout,
                    activation=self.activation,
                )
            self.root_seqTransDecoder = nn.TransformerDecoder(
                root_seqTransDecoderLayer, num_layers=self.num_layers
            )
        elif self.arch == "llama":
            self.root_seqTransEncoder = LLAMADecoder(llama_cfg)
        else:
            raise ValueError("Please choose correct architecture")
        self.root_output_process = OutputProcess(
            self.data_rep,
            self.root_output_feats,
            self.latent_dim,
            self.root_output_feats,
            1,
        )

        if self.arch == "trans_enc":
            print("TRANS_ENC init")
            if self.use_moe:
                seqTransEncoderLayer = MoETransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_size,
                    dropout=self.dropout,
                    activation=self.activation,
                    moe_cfg=self.moe_cfg,
                )
            else:
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
            if self.use_moe:
                seqTransDecoderLayer = MoETransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_size,
                    dropout=self.dropout,
                    activation=self.activation,
                    moe_cfg=self.moe_cfg,
                )
            else:
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
        elif self.arch == "llama":
            self.seqTransEncoder = LLAMADecoder(llama_cfg)
        else:
            raise ValueError("Please choose correct architecture [trans_enc]")

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )
        self.sep_root_body = sep_root_body
        if self.sep_root_body:
            print("Using separate encoders for root and body stages!")
            self.root_sequence_pos_encoder = PositionalEncoding(
                self.latent_dim, self.dropout
            )
            self.root_embed_timestep = TimestepEmbedder(
                self.latent_dim, self.root_sequence_pos_encoder
            )

        if self.cond_mode != "no_cond":
            if "text" in self.cond_mode:
                self.embed_text = nn.Linear(self.llm_dim, self.latent_dim)
                if self.sep_root_body:
                    self.root_embed_text = nn.Linear(self.llm_dim, self.latent_dim)
                print("EMBED TEXT")
                print(f"Loading {llm_type}-{llm_version}...")
                self.llm_type = llm_type
                self.llm_version = llm_version
                self.ext_models["llm"], self.tokenizer = self.load_and_freeze_llm(
                    llm_version
                )
                # text = ["translate English to German: The house is wonderful."]
                # encoded_text = self.encode_text(text)

        self.output_process = OutputProcess(
            self.data_rep,
            self.input_feats - self.root_output_feats,
            self.latent_dim,
            self.njoints - self.root_output_feats,
            self.nfeats,
        )

        if pretrained_checkpoint is not None:
            state_dict = torch.load(pretrained_checkpoint, map_location="cpu")
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]
            self.load_model_wo_llm(state_dict)

    def cleanup(self):
        return

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

        # handle empty string
        empty_text_mask = [len(text) == 0 for text in raw_text]
        encoded_text[empty_text_mask] = 0

        self.last_text = raw_text
        self.last_encoded_text = encoded_text
        return encoded_text

    def convert_root_global_to_local(self, root_motion):
        root_motion = root_motion.permute(0, 2, 3, 1)  # [batch, 1, num_steps, 5]
        if self.normalize_global_pos:
            root_motion = root_motion * self.motion_global_std.to(
                root_motion.device
            ) + self.motion_global_mean.to(root_motion.device)

        r_pos, rot_cos, rot_sin = (
            root_motion[..., :3],
            root_motion[..., 3],
            root_motion[..., 4],
        )
        r_rot_ang = torch.atan2(rot_sin, rot_cos)
        r_rot_quat_inv = torch.stack(
            [
                torch.cos(r_rot_ang / 2),
                torch.zeros_like(rot_cos),
                -torch.sin(r_rot_ang / 2),
                torch.zeros_like(rot_cos),
            ],
            dim=-1,
        )
        r_pos_y = r_pos[..., [1]].clone()
        r_pos = r_pos[:, :, 1:] - r_pos[:, :, :-1]
        r_pos = torch.cat([r_pos, r_pos[:, :, [-1]]], dim=2)
        r_pos = qrot(r_rot_quat_inv, r_pos)
        r_rot_ang = r_rot_ang.unsqueeze(-1)
        ang_v = r_rot_ang[:, :, 1:] - r_rot_ang[:, :, :-1]
        ang_v[ang_v > np.pi] -= 2 * np.pi
        ang_v[ang_v < -np.pi] += 2 * np.pi
        ang_v = torch.cat([ang_v, ang_v[:, :, [-1]]], dim=2)
        local_motion = torch.cat([ang_v, r_pos[..., [0, 2]], r_pos_y], dim=-1)
        local_motion_norm = (
            local_motion - self.motion_mean[:4].to(local_motion.device)
        ) / self.motion_std[:4].to(local_motion.device)
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

        if self.arch in {"trans_enc", "llama"}:
            # adding the timestep embed
            pose_start_ind = 1
            xseq = torch.cat((emb, xr), axis=0)  # [seqlen+1, bs, d]
            if self.text_enc_mode == "seq_concat":
                xseq = torch.cat(
                    (emb_text.transpose(0, 1), xseq), axis=0
                )  # [max_text_len + 1 + seqlen, bs, d]
                pose_start_ind += emb_text.shape[1]
            xseq = (
                self.root_sequence_pos_encoder(xseq)
                if self.sep_root_body
                else self.sequence_pos_encoder(xseq)
            )  # [seqlen+1, bs, d]
            src_mask = None
            src_key_padding_mask = None
            if self.padding_mask and "mask" in y:
                src_key_padding_mask = torch.zeros(
                    (xseq.shape[1], xseq.shape[0]), dtype=torch.bool, device=xseq.device
                )  # [bs, max_text_len + 1 + seqlen]
                src_key_padding_mask[:, pose_start_ind:] = ~y["mask"][:, 0, 0]
            output = self.root_seqTransEncoder(
                xseq, mask=src_mask, src_key_padding_mask=src_key_padding_mask
            )[pose_start_ind:]

        elif self.arch == "trans_dec":
            xseq = xr
            pose_start_ind = 0
            memory = emb
            if self.text_enc_mode == "seq_concat":
                memory = [emb_text.transpose(0, 1), emb]
                if self.add_x_to_memory:
                    memory.append(xr)
                memory = torch.cat(memory, axis=0)
            if self.emb_trans_dec:
                xseq = torch.cat((emb, xr), axis=0)
                pose_start_ind += 1
            xseq = (
                self.root_sequence_pos_encoder(xseq)
                if self.sep_root_body
                else self.sequence_pos_encoder(xseq)
            )  # [seqlen+1, bs, d]
            memory = (
                self.root_sequence_pos_encoder(memory)
                if self.sep_root_body
                else self.sequence_pos_encoder(memory)
            )  # [seqlen+1, bs, d]
            tgt_mask = memory_mask = None
            tgt_key_padding_mask = None
            if self.padding_mask and "mask" in y:
                tgt_key_padding_mask = torch.zeros(
                    (xseq.shape[1], xseq.shape[0]), dtype=torch.bool, device=xseq.device
                )
                tgt_key_padding_mask[:, pose_start_ind:] = ~y["mask"][:, 0, 0]

            output = self.root_seqTransDecoder(
                tgt=xseq,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )[pose_start_ind:]  # [seqlen, bs, d] # FIXME - maybe add a causal mask

        xr = self.root_output_process(output)
        return xr

    def project_motion(
        self,
        pred_out,
        interp_mask,
        observed_motion,
        num_interp_feats,
        pos_weight=1e-3,
        vel_weight=1.0,
        accel_weight=1.0,
    ):
        # pos anchors to original values
        # print(pred_out.shape) # [batch, num_feats, 1, num_frames]
        # print(interp_mask.shape) # [batch, num_feats, 1, num_frames]
        # print(observed_motion.shape) # [batch, num_feats, 1, num_frames]

        pred_out = pred_out.permute(0, 1, 3, 2)  # [batch, num_feats, num_frames, 1]
        interp_mask = interp_mask[:, :num_interp_feats].permute(
            0, 1, 3, 2
        )  # [batch, num_interp_feats, num_frames, 1]
        observed_motion = observed_motion[:, :num_interp_feats].permute(
            0, 1, 3, 2
        )  # [batch, num_interp_feats, num_frames, 1]

        # NOTE: only support full-body keyframes -- every interpolated channel should share the same mask and all channels should be constrained
        # print(torch.sum(interp_mask[:,:num_interp_feats], dim=(2,3)))
        assert (
            torch.sum(interp_mask[:, :num_interp_feats] - interp_mask[:, 0:1]) < 1e-6
        ), "Projection only supports full-body keyframes"

        do_interp = (torch.sum(interp_mask, dim=(1, 2, 3)) > 0).to(
            interp_mask
        )  # only do this if there are keyframes in the mask

        bsize, _, num_frames, _ = pred_out.shape
        interp_rhs_mat = torch.zeros(  # [batch_size, num_frames, num_frames]
            bsize,
            num_frames,
            num_frames,
        ).to(pred_out)
        interp_lhs_mat = torch.zeros_like(interp_rhs_mat)

        for bi in range(pred_out.shape[0]):
            cur_lhs, cur_rhs = interp_mats(
                interp_mask[bi, 0, :, 0], pos_weight, vel_weight, accel_weight
            )
            interp_rhs_mat[bi] = cur_rhs
            interp_lhs_mat[bi] = cur_lhs

        # do the interpolation
        xdiff = (
            pred_out[:, :num_interp_feats] - interp_mask * observed_motion
        )  # [batch, num_interp_feats, num_frames, 1]

        rhs = torch.matmul(interp_rhs_mat.unsqueeze(1), xdiff)
        rhs *= 1 - interp_mask
        rhs += interp_mask * observed_motion

        pred_out[:, :num_interp_feats] = (
            do_interp[:, None, None, None]
            * torch.matmul(interp_lhs_mat.unsqueeze(1), rhs)
            + (1 - do_interp[:, None, None, None]) * pred_out[:, :num_interp_feats]
        )
        return pred_out.permute(0, 1, 3, 2)

        # xdiff = (
        #     pred_out[:,:num_interp_feats].permute(0,1,3,2) -
        #     interp_mask[None,None,:,None] * observed_motion[:,:num_interp_feats].permute(0,1,3,2)
        # )

        # rhs = torch.matmul(interp_rhs_mat[None,None,:,:], xdiff)
        # rhs *= 1-interp_mask[None,None,:,None]
        # rhs += interp_mask[None,None,:,None] * observed_motion[:,:num_interp_feats].permute(0,1,3,2)

        # pred_out[:,:num_interp_feats] = (
        #     do_interp * torch.matmul(interp_lhs_mat[None,None,:,:], rhs).permute(0,1,3,2) +
        #     (1- do_interp) * pred_out[:,:num_interp_feats]
        # )

    def forward(
        self,
        x,
        timesteps,
        y=None,
        motion_mask=None,
        observed_motion=None,
        rm_text_flag=None,
        gt_motion=None,
        using_gt_root_cond=None,
        using_gt_joints_cond=None,
        fixed_root_input=None,
        **kwargs,
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
        root_emb = self.root_embed_timestep(timesteps) if self.sep_root_body else emb

        if "text" in self.cond_mode:
            if "enc_text" not in y:
                enc_text = self.encode_text(
                    y["text"]
                )  # enc_text: [B, max_text_len, llm_dim], emb: [1, B, d]
            else:
                enc_text = y["enc_text"]
            force_mask = y.get("uncond", False)
            text_cond = self.mask_cond(
                enc_text, force_mask=force_mask, rm_text_flag=rm_text_flag
            )
            emb_text = self.embed_text(text_cond)
            if self.text_enc_mode == "mean":
                emb_text = emb_text.mean(dim=1)
                emb += emb_text
            root_emb_text = (
                self.root_embed_text(text_cond) if self.sep_root_body else emb_text
            )

        if self.use_motion_mask:
            if motion_mask is not None:
                assert observed_motion is not None
                x = x * (1 - motion_mask) + observed_motion * motion_mask
                x = torch.cat([x, motion_mask], axis=1)
            else:
                x = torch.cat([x, torch.zeros_like(x)], axis=1)

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
                x, root_emb, root_emb_text, y, x_gt, using_gt_joints_cond
            )

        if self.project_root_to_mask and motion_mask is not None:
            root_motion_pred = self.project_motion(
                root_motion_pred, motion_mask, observed_motion, self.motion_root_dim
            )  # project only root

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

        x_new = torch.cat(
            [root_motion_local.detach(), x[:, self.motion_root_dim :]], axis=1
        )

        x = self.input_process(x_new)

        if self.arch in {"trans_enc", "llama"}:
            # adding the timestep embed
            pose_start_ind = 1
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            if self.text_enc_mode == "seq_concat":
                xseq = torch.cat((emb_text.transpose(0, 1), xseq), axis=0)
                pose_start_ind += emb_text.shape[1]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            src_mask = None
            src_key_padding_mask = None
            if self.padding_mask and "mask" in y:
                src_key_padding_mask = torch.zeros(
                    (xseq.shape[1], xseq.shape[0]), dtype=torch.bool, device=xseq.device
                )
                src_key_padding_mask[:, pose_start_ind:] = ~y["mask"][:, 0, 0]
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
            tgt_key_padding_mask = None
            if self.padding_mask and "mask" in y:
                tgt_key_padding_mask = torch.zeros(
                    (xseq.shape[1], xseq.shape[0]), dtype=torch.bool, device=xseq.device
                )
                tgt_key_padding_mask[:, pose_start_ind:] = ~y["mask"][:, 0, 0]

            output = self.seqTransDecoder(
                tgt=xseq,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )[pose_start_ind:]  # [seqlen, bs, d] # FIXME - maybe add a causal mask

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]

        output = torch.cat([root_motion_pred, output], axis=1)

        if self.project_output_to_mask and motion_mask is not None:
            output = self.project_motion(
                output, motion_mask, observed_motion, self.motion_root_joints_dim
            )  # project root + local joint pos

        if self.use_motion_mask and motion_mask is not None:
            if self.obs_motion_constraints == "hard":
                output = output * (1 - motion_mask) + observed_motion * motion_mask
        return output

    def get_excluded_keys(self):
        return ["llm_model"]


# precomputations for the interpolation stuff:
def interp_mats(interp_mask, pos_weight, vel_weight, acc_weight):
    # Matrix that calculates velocities:
    V = torch.zeros(
        interp_mask.shape[0],
        interp_mask.shape[0],
        dtype=interp_mask.dtype,
        device=interp_mask.device,
    )
    V[1:, 1:] += torch.diag_embed(torch.full_like(interp_mask[:-1], 1))
    V[1:, :-1] += torch.diag_embed(torch.full_like(interp_mask[:-1], -1))

    # Matrix that calculates accelerations:
    A = torch.zeros_like(V)
    A[1:-1, :-2] += torch.diag_embed(torch.full_like(interp_mask[:-2], -1))
    A[1:-1, 1:-1] += torch.diag_embed(torch.full_like(interp_mask[:-2], 2))
    A[1:-1, 2:] += torch.diag_embed(torch.full_like(interp_mask[:-2], -1))

    # Matrix for building the right hand side of the linear system:
    interp_rhs_mat = torch.zeros_like(V)
    interp_rhs_mat += pos_weight * torch.diag_embed(torch.full_like(interp_mask, 1))
    interp_rhs_mat += vel_weight * torch.matmul(V.T, V)
    interp_rhs_mat += acc_weight * torch.matmul(A.T, A)

    # Left hand side matrix for solving the linear system:
    interp_lhs_mat = (
        interp_rhs_mat * (1 - interp_mask[:, None]) * (1 - interp_mask[None, :])
    )
    interp_lhs_mat += torch.diag_embed(interp_mask)
    interp_lhs_mat = torch.linalg.inv(interp_lhs_mat)

    return interp_lhs_mat, interp_rhs_mat
