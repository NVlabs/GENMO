import atexit

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer

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
        use_motion_mask=False,
        obs_motion_constraints="hard",
        use_obs_diff=False,
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

        input_dim = (
            self.input_feats * 2 if use_motion_mask else self.input_feats
        ) + self.gru_emb_dim
        if use_obs_diff:
            input_dim += 67
        self.input_process = InputProcess(self.data_rep, input_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        self.add_x_to_memory = add_x_to_memory
        self.autoregressive = autoregressive
        self.last_text = None
        self.last_encoded_text = None
        self.obs_motion_constraints = obs_motion_constraints
        self.use_obs_diff = use_obs_diff

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

        self.output_process = OutputProcess(
            self.data_rep,
            self.input_feats + 3,
            self.latent_dim,
            self.njoints + 3,
            self.nfeats,
        )

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

        self.last_text = raw_text
        self.last_encoded_text = encoded_text
        return encoded_text

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
        **kwargs,
    ):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        assert (y is not None) == (self.cond_mode != "no_cond"), (
            "must specify y if and only if the model is class-conditional"
        )
        x = x.transpose(1, 2).unsqueeze(2)
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        emb_text = y["encoded_text"]

        # if 'text' in self.cond_mode:
        #     if 'enc_text' not in y:
        #         enc_text = self.encode_text(y['text'])  # enc_text: [B, max_text_len, llm_dim], emb: [1, B, d]
        #     else:
        #         enc_text = y['enc_text']
        #     force_mask = y.get('uncond', False)
        #     emb_text = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask, rm_text_flag=rm_text_flag))
        #     if self.text_enc_mode == 'mean':
        #         emb_text = emb_text.mean(dim=1)
        #         emb += emb_text

        x = self.input_process(x)

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

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        output = output.squeeze(2).transpose(1, 2)  # [bs, nfeats, njoints, nframes]

        pred_x_start = output[..., :-3]
        static_conf_logits = pred_x_start[..., :6]
        pred_x = pred_x_start[..., 6:]
        pred_cam = output[..., -3:]

        return {
            "pred_x_start": pred_x_start,
            "pred_x": pred_x,
            "static_conf_logits": static_conf_logits,
            "pred_cam": pred_cam,
        }
