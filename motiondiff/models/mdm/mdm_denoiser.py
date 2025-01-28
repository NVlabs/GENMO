import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        legacy=False,
        data_rep="rot",
        dataset="amass",
        clip_dim=512,
        arch="trans_enc",
        emb_trans_dec=False,
        clip_version=None,
        pretrained_checkpoint=None,
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

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get("action_emb", None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get("normalize_encoder_output", False)

        self.cond_mode = kargs.get("cond_mode", "no_cond")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == "gru" else 0
        self.input_process = InputProcess(
            self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim
        )

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

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
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print("EMBED TEXT")
                print("Loading CLIP...")
                self.clip_version = clip_version
                self.ext_models["clip"] = self.load_and_freeze_clip(clip_version)
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
            self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats
        )

        self.rot2xyz = Rotation2xyz(dataset=self.dataset)
        if pretrained_checkpoint is not None:
            state_dict = torch.load(pretrained_checkpoint, map_location="cpu")
            self.load_model_wo_clip(state_dict)

    def load_model_wo_clip(self, state_dict):
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith("clip_model.") for k in missing_keys])

    def parameters_wo_clip(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(
            clip_version, device="cpu", jit=False
        )  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model
        )  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        with torch.cuda.amp.autocast(enabled=False):
            device = next(self.parameters()).device
            move_module_dict_to_device(self.ext_models, device)
            max_text_len = (
                20 if self.dataset in ["humanml", "kit"] else None
            )  # Specific hardcoding for humanml dataset
            if max_text_len is not None:
                default_context_length = 77
                context_length = max_text_len + 2  # start_token + 20 + end_token
                assert context_length < default_context_length
                texts = clip.tokenize(
                    raw_text, context_length=context_length, truncate=True
                ).to(
                    device
                )  # [bs, context_length] # if n_tokens > context_length -> will truncate
                # print('texts', texts.shape)
                zero_pad = torch.zeros(
                    [texts.shape[0], default_context_length - context_length],
                    dtype=texts.dtype,
                    device=texts.device,
                )
                texts = torch.cat([texts, zero_pad], dim=1)
                # print('texts after pad', texts.shape, texts)
            else:
                texts = clip.tokenize(raw_text, truncate=True).to(
                    device
                )  # [bs, context_length] # if n_tokens > 77 -> will truncate
            return self.ext_models["clip"].encode_text(texts).float()

    def forward(self, x, timesteps, y=None, **kwargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        assert (y is not None) == (self.cond_mode != "no_cond"), (
            "must specify y if and only if the model is class-conditional"
        )
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        if "text" in self.cond_mode:
            enc_text = self.encode_text(y["text"])
            force_mask = y.get("uncond", False)
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if "action" in self.cond_mode:
            if not (
                y["action"] == -1
            ).any():  # FIXME - a hack so we can use already trained models
                action_emb = self.embed_action(y["action"])
                emb += self.mask_cond(action_emb)

        if self.arch == "gru":
            x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)  # [#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)  # [bs, d, #frames]
            emb_gru = emb_gru.reshape(
                bs, self.latent_dim, 1, nframes
            )  # [bs, d, 1, #frames]
            x = torch.cat(
                (x_reshaped, emb_gru), axis=1
            )  # [bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        if self.arch == "trans_enc":
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[
                1:
            ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == "trans_dec":
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[
                    1:
                ]  # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == "gru":
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.rot2xyz.smpl_model.to(*args, **kwargs)

    def eval(self, *args, **kwargs):
        super().eval(*args, **kwargs)
        self.rot2xyz.smpl_model.eval(*args, **kwargs)

    def get_excluded_keys(self):
        return [k for k in self.state_dict().keys() if "clip_model" in k]
