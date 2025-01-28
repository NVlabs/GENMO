import atexit

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer

from motiondiff.models.common.transformer import generate_ar_mask
from motiondiff.utils.tools import are_arrays_equal
from motiondiff.utils.torch_utils import move_module_dict_to_device

from .modules import *
from .rotation2xyz import Rotation2xyz


class ARMDMDenoiser(nn.Module):
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
        self.input_process = InputProcess(
            self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim
        )

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
            self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats
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

    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
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

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        assert (y is not None) == (self.cond_mode != "no_cond"), (
            "must specify y if and only if the model is class-conditional"
        )
        bs, njoints, nfeats, nframes = x.shape
        motion_prefix = y["motion_prefix"]  # [bs, njoints, nfeats, motion_prefix_len]
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        if "text" in self.cond_mode:
            enc_text = self.encode_text(
                y["text"]
            )  # enc_text: [B, max_text_len, llm_dim], emb: [1, B, d]
            force_mask = y.get("uncond", False)
            emb_text = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
            if self.text_enc_mode == "mean":
                emb_text = emb_text.mean(dim=1)
                emb += emb_text
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
        motion_prefix = self.input_process(motion_prefix)

        if self.arch == "trans_enc":
            # adding the timestep embed
            pose_start_ind = 1
            xseq = [emb]
            # add text
            if self.text_enc_mode == "seq_concat":
                xseq += [emb_text.transpose(0, 1)]
                pose_start_ind += emb_text.shape[1]
            # add motion prefix
            xseq += [motion_prefix]
            pose_start_ind += motion_prefix.shape[0]
            # finally add motion
            xseq += [x]
            xseq = torch.cat(xseq, axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            src_mask = None
            if self.autoregressive:
                src_mask = generate_ar_mask(
                    xseq.shape[0],
                    xseq.shape[0],
                    tgt_start_dim=pose_start_ind,
                    src_start_dim=pose_start_ind,
                ).to(x.device)
            output = self.seqTransEncoder(xseq, mask=src_mask)[
                pose_start_ind:
            ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

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
        return ["llm_model"]
