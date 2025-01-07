import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lmdb
import atexit
from transformers import T5Tokenizer, T5EncoderModel
from motiondiff.models.mdm.modules import *
from motiondiff.models.common.transformer import generate_ar_mask
from motiondiff.utils.torch_utils import move_module_dict_to_device
from motiondiff.utils.tools import are_arrays_equal
from hmr4d.network.base_arch.transformer.encoder_rope_s import EncoderRoPEBlock
from hmr4d.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock as EncoderRoPEBlock_old
from timm.models.vision_transformer import Mlp
from einops import einsum, rearrange, repeat
from hmr4d.network.base_arch.transformer.layer import zero_module


class MDMDenoiserROPE(nn.Module):
    def __init__(self, pl_module, modeltype, njoints, nfeats, num_frames, num_actions, translation, pose_rep, glob, glob_rot, output_dim,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", padding_mask=False, legacy=False, data_rep='rot', dataset='amass', llm_dim=512, max_text_len=20,
                 arch='trans_enc', emb_trans_dec=False, add_x_to_memory=False, autoregressive=False, llm_type="t5", llm_version="t5-small", text_enc_mode='mean', pretrained_checkpoint=None, 
                 use_precomp_text_embed=False, llm_embed_lmdb_path=None, use_motion_mask=False, obs_motion_constraints='hard', use_obs_diff=False, use_global_constraints=False, 
                 global_feat_type=['g_motion_l_joints', 'l_joints_diff', 'global_joint_mask'], args=None, **kargs):
        super().__init__()

        self.ext_models = dict()
        self.args = args

        self.ext_models['pl_module'] = pl_module
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset
        self.num_frames = num_frames

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
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats
        self.output_dim = output_dim

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.cond_kptcammask_prob = kargs.get('cond_kptcammask_prob', 0.5)
        self.cond_camvelmask_prob = kargs.get("cond_camvelmask_prob", 0.5)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.use_motion_mask = use_motion_mask
        self.use_global_constraints = use_global_constraints
        self.global_feat_type = global_feat_type

        input_dim = (self.input_feats * 2 if use_motion_mask else self.input_feats) + self.gru_emb_dim
        if use_obs_diff:
            input_dim += 67
        if use_global_constraints:
            self.global_feat_dims = 66 * len(global_feat_type)
            input_dim += self.global_feat_dims
        # self.input_process = InputProcess(self.data_rep, input_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.sequence_time_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        motion_pe = torch.zeros(1, 1, self.latent_dim)
        self.motion_pe = nn.Parameter(motion_pe, requires_grad=False)
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

        self.final_layer = Mlp(self.latent_dim, out_features=self.output_dim)
        self.add_cond_linear = nn.Linear(self.input_feats * 2 + self.latent_dim, self.latent_dim)

        if self.arch == 'trans_enc':
            print("TRANS_ROPE init")
            mlp_ratio=4.0

            # self.seqTransEncoder = nn.ModuleList(
            #     [
            #         EncoderRoPEBlock(
            #             self.latent_dim,
            #             self.num_heads,
            #             mlp_ratio=mlp_ratio,
            #             dropout=self.dropout,
            #         )
            #         for _ in range(self.num_layers)
            #     ]
            # )
            self.seqTransEncoder_old = nn.ModuleList(
                [
                    EncoderRoPEBlock_old(
                        self.latent_dim,
                        self.num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=self.dropout,
                    )
                    for _ in range(self.num_layers)
                ]
            )

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_time_encoder
        )

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.llm_dim, self.latent_dim)
                print('EMBED TEXT')
                print(f'Loading {llm_type}-{llm_version}...')
                self.llm_type = llm_type
                self.llm_version = llm_version
                self.ext_models['llm'], self.tokenizer = self.load_and_freeze_llm(llm_version)
                # text = ["translate English to German: The house is wonderful."]
                # encoded_text = self.encode_text(text)
            self.output_orient = None

        # self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
        #                                     self.nfeats)

        static_conf_dim = 6
        self.static_conf_head = Mlp(self.latent_dim, out_features=static_conf_dim)
        if 'pred_cam' in self.args.out_attr:
            pred_cam_dim = self.args.out_attr.pred_cam
            self.pred_cam_head = Mlp(self.latent_dim, out_features=pred_cam_dim)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)

        if 'cam_t_vel' in self.args.out_attr:
            cam_t_vel_dim = self.args.out_attr.cam_t_vel
            self.cam_t_vel_head = Mlp(self.latent_dim, out_features=cam_t_vel_dim)
            self.cam_vel_dim = cam_t_vel_dim

        if 'cam_scale' in self.args.out_attr:
            cam_scale_dim = self.args.out_attr.cam_scale
            self.cam_scale_head = Mlp(self.latent_dim, out_features=cam_scale_dim)
            self.cam_scale_dim = cam_scale_dim
        # self.cam_angvel_dim = 6
        # self.cam_angvel_embedder = nn.Sequential(
        #     nn.Linear(self.cam_angvel_dim, self.latent_dim),
        #     nn.SiLU(),
        #     nn.Dropout(dropout),
        #     zero_module(nn.Linear(self.latent_dim, self.latent_dim)),
        # )
        # self.cam_vel_embedder = nn.Sequential(
        #     nn.Linear(self.cam_vel_dim, self.latent_dim),
        #     nn.SiLU(),
        #     nn.Dropout(dropout),
        #     zero_module(nn.Linear(self.latent_dim, self.latent_dim)),
        # )

        if pretrained_checkpoint is not None:
            state_dict = torch.load(pretrained_checkpoint, map_location='cpu')
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
            self.load_model_wo_llm(state_dict)

    def cleanup(self):
        if self.use_precomp_text_embed:
            self.env.close()

    def load_model_wo_llm(self, state_dict):
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('llm_model.') for k in missing_keys])

    def parameters_wo_llm(self):
        return [p for name, p in self.named_parameters() if not name.startswith('llm_model.')]

    
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
            return cond * (1. - rm_text_flag.view((bs,) + (1,) * len(cond.shape[1:])))
        elif force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view((bs,) + (1,) * len(cond.shape[1:]))  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
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
                        raise Exception(f'No data for text {text}')
                    embed = np.frombuffer(data, dtype=np.float32).reshape(-1, self.llm_dim)
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
                truncation=True
            )
            # We expect all the processing is done in GPU.
            input_ids = encoded.input_ids.to(device)
            attn_mask = encoded.attention_mask.to(device)

            with torch.no_grad():
                output = self.ext_models['llm'](input_ids=input_ids, attention_mask=attn_mask)
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

    def forward(self, x, timesteps, y=None, motion_mask=None, observed_motion=None, rm_text_flag=None, rm_kpt_flag=None, global_motion=None, global_joint_mask=None, global_joint_func=None, return_aux=False, clip_cam=True, **kwargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        assert (y is not None) == (self.cond_mode != 'no_cond'
                                   ), "must specify y if and only if the model is class-conditional"
        bs, nframes, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        f_cond = y["f_cond"]

        x = x[:, :, self.s_pred_ind:]
        if motion_mask is not None:
            motion_mask = motion_mask[:, :, self.s_pred_ind:]
        if observed_motion is not None:
            observed_motion = observed_motion[:, :, self.s_pred_ind:]

        if 'text' in self.cond_mode:
            if 'enc_text' not in y:
                enc_text = self.encode_text(y['text'])  # enc_text: [B, max_text_len, llm_dim], emb: [1, B, d]
            else:
                enc_text = y['enc_text']
            force_mask = y.get('uncond', False)
            emb_text = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask, rm_text_flag=rm_text_flag))
            if self.text_enc_mode == 'mean':
                emb_text = emb_text.mean(dim=1)
                emb += emb_text

        if self.use_motion_mask:
            force_motion_mask = y.get("force_motion_mask", None)
            init_motion_mask = y.get("init_motion_mask", None)
            init_observed_motion = y.get("init_observed_motion", None)

            if motion_mask is not None:
                assert observed_motion is not None
                x_orig = x
                if force_motion_mask is not None:
                    raise NotImplementedError
                    force_motion_mask = force_motion_mask.reshape(bs, 1, 1, 1)
                    init_motion_mask = init_motion_mask.transpose(1, 2).unsqueeze(2)
                    init_observed_motion = init_observed_motion.transpose(1, 2).unsqueeze(2)
                    motion_mask = motion_mask * (1 - force_motion_mask) + init_motion_mask * force_motion_mask
                    observed_motion = observed_motion * (1 - force_motion_mask) + init_observed_motion * force_motion_mask
                assert x.shape == motion_mask.shape, (x.shape, motion_mask.shape)
                x = x * (1 - motion_mask) + observed_motion * motion_mask

                x = torch.cat([x, motion_mask], axis=2)
            else:
                if force_motion_mask is not None:
                    raise NotImplementedError
                    force_motion_mask = force_motion_mask.reshape(bs, 1, 1, 1)
                    init_motion_mask = init_motion_mask.transpose(1, 2).unsqueeze(2)
                    init_observed_motion = init_observed_motion.transpose(1, 2).unsqueeze(2)
                    motion_mask = motion_mask * (1 - force_motion_mask) + init_motion_mask * force_motion_mask
                    observed_motion = observed_motion * (1 - force_motion_mask) + init_observed_motion * force_motion_mask
                else:
                    observed_motion = x
                    motion_mask = torch.zeros_like(x)
                x = x * (1 - motion_mask) + observed_motion * motion_mask

                x = torch.cat([x, motion_mask], axis=2)
            assert not self.use_obs_diff

        xseq = self.add_cond_linear(torch.cat([x, f_cond], dim=2))

        # adding the timestep embed
        pose_start_ind = 1

        xseq = torch.cat((emb.transpose(0, 1), xseq), axis=1)  # [bs, seqlen+1, d]
        if self.text_enc_mode == 'seq_concat':
            xseq = torch.cat((emb_text, xseq), axis=1)
            pose_start_ind += emb_text.shape[1]

        xseq_start = xseq[:, :pose_start_ind, :]
        xseq_start = self.sequence_pos_encoder(xseq_start, batch_first=True)  # [bs, seqlen+1, 2d]
        xseq[:, :pose_start_ind, :] = xseq_start
        # xseq[pose_start_ind:] = xseq[pose_start_ind:] + self.motion_pe
        # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, 2d]

        if nframes > self.num_frames:
            attnmask = torch.ones((nframes, nframes), device=x.device, dtype=torch.bool)
            for i in range(nframes):
                min_ind = max(0, i - self.num_frames // 2)
                max_ind = min(nframes, i + self.num_frames // 2)
                max_ind = max(self.num_frames, max_ind)
                min_ind = min(nframes - self.num_frames, min_ind)
                attnmask[i, min_ind:max_ind] = False
            attnmask_full = torch.ones((nframes + pose_start_ind, nframes + pose_start_ind), device=x.device, dtype=torch.bool)
            attnmask_full[pose_start_ind:, pose_start_ind:] = attnmask
            attnmask = attnmask_full
            attnmask = attnmask_full[pose_start_ind:, pose_start_ind:]
        else:
            attnmask = None

        pmask = ~y['mask']  # (B, L)
        pmask = torch.cat([torch.zeros_like(pmask[:, :1]).repeat(1, pose_start_ind), pmask], dim=1)

        # for block in self.seqTransEncoder:
        #     xseq = block(
        #         xseq,
        #         attn_mask=attnmask,
        #         tgt_key_padding_mask=pmask,
        #         pose_start_ind=pose_start_ind,
        #     )
        xseq = xseq[:, pose_start_ind:]
        pmask = ~y["mask"]
        for block in self.seqTransEncoder_old:
            xseq = block(xseq, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        out_aux = {}

        # Output
        sample = self.final_layer(xseq)  # (B, L, C)

        length = y['length']
        L = sample.shape[1]
        if True:
            s_ind = self.s_pred_ind + 6
            betas = (sample[..., s_ind + 126:s_ind + 136] * (~pmask[..., None])).sum(1) / length[:, None]  # (B, C)
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :s_ind + 126], betas, sample[..., s_ind + 136:]], dim=-1)

        static_conf_logits = self.static_conf_head(xseq)
        sample[..., self.s_pred_ind + 3 :self.s_pred_ind + 3 + 6] = static_conf_logits

        if self.use_motion_mask and motion_mask is not None:
            if self.obs_motion_constraints == 'hard':
                sample[..., self.s_pred_ind:] = sample[..., self.s_pred_ind:] * (1 - motion_mask) + observed_motion * motion_mask

        output = {"pred_x_start": sample}
        # predict camera
        if 'pred_cam' in self.args.out_attr:
            pred_cam = self.pred_cam_head(xseq)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            if clip_cam:
                torch.clamp_min_(pred_cam[..., 0], 0.25)  # min_clamp s to 0.25 (prevent negative prediction)
            output["pred_cam"] = pred_cam

        if 'cam_t_vel' in self.args.out_attr:
            pred_cam_t_vel = self.cam_t_vel_head(xseq)
            output["pred_cam_t_vel"] = pred_cam_t_vel

        if 'cam_scale' in self.args.out_attr:
            pred_cam_scale = self.cam_scale_head(xseq)
            pred_cam_scale = (pred_cam_scale * ~pmask[..., None]).sum(1) / length[:, None]
            pred_cam_scale = repeat(pred_cam_scale, "b c -> b l c", l=L).clone()
            output["pred_cam_scale"] = torch.clamp_min_(pred_cam_scale, 0.1)  # min_clamp s to 0.1 (prevent negative prediction)

        return output


    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # self.rot2xyz.smpl_model.to(*args, **kwargs)


    def eval(self, *args, **kwargs):
        super().eval(*args, **kwargs)
        # self.rot2xyz.smpl_model.eval(*args, **kwargs)


    def get_excluded_keys(self):
        return ['llm_model']
