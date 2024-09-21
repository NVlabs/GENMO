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


class MDMDenoiserROPE(nn.Module):
    def __init__(self, pl_module, modeltype, njoints, nfeats, num_frames, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", padding_mask=False, legacy=False, data_rep='rot', dataset='amass', llm_dim=512, max_text_len=20,
                 arch='trans_enc', emb_trans_dec=False, add_x_to_memory=False, autoregressive=False, llm_type="t5", llm_version="t5-small", text_enc_mode='mean', pretrained_checkpoint=None, 
                 use_precomp_text_embed=False, llm_embed_lmdb_path=None, use_motion_mask=False, obs_motion_constraints='hard', use_obs_diff=False, use_global_constraints=False, 
                 global_feat_type=['g_motion_l_joints', 'l_joints_diff', 'global_joint_mask'], **kargs):
        super().__init__()

        self.ext_models = dict()

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
        self.input_process = InputProcess(self.data_rep, input_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.sequence_time_encoder = PositionalEncoding(self.latent_dim, self.dropout)
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
        

        if self.arch == 'trans_enc':
            print("TRANS_ROPE init")
            mlp_ratio=4.0

            self.seqTransEncoder = nn.ModuleList(
                [
                    EncoderRoPEBlock(
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

            if "action" in self.cond_mode:
                if self.action_emb == 'scalar':
                    self.embed_action = EmbedActionScalar(in_features=1, out_features=self.latent_dim,
                                                          activation=self.activation)
                elif self.action_emb == 'tensor':
                    self.embed_action = EmbedActionTensor(self.num_actions, self.latent_dim)
                else:
                    raise Exception(f'Unknown action embedding {self.action_emb}.')
                print('EMBED ACTION')
            if 'kpt2d+cam_angvel' in self.cond_mode:
                self.embed_kpt2d = EmbedActionScalar(
                    in_features=37 * 2,
                    out_features=self.latent_dim // 2,
                    activation=self.activation,
                )
                self.embed_cam_angvel = EmbedActionScalar(
                    in_features=6,
                    out_features=self.latent_dim // 2,
                    activation=self.activation,
                )
                n_joints = 17
                self.mask_kpt_embedding = nn.Parameter(torch.zeros(1, 1, n_joints, 2))
                self.embed_kptproj = nn.Linear(self.latent_dim * 2, self.latent_dim)
                print("EMBED KPT2D+CAM_ANGVEL")

                self.output_orient = nn.Linear(self.latent_dim, 6)
            if 'local_smplfeat' in self.cond_mode:
                self.embed_local_smplfeat = EmbedActionScalar(
                    in_features=147,
                    out_features=self.latent_dim,
                    activation=self.activation,
                )
                print('EMBED LOCAL SMPL_FEAT')
                self.output_orient = nn.Linear(self.latent_dim, 6)
            
            if 'global_smplfeat' in self.cond_mode:
                self.embed_global_smplfeat = EmbedActionScalar(
                    in_features=147,
                    out_features=self.latent_dim,
                    activation=self.activation,
                )
                print('EMBED GLOBAL SMPL_FEAT')
                self.output_orient = nn.Linear(self.latent_dim, 6)
            if "cam_vel" in self.cond_mode:
                self.embed_cam_vel = EmbedActionScalar(
                    in_features=9,
                    out_features=self.latent_dim,
                    activation=self.activation,
                )
                self.embed_camvelproj = nn.Linear(self.latent_dim * 2, self.latent_dim)
                print("EMBED CAM_VEL")

            if 'cam2world' in self.cond_mode:
                self.embed_cam2world = nn.Linear(9, self.latent_dim)

                print('EMBED CAM2WORLD')
                self.output_orient = nn.Linear(self.latent_dim, 6)

            if "plucker_kpt" in self.cond_mode:
                self.embed_plucker = EmbedActionScalar(
                    in_features=17 * (6 + 1),
                    out_features=self.latent_dim,
                    activation=self.activation,
                )
                print("EMBED PLUCKER KPT")
                self.output_orient = nn.Linear(self.latent_dim, 6)

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
        self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)

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

    def forward(self, x, timesteps, y=None, motion_mask=None, observed_motion=None, rm_text_flag=None, rm_kpt_flag=None, global_motion=None, global_joint_mask=None, global_joint_func=None, return_aux=False, **kwargs):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        assert (y is not None) == (self.cond_mode != 'no_cond'
                                   ), "must specify y if and only if the model is class-conditional"
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        if global_joint_func is not None:
            g_motion_l_joints, l_joints_diff = global_joint_func(x, global_motion, global_joint_mask)

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
        if 'action' in self.cond_mode:
            if not (y['action'] == -1).any():  # FIXME - a hack so we can use already trained models
                action_emb = self.embed_action(y['action'])
                emb += self.mask_cond(action_emb)

        if self.use_motion_mask:
            force_motion_mask = y.get("force_motion_mask", None)
            init_motion_mask = y.get("init_motion_mask", None)
            init_observed_motion = y.get("init_observed_motion", None)

            if motion_mask is not None:
                assert observed_motion is not None
                x_orig = x
                if force_motion_mask is not None:
                    force_motion_mask = force_motion_mask.reshape(bs, 1, 1, 1)
                    init_motion_mask = init_motion_mask.transpose(1, 2).unsqueeze(2)
                    init_observed_motion = init_observed_motion.transpose(1, 2).unsqueeze(2)
                    motion_mask = motion_mask * (1 - force_motion_mask) + init_motion_mask * force_motion_mask
                    observed_motion = observed_motion * (1 - force_motion_mask) + init_observed_motion * force_motion_mask

                x = x * (1 - motion_mask) + observed_motion * motion_mask

                if self.use_obs_diff:
                    motion_diff = ((observed_motion - x_orig) * motion_mask)[:, :67]
                    x = torch.cat([x, motion_mask, motion_diff], axis=1)
                else:
                    x = torch.cat([x, motion_mask], axis=1)
            else:
                if force_motion_mask is not None:
                    force_motion_mask = force_motion_mask.reshape(bs, 1, 1, 1)
                    init_motion_mask = init_motion_mask.transpose(1, 2).unsqueeze(2)
                    init_observed_motion = init_observed_motion.transpose(1, 2).unsqueeze(2)
                    motion_mask = motion_mask * (1 - force_motion_mask) + init_motion_mask * force_motion_mask
                    observed_motion = observed_motion * (1 - force_motion_mask) + init_observed_motion * force_motion_mask
                else:
                    observed_motion = x
                    motion_mask = torch.zeros_like(x)
                x = x * (1 - motion_mask) + observed_motion * motion_mask

                if self.use_obs_diff:
                    x = torch.cat([x, motion_mask, torch.zeros_like(x[:, :67])], axis=1)
                else:
                    x = torch.cat([x, motion_mask], axis=1)
            assert not self.use_obs_diff
            
        if self.use_global_constraints:
            if global_motion is not None:
                local_vars = locals()
                global_feat = torch.cat([local_vars[feat] for feat in self.global_feat_type], dim=-1)
                x = torch.cat([x, global_feat.transpose(1, 2).unsqueeze(2)], axis=1)
            else:
                x = torch.cat([x, torch.zeros((x.shape[0], self.global_feat_dims, *x.shape[2:]), device=x.device, dtype=x.dtype)], axis=1)

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            pose_start_ind = 1
            xseq = x

            # xseq = torch.cat((emb, xseq), axis=0)  # [seqlen+1, bs, d]
            xseq = torch.cat((emb, xseq), axis=0)  # [seqlen+1, bs, d]
            if self.text_enc_mode == "seq_concat":
                xseq = torch.cat((emb_text.transpose(0, 1), xseq), axis=0)
                pose_start_ind += emb_text.shape[1]

            xseq_start = xseq[:pose_start_ind]
            xseq_start = self.sequence_pos_encoder(xseq_start)  # [seqlen+1, bs, 2d]
            xseq[:pose_start_ind] = xseq_start
            xseq[pose_start_ind:] = xseq[pose_start_ind:] + self.sequence_pos_encoder.pe[-1:, :]
            # xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, 2d]

            if nframes > self.num_frames:
                attnmask = torch.ones((nframes, nframes), device=x.device, dtype=torch.bool)
                for i in range(nframes):
                    min_ind = max(0, i - self.num_frames // 2)
                    max_ind = min(nframes, i + self.num_frames // 2)
                    max_ind = max(self.num_frames, max_ind)
                    min_ind = min(nframes - self.num_frames, min_ind)
                    attnmask[i, min_ind:max_ind] = False
            else:
                attnmask = None

            pmask = ~y['mask'][:, 0, 0, :]
            pmask = torch.cat([torch.zeros_like(pmask[:, :1]).repeat(1, pose_start_ind), pmask], dim=1)
            xseq = xseq.transpose(0, 1)
            for block in self.seqTransEncoder:
                xseq = block(
                    xseq,
                    attn_mask=attnmask,
                    tgt_key_padding_mask=pmask,
                    pose_start_ind=pose_start_ind,
                )
            output = xseq.transpose(0, 1)[pose_start_ind:]

        out_aux = {}

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        # predict camera
        pred_cam = output[:, 17 * 2 + 3 + 6 : 17 * 2 + 3 + 6 + 3, :, :]     # [bs, 3, nfeats, nframes]
        pred_cam = pred_cam * self.pred_cam_std.reshape(1, 3, 1, 1) + self.pred_cam_mean.reshape(1, 3, 1, 1)
        torch.clamp_min_(pred_cam[..., 0], 0.25)  # min_clamp s to 0.25 (prevent negative prediction)
        output[:, 17 * 2 + 3 + 6 : 17 * 2 + 3 + 6 + 3, :, :] = pred_cam

        if self.use_motion_mask and motion_mask is not None:
            if self.obs_motion_constraints == 'hard':
                output = output * (1 - motion_mask) + observed_motion * motion_mask
        
        if return_aux:
            return output, out_aux
        else:
            return output


    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # self.rot2xyz.smpl_model.to(*args, **kwargs)


    def eval(self, *args, **kwargs):
        super().eval(*args, **kwargs)
        # self.rot2xyz.smpl_model.eval(*args, **kwargs)


    def get_excluded_keys(self):
        return ['llm_model']
