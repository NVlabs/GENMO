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



class MDMDenoiser(nn.Module):
    def __init__(self, pl_module, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
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
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
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
        if "kpt2d+cam_angvel" in self.cond_mode:
            # emb: [1, B, d]
            # mask: [B, 1, 1]
            # kpt_mask: [B, T, 17]
            # self.mask_kpt_embedding: [1, 1, 17, 2]
            force_mask = y.get("uncond", False)
            if rm_kpt_flag is not None or force_mask:
                cond_mask_2d = torch.zeros(bs, 1, 1, device=x.device)
                kpt_mask = torch.zeros(bs, nframes, 17, device=x.device)
                normed_kpt2d = torch.zeros(bs, nframes, 17 * 2 + 3, device=x.device)
                cam_angvel = torch.zeros(bs, nframes, 6, device=x.device)
            else:
                kpt_mask = y['kpt_mask']
                normed_kpt2d = y['normed_kpt2d']
                cam_angvel = y['cam_angvel']

                if self.training:
                    cond_mask_2d = 1 - torch.bernoulli(torch.ones(bs, device=x.device) * self.cond_kptcammask_prob).view(bs, 1, 1)
                else:
                    cond_mask_2d = torch.ones(bs, 1, 1, device=x.device)

            # Treat masked keypoints
            # mask_kpt_embedding = (1 - kpt_mask.unsqueeze(-1)) * self.mask_kpt_embedding
            # # _kpt_mask = kpt_mask.unsqueeze(-1).repeat(1, 1, 1, 2).reshape(bs, nframes, -1)
            # # _kpt_mask = torch.cat((_kpt_mask, torch.ones_like(_kpt_mask[..., :3])), dim=-1)

            # _mask_kpt_embedding = mask_kpt_embedding.reshape(bs, nframes, -1)
            # _mask_kpt_embedding = torch.cat(
            #     (_mask_kpt_embedding, torch.zeros_like(_mask_kpt_embedding[..., :3])),
            #     dim=-1,
            # )
            # _mask_kpt_embedding = _mask_kpt_embedding * cond_mask_2d

            _kpt_mask = (
                kpt_mask.unsqueeze(-1).repeat(1, 1, 1, 2).contiguous().reshape(bs, nframes, -1)
            )
            _kpt_mask = torch.cat((_kpt_mask, torch.ones_like(_kpt_mask[..., :3])), dim=-1)
            normed_kpt2d = normed_kpt2d * _kpt_mask
            normed_kpt2d = torch.cat((normed_kpt2d, _kpt_mask), dim=-1)

            kpt2d_emb = self.embed_kpt2d(cond_mask_2d * normed_kpt2d) # [B, T, N]
            cam_angvel_emb = self.embed_cam_angvel(cond_mask_2d * cam_angvel)  # [B, T, N]
            # emb += self.mask_cond(kpt2d_emb) + self.mask_cond(cam_angvel_emb)

        if 'local_smplfeat' in self.cond_mode:
            force_mask = y.get("uncond", False)
            if rm_kpt_flag is not None or force_mask:
                local_smplfeat = torch.zeros(bs, nframes, 147, device=x.device)
                cond_mask_local_smplfeat = torch.zeros(bs, 1, 1, device=x.device)
            else:
                local_smplfeat = y['local_smplfeat']
                if self.training:
                    cond_mask_local_smplfeat = 1 - torch.bernoulli(torch.ones(bs, device=x.device) * self.cond_kptcammask_prob).view(bs, 1, 1)
                else:
                    cond_mask_local_smplfeat = torch.ones(bs, 1, 1, device=x.device)
            local_smplfeat_emb = self.embed_local_smplfeat(cond_mask_local_smplfeat * local_smplfeat)

        if 'global_smplfeat' in self.cond_mode:
            force_mask = y.get("uncond", False)
            if rm_kpt_flag is not None or force_mask:
                global_smplfeat = torch.zeros(bs, nframes, 147, device=x.device)
                cond_mask_global_smplfeat = torch.zeros(bs, 1, 1, device=x.device)
            else:
                global_smplfeat = y['global_smplfeat']
                if self.training:
                    cond_mask_global_smplfeat = 1 - torch.bernoulli(torch.ones(bs, device=x.device) * self.cond_kptcammask_prob).view(bs, 1, 1)
                else:
                    cond_mask_global_smplfeat = torch.ones(bs, 1, 1, device=x.device)
            global_smplfeat_emb = self.embed_global_smplfeat(cond_mask_local_smplfeat * local_smplfeat)

        if 'cam2world' in self.cond_mode:
            force_mask = y.get("uncond", False)
            if rm_kpt_flag is not None or force_mask:
                cam2world = torch.zeros(bs, nframes, 9, device=x.device)
                cond_mask_cam2world = torch.zeros(bs, 1, 1, device=x.device)
            else:
                cam2world = y['cam2world']
                if self.training:
                    cond_mask_cam2world = 1 - torch.bernoulli(torch.ones(bs, device=x.device) * self.cond_kptcammask_prob).view(bs, 1, 1)
                    if 'local_smplfeat' in self.cond_mode:
                        cond_mask_cam2world = cond_mask_cam2world * cond_mask_local_smplfeat
                    if 'global_smplfeat' in self.cond_mode:
                        cond_mask_cam2world = cond_mask_cam2world * cond_mask_global_smplfeat
                    if 'kpt2d' in self.cond_mode:
                        cond_mask_cam2world = cond_mask_cam2world * cond_mask_2d
                else:
                    cond_mask_cam2world = torch.ones(bs, 1, 1, device=x.device)
            cam2world_emb = self.embed_cam2world(cond_mask_cam2world * cam2world)

        if 'cam_vel' in self.cond_mode:
            force_mask = y.get("uncond", False)
            if rm_kpt_flag is not None or force_mask:
                cam_vel = torch.zeros(bs, nframes, 9, device=x.device)
                cond_mask_camvel = torch.zeros(bs, 1, 1, device=x.device)
            else:
                cam_vel = y["cam_vel"]
                if self.training:
                    cond_mask_camvel = 1 - torch.bernoulli(torch.ones(bs, device=x.device) * self.cond_camvelmask_prob).view(bs, 1, 1)

                    if 'local_smplfeat' in self.cond_mode:
                        cond_mask_camvel = cond_mask_camvel * cond_mask_local_smplfeat
                    if 'global_smplfeat' in self.cond_mode:
                        cond_mask_camvel = cond_mask_camvel * cond_mask_global_smplfeat
                    if 'kpt2d' in self.cond_mode:
                        cond_mask_camvel = cond_mask_camvel * cond_mask_2d
                else:
                    cond_mask_camvel = torch.ones(bs, 1, 1, device=x.device)

            cam_vel_emb = self.embed_cam_vel(cond_mask_camvel * cam_vel)

        if 'plucker_kpt' in self.cond_mode:
            force_mask = y.get("uncond", False)
            if rm_kpt_flag is not None or force_mask:
                cond_mask_2d = torch.zeros(bs, 1, 1, device=x.device)
                kpt_mask = torch.zeros(bs, nframes, 17, 1, device=x.device)
                plucker_kpt = torch.zeros(bs, nframes, 17, 6, device=x.device)
            else:
                kpt_mask = y['kpt_mask'].reshape(bs, nframes, 17, 1)    # [B, T, 17]
                plucker_kpt = y['plucker_kpt']  # [B, T, 17, 6]

                if self.training:
                    cond_mask_2d = 1 - torch.bernoulli(torch.ones(bs, device=x.device) * self.cond_kptcammask_prob).view(bs, 1, 1)
                else:
                    cond_mask_2d = torch.ones(bs, 1, 1, device=x.device)

            plucker_kpt = plucker_kpt * kpt_mask
            plucker_kpt = torch.cat((plucker_kpt, kpt_mask), dim=-1).reshape(bs, nframes, 17 * 7)
            plucker_kpt_emb = self.embed_plucker(cond_mask_2d * plucker_kpt)

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

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
            if "kpt2d+cam_angvel" in self.cond_mode:
                xseq = torch.cat((x, kpt2d_emb.transpose(0, 1), cam_angvel_emb.transpose(0, 1)), axis=2)  # [seqlen, bs, 2d]
                xseq = self.embed_kptproj(xseq)  # [seqlen+1, bs, d]
                if "cam_vel" in self.cond_mode:
                    xseq = torch.cat((xseq, cam_vel_emb.transpose(0, 1)), axis=2)
                    xseq = self.embed_camvelproj(xseq)
            if "plucker_kpt" in self.cond_mode:
                xseq = xseq + plucker_kpt_emb.transpose(0, 1)
            if 'local_smplfeat' in self.cond_mode:
                xseq = xseq + local_smplfeat_emb.transpose(0, 1)
            if 'global_smplfeat' in self.cond_mode:
                xseq = xseq + global_smplfeat_emb.transpose(0, 1)
            if 'cam_vel' in self.cond_mode:
                xseq = xseq + cam_vel_emb.transpose(0, 1)
            if 'cam2world' in self.cond_mode:
                xseq = xseq + cam2world_emb.transpose(0, 1)

            xseq = torch.cat((emb, xseq), axis=0)  # [seqlen+1, bs, d]
            if self.text_enc_mode == 'seq_concat':
                xseq = torch.cat((emb_text.transpose(0, 1), xseq), axis=0)
                pose_start_ind += emb_text.shape[1]

            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, 2d]
            src_mask = None
            src_key_padding_mask = None
            if self.padding_mask:
                src_key_padding_mask = torch.zeros((xseq.shape[1], xseq.shape[0]), dtype=torch.bool, device=xseq.device)
                src_key_padding_mask[:, pose_start_ind:] = ~y['mask'][:, 0, 0]
            if self.autoregressive:
                src_mask = generate_ar_mask(xseq.shape[0], xseq.shape[0], tgt_start_dim=pose_start_ind, src_start_dim=pose_start_ind).to(x.device)
            output = self.seqTransEncoder(xseq, mask=src_mask, src_key_padding_mask=src_key_padding_mask)[pose_start_ind:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            xseq = x
            pose_start_ind = 0
            memory = emb
            if self.text_enc_mode == 'seq_concat':
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
                tgt_mask = generate_ar_mask(xseq.shape[0], xseq.shape[0], tgt_start_dim=pose_start_ind, src_start_dim=pose_start_ind).to(x.device)
                if self.add_x_to_memory:
                    memory_mask = generate_ar_mask(xseq.shape[0], memory.shape[0], tgt_start_dim=pose_start_ind, src_start_dim=emb_text.shape[1] + 1).to(x.device)

            output = self.seqTransDecoder(tgt=xseq, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)[pose_start_ind:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        out_aux = {}
        if return_aux and self.output_orient is not None:
            # output: [seqlen, bs, d]
            pred_local_orient = self.output_orient(output)
            pred_local_orient = pred_local_orient.transpose(0, 1)  # [bs, seqlen, 6]
            out_aux["pred_local_orient"] = pred_local_orient
            out_aux["cond_mask_2d"] = cond_mask_2d

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
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
