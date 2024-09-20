import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

from hmr4d.utils.net_utils import length_to_mask
from hmr4d.network.base_arch.transformer.layer import zero_module
from motiondiff.models.model_util import create_gaussian_diffusion
from motiondiff.models.common.cfg_sampler import ClassifierFreeSampleModel
from motiondiff.diffusion.resample import create_named_schedule_sampler
from hmr4d.utils.geo.hmr_cam import (
    compute_bbox_info_bedlam,
    compute_transl_full_cam,
    get_a_pred_cam,
    project_to_bi01,
)
from hmr4d.utils.geo.hmr_global import (
    rollout_local_transl_vel,
    get_static_joint_mask,
    get_tgtcoord_rootparam,
)

def import_type_from_str(s):
    module_name, type_name = s.rsplit(".", 1)
    module = importlib.import_module(module_name)
    type_to_import = getattr(module, type_name)
    return type_to_import



class MDMBase(nn.Module):
    def __init__(
        self,
        model_cfg,
        motion_rep_dim,
        max_len=120,
        # condition
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        # intermediate
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        # output
        pred_cam_dim=3,
        static_conf_dim=6,
        # training
        dropout=0.1,
        # other
        avgbeta=True,
        **kwargs,
    ):
        super().__init__()
        self.model_cfg = model_cfg

        self.motion_rep_dim = motion_rep_dim
        self.max_len = max_len
        self.model_cfg.denoiser.njoints = self.motion_rep_dim

        # condition
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim

        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.learned_pos_params = nn.Parameter(torch.randn(17, 2), requires_grad=True)
        self.imgseq_embedder = nn.Sequential(
            nn.LayerNorm(self.imgseq_dim),
            zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
        )

        self.load_ext_models()

        return

    def load_pretrain_checkpoint(self):
        if 'pretrained_checkpoint' in self.model_cfg:
            cp_cfg = self.model_cfg.pretrained_checkpoint
            state_dict = torch.load(cp_cfg.path, map_location='cpu')['state_dict']
            filter_keys = cp_cfg.get('filter_keys', [])
            try_load = cp_cfg.get('try_load', False)
            if len(filter_keys) > 0:
                print(f'Filtering checkpoint keys: {filter_keys}')
                skipped_keys = [k for k in state_dict.keys() if any(key in k for key in filter_keys)]
                print(f'Skipped keys: {skipped_keys}')
                state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in filter_keys)}
            if try_load:
                model_state = self.state_dict()
                state_dict = {k: v for k, v in state_dict.items()
                            if k in model_state and v.size() == model_state[k].size()}

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=cp_cfg.get('strict', True))
            if len(missing_keys) > 0:
                print(f'Missing keys: {missing_keys}')
            if len(unexpected_keys) > 0:
                print(f'Unexpected keys: {unexpected_keys}')


    def load_ext_models(self):
        self.ext_models = {}
        em_cfg = self.model_cfg.get('ext_models', {})
        for name, cfg in em_cfg.items():
            em_cfg = import_type_from_str(cfg.config.type)(**cfg.config.args)
            em = import_type_from_str(em_cfg.model.type)(em_cfg, is_inference=True, preload_checkpoint=False)
            checkpoint = torch.load(cfg.checkpoint, map_location='cpu')['state_dict']
            em.load_state_dict(checkpoint)
            em.eval()
            self.ext_models[name] = em

    def to(self, device):
        super().to(device)
        for key in self.ext_models:
            self.ext_models[key].to(device)
        return

    def init_diffusion(self):
        self.train_diffusion = create_gaussian_diffusion(self.model_cfg.diffusion, training=True)
        self.test_diffusion = create_gaussian_diffusion(self.model_cfg.diffusion, training=False)
        self.schedule_sampler = create_named_schedule_sampler(self.model_cfg.diffusion.schedule_sampler_type, self.train_diffusion)
        self.guided_denoiser = ClassifierFreeSampleModel(self.denoiser)
        return

    def generate_motion_rep(
        self, batch, target_x, gt_pred_cam, static_gt
    ):
        f_condition = batch["f_condition"]

        length = f_condition["length"]
        obs = f_condition["obs"]
        f_cliffcam = f_condition["f_cliffcam"]  # (B, L, 3)
        f_cam_angvel = f_condition["f_cam_angvel"]  # (B, L, 6)
        f_imgseq = f_condition["f_imgseq"]  # (B, L, C)

        B, L, J, C = obs.shape
        assert J == 17 and C == 3
        # Main token from observation (2D pose)
        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
        obs[~visible_mask[..., 0]] = 0  # set low-conf to all zeros
        f_obs = obs[..., :2]  # (B, L, J, 32)
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask  # (B, L, J, 2)
        f_obs = f_obs.reshape(B, L, -1)

        vis_mask = length_to_mask(length, L)  # (B, L)
        pmask = ~vis_mask  # (B, L)

        # Condition
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            f_imgseq = self.imgseq_embedder(f_imgseq)

        # 34 + 3 + 6 + 3 + 6 + 151
        motion = torch.cat([f_obs, f_cliffcam, f_cam_angvel, gt_pred_cam, static_gt, target_x], dim=-1)
        motion_mask = torch.zeros_like(motion)
        motion_mask[..., :17 * 2 + 3 + 6] = 1
        motion_mask = motion_mask * vis_mask[..., None]

        motion = motion * vis_mask[..., None]
        return motion, motion_mask

    def get_diffusion_pred_target(self, batch, target_x, gt_pred_cam, static_gt):
        diffusion = self.train_diffusion if self.training else self.test_diffusion

        outputs = dict()
        f_condition = batch["f_condition"]
        length = f_condition["length"]

        motion, motion_mask = self.generate_motion_rep(
            batch, target_x, gt_pred_cam, static_gt
        )
        B, L, _ = motion.shape
        motion = motion.permute(0, 2, 1).unsqueeze(2).contiguous()
        motion_mask = motion_mask.permute(0, 2, 1).unsqueeze(2).contiguous()
        # Setup length and make padding mask
        vis_mask = length_to_mask(length, L)  # (B, L)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=motion.device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - self.max_len // 2)
                max_ind = min(L, i + self.max_len // 2)
                max_ind = max(self.max_len, max_ind)
                min_ind = min(L - self.max_len, min_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        t, t_weights = self.schedule_sampler.sample(motion.shape[0], motion.device)
        x_start = motion
        x_t = self.train_diffusion.q_sample(x_start, t)

        denoiser_kwargs = {
            "y": {
                "text": [""] * B,
                "mask": vis_mask,
            },
            "motion_mask": motion_mask,
            "observed_motion": motion,
        }
        pred_x_start = self.denoiser(
            x_t, diffusion._scale_timesteps(t), return_aux=False, **denoiser_kwargs
        )
        pred_x_start = pred_x_start.squeeze(2).permute(0, 2, 1).contiguous()

        pred_cam = pred_x_start[:, :, 17 * 2 + 3 + 6:17 * 2 + 3 + 6 + 3]
        static_conf_logits = pred_x_start[:, :, 17 * 2 + 3 + 6 + 3:17 * 2 + 3 + 6 + 3 + 6]
        sample = pred_x_start[:, :, -151:]

        output = {
            "pred_x": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
        }
        return output

    def forward_train(self, inputs, train=False, postproc=False, static_cam=False):
        outputs = dict()
        length = inputs["length"]  # (B,) effective length of each sample

        target_x = self.endecoder.encode(inputs)  # (B, L, C)

        gt_transl = inputs["smpl_params_c"]["transl"]  # (B, L, 3)
        gt_pred_cam = get_a_pred_cam(gt_transl, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
        gt_pred_cam[gt_pred_cam.isinf()] = -1  # this will be handled by valid_mask
        gt_j3d_z_min = inputs["gt_j3d"][..., 2].min(dim=-1)[0]
        valid_cam_mask = (
            (gt_j3d_z_min > 0.3)
            * (gt_pred_cam[..., 0] > 0.3)
            * (gt_pred_cam[..., 0] < 5.0)
            * (gt_pred_cam[..., 1] > -3.0)
            * (gt_pred_cam[..., 1] < 3.0)
            * (gt_pred_cam[..., 2] > -3.0)
            * (gt_pred_cam[..., 2] < 3.0)
            * (inputs["bbx_xys"][..., 2] > 0)
        )[..., None].float()

        # vel_thr = args.static_conf.vel_thr
        # assert vel_thr > 0
        vel_thr = 0.15
        joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
        gt_w_j3d = self.endecoder.fk_v2(**inputs["smpl_params_w"])  # (B, L, J=22, 3)
        static_gt = get_static_joint_mask(gt_w_j3d, vel_thr=vel_thr, repeat_last=True)  # (B, L, J)
        static_gt = static_gt[:, :, joint_ids].float()  # (B, L, J')

        output = self.get_diffusion_pred_target(
            batch=inputs,
            target_x=target_x,
            gt_pred_cam=gt_pred_cam,
            static_gt=static_gt,
        )
        return output

    def forward_test(self, inputs, train=False, postproc=False, static_cam=False, progress=False):
        diffusion = self.test_diffusion
        denoiser = self.guided_denoiser
        length = inputs["length"]  # (B,) effective length of each sample

        f_condition = inputs["f_condition"]
        cliff_cam = f_condition["f_cliffcam"]
        B, L = cliff_cam.shape[:2]

        target_x = torch.zeros(B, L, 206).to(cliff_cam)
        gt_pred_cam = torch.zeros_like(cliff_cam)
        static_gt = torch.zeros(B, L, 6).to(cliff_cam)
        motion, motion_mask = self.generate_motion_rep(
            inputs, target_x, gt_pred_cam, static_gt
        )
        motion = motion.permute(0, 2, 1).unsqueeze(2).contiguous()
        motion_mask = motion_mask.permute(0, 2, 1).unsqueeze(2).contiguous()

        vis_mask = length_to_mask(length, L)  # (B, L)

        cond = {
            "y": {
                "text": [""] * B,
                "mask": vis_mask,
            },
            "motion_mask": motion_mask,
            "observed_motion": motion,
        }

        length = inputs["length"]  # (B,) effective length of each sample
        batch_size = length.shape[0]

        cond["y"]["scale"] = (
            torch.ones(batch_size, device=length.device)
            * self.model_cfg.diffusion.guidance_param
        )
        diff_sampler = self.model_cfg.diffusion.get("sampler", "ddim")
        if diff_sampler == "ddim":
            sample_fn = diffusion.ddim_sample_loop
            kwargs = {"eta": self.model_cfg.diffusion.ddim_eta}
        else:
            sample_fn = diffusion.p_sample_loop
            kwargs = {}

        samples = sample_fn(
            denoiser,
            (batch_size, self.denoiser.njoints, self.denoiser.nfeats, self.max_len),
            clip_denoised=False,
            model_kwargs=cond,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=progress,
            dump_steps=None,
            noise=None,
            const_noise=False,
            **kwargs,
        )
        return samples

    def forward(self, inputs, train=False, postproc=False, static_cam=False):
        if train:
            return self.forward_train(inputs, train, postproc, static_cam)
        else:
            return self.forward_test(inputs, train, postproc, static_cam)