import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from timm.models.vision_transformer import Mlp

from hmr4d.configs import MainStore, builds
from hmr4d.model.gvhmr.utils.geom import lookat_correct, spherical_to_cartesian
from hmr4d.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock
from hmr4d.network.base_arch.transformer.layer import zero_module
from hmr4d.utils.geo.hmr_cam import (
    get_bbx_xys,
    normalize_kp2d,
    perspective_projection,
    safely_render_x3d_K,
)
from hmr4d.utils.net_utils import length_to_mask
from motiondiff.models.mdm.modules import PositionalEncoding
from motiondiff.utils.torch_transform import (
    angle_axis_to_quaternion,
    angle_axis_to_rotation_matrix,
    inverse_transform,
    make_transform,
    quat_mul,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_angle_axis,
)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])


class NetworkEncoderRoPE(nn.Module):
    def __init__(
        self,
        # x
        output_dim=151,
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
        num_views=4,
        default_radius=8.0,
        use_gv_for_mv2d=True,
        use_fp32_for_cam=False,
        cam_gen_nograd=False,
        use_elevation=None,
        use_radius=None,
        use_tilt=None,
        clamp_elevation=None,
        clamp_tilt=None,
        average_cam_across_frames=False,
    ):
        super().__init__()

        # input
        self.output_dim = output_dim
        self.max_len = max_len

        # condition
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim

        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # ===== build model ===== #
        # Input (Kp2d)
        # Main token: map d_obs 2 to 32
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(17, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            17 * 32,
            hidden_features=self.latent_dim * 2,
            out_features=self.latent_dim,
            drop=dropout,
        )

        self._build_condition_embedder()

        self.learned_pos_linear_3d = nn.Linear(2, 32)
        self.learned_pos_params_3d = nn.Parameter(
            torch.randn(17, 32), requires_grad=True
        )
        self.embed_noisyobs_3d = Mlp(
            17 * 32,
            hidden_features=self.latent_dim * 2,
            out_features=self.latent_dim,
            drop=dropout,
        )

        # Transformer
        self.blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(
                    self.latent_dim,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.blocks_singleview2d = nn.ModuleList(
            [
                EncoderRoPEBlock(
                    self.latent_dim,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.num_views = num_views
        self.mv2d_dim = num_views * 17 * 2
        # self.mv2d_head = Mlp(self.latent_dim, out_features=self.mv2d_dim)

        self.singleview2d_head = Mlp(self.latent_dim, out_features=17 * 2)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        # Output heads
        self.final_layer = Mlp(self.latent_dim, out_features=self.output_dim)
        self.pred_cam_head = (
            pred_cam_dim > 0
        )  # keep extra_output for easy-loading old ckpt
        if self.pred_cam_head:
            self.pred_cam_head = Mlp(self.latent_dim, out_features=pred_cam_dim)
            self.register_buffer(
                "pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False
            )
            self.register_buffer(
                "pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False
            )
        self.proj_2d_cam_head = Mlp(self.latent_dim, out_features=3)

        self.static_conf_head = static_conf_dim > 0
        if self.static_conf_head:
            self.static_conf_head = Mlp(self.latent_dim, out_features=static_conf_dim)

        self.avgbeta = avgbeta
        self.endecoder = None
        self.img_h = self.img_w = 1024
        self.get_naive_intrinsics((self.img_w, self.img_h), focal_scale=2.0)
        self.default_radius = default_radius
        self.use_gv_for_mv2d = use_gv_for_mv2d
        self.use_fp32_for_cam = use_fp32_for_cam
        self.cam_gen_nograd = cam_gen_nograd
        self.use_elevation = use_elevation
        self.use_radius = use_radius
        self.use_tilt = use_tilt
        self.clamp_elevation = clamp_elevation
        self.clamp_tilt = clamp_tilt
        self.average_cam_across_frames = average_cam_across_frames

    def _build_condition_embedder(self):
        latent_dim = self.latent_dim
        dropout = self.dropout
        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(self.cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        if self.cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(self.cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if self.imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
            )
            self.imgseq_embedder_3d = nn.Sequential(
                nn.LayerNorm(self.imgseq_dim),
                zero_module(nn.Linear(self.imgseq_dim, latent_dim)),
            )

    def forward(
        self,
        length,
        obs=None,
        f_cliffcam=None,
        f_cam_angvel=None,
        f_imgseq=None,
        detach_j3d_for_mv2d=False,
        detach_cam_for_mv2d=False,
        proj_2d_cam=None,
        **kwargs,
    ):
        """
        Args:
            x: None we do not use it
            timesteps: (B,)
            length: (B), valid length of x, if None then use x.shape[2]
            f_imgseq: (B, L, C)
            f_cliffcam: (B, L, 3), CLIFF-Cam parameters (bbx-detection in the full-image)
            f_noisyobs: (B, L, C), nosiy pose observation
            f_cam_angvel: (B, L, 6), Camera angular velocity
        """
        B, L, J, C = obs.shape
        assert J == 17 and C == 3

        # Main token from observation (2D pose)
        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
        obs[~visible_mask[..., 0]] = 0  # set low-conf to all zeros
        f_obs = self.learned_pos_linear_3d(obs[..., :2])  # (B, L, J, 32)
        f_obs = (
            f_obs * visible_mask
            + self.learned_pos_params_3d.repeat(B, L, 1, 1) * ~visible_mask
        )
        x = self.embed_noisyobs_3d(f_obs.view(B, L, -1))  # (B, L, J*32) -> (B, L, C)

        # Condition
        f_to_add = []
        f_to_add.append(self.cliffcam_embedder(f_cliffcam))
        if hasattr(self, "cam_angvel_embedder"):
            f_to_add.append(self.cam_angvel_embedder(f_cam_angvel))
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            f_to_add.append(self.imgseq_embedder_3d(f_imgseq))

        for f_delta in f_to_add:
            x = x + f_delta

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - self.max_len // 2)
                max_ind = min(L, i + self.max_len // 2)
                max_ind = max(self.max_len, max_ind)
                min_ind = min(L - self.max_len, min_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        # Transformer
        for block in self.blocks:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        # MV2D
        # mv2d = self.mv2d_head(x)
        # mv2d = mv2d.view(B, L, self.num_views, 17, 2)

        # Output
        sample = self.final_layer(x)  # (B, L, C)
        if self.avgbeta:
            betas = (sample[..., 126:136] * (~pmask[..., None])).sum(1) / length[
                :, None
            ]  # (B, C)
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)

        # Output (extra)
        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(
                pred_cam[..., 0], 0.25
            )  # min_clamp s to 0.25 (prevent negative prediction)

        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x)  # (B, L, C')

        if proj_2d_cam is None:
            proj_2d_cam = self.proj_2d_cam_head(x)

        output = {
            "pred_context": x,
            "pred_x": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
            "proj_2d_cam": proj_2d_cam,
            # "mv2d": mv2d,
        }

        endecoder = self.endecoder[0]
        decode_dict = endecoder.decode(output["pred_x"])  # (B, L, C) -> dict

        grot_key = "global_orient_gv" if self.use_gv_for_mv2d else "global_orient"
        grot_mat = angle_axis_to_rotation_matrix(decode_dict[grot_key])
        base_rot = angle_axis_to_rotation_matrix(
            torch.tensor([[0.0, 0.0, -0.5 * np.pi]]).to(grot_mat)
        ) @ angle_axis_to_rotation_matrix(
            torch.tensor([[-0.5 * np.pi, 0.0, 0.0]]).to(grot_mat)
        )
        grot_world = base_rot @ grot_mat
        grot_world = rotation_matrix_to_angle_axis(grot_world)
        smpl_gv_dict = {
            "body_pose": decode_dict["body_pose"],  # (B, L, 63)
            "betas": decode_dict["betas"],  # (B, L, 10)
            "global_orient": grot_world,  # (B, L, 3)
            "transl": torch.zeros_like(grot_world),  # (B, L, 3)
        }
        with torch.set_grad_enabled(not detach_j3d_for_mv2d):
            _, j3d = endecoder.smplx_model(**smpl_gv_dict)

        if detach_cam_for_mv2d:
            proj_2d_cam = proj_2d_cam.detach()
        mv2d_cam_params = self.obtain_mv2d_cam_params(proj_2d_cam)
        with torch.autocast(device_type="cuda", enabled=not self.use_fp32_for_cam):
            with torch.set_grad_enabled(not self.cam_gen_nograd):
                cam_dict = self.generate_cam(mv2d_cam_params)
        kp2d = self.project_keypoints(j3d, cam_dict)
        bbx_xys = get_bbx_xys(kp2d, do_augment=False)
        kp2d_norm = normalize_kp2d(
            torch.cat([kp2d, torch.ones_like(kp2d[..., :1])], dim=-1), bbx_xys
        )[..., :2]
        output["mv2d"] = kp2d_norm
        output["mv2d_cam_params"] = mv2d_cam_params
        output["decode_dict"] = decode_dict
        return output

    def forward_singleview(self, length, obs_x_t, t, f_imgseq=None, **kwargs):
        """
        Args:
            x: None we do not use it
            timesteps: (B,)
            length: (B), valid length of x, if None then use x.shape[2]
            f_imgseq: (B, L, C)
            f_cliffcam: (B, L, 3), CLIFF-Cam parameters (bbx-detection in the full-image)
            f_noisyobs: (B, L, C), nosiy pose observation
            f_cam_angvel: (B, L, 6), Camera angular velocity
        """
        B, L, J, C = obs_x_t.shape
        assert J == 17

        # Main token from observation (2D pose)
        obs = obs_x_t.clone()
        f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
        x = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, J*32) -> (B, L, C)

        emb = self.embed_timestep(t)  # [1, bs, d]
        x = x + emb

        # Condition
        f_to_add = []
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            f_to_add.append(self.imgseq_embedder(f_imgseq))

        for f_delta in f_to_add:
            x = x + f_delta

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - self.max_len // 2)
                max_ind = min(L, i + self.max_len // 2)
                max_ind = max(self.max_len, max_ind)
                min_ind = min(L - self.max_len, min_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        # Transformer
        for block in self.blocks_singleview2d:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        # MV2D
        singleview_2d = self.singleview2d_head(x)
        singleview_2d = singleview_2d.view(B, L, 17, 2)

        output = {
            "pred_context": x,
            "singleview_2d": singleview_2d,
        }
        return output

    def get_denoiser(self):
        def denoiser(x, t, reshape=False, **kwargs):
            if reshape:
                shape = x.shape
                x = x.view(shape[0], -1, 2, shape[-1]).permute(0, 3, 1, 2)
            res = self.forward_singleview(obs_x_t=x, t=t, **kwargs)
            x0 = res["singleview_2d"]
            if reshape:
                x0 = x0.permute(0, 2, 3, 1).view(shape)
            return x0

        return denoiser

    def get_naive_intrinsics(self, res, focal_scale=1.0):
        # Assume 45 degree FOV
        img_w, img_h = res
        self.focal_length = (img_w * img_w + img_h * img_h) ** 0.5 * focal_scale
        self.cam_intrinsics = torch.eye(3).repeat(1, 1, 1).float()
        self.cam_intrinsics[:, 0, 0] = self.focal_length
        self.cam_intrinsics[:, 1, 1] = self.focal_length
        self.cam_intrinsics[:, 0, 2] = img_w / 2.0
        self.cam_intrinsics[:, 1, 2] = img_h / 2.0

    def obtain_mv2d_cam_params(self, proj_2d_cam):
        elevations, radius, tilt = (
            proj_2d_cam[..., 0],
            proj_2d_cam[..., 1],
            proj_2d_cam[..., 2],
        )
        radius += self.default_radius
        elevations *= 10.0
        tilt *= 10.0
        if self.average_cam_across_frames:
            num_frames = proj_2d_cam.shape[1]
            elevations = elevations.mean(dim=1, keepdim=True).repeat(1, num_frames)
            radius = radius.mean(dim=1, keepdim=True).repeat(1, num_frames)
            tilt = tilt.mean(dim=1, keepdim=True).repeat(1, num_frames)
        if self.use_elevation is not None:
            elevations = torch.ones_like(elevations) * self.use_elevation
        if self.use_radius is not None:
            radius = torch.ones_like(elevations) * self.use_radius
        if self.use_tilt is not None:
            tilt = torch.ones_like(elevations) * self.use_tilt
        if self.clamp_elevation is not None:
            elevations = torch.clamp(
                elevations, min=-self.clamp_elevation, max=self.clamp_elevation
            )
        if self.clamp_tilt is not None:
            tilt = torch.clamp(tilt, min=-self.clamp_tilt, max=self.clamp_tilt)
        return {
            "elevations": elevations,
            "radius": radius,
            "tilt": tilt,
        }

    def generate_cam(self, mv2d_cam_params):
        device = mv2d_cam_params["elevations"].device
        orig_shape = mv2d_cam_params["elevations"].shape[:2]
        elevations = mv2d_cam_params["elevations"].view(-1, 1)
        radius = mv2d_cam_params["radius"].view(-1, 1)
        tilt = mv2d_cam_params["tilt"].view(-1, 1)

        def lookat_correct(eye, at, up):
            zaxis = (at - eye) / torch.norm(at - eye, dim=-1, keepdim=True)
            xaxis = torch.cross(up, zaxis, dim=-1)
            xaxis = xaxis / torch.norm(xaxis, dim=-1, keepdim=True)
            yaxis = torch.cross(zaxis, xaxis, dim=-1)
            view_matrix = torch.zeros((eye.shape[0], 4, 4), device=device)
            view_matrix[:, :3, 0] = xaxis
            view_matrix[:, :3, 1] = yaxis
            view_matrix[:, :3, 2] = zaxis
            view_matrix[:, :3, 3] = eye
            view_matrix[:, 3, 3] = 1.0
            return view_matrix

        def spherical_to_cartesian(r, azimuth, elevation):
            azimuth = torch.deg2rad(azimuth)
            elevation = torch.deg2rad(elevation)
            x = r * torch.cos(elevation) * torch.cos(azimuth)
            y = r * torch.cos(elevation) * torch.sin(azimuth)
            z = r * torch.sin(elevation)
            return torch.stack([x, y, z], dim=-1)

        # elevations = (torch.ones((1, self.num_views)) * 0.0).to(device)
        # radius = (torch.ones((1, self.num_views)) * 8.0).to(device)
        # tilt = (torch.ones((1, self.num_views))).to(device)
        elevations = elevations.repeat(1, self.num_views)
        radius = radius.repeat(1, self.num_views)
        tilt = tilt.repeat(1, self.num_views)
        azimuths = torch.linspace(0, 360, self.num_views + 1)[: self.num_views].to(
            device
        )
        azimuths = azimuths.unsqueeze(0).expand(elevations.shape[0], -1)
        eyes = spherical_to_cartesian(radius, azimuths, elevations)

        eyes_flat = eyes.view(-1, 3)
        at = torch.zeros((eyes_flat.shape[0], 3), device=device)
        up = torch.tensor([0.0, 0.0, 1.0], device=device)[None, :].expand(
            eyes_flat.shape[0], -1
        )
        c2w = lookat_correct(eyes_flat, at, up).reshape(
            eyes.shape[0], self.num_views, 4, 4
        )
        if torch.norm(tilt) > 0:
            tilt_rot = angle_axis_to_rotation_matrix(
                torch.stack(
                    [
                        torch.zeros_like(tilt),
                        torch.zeros_like(tilt),
                        torch.deg2rad(tilt),
                    ],
                    dim=-1,
                )
            )
            c2w = c2w @ make_transform(tilt_rot, torch.zeros(3).to(tilt_rot))
        w2c = inverse_transform(c2w)
        intrinsics = (
            self.cam_intrinsics.to(device)
            .unsqueeze(0)
            .repeat(eyes.shape[0], self.num_views, 1, 1)
        )
        P = torch.matmul(intrinsics, w2c[..., :3, :])

        cam_dict = {
            "c2w": c2w.reshape(orig_shape + c2w.shape[1:]),
            "w2c": w2c.reshape(orig_shape + w2c.shape[1:]),
            "intrinsics": intrinsics.reshape(orig_shape + intrinsics.shape[1:]),
            "P": P.reshape(orig_shape + P.shape[1:]),
            "azimuths": azimuths.reshape(orig_shape + azimuths.shape[1:]),
            "elevations": elevations.reshape(orig_shape + elevations.shape[1:]),
            "radius": radius.reshape(orig_shape + radius.shape[1:]),
        }

        return cam_dict

    def project_keypoints(self, kpt3d, cam_dict):
        kpt3d_pad = torch.cat((kpt3d, torch.ones_like(kpt3d[..., :1])), dim=-1)
        local_kpt2d_new = (
            cam_dict["P"] @ kpt3d_pad[:, :, None].transpose(-1, -2)
        ).transpose(-1, -2)
        local_kpt2d_new = local_kpt2d_new[..., :2] / local_kpt2d_new[..., 2:]
        local_kpt2d_new[..., 1] = self.img_h - local_kpt2d_new[..., 1]
        return local_kpt2d_new


# Add to MainStore
group_name = "network/mv2d"
MainStore.store(
    name="relative_transformer",
    node=builds(NetworkEncoderRoPE, populate_full_signature=True),
    group=group_name,
)
