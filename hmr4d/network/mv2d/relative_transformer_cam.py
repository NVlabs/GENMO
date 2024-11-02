import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import einsum, rearrange, repeat
from hmr4d.configs import MainStore, builds

from hmr4d.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock
from hmr4d.network.base_arch.transformer.layer import zero_module
from hmr4d.utils.geo.hmr_cam import perspective_projection, normalize_kp2d, safely_render_x3d_K, get_bbx_xys
from hmr4d.model.gvhmr.utils.geom import lookat_correct, spherical_to_cartesian
from motiondiff.utils.torch_transform import rotation_matrix_to_angle_axis, quaternion_to_angle_axis, quat_mul, angle_axis_to_quaternion, angle_axis_to_rotation_matrix, quaternion_to_rotation_matrix

from hmr4d.utils.net_utils import length_to_mask
from timm.models.vision_transformer import Mlp


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
        num_views=4
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
            17 * 32, hidden_features=self.latent_dim * 2, out_features=self.latent_dim, drop=dropout
        )

        self._build_condition_embedder()

        # Transformer
        self.blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(self.num_layers)
            ]
        )
        
        self.num_views = num_views
        self.mv2d_dim = num_views * 17 * 2
        # self.mv2d_head = Mlp(self.latent_dim, out_features=self.mv2d_dim)

        # Output heads
        self.final_layer = Mlp(self.latent_dim, out_features=self.output_dim)
        self.pred_cam_head = pred_cam_dim > 0  # keep extra_output for easy-loading old ckpt
        if self.pred_cam_head:
            self.pred_cam_head = Mlp(self.latent_dim, out_features=pred_cam_dim)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)

        self.static_conf_head = static_conf_dim > 0
        if self.static_conf_head:
            self.static_conf_head = Mlp(self.latent_dim, out_features=static_conf_dim)

        self.avgbeta = avgbeta
        self.endecoder = None
        self.img_h = self.img_w = 1024
        self.get_naive_intrinsics((self.img_w, self.img_h), focal_scale=2.0)

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

    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_imgseq=None, **kwargs):
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
        f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        x = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, J*32) -> (B, L, C)

        # Condition
        f_to_add = []
        f_to_add.append(self.cliffcam_embedder(f_cliffcam))
        if hasattr(self, "cam_angvel_embedder"):
            f_to_add.append(self.cam_angvel_embedder(f_cam_angvel))
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
        for block in self.blocks:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        # MV2D
        # mv2d = self.mv2d_head(x)
        # mv2d = mv2d.view(B, L, self.num_views, 17, 2)
        
        # Output
        sample = self.final_layer(x)  # (B, L, C)
        if self.avgbeta:
            betas = (sample[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]  # (B, C)
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)

        # Output (extra)
        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam[..., 0], 0.25)  # min_clamp s to 0.25 (prevent negative prediction)

        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x)  # (B, L, C')

        output = {
            "pred_context": x,
            "pred_x": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
            # "mv2d": mv2d,
        }
        
        endecoder = self.endecoder[0]
        decode_dict = endecoder.decode(output["pred_x"])  # (B, L, C) -> dict
        
        grot_mat = angle_axis_to_rotation_matrix(decode_dict["global_orient"])
        base_rot = angle_axis_to_rotation_matrix(torch.tensor([[0., 0., -0.5 * np.pi]]).to(grot_mat)) @ angle_axis_to_rotation_matrix(torch.tensor([[-0.5 * np.pi, 0., 0.]]).to(grot_mat))
        grot_world = base_rot @ grot_mat
        grot_world = rotation_matrix_to_angle_axis(grot_world)
        smpl_gv_dict = {
            "body_pose": decode_dict["body_pose"],  # (B, L, 63)
            "betas": decode_dict["betas"],  # (B, L, 10)
            "global_orient": grot_world,  # (B, L, 3)
            "transl": torch.zeros_like(grot_world),  # (B, L, 3)
        }
        _, j3d = endecoder.smplx_model(**smpl_gv_dict)
        
        cam_dict = self.generate_cam(device=j3d.device)
        kp2d = self.project_keypoints(j3d, cam_dict)
        # cam_dict2 = self.generate_cam2(device=j3d.device)
        # kp2d = self.project_keypoints(j3d, cam_dict2)
        bbx_xys = get_bbx_xys(kp2d, do_augment=False)
        kp2d_norm = normalize_kp2d(torch.cat([kp2d, torch.ones_like(kp2d[..., :1])], dim=-1), bbx_xys)[..., :2]
        output['mv2d'] = kp2d_norm
        output['mv2d_proj'] = kp2d_norm.clone()
        
        output['decode_dict'] = decode_dict
        return output
    
    def get_naive_intrinsics(self, res, focal_scale=1.0):
        # Assume 45 degree FOV
        img_w, img_h = res
        self.focal_length = (img_w * img_w + img_h * img_h) ** 0.5 * focal_scale
        self.cam_intrinsics = torch.eye(3).repeat(1, 1, 1).float()
        self.cam_intrinsics[:, 0, 0] = self.focal_length
        self.cam_intrinsics[:, 1, 1] = self.focal_length
        self.cam_intrinsics[:, 0, 2] = img_w/2.
        self.cam_intrinsics[:, 1, 2] = img_h/2.

    def generate_cam(self, device):
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

        cam_dict = []
        azimuths = torch.linspace(0, 360, self.num_views + 1)[:self.num_views].to(device)
        elevations = (torch.ones((1, self.num_views)) * 0.0).to(device)
        radius = (torch.ones((1, self.num_views)) * 8.0).to(device)
        azimuths = azimuths.unsqueeze(0).expand(elevations.shape[0], -1)
        eyes = spherical_to_cartesian(radius, azimuths, elevations)
        
        eyes_flat = eyes.view(-1, 3)
        at = torch.zeros((eyes_flat.shape[0], 3), device=device)
        up = torch.tensor([0., 0., 1.], device=device)[None, :].expand(eyes_flat.shape[0], -1)
        c2w = lookat_correct(eyes_flat, at, up).reshape(eyes.shape[0], self.num_views, 4, 4)
        w2c = torch.inverse(c2w)
        intrinsics = self.cam_intrinsics.to(device).unsqueeze(0).repeat(eyes.shape[0], self.num_views, 1, 1)
        P = torch.matmul(intrinsics, w2c[..., :3, :])
        
        cam_dict = {
            'c2w': c2w,
            'w2c': w2c,
            'intrinsics': intrinsics,
            'P': P,
            'azimuths': azimuths,
            'elevations': elevations,
            'radius': radius,
        }
        
        return cam_dict
    
    def project_keypoints(self, kpt3d, cam_dict):
        kpt3d_pad = torch.cat((kpt3d, torch.ones_like(kpt3d[..., :1])), dim=-1)
        local_kpt2d_new = (cam_dict['P'][:, None] @ kpt3d_pad[:, :, None].transpose(-1, -2)).transpose(-1, -2)
        local_kpt2d_new = local_kpt2d_new[..., :2] / local_kpt2d_new[..., 2:]
        local_kpt2d_new[..., 1] = self.img_h - local_kpt2d_new[..., 1]
        return local_kpt2d_new
    
    def generate_eyes2(self):
        azimuths = np.linspace(0, 360, self.num_views, endpoint=False)
        elevations = np.ones(self.num_views) * 0
        radius = np.ones(self.num_views) * 8
        eyes = np.stack([spherical_to_cartesian(r, azimuth, elevation) for azimuth, elevation, r in zip(azimuths, elevations, radius)], axis=0)
        return eyes, azimuths, elevations, radius
    
    def generate_cam2(self, device):
        cam_dict = []
        eyes, azimuths, elevations, radius = self.generate_eyes2()
        for i in range(self.num_views):
            eye = eyes[i]
            at = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            c2w = torch.tensor(lookat_correct(eye, at, up)).float().to(device)
            w2c = torch.inverse(c2w)
            P = torch.matmul(self.cam_intrinsics.to(device), w2c[:3, :])
            cam_dict.append({
                'c2w': c2w,
                'w2c': w2c,
                'intrinsics': self.cam_intrinsics,
                'P': P.squeeze(0),
            })
        cam_dict = {k: torch.stack([x[k] for x in cam_dict]) for k in cam_dict[0]}
        cam_dict['azimuths'] = torch.tensor(azimuths).float()
        cam_dict['elevations'] = torch.tensor(elevations).float()
        cam_dict['radius'] = torch.tensor(radius).float()
        cam_dict['P'] = cam_dict['P'].unsqueeze(0)
        return cam_dict


# Add to MainStore
group_name = "network/mv2d"
MainStore.store(
    name="relative_transformer",
    node=builds(NetworkEncoderRoPE, populate_full_signature=True),
    group=group_name,
)
