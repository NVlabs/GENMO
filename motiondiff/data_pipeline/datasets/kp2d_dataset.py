# kp2d_dataset.py
import torch
import numpy as np
from motiondiff.utils.geom import perspective_projection, lookat_correct, spherical_to_cartesian
from motiondiff.utils.torch_transform import angle_axis_to_rotation_matrix
from motiondiff.models.mv2d.mv2d_utils import draw_motion_2d, draw_mv_imgs
from motiondiff.data_pipeline.tensors import collate


# an adapter to our collate func
def kp2d_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[1].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'target': 0,
        'text': b[0], #b[0]['caption']
        'lengths': b[2],
        'cam_dict': b[3],
        'aux_data': b[4],
        'info': b[5] if len(b) > 5 else {}
    } for b in batch]
    return collate(adapted_batch)



class KP2DDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames, img_w=1024, img_h=1024, num_views=4, cam_radius=8, cam_elevation=10, focal_scale=2, 
                 synthetic_view_type='even', normalize_type='image_size', bbox_scale=1.4, tilt_prob=0.0, tilt_std=15, 
                 normalize_stats_dir=None, joint_parents_file='assets/mv2d/smplx_joint_parents.p',
                 front_view_joints=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]):
        super(KP2DDataset, self).__init__()
        self.num_frames = num_frames
        self.img_w = img_w
        self.img_h = img_h
        self.num_views = num_views
        self.normalize_type = normalize_type
        self.bbox_scale = bbox_scale
        self.cam_radius = cam_radius
        self.cam_elevation = cam_elevation
        self.focal_length = (img_w * img_w + img_h * img_h) ** 0.5 * focal_scale
        self.cam_intrinsics = torch.eye(3).float()
        self.cam_intrinsics[0, 0] = self.focal_length
        self.cam_intrinsics[1, 1] = self.focal_length
        self.cam_intrinsics[0, 2] = img_w / 2.
        self.cam_intrinsics[1, 2] = img_h / 2.
        self.synthetic_view_type = synthetic_view_type
        self.tilt_prob = tilt_prob
        self.tilt_std = tilt_std
        self.joint_parents = torch.load(joint_parents_file)
        self.smplx_bbox_joints = front_view_joints

        if normalize_stats_dir is not None:
            self.motion_mean = np.load(f'{normalize_stats_dir}/mean.npy')
            self.motion_std = np.load(f'{normalize_stats_dir}/std.npy')
            self.motion_mean = np.tile(self.motion_mean, (num_views,))
            self.motion_std = np.tile(self.motion_std, (num_views,))
        else:
            self.motion_mean = None
            self.motion_std = None
        self.normalize_motion = self.motion_mean is not None

    def generate_eyes(self):
        azimuths = np.linspace(0, 360, self.num_views, endpoint=False) + np.random.rand() * 360 / self.num_views
        elevations = np.ones(self.num_views) * self.cam_elevation
        radius = np.ones(self.num_views) * self.cam_radius
        eyes = np.stack([spherical_to_cartesian(r, azimuth, elevation) for azimuth, elevation, r in zip(azimuths, elevations, radius)], axis=0)
        return eyes

    def extract_kpt3d(self, joints_pos):
        kpt3d = joints_pos - joints_pos[:, :1]
        return kpt3d
    
    def tilt_kpt3d(self, kpt3d):
        tilt_angles = torch.tensor(np.random.randn(1, 2) * self.tilt_std).expand(kpt3d.shape[0], 2).float()
        # tilt_angles[:, 0] = 0
        # tilt_angles[:, 1] = 45
        x_rot = angle_axis_to_rotation_matrix(torch.tensor([1., 0., 0.]) * torch.deg2rad(tilt_angles[:, [0]]))
        y_rot = angle_axis_to_rotation_matrix(torch.tensor([0., 1., 0.]) * torch.deg2rad(tilt_angles[:, [1]]))
        all_rot = torch.matmul(x_rot, y_rot)
        kpt3d_tilted = torch.matmul(all_rot, kpt3d.transpose(-1, -2)).transpose(-1, -2)
        return kpt3d_tilted
    
    def normalize(self, motion):
        return (motion - self.motion_mean) / self.motion_std

    def generate_cam_and_2d_motion(self, motion):
        cam_dict = []
        motion = torch.tensor(motion).float()
        kpt3d = self.extract_kpt3d(motion)
        if np.random.rand() < self.tilt_prob:
            kpt3d = self.tilt_kpt3d(kpt3d)
        kpt3d_pad = torch.cat((kpt3d, torch.ones_like(kpt3d[:, :, :1])), dim=-1)
        local_kpt2d_arr = []
        eyes = self.generate_eyes()
        for i in range(self.num_views):
            eye = eyes[i]
            at = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            c2w = torch.tensor(lookat_correct(eye, at, up)).float()
            w2c = torch.inverse(c2w)
            P = torch.matmul(self.cam_intrinsics, w2c[:3, :])
            local_kpt2d = (P @ kpt3d_pad.transpose(-1, -2)).transpose(-1, -2)
            local_kpt2d = local_kpt2d[..., :2] / local_kpt2d[..., 2:]
            local_kpt2d[:, :, 1] = self.img_h - local_kpt2d[:, :, 1]
            cam_dict.append({
                'c2w': c2w,
                'w2c': w2c,
                'intrinsics': self.cam_intrinsics,
                'P': P,
            })
            local_kpt2d_arr.append(local_kpt2d)
        cam_dict = {k: torch.stack([x[k] for x in cam_dict]) for k in cam_dict[0]}
        local_kpt2d = torch.stack(local_kpt2d_arr, dim=1)
        # draw_motion_2d(local_kpt2d, 'out/vis/test_humanml.mp4', self.joint_parents, self.img_w, self.img_h, fps=30)
        if self.normalize_type == 'image_size':
            motion_2d = (local_kpt2d / torch.tensor([self.img_w, self.img_h]).float() - 0.5) * 2
        elif self.normalize_type in {'bbox_frame', 'bbox_seq'}:
            # local_kpt2d: [bs, num_views, njoints, 2]
            front_view = local_kpt2d[:, 0, self.smplx_bbox_joints]     # only use limb joints for bbox size, and only use front views
            bbox_min = front_view.min(dim=1)[0]
            bbox_max = front_view.max(dim=1)[0]
            center = (local_kpt2d[:, :, 1] + local_kpt2d[:, :, 2]) * 0.5
            bbox_size = (bbox_max - bbox_min).max(dim=1)[0]
            if self.normalize_type == 'bbox_seq':
                normalize_size = bbox_size.mean() * self.bbox_scale
                motion_2d = (local_kpt2d - center[:, :, None]) / normalize_size * 2
            elif self.normalize_type == 'bbox_frame':
                normalize_size = bbox_size[:, None, None, None] * self.bbox_scale
                motion_2d = (local_kpt2d - center[:, :, None]) / normalize_size * 2
            
            # size = motion_2d[:, 0, self.smplx_bbox_joints].max(dim=1)[0] - motion_2d[:, 0, self.smplx_bbox_joints].min(dim=1)[0]
            # motion_2d_vis = (motion_2d + 1) * 0.5 * torch.tensor([self.img_w, self.img_w]).float()
            # draw_motion_2d(motion_2d_vis, 'out/vis/bones2d_v2.mp4', self.joint_parents, self.img_w, self.img_h, fps=30, show_joints=self.smplx_valid_joints)
        else:
            raise ValueError
        motion_2d = motion_2d.reshape(motion_2d.shape[0], -1).numpy()
        if self.normalize_motion:
            motion_2d = self.normalize(motion_2d)
        return motion_2d, kpt3d.numpy(), cam_dict
    
    def pad_motion(self, motion, kpt3d):
        if motion.shape[0] >= self.num_frames:
            m_length = self.num_frames
            idx = np.random.randint(0, motion.shape[0] - self.num_frames + 1)
            motion = motion[idx:idx + self.num_frames]
            kpt3d = kpt3d[idx:idx + self.num_frames]
        else:
            m_length = motion.shape[0]
            motion = np.concatenate([motion, np.zeros((self.num_frames - motion.shape[0], motion.shape[1]))], axis=0)
            kpt3d = np.concatenate([kpt3d, np.zeros((self.num_frames - kpt3d.shape[0], kpt3d.shape[1], kpt3d.shape[2]))], axis=0)
        return motion, kpt3d, m_length
