# kp2d_dataset.py
import torch
import numpy as np
from motiondiff.utils.geom import perspective_projection, lookat_correct, spherical_to_cartesian
from motiondiff.utils.torch_transform import angle_axis_to_rotation_matrix
from motiondiff.models.mv2d.mv2d_utils import draw_motion_2d, draw_mv_imgs
from motiondiff.data_pipeline.tensors import collate
from motiondiff.callbacks.whamlib.data.utils.normalizer import Normalizer
from motiondiff.data_pipeline.utils.augment import randomly_modify_hands_legs


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



class KP2DDatasetV2(torch.utils.data.Dataset):
    def __init__(self, num_frames, img_w=1024, img_h=1024, num_views=4, cam_radius=8, cam_elevation=10,
                 focal_scale=2, synthetic_view_type='even', normalize_type='image_size', bbox_scale=1.4,
                 normalize_stats_dir=None, cam_aug_cfg={}, use_our_normalization=False, hand_leg_aug=False):
        super().__init__()
        self.num_frames = num_frames
        self.img_w = img_w
        self.img_h = img_h
        self.num_views = num_views
        self.normalize_type = normalize_type
        self.bbox_scale = bbox_scale
        self.cam_radius = cam_radius
        self.cam_elevation = cam_elevation
        self.get_naive_intrinsics((self.img_w, self.img_h), focal_scale)
        self.cam_aug_cfg = cam_aug_cfg
        self.use_our_normalization = use_our_normalization
        self.synthetic_view_type = synthetic_view_type
        self.joint_parents = torch.load('assets/mv2d/smplx_joint_parents.p')
        self.smplx_bbox_joints = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        if normalize_stats_dir is not None:
            self.motion_mean = np.load(f'{normalize_stats_dir}/mean.npy')
            self.motion_std = np.load(f'{normalize_stats_dir}/std.npy')
            self.motion_mean = np.tile(self.motion_mean, (num_views,))
            self.motion_std = np.tile(self.motion_std, (num_views,))
        else:
            self.motion_mean = None
            self.motion_std = None
        self.normalize_motion = self.motion_mean is not None
        self.keypoints_normalizer = Normalizer(None)
        self.video_augmentor = None
        self.hand_leg_aug = hand_leg_aug
        
    def get_naive_intrinsics(self, res, focal_scale=1.0):
        # Assume 45 degree FOV
        img_w, img_h = res
        self.focal_length = (img_w * img_w + img_h * img_h) ** 0.5 * focal_scale
        self.cam_intrinsics = torch.eye(3).repeat(1, 1, 1).float()
        self.cam_intrinsics[:, 0, 0] = self.focal_length
        self.cam_intrinsics[:, 1, 1] = self.focal_length
        self.cam_intrinsics[:, 0, 2] = img_w/2.
        self.cam_intrinsics[:, 1, 2] = img_h/2.

    def generate_eyes(self, target):
        rng = target['rng']
        azimuths = np.linspace(0, 360, self.num_views, endpoint=False)
        if 'elevation_std' in self.cam_aug_cfg:
            if rng is not None and 'elevation' in rng:
                rng_val = rng['elevation']
            else:
                rng_val = np.random.randn(1)
            elevations = rng_val * self.cam_aug_cfg['elevation_std'] + self.cam_aug_cfg['elevation_mean']
            elevations = np.repeat(elevations, self.num_views)
        else:
            elevations = np.ones(self.num_views) * self.cam_elevation
        if 'radius_min' in self.cam_aug_cfg:
            if rng is not None and 'radius' in rng:
                rng_val = rng['radius']
            else:
                rng_val = np.random.rand(1)
            radius = rng_val * (self.cam_aug_cfg['radius_max'] - self.cam_aug_cfg['radius_min']) + self.cam_aug_cfg['radius_min']
            radius = np.repeat(radius, self.num_views)
        else:
            radius = np.ones(self.num_views) * self.cam_radius
        eyes = np.stack([spherical_to_cartesian(r, azimuth, elevation) for azimuth, elevation, r in zip(azimuths, elevations, radius)], axis=0)
        return eyes, azimuths, elevations, radius
    
    def generate_cam(self, target):
        cam_dict = []
        eyes, azimuths, elevations, radius = self.generate_eyes(target)
        for i in range(self.num_views):
            eye = eyes[i]
            at = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            c2w = torch.tensor(lookat_correct(eye, at, up)).float()
            w2c = torch.inverse(c2w)
            P = torch.matmul(self.cam_intrinsics, w2c[:3, :])
            cam_dict.append({
                'c2w': c2w,
                'w2c': w2c,
                'intrinsics': self.cam_intrinsics,
                'P': P,
            })
        cam_dict = {k: torch.stack([x[k] for x in cam_dict]) for k in cam_dict[0]}
        cam_dict['azimuths'] = torch.tensor(azimuths).float()
        cam_dict['elevations'] = torch.tensor(elevations).float()
        cam_dict['radius'] = torch.tensor(radius).float()
        return cam_dict
    
    def project_keypoints(self, kpt3d, cam_dict):
        kpt3d_pad = torch.cat((kpt3d, torch.ones_like(kpt3d[:, :, :1])), dim=-1)
        local_kpt2d_new = (cam_dict['P'].transpose(0, 1) @ kpt3d_pad[:, None].transpose(-1, -2)).transpose(-1, -2)
        local_kpt2d_new = local_kpt2d_new[..., :2] / local_kpt2d_new[..., 2:]
        local_kpt2d_new[..., 1] = self.img_h - local_kpt2d_new[..., 1]
        return local_kpt2d_new
    
    def normalize_ours(self, local_kpt2d, center=None, normalize_size=None):
        if self.normalize_type == 'image_size':
            motion_2d = (local_kpt2d / torch.tensor([self.img_w, self.img_h]).float() - 0.5) * 2
            center = normalize_size = None
        elif self.normalize_type in {'bbox_frame', 'bbox_seq'}:
            # local_kpt2d: [bs, num_views, njoints, 2]
            if center is not None:
                motion_2d = (local_kpt2d - center[:, :, None]) / normalize_size * 2
            else:
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
            # draw_motion_2d(motion_2d_vis, 'out/vis/test_amass1.mp4', self.joint_parents, self.img_w, self.img_h, fps=30)
        else:
            raise ValueError
        return motion_2d, center, normalize_size
    
    def get_input(self, target):
        gt_kp3d = target['kp3d']
        
        inpt_kp3d = gt_kp3d.clone()
        if self.video_augmentor is not None:
            inpt_kp3d = self.video_augmentor(inpt_kp3d)
        if self.hand_leg_aug:
            inpt_kp3d = randomly_modify_hands_legs(inpt_kp3d)
        cam_dict = self.generate_cam(target)
        in_kp2d = self.project_keypoints(inpt_kp3d, cam_dict)
        gt_kp2d = self.project_keypoints(gt_kp3d, cam_dict)
        target['raw_gt_kp2d'] = gt_kp2d.clone()
        
        if self.use_our_normalization:
            gt_kp2d, center, normalize_size = self.normalize_ours(gt_kp2d)
            in_kp2d, _, _ = self.normalize_ours(in_kp2d, center, normalize_size)
        else:
            gt_kp2d, bbox = self.keypoints_normalizer(gt_kp2d.reshape(-1, *gt_kp2d.shape[2:]), target['res'], self.cam_intrinsics, 224, 224, use_multi_view_bbox=True, num_views=self.num_views, do_augment=False)
            in_kp2d, bbox = self.keypoints_normalizer(in_kp2d.reshape(-1, *in_kp2d.shape[2:]), target['res'], self.cam_intrinsics, 224, 224, bbox=bbox, use_multi_view_bbox=True, num_views=self.num_views, do_augment=False)
            in_kp2d = in_kp2d[..., :-3].reshape(-1, self.num_views, 25, 2)
            gt_kp2d = gt_kp2d[..., :-3].reshape(-1, self.num_views, 25, 2)
            bbox = bbox.reshape(-1, self.num_views, 3)
        
        target['obs_kp2d'] = in_kp2d.reshape(in_kp2d.shape[0], -1).numpy()
        target['gt_kp2d'] = gt_kp2d.reshape(gt_kp2d.shape[0], -1).numpy()
        target['cam_dict'] = cam_dict
        
        # motion_2d_vis = (gt_kp2d + 1) * 0.5 * torch.tensor([self.img_w, self.img_w]).float()
        # draw_motion_2d(motion_2d_vis, f'out/vis/{self.dataset_name}_gt.mp4', self.joint_parents, self.img_w, self.img_h, fps=30)
        # motion_2d_vis = (in_kp2d + 1) * 0.5 * torch.tensor([self.img_w, self.img_w]).float()
        # draw_motion_2d(motion_2d_vis, f'out/vis/{self.dataset_name}_in.mp4', self.joint_parents, self.img_w, self.img_h, fps=30)
        return target
    
    def pad_motion(self, target):
        m_length = target['obs_kp2d'].shape[0]
        # for key, val in target.items():
        #     if key not in ['cam_dict']:
        #         print(key, val.shape)
        for key in {'obs_kp2d', 'gt_kp2d', 'raw_gt_kp2d', 'pose', 'betas', 'kp3d', 'f_imgseq'}:
            if key in target and isinstance(target[key], torch.Tensor):
                target[key] = target[key].numpy()
        
        if m_length < self.num_frames:
            for key, val in target.items():
                if key in {'obs_kp2d', 'gt_kp2d', 'raw_gt_kp2d', 'pose', 'betas', 'kp3d', 'f_imgseq'}:
                    target[key] = np.concatenate([val, np.zeros((self.num_frames - val.shape[0],) + val.shape[1:])], axis=0)
        return
