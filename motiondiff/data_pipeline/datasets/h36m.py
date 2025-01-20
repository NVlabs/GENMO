import torch
from torch.utils import data
from pathlib import Path
import numpy as np
import sys
sys.path.append('./')

from motiondiff.data_pipeline.datasets.kp2d_dataset_v2 import KP2DDatasetV2
from motiondiff.models.common.utils.human_models import smpl_x, smpl
from motiondiff.utils.torch_transform import rotation_matrix_to_angle_axis, quaternion_to_angle_axis, quat_mul, angle_axis_to_quaternion, angle_axis_to_rotation_matrix, quaternion_to_rotation_matrix
from motiondiff.callbacks.whamlib.models import build_body_model
from motiondiff.callbacks.whamlib.utils import transforms
from motiondiff.callbacks.whamlib.data.utils.augmentor import VideoAugmentor
from motiondiff.data_pipeline.utils.augment import augment_betas
from motiondiff.models.common.utils.preprocessing import transform_joint_to_other_db


class H36mSmplDataset(KP2DDatasetV2):
    def __init__(self, num_keypoints=14, num_frames=81, split="train", debug=False, rng=None, img_w=1024, img_h=1024, num_views=4,
                 cam_radius=8, cam_elevation=0, focal_scale=2, synthetic_view_type='even', normalize_type='image_size', bbox_scale=1.4, 
                 use_coco_pelvis=False,normalize_stats_dir=None, sample_beta=False, cam_aug_cfg={}, use_our_normalization=False, size_multiplier=1, hand_leg_aug=False, **kwargs):
        
        super().__init__(num_frames, img_w, img_h, num_views, cam_radius, cam_elevation, focal_scale, synthetic_view_type, normalize_type, bbox_scale, normalize_stats_dir, cam_aug_cfg, use_our_normalization, hand_leg_aug)
        self.debug = debug
        self.get_coco_keypoints = num_keypoints == 25
        self.sample_beta = sample_beta
        self.mean = None
        self.std = None
        self.split = split
        self.num_keypoints = num_keypoints
        self.use_coco_pelvis = use_coco_pelvis
        self.wham_regressor = torch.tensor(np.load('data/smpl/J_regressor_wham.npy'))
        self.smpl_joints = smpl.all_joints_name
        self.coco_joints = ['Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
                            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']
        self.smplx_valid_joints = [i for i, j in enumerate(smpl_x.joints_name[:num_keypoints]) if j in self.coco_joints or j in self.smpl_joints]
        self.coco_smplx_ind = [smpl_x.joints_name.index(j) for j in self.coco_joints]   # [24, 22, 23, 20, 21, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 6]
        self.video_augmentor = VideoAugmentor(coco_smplx_ind=self.coco_smplx_ind, num_frames=num_frames)
        self.base_rot = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        self.base_rot_mat = quaternion_to_rotation_matrix(torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
        self.smpl = build_body_model('cpu', num_frames)
        # Path
        self.root = Path('dataset/GVHMR/H36M/hmr4d_support')

        self.motion_frames = num_frames
        self.dataset_name = "H36M"
        self._load_dataset()
        self._get_idx2meta()  # -> Set self.idx2meta
        self.total_len = len(self.idx2meta) * size_multiplier

    def _load_dataset(self):
        # smplpose
        fn = self.root / "smplxpose_v1.pt"
        self.motion_files = torch.load(fn)
        # Dict of {
        #          "smpl_params_glob": {'body_pose', 'global_orient', 'transl', 'betas'}, FxC
        #          "cam_Rt": tensor(F, 3),
        #          "cam_K": tensor(1, 10),
        #         }
        self.seqs = list(self.motion_files.keys())

        # img(as feature)
        # vid -> (features, vid, meta {bbx_xys, K_fullimg})
        fn = self.root / "vitfeat_h36m.pt"
        self.f_img_dicts = torch.load(fn)

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []
        for vid in self.f_img_dicts:
            seq_length = self.f_img_dicts[vid]["bbx_xys"].shape[0]
            num_samples = max(seq_length // self.motion_frames, 1)
            seq_lengths.append(seq_length)
            self.idx2meta.extend([vid] * num_samples)
        hours = sum(seq_lengths) / 25 / 3600

    def _load_data(self, idx):
        sampled_motion = {}
        vid = self.idx2meta[idx]
        motion = self.motion_files[vid]
        seq_length = self.f_img_dicts[vid]["bbx_xys"].shape[0]  # this is a better choice
        sampled_motion["vid"] = vid

        # Random select a subset
        target_length = self.motion_frames
        if target_length > seq_length:  # this should not happen
            start = 0
            length = seq_length
        else:
            start = np.random.randint(0, seq_length - target_length)
            length = target_length
        end = start + length
        sampled_motion["length"] = length
        sampled_motion["start_end"] = (start, end)

        # Select motion subset
        # body_pose, global_orient, transl, betas
        sampled_motion["smpl_params_global"] = {k: v[start:end] for k, v in motion["smpl_params_glob"].items()}

        # Image as feature
        f_img_dict = self.f_img_dicts[vid]
        sampled_motion["f_imgseq"] = f_img_dict["features"][start:end].float()  # (L, 1024)
        sampled_motion["bbx_xys"] = f_img_dict["bbx_xys"][start:end]
        sampled_motion["K_fullimg"] = f_img_dict["K_fullimg"]
        # sampled_motion["kp2d"] = self.vitpose[vid][start:end].float()  # (L, 17, 3)
        sampled_motion["kp2d"] = torch.zeros((end - start), 17, 3)  # (L, 17, 3)

        # Camera
        sampled_motion["T_w2c"] = motion["cam_Rt"]  # (4, 4)

        return sampled_motion
    
    def get_global_rot_augmentation(self, rng):
        """Global coordinate augmentation. Random rotation around y-axis"""
        if rng is not None and 'angle_y' in rng:
            rng_val = rng['angle_y']
        else:
            rng_val = torch.rand(1)
        angle_y = rng_val * 2 * np.pi
        aa = torch.tensor([0.0, 0.0, angle_y]).float().unsqueeze(0)
        rmat = angle_axis_to_rotation_matrix(aa)
        return rmat

    def _process_data(self, data, idx):
        length = data["length"]
        # SMPL params in world
        smpl_params_w = data["smpl_params_global"].copy()  # in az
        # World params
        pose = smpl_params_w["body_pose"]
        grot = smpl_params_w["global_orient"]
        grot_mat = angle_axis_to_rotation_matrix(grot)
        rmat = self.get_global_rot_augmentation(None)
        grot_world = rmat @ grot_mat
        grot_world = rotation_matrix_to_angle_axis(grot_world)
        pose = torch.cat([grot_world, pose, torch.zeros_like(pose[..., :6])], dim=-1)
        shape = smpl_params_w["betas"]
        shape = augment_betas(shape, std=0.1)
        target = {
            'rng': None,
            'pose': angle_axis_to_rotation_matrix(pose.view(-1, 24, 3)),
            'betas': shape,
            'res': torch.tensor([self.img_w, self.img_h]).float(),
            'f_imgseq': data["f_imgseq"].float()
        }
        
        output = self.smpl.get_output(
            body_pose=target['pose'][:, 1:],
            global_orient=target['pose'][:, :1],
            betas=target['betas'],
            pose2rot=False)
        
        smpl_joint_world = output.orig_joints[:, :len(smpl.all_joints_name)].numpy()
        smplx_joint_world = transform_joint_to_other_db(smpl_joint_world.transpose(1, 0, 2), smpl.all_joints_name, smpl_x.joints_name[:25]).transpose(1, 0, 2)
        coco_joints = output.joints[:, :17]
        smplx_joint_world[:, self.coco_smplx_ind] = coco_joints
        smplx_joint_world[:, 0] = (smplx_joint_world[:, 1] + smplx_joint_world[:, 2]) / 2
        smplx_joint_world -= smplx_joint_world[:, :1]
        target['kp3d'] = torch.tensor(smplx_joint_world).float()
        target['pose'] = transforms.matrix_to_rotation_6d(target['pose'])
        
        self.get_input(target)
        self.pad_motion(target)
 
        text = ''
        motion = target['gt_kp2d']
        cam_dict = target['cam_dict']
        m_length = motion.shape[0]
        info = {
            'smpl_valid_joints': self.smplx_valid_joints,
            'dataset_name': 'h36m'
        }
        aux_data = {
            'kpt3d': target['kp3d'],
            'obs_kpt2d': target['obs_kp2d'],
            'raw_gt_kpt2d': target['raw_gt_kp2d'],
            'smpl_pose': target['pose'],
            'smpl_shape': target['betas'],
            'img_features': target["f_imgseq"],
        }   
        return text, motion, m_length, cam_dict, aux_data, info
    
    def __getitem__(self, idx):
        idx = idx % len(self.idx2meta)
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data
    
    def __len__(self):
        return self.total_len


# 3DPW
if __name__ == "__main__":
    dataset = H36mSmplDataset(use_our_normalization=False)
    print(len(dataset))
    for i in range(10):
        a = dataset[i]
        print(i)
