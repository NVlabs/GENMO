import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
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
from einops import rearrange


def aa_to_r6d(x):
    return transforms.matrix_to_rotation_6d(angle_axis_to_rotation_matrix(x))


def r6d_to_aa(x):
    return rotation_matrix_to_angle_axis(transforms.rotation_6d_to_matrix(x))


def interpolate_smpl_params(smpl_params, tgt_len):
    """
    smpl_params['body_pose'] (L, 63)
    tgt_len: L->L'
    """
    betas = smpl_params["betas"]
    body_pose = smpl_params["body_pose"]
    global_orient = smpl_params["global_orient"]  # (L, 3)
    transl = smpl_params["transl"]  # (L, 3)

    # Interpolate
    body_pose = rearrange(aa_to_r6d(body_pose.reshape(-1, 21, 3)), "l j c -> c j l")
    body_pose = F.interpolate(body_pose, tgt_len, mode="linear", align_corners=True)
    body_pose = r6d_to_aa(rearrange(body_pose, "c j l -> l j c")).reshape(-1, 63)

    # although this should be the same as above, we do it for consistency
    betas = rearrange(betas, "l c -> c 1 l")
    betas = F.interpolate(betas, tgt_len, mode="linear", align_corners=True)
    betas = rearrange(betas, "c 1 l -> l c")

    global_orient = rearrange(aa_to_r6d(global_orient.reshape(-1, 1, 3)), "l j c -> c j l")
    global_orient = F.interpolate(global_orient, tgt_len, mode="linear", align_corners=True)
    global_orient = r6d_to_aa(rearrange(global_orient, "c j l -> l j c")).reshape(-1, 3)

    transl = rearrange(transl, "l c -> c 1 l")
    transl = F.interpolate(transl, tgt_len, mode="linear", align_corners=True)
    transl = rearrange(transl, "c 1 l -> l c")

    return {"body_pose": body_pose, "betas": betas, "global_orient": global_orient, "transl": transl}


class AmassDataset(KP2DDatasetV2):
    def __init__(self, num_keypoints=14, num_frames=81, split="train", debug=False, rng=None, img_w=1024, img_h=1024, num_views=4,
                 cam_radius=8, cam_elevation=0, focal_scale=2, synthetic_view_type='even', normalize_type='image_size', bbox_scale=1.4, 
                 use_coco_pelvis=False,normalize_stats_dir=None, sample_beta=False, cam_aug_cfg={}, use_our_normalization=False, size_multiplier=1, skip_moyo=False, l_factor=1.5, hand_leg_aug=False, **kwargs):
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
        self.root = Path("dataset/GVHMR/AMASS/hmr4d_support")
        self.motion_frames = num_frames
        self.l_factor = l_factor
        self.skip_moyo = skip_moyo
        self.dataset_name = "AMASS"
        self._load_dataset()
        self._get_idx2meta()  # -> Set self.idx2meta
        self.total_len = len(self.idx2meta) * size_multiplier

    def _load_dataset(self):
        filename = self.root / "smplxpose_v2.pth"
        self.motion_files = torch.load(filename)
        self.seqs = list(self.motion_files.keys())

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []

        # Skip too-long idle-prefix
        motion_start_id = {}
        for vid in self.motion_files:
            if self.skip_moyo and "moyo_smplxn" in vid:
                continue
            seq_length = self.motion_files[vid]["pose"].shape[0]
            start_id = motion_start_id[vid] if vid in motion_start_id else 0
            seq_length = seq_length - start_id
            if seq_length < 25:  # Skip clips that are too short
                continue
            num_samples = max(seq_length // self.motion_frames, 1)
            seq_lengths.append(seq_length)
            self.idx2meta.extend([(vid, start_id)] * num_samples)
        hours = sum(seq_lengths) / 30 / 3600

    def _load_data(self, idx):
        """
        - Load original data
        - Augmentation: speed-augmentation to L frames
        """
        # Load original data
        mid, start_id = self.idx2meta[idx]
        raw_data = self.motion_files[mid]
        raw_len = raw_data["pose"].shape[0] - start_id
        data = {
            "body_pose": raw_data["pose"][start_id:, 3:],  # (F, 63)
            "betas": raw_data["beta"].repeat(raw_len, 1),  # (10)
            "global_orient": raw_data["pose"][start_id:, :3],  # (F, 3)
            "transl": raw_data["trans"][start_id:],  # (F, 3)
        }

        # Get {tgt_len} frames from data
        # Random select a subset with speed augmentation  [start, end)
        tgt_len = self.motion_frames
        raw_subset_len = np.random.randint(int(tgt_len / self.l_factor), int(tgt_len * self.l_factor))
        if raw_subset_len <= raw_len:
            start = np.random.randint(0, raw_len - raw_subset_len + 1)
            end = start + raw_subset_len
        else:  # interpolation will use all possible frames (results in a slow motion)
            start = 0
            end = raw_len
        data = {k: v[start:end] for k, v in data.items()}

        # Interpolation (vec + r6d)
        data_interpolated = interpolate_smpl_params(data, tgt_len)

        # # AZ -> AY
        # data_interpolated["global_orient"], data_interpolated["transl"], _ = get_tgtcoord_rootparam(
        #     data_interpolated["global_orient"],
        #     data_interpolated["transl"],
        #     tsf="az->ay",
        # )

        data_interpolated["data_name"] = "amass"
        return data_interpolated


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
        length = data["body_pose"].shape[0]
        body_pose = data["body_pose"]
        # betas = augment_betas(data["betas"], std=0.1)
        betas = data["betas"]
        global_orient = data["global_orient"]
        del data

        pose = body_pose
        grot = global_orient
        grot_mat = angle_axis_to_rotation_matrix(grot)
        rmat = self.get_global_rot_augmentation(None)
        # rmat = self.base_rot_mat @ rmat
        grot_world = rmat @ grot_mat
        grot_world = rotation_matrix_to_angle_axis(grot_world)
        pose = torch.cat([grot_world, pose, torch.zeros_like(pose[..., :6])], dim=-1)
        shape = augment_betas(betas, std=0.1)
        target = {
            'rng': None,
            'pose': angle_axis_to_rotation_matrix(pose.view(-1, 24, 3)),
            'betas': shape,
            'res': torch.tensor([self.img_w, self.img_h]).float()
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
            'dataset_name': 'amass_v2'
        }
        aux_data = {
            'kpt3d': target['kp3d'],
            'obs_kpt2d': target['obs_kp2d'],
            'raw_gt_kpt2d': target['raw_gt_kp2d'],
            'smpl_pose': target['pose'],
            'smpl_shape': target['betas'],
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
    np.random.seed(0)
    dataset = AmassDataset(use_our_normalization=False, hand_leg_aug=True)
    print(len(dataset))
    for i in range(10):
        a = dataset[i + 100]
        print(i)
