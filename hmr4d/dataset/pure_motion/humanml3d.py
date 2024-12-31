from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hmr4d.configs import MainStore, builds
from hmr4d.utils.geo.hmr_global import (
    get_c_rootparam,
    get_R_c2gv,
    get_tgtcoord_rootparam,
)
from hmr4d.utils.geo_transform import (
    apply_T_on_points,
    compute_cam_angvel,
    compute_cam_tvel,
    cvt_p2d_from_i_to_c,
    project_p2d,
)
from hmr4d.utils.net_utils import (
    get_valid_mask,
    repeat_to_max_len,
    repeat_to_max_len_dict,
)
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.wis3d_utils import (
    add_motion_as_lines,
    convert_motion_as_line_mesh,
    make_wis3d,
)
from hmr4d.utils.geo_transform import normalize_T_w2c, as_identity

from .cam_traj_utils import CameraAugmentorV11
from hmr4d.utils.geo.hmr_cam import create_camera_sensor
from .base_dataset import BaseDataset
from .utils import *


class Humanml3dDataset(BaseDataset):
    def __init__(
        self,
        motion_frames=120,
        l_factor=1.5,  # speed augmentation
        skip_moyo=True,  # not contained in the ICCV19 released version
        cam_augmentation="v11",
        random1024=False,  # DEBUG
        limit_size=None,
    ):
        self.root = Path("inputs/HumanML3D_SMPL/hmr4d_support")
        self.motion_frames = motion_frames
        self.l_factor = l_factor
        self.random1024 = random1024
        self.skip_moyo = skip_moyo
        self.dataset_name = "HumanML3D"
        self.smplx_neutral = make_smplx(type="supermotion_smpl24")
        self.smplx_male = make_smplx(type="supermotion_smpl24_male")
        self.smplx_female = make_smplx(type="supermotion_smpl24_female")
        self.smplx_dict = {
            "male": self.smplx_male,
            "female": self.smplx_female,
            "neutral": self.smplx_neutral,
        }

        super().__init__(cam_augmentation, limit_size)

    def _load_dataset(self):
        filename = self.root / "humanml3d_smplhpose.pth"
        Log.info(f"[{self.dataset_name}] Loading from {filename} ...")
        tic = Log.time()
        if self.random1024:  # Debug, faster loading
            try:
                Log.info(f"[{self.dataset_name}] Loading 1024 samples for debugging ...")
                self.motion_files = torch.load(self.root / "smplxpose_v2_random1024.pth")
            except:
                Log.info(f"[{self.dataset_name}] Not found! Saving 1024 samples for debugging ...")
                self.motion_files = torch.load(filename)
                keys = list(self.motion_files.keys())
                keys = np.random.choice(keys, 1024, replace=False)
                self.motion_files = {k: self.motion_files[k] for k in keys}
                torch.save(self.motion_files, self.root / "humanml3d_smplhpose_random1024.pth")
        else:
            self.motion_files = torch.load(filename)
        self.seqs = list(self.motion_files.keys())
        Log.info(f"[{self.dataset_name}] {len(self.seqs)} sequences. Elapsed: {Log.time() - tic:.2f}s")

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []

        # Skip too-long idle-prefix
        motion_start_id = {}
        for vid in self.motion_files:
            seq_length = self.motion_files[vid]["pose"].shape[0]
            start_id = motion_start_id[vid] if vid in motion_start_id else 0
            seq_length = seq_length - start_id
            if seq_length < 25:  # Skip clips that are too short
                continue
            num_samples = max(seq_length // self.motion_frames, 1)
            seq_lengths.append(seq_length)
            self.idx2meta.extend([(vid, start_id)] * num_samples)
            assert start_id == 0, f"start_id is not 0 for {vid}"
        hours = sum(seq_lengths) / 30 / 3600
        Log.info(f"[{self.dataset_name}] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples.")

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

        # AZ -> AY
        data_interpolated["global_orient"], data_interpolated["transl"], _ = get_tgtcoord_rootparam(
            data_interpolated["global_orient"],
            data_interpolated["transl"],
            tsf="az->ay",
        )

        data_interpolated["data_name"] = "humanml3d"
        data_interpolated["gender"] = raw_data["gender"]
        return data_interpolated

    def _process_data(self, data, idx):
        """
        Args:
            data: dict {
                "body_pose": (F, 63),
                "betas": (F, 10),
                "global_orient": (F, 3),  in the AY coordinates
                "transl": (F, 3),  in the AY coordinates
            }
        """
        data_name = data["data_name"]
        length = data["body_pose"].shape[0]
        # Augmentation: betas, SMPL (gravity-axis)
        gender = str(data['gender'])
        body_pose = data["body_pose"]
        betas = augment_betas(data["betas"], std=0.1)
        global_orient_w, transl_w = rotate_around_axis(data["global_orient"], data["transl"], axis="y")
        del data

        # SMPL_params in world
        smpl_params_w = {
            "body_pose": body_pose,  # (F, 63)
            "betas": betas,  # (F, 10)
            "global_orient": global_orient_w,  # (F, 3)
            "transl": transl_w,  # (F, 3)
        }

        # Camera trajectory augmentation
        if self.cam_augmentation == "v11":
            # interleave repeat to original length (faster)
            N = 10
            smpl_layer = self.smplx_dict[gender]
            w_j3d = smpl_layer(
                smpl_params_w["body_pose"][::N],
                smpl_params_w["betas"][::N],
                smpl_params_w["global_orient"][::N],
                None,
            )
            w_j3d = w_j3d.repeat_interleave(N, dim=0) + smpl_params_w["transl"][:, None]  # (F, 24, 3)

            if False:
                wis3d = make_wis3d(name="debug_amass")
                add_motion_as_lines(w_j3d, wis3d, "w_j3d")

            width, height, K_fullimg = create_camera_sensor(1000, 1000, 43.3)  # WHAM
            focal_length = K_fullimg[0, 0]
            wham_cam_augmentor = CameraAugmentorV11()
            T_w2c = wham_cam_augmentor(w_j3d, length)  # (F, 4, 4)

        else:
            raise NotImplementedError

        T_c2w = T_w2c.inverse()
        noisy_T_c2w = T_c2w.clone()
        R_c2w = as_identity(T_c2w[:, :3, :3])
        t_c2w = T_c2w[:, :3, 3]
        rand_scale = min(max(0.1, torch.randn(1) + 3), 10)
        noisy_t_c2w = t_c2w / rand_scale
        noisy_T_c2w[:, :3, 3] = noisy_t_c2w
        noisy_T_w2c = noisy_T_c2w.inverse()
        del noisy_T_c2w

        normed_noisy_T_w2c = normalize_T_w2c(noisy_T_w2c)
        normed_T_w2c = normalize_T_w2c(T_w2c)
        normed_R_w2c = as_identity(normed_T_w2c[:, :3, :3])
        normed_t_w2c = normed_T_w2c[:, :3, 3]
        normed_noisy_R_w2c = as_identity(normed_noisy_T_w2c[:, :3, :3])
        normed_noisy_t_w2c = normed_noisy_T_w2c[:, :3, 3]
        del noisy_T_w2c

        # SMPL params in cam
        offset = self.smplx.get_skeleton(smpl_params_w["betas"][0])[0]  # (3)
        global_orient_c, transl_c = get_c_rootparam(
            smpl_params_w["global_orient"],
            smpl_params_w["transl"],
            T_w2c,
            offset,
        )
        smpl_params_c = {
            "body_pose": smpl_params_w["body_pose"].clone(),  # (F, 63)
            "betas": smpl_params_w["betas"].clone(),  # (F, 10)
            "global_orient": global_orient_c,  # (F, 3)
            "transl": transl_c,  # (F, 3)
        }

        # World params
        gravity_vec = torch.tensor([0, -1, 0], dtype=torch.float32)  # (3), BEDLAM is ay
        R_c2gv = get_R_c2gv(T_w2c[:, :3, :3], gravity_vec)  # (F, 3, 3)

        # Image
        K_fullimg = K_fullimg.repeat(length, 1, 1)  # (F, 3, 3)
        cam_angvel = compute_cam_angvel(normed_T_w2c[:, :3, :3])  # (F, 6)
        cam_tvel = compute_cam_tvel(normed_T_w2c[:, :3, 3])  # (F, 3)
        noisy_cam_tvel = compute_cam_tvel(normed_noisy_T_w2c[:, :3, 3])  # (F, 3)

        # Returns: do not forget to make it batchable! (last lines)
        # NOTE: bbx_xys and f_imgseq will be added later
        max_len = length
        return_data = {
            "meta": {"data_name": data_name, "idx": idx, "T_w2c": T_w2c},
            "length": length,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w,
            "R_c2gv": R_c2gv,  # (F, 3, 3)
            "gravity_vec": gravity_vec,  # (3)
            "bbx_xys": torch.zeros((length, 3)),  # (F, 3)  # NOTE: a placeholder
            "K_fullimg": K_fullimg,  # (F, 3, 3)
            "f_imgseq": torch.zeros((length, 1024)),  # (F, D)  # NOTE: a placeholder
            "kp2d": torch.zeros(length, 17, 3),  # (F, 17, 3)
            "cam_angvel": cam_angvel,  # (F, 6)
            "cam_tvel": cam_tvel,  # (F, 3),
            "noisy_cam_tvel": noisy_cam_tvel,  # (F, 3),
            "T_w2c": normed_T_w2c,
            "mask": {
                "valid": get_valid_mask(length, length),
                "vitpose": False,
                "bbx_xys": False,
                "f_imgseq": False,
                "spv_incam_only": False,
            },
        }

        # Batchable
        return_data["smpl_params_c"] = repeat_to_max_len_dict(return_data["smpl_params_c"], max_len)
        return_data["smpl_params_w"] = repeat_to_max_len_dict(return_data["smpl_params_w"], max_len)
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
        return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)
        return_data["cam_tvel"] = repeat_to_max_len(return_data["cam_tvel"], max_len)
        return_data["noisy_cam_tvel"] = repeat_to_max_len(return_data["noisy_cam_tvel"], max_len)
        return_data["T_w2c"] = repeat_to_max_len(return_data["T_w2c"], max_len)
        return return_data


group_name = "train_datasets/pure_motion_humanml3d"
MainStore.store(name="v11", node=builds(Humanml3dDataset, cam_augmentation="v11"), group=group_name)
