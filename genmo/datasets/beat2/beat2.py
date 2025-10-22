from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils import data

from genmo.datasets.pure_motion.cam_traj_utils import CameraAugmentorV11
from genmo.utils.geo_transform import (
    compute_cam_angvel,
    compute_cam_tvel,
    normalize_T_w2c,
)
from genmo.utils.net_utils import (
    get_valid_mask,
    repeat_to_max_len,
    repeat_to_max_len_dict,
)
from genmo.utils.pylogger import Log
from third_party.GVHMR.hmr4d.utils.geo.hmr_cam import create_camera_sensor
from third_party.GVHMR.hmr4d.utils.geo.hmr_global import get_c_rootparam, get_R_c2gv
from third_party.GVHMR.hmr4d.utils.smplx_utils import make_smplx
from third_party.GVHMR.hmr4d.utils.wis3d_utils import add_motion_as_lines, make_wis3d


class BEAT2SmplDataset(data.Dataset):
    def __init__(
        self,
        root="inputs/BEAT2",
        split="train",
        motion_frames=120,
        cam_augmentation="static",
        lazy_load=False,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.motion_frames = motion_frames
        self.lazy_load = lazy_load

        self.audio_fps = 18000
        self.video_fps = 30

        self.cam_augmentation = cam_augmentation

        self.smplx = make_smplx("supermotion")

        self.smplx_neutral = make_smplx(type="supermotion_smpl24")
        self.smplx_dict = {
            "male": self.smplx_neutral,
            "female": self.smplx_neutral,
            "neutral": self.smplx_neutral,
        }

        self._load_dataset()
        self._get_idx2meta()

    def _load_dataset(self):
        # smplpose
        tic = Log.time()
        fn = self.root / "all_splits.pth"
        self.smpl_model = make_smplx("supermotion")
        Log.info(f"[BEAT2] Loading from {fn} ...")
        self.motion_files = torch.load(fn)[self.split]
        Log.info(
            f"[BEAT2] {len(self.motion_files)} sequences. Elapsed: {Log.time() - tic:.2f}s"
        )

    def _get_idx2meta(self):
        seq_lengths = []
        self.idx2meta = []
        for item in self.motion_files:
            seq_length = item["length"]
            # vid = item["video_id"]
            seq_lengths.append(seq_length)
            self.idx2meta.append(item)
        hours = sum(seq_lengths) / 30 / 3600
        Log.info(
            f"[BEAT2] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples."
        )

    def __len__(self):
        return len(self.idx2meta)

    def _load_data(self, idx):
        sampled_motion = {}
        item = self.idx2meta[idx]
        vid = item["video_id"]
        subset = item["subset"]
        motion = np.load(self.root / f"{subset}/smplxflame_30/{vid}.npz")
        seq_length = motion["poses"].shape[0]

        audio_array, audio_fps = librosa.load(
            self.root / f"{subset}/wave16k/{vid}.wav", sr=self.audio_fps
        )
        audio_array = audio_array[: seq_length * int(self.audio_fps / self.video_fps)]
        audio_array = torch.from_numpy(audio_array).float()

        # Random select a subset
        target_length = self.motion_frames
        if target_length > seq_length:
            start = 0
            length = seq_length
            Log.info(
                f"[BEAT2] ({idx}) target length < sequence length: {target_length} <= {seq_length}"
            )
        elif self.split in ["train", "minitrain"]:
            start = np.random.randint(0, seq_length - target_length)
            length = target_length
        else:
            start = 0
            length = seq_length
        end = start + length
        sampled_motion["vid"] = vid
        sampled_motion["length"] = length
        sampled_motion["start_end"] = (start, end)

        # Select motion subset
        audio_array = audio_array[
            start * int(self.audio_fps / self.video_fps) : end
            * int(self.audio_fps / self.video_fps)
        ]
        fb_poses = motion["poses"]
        fb_betas = motion["betas"]
        transl = motion["trans"]

        body_pose = fb_poses[start:end, 3:66]
        global_orient = fb_poses[start:end, :3]
        betas = fb_betas[None].repeat(length, axis=0)[:, :10]
        transl = transl[start:end]

        sampled_motion["smpl_params_glob"] = {
            "body_pose": torch.from_numpy(body_pose).float(),
            "betas": torch.from_numpy(betas).float(),
            "global_orient": torch.from_numpy(global_orient).float(),
            "transl": torch.from_numpy(transl).float(),
        }
        sampled_motion["gender"] = motion["gender"]

        sampled_motion["f_imgseq"] = torch.zeros((length, 1024)).float()
        sampled_motion["kp2d"] = torch.zeros((length, 17, 3)).float()

        sampled_motion["audio_array"] = audio_array
        sampled_motion["audio_fps"] = self.audio_fps

        return sampled_motion

    def _process_data(self, data, idx):
        length = data["length"]
        gender = str(data["gender"])

        # SMPL params in world
        smpl_params_w = data["smpl_params_glob"]
        audio_fps = data["audio_fps"]

        if self.cam_augmentation == "v11":
            N = 10
            smpl_layer = self.smplx_dict[gender]
            w_j3d = smpl_layer(
                smpl_params_w["body_pose"][::N],
                smpl_params_w["betas"][::N],
                smpl_params_w["global_orient"][::N],
                None,
            )
            w_j3d = (
                w_j3d.repeat_interleave(N, dim=0)[:length]
                + smpl_params_w["transl"][:, None]
            )  # (F, 24, 3)

            width, height, K_fullimg = create_camera_sensor(1000, 1000, 43.3)  # WHAM
            # focal_length = K_fullimg[0, 0]
            wham_cam_augmentor = CameraAugmentorV11()
            T_w2c = wham_cam_augmentor(w_j3d, length)  # (F, 4, 4)
        elif self.cam_augmentation == "static":
            # interleave repeat to original length (faster)
            N = 10
            smpl_layer = self.smplx_dict[gender]
            w_j3d = smpl_layer(
                smpl_params_w["body_pose"][::N],
                smpl_params_w["betas"][::N],
                smpl_params_w["global_orient"][::N],
                None,
            )
            w_j3d = (
                w_j3d.repeat_interleave(N, dim=0)[:length]
                + smpl_params_w["transl"][:, None]
            )  # (F, 24, 3)

            if False:
                wis3d = make_wis3d(name="debug_amass")
                add_motion_as_lines(w_j3d, wis3d, "w_j3d")

            width, height, K_fullimg = create_camera_sensor(1000, 1000, 43.3)  # WHAM
            # focal_length = K_fullimg[0, 0]
            wham_cam_augmentor = CameraAugmentorV11()
            T_w2c = wham_cam_augmentor(w_j3d, length, camera_type="static")  # (F, 4, 4)
        else:
            raise NotImplementedError

        T_c2w = T_w2c.inverse()
        noisy_T_c2w = T_c2w.clone()
        # R_c2w = as_identity(T_c2w[:, :3, :3])
        t_c2w = T_c2w[:, :3, 3]
        rand_scale = min(max(0.1, torch.randn(1) + 3), 10)
        noisy_t_c2w = t_c2w / rand_scale
        noisy_T_c2w[:, :3, 3] = noisy_t_c2w
        noisy_T_w2c = noisy_T_c2w.inverse()
        del noisy_T_c2w

        normed_noisy_T_w2c = normalize_T_w2c(noisy_T_w2c)
        normed_T_w2c = normalize_T_w2c(T_w2c)
        # normed_R_w2c = as_identity(normed_T_w2c[:, :3, :3])
        # normed_t_w2c = normed_T_w2c[:, :3, 3]
        # normed_noisy_R_w2c = as_identity(normed_noisy_T_w2c[:, :3, :3])
        # normed_noisy_t_w2c = normed_noisy_T_w2c[:, :3, 3]
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
            "meta": {
                "data_name": "beat2",
                "idx": idx,
                "vid": data["vid"],
                "eval_gen_only": True,
            },
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
            "audio_array": data["audio_array"],
            "audio_fps": audio_fps,
            # "has_audio": True,
            "mask": {
                "valid": get_valid_mask(length, length),
                "has_img_mask": get_valid_mask(length, 0),
                "has_2d_mask": get_valid_mask(length, length),
                "has_cam_mask": get_valid_mask(length, length),
                "has_audio_mask": get_valid_mask(length, length),
                "has_music_mask": get_valid_mask(length, 0),
                "2d_only": False,
                "vitpose": False,
                "bbx_xys": False,
                "f_imgseq": False,
                "spv_incam_only": False,
                "invalid_contact": False,
            },
        }

        # Batchable
        return_data["smpl_params_c"] = repeat_to_max_len_dict(
            return_data["smpl_params_c"], max_len
        )
        return_data["smpl_params_w"] = repeat_to_max_len_dict(
            return_data["smpl_params_w"], max_len
        )
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["bbx_xys"] = repeat_to_max_len(return_data["bbx_xys"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
        return_data["f_imgseq"] = repeat_to_max_len(return_data["f_imgseq"], max_len)
        return_data["audio_array"] = repeat_to_max_len(
            return_data["audio_array"], int(max_len * self.audio_fps / self.video_fps)
        )
        return_data["kp2d"] = repeat_to_max_len(return_data["kp2d"], max_len)
        return_data["cam_angvel"] = repeat_to_max_len(
            return_data["cam_angvel"], max_len
        )
        return_data["cam_tvel"] = repeat_to_max_len(return_data["cam_tvel"], max_len)
        return_data["noisy_cam_tvel"] = repeat_to_max_len(
            return_data["noisy_cam_tvel"], max_len
        )
        return_data["T_w2c"] = repeat_to_max_len(return_data["T_w2c"], max_len)
        return return_data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data
