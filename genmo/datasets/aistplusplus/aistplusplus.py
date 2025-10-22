import os
from pathlib import Path

import numpy as np
import torch
from moviepy.editor import AudioFileClip
from torch.utils import data

from genmo.utils.geo_transform import (
    compute_cam_angvel,
    compute_cam_tvel,
    get_bbx_xys_from_xyxy,
    normalize_T_w2c,
)
from genmo.utils.net_utils import (
    get_valid_mask,
    repeat_to_max_len,
    repeat_to_max_len_dict,
)
from genmo.utils.pylogger import Log
from third_party.GVHMR.hmr4d.utils.geo.hmr_global import get_c_rootparam, get_R_c2gv
from third_party.GVHMR.hmr4d.utils.smplx_utils import make_smplx


class AISTPlusPlusSmplDataset(data.Dataset):
    def __init__(
        self,
        root="inputs/AIST++",
        split="train",
        motion_frames=120,
        lazy_load=False,
        eval_gen_only=True,
        feat_version="v1",
    ):
        super().__init__()
        # Path
        self.root = Path(root)

        # Setting
        self.motion_frames = motion_frames
        self.lazy_load = lazy_load
        self.split = split
        self.eval_gen_only = eval_gen_only
        self.feat_version = feat_version
        self._load_dataset()
        self._get_idx2meta()

    def _load_dataset(self):
        # smplpose
        tic = Log.time()
        fn = self.root / "annot_aist_30fps.pt"
        self.smpl_model = make_smplx("supermotion")
        Log.info(f"[AIST++ {self.feat_version}] Loading from {fn} ...")
        self.motion_files = torch.load(fn)
        self.split_set = torch.load(self.root / f"{self.split}.pt")
        # Dict of {
        #          "smpl_params_glob": {'body_pose', 'global_orient', 'transl', 'betas'}, FxC
        #          "cam_Rt": tensor(F, 3),
        #          "cam_K": tensor(1, 10),
        #         }
        self.seqs = list(self.motion_files.keys())
        Log.info(
            f"[AIST++ {self.feat_version}] {len(self.seqs)} sequences. Elapsed: {Log.time() - tic:.2f}s"
        )

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []
        for vid in self.motion_files:
            if vid not in self.split_set:
                continue
            seq_length = self.motion_files[vid]["bbox_xyxy"].shape[0]
            seq_lengths.append(seq_length)
            self.idx2meta.extend([vid])
        hours = sum(seq_lengths) / 30 / 3600
        Log.info(
            f"[AIST++] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples."
        )

    def __len__(self):
        return len(self.idx2meta)

    def _load_data(self, idx):
        sampled_motion = {}
        vid = self.idx2meta[idx]
        motion = self.motion_files[vid]

        music_feat = torch.load(
            self.root / f"musicfeat_{self.feat_version}/{vid}_musicfeat_fps30.pt"
        )
        seq_length = min(music_feat.shape[0], motion["bbox_xyxy"].shape[0])
        sampled_motion["vid"] = vid

        # Random select a subset
        target_length = self.motion_frames
        if target_length > seq_length:  # this should not happen
            start = 0
            length = seq_length
            Log.info(
                f"[AIST++] ({idx}) target length < sequence length: {target_length} <= {seq_length}"
            )
        elif self.split in ["train", "minitrain"]:
            start = np.random.randint(0, seq_length - target_length)
            length = target_length
        else:
            start = 0
            length = seq_length
        end = start + length
        sampled_motion["length"] = length
        sampled_motion["start_end"] = (start, end)

        music_beats = torch.load(self.root / f"musicfeat/{vid}_musicfeat_fps30.pt")[
            ..., 53
        ]
        sampled_motion["music_beats"] = torch.from_numpy(music_beats[start:end]).float()

        # Select motion subset
        # body_pose, global_orient, transl, betas
        sampled_motion["smpl_params_glob"] = {
            "body_pose": torch.from_numpy(
                motion["smpl_pose_global"][start:end][:, 3:66]
            ).float(),
            "betas": torch.zeros((length, 10)).float(),
            "global_orient": torch.from_numpy(
                motion["smpl_pose_global"][start:end][:, :3]
            ).float(),
            "transl": torch.from_numpy(motion["smpl_trans_global"][start:end]).float(),
        }

        sampled_motion["smpl_params_cam"] = {
            "body_pose": torch.from_numpy(
                motion["smpl_pose"][start:end][:, 3:66]
            ).float(),
            "betas": torch.zeros((length, 10)).float(),
            "global_orient": torch.from_numpy(
                motion["smpl_pose"][start:end][:, :3]
            ).float(),
            "transl": torch.from_numpy(motion["smpl_trans"][start:end]).float(),
        }

        # Image as feature
        # vitfeat = torch.load(self.root / f"vitfeat/{vid}_vitfeat.pt")
        # vitfeat = vitfeat[::2]  # 60fps -> 30fps
        # sampled_motion["f_imgseq"] = vitfeat[start:end].float()  # (L, 1024)
        sampled_motion["f_imgseq"] = torch.zeros((length, 1024)).float()

        bbx_xys = get_bbx_xys_from_xyxy(
            torch.from_numpy(motion["bbox_xyxy"][start:end]), base_enlarge=1.2
        )
        sampled_motion["bbx_xys"] = bbx_xys.float()
        sampled_motion["K_fullimg"] = motion["intrinsics"]
        # sampled_motion["kp2d"] = self.vitpose[vid][start:end].float()  # (L, 17, 3)
        # vitpose = torch.load(self.root / f"vitpose/{vid}_vitpose.pt")[0][::2]  # 60fps -> 30fps
        # sampled_motion["kp2d"] = vitpose[start:end].float()  # (L, 17, 3)
        sampled_motion["kp2d"] = torch.zeros((length, 17, 3)).float()

        # Camera
        sampled_motion["T_w2c"] = motion["T_w2c"]  # (4, 4)

        sampled_motion["music_embed"] = torch.from_numpy(
            music_feat[start:end]
        ).float()  # (L, 1024)

        # load audio
        if self.split in ["train", "minitrain"]:
            music_fps = 30
            music_array = torch.zeros((length, 1024)).float()
        else:
            music_array = torch.load(os.path.join(self.root, f"audio_array/{vid}.pt"))
            music_array = torch.from_numpy(music_array).float()
            music = AudioFileClip(os.path.join(self.root, f"audio/{vid}.mp3"))
            music_fps = music.fps
            start_audio = int(start * music_fps / 30)
            end_audio = int(end * music_fps / 30)
            music_array = music_array[start_audio:end_audio]
        # music_fps = 30

        sampled_motion["music_array"] = music_array
        sampled_motion["music_fps"] = music_fps
        sampled_motion["height"] = motion["height"]
        sampled_motion["width"] = motion["width"]
        return sampled_motion

    def _process_data(self, data, idx):
        length = data["length"]

        # SMPL params in world
        smpl_params_w = data["smpl_params_glob"]  # in az
        old_smpl_params_c = data["smpl_params_cam"]
        music_fps = data["music_fps"]

        # SMPL params in cam
        T_w2c = data["T_w2c"]  # (4, 4)
        offset = self.smpl_model.get_skeleton(smpl_params_w["betas"][0])[0]  # (3)
        global_orient_c, transl_c = get_c_rootparam(
            smpl_params_w["global_orient"],
            smpl_params_w["transl"],
            T_w2c,
            offset,
        )
        assert (
            old_smpl_params_c["global_orient"] - global_orient_c
        ).abs().max() < 1e-4, (
            (old_smpl_params_c["global_orient"] - global_orient_c).abs().max(),
            data["vid"],
            data["start_end"],
        )
        assert (old_smpl_params_c["transl"] - transl_c).abs().max() < 1e-4, (
            (old_smpl_params_c["transl"] - transl_c).abs().max(),
            data["vid"],
            data["start_end"],
        )

        smpl_params_c = {
            "body_pose": smpl_params_w["body_pose"].clone(),  # (F, 63)
            "betas": smpl_params_w["betas"].clone(),  # (F, 10)
            "global_orient": global_orient_c,  # (F, 3)
            "transl": transl_c,  # (F, 3)
        }

        # World params
        gravity_vec = torch.tensor([0, 0, -1]).float()  # (3), AIST++ is az
        T_w2c = T_w2c.repeat(length, 1, 1)  # (F, 4, 4)
        R_c2gv = get_R_c2gv(
            T_w2c[..., :3, :3], axis_gravity_in_w=gravity_vec
        )  # (F, 3, 3)

        # Image
        bbx_xys = data["bbx_xys"]  # (F, 3)
        K_fullimg = data["K_fullimg"].repeat(length, 1, 1)  # (F, 3, 3)
        f_imgseq = data["f_imgseq"]  # (F, 1024)

        normed_T_w2c = normalize_T_w2c(T_w2c)

        cam_angvel = compute_cam_angvel(
            normed_T_w2c[:, :3, :3]
        )  # (F, 6)  slightly different from WHAM
        cam_tvel = compute_cam_tvel(normed_T_w2c[:, :3, 3])  # (F, 3)
        assert cam_tvel.sum() == 0, cam_tvel

        # Returns: do not forget to make it batchable! (last lines)
        max_len = self.motion_frames if self.split in ["train", "minitrain"] else length
        return_data = {
            "meta": {
                "data_name": "aist++",
                "dataset_id": "aist++",
                "idx": idx,
                "vid": data["vid"],
                "height": data["height"],
                "width": data["width"],
                "eval_gen_only": self.eval_gen_only,
            },
            "length": length,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w,
            "R_c2gv": R_c2gv,  # (F, 3, 3)
            "gravity_vec": gravity_vec,  # (3)
            "bbx_xys": bbx_xys,  # (F, 3)
            "K_fullimg": K_fullimg,  # (F, 3, 3)
            "f_imgseq": f_imgseq,  # (F, D)
            "kp2d": data["kp2d"],  # (F, 17, 3)
            "cam_angvel": cam_angvel,  # (F, 6)
            "cam_tvel": cam_tvel,  # (F, 3)
            "noisy_cam_tvel": cam_tvel,  # (F, 3)
            "T_w2c": normed_T_w2c,  # (F, 4, 4)
            "music_embed": data["music_embed"],  # (F, C)
            "music_array": data["music_array"],  # (F / 30 * audio_fps, C)
            "music_fps": music_fps,
            "music_beats": data["music_beats"],  # (F,)
            # "has_music": True,
            "mask": {
                "valid": get_valid_mask(length, length),
                "has_img_mask": get_valid_mask(length, 0),
                "has_2d_mask": get_valid_mask(length, length),
                "has_cam_mask": get_valid_mask(length, length),
                "has_audio_mask": get_valid_mask(length, 0),
                "has_music_mask": get_valid_mask(length, length),
                "2d_only": False,
                "vitpose": False,
                "bbx_xys": False,
                "f_imgseq": False,
                "spv_incam_only": False,
                "invalid_contact": True,
            },
        }

        # Batchable
        if self.split in ["train", "minitrain"]:
            return_data["smpl_params_c"] = repeat_to_max_len_dict(
                return_data["smpl_params_c"], max_len
            )
            return_data["smpl_params_w"] = repeat_to_max_len_dict(
                return_data["smpl_params_w"], max_len
            )
            return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
            return_data["bbx_xys"] = repeat_to_max_len(return_data["bbx_xys"], max_len)
            return_data["K_fullimg"] = repeat_to_max_len(
                return_data["K_fullimg"], max_len
            )
            return_data["f_imgseq"] = repeat_to_max_len(
                return_data["f_imgseq"], max_len
            )
            return_data["music_embed"] = repeat_to_max_len(
                return_data["music_embed"], max_len
            )
            return_data["music_array"] = repeat_to_max_len(
                return_data["music_array"], int(max_len / 30 * music_fps)
            )
            return_data["music_beats"] = repeat_to_max_len(
                return_data["music_beats"], max_len
            )

            return_data["kp2d"] = repeat_to_max_len(return_data["kp2d"], max_len)
            return_data["cam_angvel"] = repeat_to_max_len(
                return_data["cam_angvel"], max_len
            )
            return_data["cam_tvel"] = repeat_to_max_len(
                return_data["cam_tvel"], max_len
            )
            return_data["noisy_cam_tvel"] = repeat_to_max_len(
                return_data["noisy_cam_tvel"], max_len
            )
            return_data["T_w2c"] = repeat_to_max_len(return_data["T_w2c"], max_len)
        return return_data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data
