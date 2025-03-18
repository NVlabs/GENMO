from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hmr4d.configs import MainStore, builds
from hmr4d.dataset.pure_motion.base_dataset import BaseDataset
from hmr4d.dataset.pure_motion.cam_traj_utils import CameraAugmentorV11
from hmr4d.dataset.pure_motion.utils import *
from hmr4d.utils.geo.hmr_cam import create_camera_sensor
from hmr4d.utils.geo.hmr_global import (
    get_c_rootparam,
    get_R_c2gv,
    get_tgtcoord_rootparam,
)
from hmr4d.utils.geo_transform import (
    apply_T_on_points,
    as_identity,
    compute_cam_angvel,
    compute_cam_tvel,
    cvt_p2d_from_i_to_c,
    normalize_T_w2c,
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


class Humanml3dDataset(BaseDataset):
    def __init__(
        self,
        mode="default",
        motion_frames=120,
        l_factor=1.5,  # speed augmentation
        skip_moyo=True,  # not contained in the ICCV19 released version
        cam_augmentation="v11",
        split="train",
        random1024=False,  # DEBUG
        limit_size=None,
        no_subsample=False,
        max_text_len=50,
        part_ind=-1,
        num_parts=-1,
        eval_text_only=False,
        use_random_subset=False,
        random_subset_size=32,
        random_subset_seed=7,
        use_multi_text=False,
        num_multi_text=3,
        multi_text_vid=None,
        motion_start_mode="first",
        enable_speed_aug=False,
        eval_seed=None,
        humanoid_data_dir=None,
        humanoid_noise_seq=None,
        humanoid_separate_task_obs=False,
        humanoid_self_obs_dim=358,
        discard_last_frame=False,
    ):
        self.root = Path("inputs/HumanML3D_SMPL/hmr4d_support")
        if split == "train":
            self.text_embed_file = Path(
                "inputs/HumanML3D_SMPL_ye/t5_embeddings_v1_half/all_text_embed.pth"
            )  # TODO: USE THE STANDARD PATH
        else:
            self.text_embed_file = Path(
                "inputs/HumanML3D_SMPL_ye/t5_embeddings_v1_half/test_text_embed.pth"
            )
        if split == "test":
            no_subsample = True
        self.mode = mode
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
        self.max_text_len = max_text_len
        self.split = split
        self.no_subsample = no_subsample
        self.num_parts = num_parts
        self.part_ind = part_ind
        self.eval_text_only = eval_text_only
        self.use_random_subset = use_random_subset
        self.random_subset_seed = random_subset_seed
        self.random_subset_size = random_subset_size
        self.use_multi_text = use_multi_text
        self.num_multi_text = num_multi_text
        self.multi_text_vid = multi_text_vid
        if self.use_multi_text:
            self.num_multi_text = len(self.multi_text_vid)
            self.vid_to_idx = {}
        self.motion_start_mode = motion_start_mode
        self.enable_speed_aug = enable_speed_aug
        self.eval_seed = eval_seed
        self.humanoid_data_dir = humanoid_data_dir
        self.humanoid_noise_seq = humanoid_noise_seq
        self.has_humanoid_data = humanoid_data_dir is not None
        self.humanoid_separate_task_obs = humanoid_separate_task_obs
        self.humanoid_self_obs_dim = humanoid_self_obs_dim
        self.discard_last_frame = discard_last_frame
        super().__init__(cam_augmentation, limit_size)
        if self.use_multi_text:
            for i, (vid, _) in enumerate(self.idx2meta):
                self.vid_to_idx[vid] = i
        return

    def _load_dataset(self):
        filename = self.root / f"humanml3d_smplhpose_{self.split}.pth"
        Log.info(f"[{self.dataset_name}] Loading from {filename} ...")
        tic = Log.time()
        if self.random1024:  # Debug, faster loading
            try:
                Log.info(
                    f"[{self.dataset_name}] Loading 1024 samples for debugging ..."
                )
                self.motion_files = torch.load(
                    self.root / "smplxpose_v2_random1024.pth"
                )
            except:
                Log.info(
                    f"[{self.dataset_name}] Not found! Saving 1024 samples for debugging ..."
                )
                self.motion_files = torch.load(filename)
                keys = list(self.motion_files.keys())
                keys = np.random.choice(keys, 1024, replace=False)
                self.motion_files = {k: self.motion_files[k] for k in keys}
                torch.save(
                    self.motion_files, self.root / "humanml3d_smplhpose_random1024.pth"
                )
        else:
            self.motion_files = torch.load(filename)
        self.text_embed_dict = torch.load(self.text_embed_file)
        self.seqs = list(self.motion_files.keys())
        Log.info(
            f"[{self.dataset_name}] {len(self.seqs)} sequences. Elapsed: {Log.time() - tic:.2f}s"
        )

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []

        # Skip too-long idle-prefix
        motion_start_id = {}
        for vid in self.motion_files:
            # test_data = self.motion_files[vid]['text_data']
            # print(vid, test_data[0]['caption'])
            seq_length = self.motion_files[vid]["pose"].shape[0]
            start_id = motion_start_id[vid] if vid in motion_start_id else 0
            seq_length = seq_length - start_id
            if (
                seq_length < 25 and not self.no_subsample
            ):  # Skip clips that are too short
                continue
            num_samples = max(seq_length // self.motion_frames, 1)
            if self.use_random_subset or self.no_subsample:
                num_samples = 1
            seq_lengths.append(seq_length)
            self.idx2meta.extend([(vid, start_id)] * num_samples)
            assert start_id == 0, f"start_id is not 0 for {vid}"

        if self.num_parts > 0:
            part_size = len(self.idx2meta) // self.num_parts
            start_idx = self.part_ind * part_size
            end_idx = (self.part_ind + 1) * part_size
            self.idx2meta = self.idx2meta[start_idx:end_idx]
            seq_lengths = seq_lengths[start_idx:end_idx]

        if self.use_random_subset:
            self.rng = np.random.RandomState(self.random_subset_seed)
            shuffle_ind = np.arange(len(self.idx2meta))
            self.rng.shuffle(shuffle_ind)
            self.idx2meta = [
                self.idx2meta[i] for i in shuffle_ind[: self.random_subset_size]
            ]
            seq_lengths = [
                seq_lengths[i] for i in shuffle_ind[: self.random_subset_size]
            ]
        else:
            self.rng = np.random
        hours = sum(seq_lengths) / 30 / 3600
        Log.info(
            f"[{self.dataset_name}] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples."
        )

    def _load_data(self, idx):
        """
        - Load original data
        - Augmentation: speed-augmentation to L frames
        """
        # Load original data
        mid, start_id = self.idx2meta[idx]
        raw_data = self.motion_files[mid]
        text_embed_data = self.text_embed_dict[mid].float()

        raw_len = raw_data["pose"].shape[0] - start_id
        offset = 1 if self.discard_last_frame else 0
        raw_len -= offset
        data = {
            "body_pose": raw_data["pose"][start_id : start_id + raw_len, 3:],  # (F, 63)
            "betas": raw_data["beta"].repeat(raw_len, 1),  # (10)
            "global_orient": raw_data["pose"][
                start_id : start_id + raw_len, :3
            ],  # (F, 3)
            "transl": raw_data["trans"][start_id : start_id + raw_len],  # (F, 3)
        }

        if self.has_humanoid_data:
            noise_seq = np.random.choice(self.humanoid_noise_seq)
            humanoid_data = torch.load(
                f"{self.humanoid_data_dir}/{noise_seq}/{mid}.pth"
            )
            raw_humanoid_len = humanoid_data["obs"].shape[0] - start_id
            if raw_humanoid_len >= raw_len:
                data["humanoid_obs"] = humanoid_data["obs"][
                    start_id : start_id + raw_len
                ]
                data["humanoid_clean_action"] = humanoid_data["clean_action"][
                    start_id : start_id + raw_len
                ]
            else:
                data["humanoid_obs"] = torch.cat(
                    [
                        humanoid_data["obs"][start_id:],
                        torch.zeros(
                            raw_len - raw_humanoid_len, *humanoid_data["obs"].shape[1:]
                        ),
                    ],
                    dim=0,
                )
                data["humanoid_clean_action"] = torch.cat(
                    [
                        humanoid_data["clean_action"][start_id:],
                        torch.zeros(
                            raw_len - raw_humanoid_len,
                            *humanoid_data["clean_action"].shape[1:],
                        ),
                    ],
                    dim=0,
                )
        else:
            humanoid_data = None

        # Get {tgt_len} frames from data
        # Random select a subset with speed augmentation  [start, end)
        tgt_len = self.motion_frames
        if self.enable_speed_aug:
            raw_subset_len = self.rng.randint(
                int(tgt_len / self.l_factor), int(tgt_len * self.l_factor)
            )
        else:
            raw_subset_len = tgt_len
        raw_subset_len = min(raw_subset_len, raw_len)
        if raw_subset_len <= raw_len:
            start = self.rng.randint(0, raw_len - raw_subset_len + 1)
            end = start + raw_subset_len
        else:  # interpolation will use all possible frames (results in a slow motion)
            start = 0
            end = raw_len
        data = {k: v[start:end] for k, v in data.items()}

        # Interpolation (vec + r6d)
        if self.enable_speed_aug:
            data_interpolated = interpolate_smpl_params(data, tgt_len)
            valid_length = tgt_len
        else:
            data_interpolated = data
            if raw_subset_len < tgt_len:
                data_interpolated = pad_data(data, tgt_len)
            valid_length = raw_subset_len

        # text_data = self.rng.choice(raw_data["text_data"])
        if self.use_random_subset or self.eval_text_only:
            text_ind = 0
        else:
            text_ind = self.rng.randint(0, len(raw_data["text_data"]))
        text_data = raw_data["text_data"][text_ind]
        text_embed = text_embed_data[text_ind]

        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        # AZ -> AY
        data_interpolated["global_orient"], data_interpolated["transl"], _ = (
            get_tgtcoord_rootparam(
                data_interpolated["global_orient"],
                data_interpolated["transl"],
                tsf="az->ay",
            )
        )

        data_interpolated["data_name"] = "humanml3d"
        data_interpolated["mid"] = mid
        data_interpolated["text_ind"] = text_ind
        data_interpolated["gender"] = raw_data["gender"]
        data_interpolated["caption"] = caption
        data_interpolated["text_embed"] = text_embed
        data_interpolated["valid_length"] = valid_length
        if self.has_humanoid_data:
            raw_subset_humanoid_len = (
                min(start + raw_subset_len, raw_humanoid_len) - start
            )
            data_interpolated["valid_humanoid_length"] = raw_subset_humanoid_len
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
        if self.motion_start_mode == "sample":
            mlength = data["body_pose"].shape[0]
            length = min(self.motion_frames, mlength)
            start = np.random.randint(0, max(mlength - length + 1, 1))
            for k, v in data.items():
                if v in ["body_pose", "betas", "global_orient", "transl"]:
                    data[k] = v[start:]

        data_name = data["data_name"]
        mid = data["mid"]
        text_ind = data["text_ind"]
        length = data["body_pose"].shape[0]
        valid_length = data["valid_length"]
        # Augmentation: betas, SMPL (gravity-axis)
        gender = str(data["gender"])
        body_pose = data["body_pose"]
        betas = augment_betas(data["betas"], std=0.1)
        global_orient_w, transl_w = rotate_around_axis(
            data["global_orient"], data["transl"], axis="y"
        )
        caption = data["caption"]
        text_embed = data["text_embed"]
        if self.has_humanoid_data:
            valid_humanoid_length = data["valid_humanoid_length"]
            humanoid_obs = data["humanoid_obs"]
            humanoid_clean_action = data["humanoid_clean_action"]
        else:
            valid_humanoid_length = 0
            humanoid_obs = None
            humanoid_clean_action = None
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
            w_j3d = (
                w_j3d.repeat_interleave(N, dim=0)[:length]
                + smpl_params_w["transl"][:, None]
            )  # (F, 24, 3)

            if False:
                wis3d = make_wis3d(name="debug_amass")
                add_motion_as_lines(w_j3d, wis3d, "w_j3d")

            width, height, K_fullimg = create_camera_sensor(1000, 1000, 43.3)  # WHAM
            focal_length = K_fullimg[0, 0]
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
            focal_length = K_fullimg[0, 0]
            wham_cam_augmentor = CameraAugmentorV11()
            T_w2c = wham_cam_augmentor(w_j3d, length, camera_type="static")  # (F, 4, 4)

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
            "meta": {
                "data_name": data_name,
                "dataset_id": "humanml3d",
                "idx": idx,
                "T_w2c": T_w2c,
                "eval_gen_only": self.eval_text_only,
                "mid": mid,
                "text_ind": text_ind,
                "mode": self.mode,
                "eval_seed": self.eval_seed,
                "has_humanoid_data": self.has_humanoid_data,
            },
            "length": valid_length,
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
            "caption": caption,
            "text_embed": text_embed,
            "mask": {
                "valid": get_valid_mask(length, valid_length),
                "humanoid": get_valid_mask(length, valid_humanoid_length),
                "vitpose": False,
                "bbx_xys": False,
                "f_imgseq": False,
                "spv_incam_only": False,
                "invalid_contact": False,
            },
        }
        if self.has_humanoid_data:
            return_data["humanoid_obs"] = humanoid_obs
            return_data["humanoid_clean_action"] = humanoid_clean_action
            if self.humanoid_separate_task_obs:
                return_data["humanoid_obs"] = humanoid_obs[
                    :, : self.humanoid_self_obs_dim
                ]
                return_data["humanoid_task_obs"] = humanoid_obs[
                    :, self.humanoid_self_obs_dim :
                ]

        # Batchable
        return_data["smpl_params_c"] = repeat_to_max_len_dict(
            return_data["smpl_params_c"], max_len
        )
        return_data["smpl_params_w"] = repeat_to_max_len_dict(
            return_data["smpl_params_w"], max_len
        )
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
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
        if self.multi_text_vid is not None:
            idx = self.vid_to_idx[self.multi_text_vid[0]]
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        if self.use_multi_text:
            all_data = [data]
            for i in range(1, self.num_multi_text):
                if self.multi_text_vid is not None:
                    new_idx = self.vid_to_idx[self.multi_text_vid[i]]
                else:
                    new_idx = self.rng.randint(0, len(self))
                data_i = self._load_data(new_idx)
                data_i = self._process_data(data_i, new_idx)
                all_data.append(data_i)
            multi_text_data = {
                "vid": [],
                "caption": [],
                "text_ind": [],
                "text_embed": [],
                "window_start": [],
                "window_end": [],
            }
            window_stride = 1 / self.num_multi_text
            for i, data_i in enumerate(all_data):
                multi_text_data["vid"].append(data_i["meta"]["mid"])
                multi_text_data["caption"].append(data_i["caption"])
                multi_text_data["text_ind"].append(data_i["meta"]["text_ind"])
                multi_text_data["text_embed"].append(data_i["text_embed"])
                window_start = i * window_stride
                window_end = (i + 1) * window_stride
                # window_start = i * 100/720
                # window_end = (i + 1) * 100/720
                # if i == len(all_data) - 1:
                #     window_end = 1
                multi_text_data["window_start"].append(window_start)
                multi_text_data["window_end"].append(window_end)
            # multi_text_data["window_end"][0] = 120/600
            # multi_text_data["window_end"][1] = 300/600
            # multi_text_data["window_start"][1] = 120/600
            # multi_text_data["window_start"][2] = 300/600
            multi_text_data["text_embed"] = torch.stack(multi_text_data["text_embed"])
            multi_text_data["window_start"] = torch.tensor(
                multi_text_data["window_start"]
            )
            multi_text_data["window_end"] = torch.tensor(multi_text_data["window_end"])
            print("vid & captions:")
            print(multi_text_data["vid"])
            print(multi_text_data["caption"])
            data["meta"]["multi_text_data"] = multi_text_data
        return data


train_group_name = "train_datasets/pure_motion_humanml3d"
MainStore.store(
    name="v11_train",
    node=builds(Humanml3dDataset, cam_augmentation="v11", split="train"),
    group=train_group_name,
)
MainStore.store(
    name="static_train",
    node=builds(Humanml3dDataset, cam_augmentation="static", split="train"),
    group=train_group_name,
)
test_group_name = "test_datasets/pure_motion_humanml3d"
MainStore.store(
    name="v11_test",
    node=builds(
        Humanml3dDataset,
        cam_augmentation="v11",
        split="test",
        eval_text_only=True,
        no_subsample=True,
    ),
    group=test_group_name,
)
MainStore.store(
    name="static_test",
    node=builds(
        Humanml3dDataset,
        cam_augmentation="static",
        split="test",
        eval_text_only=True,
        no_subsample=True,
    ),
    group=test_group_name,
)
MainStore.store(
    name="humanoid_train_v1",
    node=builds(
        Humanml3dDataset,
        cam_augmentation="static",
        split="train",
        humanoid_data_dir="inputs/humanoid/data/traj_v1",
        humanoid_noise_seq=["noise_0", "noise_0.07"],
        discard_last_frame=True,
    ),
    group=train_group_name,
)
MainStore.store(
    name="humanoid_train_im_v1",
    node=builds(
        Humanml3dDataset,
        cam_augmentation="static",
        split="train",
        humanoid_data_dir="inputs/humanoid/data/traj_im_v1",
        humanoid_noise_seq=["noise_0", "noise_0.07"],
        discard_last_frame=True,
    ),
    group=train_group_name,
)
MainStore.store(
    name="humanoid_train_im_v1_taskobs",
    node=builds(
        Humanml3dDataset,
        cam_augmentation="static",
        split="train",
        humanoid_data_dir="inputs/humanoid/data/traj_im_v1",
        humanoid_noise_seq=["noise_0", "noise_0.07"],
        humanoid_separate_task_obs=True,
        discard_last_frame=True,
    ),
    group=train_group_name,
)

if __name__ == "__main__":
    dataset = Humanml3dDataset(
        cam_augmentation="static",
        split="train",
        humanoid_data_dir="inputs/humanoid/data/traj_v1",
        humanoid_noise_seq=["noise_0", "noise_0.07"],
        discard_last_frame=True,
    )
    print(len(dataset))
    for i in range(len(dataset)):
        data = dataset[i]
        print(i)
