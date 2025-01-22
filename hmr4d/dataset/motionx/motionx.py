from pathlib import Path
import numpy as np
import torch
from hmr4d.utils.pylogger import Log
from motiondiff.models.mdm.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

from time import time

from hmr4d.configs import MainStore, builds
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.video_io_utils import read_video_np, save_video

import hmr4d.utils.matrix as matrix
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from hmr4d.dataset.bedlam.utils import mid2featname, mid2vname
from hmr4d.utils.geo_transform import compute_cam_angvel, compute_cam_tvel, apply_T_on_points
from hmr4d.utils.geo.hmr_global import get_T_w2c_from_wcparams, get_c_rootparam, get_R_c2gv
from hmr4d.utils.geo_transform import normalize_T_w2c, as_identity
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy


class MotionXDataset(ImgfeatMotionDatasetBase):
    def __init__(
        self,
        version='v2d', # [v2d, vlocal, vglobal]
        motion_start_mode='sample',  # ["sample", "first"]
        split='train',
        max_num_motions=None,
        random_permute=False,
        random_seed=7,
        max_motion_frames=120,
    ):
        self.hmr4d_support_dir = Path("inputs/MotionXpp/hmr4d_support")
        self.root = Path("inputs/MotionXpp/hmr4d_support")
        self.text_embed_file = Path("inputs/MotionXpp_ye/t5_embeddings_v1_half/all_text_embed.pth") # TODO: USE THE STANDARD PATH
        self.dataset_name = "Motion-X++"
        self.split = split
        # Setting
        self.min_motion_frames = 60
        self.max_motion_frames = max_motion_frames
        self.version = version
        self.motion_start_mode = motion_start_mode
        self.max_num_motions = max_num_motions
        self.random_permute = random_permute
        self.random_seed = random_seed
        super().__init__()

    def _load_dataset(self):
        Log.info(f"[Motion-X++] Loading from {self.root}")
        tic = time()
        self.train_labels = torch.load(self.root / f"motionxpp_smplxposev3_{self.split}.pth")
        self.f_img_folder = self.hmr4d_support_dir / "imgfeats/"
        self.text_embed_dict = torch.load(self.text_embed_file)

        Log.info(f"[Motion-X++] Motion files loaded. Elapsed: {time() - tic:.2f}s")

    def _get_idx2meta(self):
        seq_lengths = []
        for vid in self.train_labels:
            seq_length = self.train_labels[vid]["pose"].shape[0]
            seq_lengths.append(seq_length)
        self.idx2meta = list(self.train_labels.keys())
        if self.random_permute:
            rng = np.random.RandomState(self.random_seed)
            shuffle_ind = np.arange(len(self.idx2meta))
            rng.shuffle(shuffle_ind)
            self.idx2meta = [self.idx2meta[i] for i in shuffle_ind]
            seq_lengths = [seq_lengths[i] for i in shuffle_ind]
        if self.max_num_motions is not None:
            self.idx2meta = self.idx2meta[: self.max_num_motions]
            seq_lengths = seq_lengths[: self.max_num_motions]
        minutes = sum(seq_lengths) / 30 / 60
        Log.info(
            f"[{self.dataset_name}] has {minutes:.1f} minutes motion -> Resampled to {len(self.idx2meta)} samples."
        )

    def _load_data(self, idx):
        data = {}
        vid = self.idx2meta[idx]
        data = self.train_labels[vid].copy()
        text_embed = self.text_embed_dict[vid].float()
        subset = data["subset"]
        file_name = data["file_name"]

        # Random select a subset
        mlength = data["pose"].shape[0]
        min_motion_len = self.min_motion_frames
        max_motion_len = self.max_motion_frames

        if mlength < min_motion_len:  # the minimal mlength is 30 when generating data
            length = mlength
        else:
            length = min(max_motion_len, mlength)
        if self.motion_start_mode == "sample":
            start = np.random.randint(0, max(mlength - length + 1, 1))
        else:
            start = 0
        end = start + length
        data["start_end"] = (start, end)
        data["length"] = length
        mask = np.zeros((self.max_motion_frames,))
        mask[:length] = 1
        data["meta"] = {
            "data_name": self.dataset_name,
            "idx": idx,
            "vid": vid,
            "subset": subset,
            "file_name": file_name,
            "start_end": (start, end),
        }

        # Update data to a subset
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and len(v.shape) > 1 and k != "skeleton":
                data[k] = v[start:end]

        # Load img(as feature) : {mid -> 'features', 'bbx_xys', 'img_wh', 'start_end'}
        imgfeat_file = self.f_img_folder / f"{subset}/{file_name}.pth"
        features = torch.load(imgfeat_file)
        bbx_xywh = data['bbox']
        x1, y1, w, h = bbx_xywh[:, 0], bbx_xywh[:, 1], bbx_xywh[:, 2], bbx_xywh[:, 3]
        bbx_xyxy = torch.stack([x1, y1, x1 + w, y1 + h], dim=1)

        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3) apply aspect ratio and enlarge

        width, height = data['width'], data['height']
        kp2d = data["body_kpts"]
        # remap (start, end)
        data["f_imgseq"] = features[start:end].float()  # (L, 1024)

        data["bbx_xys"] = bbx_xys.float()  # (L, 4)
        data["img_wh"] = torch.tensor([width, height])  # (2)
        data["kp2d"] = kp2d.float()  # (L, 17, 3)

        fx, fy, cx, cy = data['intrins']
        K_fullimg = torch.eye(3).repeat(self.max_motion_frames, 1, 1)
        K_fullimg[:, 0, 0] = fx
        K_fullimg[:, 1, 1] = fy
        K_fullimg[:, 0, 2] = cx
        K_fullimg[:, 1, 2] = cy
        data['K_fullimg'] = K_fullimg
        data['mask'] = mask.astype(np.bool8)
        data['text_embed'] = text_embed[0]

        return data

    def _process_data(self, data, idx):
        length = data["length"]

        start, end = data['start_end']

        # SMPL params in cam
        body_pose = data["pose"][start:end]  # (F, 63)
        betas = data["beta"][start:end]  # (F, 10)
        global_orient_c = data["global_orient_c"]  # (F, 3)
        transl_c = data["trans_c"]
        smpl_params_c = {
            "body_pose": body_pose,
            "betas": betas,
            "transl": transl_c,
            "global_orient": global_orient_c,
        }

        # SMPL params in world
        global_orient_w = data["global_orient"]  # (F, 3)
        transl_w = data["trans"]  # (F, 3)
        smpl_params_w = {"body_pose": body_pose, "betas": betas, "transl": transl_w, "global_orient": global_orient_w}

        gravity_vec = torch.tensor([0, -1, 0], dtype=torch.float32)  # (3), BEDLAM is ay
        R_w2c = data['cam_R']
        t_w2c = data['cam_T']
        T_w2c = torch.eye(4).repeat(length, 1, 1)
        T_w2c[:, :3, :3] = R_w2c
        T_w2c[:, :3, 3] = t_w2c
        R_c2gv = get_R_c2gv(T_w2c[:, :3, :3], gravity_vec)  # (F, 3, 3)

        # cam_angvel (slightly different from WHAM)
        normed_T_w2c = normalize_T_w2c(T_w2c)
        noisy_normed_T_w2c = normed_T_w2c.clone()
        noisy_t_w2c = noisy_normed_T_w2c[:, :3, 3]
        rand_scale = min(max(0.1, torch.randn(1) + 3), 10)
        noisy_t_w2c = noisy_t_w2c / rand_scale
        noisy_normed_T_w2c[:, :3, 3] = noisy_t_w2c

        cam_angvel = compute_cam_angvel(normed_T_w2c[:, :3, :3])  # (F, 6)
        cam_tvel = compute_cam_tvel(normed_T_w2c[:, :3, 3])  # (F, 3)
        noisy_cam_tvel = compute_cam_tvel(noisy_normed_T_w2c[:, :3, 3])  # (F, 3)

        # Returns: do not forget to make it batchable! (last lines)
        max_len = self.max_motion_frames
        if self.version == "v2d":
            obs_kp2d = data["kp2d"][:, :, :2].reshape(-1, 1, 17, 2).numpy()
            conf = data["kp2d"][:, :, 2].numpy()
            return_data = {
                "idx": idx,
                "obs_kp2d": obs_kp2d,
                "K_fullimg": data["K_fullimg"],
                "f_imgseq": data["f_imgseq"],  # (F, D)
                "mask": data["mask"].astype(np.bool8),
                "conf": conf,
                "length": length,
                "is_2d": True,
                "use_cliffcam": True,
                "meta": {
                    "dataset_id": "motion-x++2d",
                    "eval_text_only": True,
                },
                "caption": data["text"],
                "dataset_name": self.dataset_name,
            }
            # Batchable
            return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
            return_data["f_imgseq"] = repeat_to_max_len(return_data["f_imgseq"], max_len)
            return_data["obs_kp2d"] = repeat_to_max_len(torch.from_numpy(return_data["obs_kp2d"]), max_len).numpy()
            return_data["conf"] = repeat_to_max_len(torch.from_numpy(return_data["conf"]), max_len).numpy()
            return_data["text_embed"] = data["text_embed"]
        else:
            return_data = {
                "meta": {"data_name": "bedlam", "idx": idx},
                "length": length,
                "smpl_params_c": smpl_params_c,
                "smpl_params_w": smpl_params_w,
                "R_c2gv": R_c2gv,  # (F, 3, 3)
                "gravity_vec": gravity_vec,  # (3)
                "bbx_xys": data["bbx_xys"],  # (F, 3)
                "K_fullimg": data["K_fullimg"],  # (F, 3, 3)
                "f_imgseq": data["f_imgseq"],  # (F, D)
                "kp2d": data["kp2d"],  # (F, 17, 3)
                "cam_angvel": cam_angvel,  # (F, 6)
                "cam_tvel": cam_tvel,  # (F, 3)
                "noisy_cam_tvel": noisy_cam_tvel,  # (F, 3)
                "T_w2c": normed_T_w2c,  # (F, 4, 4)
                "mask": {
                    "valid": get_valid_mask(max_len, length),
                    "vitpose": False,
                    "bbx_xys": True,
                    "f_imgseq": True,
                    "spv_incam_only": True if self.version == "vglobal" else False,
                },
                "caption": data["text"],
            }

            # Batchable
            return_data["smpl_params_c"] = repeat_to_max_len_dict(return_data["smpl_params_c"], max_len)
            return_data["smpl_params_w"] = repeat_to_max_len_dict(return_data["smpl_params_w"], max_len)
            return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
            return_data["bbx_xys"] = repeat_to_max_len(return_data["bbx_xys"], max_len)
            return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
            return_data["f_imgseq"] = repeat_to_max_len(return_data["f_imgseq"], max_len)
            return_data["kp2d"] = repeat_to_max_len(return_data["kp2d"], max_len)
            return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)
            return_data["cam_tvel"] = repeat_to_max_len(return_data["cam_tvel"], max_len)
            return_data["noisy_cam_tvel"] = repeat_to_max_len(return_data["noisy_cam_tvel"], max_len)
            return_data["T_w2c"] = repeat_to_max_len(return_data["T_w2c"], max_len)

        return return_data


MainStore.store(name="v2d", node=builds(MotionXDataset, version="v2d", motion_start_mode="sample", split="train"), group="train_2d_datasets/imgfeat_motionx")
MainStore.store(name="v2d_train", node=builds(MotionXDataset, version="v2d", motion_start_mode="sample", split="train"), group="train_2d_datasets/imgfeat_motionx")
MainStore.store(name="v2d_val", node=builds(MotionXDataset, version="v2d", motion_start_mode="sample", split="val"), group="test_datasets/imgfeat_motionx")
MainStore.store(name="v2d_test", node=builds(MotionXDataset, version="v2d", motion_start_mode="sample", split="test"), group="test_datasets/imgfeat_motionx")
MainStore.store(name="v2d_test32", node=builds(MotionXDataset, version="v2d", motion_start_mode="sample", split="test", max_num_motions=32, random_permute=True), group="test_datasets/imgfeat_motionx")
MainStore.store(name="v2d_test128", node=builds(MotionXDataset, version="v2d", motion_start_mode="sample", split="test", max_num_motions=128, random_permute=True), group="test_datasets/imgfeat_motionx")

MainStore.store(name="vlocal", node=builds(MotionXDataset, version="vlocal", motion_start_mode="sample", split="train"), group="train_datasets/imgfeat_motionx")
MainStore.store(name="vlocal_train", node=builds(MotionXDataset, version="vlocal", motion_start_mode="sample", split="train"), group="train_datasets/imgfeat_motionx")
MainStore.store(name="vlocal_val", node=builds(MotionXDataset, version="vlocal", motion_start_mode="sample", split="val"), group="test_datasets/imgfeat_motionx")
MainStore.store(name="vlocal_test", node=builds(MotionXDataset, version="vlocal", motion_start_mode="sample", split="test"), group="test_datasets/imgfeat_motionx")

MainStore.store(name="vglobal", node=builds(MotionXDataset, version="vglobal", motion_start_mode="sample", split="train"), group="train_datasets/imgfeat_motionx")
MainStore.store(name="vglobal_train", node=builds(MotionXDataset, version="vglobal", motion_start_mode="sample", split="train"), group="train_datasets/imgfeat_motionx")
MainStore.store(name="vglobal_val", node=builds(MotionXDataset, version="vglobal", motion_start_mode="sample", split="val"), group="test_datasets/imgfeat_motionx")
MainStore.store(name="vglobal_test", node=builds(MotionXDataset, version="vglobal", motion_start_mode="sample", split="test"), group="test_datasets/imgfeat_motionx")

