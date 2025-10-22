from pathlib import Path

import torch
from torch.utils import data

from genmo.utils.geo_transform import (
    compute_cam_angvel,
    compute_cam_tvel,
    normalize_T_w2c,
)
from genmo.utils.net_utils import get_valid_mask
from genmo.utils.pylogger import Log
from genmo.utils.rotation_conversions import matrix_to_axis_angle, quaternion_to_matrix
from third_party.GVHMR.hmr4d.utils.geo.flip_utils import flip_kp2d_coco17
from third_party.GVHMR.hmr4d.utils.geo.hmr_cam import estimate_K, resize_K

from .utils import EMDB1_NAMES, EMDB2_NAMES

VID_PRESETS = {1: EMDB1_NAMES, 2: EMDB2_NAMES}


def as_identity(R):
    is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
    R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
    return R


class EmdbSmplFullSeqDataset(data.Dataset):
    def __init__(self, split=1, flip_test=False):
        """
        split: 1 for EMDB-1, 2 for EMDB-2
        flip_test: if True, extra flip data will be returned
        """
        super().__init__()
        self.dataset_name = "EMDB"
        self.split = split
        self.dataset_id = f"EMDB_{split}"
        Log.info(f"[{self.dataset_name}] Full sequence, split={split}")

        # Load evaluation protocol from WHAM labels
        tic = Log.time()
        self.emdb_dir = Path("inputs/EMDB/hmr4d_support")
        # 'name', 'gender', 'smpl_params', 'mask', 'K_fullimg', 'T_w2c', 'bbx_xys', 'kp2d', 'features'
        self.labels = torch.load(self.emdb_dir / "emdb_vit_v4.pt")
        self.cam_traj = torch.load(
            self.emdb_dir / "emdb_dpvo_traj.pt"
        )  # estimated with DPVO

        self.vimo_labels = torch.load(self.emdb_dir / "emdb_vimo.pt")
        self.droid_cam_traj = torch.load(
            self.emdb_dir / "emdb_slam_traj.pt"
        )  # estimated with SLAM

        # Setup dataset index
        self.idx2meta = []
        for vid in VID_PRESETS[split]:
            seq_length = len(self.labels[vid]["mask"])
            self.idx2meta.append((vid, 0, seq_length))  # start=0, end=seq_length
        Log.info(
            f"[{self.dataset_name}] {len(self.idx2meta)} sequences. Elapsed: {Log.time() - tic:.2f}s"
        )

        # If flip_test is enabled, we will return extra data for flipped test
        self.flip_test = flip_test
        if self.flip_test:
            Log.info(f"[{self.dataset_name}] Flip test enabled")

    def __len__(self):
        return len(self.idx2meta)

    def _load_data(self, idx):
        data = {}

        # [vid, start, end]
        vid, start, end = self.idx2meta[idx]
        length = end - start
        meta = {
            "dataset_id": self.dataset_id,
            "vid": vid,
            "vid-start-end": (start, end),
        }
        data.update({"meta": meta, "length": length})

        label = self.labels[vid]
        vimo_label = self.vimo_labels[vid]
        droid_label = self.droid_cam_traj[vid]

        # smpl_params in world
        gender = label["gender"]
        smpl_params = label["smpl_params"]
        mask = label["mask"]
        mask_dict = {
            "valid": mask,
            "has_img_mask": get_valid_mask(length, length),
            "has_2d_mask": get_valid_mask(length, length),
            "has_cam_mask": get_valid_mask(length, length),
            "has_audio_mask": get_valid_mask(length, 0),
            "has_music_mask": get_valid_mask(length, 0),
            # "2d_only": False,
        }
        data.update({"smpl_params": smpl_params, "gender": gender, "mask": mask_dict})
        vimo_smpl_params = {
            "pred_cam": vimo_label["vimo_params"]["pred_cam"],
            "pred_pose": vimo_label["vimo_params"]["pred_pose"],
            "pred_shape": vimo_label["vimo_params"]["pred_shape"],
            "pred_trans_c": vimo_label["vimo_params"]["pred_trans"],
        }

        data.update({"vimo_smpl_params": vimo_smpl_params})

        # camera
        # load droid slam
        R_c2w = torch.from_numpy(droid_label["pred_cam_R"]).float()
        t_c2w = torch.from_numpy(droid_label["pred_cam_T"]).float()
        scales = torch.from_numpy(droid_label["all_scales"]).float()
        mean_scale = droid_label["scale"]
        T_c2w = torch.eye(4)[None].repeat(length, 1, 1).to(R_c2w)
        T_c2w[:, :3, :3] = R_c2w
        T_c2w[:, :3, 3] = t_c2w
        T_w2c = T_c2w.inverse()

        # K_fullimg = label["K_fullimg"]  # We use estimated K
        width_height = (1440, 1920) if vid != "P0_09_outdoor_walk" else (720, 960)
        K_fullimg = estimate_K(*width_height)
        # T_w2c = label["T_w2c"]  # use GT camera trajectory
        gt_T_w2c = label["T_w2c"]
        data.update(
            {
                "K_fullimg": K_fullimg,
                "T_w2c": T_w2c,
                "scales": scales,
                "mean_scale": mean_scale,
                "gt_T_w2c": gt_T_w2c,
            }
        )

        if "vimo_params_flip" in vimo_label:
            flipped_trans_c = vimo_label["vimo_params_flip"]["pred_trans"]
            orig_trans_c = data["vimo_smpl_params"]["pred_trans_c"]
            tz = flipped_trans_c[..., 2]
            tx = flipped_trans_c[..., 0]
            focal = K_fullimg[0, 0]
            cx = K_fullimg[0, 2]
            width = width_height[0]

            flipped_tx = tz * (width - 1 - 2 * cx) / focal - tx
            avg_trans_c = torch.zeros_like(flipped_trans_c)
            avg_trans_c[..., 0] = (flipped_tx + orig_trans_c[..., 0]) / 2
            avg_trans_c[..., 0] = orig_trans_c[..., 0]
            avg_trans_c[..., 1] = (flipped_trans_c[..., 1] + orig_trans_c[..., 1]) / 2
            avg_trans_c[..., 2] = (tz + orig_trans_c[..., 2]) / 2
            data["vimo_smpl_params"]["pred_trans_c"] = avg_trans_c

        # R_w2c -> cam_angvel
        use_DPVO = False
        if use_DPVO:
            traj = self.cam_traj[data["meta"]["vid"]]  # (L, 7)
            R_w2c = quaternion_to_matrix(traj[:, [6, 3, 4, 5]]).mT  # (L, 3, 3)
            t_c2w = traj[:, :3]
        else:  # GT
            # L = data["T_w2c"].shape[0]
            norm_T_w2c = normalize_T_w2c(data["T_w2c"])

            R_w2c = norm_T_w2c[:, :3, :3]
            t_w2c = norm_T_w2c[:, :3, 3]

            data["cam_angvel"] = compute_cam_angvel(R_w2c)  # (L, 6)
            data["cam_tvel"] = compute_cam_tvel(t_w2c)  # (L, 3)
            data["R_w2c"] = R_w2c

        # image bbx, features
        bbx_xys = label["bbx_xys"]
        f_imgseq = label["features"]
        kp2d = label["kp2d"]
        data.update({"bbx_xys": bbx_xys, "f_imgseq": f_imgseq, "kp2d": kp2d})

        # to render a video
        video_path = self.emdb_dir / f"videos/{vid}.mp4"
        frame_id = torch.where(mask)[0].long()
        resize_factor = 0.5
        width_height_render = torch.tensor(width_height) * resize_factor
        K_render = resize_K(K_fullimg, resize_factor)
        bbx_xys_render = bbx_xys * resize_factor
        data["meta_render"] = {
            "split": self.split,
            "name": vid,
            "video_path": str(video_path),
            "resize_factor": resize_factor,
            "frame_id": frame_id,
            "width_height": width_height_render.int(),
            "K": K_render,
            "bbx_xys": bbx_xys_render,
            "R_cam_type": "DPVO" if use_DPVO else "GtGyro",
        }

        # if enable flip_test
        if self.flip_test:
            imgfeat_dir = self.emdb_dir / "imgfeats/emdb_flip"
            f_img_dict = torch.load(imgfeat_dir / f"{vid}.pt")

            flipped_bbx_xys = f_img_dict["bbx_xys"].float()  # (L, 3)
            flipped_features = f_img_dict["features"].float()  # (L, 1024)
            width = width_height[0]
            flipped_kp2d = flip_kp2d_coco17(kp2d, width)  # (L, 17, 3)

            R_flip_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
            flipped_R_w2c = R_flip_x @ R_w2c.clone()
            flipped_t_w2c = (R_flip_x @ t_w2c.clone()[..., None])[..., 0]
            flipped_T_w2c = torch.eye(4)[None].repeat(length, 1, 1).to(flipped_R_w2c)
            flipped_T_w2c[:, :3, :3] = flipped_R_w2c
            flipped_T_w2c[:, :3, 3] = flipped_t_w2c

            data_flip = {
                "bbx_xys": flipped_bbx_xys,
                "f_imgseq": flipped_features,
                "kp2d": flipped_kp2d,
                "cam_angvel": compute_cam_angvel(flipped_R_w2c),
                "cam_tvel": compute_cam_tvel(flipped_t_w2c),
                "R_w2c": flipped_R_w2c,
            }
            flipped_trans_c = vimo_label["vimo_params_flip"]["pred_trans"]
            flipped_trans_c[..., 2] = avg_trans_c[..., 2]
            vimo_smpl_params_flip = {
                "pred_cam": vimo_label["vimo_params_flip"]["pred_cam"],
                "pred_pose": vimo_label["vimo_params_flip"]["pred_pose"],
                "pred_shape": vimo_label["vimo_params_flip"]["pred_shape"],
                "pred_trans_c": flipped_trans_c,
            }
            data_flip["vimo_smpl_params"] = vimo_smpl_params_flip

            flipped_K_fullimg = K_fullimg.clone()
            data_flip.update(
                {
                    "K_fullimg": flipped_K_fullimg,
                    "T_w2c": flipped_T_w2c,
                    "scales": scales,
                    "mean_scale": mean_scale,
                }
            )
            data["flip_test"] = data_flip

        return data

    def _process_data(self, data):
        length = data["length"]
        data["K_fullimg"] = data["K_fullimg"][None].repeat(length, 1, 1)
        return data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data)
        return data
