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
from third_party.GVHMR.hmr4d.utils.geo.flip_utils import flip_kp2d_coco17
from third_party.GVHMR.hmr4d.utils.geo.hmr_cam import estimate_K, resize_K

VID_HARD = []
# VID_HARD = ["downtown_bar_00_1"]


class ThreedpwOccSmplFullSeqDataset(data.Dataset):
    def __init__(self, flip_test=False, skip_invalid=False):
        super().__init__()
        self.dataset_name = "3DPW_OCC"
        self.skip_invalid = skip_invalid
        Log.info(f"[{self.dataset_name}] Full sequence")

        # Load evaluation protocol from WHAM labels
        self.threedpw_dir = Path("inputs/3DPW/hmr4d_support")
        # ['vname', 'K_fullimg', 'T_w2c', 'smpl_params', 'gender', 'mask_raw', 'mask_wham', 'img_wh']
        self.labels = torch.load(self.threedpw_dir / "test_3dpw_gt_labels.pt")
        self.vid2bbx = torch.load(self.threedpw_dir / "preproc_test_bbx.pt")
        self.vid2kp2d = torch.load(self.threedpw_dir / "preproc_test_kp2d_v0.pt")

        self.vimo_labels = torch.load(self.threedpw_dir / "test_3dpw_vimo_labels.pt")
        self.droid_cam_traj = torch.load(self.threedpw_dir / "3dpw_test_slam_traj.pt")
        # Setup dataset index
        self.idx2meta = list(self.labels)
        if len(VID_HARD) > 0:  # Pick subsets for fast testing
            self.idx2meta = VID_HARD
        Log.info(f"[{self.dataset_name}] {len(self.idx2meta)} sequences.")

        # If flip_test is enabled, we will return extra data for flipped test
        self.flip_test = flip_test
        if self.flip_test:
            Log.info(f"[{self.dataset_name}] Flip test enabled")

    def __len__(self):
        return len(self.idx2meta)

    def _load_data(self, idx):
        data = {}
        vid = self.idx2meta[idx]
        meta = {"dataset_id": self.dataset_name, "vid": vid}
        data.update({"meta": meta})

        # Add useful data
        label = self.labels[vid]
        vimo_label = self.vimo_labels[vid]
        droid_label = self.droid_cam_traj[vid]

        mask = label["mask_wham"]
        width_height = label["img_wh"]
        vimo_smpl_params = {
            "pred_cam": vimo_label["vimo_params"]["pred_cam"],
            "pred_pose": vimo_label["vimo_params"]["pred_pose"],
            "pred_shape": vimo_label["vimo_params"]["pred_shape"],
            "pred_trans_c": vimo_label["vimo_params"]["pred_trans"],
        }

        length = len(mask)
        data.update(
            {
                "length": length,  # F
                "smpl_params": label["smpl_params"],  # world
                "gender": label["gender"],  # str
                # "T_w2c": label["T_w2c"],  # (F, 4, 4)
                "mask": {
                    "valid": mask,  # (F)
                    "has_img_mask": get_valid_mask(length, length),
                    "has_2d_mask": get_valid_mask(length, length),
                    "has_cam_mask": get_valid_mask(length, length),
                    "has_audio_mask": get_valid_mask(length, 0),
                    "has_music_mask": get_valid_mask(length, 0),
                    "2d_only": False,
                },
            }
        )

        gt_T_w2c = label["T_w2c"]
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

        K_fullimg = label["K_fullimg"]  # (3, 3)
        if False:
            K_fullimg = estimate_K(*width_height)
        data["K_fullimg"] = K_fullimg
        data.update(
            {
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

        # Preprocessed:  bbx, kp2d, image as feature
        imgfeat_dir = self.threedpw_dir / "imgfeats/3dpw_occ_test"
        f_img_dict = torch.load(imgfeat_dir / f"{vid}.pth")
        # bbx_xys = self.vid2bbx[vid]["bbx_xys"]  # (F, 3)
        # kp2d = self.vid2kp2d[vid]  # (F, 17, 3)
        bbx_xys = f_img_dict["bbx_xys"].clone().float()
        kp2d = f_img_dict["vitpose"].clone().float()

        norm_T_w2c = normalize_T_w2c(data["T_w2c"])
        cam_angvel = compute_cam_angvel(norm_T_w2c[:, :3, :3])  # (L, 6)
        cam_tvel = compute_cam_tvel(norm_T_w2c[:, :3, 3])  # (L, 3)
        data.update(
            {
                "bbx_xys": bbx_xys,
                "kp2d": kp2d,
                "cam_angvel": cam_angvel,
                "cam_tvel": cam_tvel,
            }
        )
        data["R_w2c"] = norm_T_w2c[:, :3, :3]

        f_imgseq = f_img_dict["features"].float()
        kp2d = f_img_dict["vitpose"].clone().float()
        data["f_imgseq"] = f_imgseq  # (F, 1024)

        # to render a video
        vname = label["vname"]
        video_path = self.threedpw_dir / f"videos/{vname}.mp4"
        frame_id = torch.where(mask)[0].long()
        ds = 0.5
        K_render = resize_K(K_fullimg, ds)
        bbx_xys_render = bbx_xys * ds
        kp2d_render = kp2d.clone()
        kp2d_render[..., :2] *= ds
        data["meta_render"] = {
            "name": vid,
            "video_path": str(video_path),
            "ds": ds,
            "frame_id": frame_id,
            "K": K_render,
            "bbx_xys": bbx_xys_render,
            "kp2d": kp2d_render,
        }

        if self.flip_test:
            imgfeat_dir = self.threedpw_dir / "imgfeats/3dpw_occ_test_flip"
            f_img_dict = torch.load(imgfeat_dir / f"{vid}.pth")
            flipped_bbx_xys = f_img_dict["bbx_xys"].float()  # (L, 3)
            flipped_features = f_img_dict["features"].float()  # (L, 1024)
            flipped_kp2d = flip_kp2d_coco17(kp2d, width_height[0])  # (L, 17, 3)

            R_flip_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
            flipped_R_w2c = R_flip_x @ norm_T_w2c[:, :3, :3].clone()
            flipped_t_w2c = (R_flip_x @ norm_T_w2c[:, :3, 3:].clone())[..., 0]
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

        if self.skip_invalid:  # Drop all invalid frames
            mask = data["mask"].clone()
            data["length"] = sum(mask)
            data["smpl_params"] = {
                k: v[mask].clone() for k, v in data["smpl_params"].items()
            }
            data["T_w2c"] = data["T_w2c"][mask].clone()
            data["mask"] = data["mask"][mask].clone()
            data["K_fullimg"] = data["K_fullimg"][mask].clone()
            data["bbx_xys"] = data["bbx_xys"][mask].clone()
            data["kp2d"] = data["kp2d"][mask].clone()
            data["cam_angvel"] = data["cam_angvel"][mask].clone()
            data["cam_tvel"] = data["cam_tvel"][mask].clone()
            data["f_imgseq"] = data["f_imgseq"][mask].clone()
            data["flip_test"] = {
                k: v[mask].clone() for k, v in data["flip_test"].items()
            }

        return data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data)
        return data
