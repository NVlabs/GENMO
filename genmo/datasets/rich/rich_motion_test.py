from pathlib import Path

import torch
from torch.utils import data

from genmo.utils.geo_transform import (
    compute_cam_angvel,
    compute_cam_tvel,
    normalize_T_w2c,
    transform_mat,
)
from genmo.utils.net_utils import get_valid_mask
from genmo.utils.pylogger import Log
from genmo.utils.rotation_conversions import axis_angle_to_matrix
from third_party.GVHMR.hmr4d.utils.geo.hmr_cam import resize_K

from .rich_utils import (
    get_cam2params,
    get_cam_key_wham_vid,
    get_w2az_sahmr,
    parse_seqname_info,
)

VID_PRESETS = {
    "easytohard": [
        "test/Gym_013_burpee4/cam_06",
        "test/Gym_011_pushup1/cam_02",
        "test/LectureHall_019_wipingchairs1/cam_03",
        "test/ParkingLot2_009_overfence1/cam_04",
        "test/LectureHall_021_sidebalancerun1/cam_00",
        "test/Gym_010_dips2/cam_05",
    ],
}


class RichSmplFullSeqDataset(data.Dataset):
    def __init__(self, vid_presets=None):
        """
        Args:
            vid_presets is a key in VID_PRESETS
        """
        super().__init__()
        self.dataset_name = "RICH"
        self.dataset_id = "RICH"
        Log.info(f"[{self.dataset_name}] Full sequence, Test")
        tic = Log.time()

        # Load evaluation protocol from WHAM labels
        self.rich_dir = Path("inputs/RICH/hmr4d_support")
        self.labels = torch.load(self.rich_dir / "rich_test_labels.pt")
        self.preproc_data = torch.load(self.rich_dir / "rich_test_preproc.pt")
        vids = select_subset(self.labels, vid_presets)

        self.vimo_labels = torch.load(self.rich_dir / "rich_test_vimo_preproc.pt")
        # Setup dataset index
        self.idx2meta = []
        for vid in vids:
            seq_length = len(self.labels[vid]["frame_id"])
            self.idx2meta.append((vid, 0, seq_length))  # start=0, end=seq_length
        # print(sum([end - start for _, _, start, end in self.idx2meta]))

        # Prepare ground truth motion in ay-coordinate
        self.w2az = (
            get_w2az_sahmr()
        )  # scan_name -> T_w2az, w-coordinate refers to cam-1-coordinate
        self.cam2params = get_cam2params()  # cam_key -> (T_w2c, K)
        seqname_info = parse_seqname_info(
            skip_multi_persons=True
        )  # {k: (scan_name, subject_id, gender, cam_ids)}
        self.seqname_to_scanname = {k: v[0] for k, v in seqname_info.items()}

        Log.info(
            f"[RICH] {len(self.idx2meta)} sequences. Elapsed: {Log.time() - tic:.2f}s"
        )

    def __len__(self):
        return len(self.idx2meta)

    def _load_data(self, idx):
        data = {}

        # [start, end), when loading data from labels
        vid, start, end = self.idx2meta[idx]
        label = self.labels[vid]
        preproc_data = self.preproc_data[vid]

        vimo_label = self.vimo_labels[vid]

        length = end - start
        meta = {"dataset_id": "RICH", "vid": vid, "vid-start-end": (start, end)}
        data.update({"meta": meta, "length": length})

        # SMPLX
        data.update(
            {"gt_smpl_params": label["gt_smplx_params"], "gender": label["gender"]}
        )
        vimo_smpl_params = {
            "pred_cam": vimo_label["vimo_params"]["pred_cam"],
            "pred_pose": vimo_label["vimo_params"]["pred_pose"],
            "pred_shape": vimo_label["vimo_params"]["pred_shape"],
            "pred_trans_c": vimo_label["vimo_params"]["pred_trans"],
        }
        data.update({"vimo_smpl_params": vimo_smpl_params})
        data["vimo_label"] = vimo_label
        # camera
        cam_key = get_cam_key_wham_vid(vid)
        scan_name = self.seqname_to_scanname[vid.split("/")[1]]
        T_w2c, K = self.cam2params[cam_key]  # (4, 4)  (3, 3)
        T_w2az = self.w2az[scan_name]
        data.update({"T_w2c": T_w2c, "T_w2az": T_w2az, "K": K})

        # image features
        data.update(
            {
                "f_imgseq": preproc_data["f_imgseq"],
                "bbx_xys": preproc_data["bbx_xys"],
                "img_wh": preproc_data["img_wh"],
                "kp2d": preproc_data["kp2d"],
            }
        )

        # to render a video
        video_path = self.rich_dir / "video" / vid / "video.mp4"
        frame_id = label["frame_id"]  # (F,)
        width, height = data["img_wh"] / 4  #  Video saved has been downsampled 1/4
        K_render = resize_K(K, 0.25)
        bbx_xys_render = data["bbx_xys"] / 4
        data["meta_render"] = {
            "name": vid.replace("/", "@"),
            "video_path": str(video_path),
            "frame_id": frame_id,
            "width_height": (width, height),
            "K": K_render,
            "bbx_xys": bbx_xys_render,
        }

        return data

    def _process_data(self, data):
        # T_w2az is pre-computed by using floor clue. az2zy uses a rotation along x-axis.
        R_az2ay = axis_angle_to_matrix(
            torch.tensor([1.0, 0.0, 0.0]) * -torch.pi / 2
        )  # (3, 3)
        T_w2ay = (
            transform_mat(R_az2ay, R_az2ay.new([0, 0, 0])) @ data["T_w2az"]
        )  # (4, 4)

        vimo_label = data["vimo_label"]

        # process img feature with xys
        length = data["length"]
        f_imgseq = data["f_imgseq"]  # (F, 1024)
        normed_T_w2c = normalize_T_w2c(data["T_w2c"])
        R_w2c = normed_T_w2c[:, :3, :3].repeat(length, 1, 1)  # (L, 4, 4)
        t_w2c = normed_T_w2c[:, :3, 3].repeat(length, 1)  # (L, 3)
        cam_angvel = compute_cam_angvel(R_w2c)  # (L, 6)
        cam_tvel = compute_cam_tvel(t_w2c)  # (L, 3)

        K_fullimg = data["K"][None].expand(length, -1, -1)  # (L, 3, 3)

        scales = torch.ones(length)
        mean_scale = 1.0
        # Return
        return_data = {
            # --- not batched
            "task": "CAP-Seq",
            "meta": data["meta"],
            "meta_render": data["meta_render"],
            # --- we test on single sequence, so set kv manually
            "length": length,
            "f_imgseq": f_imgseq,
            "cam_angvel": cam_angvel,
            "cam_tvel": cam_tvel,
            "R_w2c": R_w2c,
            "bbx_xys": data["bbx_xys"],  # (F, 3)
            "K_fullimg": K_fullimg,  # (L, 3, 3)
            "kp2d": data["kp2d"],  # (F, 17, 3)
            # --- dataset specific
            "model": "smplx",
            "gender": data["gender"],
            "gt_smpl_params": data["gt_smpl_params"],
            "T_w2ay": T_w2ay,  # (4, 4)
            "T_w2c": data["T_w2c"],  # (4, 4)
            "scales": scales,
            "mean_scale": mean_scale,
            "vimo_smpl_params": data["vimo_smpl_params"],
            "mask": {
                "valid": get_valid_mask(length, length),
                "has_img_mask": get_valid_mask(length, length),
                "has_2d_mask": get_valid_mask(length, length),
                "has_cam_mask": get_valid_mask(length, length),
                "has_audio_mask": get_valid_mask(length, 0),
                "has_music_mask": get_valid_mask(length, 0),
            },
        }

        if "vimo_params_flip" in vimo_label:
            flipped_trans_c = vimo_label["vimo_params_flip"]["pred_trans"]
            orig_trans_c = data["vimo_smpl_params"]["pred_trans_c"]
            tz = flipped_trans_c[..., 2]
            tx = flipped_trans_c[..., 0]
            focal = K_fullimg[0, 0, 0]
            cx = K_fullimg[0, 0, 2]
            width = data["meta_render"]["width_height"][0]

            flipped_tx = tz * (width - 1 - 2 * cx) / focal - tx
            avg_trans_c = torch.zeros_like(flipped_trans_c)
            avg_trans_c[..., 0] = (flipped_tx + orig_trans_c[..., 0]) / 2
            avg_trans_c[..., 0] = orig_trans_c[..., 0]
            avg_trans_c[..., 1] = (flipped_trans_c[..., 1] + orig_trans_c[..., 1]) / 2
            avg_trans_c[..., 2] = (tz + orig_trans_c[..., 2]) / 2
            return_data["vimo_smpl_params"]["pred_trans_c"] = avg_trans_c

        return return_data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data)
        return data


def select_subset(labels, vid_presets):
    vids = list(labels.keys())
    if vid_presets is not None:  # Use a subset of the videos
        vids = VID_PRESETS[vid_presets]
    return vids
