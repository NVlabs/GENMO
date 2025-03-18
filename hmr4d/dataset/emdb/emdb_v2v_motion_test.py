from pathlib import Path

import numpy as np
import torch
from moviepy.editor import AudioFileClip
from torch.utils import data

from hmr4d.utils.geo.flip_utils import flip_kp2d_coco17
from hmr4d.utils.geo.hmr_cam import estimate_K, resize_K
from hmr4d.utils.geo_transform import (
    compute_cam_angvel,
    compute_cam_tvel,
    normalize_T_w2c,
)
from hmr4d.utils.net_utils import get_valid_mask
from hmr4d.utils.pylogger import Log
from hmr4d.utils.wis3d_utils import add_motion_as_lines, make_wis3d
from motiondiff.models.mdm.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

from .utils import EMDB1_NAMES, EMDB2_NAMES

VID_PRESETS = {1: EMDB1_NAMES, 2: EMDB2_NAMES}


from hmr4d.configs import MainStore, builds


def as_identity(R):
    is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
    R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
    return R


class EmdbSmplFullSeqVTVDataset(data.Dataset):
    def __init__(
        self, split=1, flip_test=False, multicond_args=None, music_feat_version="v1"
    ):
        """
        split: 1 for EMDB-1, 2 for EMDB-2
        flip_test: if True, extra flip data will be returned
        """
        super().__init__()
        self.dataset_name = "EMDB"
        self.split = split
        # self.type = type
        self.multicond_args = multicond_args
        self.feat_version = music_feat_version
        if multicond_args.get("use_multicond", False):
            self.start_ind = multicond_args.video1.end_ind
            self.end_ind = multicond_args.video2.start_ind
            self.text_start_ind = multicond_args.text1.start_ind
            self.text_end_ind = multicond_args.text1.end_ind
            self.type = multicond_args.type
        else:
            self.start_ind = -1
            self.end_ind = -1
            self.type = "fix"
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

        self.text_embed_file = Path(
            "inputs/HumanML3D_SMPL_ye/t5_embeddings_v1_half/test_text_embed.pth"
        )
        self.text_embed_dict = torch.load(self.text_embed_file)

        text_motion_file = (
            Path("inputs/HumanML3D_SMPL/hmr4d_support") / "humanml3d_smplhpose_test.pth"
        )
        self.text_motion_files = torch.load(text_motion_file)
        self.text_seqs = list(self.text_motion_files.keys())

        self.music_motion_files = torch.load("inputs/AIST++/annot_aist_30fps.pt")
        self.music_split_set = torch.load("inputs/AIST++/test.pt")

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

        text_seq_lengths = []
        self.text_idx2meta = []

        # Skip too-long idle-prefix
        text_motion_start_id = {}
        self.motion_frames = 120
        self.use_random_subset = False
        for vid in self.text_motion_files:
            seq_length = self.text_motion_files[vid]["pose"].shape[0]
            start_id = text_motion_start_id[vid] if vid in text_motion_start_id else 0
            seq_length = seq_length - start_id
            if seq_length < 25:  # Skip clips that are too short
                continue
            num_samples = max(seq_length // self.motion_frames, 1)
            if self.use_random_subset:
                num_samples = 1
            text_seq_lengths.append(seq_length)
            self.text_idx2meta.extend([(vid, start_id)] * num_samples)
            assert start_id == 0, f"start_id is not 0 for {vid}"
        hours = sum(text_seq_lengths) / 30 / 3600
        Log.info(
            f"[{self.dataset_name}] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples."
        )

        self.music_idx2meta = []
        for vid in self.music_motion_files:
            if vid not in self.music_split_set:
                continue
            seq_length = self.music_motion_files[vid]["bbox_xyxy"].shape[0]
            self.music_idx2meta.extend([vid])

    def __len__(self):
        return len(self.idx2meta)

    def _load_data(self, idx, text_id):
        data = {}
        # if self.type == "fix":
        #     if text_id == 0:
        #         mid = "002246"
        #     elif text_id == 1:
        #         mid = "004562"
        #         # for i, (mid, _) in enumerate(self.text_idx2meta):
        #         #     text_seq = self.text_motion_files[mid]["text_data"]
        #         #     for caption in text_seq:
        #         #         # if 'raise' in caption['caption'] and 'walk' in caption['caption']:
        #         #         #     print(caption['caption'], mid, i)
        #         #         if 'person lunges forwards' in caption['caption']:
        #         #             print(caption['caption'], mid, i)
        #         #         elif 'body climbs up something and turns back' in caption['caption']:
        #         #             print(caption['caption'], mid, i)
        #         #         # elif 'body climbs up something and falls' in caption['caption']:
        #         #         #     print(caption['caption'], mid, i)
        #         #         # f.write(f"{caption['caption']} {mid} {i}\n")
        #     else:
        #         mid, start_id = self.text_idx2meta[text_id]  # 277, 278, 406, 494, 495
        # else:
        #     mid, start_id = self.text_idx2meta[text_id]  # 277, 278, 406, 494, 495
        # mid = "004222"  # 005375, 013157
        # mid = "005375"  # 004222, 013157
        # mid = "003784"
        # mid = "M004562"
        mid, start_id = self.text_idx2meta[text_id]
        text_vid = mid
        text_seq = self.text_motion_files[text_vid]["text_data"]
        text_ind = np.random.randint(0, len(text_seq))
        if self.type == "fix":
            if mid == "002246":
                text_ind = 2
            elif mid == "004562":
                text_ind = 0
            elif mid == "M004562":
                text_ind = 0
        text_embed_data = self.text_embed_dict[text_vid].float()
        text_embed = text_embed_data[text_ind]
        caption = text_seq[text_ind]["caption"]
        print(mid, text_ind, caption)

        # with open('text_seq.txt', 'w') as f:
        #     for i, (mid, _) in enumerate(self.text_idx2meta):
        #         text_seq = self.text_motion_files[mid]["text_data"]
        #         for caption in text_seq:
        #             # if 'raise' in caption['caption'] and 'walk' in caption['caption']:
        #             #     print(caption['caption'], mid, i)
        #             if 'clockwise' in caption['caption']:
        #                 # print(caption['caption'], mid, i)
        #                 f.write(f"{caption['caption']} {mid} {i}\n")
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
        data.update({"smpl_params": smpl_params, "gender": gender, "mask": mask})
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
        use_SLAM = False
        if use_DPVO:
            traj = self.cam_traj[data["meta"]["vid"]]  # (L, 7)
            R_w2c = quaternion_to_matrix(traj[:, [6, 3, 4, 5]]).mT  # (L, 3, 3)
            t_c2w = traj[:, :3]
        elif use_SLAM:
            L = data["T_w2c"].shape[0]
            norm_T_w2c = normalize_T_w2c(data["T_w2c"])

            R_w2c = norm_T_w2c[:, :3, :3]
            t_w2c = norm_T_w2c[:, :3, 3]

            data["cam_angvel"] = compute_cam_angvel(R_w2c)  # (L, 6)
            data["cam_tvel"] = compute_cam_tvel(t_w2c)  # (L, 3)
            data["R_w2c"] = R_w2c
        else:  # GT
            L = data["T_w2c"].shape[0]
            norm_T_w2c = normalize_T_w2c(data["gt_T_w2c"])

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

        length = data["length"]
        if self.multicond_args.use_multicond:
            length = min(length, 1000)
        else:
            return data

        cam_angvel = compute_cam_angvel(torch.eye(3)[None].repeat(length, 1, 1))
        cam_tvel = compute_cam_tvel(torch.zeros(length, 3))
        return_data = {
            "meta": data["meta"],
            "length": length,
            "bbx_xys": data["bbx_xys"][:length],
            # "bbx_xys": torch.zeros((length, 3)),  # (F, 3)  # NOTE: a placeholder
            "K_fullimg": data["K_fullimg"].reshape(1, 3, 3).repeat(length, 3, 3),
            "f_imgseq": data["f_imgseq"][:length],
            # "f_imgseq": torch.zeros((length, 1024)),  # (F, D)  # NOTE: a placeholder
            "kp2d": data["kp2d"][:length],
            # "kp2d": torch.zeros(length, 17, 3),  # (F, 17, 3)
            "cam_angvel": data["cam_angvel"][:length],
            # "cam_angvel": cam_angvel,
            "cam_tvel": data["cam_tvel"][:length],
            # "cam_tvel": cam_tvel,
            "R_w2c": data["R_w2c"][:length],
            "T_w2c": data["T_w2c"][:length],
            "gt_T_w2c": data["gt_T_w2c"][:length],
            "gender": data["gender"],
            "mask": {
                "valid": get_valid_mask(length, length),
                "vitpose": False,
                "bbx_xys": False,
                "f_imgseq": False,
                "spv_incam_only": False,
            },
        }
        # return_data["meta"]["eval_gen_only"] = True
        return_data["text_embed"] = text_embed
        return_data["caption"] = caption
        multi_text_data = {
            "vid": [mid],
            "caption": [caption],
            "text_ind": [text_ind],
            "text_embed": [text_embed],
            "window_start_ind": [self.text_start_ind],
            "window_end_ind": [self.text_end_ind],
        }
        multi_text_data["text_embed"] = torch.stack(multi_text_data["text_embed"])
        multi_text_data["window_start_ind"] = torch.tensor(
            multi_text_data["window_start_ind"]
        )
        multi_text_data["window_end_ind"] = torch.tensor(
            multi_text_data["window_end_ind"]
        )

        return_data["meta"]["multi_text_data"] = multi_text_data
        # return_data["caption"] = "A person is walking while raising hands."

        return return_data

    def _process_data(self, data):
        length = data["length"]
        # data["K_fullimg"] = data["K_fullimg"][None].repeat(length, 1, 1)
        return data

    def mask_data(self, data, start_ind, end_ind):
        data["bbx_xys"][start_ind:end_ind] = torch.zeros_like(
            data["bbx_xys"][start_ind:end_ind]
        )
        data["f_imgseq"][start_ind:end_ind] = torch.zeros_like(
            data["f_imgseq"][start_ind:end_ind]
        )
        data["kp2d"][start_ind:end_ind] = torch.zeros_like(
            data["kp2d"][start_ind:end_ind]
        )
        data["cam_angvel"][start_ind:end_ind] = compute_cam_angvel(
            torch.eye(3)[None].repeat(end_ind - start_ind, 1, 1)
        )
        # data["cam_tvel"][start_ind:end_ind] = torch.zeros_like(data["cam_tvel"][start_ind:end_ind])
        # data["R_w2c"][start_ind:end_ind] = torch.zeros_like(data["R_w2c"][start_ind:end_ind])
        return data

    def mask_music(self, data, music_id, start_ind, end_ind):
        music_vid = self.music_idx2meta[music_id]
        music_feat = torch.load(
            f"inputs/AIST++/musicfeat_{self.feat_version}/{music_vid}_musicfeat_fps30.pt"
        )
        music_length = end_ind - start_ind
        length = data["length"]
        music_feat = music_feat[:music_length]
        data["music_embed"] = torch.zeros((length, 35))
        # data["music_embed"][start_ind:music_feat.shape[0]] = torch.from_numpy(music_feat).float()
        data["music_embed"][start_ind:end_ind] = torch.from_numpy(music_feat).float()

        # load audio
        music_array = torch.load(f"inputs/AIST++/audio_array/{music_vid}.pt")
        music_array = torch.from_numpy(music_array).float()
        music = AudioFileClip(f"inputs/AIST++/audio/{music_vid}.mp3")
        music_fps = music.fps
        length_music_audio = int(music_length * music_fps / 30)
        start_audio = int(start_ind * music_fps / 30)
        end_audio = int(end_ind * music_fps / 30)
        music_array = music_array[:length_music_audio]
        entire_music_array = torch.zeros(
            (length * music_fps // 30, music_array.shape[1])
        )
        # entire_music_array[start_audio:music_array.shape[0]] = music_array
        entire_music_array[start_audio:end_audio] = music_array
        data["music_array"] = entire_music_array
        data["music_fps"] = music_fps

        data["bbx_xys"][start_ind:end_ind] = torch.zeros_like(
            data["bbx_xys"][start_ind:end_ind]
        )
        data["f_imgseq"][start_ind:end_ind] = torch.zeros_like(
            data["f_imgseq"][start_ind:end_ind]
        )
        data["kp2d"][start_ind:end_ind] = torch.zeros_like(
            data["kp2d"][start_ind:end_ind]
        )
        data["cam_angvel"][start_ind:end_ind] = compute_cam_angvel(
            torch.eye(3)[None].repeat(end_ind - start_ind, 1, 1)
        )
        return data

    def __getitem__(self, idx):
        text_id1 = torch.randint(0, len(self.text_idx2meta), (1,)).item()

        # if idx == 0:
        #     idx = 3
        #     text_id1 = 0
        # elif idx == 1:
        #     idx = 3
        #     text_id1 = 1
        # elif idx == 2:
        #     idx = 6
        #     text_id1 = 0
        # elif idx == 3:
        #     idx = 6
        #     text_id1 = 1
        idx = 3
        data = self._load_data(idx, text_id1)  # v1t1
        data = self._process_data(data)  # v1t1

        if self.start_ind < self.end_ind:
            end_ind = self.end_ind
            idx_2 = torch.randint(0, len(self.idx2meta), (1,)).item()
            if self.type == "fix":
                idx_2 = 10
            data_second = self._load_data(idx_2, text_id1)
            # data_second = self._load_data(6)
            len_second = data_second["length"]
            len_first = data["length"]
            # if len_second < len_first:
            #     len_first = len_second
            #     data["length"] = len_first
            #     data["bbx_xys"] = data["bbx_xys"][:len_first]
            #     data["K_fullimg"] = data["K_fullimg"][:len_first]
            #     data["f_imgseq"] = data["f_imgseq"][:len_first]
            #     data["kp2d"] = data["kp2d"][:len_first]
            #     data["cam_angvel"] = data["cam_angvel"][:len_first]
            #     data["cam_tvel"] = data["cam_tvel"][:len_first]
            #     data["R_w2c"] = data["R_w2c"][:len_first]
            #     data["T_w2c"] = data["T_w2c"][:len_first]
            #     data["gt_T_w2c"] = data["gt_T_w2c"][:len_first]
            #     data["mask"]["valid"] = get_valid_mask(len_first, len_first)

            # min_len = min(len_second, 1000)
            data_second = self._process_data(data_second)
            # data["bbx_xys"][end_ind:len_first] = data_second["bbx_xys"][300:len_first - end_ind + 300]
            # data["f_imgseq"][end_ind:len_first] = data_second["f_imgseq"][300:len_first - end_ind + 300]
            # data["kp2d"][end_ind:len_first] = data_second["kp2d"][300:len_first - end_ind + 300]
            # data["cam_angvel"][end_ind:len_first] = data_second["cam_angvel"][300:len_first - end_ind + 300]
            # data["cam_tvel"][end_ind:len_first] = data_second["cam_tvel"][300:len_first - end_ind + 300]
            # data["R_w2c"][end_ind:len_first] = data_second["R_w2c"][300:len_first - end_ind + 300]
            # data["T_w2c"][end_ind:len_first] = data_second["T_w2c"][300:len_first - end_ind + 300]
            # data["gt_T_w2c"][end_ind:len_first] = data_second["gt_T_w2c"][300:len_first - end_ind + 300]

            data["bbx_xys"] = torch.cat(
                [data["bbx_xys"][:end_ind], data_second["bbx_xys"][end_ind:len_first]],
                dim=0,
            )
            data["f_imgseq"] = torch.cat(
                [
                    data["f_imgseq"][:end_ind],
                    data_second["f_imgseq"][end_ind:len_first],
                ],
                dim=0,
            )
            data["K_fullimg"] = torch.cat(
                [
                    data["K_fullimg"][:end_ind],
                    data_second["K_fullimg"][end_ind:len_first],
                ],
                dim=0,
            )
            data["kp2d"] = torch.cat(
                [data["kp2d"][:end_ind], data_second["kp2d"][end_ind:len_first]], dim=0
            )
            data["cam_angvel"] = torch.cat(
                [
                    data["cam_angvel"][:end_ind],
                    data_second["cam_angvel"][end_ind:len_first],
                ],
                dim=0,
            )
            data["cam_tvel"] = torch.cat(
                [
                    data["cam_tvel"][:end_ind],
                    data_second["cam_tvel"][end_ind:len_first],
                ],
                dim=0,
            )
            data["R_w2c"] = torch.cat(
                [data["R_w2c"][:end_ind], data_second["R_w2c"][end_ind:len_first]],
                dim=0,
            )
            data["T_w2c"] = torch.cat(
                [data["T_w2c"][:end_ind], data_second["T_w2c"][end_ind:len_first]],
                dim=0,
            )
            data["gt_T_w2c"] = torch.cat(
                [
                    data["gt_T_w2c"][:end_ind],
                    data_second["gt_T_w2c"][end_ind:len_first],
                ],
                dim=0,
            )
            data["length"] = data["bbx_xys"].shape[0]
            data["mask"]["valid"] = get_valid_mask(data["length"], data["length"])
            data["meta"]["vid2"] = data_second["meta"]["vid"]  # v1t1v2
            data["meta"]["multi_text_data"]["window_start"] = (
                data["meta"]["multi_text_data"]["window_start_ind"] / data["length"]
            )
            data["meta"]["multi_text_data"]["window_end"] = (
                data["meta"]["multi_text_data"]["window_end_ind"] / data["length"]
            )
            data = self.mask_data(
                data, start_ind=self.text_start_ind, end_ind=self.text_end_ind
            )
            data = self.mask_music(
                data,
                music_id=10,
                start_ind=self.multicond_args.music1.start_ind,
                end_ind=self.multicond_args.music1.end_ind,
            )

        if self.type == "fix":
            return data
        text_id2 = torch.randint(0, len(self.text_idx2meta), (1,)).item()
        idx_3 = torch.randint(0, len(self.idx2meta), (1,)).item()
        data_v1t2v2 = self._load_data(idx, text_id2)

        if self.start_ind < self.end_ind:
            end_ind = self.end_ind
            # idx_2 = torch.randint(0, len(self.idx2meta), (1,)).item()
            # data_second = self._load_data(idx_2)
            # data_second = self._load_data(6)
            len_second = data_second["length"]
            len_first = data["length"]
            # min_len = min(len_second, 1000)
            # data_second = self._process_data(data_second)
            # data["bbx_xys"][end_ind:len_first] = data_second["bbx_xys"][300:len_first - end_ind + 300]
            # data["f_imgseq"][end_ind:len_first] = data_second["f_imgseq"][300:len_first - end_ind + 300]
            # data["kp2d"][end_ind:len_first] = data_second["kp2d"][300:len_first - end_ind + 300]
            # data["cam_angvel"][end_ind:len_first] = data_second["cam_angvel"][300:len_first - end_ind + 300]
            # data["cam_tvel"][end_ind:len_first] = data_second["cam_tvel"][300:len_first - end_ind + 300]
            # data["R_w2c"][end_ind:len_first] = data_second["R_w2c"][300:len_first - end_ind + 300]
            # data["T_w2c"][end_ind:len_first] = data_second["T_w2c"][300:len_first - end_ind + 300]
            # data["gt_T_w2c"][end_ind:len_first] = data_second["gt_T_w2c"][300:len_first - end_ind + 300]
            data_v1t2v2["bbx_xys"] = torch.cat(
                [data["bbx_xys"][:end_ind], data_second["bbx_xys"][end_ind:len_first]],
                dim=0,
            )
            data_v1t2v2["f_imgseq"] = torch.cat(
                [
                    data["f_imgseq"][:end_ind],
                    data_second["f_imgseq"][end_ind:len_first],
                ],
                dim=0,
            )
            data_v1t2v2["K_fullimg"] = torch.cat(
                [
                    data["K_fullimg"][:end_ind],
                    data_second["K_fullimg"][end_ind:len_first],
                ],
                dim=0,
            )
            data_v1t2v2["kp2d"] = torch.cat(
                [data["kp2d"][:end_ind], data_second["kp2d"][end_ind:len_first]], dim=0
            )
            data_v1t2v2["cam_angvel"] = torch.cat(
                [
                    data["cam_angvel"][:end_ind],
                    data_second["cam_angvel"][end_ind:len_first],
                ],
                dim=0,
            )
            data_v1t2v2["cam_tvel"] = torch.cat(
                [
                    data["cam_tvel"][:end_ind],
                    data_second["cam_tvel"][end_ind:len_first],
                ],
                dim=0,
            )
            data_v1t2v2["R_w2c"] = torch.cat(
                [data["R_w2c"][:end_ind], data_second["R_w2c"][end_ind:len_first]],
                dim=0,
            )
            data_v1t2v2["T_w2c"] = torch.cat(
                [data["T_w2c"][:end_ind], data_second["T_w2c"][end_ind:len_first]],
                dim=0,
            )
            data_v1t2v2["gt_T_w2c"] = torch.cat(
                [
                    data["gt_T_w2c"][:end_ind],
                    data_second["gt_T_w2c"][end_ind:len_first],
                ],
                dim=0,
            )
            data_v1t2v2["length"] = data_v1t2v2["bbx_xys"].shape[0]
            data_v1t2v2["mask"]["valid"] = get_valid_mask(
                data_v1t2v2["length"], data_v1t2v2["length"]
            )
            data_v1t2v2["meta"]["vid2"] = data_second["meta"]["vid"]  # v1t2v2
            data_v1t2v2["meta"]["multi_text_data"]["window_start"] = (
                data_v1t2v2["meta"]["multi_text_data"]["window_start_ind"]
                / data_v1t2v2["length"]
            )
            data_v1t2v2["meta"]["multi_text_data"]["window_end"] = (
                data_v1t2v2["meta"]["multi_text_data"]["window_end_ind"]
                / data_v1t2v2["length"]
            )
            data_v1t2v2 = self.mask_data(
                data_v1t2v2, start_ind=self.start_ind, end_ind=self.end_ind
            )

        data_v1t2v3 = self._load_data(idx, text_id2)

        if self.start_ind < self.end_ind:
            end_ind = self.end_ind
            # idx_2 = torch.randint(0, len(self.idx2meta), (1,)).item()
            data_third = self._load_data(idx_3, text_id2)
            len_third = data_third["length"]
            len_first = data["length"]
            min_len = min(len_third, 1000)
            data_third = self._process_data(data_third)
            data_v1t2v3["bbx_xys"] = torch.cat(
                [data["bbx_xys"][:end_ind], data_third["bbx_xys"][end_ind:len_first]],
                dim=0,
            )
            data_v1t2v3["f_imgseq"] = torch.cat(
                [data["f_imgseq"][:end_ind], data_third["f_imgseq"][end_ind:len_first]],
                dim=0,
            )
            data_v1t2v3["K_fullimg"] = torch.cat(
                [
                    data["K_fullimg"][:end_ind],
                    data_third["K_fullimg"][end_ind:len_first],
                ],
                dim=0,
            )
            data_v1t2v3["kp2d"] = torch.cat(
                [data["kp2d"][:end_ind], data_third["kp2d"][end_ind:len_first]], dim=0
            )
            data_v1t2v3["cam_angvel"] = torch.cat(
                [
                    data["cam_angvel"][:end_ind],
                    data_third["cam_angvel"][end_ind:len_first],
                ],
                dim=0,
            )
            data_v1t2v3["cam_tvel"] = torch.cat(
                [data["cam_tvel"][:end_ind], data_third["cam_tvel"][end_ind:len_first]],
                dim=0,
            )
            data_v1t2v3["R_w2c"] = torch.cat(
                [data["R_w2c"][:end_ind], data_third["R_w2c"][end_ind:len_first]], dim=0
            )
            data_v1t2v3["T_w2c"] = torch.cat(
                [data["T_w2c"][:end_ind], data_third["T_w2c"][end_ind:len_first]], dim=0
            )
            data_v1t2v3["gt_T_w2c"] = torch.cat(
                [data["gt_T_w2c"][:end_ind], data_third["gt_T_w2c"][end_ind:len_first]],
                dim=0,
            )
            data_v1t2v3["length"] = data_v1t2v3["bbx_xys"].shape[0]
            data_v1t2v3["mask"]["valid"] = get_valid_mask(
                data_v1t2v3["length"], data_v1t2v3["length"]
            )
            data_v1t2v3["meta"]["vid2"] = data_third["meta"]["vid"]  # v1t2v3
            data_v1t2v3["meta"]["multi_text_data"]["window_start"] = (
                data_v1t2v3["meta"]["multi_text_data"]["window_start_ind"]
                / data_v1t2v3["length"]
            )
            data_v1t2v3["meta"]["multi_text_data"]["window_end"] = (
                data_v1t2v3["meta"]["multi_text_data"]["window_end_ind"]
                / data_v1t2v3["length"]
            )
            data_v1t2v3 = self.mask_data(
                data_v1t2v3, start_ind=self.start_ind, end_ind=self.end_ind
            )

        data_v1t1v3 = self._load_data(idx, text_id1)

        if self.start_ind < self.end_ind:
            end_ind = self.end_ind
            # idx_2 = torch.randint(0, len(self.idx2meta), (1,)).item()
            # data_third = self._load_data(idx_3)
            # len_third = data_third['length']
            len_first = data["length"]
            # min_len = min(len_third, 1000)
            # data_third = self._process_data(data_third)
            data_v1t1v3["bbx_xys"] = torch.cat(
                [data["bbx_xys"][:end_ind], data_third["bbx_xys"][end_ind:len_first]],
                dim=0,
            )
            data_v1t1v3["f_imgseq"] = torch.cat(
                [data["f_imgseq"][:end_ind], data_third["f_imgseq"][end_ind:len_first]],
                dim=0,
            )
            data_v1t1v3["K_fullimg"] = torch.cat(
                [
                    data["K_fullimg"][:end_ind],
                    data_third["K_fullimg"][end_ind:len_first],
                ],
                dim=0,
            )
            data_v1t1v3["kp2d"] = torch.cat(
                [data["kp2d"][:end_ind], data_third["kp2d"][end_ind:len_first]], dim=0
            )
            data_v1t1v3["cam_angvel"] = torch.cat(
                [
                    data["cam_angvel"][:end_ind],
                    data_third["cam_angvel"][end_ind:len_first],
                ],
                dim=0,
            )
            data_v1t1v3["cam_tvel"] = torch.cat(
                [data["cam_tvel"][:end_ind], data_third["cam_tvel"][end_ind:len_first]],
                dim=0,
            )
            data_v1t1v3["R_w2c"] = torch.cat(
                [data["R_w2c"][:end_ind], data_third["R_w2c"][end_ind:len_first]], dim=0
            )
            data_v1t1v3["T_w2c"] = torch.cat(
                [data["T_w2c"][:end_ind], data_third["T_w2c"][end_ind:len_first]], dim=0
            )
            data_v1t1v3["gt_T_w2c"] = torch.cat(
                [data["gt_T_w2c"][:end_ind], data_third["gt_T_w2c"][end_ind:len_first]],
                dim=0,
            )
            data_v1t1v3["length"] = data_v1t1v3["bbx_xys"].shape[0]
            data_v1t1v3["mask"]["valid"] = get_valid_mask(
                data_v1t1v3["length"], data_v1t1v3["length"]
            )
            data_v1t1v3["meta"]["vid2"] = data_third["meta"]["vid"]  # v1t2v3
            data_v1t1v3["meta"]["multi_text_data"]["window_start"] = (
                data_v1t1v3["meta"]["multi_text_data"]["window_start_ind"]
                / data_v1t1v3["length"]
            )
            data_v1t1v3["meta"]["multi_text_data"]["window_end"] = (
                data_v1t1v3["meta"]["multi_text_data"]["window_end_ind"]
                / data_v1t1v3["length"]
            )
            data_v1t1v3 = self.mask_data(
                data_v1t1v3, start_ind=self.start_ind, end_ind=self.end_ind
            )

        if self.type == "v1t1v2":
            return data
        elif self.type == "v1t2v3":
            return data_v1t2v3
        elif self.type == "v1t1v3":
            return data_v1t1v3
        elif self.type == "v1t2v2":
            return data_v1t2v2
        elif self.type == "fix":
            return data
        else:
            raise ValueError(f"Invalid type: {self.type}")


# # EMDB-1 and EMDB-2
# MainStore.store(
#     name="v1",
#     node=builds(EmdbSmplFullSeqVTVDataset, populate_full_signature=True),
#     group="test_datasets/emdb1",
# )
# MainStore.store(
#     name="v1_fliptest",
#     node=builds(EmdbSmplFullSeqVTVDataset, flip_test=True, populate_full_signature=True),
#     group="test_datasets/emdb1",
# )
# MainStore.store(
#     name="v2",
#     node=builds(EmdbSmplFullSeqVTVDataset, split=2, populate_full_signature=True),
#     group="test_datasets/emdb2",
# )
# MainStore.store(
#     name="v1_fliptest",
#     node=builds(
#         EmdbSmplFullSeqVTVDataset, split=2, flip_test=True, populate_full_signature=True
#     ),
#     group="test_datasets/emdb2",
# )
