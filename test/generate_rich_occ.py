import torch
from tqdm import tqdm
import glob
import cv2
import numpy as np
import time
import os
from occ_utils import rand_img_clip_transforms2
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel
import random

rich_root = "/mnt/disk1/RICH/"


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


def select_subset(labels, vid_presets):
    vids = list(labels.keys())
    if vid_presets is not None:  # Use a subset of the videos
        vids = VID_PRESETS[vid_presets]
    return vids


def draw_kpt(img, kp2d):
    for i, kp in enumerate(kp2d):
        if i in [11, 12, 13, 14, 15, 16]:
            cv2.circle(img, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)
        elif i in [0, 0, 7, 8, 9, 10]:
            cv2.circle(img, (int(kp[0]), int(kp[1])), 2, (255, 255, 255), -1)
    return img


def draw_bbox(img, bbx_xys):
    x, y, s = bbx_xys
    cv2.rectangle(img, (int(x-s/2), int(y-s/2)), (int(x+s/2), int(y+s/2)), (0, 0, 255), 2)
    return img



def get_cam_key_wham_vid(vid):
    _, sname, cname = vid.split("/")
    scene = sname.split("_")[0]
    cid = int(cname.split("_")[1])
    cam_key = f"{scene}_{cid}"
    return cam_key


if __name__ == "__main__":
    labels = torch.load("inputs/RICH/hmr4d_support/rich_test_preproc.pt")
    data = torch.load('/home/jiefengl/git/GVHMR/inputs/RICH/hmr4d_support/rich_test_labels.pt')
    occ_labels = {}
    device = "cuda:0"
    tic = time.time()
    feat_extractor = Extractor(device=device, tqdm_leave=True)
    print('Finished loading features extractor:', time.time() - tic)
    tic = time.time()
    vitpose_extractor = VitPoseExtractor(device=device, tqdm_leave=True)
    print('Finished loading pose extractor:', time.time() - tic)
    os.makedirs("outputs/tmp_rich_occ", exist_ok=True)

    keys = list(labels.keys())
    random.shuffle(keys)
    bar = tqdm(keys)
    for vid in bar:
        if os.path.exists(f"outputs/tmp_rich_occ/{vid.replace('/', '_')}.pt"):
            label_occ = torch.load(f"outputs/tmp_rich_occ/{vid.replace('/', '_')}.pt")
            occ_labels[vid] = label_occ
            continue
        bar.set_description(f'Processing {vid}:')
        label = labels[vid]
        label_occ = {
            'vid': label['vid'],
            'img_wh': label['img_wh'],
        }
        bbx_xys = label["bbx_xys"]
        kp2d = label["kp2d"]
        img_dir = f'{rich_root}/{vid}'
        cam_id = vid.split('/')[-1].split('_')[1]

        img_files_glob = sorted(glob.glob(f"{img_dir}/*.jpeg"))

        if "ParkingLot2" in vid:
            frame_id_offset = int(os.path.basename(img_files_glob[0]).split("_")[0])
        else:
            frame_id_offset = 0
        imgnames = [
            f"{img_dir}/{imgid + frame_id_offset:05d}_{cam_id}.jpeg"
            for imgid in data[vid]["frame_id"]
        ]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        sample = cv2.imread(imgnames[0])
        height, width = sample.shape[:2]
        imgscale = min(height, width)

        occ_bbx_xys = []
        bbx_cx = torch.cat([bbx_xys[:, :2].clone(), bbx_xys[:, [2]].clone(), bbx_xys[:, [2]].clone()], dim=1)
        for i in range(len(bbx_xys)):
            center, scale = rand_img_clip_transforms2(
                i, i, len(bbx_xys), kp2d.numpy(), bbx_cx.numpy(), 1, imgscale=imgscale
            )
            occ_bbx_xys_i = torch.from_numpy(np.concatenate([center, scale[:1]], axis=0)).float()
            occ_bbx_xys.append(occ_bbx_xys_i)
        occ_bbx_xys = torch.stack(occ_bbx_xys)
        assert occ_bbx_xys.shape == bbx_xys.shape, (occ_bbx_xys.shape, bbx_xys.shape)
        label_occ['bbx_xys'] = occ_bbx_xys

        # out = cv2.VideoWriter(f"outputs/vis_rich_occ_{vid.replace('/', '_')}.mp4", fourcc, 30, (width, height))

        # for i, imgname in tqdm(enumerate(imgnames)):
        #     img = cv2.imread(imgname)
        #     img = draw_bbox(img, occ_bbx_xys[i])
        #     img = draw_kpt(img, label['kp2d'][i])
        #     out.write(img)
        # out.release()
        # import ipdb; ipdb.set_trace()

        vitpose, imgs_dict = vitpose_extractor.extract(imgnames, occ_bbx_xys, img_ds=0.5, batch_size=32, path_type="image_list")
        assert vitpose.shape == label['kp2d'].shape, (vitpose.shape, label['kp2d'].shape)
        label_occ["kp2d"] = vitpose.cpu().float()

        vit_features = feat_extractor.extract_video_features(
            imgnames, bbx_xys, batch_size=32, path_type="image_list", imgs_dict=imgs_dict
        )
        assert vit_features.shape == label['f_imgseq'].shape, (vit_features.shape, label['f_imgseq'].shape)
        label_occ["f_imgseq"] = vit_features.cpu().float()

        occ_labels[vid] = label_occ
        torch.save(label_occ, f"outputs/tmp_rich_occ/{vid.replace('/', '_')}.pt")
    torch.save(occ_labels, "outputs/rich_occ_labels.pt")
