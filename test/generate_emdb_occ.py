import glob
import os
import random
import time

import cv2
import numpy as np
import torch
from occ_utils import rand_img_clip_transforms2
from tqdm import tqdm

from hmr4d.utils.preproc import Extractor, SLAMModel, Tracker, VitPoseExtractor

emdb_root = "/mnt/dhd/body-pose-dataset/EMDB"


def draw_bbox(img, bbx_xys):
    x, y, s = bbx_xys
    cv2.rectangle(
        img,
        (int(x - s / 2), int(y - s / 2)),
        (int(x + s / 2), int(y + s / 2)),
        (0, 0, 255),
        2,
    )
    return img


if __name__ == "__main__":
    labels = torch.load("inputs/EMDB/hmr4d_support/emdb_vit_v4.pt")
    occ_labels = {}
    device = "cuda:0"
    tic = time.time()
    feat_extractor = Extractor(device=device, tqdm_leave=True)
    print("Finished loading features extractor:", time.time() - tic)
    tic = time.time()
    vitpose_extractor = VitPoseExtractor(device=device, tqdm_leave=True)
    print("Finished loading pose extractor:", time.time() - tic)
    os.makedirs("outputs/tmp_emdb_occ", exist_ok=True)

    keys = list(labels.keys())
    # random.shuffle(keys)
    bar = tqdm(keys)
    for vid in bar:
        if os.path.exists(f"outputs/tmp_emdb_occ/{vid}.pt"):
            label_occ = torch.load(f"outputs/tmp_emdb_occ/{vid}.pt")
            occ_labels[vid] = label_occ
            continue
        bar.set_description(f"Processing {vid}:")
        pid = vid.split("_")[0]
        dir_name = "_".join(vid.split("_")[1:])
        label = labels[vid]
        label_occ = {
            "name": label["name"],
            "gender": label["gender"],
            "smpl_params": label["smpl_params"],
            "mask": label["mask"],
            "K_fullimg": label["K_fullimg"],
            "T_w2c": label["T_w2c"],
        }
        bbx_xys = label["bbx_xys"]
        kp2d = label["kp2d"]
        img_dir = f"{emdb_root}/{pid}/{dir_name}/images"

        occ_bbx_xys = []
        bbx_cx = torch.cat(
            [bbx_xys[:, :2].clone(), bbx_xys[:, [2]].clone(), bbx_xys[:, [2]].clone()],
            dim=1,
        )
        for i in range(len(bbx_xys)):
            center, scale = rand_img_clip_transforms2(
                i, i, len(bbx_xys), kp2d.numpy(), bbx_cx.numpy(), 1
            )
            occ_bbx_xys_i = torch.from_numpy(
                np.concatenate([center, scale[:1]], axis=0)
            ).float()
            occ_bbx_xys.append(occ_bbx_xys_i)
        occ_bbx_xys = torch.stack(occ_bbx_xys)
        assert occ_bbx_xys.shape == bbx_xys.shape, (occ_bbx_xys.shape, bbx_xys.shape)
        label_occ["bbx_xys"] = occ_bbx_xys

        imgnames = sorted(glob.glob(f"{img_dir}/*.jpg"))
        vitpose, imgs_dict = vitpose_extractor.extract(
            imgnames, occ_bbx_xys, img_ds=0.5, path_type="image_list"
        )
        assert vitpose.shape == label["kp2d"].shape, (
            vitpose.shape,
            label["kp2d"].shape,
        )
        label_occ["kp2d"] = vitpose.cpu().float()

        vit_features = feat_extractor.extract_video_features(
            imgnames,
            bbx_xys,
            batch_size=32,
            path_type="image_list",
            imgs_dict=imgs_dict,
        )
        assert vit_features.shape == label["features"].shape, (
            vit_features.shape,
            label["features"].shape,
        )
        label_occ["features"] = vit_features.cpu().float()

        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # sample = cv2.imread(imgnames[0])
        # height, width = sample.shape[:2]
        # out = cv2.VideoWriter(f"outputs/vis_emdb_occ_{vid}.mp4", fourcc, 30, (width, height))

        # for i, imgname in tqdm(enumerate(imgnames)):
        #     img = cv2.imread(imgname)
        #     img = draw_bbox(img, occ_bbx_xys[i])
        #     out.write(img)
        # out.release()
        # import ipdb; ipdb.set_trace()
        occ_labels[vid] = label_occ
        torch.save(label_occ, f"outputs/tmp_emdb_occ/{vid}.pt")
    torch.save(occ_labels, "outputs/emdb_occ_labels.pt")
