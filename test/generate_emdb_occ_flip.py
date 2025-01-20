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

emdb_root = "/mnt/dhd/body-pose-dataset/EMDB"


def draw_bbox(img, bbx_xys):
    x, y, s = bbx_xys
    cv2.rectangle(img, (int(x-s/2), int(y-s/2)), (int(x+s/2), int(y+s/2)), (0, 0, 255), 2)
    return img




if __name__ == "__main__":
    labels = torch.load("inputs/EMDB/hmr4d_support/emdb_occ_labels.pt")
    device = "cuda:1"
    tic = time.time()
    feat_extractor = Extractor(device=device, tqdm_leave=True)
    print('Finished loading features extractor:', time.time() - tic)
    tic = time.time()
    os.makedirs("outputs/tmp_emdb_occ_flip/f_dict", exist_ok=True)

    keys = list(labels.keys())
    random.shuffle(keys)
    bar = tqdm(keys)
    for vid in bar:
        if os.path.exists(f"outputs/tmp_emdb_occ_flip/f_dict/{vid}.pt"):
            label_occ = torch.load(f"outputs/tmp_emdb_occ_flip/f_dict/{vid}.pt")
            continue
        bar.set_description(f'Processing {vid}:')
        pid = vid.split('_')[0]
        dir_name = "_".join(vid.split("_")[1:])
        label = labels[vid]

        bbx_xys = label["bbx_xys"]
        img_dir = f'{emdb_root}/{pid}/{dir_name}/images'

        imgnames = sorted(glob.glob(f"{img_dir}/*.jpg"))

        sample = cv2.imread(imgnames[0])
        width = sample.shape[1]

        if os.path.exists(f"outputs/tmp_emdb_occ_flip/imgfeats/{vid}.pt"):
            vit_features = torch.load(f"outputs/tmp_emdb_occ_flip/imgfeats/{vid}.pt")
        else:
            vit_features = feat_extractor.extract_video_features(
                imgnames, bbx_xys, batch_size=32, path_type="image_list", flip=True
            )
        flipped_bbx_xys = bbx_xys.clone()
        flipped_bbx_xys[..., 0] = width - bbx_xys[..., 0] - 1 
        assert vit_features.shape == label['features'].shape, (vit_features.shape, label['features'].shape)

        data = {
            "features": vit_features,
            "bbx_xys": flipped_bbx_xys,
        }
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
        torch.save(data, f"outputs/tmp_emdb_occ_flip/f_dict/{vid}.pt")
