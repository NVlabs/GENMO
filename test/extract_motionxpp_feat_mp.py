import glob
import os
import torch.multiprocessing as mp

import json

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
import torch.multiprocessing as mp

from tqdm import tqdm
import contextlib
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy

def process_video(vid, data, extractor, device):
    # Set the device for this process
    torch.cuda.set_device(device)
    
    subset = data["subset"]
    file_name = data["file_name"]
    vid_name = f"{subset}_{file_name}"
    video_file = f"/mnt/disk3/motion-x++/video/{subset}/{file_name}.mp4"
    assert os.path.exists(video_file), (video_file, video_file)
    bbx_xywh = data["bbox"]
    x1, y1, w, h = bbx_xywh[:, 0], bbx_xywh[:, 1], bbx_xywh[:, 2], bbx_xywh[:, 3]
    bbx_xyxy = torch.stack([x1, y1, x1 + w, y1 + h], dim=1)

    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
    # with contextlib.redirect_stdout(None):
    vit_features = extractor.extract_video_features(video_file, bbx_xys, batch_size=32)
    assert vit_features.shape[0] == bbx_xys.shape[0], (vit_features.shape, bbx_xys.shape)
    if vit_features.shape[0] != data["pose"].shape[0]:
        assert vit_features.shape[0] == data["pose"].shape[0] + 1, (vit_features.shape, data["pose"].shape)
        vit_features = vit_features[:-1]
    return vid_name, vit_features.float()

if __name__ == "__main__":
    device_ids = [0, 1]  # List of GPU device IDs to use
    smplx_layer = make_smplx(type="smplx-motionx").to(f"cuda:{device_ids[0]}")
    motionx_db_feat = {}
    motionx_db = torch.load("/mnt/disk3/motion-x++/motionxpp_smplxposev2.pth")
    extractor_dict = {i: Extractor(f"cuda:{i}", tqdm_leave=False) for i in device_ids}

    with mp.Pool(processes=len(device_ids)) as pool:
        results = [pool.apply_async(process_video, args=(vid, motionx_db[vid], extractor_dict[device_ids[i % len(device_ids)]], device_ids[i % len(device_ids)]))
                   for i, vid in enumerate(motionx_db)]
        for result in tqdm(results):
            vid_name, vit_features = result.get()
            motionx_db_feat[vid_name] = vit_features

    print(f"total {len(motionx_db_feat)} samples")
    torch.save(motionx_db_feat, "/mnt/disk3/motion-x++/motionxpp_feat.pth")
