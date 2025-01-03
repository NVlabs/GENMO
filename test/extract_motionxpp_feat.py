import glob
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

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

def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box on the given frame.

    Args:
    frame (numpy.ndarray): The input frame on which to draw the bounding box.
    bbox (tuple): The bounding box coordinates (x, y, w, h).
    color (tuple): The color of the bounding box (default is green).
    thickness (int): The thickness of the bounding box lines (default is 2).

    Returns:
    numpy.ndarray: The frame with the bounding box drawn.
    """
    x, y, w, h = bbox
    top_left = (int(x), int(y))
    bottom_right = (int(x + w), int(y + h))
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)
    return frame


if __name__ == "__main__":
    device = "cuda:0"
    smplx_layer = make_smplx(type="smplx-motionx").to(device)

    motionx_db_feat = {}
    motionx_db = torch.load("/mnt/disk3/motion-x++/motionxpp_smplxposev2.pth")
    extractor = Extractor()

    for vid in tqdm(motionx_db):
        data = motionx_db[vid]
        subset = data["subset"]
        file_name = data["file_name"]
        vid_name = f"{subset}_{file_name}"
        video_file = f"/mnt/disk3/motion-x++/video/{subset}/{file_name}.mp4"
        assert os.path.exists(video_file), (video_file, video_file)
        bbx_xywh = data["bbox"]
        x1, y1, w, h = bbx_xywh[:, 0], bbx_xywh[:, 1], bbx_xywh[:, 2], bbx_xywh[:, 3]
        bbx_xyxy = torch.stack([x1, y1, x1 + w, y1 + h], dim=1)

        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3) apply aspect ratio and enlarge

        vit_features = extractor.extract_video_features(video_file, bbx_xys, batch_size=32)
        # torch.save(vit_features, paths.vit_features)
        motionx_db_feat[vid_name] = vit_features
        assert vit_features.shape[0] == bbx_xys.shape[0], (vit_features.shape, bbx_xys.shape)
        if vit_features.shape[0] != data["pose"].shape[0]:
            assert vit_features.shape[0] == data["pose"].shape[0] + 1, (vit_features.shape, data["pose"].shape)
            vit_features = vit_features[:-1]
        motionx_db_feat[vid_name] = vit_features.float()

print(f"total {len(motionx_db_feat)} samples")
torch.save(motionx_db_feat, "/mnt/disk3/motion-x++/motionxpp_feat.pth")
