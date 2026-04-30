import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")

import argparse
from glob import glob

import numpy as np
import torch
from pycocotools import mask as masktool

from hmr4d.camera import calibrate_intrinsics, run_metric_slam
from hmr4d.camera.pipeline import detect_segment_track

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, default="./examples", help="input img dir")
parser.add_argument("--out_dir", type=str, default="./results", help="output dir")
parser.add_argument(
    "--static_camera", action="store_true", help="whether the camera is static"
)
parser.add_argument(
    "--visualize_mask", action="store_true", help="save deva vos for visualization"
)
args = parser.parse_args()

# File and folders
img_folder = args.img_dir

seq_folder = args.out_dir
os.makedirs(seq_folder, exist_ok=True)

##### Detection + SAM + DEVA-Track-Anything #####
print("Detect, Segment, and Track ...")
imgfiles = sorted(glob(f"{img_folder}/*.jpg"))
nframes = len(imgfiles)
if os.path.exists(f"{seq_folder}/boxes.npy"):
    boxes_ = np.load(f"{seq_folder}/boxes.npy", allow_pickle=True)
    masks_ = np.load(f"{seq_folder}/masks.npy", allow_pickle=True)
    tracks_ = np.load(f"{seq_folder}/tracks.npy", allow_pickle=True)
else:
    boxes_, masks_, tracks_ = detect_segment_track(
        imgfiles, seq_folder, thresh=0.25, min_size=100, save_vos=args.visualize_mask
    )

##### Run Masked DROID-SLAM #####
print("Masked Metric SLAM ...")
masks = np.array([masktool.decode(m) for m in masks_])
masks = torch.from_numpy(masks)

cam_int = np.load(f"{seq_folder}/cam_int.npy", allow_pickle=True)
is_static = False
assert os.path.exists(f"{seq_folder}/cam_int.npy")
# cam_int, is_static = calibrate_intrinsics(img_folder, masks, is_static=args.static_camera)
cam_R, cam_T, all_scales, scale = run_metric_slam(
    img_folder, masks=masks, calib=cam_int, is_static=is_static
)

camera = {
    "pred_cam_R": cam_R.numpy(),
    "pred_cam_T": cam_T.numpy(),
    "all_scales": all_scales,
    "scale": scale,
    "img_focal": cam_int[0],
    "img_center": cam_int[2:],
}

np.save(f"{seq_folder}/camera.npy", camera)
np.save(f"{seq_folder}/boxes.npy", boxes_)
np.save(f"{seq_folder}/masks.npy", masks_)
np.save(f"{seq_folder}/tracks.npy", tracks_)
