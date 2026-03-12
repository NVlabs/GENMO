"""HMR2 ViT feature extractor for GEM-SMPL demo.

Adapted from GVHMR's hmr4d/utils/preproc/vitfeat_extractor.py
"""

import cv2
import numpy as np
import torch
from tqdm import tqdm

from gem.network.hmr2.utils.preproc import IMAGE_MEAN, IMAGE_STD, crop_and_resize


def _read_video_np(video_path, scale=1.0):
    """Read video frames as numpy array (L, H, W, 3) RGB uint8."""
    from gem.utils.video_io_utils import read_video_np

    return read_video_np(video_path, scale=scale)


def _get_batch(video_path, bbx_xys, img_ds=1.0, img_dst_size=256):
    """Preprocess video frames: crop, resize, normalize for HMR2.

    Args:
        video_path: path to video file
        bbx_xys: (L, 3) tensor of [center_x, center_y, bbox_size]
        img_ds: downscale factor for reading video
        img_dst_size: output crop size (256 for HMR2)

    Returns:
        imgs: (L, 3, 256, 256) normalized tensor
        bbx_xys: (L, 3) updated bounding boxes
    """
    imgs = _read_video_np(video_path, scale=img_ds)

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    gt_bbx_size_ds = gt_bbx_size * img_ds
    ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
    imgs = np.stack(
        [
            cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
            for v, d in zip(imgs, ds_factors)
        ]
    )

    # Crop and resize each frame
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i].numpy() * img_ds,
            float(gt_bbx_size[i]) * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))  # (L, 256, 256, 3), RGB
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (L, 3)

    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (L, 3, 256, 256)
    return imgs, bbx_xys


class HMR2FeatureExtractor:
    """Extract HMR2 ViT features (1024-dim) from video frames."""

    def __init__(self, checkpoint_path, device="cuda"):
        from gem.network.hmr2 import load_hmr2

        self.model = load_hmr2(checkpoint_path).to(device).eval()
        self.device = device

    @torch.no_grad()
    def extract_video_features(self, video_path, bbx_xys, img_ds=1.0, batch_size=16):
        """Extract HMR2 features for each frame.

        Args:
            video_path: path to video file
            bbx_xys: (L, 3) tensor of [center_x, center_y, bbox_size]
            img_ds: downscale factor for reading video
            batch_size: inference batch size

        Returns:
            features: (L, 1024) tensor of ViT features
        """
        imgs, bbx_xys = _get_batch(video_path, bbx_xys, img_ds=img_ds)

        L = imgs.shape[0]
        imgs = imgs.to(self.device)
        features = []
        for j in tqdm(range(0, L, batch_size), desc="HMR2 Feature", leave=True):
            imgs_batch = imgs[j : j + batch_size]
            feature = self.model({"img": imgs_batch})
            features.append(feature.detach().cpu())

        features = torch.cat(features, dim=0).clone()  # (L, 1024)
        return features
