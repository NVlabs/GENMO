import torch
from hmr4d.network.hmr2 import load_hmr2, HMR2
from hmr4d.network.vimo import load_vimo, HMR_VIMO

from hmr4d.utils.video_io_utils import read_video_np
import cv2
import numpy as np

from hmr4d.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD
from tqdm import tqdm
import time
from multiprocessing import Pool


def read_image(path, scale=1.0):
    img = cv2.imread(str(path))[..., ::-1]
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return img

def read_images_mp(image_paths, scale=1.0):
    tic = time.time()
    # with Pool() as pool:
    #     imgs = pool.starmap(read_image, [(path, scale) for path in image_paths])
    imgs = [read_image(path, scale) for path in image_paths]
    toc = time.time()
    print(f"read_images_mp: {toc - tic:.2f}s")
    return np.stack(imgs)


def get_batch(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    if path_type == "video":
        imgs = read_video_np(input_path, scale=img_ds)
    elif path_type == "image_list":
        imgs = read_images_mp(input_path, scale=img_ds)
    elif path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0  # this is safe
        imgs = input_path

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    if True:
        gt_bbx_size_ds = gt_bbx_size * img_ds
        ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
        imgs = np.stack(
            [
                # gaussian(v, sigma=(d - 1) / 2, channel_axis=2, preserve_range=True) if d > 1.1 else v
                cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
                for v, d in zip(imgs, ds_factors)
            ]
        )

    # Output
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i] * img_ds,
            gt_bbx_size[i] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))  # (F, 256, 256, 3), RGB
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (F, 3)

    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (F, 3, 256, 256
    return imgs, bbx_xys


def get_batch_vimo(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    if path_type == "video":
        imgs = read_video_np(input_path, scale=img_ds)
    elif path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0  # this is safe
        imgs = input_path

    return imgs


class Extractor:
    def __init__(self, device='cuda:0', tqdm_leave=True):
        self.extractor: HMR2 = load_hmr2().to(device).eval()
        self.device = device
        self.tqdm_leave = tqdm_leave

    def extract_video_features(self, video_path, bbx_xys, img_ds=0.5, batch_size=16, path_type='video', imgs_dict=None, flip=False):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        if imgs_dict is not None and img_ds in imgs_dict:
            imgs = imgs_dict[img_ds]
        else:
            # Get the batch
            if isinstance(video_path, str) or isinstance(video_path, list):
                imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds, path_type=path_type)
            else:
                assert isinstance(video_path, torch.Tensor)
                imgs = video_path

        # Inference
        F, _, H, W = imgs.shape  # (F, 3, H, W)
        imgs = imgs.to(self.device)
        # batch_size = 16  # 5GB GPU memory, occupies all CUDA cores of 3090
        features = []
        if self.tqdm_leave:
            bar = tqdm(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave)
        else:
            bar = range(0, F, batch_size)
        # for j in tqdm(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave):
        for j in bar:
            imgs_batch = imgs[j : j + batch_size]

            with torch.no_grad():
                if flip:
                    imgs_batch = torch.flip(imgs_batch, dims=[3])
                feature = self.extractor({"img": imgs_batch})
                features.append(feature.detach().cpu())

        features = torch.cat(features, dim=0).clone()  # (F, 1024)
        return features


class ExtractorVIMO:
    def __init__(self, tqdm_leave=True):
        self.extractor: HMR_VIMO = load_vimo().cuda().eval()
        self.tqdm_leave = tqdm_leave

    def extract_video_features(self, video_path, bbx_xys, calib, img_ds=0.5):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        # Get the batch, imgs (B, H, W, 3)
        if isinstance(video_path, str):
            imgs = get_batch_vimo(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        fx, fy, cx, cy = calib
        img_focal = float(fx)
        img_center = np.array([cx, cy])

        cx, cy, s = bbx_xys.unbind(dim=-1)
        bbx_xyxy = torch.stack((cx-s/2, cy-s/2, cx+s/2, cy+s/2), dim=-1)

        boxes_ck = np.concatenate((bbx_xyxy, np.ones((bbx_xyxy.shape[0], 1))), axis=1)

        pred_smpl = self.extractor.inference(imgs, boxes_ck, img_focal=img_focal, img_center=img_center)

        return pred_smpl
