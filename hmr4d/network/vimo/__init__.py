import torch
import numpy as np
from pathlib import Path
from hmr4d.network.vimo.hmr_vimo import HMR_VIMO
from hmr4d.network.vimo.config import update_cfg


def load_vimo(device="cuda"):
    cfg = str((Path(__file__).parent / "configs/config_vimo.yaml").resolve())
    ckpt = "inputs/checkpoints/vimo/vimo_checkpoint.pth.tar"

    cfg = update_cfg(cfg)

    cfg.DEVICE = device
    model = HMR_VIMO(cfg)
    ckpt = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()
    return model


def run_vimo(model, imgfiles, bbox_data, calib, device="cuda"):
    fx, fy, cx, cy = calib
    img_focal = float(fx)
    img_center = np.array([cx, cy])

    boxes_ck = bbox_data.cpu().numpy()
    boxes_ck = np.concatenate((boxes_ck, np.ones((boxes_ck.shape[0], 1))), axis=1)

    pred_smpl = model.inference(
        imgfiles, boxes_ck, img_focal=img_focal, img_center=img_center
    )
    return pred_smpl
