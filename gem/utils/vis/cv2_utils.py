# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import cv2
import numpy as np
import torch as _torch

# ---------------------------------------------------------------------------
# COCO-17 constants
# ---------------------------------------------------------------------------

COCO_SKELETON = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]

# Per-bone colors for the 19 COCO skeleton bones (BGR)
_COCO_BONE_COLORS = [
    (255, 190, 0),  # 0: 15-13 left leg
    (255, 190, 0),  # 1: 13-11 left leg
    (255, 0, 170),  # 2: 16-14 right leg
    (255, 0, 170),  # 3: 14-12 right leg
    (0, 215, 255),  # 4: 11-12 hip
    (0, 215, 255),  # 5: 5-11  left torso
    (0, 215, 255),  # 6: 6-12  right torso
    (0, 215, 255),  # 7: 5-6   shoulders
    (0, 255, 100),  # 8: 5-7   left upper arm
    (50, 130, 255),  # 9: 6-8   right upper arm
    (0, 255, 100),  # 10: 7-9  left forearm
    (50, 130, 255),  # 11: 8-10 right forearm
    (180, 130, 255),  # 12: 1-2  eyes
    (180, 130, 255),  # 13: 0-1  nose-left eye
    (180, 130, 255),  # 14: 0-2  nose-right eye
    (180, 130, 255),  # 15: 1-3  left ear
    (180, 130, 255),  # 16: 2-4  right ear
    (0, 255, 100),  # 17: 3-5  left ear-shoulder
    (50, 130, 255),  # 18: 4-6  right ear-shoulder
]

# Per-joint colors for 17 COCO joints (BGR)
_COCO_JOINT_COLORS = [
    (180, 130, 255),  # 0: nose
    (180, 130, 255),  # 1: left eye
    (180, 130, 255),  # 2: right eye
    (180, 130, 255),  # 3: left ear
    (180, 130, 255),  # 4: right ear
    (0, 255, 100),  # 5: left shoulder
    (50, 130, 255),  # 6: right shoulder
    (0, 255, 100),  # 7: left elbow
    (50, 130, 255),  # 8: right elbow
    (0, 255, 100),  # 9: left wrist
    (50, 130, 255),  # 10: right wrist
    (255, 190, 0),  # 11: left hip
    (255, 0, 170),  # 12: right hip
    (255, 190, 0),  # 13: left knee
    (255, 0, 170),  # 14: right knee
    (255, 190, 0),  # 15: left ankle
    (255, 0, 170),  # 16: right ankle
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _draw_ellipse_bone(canvas, pt1, pt2, color, stickwidth):
    """Draw a bone as a filled ellipse between two joints."""
    x1, y1 = pt1
    x2, y2 = pt2
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    length = math.hypot(x1 - x2, y1 - y2)
    if length < 1:
        return
    angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
    polygon = cv2.ellipse2Poly(
        (int(mx), int(my)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
    )
    cv2.fillConvexPoly(canvas, polygon, color, lineType=cv2.LINE_AA)


def get_colors_by_conf(conf):
    """Convert confidence scores [0, 1] to RGB colors (green=high, red=low).

    Args:
        conf: array-like of shape (J,)
    Returns:
        colors: (J, 3) float tensor in [0, 255]
    """
    conf_t = _torch.as_tensor(conf).float().clamp(0, 1).reshape(-1)
    colors = _torch.zeros(len(conf_t), 3)
    colors[:, 0] = (1 - conf_t) * 255  # R
    colors[:, 1] = conf_t * 255  # G
    return colors


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x.copy()
    elif isinstance(x, list):
        return np.array(x)
    return x.clone().cpu().numpy()


def draw_bbx_xys_on_image(bbx_xys, image, conf=True):
    assert isinstance(bbx_xys, np.ndarray)
    assert isinstance(image, np.ndarray)
    image = image.copy()
    lu_point = (bbx_xys[:2] - bbx_xys[2:] / 2).astype(int)
    rd_point = (bbx_xys[:2] + bbx_xys[2:] / 2).astype(int)
    color = (255, 178, 102) if conf else (128, 128, 128)  # orange or gray
    image = cv2.rectangle(image, lu_point, rd_point, color, 2)
    return image


def draw_bbx_xys_on_image_batch(bbx_xys_batch, image_batch, conf=None):
    """conf: if provided, list of bool"""
    use_conf = conf is not None
    bbx_xys_batch = to_numpy(bbx_xys_batch)
    assert len(bbx_xys_batch) == len(image_batch)
    image_batch_out = []
    for i in range(len(bbx_xys_batch)):
        if use_conf:
            image_batch_out.append(draw_bbx_xys_on_image(bbx_xys_batch[i], image_batch[i], conf[i]))
        else:
            image_batch_out.append(draw_bbx_xys_on_image(bbx_xys_batch[i], image_batch[i]))
    return image_batch_out


def draw_bbx_xyxy_on_image(bbx_xys, image, conf=True):
    bbx_xys = to_numpy(bbx_xys)
    image = to_numpy(image)
    color = (255, 178, 102) if conf else (128, 128, 128)  # orange or gray
    image = cv2.rectangle(
        image,
        (int(bbx_xys[0]), int(bbx_xys[1])),
        (int(bbx_xys[2]), int(bbx_xys[3])),
        color,
        2,
    )
    return image


def draw_bbx_xyxy_on_image_batch(bbx_xyxy_batch, image_batch, mask=None, conf=None):
    """
    Args:
        conf: if provided, list of bool, mutually exclusive with mask
        mask: whether to draw, historically used
    """
    if mask is not None:
        assert conf is None
    if conf is not None:
        assert mask is None
    use_conf = conf is not None
    bbx_xyxy_batch = to_numpy(bbx_xyxy_batch)
    image_batch = to_numpy(image_batch)
    assert len(bbx_xyxy_batch) == len(image_batch)
    image_batch_out = []
    for i in range(len(bbx_xyxy_batch)):
        if use_conf:
            image_batch_out.append(
                draw_bbx_xyxy_on_image(bbx_xyxy_batch[i], image_batch[i], conf[i])
            )
        else:
            if mask is None or mask[i]:
                image_batch_out.append(draw_bbx_xyxy_on_image(bbx_xyxy_batch[i], image_batch[i]))
            else:
                image_batch_out.append(image_batch[i])
    return image_batch_out


def draw_kpts(frame, keypoints, color=(0, 255, 0), thickness=2):
    frame_ = frame.copy()
    for x, y in keypoints:
        cv2.circle(frame_, (int(x), int(y)), thickness, color, -1)
    return frame_


def draw_kpts_with_conf(frame, kp2d, conf, thickness=2):
    """
    Args:
        kp2d: (J, 2),
        conf: (J,)
    """
    frame_ = frame.copy()
    conf = conf.reshape(-1)
    colors = get_colors_by_conf(conf)  # (J, 3)
    colors = colors[:, [2, 1, 0]].int().numpy().tolist()
    for j in range(kp2d.shape[0]):
        x, y = kp2d[j, :2]
        c = colors[j]
        cv2.circle(frame_, (int(x), int(y)), thickness, c, -1)
    return frame_


def draw_kpts_with_conf_batch(frames, kp2d_batch, conf_batch, thickness=2):
    """
    Args:
        kp2d_batch: (B, J, 2),
        conf_batch: (B, J)
    """
    assert len(frames) == len(kp2d_batch)
    assert len(frames) == len(conf_batch)
    frames_ = []
    for i in range(len(frames)):
        frames_.append(draw_kpts_with_conf(frames[i], kp2d_batch[i], conf_batch[i], thickness))
    return frames_


def draw_coco17_skeleton(img, keypoints, conf_thr=0):
    use_conf = keypoints.shape[1] == 3
    img = img.copy()

    # Draw bones as filled ellipses with per-bone colors and alpha blending
    for bone_idx, (a, b) in enumerate(COCO_SKELETON):
        if use_conf and (keypoints[a][2] <= conf_thr or keypoints[b][2] <= conf_thr):
            continue
        pt1 = keypoints[a][:2].tolist()
        pt2 = keypoints[b][:2].tolist()
        cur_canvas = img.copy()
        _draw_ellipse_bone(cur_canvas, pt1, pt2, _COCO_BONE_COLORS[bone_idx], 4)
        img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)

    # Draw joints with black outline + colored fill
    for j in range(keypoints.shape[0]):
        if use_conf and keypoints[j][2] <= conf_thr:
            continue
        x, y = keypoints[j][:2].astype(int)
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 3, _COCO_JOINT_COLORS[j], -1, cv2.LINE_AA)

    return img


def draw_coco17_skeleton_batch(imgs, keypoints_batch, conf_thr=0):
    assert len(imgs) == len(keypoints_batch)
    keypoints_batch = to_numpy(keypoints_batch)
    imgs_out = []
    for i in range(len(imgs)):
        imgs_out.append(draw_coco17_skeleton(imgs[i], keypoints_batch[i], conf_thr))
    return imgs_out
