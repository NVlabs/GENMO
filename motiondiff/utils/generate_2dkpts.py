import cv2
import numpy as np
import torch
from tqdm import tqdm

# from rtmlib import Wholebody, Body
# from rtmlib.visualization.draw import draw_mmpose

from .mmpose_wraper import BodyPose, draw_mmpose

coco17 = dict(
    name="coco17",
    keypoint_info={
        0: dict(name="nose", id=0, color=[51, 153, 255], swap=""),
        1: dict(name="left_eye", id=1, color=[51, 153, 255], swap="right_eye"),
        2: dict(name="right_eye", id=2, color=[51, 153, 255], swap="left_eye"),
        3: dict(name="left_ear", id=3, color=[51, 153, 255], swap="right_ear"),
        4: dict(name="right_ear", id=4, color=[51, 153, 255], swap="left_ear"),
        5: dict(name="left_shoulder", id=5, color=[0, 255, 0], swap="right_shoulder"),
        6: dict(name="right_shoulder", id=6, color=[255, 128, 0], swap="left_shoulder"),
        7: dict(name="left_elbow", id=7, color=[0, 255, 0], swap="right_elbow"),
        8: dict(name="right_elbow", id=8, color=[255, 128, 0], swap="left_elbow"),
        9: dict(name="left_wrist", id=9, color=[0, 255, 0], swap="right_wrist"),
        10: dict(name="right_wrist", id=10, color=[255, 128, 0], swap="left_wrist"),
        11: dict(name="left_hip", id=11, color=[0, 255, 0], swap="right_hip"),
        12: dict(name="right_hip", id=12, color=[255, 128, 0], swap="left_hip"),
        13: dict(name="left_knee", id=13, color=[0, 255, 0], swap="right_knee"),
        14: dict(name="right_knee", id=14, color=[255, 128, 0], swap="left_knee"),
        15: dict(name="left_ankle", id=15, color=[0, 255, 0], swap="right_ankle"),
        16: dict(name="right_ankle", id=16, color=[255, 128, 0], swap="left_ankle"),
    },
    skeleton_info={
        0: dict(link=("left_ankle", "left_knee"), id=0, color=[0, 255, 0]),
        1: dict(link=("left_knee", "left_hip"), id=1, color=[0, 255, 0]),
        2: dict(link=("right_ankle", "right_knee"), id=2, color=[255, 128, 0]),
        3: dict(link=("right_knee", "right_hip"), id=3, color=[255, 128, 0]),
        4: dict(link=("left_hip", "right_hip"), id=4, color=[51, 153, 255]),
        5: dict(link=("left_shoulder", "left_hip"), id=5, color=[51, 153, 255]),
        6: dict(link=("right_shoulder", "right_hip"), id=6, color=[51, 153, 255]),
        7: dict(link=("left_shoulder", "right_shoulder"), id=7, color=[51, 153, 255]),
        8: dict(link=("left_shoulder", "left_elbow"), id=8, color=[0, 255, 0]),
        9: dict(link=("right_shoulder", "right_elbow"), id=9, color=[255, 128, 0]),
        10: dict(link=("left_elbow", "left_wrist"), id=10, color=[0, 255, 0]),
        11: dict(link=("right_elbow", "right_wrist"), id=11, color=[255, 128, 0]),
        12: dict(link=("left_eye", "right_eye"), id=12, color=[51, 153, 255]),
        13: dict(link=("nose", "left_eye"), id=13, color=[51, 153, 255]),
        14: dict(link=("nose", "right_eye"), id=14, color=[51, 153, 255]),
        15: dict(link=("left_eye", "left_ear"), id=15, color=[51, 153, 255]),
        16: dict(link=("right_eye", "right_ear"), id=16, color=[51, 153, 255]),
        17: dict(link=("left_ear", "left_shoulder"), id=17, color=[51, 153, 255]),
        18: dict(link=("right_ear", "right_shoulder"), id=18, color=[51, 153, 255]),
    },
)

skeleton_dict = {
    "coco17": coco17,
}


def draw_skeleton(
    img, keypoints, scores, kpt_thr=0.5, radius=2, line_width=2, skeleton_type="coco17"
):
    # num_keypoints = keypoints.shape[1]

    keypoint_info = skeleton_dict[skeleton_type]["keypoint_info"]
    skeleton_info = skeleton_dict[skeleton_type]["skeleton_info"]

    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :, :]

    num_instance = keypoints.shape[0]

    for i in range(num_instance):
        img = draw_mmpose(
            img,
            keypoints[i],
            scores[i],
            keypoint_info,
            skeleton_info,
            kpt_thr,
            radius,
            line_width,
        )

    return img


def get_rtmpose_model(device="cpu"):
    backend = "onnxruntime"  # opencv, onnxruntime, openvino

    # openpose_skeleton = False  # True for openpose-style, False for mmpose-style

    # model = Wholebody(
    #     to_openpose=openpose_skeleton,
    #     mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
    #     backend=backend, device=device)
    model = BodyPose(backend=backend, device=device)
    return model


def calculate_iou_batch(bboxes1, bboxes2):
    # Expand dimensions to allow broadcasting
    bboxes1 = bboxes1[:, np.newaxis, :]
    bboxes2 = bboxes2[np.newaxis, :, :]

    # Extract coordinates of the rectangles
    x1, y1, x2, y2 = np.split(bboxes1, 4, axis=2)
    x3, y3, x4, y4 = np.split(bboxes2, 4, axis=2)

    # Calculate the coordinates of the intersection rectangle
    x_intersection = np.maximum(x1, x3)
    y_intersection = np.maximum(y1, y3)
    width_intersection = np.maximum(0, np.minimum(x2, x4) - x_intersection)
    height_intersection = np.maximum(0, np.minimum(y2, y4) - y_intersection)

    # Calculate the area of intersection
    area_intersection = width_intersection * height_intersection

    # Calculate the area of both rectangles
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)

    # Calculate the union area
    area_union = area_box1 + area_box2 - area_intersection

    # Calculate the IoU
    iou = area_intersection / area_union

    return iou


@torch.no_grad()
def generate_2dkpts(model, img, bboxes_xyxy=None, prev_bbox=None, return_single=True):
    not_found = False
    if bboxes_xyxy is None:
        bboxes_xyxy = model.det_model(img)
        if prev_bbox is not None:
            # simple tracking
            if bboxes_xyxy.shape[0] == 0:
                bboxes_xyxy = prev_bbox[None]
                not_found = True
            else:
                iou = calculate_iou_batch(prev_bbox[None], bboxes_xyxy)
                max_idx = int(iou.argmax(axis=1))
                if iou[:, max_idx] < 0.2:
                    bboxes_xyxy = prev_bbox[None]
                    not_found = True
                else:
                    bboxes_xyxy = bboxes_xyxy[max_idx][None]
                    not_found = False

    keypoints, scores = model.pose_model(img, bboxes=bboxes_xyxy)
    if return_single:
        mean_scores = np.mean(scores, axis=1)
        max_idx = mean_scores.argmax()
        keypoints = keypoints[max_idx][None]
        scores = scores[max_idx][None]
        bboxes_xyxy = bboxes_xyxy[max_idx][None]

    if not_found:
        xmin = keypoints[..., 0].min()
        ymin = keypoints[..., 1].min()
        xmax = keypoints[..., 0].max()
        ymax = keypoints[..., 1].max()
        bboxes_xyxy = np.array([xmin, ymin, xmax, ymax])

    # scores = scores * 0.1
    keypoints = np.concatenate([keypoints, scores[..., None]], axis=-1)

    keypoints = keypoints.reshape([-1, 3])

    if bboxes_xyxy.ndim == 2 and bboxes_xyxy.shape[0] > 1:
        area = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) * (
            bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]
        )
        max_idx = area.argmax()
        bboxes_xyxy = bboxes_xyxy[[max_idx]]

    bboxes_xyxy = bboxes_xyxy.reshape([4])

    return keypoints, bboxes_xyxy


@torch.no_grad()
def generate_2dkpts_dir(model, img_files, bboxes_list=None):
    keypoints_list = []
    if bboxes_list is None:
        bboxes_list = []
        use_existing_box = False
    else:
        use_existing_box = True

    prev_bbox = None
    for n, img_file in tqdm(enumerate(img_files)):
        img = cv2.imread(img_file)
        if use_existing_box:
            keypoints, _ = generate_2dkpts(model, img, bboxes_xyxy=bboxes_list[n][None])
        else:
            keypoints, bboxes = generate_2dkpts(model, img, prev_bbox=prev_bbox)
            prev_bbox = bboxes
            bboxes_list.append(bboxes)
        keypoints_list.append(keypoints)

    return keypoints_list, bboxes_list
