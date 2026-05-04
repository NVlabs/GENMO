# SPDX-License-Identifier: Apache-2.0
"""YOLOX person detection (ONNX) + ByteTrack multi-object tracking.

YOLOX — Apache-2.0 (Megvii)
ByteTrack — MIT (Megvii)

This module is self-contained: it only depends on numpy, scipy, cv2,
and onnxruntime.  No external YOLOX / ByteTrack packages are required.
"""

from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Pre-trained YOLOX-X on HumanArt (COCO-person + art data) — good for
# real-world person detection.  Hosted on OpenMMLab CDN.
YOLOX_X_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip"
)
_CACHE_DIR = Path(
    os.getenv("TORCH_HOME", os.path.expanduser("~/.cache/rtmlib"))
) / "hub" / "checkpoints"


# ---------------------------------------------------------------------------
#  ONNX model download helper
# ---------------------------------------------------------------------------

def _download_yolox_onnx(url: str = YOLOX_X_URL) -> str:
    """Download and unzip YOLOX ONNX if not cached. Returns onnx path."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parts = urlparse(url)
    zip_name = os.path.basename(parts.path)
    onnx_name = zip_name.rsplit(".", 1)[0] + ".onnx"
    onnx_path = _CACHE_DIR / onnx_name

    if onnx_path.exists():
        return str(onnx_path)

    zip_path = _CACHE_DIR / zip_name
    if not zip_path.exists():
        sys.stderr.write(f'Downloading YOLOX: "{url}" → {zip_path}\n')
        req = Request(url, headers={"User-Agent": "gem-yolox"})
        with urlopen(req) as resp, open(zip_path, "wb") as f:
            while True:
                buf = resp.read(8192)
                if not buf:
                    break
                f.write(buf)

    # Extract end2end.onnx from the zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("end2end.onnx"):
                with zf.open(member) as src, open(onnx_path, "wb") as dst:
                    dst.write(src.read())
                break
    zip_path.unlink(missing_ok=True)
    return str(onnx_path)


# ---------------------------------------------------------------------------
#  YOLOX ONNX Detector
# ---------------------------------------------------------------------------

class YOLOXDetector:
    """YOLOX person detector using ONNX Runtime.

    Parameters
    ----------
    onnx_path : str or None
        Path to YOLOX ONNX model.  If None, auto-downloads YOLOX-X.
    input_size : tuple
        Model input resolution (H, W).  Default (640, 640).
    score_thr : float
        Minimum person confidence threshold.
    nms_thr : float
        NMS IoU threshold.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        input_size: Tuple[int, int] = (640, 640),
        score_thr: float = 0.5,
        nms_thr: float = 0.45,
        device: str = "cuda",
    ):
        import onnxruntime as ort

        if onnx_path is None:
            onnx_path = _download_yolox_onnx()

        providers = []
        if device == "cuda":
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
        if sys.platform == "darwin":
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        active = self.sess.get_providers()
        print(f"[YOLOXDetector] Loaded {Path(onnx_path).name} (EP={active})")

        self.input_size = input_size  # (H, W)
        self.score_thr = score_thr
        self.nms_thr = nms_thr

    # -- public API ---------------------------------------------------------

    def detect(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect persons in a single BGR image.

        Returns
        -------
        boxes : ndarray (N, 4)  — xyxy in original image coords
        scores : ndarray (N,)
        """
        padded, ratio = self._preprocess(img_bgr)
        inp = padded.transpose(2, 0, 1)[None].astype(np.float32)
        ort_in = {self.sess.get_inputs()[0].name: inp}
        outputs = self.sess.run(None, ort_in)
        return self._postprocess(outputs[0], ratio)

    # -- internal -----------------------------------------------------------

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        ratio = min(self.input_size[0] / h, self.input_size[1] / w)
        resized = cv2.resize(
            img,
            (int(w * ratio), int(h * ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        padded = np.full(
            (self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8
        )
        padded[: resized.shape[0], : resized.shape[1]] = resized
        return padded, ratio

    def _postprocess(
        self, outputs: np.ndarray, ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        if outputs.shape[-1] == 5:
            # ONNX with NMS baked in: shape (1, N, 5) = [x1, y1, x2, y2, score]
            boxes = outputs[0, :, :4] / ratio
            scores = outputs[0, :, 4]
            mask = scores > self.score_thr
            return boxes[mask], scores[mask]

        # ONNX without NMS: shape (1, N, 85) for COCO
        # Decode grid predictions
        strides = [8, 16, 32]
        grids, exp_strides = [], []
        for s in strides:
            hs, ws = self.input_size[0] // s, self.input_size[1] // s
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            exp_strides.append(np.full((*grid.shape[:2], 1), s))
        grids = np.concatenate(grids, 1)
        exp_strides = np.concatenate(exp_strides, 1)

        outputs[..., :2] = (outputs[..., :2] + grids) * exp_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * exp_strides

        preds = outputs[0]
        # Convert cxcywh → xyxy
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = (cx - bw / 2) / ratio
        y1 = (cy - bh / 2) / ratio
        x2 = (cx + bw / 2) / ratio
        y2 = (cy + bh / 2) / ratio
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        obj_conf = preds[:, 4]
        cls_scores = preds[:, 5:]  # 80 COCO classes
        person_score = obj_conf * cls_scores[:, 0]  # class 0 = person

        mask = person_score > self.score_thr
        boxes_f = boxes_xyxy[mask]
        scores_f = person_score[mask]

        # NMS
        keep = _nms(boxes_f, scores_f, self.nms_thr)
        return boxes_f[keep], scores_f[keep]


# ---------------------------------------------------------------------------
#  NMS (numpy)
# ---------------------------------------------------------------------------

def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> List[int]:
    """Standard greedy NMS on (N, 4) xyxy boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= thr)[0] + 1]
    return keep


# ---------------------------------------------------------------------------
#  IoU helpers
# ---------------------------------------------------------------------------

def _iou_batch(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """Pairwise IoU between (M, 4) and (N, 4) xyxy boxes → (M, N)."""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


# ---------------------------------------------------------------------------
#  Kalman Filter for bounding box tracking (xyah state)
# ---------------------------------------------------------------------------

class _KalmanBoxTracker:
    """Tracks a single object with a Kalman filter over [x, y, a, h] state.

    State vector: [cx, cy, aspect_ratio, height, vx, vy, va, vh].
    Measurement:  [cx, cy, aspect_ratio, height].
    """

    _count = 0

    def __init__(self, bbox: np.ndarray, score: float):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        a = w / (h + 1e-7)

        self.x = np.array([cx, cy, a, h, 0, 0, 0, 0], dtype=np.float64)
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty on velocities

        # Process noise
        self.Q = np.eye(8) * 1.0
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.diag([1.0, 1.0, 10.0, 1.0])

        # Transition and observation matrices
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)  # position += velocity
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        self.id = _KalmanBoxTracker._count
        _KalmanBoxTracker._count += 1

        self.hits = 1
        self.time_since_update = 0
        self.score = score

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self._state_to_xyxy()

    def update(self, bbox: np.ndarray, score: float):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w / (h + 1e-7), h])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.hits += 1
        self.time_since_update = 0
        self.score = score

    def _state_to_xyxy(self) -> np.ndarray:
        cx, cy, a, h = self.x[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ---------------------------------------------------------------------------
#  ByteTrack Tracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """ByteTrack multi-object tracker (MIT license).

    Two-stage association:
      1. Match high-score detections to existing tracks (IoU).
      2. Match remaining low-score detections to unmatched tracks (IoU).
    No appearance model — pure motion-based, very fast.

    Parameters
    ----------
    high_thr : float
        Score threshold to split detections into high / low groups.
    match_thr : float
        IoU threshold for matching.
    max_lost : int
        Frames to keep a lost track before removal.
    """

    def __init__(
        self,
        high_thr: float = 0.5,
        match_thr: float = 0.3,
        max_lost: int = 30,
    ):
        self.high_thr = high_thr
        self.match_thr = match_thr
        self.max_lost = max_lost
        self.tracks: List[_KalmanBoxTracker] = []
        _KalmanBoxTracker._count = 0

    def update(
        self, boxes: np.ndarray, scores: np.ndarray
    ) -> List[Tuple[np.ndarray, int, float]]:
        """Process one frame of detections.

        Parameters
        ----------
        boxes : (N, 4) xyxy
        scores : (N,)

        Returns
        -------
        results : list of (box_xyxy, track_id, score)
        """
        # Predict all existing tracks
        for t in self.tracks:
            t.predict()

        # Split detections into high and low score groups
        high_mask = scores >= self.high_thr
        low_mask = ~high_mask & (scores > 0.1)

        high_boxes, high_scores = boxes[high_mask], scores[high_mask]
        low_boxes, low_scores = boxes[low_mask], scores[low_mask]

        # --- Stage 1: match high-score dets to tracks ---
        matched_t, matched_d, unmatched_tracks, unmatched_dets = self._match(
            self.tracks, high_boxes, self.match_thr
        )

        for t_idx, d_idx in zip(matched_t, matched_d):
            self.tracks[t_idx].update(high_boxes[d_idx], high_scores[d_idx])

        # --- Stage 2: match low-score dets to remaining tracks ---
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        if len(remaining_tracks) > 0 and len(low_boxes) > 0:
            mt2, md2, still_unmatched, _ = self._match(
                remaining_tracks, low_boxes, self.match_thr
            )
            for t_idx, d_idx in zip(mt2, md2):
                remaining_tracks[t_idx].update(low_boxes[d_idx], low_scores[d_idx])
            unmatched_tracks = [unmatched_tracks[i] for i in still_unmatched]

        # --- Create new tracks from unmatched high-score detections ---
        for d_idx in unmatched_dets:
            self.tracks.append(
                _KalmanBoxTracker(high_boxes[d_idx], high_scores[d_idx])
            )

        # --- Remove dead tracks ---
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_lost
        ]

        # --- Collect active results ---
        results = []
        for t in self.tracks:
            if t.time_since_update == 0:
                results.append((t._state_to_xyxy(), t.id, t.score))

        return results

    @staticmethod
    def _match(
        tracks: List[_KalmanBoxTracker],
        boxes: np.ndarray,
        thr: float,
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Hungarian matching between tracks and detections."""
        if len(tracks) == 0 or len(boxes) == 0:
            return [], [], list(range(len(tracks))), list(range(len(boxes)))

        from scipy.optimize import linear_sum_assignment

        track_boxes = np.array([t._state_to_xyxy() for t in tracks])
        iou_matrix = _iou_batch(track_boxes, boxes)
        cost = 1.0 - iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_t, matched_d = [], []
        unmatched_t = set(range(len(tracks)))
        unmatched_d = set(range(len(boxes)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= thr:
                matched_t.append(r)
                matched_d.append(c)
                unmatched_t.discard(r)
                unmatched_d.discard(c)

        return matched_t, matched_d, sorted(unmatched_t), sorted(unmatched_d)


# ---------------------------------------------------------------------------
#  High-level API: detect + track across a video
# ---------------------------------------------------------------------------

def detect_and_track(
    frames: np.ndarray,
    detector: YOLOXDetector,
    *,
    detect_every: int = 1,
    tracker_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run YOLOX detection + ByteTrack on a video and return the primary
    person's bounding boxes.

    Parameters
    ----------
    frames : (T, H, W, 3) uint8 RGB array
    detector : YOLOXDetector instance
    detect_every : int
        Run detection every N frames.  Intermediate frames use Kalman
        prediction only (faster but less accurate for fast motion).
        Set to 1 to detect every frame.
    tracker_kwargs : dict
        Extra kwargs for ByteTracker.

    Returns
    -------
    bbx_xyxy : (T, 4) float32 — primary person boxes (xyxy)
    track_ids : (T,) int — track ID per frame (-1 if no detection)
    """
    from tqdm import tqdm

    T, H, W, _ = frames.shape
    tracker = ByteTracker(**(tracker_kwargs or {}))

    # Per-frame results: list of (track_results)
    per_frame: List[List[Tuple[np.ndarray, int, float]]] = []

    for i in tqdm(range(T), desc="YOLOX+ByteTrack"):
        if i % detect_every == 0:
            img_bgr = frames[i][..., ::-1].copy()
            boxes, scores = detector.detect(img_bgr)
        else:
            # No detection — feed empty to tracker (Kalman prediction only)
            boxes, scores = np.empty((0, 4)), np.empty((0,))
        results = tracker.update(boxes, scores)
        per_frame.append(results)

    # Select primary track: the one with largest cumulative area
    return _select_primary_track(per_frame, T, H, W)


def _select_primary_track(
    per_frame: List[List[Tuple[np.ndarray, int, float]]],
    T: int,
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick the dominant track and extract per-frame boxes."""
    from collections import defaultdict

    # Accumulate area per track ID
    track_area: dict = defaultdict(float)
    track_frames: dict = defaultdict(list)

    for t, results in enumerate(per_frame):
        for box, tid, score in results:
            area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            track_area[tid] += area
            track_frames[tid].append((t, box, score))

    if not track_area:
        # No detections at all — fall back to full frame
        default = np.array([0.0, 0.0, W - 1, H - 1], dtype=np.float32)
        return np.tile(default, (T, 1)), np.full(T, -1, dtype=np.int32)

    # Primary track = largest cumulative area
    primary_id = max(track_area, key=track_area.get)
    primary_dict = {t: box for t, box, _ in track_frames[primary_id]}

    bbx_xyxy = np.zeros((T, 4), dtype=np.float32)
    track_ids = np.full(T, -1, dtype=np.int32)

    for t in range(T):
        if t in primary_dict:
            bbx_xyxy[t] = primary_dict[t]
            track_ids[t] = primary_id

    # Interpolate gaps (frames where primary track has no observation)
    _interpolate_gaps(bbx_xyxy, track_ids, primary_id, H, W)

    # Clamp
    bbx_xyxy[:, [0, 2]] = np.clip(bbx_xyxy[:, [0, 2]], 0, W - 1)
    bbx_xyxy[:, [1, 3]] = np.clip(bbx_xyxy[:, [1, 3]], 0, H - 1)

    return bbx_xyxy, track_ids


def _interpolate_gaps(
    bbx: np.ndarray,
    ids: np.ndarray,
    primary_id: int,
    H: int,
    W: int,
):
    """Fill gaps in the primary track via linear interpolation."""
    T = len(bbx)
    valid = ids == primary_id
    if valid.sum() == 0:
        bbx[:] = [0, 0, W - 1, H - 1]
        return

    valid_idx = np.where(valid)[0]

    # Fill leading frames with first valid box
    if valid_idx[0] > 0:
        bbx[: valid_idx[0]] = bbx[valid_idx[0]]
        ids[: valid_idx[0]] = primary_id

    # Fill trailing frames with last valid box
    if valid_idx[-1] < T - 1:
        bbx[valid_idx[-1] + 1 :] = bbx[valid_idx[-1]]
        ids[valid_idx[-1] + 1 :] = primary_id

    # Interpolate internal gaps
    for k in range(len(valid_idx) - 1):
        a, b = valid_idx[k], valid_idx[k + 1]
        if b - a > 1:
            for t in range(a + 1, b):
                alpha = (t - a) / (b - a)
                bbx[t] = (1 - alpha) * bbx[a] + alpha * bbx[b]
                ids[t] = primary_id
