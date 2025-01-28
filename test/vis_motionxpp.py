import os

import cv2
import torch

from motiondiff.models.mdm.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
from motiondiff.utils.vis_scenepic import ScenepicVisualizer

motion_files = torch.load(
    "inputs/MotionXpp/hmr4d_support/motionxpp_smplxposev3_aligned.pth"
)

device = torch.device("cuda")
sp_visualizer = ScenepicVisualizer(
    "/home/jiefengl/git/physdiff_megm/data/smpl_data", device=device
)
smpl_dict = sp_visualizer.smpl_dict


def draw_keypoints(body_kpts, img):
    for kpt in body_kpts:
        cv2.circle(img, (int(kpt[0]), int(kpt[1])), 2, (0, 0, 255), -1)
    return img


for idx, vid in enumerate(motion_files):
    motion_data = motion_files[vid]
    smpl_layer = smpl_dict["neutral"]
    subset = motion_data["subset"]
    if subset == "music":
        continue

    vname = motion_data["file_name"]
    video_path = f"/mnt/disk3/motion-x++/video/{subset}/{vname}.mp4"
    assert os.path.exists(video_path), f"{video_path} not found"

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = f"vis/motionxpp/{subset}/motionxpp_{idx:05d}.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tot_frames = motion_data["pose"].shape[0]
    for i in range(tot_frames):
        ret, frame = cap.read()
        kpt = motion_data["body_kpts"][i]
        frame = draw_keypoints(kpt, frame)
        out.write(frame)

    out.release()
    cap.release()
    continue

    body_pose = motion_data["pose"].reshape(-1, 21, 3)
    global_orient = motion_data["global_orient"].reshape(-1, 1, 3)
    beta = motion_data["beta"]
    trans = motion_data["trans"]

    pose_pad = torch.cat([body_pose, torch.zeros_like(body_pose[:, :2])], dim=1)
    smpl_output = smpl_layer(
        betas=beta.to(device),
        global_orient=global_orient.to(device),
        body_pose=pose_pad.to(device),
        transl=trans.to(device),
        orig_joints=True,
        pose2rot=True,
    )
    j3d = smpl_output.joints.detach()

    smpl_seq = {
        "text": "",
        "joints_pos": j3d,
        # "joints_pos": torch.from_numpy(data_jts[:, :24]).float(),
    }

    html_file = f"vis/motionxpp/{subset}/motionxpp_{idx:05d}.html"
    os.makedirs(os.path.dirname(html_file), exist_ok=True)
    sp_visualizer.vis_smpl_scene(smpl_seq, html_file)
    print(vid)
