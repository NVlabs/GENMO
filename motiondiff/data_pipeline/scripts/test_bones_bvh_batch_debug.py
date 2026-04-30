import os
import os.path as osp
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.append("./")
import argparse

import pandas as pd

from motiondiff.data_pipeline.scripts.vis_bones import SMPLVisualizer
from motiondiff.data_pipeline.utils.skeleton import Skeleton, load_bvh_animation
from motiondiff.utils.hybrik import batch_rigid_transform

# create start and end arg
argparser = argparse.ArgumentParser()
argparser.add_argument("--start", type=int, default=0)
argparser.add_argument("--end", type=int, default=10)
argparser.add_argument("--job_id", type=int, default=0)
args = argparser.parse_args()

bvh_dir = "/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw"
out_dir = "out/bones/vis_videos_v1"
tmp_dir = "out/bones/tmp"
meta_file = "data/bones_meta.csv"
# meta_file = f'{bvh_dir}/Metadata - 350 000 moves.csv'
meta = pd.read_csv(meta_file)


def break_sentence(sentence, max_chars_per_line=160):
    words = sentence.split()
    lines = []
    current_line = ""
    for word in words:
        if (
            len(current_line) + len(word) + 1 <= max_chars_per_line
        ):  # Adding 1 for space
            if current_line:  # If not the first word in the line, add a space
                current_line += " "
            current_line += word
        else:
            lines.append(current_line)
            current_line = " " * 4 + word
    # Add the remaining part of the sentence
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)


def hstack_videos(
    video1_path,
    video2_path,
    out_path,
    crf=15,
    verbose=True,
    text1=None,
    text2=None,
    text_color="black",
    text_size=20,
):
    FFMPEG_PATH = "/usr/bin/ffmpeg" if osp.exists("/usr/bin/ffmpeg") else "ffmpeg"
    if not (text1 is None):
        write_text = True
        tmp_file = f"{osp.splitext(out_path)[0]}_tmp.mp4"
    else:
        write_text = False
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i",
        video1_path,
        "-i",
        video2_path,
        "-filter_complex",
        "hstack,format=yuv420p",
        "-vcodec",
        "libx264",
        "-crf",
        f"{crf}",
        tmp_file if write_text else out_path,
    ]
    if not verbose:
        cmd += ["-hide_banner", "-loglevel", "error"]
    subprocess.run(cmd)

    if write_text:
        font_file = "/usr/share/fonts/truetype/lato/Lato-Regular.ttf"
        text1 = text1.replace(",", "\,")
        text1 = text1.replace(":", "\:")
        text1 = text1.replace("'", "\u2019")
        draw_str = f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={text_color}:text='{text1}':x=0:y=20:line_spacing={text_size // 2}"
        cmd = [
            FFMPEG_PATH,
            "-i",
            tmp_file,
            "-y",
            "-vf",
            draw_str,
            "-c:a",
            "copy",
            out_path,
        ]
        if not verbose:
            cmd += ["-hide_banner", "-loglevel", "error"]
        subprocess.run(cmd)
        os.remove(tmp_file)


vis = None

i = 321420
t0 = time.time()
row = meta.iloc[i + 1]
bvh_fname = row["move_bvh_path"]
bvh_file = f"{bvh_dir}/{bvh_fname}"
take_name = os.path.basename(bvh_fname)[:-4]
video_path = f"{out_dir}/{i // 1000:04d}/{i:06d}-{take_name}.mp4"
# if os.path.exists(video_path):
#     print(f'skip {video_path}')
#     continue

print(
    f"{i}/({args.start}, {args.end}) start processing {bvh_fname}, time: {time.time() - t0:.2f}"
)

texts = [
    break_sentence(str(row["content_natural_desc_1"])),
    break_sentence(str(row["content_natural_desc_2"])),
    break_sentence(str(row["content_natural_desc_3"])),
    break_sentence(str(row["content_technical_description"])),
    break_sentence(str(row["content_short_description"])),
]
skeleton = Skeleton()
skeleton.load_from_bvh(bvh_file)
root_trans, joint_rot_mats = load_bvh_animation(bvh_file, skeleton)
parent_indices = skeleton.get_parent_indices()
joints = skeleton.get_neutral_joints()

rot_mats = torch.tensor(joint_rot_mats)
joints = torch.tensor(joints).unsqueeze(0).repeat(rot_mats.shape[0], 1, 1)
parents = torch.LongTensor(parent_indices)
joints -= joints[:, [0]]

posed_joints, global_rot_mat = batch_rigid_transform(rot_mats, joints, parents)
posed_joints += torch.tensor(root_trans).unsqueeze(1)
posed_joints = torch.stack(
    [-posed_joints[:, :, 0], posed_joints[:, :, 2], posed_joints[:, :, 1]], dim=-1
)

if vis is None:
    vis = SMPLVisualizer(
        joint_parents=parents,
        distance=7,
        elevation=10,
        verbose=False,
        display_num=f":{args.job_id + 10}",
    )

smpl_seq = {
    "gt": {
        "joints_pos": posed_joints.float() * 0.01,
    }
}

video_path_f = f"{tmp_dir}/{i // 1000:04d}/{i:06d}-{take_name}_front.mp4"
video_path_s = f"{tmp_dir}/{i // 1000:04d}/{i:06d}-{take_name}_side.mp4"
frame_dir = f"{tmp_dir}/{i // 1000:04d}/{i:06d}-{take_name}_frames"
os.makedirs(os.path.dirname(video_path), exist_ok=True)
os.makedirs(os.path.dirname(video_path_f), exist_ok=True)
vis.save_animation_as_video(
    video_path_f,
    init_args={"smpl_seq": smpl_seq, "mode": "gt", "camera": {"azimuth": 0}},
    window_size=(800, 800),
    frame_dir=frame_dir,
    fps=120,
    crf=15,
)
vis.save_animation_as_video(
    video_path_s,
    init_args={"smpl_seq": smpl_seq, "mode": "gt", "camera": {"azimuth": -90}},
    window_size=(800, 800),
    frame_dir=frame_dir,
    fps=120,
    crf=15,
)
hstack_videos(
    video_path_f,
    video_path_s,
    video_path,
    crf=15,
    verbose=False,
    text1="\n".join(texts),
)
# os.remove(video_path_f)
# os.remove(video_path_s)
print(
    f"{i}/({args.start}, {args.end}) processed {bvh_fname}, time: {time.time() - t0:.2f}"
)
