import sys
import torch
import numpy as np
import time
import os
import os.path as osp
import subprocess
sys.path.append('./')
from motiondiff.data_pipeline.utils.skeleton import Skeleton, load_bvh_animation
from motiondiff.utils.hybrik import batch_rigid_transform
from motiondiff.data_pipeline.scripts.vis_bones import SMPLVisualizer
import pandas as pd
import argparse
import wandb
import csv
# create start and end arg
argparser = argparse.ArgumentParser()
argparser.add_argument('--start', type=int, default=0)
argparser.add_argument('--end', type=int, default=50)
argparser.add_argument('--job_id', type=int, default=0)
args = argparser.parse_args()

bvh_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw_v14'
out_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_verify_mirrors_v14'
# bvh_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_full_raw'
# out_dir = '/lustre/fs5/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/bones_verify_mirrors_v3'
# bvh_dir = '../bones_data/foundation_data'
# out_dir = './out/bones_mirror_v3'

# meta_file = 'data/bones_meta.csv'
# meta_file = f'{bvh_dir}/Metadata - 350 000 moves.csv'
# meta = pd.read_csv(meta_file)

def break_sentence(sentence, max_chars_per_line=160):
    words = sentence.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars_per_line:  # Adding 1 for space
            if current_line:  # If not the first word in the line, add a space
                current_line += " "
            current_line += word
        else:
            lines.append(current_line)
            current_line = ' ' * 4 + word
    # Add the remaining part of the sentence
    if current_line:
        lines.append(current_line)
    return '\n'.join(lines)

def hstack_videos(video1_path, video2_path, out_path, crf=15, verbose=True, text1=None, text2=None, text_size=12, vid_width=800):
    FFMPEG_PATH = '/usr/bin/ffmpeg' if osp.exists('/usr/bin/ffmpeg') else 'ffmpeg'
    if text1 is not None or text2 is not None:
        write_text = True
        tmp_file = f'{osp.splitext(out_path)[0]}_tmp.mp4'
    else:
        write_text = False
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cmd = [FFMPEG_PATH, '-y', '-i', video1_path, '-i', video2_path, '-filter_complex', 'hstack,format=yuv420p', 
           '-vcodec', 'libx264', '-crf', f'{crf}', tmp_file if write_text else out_path]
    if not verbose:
        cmd += ['-hide_banner', '-loglevel', 'error']
    subprocess.run(cmd)

    if write_text:
        text1 = '' if text1 is None else text1
        text2 = '' if text2 is None else text2
        font_file = '/usr/share/fonts/truetype/lato/Lato-Regular.ttf'
        draw_str = f"drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={'black'}:text='{text1}':x=0:y=20:line_spacing={text_size//2}" + \
                  f",drawtext=fontsize={text_size}:fontfile={font_file}:fontcolor={'red'}:text='{text2}':x={vid_width}:y=20:line_spacing={text_size//2}"
        cmd = [FFMPEG_PATH, '-i', tmp_file, '-y', '-vf', draw_str, '-c:a', 'copy', out_path]
        if not verbose:
            cmd += ['-hide_banner', '-loglevel', 'error']
        subprocess.run(cmd)
        os.remove(tmp_file)

def load_motion(bvh_fname, row):
    bvh_file = f'{bvh_dir}/{bvh_fname}'
    texts = [
        break_sentence('1. ' + row['content_natural_desc_1']),
        break_sentence('2. ' + row['content_natural_desc_2']),
        break_sentence('3. ' + row['content_natural_desc_3']),
        break_sentence('4. ' + row['content_short_description']),
        break_sentence('5. ' + row['content_short_description_2']),
        break_sentence('6. ' + row['content_technical_description']),
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
    # NOTE: have to negate x so that left/right annotations align with the rendered motion
    posed_joints = torch.stack([-posed_joints[:, :, 0], posed_joints[:, :, 2], posed_joints[:, :, 1]], dim=-1)

    return posed_joints, texts, parents

non_mirror_names = []
name2row = dict()
# meta_csv = os.path.join(bvh_dir, 'Metadata - 350 000 moves.csv')
meta_csv = os.path.join(bvh_dir, 'metadata_240527_v014.csv')
print('Loading metadata...')
with open(meta_csv, 'r') as f:
    reader = csv.DictReader(f)
    for ri, row in enumerate(reader):
        move_name = row['move_org_name']
        bvh_path = row['move_bvh_path']
        name2row[move_name] = row
        if os.path.splitext(bvh_path)[0][-2:] != '_M':
            non_mirror_names.append(move_name)

print(f'{len(name2row)} entries in metadata.')
print(f'Processing {len(non_mirror_names)} non-mirrored entries...')

# # check if all have a mirror
# print(f'Checking all mirrors are present...')
# for i in range(args.start, args.end):
#     if i > len(non_mirror_names)-1:
#         break
#     cur_name = non_mirror_names[i]
#     row = name2row[cur_name]

#     # load non-mirrored
#     bvh_fname = row['move_bvh_path']
#     mirrored_name = cur_name + '_M'
#     if mirrored_name not in name2row:
#         print(f'No mirrored bvh found for {bvh_fname}!')
#         error_path = f'{out_dir}/{i // 1000:04d}/missing_mirrors.txt'
#         os.makedirs(os.path.dirname(error_path), exist_ok=True)
#         with open(error_path, 'a') as f:
#             f.write(f'{bvh_fname}\n')

print(f'Rendering mirrors...')
for i in range(args.start, args.end, 100):
    if i > len(non_mirror_names)-1:
        break
    try:
        t0 = time.time()
        cur_name = non_mirror_names[i]
        row = name2row[cur_name]

        # load non-mirrored
        bvh_fname = row['move_bvh_path']
        print(bvh_fname)
        posed_joints, texts, parents = load_motion(bvh_fname, row)
        # print(texts)
        # load mirrored
        mirrored_name = cur_name + '_M'
        if mirrored_name not in name2row:
            continue
        mirror_row = name2row[mirrored_name]
        mirror_bvh_fname = mirror_row['move_bvh_path']
        print(mirror_bvh_fname)
        mirror_joints, mirror_texts, mirror_parents = load_motion(mirror_bvh_fname, mirror_row)
        # print(mirror_texts)

        vis = SMPLVisualizer(joint_parents=parents, distance=7, elevation=10, verbose=False, color_left_right=True, display_num=f':{args.job_id + 500}')
        smpl_seq = {
            'gt':{
                'joints_pos': posed_joints.float() * 0.01,
            }
        }
        mirror_smpl_seq = {
            'gt':{
                'joints_pos': mirror_joints.float() * 0.01,
            }
        }
        take_name = os.path.basename(bvh_fname)[:-4]
        video_path_f = f'out/tmp/{take_name}.mp4'
        video_path_s = f'out/tmp/{take_name}_M.mp4'
        frame_dir = f'out/tmp/frames/{take_name}'
        video_path = f'{out_dir}/{i // 5000:04d}/{i:06d}-{take_name}.mp4'

        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        os.makedirs(os.path.dirname(video_path_f), exist_ok=True)

        vis.save_animation_as_video(video_path_f, init_args={'smpl_seq': smpl_seq, 'mode': 'gt', 'camera': {'azimuth': -180}}, window_size=(800, 800), frame_dir=frame_dir, fps=120, crf=15)
        vis.save_animation_as_video(video_path_s, init_args={'smpl_seq': mirror_smpl_seq, 'mode': 'gt'}, window_size=(800, 800), frame_dir=frame_dir, fps=120, crf=15)

        hstack_videos(video_path_f, video_path_s, video_path, crf=15, verbose=False, text1='\n'.join(texts), text2='\n'.join(mirror_texts), text_size=10, vid_width=800)

        os.remove(video_path_f)
        os.remove(video_path_s)

        print(f'{i}/({args.start}, {args.end}) processed {bvh_fname}, time: {time.time() - t0:.2f}')
    except Exception as e:
        print(f'Error in {bvh_fname}: {e}')
        continue