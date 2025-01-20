import json
import glob
import subprocess
import os
import time
import argparse
import re

# with open('out/job_queue.txt', 'r') as file:
#     job_info = file.read()
# # Use regular expression to find all occurrences of the pattern
# job_indices = re.findall(r'50llama-bones\.(\d+)-', job_info)

# # Convert the extracted indices to integers
# skip_index = [int(index) for index in job_indices]
# print(skip_index)
skip_index = []

parser = argparse.ArgumentParser(description='Process bones')
parser.add_argument('-pt', '--partition', default='cpu', help='slurm partition')
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=None)
parser.add_argument('--motion_start', type=int, default=0)
parser.add_argument('--motion_end', type=int, default=350000)
parser.add_argument('--num_jobs', type=int, default=200)
parser.add_argument('--local', action='store_true')
parser.add_argument('--wandb', action='store_true')
args = parser.parse_args()

total_num_motion = args.motion_end - args.motion_start
total_jobs = args.num_jobs
num_motion_per_job = total_num_motion // total_jobs
if args.end_index is None:
    args.end_index = total_jobs

for job_index in range(args.start_index, args.end_index):
    if job_index in skip_index:
        continue
    
    motion_start_index = job_index * num_motion_per_job + args.motion_start
    motion_end_index = (job_index + 1) * num_motion_per_job + args.motion_start

    cmd = f'python motiondiff/data_pipeline/augment/llama_aug_v5.py --start {motion_start_index} --end {motion_end_index}'
    if args.wandb:
        cmd += ' --wandb'
    print(cmd)
    if args.local:
        subprocess.run(cmd, shell=True)
    else:
        job_cmd = f'mkdir -p /repo; cd /repo; git clone ssh://git@gitlab-master.nvidia.com:12051/ediff-i-motion/physdiff.git; cd physdiff; git checkout bones; pip install --upgrade openai; {cmd}'
        
        ssh_cmd = \
        f'submit_job --partition {args.partition} --duration 24 --nodes 1 --cpu 16 --mem 32 --account nvr_torontoai_humanmotionfm --exclude_hosts cpu-00016,cpu-00038,cpu-00039,cpu-00026,cpu-00009,cpu-00025,cpu-00036,cpu-00028,cpu-00029,cpu-00033,cpu-00031,cpu-00021 --email_mode fail --image /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/docker/motiondiff_0.3.5.sqsh' + \
        f' --name "{total_jobs}llama-bones.{job_index:03d}-{total_jobs}" --command "{job_cmd}"'
        print(ssh_cmd)

        subprocess.run(f"ssh drempe@cs-oci-ord-login-01 '{ssh_cmd}'", shell=True)

# run the ones that got missed somehow
# missing_idx = [9922, 9958, 10020, 10031, 10043, 10197, 10200, 10206, 10237, 10242, 10244, 10263, 11972, 11985, 11994, 12018, 12050, 12064, 12076, 12085, 12088, 12100, 12114, 12116, 81846, 81865, 81926, 81972, 98986, 135454, 135476, 135484, 135485, 135493, 135498, 135505, 135515, 135523, 135531, 135533, 135540, 135545, 135555, 135565, 137694, 137757, 137761, 137791, 137793, 137802, 137818, 137829, 137841, 137869, 137876, 137884, 137897, 137899, 137964, 150347, 150348, 150351, 150352, 150353, 150354, 158054, 169251, 169262, 169308, 169318, 169322, 169328, 169330, 169332, 169336, 169340, 169342, 169344, 169580, 169582, 170589, 170610, 170614, 170622, 170627, 170633, 170647, 170652, 170697, 170844, 198653, 215620, 215636, 215656, 215692, 215699, 215705, 215724, 215725, 215727, 215732, 215739, 215743, 215789, 215815, 215926, 218020, 218050, 218374, 218382, 218384, 218398, 218404, 218414, 218417, 218428, 218430, 218445, 218450, 218519, 218524, 228889, 228897, 228911, 228929, 228933, 228953, 228986, 229017, 229038, 229050, 229052, 229063, 229066, 229078, 229079, 230137, 230145, 230201, 230274, 230301, 230306, 230318, 230352, 230356, 230378, 230385, 230420, 230530, 230635, 230872, 243939, 243985, 244016, 244042, 244819, 244823, 244838, 244839, 244874, 244888, 244891, 244910, 244912, 244969, 245018, 246875, 247011, 247064, 247093, 247096, 247132, 247145, 247161, 247163, 247195, 247206, 247216, 247219, 247245, 247295, 264152, 264225, 264237, 264238, 264242, 264273, 264300, 264320, 264325, 264328, 264332, 264333, 264348, 264380, 264390, 265622, 266437, 266461, 266540, 266567, 266574, 266613, 266633, 266645, 266657, 266669, 266693, 267552, 267557, 267561, 282810, 282811, 282819, 282831, 282840, 282843, 282847, 282854, 282859, 282867, 282871, 282885, 282891, 282903, 282905, 285267, 285304, 285339, 285350, 285352, 285355, 285356, 285364, 285374, 285375, 285378, 285383, 285388, 285395, 285401, 329095, 329098, 329102, 329103, 329106, 329114, 329126, 329140, 329756, 332667, 332669, 332674, 332675, 332677, 332681, 332683, 332689, 332691, 332699, 332705, 332714, 332723, 332729, 332735, 334134, 334154, 334155, 334170, 334171, 334176, 334178, 334179, 334186, 334192, 334194, 334200, 334203, 334206, 334224, 339562, 339563, 349202]
# total_jobs = len(missing_idx)
# for cur_missing in missing_idx:        
#     motion_start_index = cur_missing #job_index * num_motion_per_job + args.motion_start
#     motion_end_index = cur_missing + 1 #(job_index + 1) * num_motion_per_job + args.motion_start

#     cmd = f'python motiondiff/data_pipeline/augment/llama_aug_v5.py --start {motion_start_index} --end {motion_end_index}'
#     if args.wandb:
#         cmd += ' --wandb'
#     print(cmd)
#     if args.local:
#         subprocess.run(cmd, shell=True)
#     else:
#         job_cmd = f'mkdir -p /repo; cd /repo; git clone ssh://git@gitlab-master.nvidia.com:12051/ediff-i-motion/physdiff.git; cd physdiff; git checkout bones; pip install --upgrade openai; {cmd}'
        
#         ssh_cmd = \
#         f'submit_job --partition {args.partition} --duration 24 --nodes 1 --cpu 16 --mem 32 --account nvr_torontoai_humanmotionfm --exclude_hosts cpu-00016,cpu-00038,cpu-00039,cpu-00026,cpu-00009,cpu-00025,cpu-00036,cpu-00028,cpu-00029,cpu-00033,cpu-00031,cpu-00021 --email_mode fail --image /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/docker/motiondiff_0.3.5.sqsh' + \
#         f' --name "{total_jobs}llama-bones.{motion_start_index:03d}-{total_jobs}" --command "{job_cmd}"'
#         print(ssh_cmd)

#         subprocess.run(f"ssh drempe@cs-oci-ord-login-01 '{ssh_cmd}'", shell=True)