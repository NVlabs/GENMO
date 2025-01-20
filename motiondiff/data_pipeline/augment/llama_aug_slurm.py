import json
import glob
import subprocess
import os
import time
import argparse
import re

with open('out/job_queue.txt', 'r') as file:
    job_info = file.read()
# Use regular expression to find all occurrences of the pattern
job_indices = re.findall(r'50llama-bones\.(\d+)-', job_info)

# Convert the extracted indices to integers
skip_index = [int(index) for index in job_indices]
print(skip_index)


parser = argparse.ArgumentParser(description='Process bones')
parser.add_argument('-pt', '--partition', default='cpu', help='slurm partition')
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=None)
parser.add_argument('--motion_start', type=int, default=0)
parser.add_argument('--motion_end', type=int, default=350000)
parser.add_argument('--num_jobs', type=int, default=100)
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

    cmd = f'python motiondiff/data_pipeline/augment/llama_aug.py --start {motion_start_index} --end {motion_end_index}'
    if args.wandb:
        cmd += ' --wandb'
    print(cmd)
    if args.local:
        subprocess.run(cmd, shell=True)
    else:
        job_cmd = f'mkdir -p /repo; cd /repo; git clone ssh://git@gitlab-master.nvidia.com:12051/ediff-i-motion/physdiff.git; cd physdiff; git checkout bones; pip install --upgrade openai; {cmd}'
        
        ssh_cmd = \
        f'submit_job --partition {args.partition} --duration 24 --nodes 1 --cpu 16 --account nvr_torontoai_humanmotionfm --exclude_hosts cpu-00016,cpu-00038,cpu-00039,cpu-00026,cpu-00009,cpu-00025,cpu-00036,cpu-00028,cpu-00029 --email_mode fail --image /lustre/fsw/portfolios/nvr/projects/nvr_lpr_digitalhuman/docker/dh_foundation_0.1.sqsh' + \
        f' --name "{total_jobs}llama-bones.{job_index:03d}-{total_jobs}" --command "{job_cmd}"'
        print(ssh_cmd)

        subprocess.run(f"ssh yey@cs-oci-ord-login-01 '{ssh_cmd}'", shell=True)
    