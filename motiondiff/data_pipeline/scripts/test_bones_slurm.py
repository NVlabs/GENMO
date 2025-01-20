import json
import glob
import subprocess
import os
import time
import argparse


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
    
    motion_start_index = job_index * num_motion_per_job + args.motion_start
    motion_end_index = (job_index + 1) * num_motion_per_job + args.motion_start

    cmd = f'python motiondiff/data_pipeline/scripts/test_bones_bvh_batch.py --start {motion_start_index} --end {motion_end_index} --job_id {job_index}'
    if args.wandb:
        cmd += ' --wandb'
    print(cmd)
    if args.local:
        subprocess.run(cmd, shell=True)
    else:
        job_cmd = f'mkdir -p /repo; cd /repo; git clone ssh://git@gitlab-master.nvidia.com:12051/ediff-i-motion/physdiff.git; cd physdiff; pip install lxml bvh pyrender pyvista==0.38.6; apt-get update; apt install -y libgl1-mesa-glx xvfb; {cmd}'
        
        ssh_cmd = \
        f'submit_job --partition {args.partition} --duration 24 --nodes 1 --cpu 8 --exclude_hosts cpu-00016,cpu-00038,cpu-00039,cpu-00026,cpu-00009,cpu-00025 --email_mode fail --image /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/docker/motiondiff_0.3.5.sqsh' + \
        f' --name "{total_jobs}bones.{job_index:03d}-{total_jobs}" --command "{job_cmd}"'
        print(ssh_cmd)

        subprocess.run(f"ssh yey@cs-oci-ord-login-01 '{ssh_cmd}'", shell=True)
    