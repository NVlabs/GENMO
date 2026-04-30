import argparse
import glob
import json
import os
import subprocess
import time

parser = argparse.ArgumentParser(description="Process bones")
parser.add_argument("-pt", "--partition", default="cpu", help="slurm partition")
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--end_index", type=int, default=2)
parser.add_argument("--local", action="store_true")
args = parser.parse_args()

# 150k motions / 5000 motions per job = 35 total jobs
#   only render 1 of every 100 motions within each of these jobs, i.e. 5000 / 100 = 50 video
num_motion_per_job = 5000
total_jobs = 35

for job_index in range(args.start_index, args.end_index):
    motion_start_index = job_index * num_motion_per_job
    motion_end_index = motion_start_index + num_motion_per_job

    cmd = f"python motiondiff/data_pipeline/scripts/test_bones_bvh_refcams.py --start {motion_start_index} --end {motion_end_index} --job_id {job_index}"
    print(cmd)
    if args.local:
        subprocess.run(cmd, shell=True)
    else:
        job_cmd = f"mkdir -p /repo; cd /repo; git clone ssh://git@gitlab-master.nvidia.com:12051/ediff-i-motion/physdiff.git; cd physdiff; pip install lxml bvh pyrender pyvista==0.38.6; apt-get update; apt install -y libgl1-mesa-glx xvfb; {cmd}"

        ssh_cmd = (
            f"submit_job --partition {args.partition} --duration 24 --nodes 1 --cpu 8 --image /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/docker/motiondiff_0.3.5.sqsh"
            + f' --name "bones_refcams.{job_index:04d}" --command "{job_cmd}"'
        )
        print(ssh_cmd)

        subprocess.run(f"ssh drempe@cs-oci-ord-login-01 '{ssh_cmd}'", shell=True)
