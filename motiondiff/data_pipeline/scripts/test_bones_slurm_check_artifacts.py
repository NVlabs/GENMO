import argparse
import glob
import json
import os
import subprocess
import time

parser = argparse.ArgumentParser(description="Process bones")
parser.add_argument("-pt", "--partition", default="cpu_long", help="slurm partition")
parser.add_argument("--local", action="store_true")
args = parser.parse_args()

cmd = f"python motiondiff/data_pipeline/scripts/test_bones_bvh_check_artifacts.py"
print(cmd)
if args.local:
    subprocess.run(cmd, shell=True)
else:
    job_cmd = f"mkdir -p /repo; cd /repo; git clone ssh://git@gitlab-master.nvidia.com:12051/ediff-i-motion/physdiff.git; cd physdiff; git pull origin main; git fetch; git checkout bones; pip install lxml bvh pyrender pyvista==0.38.6; apt-get update; apt install -y libgl1-mesa-glx xvfb; {cmd}"

    ssh_cmd = (
        f"submit_job --partition {args.partition} --duration 48 --nodes 1 --cpu 96 --image /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/docker/motiondiff_0.3.5.sqsh"
        + f' --name "bones.check_artifacts" --command "{job_cmd}"'
    )
    print(ssh_cmd)

    subprocess.run(f"ssh drempe@cs-oci-ord-login-01 '{ssh_cmd}'", shell=True)
