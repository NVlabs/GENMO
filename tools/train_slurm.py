import argparse
import datetime
import glob
import json
import os
import os.path as osp
import pathlib
import sys
import time

import numpy as np

sys.path.append(os.getcwd())

from motiondiff.utils.tools import subprocess_run

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cfg_query", nargs="+", default=[])
parser.add_argument("-d", "--dir_query", nargs="+", default=[])
parser.add_argument("-v", "--cfg_var", type=str, default="", help="exp name var")
parser.add_argument("-env", "--env_var", default="DUMMYFLAG=1")
parser.add_argument("-m", "--mode", default="default")
parser.add_argument("-g", "--gpus", type=int, default=1, help="gpus used per node")
parser.add_argument("-n", "--nodes", type=int, default=1, help="number of nodes")
parser.add_argument("-ar_bt", "--autoresume_before_timelimit", type=int, default=20)
parser.add_argument(
    "-pt", "--partition", default="polar,polar3,polar4,grizzly", help="slurm partition"
)
parser.add_argument(
    "-t", "--time", type=int, default=4, help="single slurm job time duration in hours"
)
parser.add_argument("-db", "--debug", action="store_true")
parser.add_argument(
    "-exc", "--exclude", nargs="+", default=[], help="exclude cfg by keywords"
)
parser.add_argument("-s", "--stage", default="opt")
parser.add_argument("-l", "--local", action="store_true")
parser.add_argument("-i", "--interactive", action="store_true")
parser.add_argument("-u", "--user", help="cluster username", required=True)
parser.add_argument(
    "-a",
    "--account",
    help="cluster account/team (nvr_torontoai_humanmotionfm|nvr_lpr_digitalhuman)",
    default="nvr_torontoai_humanmotionfm",
)
parser.add_argument(
    "-b",
    "--branch",
    default="main",
    help="git branch of the code base to run on cluster",
)
parser.add_argument("-p", "--push_changes", action="store_true")
parser.add_argument("-j", "--job_tag", default="mfm-hmr")
parser.add_argument("-group", "--wandb_group", default=None)
parser.add_argument("-dg", "--disable_wandb_group", action="store_true")
parser.add_argument(
    "-si", "--start_ind", type=int, default=0, help="start index of cfgs"
)
parser.add_argument(
    "-ei", "--end_ind", type=int, default=None, help="end index of cfgs"
)
parser.add_argument(
    "-nc", "--num_cfg", type=int, default=None, help="number of cfgs to run"
)
parser.add_argument("-gm", "--git_message", default=None, help="git commit message")
parser.add_argument(
    "-slack", "--slack_mode", default="fail", help="slack mode for ADLR script"
)
parser.add_argument(
    "-test_ar", "--test_autoresume_timer", help="in minutes", type=int, default=-1
)
parser.add_argument("-r", "--resume", action="store_true")
parser.add_argument("-rcp", "--resume_cp", default="last")
parser.add_argument("-ag", "--additional_args", default="")
args = parser.parse_args()

config_dir = "hmr4d/configs/exp"
job_tag = args.job_tag
stages = [args.stage]
cfg_files = []
for query in args.dir_query:
    cfg_path = f"{config_dir}/**/{query}/**/*.yaml"
    cfg_files += sorted(glob.glob(cfg_path, recursive=True))
for query in args.cfg_query:
    cfg_path = f"{config_dir}/**/{query}.yaml"
    cfg_files += sorted(glob.glob(cfg_path, recursive=True))
cfg_files = sorted(list(set(cfg_files)))
print("cfgs:")

# ecluded cfg files
if len(args.exclude) > 0:
    exc_cfg_files = []
    for query in args.exclude:
        cfg_path = f"{config_dir}/**/{query}.yaml"
        exc_cfg_files += sorted(glob.glob(cfg_path, recursive=True))
        print("excluded:", exc_cfg_files)
    cfg_files = [c for c in cfg_files if c not in exc_cfg_files]
    for cfg in cfg_files:
        print(cfg)
    print("total after exclusion:", len(cfg_files))
else:
    for cfg in cfg_files:
        print(cfg)
    print("total:", len(cfg_files))


slurm_cmds = []
for cfg_f in cfg_files:
    cfg = osp.splitext(cfg_f)[0]
    cfg = osp.relpath(cfg, config_dir)
    if not args.debug:
        if args.mode == "humanoid":
            cmd = f"/workspace/isaaclab/_isaac_sim/python.sh tools/train_ilab.py --headless --enable_cameras exp={cfg} exp_name_var={args.cfg_var} ++pl_trainer.devices={args.gpus} ++pl_trainer.num_nodes={args.nodes} {args.additional_args}"
            if args.gpus > 1:
                cmd += " --distributed"
        else:
            cmd = f"python tools/train_v2.py exp={cfg} exp_name_var={args.cfg_var} ++pl_trainer.devices={args.gpus} ++pl_trainer.num_nodes={args.nodes} {args.additional_args}"
        cmd += " ++data.loader_opts.train.num_workers=8 ++data.loader_opts.train_2d.num_workers=8"
        if args.resume:
            cmd += f" resume_mode={args.resume_cp}"
    else:
        args.partition = "interactive"
        cmd = " --version; sleep 1000"

    cfg_tag = f"{cfg.replace('/', '_')}_{args.cfg_var}"
    if args.gpus > 1:
        cfg_tag += f"_{args.gpus}gpus"
    if args.nodes > 1:
        cfg_tag += f"_{args.nodes}nodes"

    tag = f"{args.job_tag}.{cfg_tag}"
    tag = tag[:110] if len(tag) > 110 else tag
    slurm_cmds.append((cmd, tag, cfg_tag))
    # print(cmd)


if len(slurm_cmds) == 0:
    print("No commands to run")
    sys.exit(0)

exp_base_folder = f"/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/workspaces/motiondiff/motiondiff_exp/{args.user}"
tmp_base_folder = f"/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/workspaces/phc/{args.user}/tmp"
subprocess_run(
    f"ssh {args.user}@cs-oci-ord-login-02 'mkdir -p {exp_base_folder}'", shell=True
)

now_str = (
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.interactive
    else "interactive"
)
exp_folder = f"{exp_base_folder}/gvhmr-{now_str}"
tmp_folder = f"{tmp_base_folder}/tmp-{now_str}"
repo_folder = str(
    pathlib.Path(__file__).resolve().parent.parent
)  # the root of the project

# can only include top
include_files = [
    "motiondiff/**",
    "hmr4d/**",
    "tools/**",
    "setup.py",
    "dyn_ddns.py",
    "third-party/**",  # thrid-party/PHC_Lab/** won't work
]
exclude_files = [
    "/inputs",
    "/outputs*",
    "/glove",
    "**/out",
    "**/wandb",
    "**/__pycache__",
    "**/doc",
    "**/docs",
    "*.ipynb",
    "*.safetensors",
    "third-party/DPVO",
    "third-party/PHC_Lab/phc/data",
    "third-party/**/*.pth",
]
include_str = " ".join([f'--include="{f}"' for f in include_files])
exclude_str = " ".join([f'--exclude="{f}"' for f in exclude_files])


rsync_cmd = f'rsync -az -m --partial --chmod=775 {exclude_str} {include_str} --exclude="*/*" {repo_folder}/ {args.user}@cs-oci-ord-dc-02:{exp_folder}/'
print(rsync_cmd)
if not args.debug:
    subprocess_run(rsync_cmd, shell=True)

if args.interactive:
    print(f"Interactive mode, skipping slurm jobs, please go to {exp_folder}")
    sys.exit(0)

for cmd, tag, cfg_tag in slurm_cmds:
    if args.mode == "humanoid":
        script_name = "slurm_job_ilab.sh"
        docker_image = "/lustre/fsw/portfolios/nvr/projects/nvr_lpr_digitalhuman/docker/ye_isaac_lab_2.1.1.sqsh"
        tmp_cfg_folder = f"{tmp_folder}-{cfg_tag}"
        job_cmd = f"cd {exp_folder}; tools/{script_name} {args.user} {args.branch} {args.env_var} {tmp_cfg_folder} {cmd}"
    else:
        script_name = "slurm_job.sh"
        docker_image = "/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/docker/gvhmr+v1.sqsh"
        job_cmd = f"cd {exp_folder}; tools/{script_name} {args.user} {args.branch} {args.env_var} {cmd}"
    print("job_cmd:", job_cmd)

    autoresume_str = (
        f"--autoresume_timer {args.test_autoresume_timer}"
        if args.test_autoresume_timer > 0
        else f"--autoresume_before_timelimit {args.autoresume_before_timelimit}"
    )
    autoresume_str += " --autoresume_ignore_failure"
    account_str = f"--account {args.account}" if args.account is not None else ""
    ssh_cmd = (
        f"submit_job --partition {args.partition} {account_str} --duration {args.time} --gpu {args.gpus} --nodes {args.nodes} --tasks_per_node {args.gpus} {autoresume_str} --email_mode {args.slack_mode} --image {docker_image}"
        + f' --name {tag} --command "{job_cmd}"'
    )

    print(f"ssh {args.user}@cs-oci-ord-login-02 '{ssh_cmd}'")
    if not args.debug:
        subprocess_run(f"ssh {args.user}@cs-oci-ord-login-02 '{ssh_cmd}'", shell=True)
    else:
        print(ssh_cmd)
