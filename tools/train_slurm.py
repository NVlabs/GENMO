import os
import os.path as osp
import numpy as np
import glob
import sys
import argparse
import pathlib
import time
import json
import datetime

sys.path.append(os.getcwd())

from motiondiff.utils.tools import subprocess_run

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cfg", type=str, required=True, help="config file")
parser.add_argument("-v", "--cfg_var", type=str, default='', help="exp name var")
parser.add_argument("-g", "--gpus", type=int, default=1, help="gpus used per node")
parser.add_argument("-n", "--nodes", type=int, default=1, help="number of nodes")
parser.add_argument('-ar_bt', '--autoresume_before_timelimit', type=int, default=30)
parser.add_argument('-pt', '--partition', default='polar,polar2,polar3,polar4,grizzly', help='slurm partition')
parser.add_argument('-t', '--time', type=int, default=4, help='single slurm job time duration in hours')
parser.add_argument('-db', '--debug', action="store_true")
parser.add_argument('-f', '--filter', nargs='+', default=[], help='filter cfg by keywords')
parser.add_argument('-exc', '--exclude', nargs='+', default=[], help='exclude cfg by keywords')
parser.add_argument('-s', '--stage', default='opt')
parser.add_argument('-l', '--local', action="store_true")
parser.add_argument('-u', '--user', help='cluster username', required=True)
parser.add_argument('-a', '--account', help='cluster account/team', default='nvr_torontoai_humanmotionfm')
parser.add_argument('-b', '--branch', default='main', help='git branch of the code base to run on cluster')
parser.add_argument('-p', '--push_changes', action="store_true")
parser.add_argument('-j', '--job_tag', default="motiondiff")
parser.add_argument('-group', '--wandb_group', default=None)
parser.add_argument('-dg', '--disable_wandb_group', action="store_true")
parser.add_argument('-si', '--start_ind', type=int, default=0, help='start index of cfgs')
parser.add_argument('-ei', '--end_ind', type=int, default=None, help='end index of cfgs')
parser.add_argument('-nc', '--num_cfg', type=int, default=None, help='number of cfgs to run')
parser.add_argument('-gm', '--git_message', default=None, help='git commit message')
parser.add_argument('-slack', '--slack_mode', default='fail', help='slack mode for ADLR script')
parser.add_argument('-test_ar', '--test_autoresume_timer', type=int, default=-1)
parser.add_argument('-r', '--resume', action="store_true")
parser.add_argument('-rcp', '--resume_cp', default="last")
args = parser.parse_args()


cmd = f"python tools/train.py exp={args.cfg} exp_name_var={args.cfg_var} pl_trainer.devices={args.gpus}"
if args.resume:
    cmd += f" --resume --cp {args.resume_cp}"