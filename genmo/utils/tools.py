import datetime
import glob
import importlib
import itertools
import os
import os.path as osp
import subprocess
import time

import numpy as np
import wandb


class AverageMeter(object):
    def __init__(self, avg=None, count=1):
        self.reset()
        if avg is not None:
            self.val = avg
            self.avg = avg
            self.count = count
            self.sum = avg * count

    def __repr__(self) -> str:
        return f"{self.avg: .4f}"

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def worker_init_fn(worker_id):
    os.environ["worker_id"] = str(worker_id)
    np.random.seed(np.random.get_state()[1][0] + worker_id * 7)


def find_last_version(folder, prefix="version_", cp="last"):
    version_folders = glob.glob(f"{folder}/{prefix}*")
    if cp is not None:
        if cp == "last":
            suffix = "last.ckpt"
        elif cp == "best":
            suffix = "*best*.ckpt"
        elif cp.isdigit():
            suffix = f"*{int(cp):07d}.ckpt"
        else:
            suffix = f"{cp}.ckpt"
        version_folders = [
            x for x in version_folders if len(glob.glob(f"{x}/**/{suffix}")) > 0
        ]
    version_numbers = sorted(
        [int(osp.basename(x)[len(prefix) :]) for x in version_folders]
    )
    if len(version_numbers) == 0:
        return None
    last_version = version_numbers[-1]
    return last_version


def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return convert_sec_to_time(eta)


def convert_sec_to_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))


def concat_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


def find_consecutive_runs(x, min_len=1):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:] - 1, out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))
        ind = run_lengths >= min_len
        run_starts = run_starts[ind]
        run_lengths = run_lengths[ind]

        # find run values
        run_values = [
            x[start : start + length] for start, length in zip(run_starts, run_lengths)
        ]
        # assert np.allclose(np.concatenate(run_values), x)

        return run_values, run_starts, run_lengths


def get_checkpoint_path(checkpoint_dir, cp, return_name=False):
    if cp == "last":  # use last epoch
        cp_name = "last.ckpt"
    elif cp == "best":  # use best epoch
        cp_name = osp.basename(sorted(glob.glob(f"{checkpoint_dir}/*best*.ckpt"))[-1])
    else:
        cp_name = osp.basename(sorted(glob.glob(f"{checkpoint_dir}/{cp}.ckpt"))[-1])
    cp_path = f"{checkpoint_dir}/{cp_name}"
    if return_name:
        return cp_path, cp_name
    return cp_path


def subprocess_run(cmd, ignore_err=False, **kwargs):
    try:
        result = subprocess.run(cmd, **kwargs)
    except subprocess.CalledProcessError as err:
        print("####### subprocess-run error message ######")
        print(f"{err} {err.stderr.decode('utf8')}")
    if result.returncode != 0:
        if not ignore_err:
            raise Exception("error in subprocess_run!")
    return result


def import_type_from_str(s):
    module_name, type_name = s.rsplit(".", 1)
    module = importlib.import_module(module_name)
    type_to_import = getattr(module, type_name)
    return type_to_import


def build_object_from_dict(d, type_field="type", **add_kwargs):
    d = d.copy()
    _type = import_type_from_str(d.pop(type_field))
    return _type(**d, **add_kwargs)


def write_list_to_file(filename, string_list):
    with open(filename, "w") as file:
        for item in string_list:
            file.write(item + "\n")


def are_arrays_equal(array1, array2, sort=False):
    if array1 is None or array2 is None:
        return False
    # if array1 == array2:
    #     return True
    if len(array1) != len(array2):
        return False

    # Sort both arrays
    if sort:
        array1 = sorted(array1)
        array2 = sorted(array2)

    # Compare each element
    for i in range(len(array1)):
        if array1[i] != array2[i]:
            return False

    return True


def wandb_run_exists():
    return isinstance(wandb.run, wandb.sdk.wandb_run.Run)


def load_ema_weights_from_checkpoint(model, checkpoint):
    ema_params = checkpoint["optimizer_states"][0]["ema"]
    for param, ema_param in zip(model.parameters(), ema_params):
        param.data.copy_(ema_param.data)
    return


def rsync_file_from_remote(fname, remote_dir, local_dir, hostname):
    remote_fname = fname.replace(local_dir, f"{remote_dir}/./")
    cmd = f"rsync -avzP -m --relative {hostname}:{remote_fname} {local_dir}/"
    subprocess_run(cmd, shell=True)
    return


# Global variable for timing indentation level
timer_indent_level = 0


# Context manager for timing
class Timer:
    def __init__(self, name="", enabled=True, show_rank=False, rank_zero_only=True):
        self.name = name
        self.start_time = None
        self.enabled = enabled
        if "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])
        else:
            self.rank = 0
        self.show_rank = show_rank
        self.rank_zero_only = rank_zero_only

    def __enter__(self):
        if (not self.enabled) or (self.rank_zero_only and self.rank != 0):
            return self
        global timer_indent_level
        self.start_time = time.perf_counter()
        self.current_indent = timer_indent_level  # Capture current indent level
        timer_indent_level += 1  # Increment global indent level for next call
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False  # Re-raise the exception
        if (not self.enabled) or (self.rank_zero_only and self.rank != 0):
            return self
        global timer_indent_level
        elapsed_time = time.perf_counter() - self.start_time
        indent = "    " * self.current_indent  # 4 spaces per indent level
        rank_str = f"[rank{self.rank}] " if self.show_rank else ""
        print(f"{indent}{rank_str}[{self.name}] time: {elapsed_time:.4f} seconds")
        timer_indent_level -= 1  # Decrement global indent level after finishing
