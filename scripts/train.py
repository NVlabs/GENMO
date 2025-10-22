import builtins
import os
from datetime import datetime

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import wandb
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks.checkpoint import Checkpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from genmo.callbacks.autoresume_callback import AutoResume, AutoResumeCallback
from genmo.utils.net_utils import get_resume_ckpt_path, load_pretrained_model
from genmo.utils.pylogger import Log
from genmo.utils.tools import (
    find_last_version,
    get_checkpoint_path,
    rsync_file_from_remote,
)
from genmo.utils.vis.rich_logger import print_cfg

OmegaConf.register_new_resolver("eval", builtins.eval)


def _get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return 0


global_rank = _get_rank()


def wandb_run_exists():
    return isinstance(wandb.run, wandb.sdk.wandb_run.Run)


def get_callbacks(cfg: DictConfig) -> list:
    """Parse and instantiate all the callbacks in the config.

    Supports both flat and nested callback configs. Only nodes containing
    a `_target_` are instantiated.
    """
    if not hasattr(cfg, "callbacks") or cfg.callbacks is None:
        return None

    def _collect_callback_nodes(node):
        collected = []
        if node is None:
            return collected
        # Dict-like node
        if isinstance(node, (DictConfig, dict)):
            # direct instantiable config
            if "_target_" in node:
                collected.append(node)
            else:
                for child in node.values():
                    collected.extend(_collect_callback_nodes(child))
        # List-like node
        elif isinstance(node, (ListConfig, list, tuple)):
            for child in node:
                collected.extend(_collect_callback_nodes(child))
        # primitives are ignored
        return collected

    enable_checkpointing = cfg.pl_trainer.get("enable_checkpointing", True)
    callbacks = []
    for cb_conf in _collect_callback_nodes(cfg.callbacks):
        cb = hydra.utils.instantiate(cb_conf, _recursive_=False)
        if not enable_checkpointing and isinstance(cb, Checkpoint):
            continue
        callbacks.append(cb)
    return callbacks


def train(cfg: DictConfig) -> None:
    """Train/Test"""
    Log.info(f"[Exp Name]: {cfg.exp_name}")
    # use total batch size
    if cfg.task == "fit":
        Log.info(
            f"[GPU x Batch] = {cfg.pl_trainer.devices} x {cfg.data.loader_opts.train.batch_size}"
        )
    num_nodes = cfg.pl_trainer.get("num_nodes", 1)
    cfg.num_test_data *= cfg.pl_trainer.devices * num_nodes
    if (
        "imgfeat_motionx" in cfg.test_datasets
        and "max_num_motions" in cfg.test_datasets.imgfeat_motionx
    ):
        cfg.test_datasets.imgfeat_motionx.max_num_motions *= (
            cfg.pl_trainer.devices * num_nodes
        )
    pl.seed_everything(cfg.seed)
    torch.cuda.set_device(global_rank % 8)  # for tinycudann default memory
    wandb_run = None
    version = None

    if cfg.get("timing", False):
        os.environ["DEBUG_TIMING"] = "TRUE"

    if AutoResume is not None:
        details = AutoResume.get_resume_details()
        if details:
            cfg.resume_mode = "last"
            if "wandb_id" in details:
                wandb_run = details["wandb_id"]
                version = int(details["version"])
            print(
                f"[Auto Resume] Loading. checkpoint: {details['checkpoint']} wandb_id: {details.get('wandb_id', None)}"
            )

    if cfg.task == "test" and not cfg.get("no_checkpoint", False):
        test_cp = cfg.get("test_checkpoint", "last")
        remote_run_dir = cfg.output_dir.replace("outputs", cfg.remote_results_path)
        version = find_last_version(remote_run_dir, cp=test_cp)
        checkpoint_dir = f"{remote_run_dir}/version_{version}/checkpoints"
        remote_ckpt_path = get_checkpoint_path(checkpoint_dir, test_cp)
        if cfg.get("rsync_ckpt", False):
            cfg.ckpt_path = remote_ckpt_path.replace(cfg.remote_results_path, "outputs")
            if not os.path.exists(cfg.ckpt_path):
                print(f"rsyncing from remote: {remote_ckpt_path}")
                print(f"output_dir: {cfg.output_dir}")
                rsync_file_from_remote(
                    cfg.ckpt_path,
                    remote_run_dir,
                    cfg.output_dir,
                    hostname="cs-oci-ord-dc-03",
                )
        else:
            cfg.ckpt_path = remote_ckpt_path
        print("ckpt path:", cfg.ckpt_path)
        cfg.output_dir = f"{cfg.output_dir}/version_{version}"
        cfg.logger.name = (
            f"{cfg.exp_name}_v{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
    else:
        run_root_dir = cfg.output_dir
        if version is None and cfg.resume_mode == "last":
            version = find_last_version(run_root_dir, cp="last")

    # preparation
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data, _recursive_=False
    )
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    if (
        cfg.get("pretrain_ckpt", None) is not None
        and cfg.ckpt_path is None
        and cfg.resume_mode is None
    ):
        cfg.ckpt_path = cfg.pretrain_ckpt

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    if cfg.ckpt_path is not None:
        if cfg.get("rsync_ckpt", False) and not os.path.exists(cfg.ckpt_path):
            print(f"rsyncing from remote: {cfg.ckpt_path}")
            cfg.ckpt_path = cfg.ckpt_path.replace(cfg.remote_results_path, "outputs")
            local_dir = cfg.ckpt_path.split("/version_")[0]
            os.makedirs(local_dir, exist_ok=True)
            rsync_file_from_remote(
                cfg.ckpt_path,
                cfg.remote_results_path,
                "outputs",
                hostname="cs-oci-ord-dc-03",
            )

        ckpt = load_pretrained_model(model, cfg.ckpt_path)
        print(f"Loaded pretrained model from {cfg.ckpt_path}")
        if ckpt is not None:
            wandb_cfg["pretrained_ckpt_info"] = {
                "global_step": ckpt["global_step"],
                "epoch": ckpt["epoch"],
            }
            print("pretrained ckpt info:", wandb_cfg["pretrained_ckpt_info"])

    # PL callbacks and logger
    if cfg.task == "fit":
        if global_rank == 0:
            tb_logger = TensorBoardLogger(run_root_dir, version=version, name="")
            version = tb_logger.version
            os.makedirs(tb_logger.log_dir, exist_ok=True)
            cfg.output_dir = tb_logger.log_dir

            slurm_job_id = int(os.environ.get("SLURM_JOB_ID", "-1"))
            run_name = (
                f"{cfg.exp_name}_v{version}_{slurm_job_id}"
                if slurm_job_id > 0
                else f"{cfg.exp_name}_v{version}"
            )
            cfg.logger.name = run_name
            # cfg.logger.version = version  # shouldn't set version for Wandb

            if cfg.resume_mode == "last" and os.path.exists(
                f"{tb_logger.log_dir}/meta.yaml"
            ):
                meta = yaml.safe_load(open(f"{tb_logger.log_dir}/meta.yaml", "r"))
                if wandb_run is None:
                    wandb_run = meta["wandb_run"]
            if wandb_run is None:
                wandb_run = f"{cfg.exp_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            cfg.logger.id = wandb_run

        if cfg.pl_trainer.devices > 1 and "RANK" in os.environ:
            dist.init_process_group("nccl")
            dist.barrier()

        if global_rank != 0:
            if version is None:
                version = find_last_version(run_root_dir, cp=None)
            cfg.output_dir = f"{run_root_dir}/version_{version}"

    callbacks = get_callbacks(cfg)
    has_ckpt_cb = any([isinstance(cb, Checkpoint) for cb in callbacks])
    if not has_ckpt_cb and cfg.pl_trainer.get("enable_checkpointing", True):
        Log.warning("No checkpoint-callback found. Disabling PL auto checkpointing.")
        cfg.pl_trainer = {**cfg.pl_trainer, "enable_checkpointing": False}
    if AutoResume is not None:
        callbacks.append(AutoResumeCallback(version))

    cfg.logger.config = wandb_cfg
    if cfg.use_wandb:
        logger = hydra.utils.instantiate(
            cfg.logger, _recursive_=False, _convert_="partial"
        )
    else:
        logger = tb_logger

    if cfg.task == "fit" and global_rank == 0:
        # wandb.config.update({"cfg": OmegaConf.to_container(cfg)}, allow_val_change=True)
        assert cfg.logger.id is not None
        meta = {"wandb_run": cfg.logger.id}
        yaml.safe_dump(meta, open(f"{tb_logger.log_dir}/meta.yaml", "w"))
        print("saved meta:", meta)

    # PL-Trainer
    if cfg.task == "test":
        Log.info("Test mode forces full-precision.")
        cfg.pl_trainer = {**cfg.pl_trainer, "precision": 32}
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )

    print("=" * 20)
    print("version:", version)

    if cfg.task == "fit":
        resume_path = None
        if cfg.resume_mode is not None:
            save_dir = cfg.output_dir + "/checkpoints"
            resume_path = get_resume_ckpt_path(cfg.resume_mode, ckpt_dir=save_dir)
        Log.info("Start Fitting...")
        trainer.fit(
            model,
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            ckpt_path=resume_path,
        )
    elif cfg.task == "test":
        Log.info("Start Testing...")
        trainer.test(model, datamodule.test_dataloader())
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    Log.info("End of script.")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg) -> None:
    print_cfg(cfg, use_rich=True)
    train(cfg)


if __name__ == "__main__":
    main()
