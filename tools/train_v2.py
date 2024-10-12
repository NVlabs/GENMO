import hydra
import pytorch_lightning as pl
import os
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks.checkpoint import Checkpoint

import wandb
from hmr4d.utils.pylogger import Log
from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.vis.rich_logger import print_cfg
from hmr4d.utils.net_utils import load_pretrained_model, get_resume_ckpt_path
from motiondiff.utils.tools import find_last_version
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.utils.callbacks.autoresume_callback import AutoResumeCallback, AutoResume
import yaml
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


def wandb_run_exists():
    return isinstance(wandb.run, wandb.sdk.wandb_run.Run)


def get_callbacks(cfg: DictConfig) -> list:
    """Parse and instantiate all the callbacks in the config."""
    if not hasattr(cfg, "callbacks") or cfg.callbacks is None:
        return None
    # Handle special callbacks
    enable_checkpointing = cfg.pl_trainer.get("enable_checkpointing", True)
    # Instantiate all the callbacks
    callbacks = []
    for callback in cfg.callbacks.values():
        if callback is not None:
            cb = hydra.utils.instantiate(callback, _recursive_=False)
            # skip when disable checkpointing and the callback is Checkpoint
            if not enable_checkpointing and isinstance(cb, Checkpoint):
                continue
            else:
                callbacks.append(cb)
    return callbacks


def train(cfg: DictConfig) -> None:
    """Train/Test"""
    Log.info(f"[Exp Name]: {cfg.exp_name}")
    # use total batch size
    cfg.data.loader_opts.train.batch_size = cfg.data.loader_opts.train.batch_size // cfg.pl_trainer.devices
    if cfg.task == "fit":
        Log.info(f"[GPU x Batch] = {cfg.pl_trainer.devices} x {cfg.data.loader_opts.train.batch_size}")
    pl.seed_everything(cfg.seed)

    if AutoResume is not None:
        details = AutoResume.get_resume_details()
        if details:
            cfg.resume_mode = 'last'
            if 'wandb_id' in details:
                cfg.wandb_run = details['wandb_id']
                cfg.logger.version = details['version']
            print(f"[Auto Resume] Loading. checkpoint: {details['checkpoint']} wandb_id: {details.get('wandb_id', None)}")
        
    # preparation
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)
    if cfg.ckpt_path is not None:
        load_pretrained_model(model, cfg.ckpt_path)

    # PL callbacks and logger

    global_rank = rank_zero_only.rank if rank_zero_only.rank is not None else 0
    if global_rank == 0:

        slurm_job_id = int(os.environ.get("SLURM_JOB_ID", "-1"))
        run_name = f'{cfg.exp_name}_{slurm_job_id}' if slurm_job_id > 0 else f'{cfg.exp_name}'
        cfg.logger.name = run_name

        tb_logger = TensorBoardLogger(f'{cfg.logger.save_dir}', version=None, name='')
        version = tb_logger.version

        meta = {'wandb_run': wandb.run.id if wandb_run_exists() else None}
        yaml.safe_dump(meta, open(f'{tb_logger.log_dir}/meta.yaml', 'w'))
        cfg.logger.version = version

    callbacks = get_callbacks(cfg)
    has_ckpt_cb = any([isinstance(cb, Checkpoint) for cb in callbacks])
    if not has_ckpt_cb and cfg.pl_trainer.get("enable_checkpointing", True):
        Log.warning("No checkpoint-callback found. Disabling PL auto checkpointing.")
        cfg.pl_trainer = {**cfg.pl_trainer, "enable_checkpointing": False}
    if AutoResume is not None:
        callbacks.append(AutoResumeCallback(cfg.logger.version))

    logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)

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

    if cfg.task == "fit":
        resume_path = None
        if cfg.resume_mode is not None:
            save_dir = cfg.logger.save_dir + '/checkpoints'
            resume_path = get_resume_ckpt_path(cfg.resume_mode, ckpt_dir=save_dir)
            Log.info(f"Resume training from {resume_path}")
        Log.info("Start Fitiing...")
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=resume_path)
    elif cfg.task == "test":
        Log.info("Start Testing...")
        trainer.test(model, datamodule.test_dataloader())
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    Log.info("End of script.")


@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg) -> None:
    print_cfg(cfg, use_rich=True)
    train(cfg)


if __name__ == "__main__":
    register_store_gvhmr()
    main()
