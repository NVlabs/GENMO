import hydra
import pytorch_lightning as pl
import os
import torch
import torch.distributed as dist
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks.checkpoint import Checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hmr4d.utils.pylogger import Log
from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.vis.rich_logger import print_cfg
from hmr4d.utils.net_utils import load_pretrained_model, get_resume_ckpt_path
from motiondiff.utils.tools import find_last_version, get_checkpoint_path, rsync_file_from_remote
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.utils.callbacks.autoresume_callback import AutoResumeCallback, AutoResume
from motiondiff.utils.torch_utils import tensor_to
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from hmr4d.model.gvhmr.utils.vis_smpl import SMPLVisualizer


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
    pl.seed_everything(cfg.seed)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))   # for tinycudann default memory
    torch.set_grad_enabled(False)
    version = None
    
    if cfg.task == 'test' and not cfg.get('no_checkpoint', False):
        test_cp = cfg.get('test_checkpoint', 'last')
        remote_run_dir = cfg.output_dir.replace('outputs', cfg.remote_results_path)
        version = find_last_version(remote_run_dir, cp=test_cp)
        checkpoint_dir = f'{remote_run_dir}/version_{version}/checkpoints'
        remote_ckpt_path = get_checkpoint_path(checkpoint_dir, test_cp)
        if cfg.get('rsync_ckpt', False):
            cfg.ckpt_path = remote_ckpt_path.replace(cfg.remote_results_path, 'outputs')
            if not os.path.exists(cfg.ckpt_path):
                print(f"rsyncing from remote: {remote_ckpt_path}")
                print(f"output_dir: {cfg.output_dir}")
                rsync_file_from_remote(cfg.ckpt_path, remote_run_dir, cfg.output_dir, hostname='cs-oci-ord-dc-03')
        else:
            cfg.ckpt_path = remote_ckpt_path
        print("ckpt path:", cfg.ckpt_path)
        cfg.output_dir = f'{cfg.output_dir}/version_{version}'
        cfg.logger.name = f"{cfg.exp_name}_v{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:    
        run_root_dir = cfg.output_dir
        if version is None:
            version = find_last_version(run_root_dir, cp='last')


    # preparation
    # datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)
    
    if cfg.get('pretrain_ckpt', None) is not None and cfg.ckpt_path is None and cfg.resume_mode is None:
        load_pretrained_model(model, cfg.pretrain_ckpt)
        print(f"Loaded pretrained model from {cfg.pretrain_ckpt}")
    if cfg.ckpt_path is not None:
        ckpt = load_pretrained_model(model, cfg.ckpt_path)
        
    model.eval()
    model.cuda()
    
    # dataloader = datamodule.test_dataloader()
    # data = next(iter(dataloader))[0]
    # data = tensor_to(data, 'cuda')
    # torch.save(data, 'out/data.pth')
    
    data = torch.load('out/data.pth', map_location='cuda')
    
    outputs = model.validation_3d(data, 0)
    
    vis = SMPLVisualizer(distance=7, elevation=10)
    vis.show_animation(init_args={'smpl_seq': outputs['pred_smpl_params_global']}, window_size=(1500, 1500), fps=30)
    

    print('='*20)
    print('version:', version)


    Log.info("End of script.")


@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg) -> None:
    print_cfg(cfg, use_rich=True)
    train(cfg)


if __name__ == "__main__":
    register_store_gvhmr()
    main()
