from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.checkpoint import Checkpoint
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds
from copy import deepcopy

from typing import Any, Dict, Optional, Set
from torch import Tensor


class SimpleCkptSaver(Checkpoint):
    """
    This callback runs at the end of each training epoch.
    Check {every_n_epochs} and save at most {save_top_k} model if it is time.
    """

    def __init__(
        self,
        output_dir,
        filename="e{epoch:03d}-s{step:06d}.ckpt",
        save_top_k=1,
        every_n_epochs=1,
        save_last=None,
        save_weights_only=False,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.save_top_k = save_top_k
        self.every_n_epochs = every_n_epochs
        self.save_last = save_last
        self.save_weights_only = save_weights_only

        # Setup output dir
        if rank_zero_only.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            Log.info(f"[Simple Ckpt Saver]: Save to `{self.output_dir}'")

    def _monitor_candidates(self, trainer: "pl.Trainer") -> Dict[str, Tensor]:
        monitor_candidates = deepcopy(trainer.callback_metrics)
        # cast to int if necessary because `self.log("epoch", 123)` will convert it to float. if it's not a tensor
        # or does not exist we overwrite it as it's likely an error
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
        step = monitor_candidates.get("step")
        monitor_candidates["step"] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)
        return monitor_candidates

    # def _save_last_checkpoint(self, trainer, pl_module, monitor_candidates) -> None:
    #     lastpath = self.output_dir / 'last.ckpt'
    #     checkpoint = {
    #         "epoch": trainer.current_epoch,
    #         "global_step": trainer.global_step,
    #         "pytorch-lightning_version": pl.__version__,
    #         "state_dict": pl_module.state_dict(),
    #     }
    #     pl_module.on_save_checkpoint(checkpoint)

    #     if not self.save_weights_only:
    #         # optimizer
    #         optimizer_states = []
    #         for i, optimizer in enumerate(trainer.optimizers):
    #             # Rely on accelerator to dump optimizer state
    #             optimizer_state = trainer.strategy.optimizer_state(optimizer)
    #             optimizer_states.append(optimizer_state)
    #         checkpoint["optimizer_states"] = optimizer_states

    #         # lr_scheduler
    #         lr_schedulers = []
    #         for config in trainer.lr_scheduler_configs:
    #             lr_schedulers.append(config.scheduler.state_dict())
    #         checkpoint["lr_schedulers"] = lr_schedulers

    #     # trainer.strategy.checkpoint_io.save_checkpoint(checkpoint, filepath)
    #     torch.save(checkpoint, lastpath)

    # @rank_zero_only
    # def on_train_epoch_end(self, trainer, pl_module):
    #     """Save a checkpoint at the end of the training epoch."""
    #     if self.every_n_epochs >= 1 and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
    #         if self.save_top_k == 0:
    #             return

    #         # Current saved ckpts in the output_dir
    #         model_paths = []
    #         for p in sorted(list(self.output_dir.glob("*.ckpt"))):
    #             model_paths.append(p)
    #         model_to_remove = model_paths[0] if len(model_paths) >= self.save_top_k else None

    #         # Save cureent checkpoint
    #         filepath = self.output_dir / self.filename.format(epoch=trainer.current_epoch, step=trainer.global_step)
    #         lastpath = self.output_dir / 'last_epoch.ckpt'
    #         checkpoint = {
    #             "epoch": trainer.current_epoch,
    #             "global_step": trainer.global_step,
    #             "pytorch-lightning_version": pl.__version__,
    #             "state_dict": pl_module.state_dict(),
    #         }
    #         pl_module.on_save_checkpoint(checkpoint)

    #         if not self.save_weights_only:
    #             # optimizer
    #             optimizer_states = []
    #             for i, optimizer in enumerate(trainer.optimizers):
    #                 # Rely on accelerator to dump optimizer state
    #                 optimizer_state = trainer.strategy.optimizer_state(optimizer)
    #                 optimizer_states.append(optimizer_state)
    #             checkpoint["optimizer_states"] = optimizer_states

    #             # lr_scheduler
    #             lr_schedulers = []
    #             for config in trainer.lr_scheduler_configs:
    #                 lr_schedulers.append(config.scheduler.state_dict())
    #             checkpoint["lr_schedulers"] = lr_schedulers

    #         # trainer.strategy.checkpoint_io.save_checkpoint(checkpoint, filepath)
    #         torch.save(checkpoint, filepath)
    #         # torch.save(checkpoint, lastpath)

    #         # Remove the earliest checkpoint
    #         if model_to_remove:
    #             trainer.strategy.remove_checkpoint(model_paths[0])
    
        
    def _save_last_checkpoint(self, trainer, pl_module, monitor_candidates) -> None:
        lastpath = self.output_dir / 'last.ckpt'
        checkpoint = trainer._checkpoint_connector.dump_checkpoint()
        trainer.strategy.save_checkpoint(checkpoint, lastpath)
        
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """Save a checkpoint at the end of the training epoch."""
        if self.every_n_epochs >= 1 and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            if self.save_top_k == 0:
                return

            # Save cureent checkpoint
            filepath = self.output_dir / self.filename.format(epoch=trainer.current_epoch, step=trainer.global_step)
            lastpath = self.output_dir / 'last.ckpt'
            checkpoint = trainer._checkpoint_connector.dump_checkpoint()
            trainer.strategy.save_checkpoint(checkpoint, filepath)
            trainer.strategy.save_checkpoint(checkpoint, lastpath)


    # def on_train_batch_end(
    #     self,
    #     trainer,
    #     pl_module,
    #     outputs,
    #     batch,
    #     batch_idx: int,
    # ) -> None:
    #     self._check_autoresume(trainer, pl_module)


group_name = "callbacks/simple_ckpt_saver"
base = builds(SimpleCkptSaver, output_dir="${output_dir}/checkpoints/", populate_full_signature=True)
MainStore.store(name="base", node=base, group=group_name)
MainStore.store(name="every1e", node=base, group=group_name)
MainStore.store(name="every2e", node=base(every_n_epochs=2), group=group_name)
MainStore.store(name="every5e", node=base(every_n_epochs=5), group=group_name)
MainStore.store(name="every5e_top100", node=base(every_n_epochs=5, save_top_k=100), group=group_name)
MainStore.store(name="every10e", node=base(every_n_epochs=10), group=group_name)
MainStore.store(name="every10e_top100", node=base(every_n_epochs=10, save_top_k=100), group=group_name)
MainStore.store(name="every100e_top100", node=base(every_n_epochs=100, save_top_k=100), group=group_name)
MainStore.store(name="every1000e_top100", node=base(every_n_epochs=1000, save_top_k=100), group=group_name)
