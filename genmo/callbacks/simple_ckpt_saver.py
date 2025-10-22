import os
from copy import deepcopy
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.checkpoint import Checkpoint
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor

from genmo.utils.pylogger import Log


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
        every_n_epochs=None,
        every_n_steps=None,
        save_last=None,
        save_weights_only=False,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.save_top_k = save_top_k
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps
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
        monitor_candidates["epoch"] = (
            epoch.int()
            if isinstance(epoch, Tensor)
            else torch.tensor(trainer.current_epoch)
        )
        step = monitor_candidates.get("step")
        monitor_candidates["step"] = (
            step.int()
            if isinstance(step, Tensor)
            else torch.tensor(trainer.global_step)
        )
        return monitor_candidates

    def _save_last_checkpoint(self, trainer, pl_module, monitor_candidates) -> None:
        lastpath = self.output_dir / "last.ckpt"
        checkpoint = trainer._checkpoint_connector.dump_checkpoint()
        trainer.strategy.save_checkpoint(checkpoint, lastpath)
        os.chmod(lastpath, 0o755)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """Save a checkpoint at the end of the training epoch."""
        if (
            self.every_n_epochs is not None
            and (trainer.current_epoch + 1) % self.every_n_epochs == 0
        ):
            if self.save_top_k == 0:
                return

            # Save cureent checkpoint
            filepath = self.output_dir / self.filename.format(
                epoch=trainer.current_epoch, step=trainer.global_step
            )
            lastpath = self.output_dir / "last.ckpt"
            checkpoint = trainer._checkpoint_connector.dump_checkpoint()
            trainer.strategy.save_checkpoint(checkpoint, filepath)
            trainer.strategy.save_checkpoint(checkpoint, lastpath)
            os.chmod(filepath, 0o755)
            os.chmod(lastpath, 0o755)

    @rank_zero_only
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        """Save a checkpoint at the end of the training epoch."""
        if (
            self.every_n_steps is not None
            and trainer.global_step % self.every_n_steps == 0
        ):
            if self.save_top_k == 0:
                return

            # Save cureent checkpoint
            filepath = self.output_dir / "s{step:06d}.ckpt".format(
                step=trainer.global_step
            )
            lastpath = self.output_dir / "last.ckpt"
            checkpoint = trainer._checkpoint_connector.dump_checkpoint()
            trainer.strategy.save_checkpoint(checkpoint, filepath)
            trainer.strategy.save_checkpoint(checkpoint, lastpath)
            os.chmod(filepath, 0o755)
            os.chmod(lastpath, 0o755)
