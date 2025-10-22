import os
from typing import Any

import wandb
from pytorch_lightning import Callback, LightningModule, Trainer

from genmo.utils.tools import wandb_run_exists

try:
    import sys

    sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
    from userlib.auto_resume import AutoResume
except ModuleNotFoundError:
    AutoResume = None


class AutoResumeCallback(Callback):
    def __init__(self, version=None) -> None:
        if AutoResume is not None:
            AutoResume.init()
        self.version = version
        self.last_epoch_checkpoint = None

    def _dump_current_checkpoint(self, trainer):
        cp = trainer.checkpoint_callback
        try:
            checkpoint_dict = trainer._checkpoint_connector.dump_checkpoint(
                cp.save_weights_only
            )
        except ValueError:
            # This can happen for 16 precision for the first iterations
            # when the GradScaler is still adapting and skipping steps
            checkpoint_dict = None
        return checkpoint_dict

    def _save_checkpoint(self, trainer, checkpoint, filepath):
        """Save a checkpoint to the memory."""

        trainer.strategy.save_checkpoint(
            checkpoint,
            filepath,
        )
        # trainer.strategy.barrier("Trainer.save_checkpoint")
        os.chmod(filepath, 0o755)

    def _save_last_checkpoints(self, trainer):
        cp = trainer.checkpoint_callback
        # monitor_candidates = cp._monitor_candidates(trainer)

        # Save last epoch. This is the one we will use for the autoresume
        filepath_epoch = cp.output_dir / "last.ckpt"
        if self.last_epoch_checkpoint is not None:
            self._save_checkpoint(trainer, self.last_epoch_checkpoint, filepath_epoch)
        else:
            filepath_epoch = None

        # Save last step just in case
        # filepath_step = cp.output_dir / 'last_step.ckpt'
        # last_step_checkpoint = self._dump_current_checkpoint(trainer)
        # if last_step_checkpoint is not None:
        #     self._save_checkpoint(trainer, last_step_checkpoint, filepath_step)

        # save top k
        # self._save_topk_checkpoint(trainer)
        return filepath_epoch

    def _check_autoresume(self, trainer, pl_module):
        if AutoResume is not None and AutoResume.termination_requested():
            if trainer.global_rank == 0:
                checkpoint = self._save_last_checkpoints(trainer)
                details = {
                    "checkpoint": checkpoint,
                    "wandb_id": wandb.run.id if wandb_run_exists() else "",
                    "version": str(self.version),
                }
                message = f"[Auto Resume] Terminateing. checkpoint: {checkpoint} wandb_id: {details['wandb_id']} version: {details['version']}"
                print(message, flush=True)
                AutoResume.request_resume(details, message=message)
                if wandb_run_exists():
                    wandb.run.finish()
                trainer.should_stop = True
                trainer.limit_val_batches = 0
            else:
                print(f"[Auto Resume] Rank {trainer.global_rank} exiting.", flush=True)

            if hasattr(pl_module, "cleanup_for_autoresume"):
                pl_module.cleanup_for_autoresume()

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        # only for rank 0
        if trainer.global_rank == 0:
            # save the last epoch checkpoint in memory
            # this is the one we will dump for the "last" checkpoint
            checkpoint_dict = self._dump_current_checkpoint(trainer)
            self.last_epoch_checkpoint = checkpoint_dict

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._check_autoresume(trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pass
        # self._check_autoresume(trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pass
        # self._check_autoresume(trainer, pl_module)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pass
        # self._check_autoresume(trainer, pl_module)
