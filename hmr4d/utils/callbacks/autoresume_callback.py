import os
from typing import Any
import wandb

from pytorch_lightning import Callback, LightningModule, Trainer
from motiondiff.utils.tools import wandb_run_exists

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

    def _check_autoresume(self, trainer, pl_module):
        if AutoResume is not None and AutoResume.termination_requested():
            if trainer.global_rank == 0:
                cp = trainer.checkpoint_callback
                monitor_candidates = cp._monitor_candidates(trainer)
                cp._save_last_checkpoint(trainer, pl_module, monitor_candidates)
                checkpoint = cp.output_dir / 'last.ckpt'
                details = {
                    "checkpoint": checkpoint,
                    "wandb_id": wandb.run.id if wandb_run_exists() else "",
                    "version": str(self.version),
                }
                message = f"[Auto Resume] Terminateing. checkpoint: {checkpoint} wandb_id: {details['wandb_id']} version: {details['version']}"
                print(message, flush=True)
                AutoResume.request_resume(
                    details,
                    message=message
                )
                if wandb_run_exists():
                    wandb.run.finish()
                trainer.should_stop = True
                trainer.limit_val_batches = 0
            else:
                print(f"[Auto Resume] Rank {trainer.global_rank} exiting.", flush=True)

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