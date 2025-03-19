import os

import torch.distributed as dist
from pytorch_lightning.callbacks import Callback


class PHCDataCollectionCallback(Callback):
    """Callback to rerun PHC data collection at the start of each epoch."""

    def __init__(self, collect_kwargs, every_n_epochs=1):
        """Initialize the callback.

        Args:
            data_path: Path where the PHC data will be saved
        """
        super().__init__()
        self.collect_kwargs = collect_kwargs
        self.every_n_epochs = every_n_epochs

    def _collect_phc_data(self, trainer, pl_module):
        """Run PHC data collection.

        Args:
            pl_module: The PyTorch Lightning module containing the humanoid
        """
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        # Get the humanoid instance from the model
        phc_dataset = None
        for dataset in trainer.train_dataloader.dataset.datasets:
            if dataset.__class__.__name__ == "PHCDataset":
                phc_dataset = dataset
                break

        print(
            f"Re-collecting PHC data at the start of epoch {trainer.current_epoch + 1}"
        )
        humanoid = pl_module.humanoid
        # Run data collection
        humanoid.player.run_data_collection(
            train_info={
                "global_step": pl_module.global_step,
            },
            rank=trainer.global_rank,
            **self.collect_kwargs,
        )

        phc_dataset.reload_data()
        return

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch begins."""
        self._collect_phc_data(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the train batch begins."""
        # self._collect_phc_data(trainer, pl_module)
        pass
