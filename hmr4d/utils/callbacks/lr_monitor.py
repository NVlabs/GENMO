from pytorch_lightning.callbacks import LearningRateMonitor

from hmr4d.configs import MainStore, builds

MainStore.store(
    name="pl", node=builds(LearningRateMonitor), group="callbacks/lr_monitor"
)
