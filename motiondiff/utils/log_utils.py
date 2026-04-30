import logging
import os
import time

from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from motiondiff.utils.tools import convert_sec_to_time, get_eta_str


def create_logger(file_path, file_handle=True):
    # create logger
    logger = logging.getLogger(file_path)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)

    if file_handle:
        # create file handler which logs even debug messages
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fh = logging.FileHandler(file_path, mode="a")
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("[%(asctime)s] %(message)s")
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger


class TextLogger(Logger):
    def __init__(
        self,
        file_path,
        cfg=None,
        write_file=True,
        training=True,
        max_steps=-1,
        log_nsteps=1,
        loss_fmt="%.4f",
    ):
        super().__init__()
        self.training = training
        self.write_file = write_file
        self.cfg = cfg
        self.max_steps = max_steps
        self.log_nsteps = log_nsteps
        self.loss_fmt = loss_fmt
        self.cfg_name = cfg.id if cfg is not None else "Exp"
        self.setup_log(file_path, write_file)
        self.cur_metrics = dict()
        self.metrics_to_ignore = set(["step", "epoch"])
        self.last_step = None
        self.last_step_time = time.time()

    def setup_log(self, file_path, write_file):
        self.log = log = logging.getLogger(file_path)
        log.propagate = False
        log.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        stream_formatter = logging.Formatter("%(message)s")
        ch.setFormatter(stream_formatter)
        log.addHandler(ch)

        if write_file:
            # create file handler which logs even debug messages
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            fh = logging.FileHandler(file_path, mode="a")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
            log.addHandler(fh)

    def log_metrics(self, metrics, step):
        self.cur_metrics.update(metrics)
        self.cur_metrics["step"] = step
        if step != self.last_step:
            if self.training:
                self.log_train()
            else:
                self.log_eval()
        self.last_step = step

    @rank_zero_only
    def log_train(self):
        log = self.log
        step = self.cur_metrics["step"]
        if (step + 1) % self.log_nsteps != 0:
            return
        step_secs = time.time() - self.last_step_time
        eta_str = get_eta_str(step, self.max_steps, step_secs / self.log_nsteps)
        loss_fmt_str = "%s: " + self.loss_fmt
        loss_str = " | ".join(
            [
                (loss_fmt_str % (x, y))
                for x, y in self.cur_metrics.items()
                if x not in self.metrics_to_ignore
            ]
        )
        info_str = f"{self.cfg_name} | {step + 1:4d}/{self.max_steps} | TE: {convert_sec_to_time(step_secs)} ETA: {eta_str} | {loss_str}"
        log.info(info_str)
        self.last_step_time = time.time()

    def log_eval(self):
        pass

    @property
    def experiment():
        pass

    @property
    def name(self):
        return "textlogger"

    def log_hyperparams(self, hparams):
        pass

    @property
    def version(self):
        pass
