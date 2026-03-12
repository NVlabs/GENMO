# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from time import time

import torch
from colorlog import ColoredFormatter

_handler = logging.StreamHandler()
_handler.setFormatter(
    ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
)
_logger = logging.getLogger("gem")
_logger.addHandler(_handler)
_logger.setLevel(logging.DEBUG)
_logger.propagate = False


class Log:
    """Minimal structured logger used throughout the project."""

    @staticmethod
    def info(msg, *args, **kwargs):
        _logger.info(msg, *args, **kwargs)

    @staticmethod
    def warning(msg, *args, **kwargs):
        _logger.warning(msg, *args, **kwargs)

    # Alias
    warn = warning

    @staticmethod
    def debug(msg, *args, **kwargs):
        _logger.debug(msg, *args, **kwargs)

    @staticmethod
    def error(msg, *args, **kwargs):
        _logger.error(msg, *args, **kwargs)

    @staticmethod
    def time():
        return time()


def timer(sync_cuda=False, mem=False, loop=1):
    """
    Args:
        func: function
        sync_cuda: bool, whether to synchronize cuda
        mem: bool, whether to log memory
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if mem:
                start_mem = torch.cuda.memory_allocated() / 1024**2
            if sync_cuda:
                torch.cuda.synchronize()

            start = Log.time()
            for _ in range(loop):
                result = func(*args, **kwargs)

            if sync_cuda:
                torch.cuda.synchronize()
            if loop == 1:
                message = f"{func.__name__} took {Log.time() - start:.3f} s."
            else:
                message = f"{func.__name__} took {(Log.time() - start) / loop:.3f} s. (loop={loop})"

            if mem:
                end_max_mem = torch.cuda.max_memory_allocated() / 1024**2
                message += f" Start_Mem {start_mem:.1f} Max {end_max_mem:.1f} MB"
            Log.info(message)

            return result

        return wrapper

    return decorator


def timed(fn):
    """example usage: timed(lambda: model(inp))"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000
