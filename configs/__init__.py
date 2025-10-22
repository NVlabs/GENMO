import argparse
import os

from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore

os.environ["HYDRA_FULL_ERROR"] = "1"

MainStore = ConfigStore.instance()


def parse_args_to_cfg():
    """
    Use minimal Hydra API to parse args and return cfg.
    This function don't do _run_hydra which create log file hierarchy.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", "-cn", default="train")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    # Cfg
    with initialize_config_module(version_base="1.3", config_module="configs"):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    return cfg
