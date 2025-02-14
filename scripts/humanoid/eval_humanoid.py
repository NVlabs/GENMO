import sys

import hydra

sys.path.append("third-party/PHC_Lab/")
from omegaconf import DictConfig, OmegaConf

from phc.humanoid import Humanoid


@hydra.main(
    version_base=None,
    config_path="../../third-party/PHC_Lab/phc/cfg",
    config_name="config",
)
def main(cfg: DictConfig):
    # Initialize the humanoid with the hydra config
    humanoid = Humanoid(
        cfg_hydra=cfg,
        seed=0,
    )

    humanoid.eval()
    humanoid.close()


if __name__ == "__main__":
    main()
