"""Backward-compatible entry point for general commands.

Prefer the focused entry points:

  - Training / evaluation       : ``python -m src.cli_train``
  - Dataset preprocessing       : ``python -m src.cli_preprocess``
  - Motion feature extraction   : ``python -m src.cli_motionfeatures``
  - Respiratory video extraction: ``python -m src.cli_extract_respiratory``
"""

import hydra
from omegaconf import DictConfig

from src import cli_preprocess, cli_train
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def main(cfg: DictConfig):
    setup_logging(cfg.log.dir, cfg.log.name)

    if cfg.command in ("preprocess", "preprocess_respiratory_edf"):
        cli_preprocess.run(cfg)
    elif cfg.command == "motionfeatures":
        logger.error("Use `python -m src.cli_motionfeatures` for motion feature extraction.")
    elif cfg.command == "extract_respiratory":
        logger.error("Use `python -m src.cli_extract_respiratory` for respiratory video extraction.")
    elif cfg.command in ("pretrain", "finetune", "test", "transfer_to_video"):
        cli_train.run(cfg)
    else:
        logger.error(f"Unknown command: {cfg.command}")


if __name__ == "__main__":
    main()
