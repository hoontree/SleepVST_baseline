import sys

import hydra
from omegaconf import DictConfig

from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def _run_preprocess(cfg: DictConfig):
    """SNUH/KISS dataset preprocessing."""
    dataset_name = cfg.get("dataset", {}).get("name")
    nested_name = cfg.get("preprocess", {}).get("dataset", {}).get("name")
    if dataset_name == "KISS" or nested_name == "KISS":
        from src.data.preprocess.preprocess_SNUH import process_snuh_dataset
        logger.info("Starting SNUH/KISS dataset preprocessing")

        preprocess_cfg = cfg.preprocess if hasattr(cfg, "preprocess") else cfg
        results = process_snuh_dataset(preprocess_cfg)

        logger.info("SNUH preprocessing completed")
        logger.info(f"Results: {results}")
    else:
        logger.error(f"Unknown dataset for preprocessing: {dataset_name or 'Unknown'}")
        sys.exit(1)


def _run_preprocess_respiratory_edf(cfg: DictConfig):
    """Respiratory EDF dataset preprocessing."""
    from src.data.preprocess.preprocess_respiratory_edf import process_respiratory_edf_dataset
    logger.info("Starting Respiratory EDF dataset preprocessing")
    logger.info("Using respiratory_extraction.py signal processing pipeline")

    preprocess_cfg = cfg.preprocess if hasattr(cfg, "preprocess") else cfg
    results = process_respiratory_edf_dataset(preprocess_cfg)

    logger.info("Respiratory EDF preprocessing completed")
    logger.info(f"Results: {results}")


def run(cfg: DictConfig):
    if cfg.command == "preprocess":
        _run_preprocess(cfg)
    elif cfg.command == "preprocess_respiratory_edf":
        _run_preprocess_respiratory_edf(cfg)
    else:
        logger.error(
            f"Unknown preprocessing command: {cfg.command}. "
            "Use `python -m src.cli_motionfeatures` for motion features, "
            "`python -m src.cli_extract_respiratory` for video respiratory extraction, "
            "or `python -m src.cli_train` for training."
        )


@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def main(cfg: DictConfig):
    setup_logging(cfg.log.dir, cfg.log.name)
    run(cfg)


if __name__ == "__main__":
    main()
