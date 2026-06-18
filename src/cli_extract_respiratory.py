import sys

import hydra
from omegaconf import DictConfig

from src.data.preprocess.respiratory_pipeline_mp import process_all_videos
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def _log_processing_summary(results):
    successful = sum(1 for r in results if r.startswith("Success"))
    skipped = sum(1 for r in results if r.startswith("Skipped"))
    failed = sum(1 for r in results if r.startswith("Failed") or r.startswith("Error"))

    logger.info("=" * 50)
    logger.info("RESPIRATORY EXTRACTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total videos: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 50)

    failed_videos = [r for r in results if r.startswith("Failed") or r.startswith("Error")]
    if failed_videos:
        logger.info("Failed videos:")
        for failed_video in failed_videos:
            logger.info(f"  {failed_video}")


def run(cfg: DictConfig):
    logger.info("Starting respiratory signal extraction")
    logger.info(f"Video path: {cfg.video.input_path}")
    logger.info(f"Output directory: {cfg.output.dir}")
    logger.info(f"Magnification factor: {cfg.magnification.mag_factor}")
    logger.info(f"Frequency range: {cfg.magnification.freq_range}")

    mp_enabled = cfg.get("multiprocessing", {}).get("enabled", False)
    if mp_enabled:
        num_workers = cfg.get("multiprocessing", {}).get("num_workers", None)
        logger.info("Multiprocessing: ENABLED")
        if num_workers:
            logger.info(f"Number of workers: {num_workers}")
        else:
            logger.info("Number of workers: auto (CPU count - 1)")
    else:
        logger.info("Multiprocessing: DISABLED (sequential processing)")

    results = process_all_videos(cfg)
    if not results:
        logger.error("No videos were processed. Exiting.")
        sys.exit(1)

    _log_processing_summary(results)
    logger.info("Respiratory signal extraction completed")


@hydra.main(version_base=None, config_path="../config", config_name="preprocess/respiratory")
def main(cfg: DictConfig):
    setup_logging(cfg.log.dir, cfg.log.name)
    run(cfg)


if __name__ == "__main__":
    main()
