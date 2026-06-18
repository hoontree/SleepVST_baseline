import sys

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig

from src.data.preprocess.motion_extractor import build_motion_processing_config, process_all_videos
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def _log_processing_summary(results):
    successful = sum(1 for r in results if r.startswith("Success"))
    skipped = sum(1 for r in results if r.startswith("Skipped"))
    failed = sum(1 for r in results if r.startswith("Failed") or r.startswith("Error"))

    logger.info("=" * 50)
    logger.info("PROCESSING SUMMARY")
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
    config = instantiate(cfg.config)

    logger.info(f"Starting motion feature extraction with {config.num_workers} workers")
    logger.info(f"Video path: {config.video_input_path}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Target FPS: {config.target_fps}")
    logger.info(f"Output size: {config.output_size}")

    results = process_all_videos(config)
    if not results:
        logger.error("No videos were processed. Exiting.")
        sys.exit(1)

    _log_processing_summary(results)
    logger.info("Motion feature extraction completed")


@hydra.main(version_base=None, config_path="../config", config_name="preprocess/motionfeatures")
def main(cfg: DictConfig):
    setup_logging(cfg.log.dir, cfg.log.name)
    run(cfg)


if __name__ == "__main__":
    main()
