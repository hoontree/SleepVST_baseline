import hydra
import sys
import os
from omegaconf import DictConfig, OmegaConf
from src.common.logger import Logger
from src.common.utils import setup_device
from src.train.loop import pretrain, finetune, test
from src.train.transfer import transfer_to_video
from src.data.registry import get_dataset
from src.data.datasets.KVSS import KVSS
from src.data.datasets.MESA import MESA
from src.data.datasets.SHHS import SHHS
from src.data.base_datamodule import BaseDataModule
from src.models.registry import get_model
from torch.utils.data import ConcatDataset
from src.data.preprocess.motion_extractor import process_all_videos

@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def main(cfg: DictConfig):
    cfg_log = cfg.log
    logger = Logger(cfg_log.dir, name=cfg_log.name)
    gpu_ids = cfg.system.get("gpu_ids", "4")
    device = setup_device(cfg.system.device, gpu_ids)
    
    # Setup wandb if needed
    wandb_run = None
    if cfg.command in ["pretrain", "finetune"]:
        try:
            import wandb
            wandb_key = os.environ.get("WANDB_API_KEY")
            if wandb_key:
                wandb.login(key=wandb_key)
                wandb_run = wandb.init(
                    project=cfg.get("wandb", {}).get("project", "SleepVST"),
                    entity=cfg.get("wandb", {}).get("entity", None),
                    name=cfg.train.get("log_name", cfg.command),
                    config=OmegaConf.to_container(cfg, resolve=True)
                )
                logger.info("W&B logging enabled")
            else:
                logger.info("W&B API key not found, skipping W&B logging")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
    
    if cfg.command == "pretrain":
        logger.info("Starting pretrain command")
        pretrain(cfg, logger, wandb_run)
        if wandb_run:
            wandb_run.finish()

    elif cfg.command == "finetune":
        logger.info("Starting finetune command")
        finetune(cfg, logger, wandb_run)
        if wandb_run:
            wandb_run.finish()
    
    elif cfg.command == "test":
        logger.info("Starting test command")
        test(cfg, logger)
    elif cfg.command == "preprocess":
        if cfg.get("dataset", {}).get("name") == "KISS" or cfg.get("preprocess", {}).get("dataset", {}).get("name") == "KISS":
            from src.data.preprocess.preprocess_SNUH import process_snuh_dataset
            logger = Logger(cfg.log.dir, name=cfg.log.name)
            logger.info("Starting SNUH/KISS dataset preprocessing")

            # Use preprocess config if available, otherwise use root config
            preprocess_cfg = cfg.preprocess if hasattr(cfg, 'preprocess') else cfg
            results = process_snuh_dataset(preprocess_cfg, logger)

            logger.info("SNUH preprocessing completed")
            logger.info(f"Results: {results}")
        else:
            logger = Logger(cfg.log.dir, name=cfg.log.name)
            logger.error(f"Unknown dataset for preprocessing: {cfg.get('dataset', {}).get('name', 'Unknown')}")
            sys.exit(1)
    elif cfg.command == "preprocess_respiratory_edf":
        from src.data.preprocess.preprocess_respiratory_edf import process_respiratory_edf_dataset
        logger = Logger(cfg.log.dir, name=cfg.log.name)
        logger.info("Starting Respiratory EDF dataset preprocessing")
        logger.info("Using respiratory_extraction.py signal processing pipeline")

        # Use preprocess config if available, otherwise use root config
        preprocess_cfg = cfg.preprocess if hasattr(cfg, 'preprocess') else cfg
        results = process_respiratory_edf_dataset(preprocess_cfg, logger)

        logger.info("Respiratory EDF preprocessing completed")
        logger.info(f"Results: {results}")
    elif cfg.command == "motionfeatures":
        cfg = cfg.preprocess
        logger = Logger(cfg.log.dir, name=cfg.log.name)
        logger.info(f"Starting motion feature extraction with {cfg.num_workers} workers")
        logger.info(f"Video path: {cfg.video.input_path}")
        logger.info(f"Output directory: {cfg.output.dir}")
        logger.info(f"Target FPS: {cfg.target_fps}")
        logger.info(f"Output size: {cfg.output.size}")
        
        # Process videos
        results = process_all_videos(cfg, logger)
        
        if not results:
            logger.error("No videos were processed. Exiting.")
            sys.exit(1)
        
        # Summary
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
        
        # Log failed videos for debugging
        failed_videos = [r for r in results if r.startswith("Failed") or r.startswith("Error")]
        if failed_videos:
            logger.info("Failed videos:")
            for failed in failed_videos:
                logger.info(f"  {failed}")
        
        logger.info("Motion feature extraction completed")
        pass
    elif cfg.command == "extract_respiratory":
        from src.data.preprocess.respiratory_pipeline_mp import process_all_videos as process_respiratory
        cfg = cfg.preprocess
        logger = Logger(cfg.log.dir, name=cfg.log.name)
        logger.info("Starting respiratory signal extraction")
        logger.info(f"Video path: {cfg.video.input_path}")
        logger.info(f"Output directory: {cfg.output.dir}")
        logger.info(f"Magnification factor: {cfg.magnification.mag_factor}")
        logger.info(f"Frequency range: {cfg.magnification.freq_range}")

        # Log multiprocessing configuration
        mp_enabled = cfg.get('multiprocessing', {}).get('enabled', False)
        if mp_enabled:
            num_workers = cfg.get('multiprocessing', {}).get('num_workers', None)
            logger.info(f"Multiprocessing: ENABLED")
            if num_workers:
                logger.info(f"Number of workers: {num_workers}")
            else:
                logger.info(f"Number of workers: auto (CPU count - 1)")
        else:
            logger.info(f"Multiprocessing: DISABLED (sequential processing)")

        # Process videos
        results = process_respiratory(cfg, logger)

        if not results:
            logger.error("No videos were processed. Exiting.")
            sys.exit(1)

        # Summary
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

        # Log failed videos for debugging
        failed_videos = [r for r in results if r.startswith("Failed") or r.startswith("Error")]
        if failed_videos:
            logger.info("Failed videos:")
            for failed in failed_videos:
                logger.info(f"  {failed}")

        logger.info("Respiratory signal extraction completed")
    elif cfg.command == "transfer_to_video":
        logger = Logger(cfg.log.dir, name=cfg.log.name)
        transfer_to_video(cfg, logger=logger)
        return
    else:
        return logger.error(f"Unknown command: {cfg.command}")
if __name__ == "__main__":
    main()