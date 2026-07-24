import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.logger import setup_logging, get_logger
from src.utils.utils import setup_device
from src.train.loop import prepare_pretrain, prepare_finetune, run_test_sets, test
from src.train.transfer import transfer_to_video

logger = get_logger(__name__)


def _setup_wandb(cfg: DictConfig):
    """Initialize a W&B run for training commands, if configured."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping W&B logging")
        return None

    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        logger.info("W&B API key not found, skipping W&B logging")
        return None

    wandb.login(key=wandb_key)
    wandb_run = wandb.init(
        project=cfg.get("wandb", {}).get("project", "SleepVST"),
        entity=cfg.get("wandb", {}).get("entity", None),
        name=cfg.train.get("log_name", cfg.command),
        group=cfg.get("wandb", {}).get("command", None),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    logger.info("W&B logging enabled")
    return wandb_run


def run(cfg: DictConfig):
    """Dispatch a training/eval command. Reusable from the compat shim in cli.py."""
    setup_logging(cfg.log.dir, cfg.log.name)
    gpu_ids = cfg.system.get("gpu_ids", "4")
    device = setup_device(cfg.system.device, gpu_ids)

    if cfg.command == "pretrain":
        logger.info("Starting pretrain command")
        wandb_run = _setup_wandb(cfg)
        setup = prepare_pretrain(cfg, wandb_run)
        setup.trainer.fit(setup.train_loader, setup.val_loader, label="Pretraining")
        run_test_sets(setup.trainer, setup.test_loaders, wandb_run)
        if wandb_run:
            wandb_run.finish()

    elif cfg.command == "finetune":
        # The transfer-oriented default config does not carry the finetune-only
        # keys (train.pretrained_checkpoint, mode=finetune). Fail with guidance
        # instead of a cryptic ConfigKeyError from deep inside prepare_finetune.
        if "pretrained_checkpoint" not in cfg.get("train", {}):
            logger.error(
                "'finetune' is not wired into the default config. Run it with the "
                "dedicated config:\n"
                "    python -m src.cli_train --config-name command2/finetune"
            )
            return
        logger.info("Starting finetune command")
        wandb_run = _setup_wandb(cfg)
        setup = prepare_finetune(cfg, wandb_run)
        setup.trainer.fit(setup.train_loader, setup.val_loader, label="Finetuning")
        run_test_sets(setup.trainer, setup.test_loaders, wandb_run)
        if wandb_run:
            wandb_run.finish()

    elif cfg.command == "test":
        # The default config has no `test:` block (checkpoint / datasets). Fail
        # with guidance instead of a cryptic ConfigKeyError inside test().
        if "test" not in cfg:
            logger.error(
                "'test' is not wired into the default config. Run it with the "
                "dedicated config:\n"
                "    python -m src.cli_train --config-name command2/test"
            )
            return
        logger.info("Starting test command")
        wandb_run = _setup_wandb(cfg)
        test(cfg, wandb_run)
        if wandb_run:
            wandb_run.finish()

    elif cfg.command == "transfer_to_video":
        logger.info("Starting transfer_to_video command")
        transfer_to_video(cfg)

    else:
        logger.error(
            f"Unknown training command: {cfg.command}. "
            "Expected one of: pretrain, finetune, test, transfer_to_video. "
            "For data preprocessing use `python -m src.cli_preprocess`."
        )


@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
