import hydra
from omegaconf import DictConfig

from src.train.transfer import transfer_to_video
from src.utils.logger import setup_logging, get_logger
from src.utils.utils import setup_device

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def main(cfg: DictConfig):
    # Match cli_train.run(): configure file logging and select the GPU before
    # running. Without these the shim logged nowhere and ignored system.gpu_ids.
    setup_logging(cfg.log.dir, cfg.log.name)
    setup_device(cfg.system.device, cfg.system.get("gpu_ids", "4"))
    transfer_to_video(cfg)


if __name__ == "__main__":
    main()
