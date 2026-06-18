import hydra
from omegaconf import DictConfig
from src.train.transfer import transfer_to_video


@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def main(cfg: DictConfig):
    transfer_to_video(cfg)

if __name__ == "__main__":
    main()