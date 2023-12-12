import hydra
from omegaconf import DictConfig

from runners.iris_trainer import Trainer


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
