import hydra
from f_ppsci import eval
from f_ppsci import train
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="./config", config_name="f.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        eval(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
