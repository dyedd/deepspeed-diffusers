from omegaconf import OmegaConf


def load_training_config(config_path: str):
    data_dict = OmegaConf.load(config_path)
    return data_dict
