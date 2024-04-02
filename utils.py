import random

import numpy as np
import torch
from deepspeed import get_accelerator
from omegaconf import OmegaConf
from transformers import set_seed


def load_training_config(config_path: str):
    data_dict = OmegaConf.load(config_path)
    return data_dict


def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    elif is_rank_0():
        print(msg)


def is_rank_0():
    """检测是否rank 0."""
    # 全局rank，单节点就是local_rank
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
