import os
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import datetime
import pathlib
import torch
import shutil
import glob

DEEPSPEED_CP_AUX_FILENAME = 'auxiliary.pt'
KEEP_N_CHECKPOINTS = 10


def load_training_config(config_path: str):
    data_dict = OmegaConf.load(config_path)
    return data_dict


def plot_loss_curve_and_save(loss_list):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Training Loss')
    plt.title('Loss Curve')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    # 生成基于当前时间的唯一文件名
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"loss_curve_{current_time}.png"
    # 保存图像文件
    plt.savefig(filename)
    plt.close()  # 关闭绘图窗口，防止在后台累积过多图像

def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, pathlib.Path):
        cp_path = pathlib.Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = pathlib.Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir

### Checkpointing
@torch.no_grad()
def save_model(model, path: str, epoch=0, opt=None):
    save_obj = {'epoch': epoch, }
    cp_dir = cp_path_to_dir(path, 'ds')
    if KEEP_N_CHECKPOINTS is not None:
        checkpoints = sorted(glob.glob(str(cp_dir / "global*")), key=os.path.getmtime, reverse=True)
        for checkpoint in checkpoints[KEEP_N_CHECKPOINTS:]:
            shutil.rmtree(checkpoint)

    model.save_checkpoint(cp_dir, client_state=save_obj)
    # Save a nonsense value that directs the user to convert the checkpoint to a normal 32-bit pytorch model.
    save_obj = {
        **save_obj,
        'weights': (
            'To get a working standard checkpoint, '
            'look into consolidating DeepSpeed checkpoints.'
        ),
    }
    torch.save(save_obj, str(cp_dir / DEEPSPEED_CP_AUX_FILENAME))
    save_obj = { **save_obj, 'weights': model.state_dict(), }
    if opt is not None:
        save_obj = { **save_obj, 'opt_state': opt.state_dict(), }
    torch.save(save_obj, path)