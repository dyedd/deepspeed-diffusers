import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import argparse
from utils import load_training_config


def sample(cfg, prompt, torch_dtype=torch.float16):
    if cfg.use_lora.action:
        pipe = StableDiffusionPipeline.from_pretrained(cfg.pretrained_model_name_or_path, local_files_only=True,
                                                       use_safetensors=True,
                                                       torch_dtype=torch_dtype,
                                                       safety_checker=None, requires_safety_checker=False)
        pipe.unet.load_attn_procs(cfg.use_lora.output_dir + "/" + cfg.ckpt_name + ".safetensor")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(cfg.output_dir + '/' + cfg.ckpt_name,
                                                       use_safetensors=True,
                                                       safety_checker=None, requires_safety_checker=False)
    pipe.to("cuda")
    scale = 7.5
    n_samples = 4

    with autocast("cuda"):
        images = pipe(n_samples * [prompt], guidance_scale=scale).images

    for idx, im in enumerate(images):
        im.save(f"{idx:06}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='采样/推理测试脚本')
    parser.add_argument('--cfg', type=str, default="./cfg.json", help='配置文件路径')
    args = parser.parse_args()
    cfg_path = args.cfg
    cfg = load_training_config(cfg_path)
    prompt = "A pokemon with green eyes and red legs."
    sample(cfg, prompt)
