import argparse
import time
import os

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from matplotlib import pyplot as plt
from torch import autocast
from utils import load_training_config


def sample(cfg, weightst_dir, result_dir, prompt, negative_prompt, show_baseline, compare_weights, torch_dtype=torch.float16):
    # 基准
    pipe_baseline = StableDiffusionPipeline.from_pretrained(cfg.pretrained_model_name_or_path, local_files_only=True,
                                                            # use_safetensors=True,
                                                            torch_dtype=torch_dtype,
                                                            safety_checker=None, requires_safety_checker=False)
    pipe_baseline.scheduler = DPMSolverMultistepScheduler.from_config(pipe_baseline.scheduler.config)
    pipe_baseline.to("cuda")

    # 微调后的
    pipe = StableDiffusionPipeline.from_pretrained(cfg.pretrained_model_name_or_path, local_files_only=True,
                                                   use_safetensors=True,
                                                   torch_dtype=torch_dtype,
                                                   safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)



    if not compare_weights:
        if cfg.use_lora.action:
            pipe.unet.load_attn_procs(weightst_dir)
        else:
            pipe.unet = UNet2DConditionModel.from_pretrained(weightst_dir, local_files_only=True,
                                                             use_safetensors=True,
                                                             torch_dtype=torch_dtype,
                                                             safety_checker=None, requires_safety_checker=False)
    pipe.to("cuda")
    os.makedirs(result_dir, exist_ok=True)
    with autocast("cuda"):
        if compare_weights:
            all_images = []
            global_steps = []
            for subdir in os.listdir(weights_dir):
                if "global_step" in subdir:
                    global_steps.append(subdir)
                    subdir_path = os.path.join(weights_dir, subdir)
                    for weights_file in os.listdir(subdir_path):
                        if cfg.use_lora.action:
                            pipe.unet.load_attn_procs(os.path.join(subdir_path, weights_file))
                        else:
                            pipe.unet = UNet2DConditionModel.from_pretrained(os.path.join(subdir_path, weights_file),
                                                                             local_files_only=True,
                                                                             use_safetensors=True,
                                                                             torch_dtype=torch_dtype,
                                                                             safety_checker=None,
                                                                             requires_safety_checker=False)
                        images = []
                        for step in [1,5,10,15,20,25,30,35]:
                            image = pipe(prompt[0], num_inference_steps=step, negative_prompt=negative_prompt[0],
                                     clip_skip=2,
                                     guidance_scale=8,
                                    generator=torch.Generator(device="cuda").manual_seed(58642352)).images[0]
                            images.append(image)
                        images = [np.array(im) for im in images]
                        combined_image = np.hstack(images)
                        all_images.append(combined_image)
            safe_prompt = "-".join(word for word in prompt[0].split())
            all_images = np.vstack(all_images)
            combined_image = Image.fromarray(all_images)
            fig, ax = plt.subplots(figsize=(20, 20))
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            ax.imshow(combined_image)
            ax.set_xlabel('Inference Steps')
            ax.set_ylabel('Global Steps')

            image_width = combined_image.size[0] // len([1,5,10,15,20,25,30,35])
            x_ticks = [image_width * (i + 0.5) for i in range(len([1,5,10,15,20,25,30,35]))]
            plt.xticks(x_ticks, [1,5,10,15,20,25,30,35])

            image_height = combined_image.size[1] // len(global_steps)
            y_ticks = [image_height * (i + 0.5) for i in range(len(global_steps))]
            plt.yticks(y_ticks, global_steps)
            plt.savefig(f"{result_dir}/{safe_prompt}_combined_{time.time()}.png")
        else:
            if show_baseline:
                print("Baseline:")
                images1 = pipe_baseline(prompt, num_inference_steps=35,negative_prompt=negative_prompt, clip_skip=2, guidance_scale=8).images
                for prompt_text, im in zip(prompt, images1):
                    safe_prompt = "-".join(word for word in prompt_text.split())
                    im.save(f"{result_dir}/baseline_{safe_prompt}_{time.time()}.png")

            print("微调：")
            images2 = pipe(prompt, num_inference_steps=35, negative_prompt=negative_prompt, clip_skip=2, guidance_scale=8).images
            for prompt_text, im in zip(prompt, images2):
                safe_prompt = "-".join(word for word in prompt_text.split())
                im.save(f"{result_dir}/tuned_{safe_prompt}_{time.time()}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='推理测试脚本')
    parser.add_argument('--cfg', type=str, default="./cfg/default.json", help='配置文件路径')
    parser.add_argument('--weights', type=str,
                        default="/home/dcuuser/dxm/diffusers/stable_diffusion/sd-pokemon/global_step9404/unet_ema",
                        help='训练文件路径')
    parser.add_argument('--result', type=str,
                        default="/home/dcuuser/dxm/diffusers/stable_diffusion/results/509-35-fp16",
                        help='保存图片文件路径')
    parser.add_argument('--show_baseline', action='store_true', default=False, help='是否显示Baseline图像')
    parser.add_argument('--compare_weights', action='store_true', default=False, help='是否比较权重，请把weights改成global_step前的目录')
    args = parser.parse_args()
    weights_dir = args.weights
    result_dir = args.result
    show_baseline = args.show_baseline
    compare_weights = args.compare_weights
    cfg_path = args.cfg
    cfg = load_training_config(cfg_path)
    prompt = [
        "pokemon style, fire pokemon, dragon, full body, masterpiece, high quality, best quality, high-definition, ultra-detailed, simple background",
        "pokemon style, ice, bear, wings, pink eyes, full body, masterpiece, high quality, best quality, high-definition, ultra-detailed, simple background",
        "pokemon style, fat bee, full body, masterpiece, high quality, best quality, high-definition, ultra-detailed, simple background",
        # # # 水晶海马
        # "crystal seahorse",
        # # # 火焰狐狸
        # "flame fox",
        # # # 雷霆龙
        # "thunder dragon",
        # # #花精灵
        # "flower fairy",
        # # # 冰晶熊
        # "ice-crystal bear",
    ]
    negative_prompt=["nsfw, human, 1boy, 1girl, (worst quality, low quality:1.4), (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0), greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text, title, logo, signature,watermark,username, artist name, bad anatomy"]*len(prompt)
    sample(cfg, weights_dir, result_dir, prompt, negative_prompt, show_baseline, compare_weights)
