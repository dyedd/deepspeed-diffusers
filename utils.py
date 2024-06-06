import random
import logging
import numpy as np
import torch
from deepspeed import get_accelerator
import wandb

from diffusers import StableDiffusionPipeline
from omegaconf import OmegaConf
from transformers import set_seed

from datasets import load_dataset
from torchvision import transforms


def load_custom_dataset(cfg, tokenizer, imagefolder=True, image_column_mapping=None, caption_column_mapping=None):
    # 加载数据集
    if imagefolder:
        dataset = load_dataset("imagefolder", data_dir=cfg.dataset_dir)
    else:
        dataset = load_dataset(cfg.dataset_dir)
    column_names = dataset["train"].column_names
    if image_column_mapping is None:
        image_column = column_names[0]
    else:
        image_column = image_column_mapping
    if caption_column_mapping is None:
        caption_column = column_names[1]
    else:
        caption_column = caption_column_mapping

    train_transforms = transforms.Compose(
        [
            transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(cfg.resolution) if cfg.center_crop else transforms.RandomCrop(cfg.resolution),
            transforms.RandomHorizontalFlip() if cfg.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # 预处理文本的函数
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        return inputs.input_ids

    # 自定义数据转换函数
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    # 自定义批处理函数
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    return dataset['train'].with_transform(preprocess_train), collate_fn


def load_training_config(config_path: str):
    data_dict = OmegaConf.load(config_path)
    return data_dict


def deepspeed_config_from_args(args):
    return {
        # train_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation *GPU
        'train_micro_batch_size_per_gpu': args.train_micro_batch_size_per_gpu,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'fp16': args.use_fp16,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "wall_clock_breakdown": args.wall_clock_breakdown,
        "wandb": args.wandb,
        "flops_profiler": args.flops_profiler,
        "zero_optimization": args.zero_optimization
    }


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


def log_validation(unet, args, device, weight_dtype):
    logging.info("Running validation... ")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        local_files_only=True,
        use_safetensors=True,
        safety_checker=None, requires_safety_checker=False,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if "seed" in args:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
        images.append(image)
    wandb.log(
        {
            "validation": [
                wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                for i, image in enumerate(images)
            ]
        }
    )
    del pipeline
    torch.cuda.empty_cache()
    return images
