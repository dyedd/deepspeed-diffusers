import random
import time
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from torchvision import transforms
import deepspeed
from loguru import logger as logging
from datasets import load_dataset
import argparse
import os
from model import Diffusion
from deepspeed_config import  deepspeed_config_from_args
from utils import plot_loss_curve_and_save, save_model, load_training_config


# os.environ['https_proxy']="http://127.0.0.1:7890"


def main():
    parser = argparse.ArgumentParser(description='deepspeed训练SD脚本')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='传递给deepspeed的hook,程序启动会自动补充')
    parser.add_argument('--cfg', type=str, default="./cfg.json", help='配置文件路径')
    args = parser.parse_args()
    cfg_path = args.cfg
    cfg = load_training_config(cfg_path)

    # 初始化分布式训练
    deepspeed.init_distributed()
    # 初始化数据集
    logging.debug("init dataset")
    # 加载数据集
    dataset = load_dataset(cfg.dataset_dir)
    column_names = dataset["train"].column_names
    image_column, caption_column = column_names

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
        inputs = model.tokenizer(
            captions, max_length=model.tokenizer.model_max_length, padding="max_length", truncation=True,
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

    # 如果设置一个gpu转换会怎么样？
    train_dataset = dataset['train'].with_transform(preprocess_train)


    # 初始化模型
    logging.debug("init model")
    if cfg.use_fp16:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    model = Diffusion(cfg.pretrained_model_name_or_path, weight_dtype=weight_dtype, is_lora=cfg.lora, rank=int(os.environ["WORLD_SIZE"]))

    # 初始化引擎
    logging.debug("init engine")
    deepspeed_config = deepspeed_config_from_args(cfg)
    parameters = filter(lambda p: p.requires_grad, model.unet.parameters())
    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters,
        training_data=train_dataset,
        collate_fn=collate_fn,
        config_params=deepspeed_config,
    )

    local_rank = model_engine.local_rank

    # 加载检查点
    if os.path.exists(cfg.output_dir):
        model_engine.load_checkpoint(f"./{cfg.output_dir}/")

    # train
    start_time = time.time()
    for epoch in range(cfg.num_epochs) :
        model_engine.train()
        # 记录训练开始时间
        last_time = time.time()
        running_loss = 0.0
        for i, data in enumerate(training_dataloader):
            images,texts = data['pixel_values'].to(model_engine.device,dtype=weight_dtype), data['input_ids'].to(model_engine.device, dtype=torch.long)
            loss = model_engine(images, texts)
            model_engine.backward(loss)
            model_engine.step()
            running_loss += loss.item()
            if i % cfg.log_interval == 0:
                if local_rank == 0:
                    used_time = time.time() - last_time
                    logging.info(
                        f"[epoch: {epoch + 1 : d}, step: {i + 1 : 5d}] Loss: {running_loss/cfg.log_interval : .3f} Time/Batch: {used_time/cfg.log_interval:6.4f}s")
                    last_time = time.time()
                running_loss = 0.0
            # save checkpoint
            model_engine.save_checkpoint(f"{cfg.output_dir}")
    if local_rank == 0:
        logging.info(f"Total training time: {time.time() - start_time:6.4f}s")
        if not cfg.lora:
            pipeline = StableDiffusionPipeline.from_pretrained(
                cfg.pretrained_model_name_or_path,
                text_encoder=model_engine.text_encoder,
                vae=model_engine.vae,
                unet=model_engine.unet,
                variant=weight_dtype,
            )
            pipeline.save_pretrained(cfg.output_dir)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()

