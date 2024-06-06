import random

import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms


def load_custom_dataset(cfg, tokenizer, image_column_mapping=None, caption_column_mapping=None):
    # 加载数据集
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
        if caption_column_mapping and caption_column_mapping.startswith("zh"):
            padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
            return {
                "pixel_values": pixel_values,
                "input_ids": padded_tokens.input_ids,
                "attention_mask": padded_tokens.attention_mask
            }
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    return dataset['train'].with_transform(preprocess_train), collate_fn
