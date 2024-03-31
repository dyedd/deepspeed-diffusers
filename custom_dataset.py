import torch
from torchvision import transforms
from datasets import load_dataset
import random
import numpy as np

def load_custom_dataset(cfg, model):
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

    return dataset['train'].with_transform(preprocess_train), collate_fn