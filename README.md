# 🧩 Deepspeed-diffusers

*Read this in [English](README_en.md).*

**`deepspeed-diffusers`** 是一个结合`Deepspeed`和`Diffusers`库来训练扩散模型（Diffusion Models）的项目。

`Diffusers`是目前最受欢迎的预训练扩散模型的首选集成库。然而由于产品的绑定，许多用于`Diffusers`并行训练的脚本绝大部分都是通过`Huggingface`另一个产品`Accelerate`集成的。

> [!IMPORTANT]
> `Accelerate`同样也是一个统一API操作的库，但截止目前本项目的发布，`Accelerate`明确表示并不完全支持`Deepspeed`。
> 
> 并且让人疑惑的是，`Deepspeed`仓库中用来示例`Stable Diffusion`的脚本竟然是`Accelerate`?🤔
> 
> 为了充分发挥`Deepspeed`的能力，本项目就这么诞生了。

此外，本项目也借鉴了[OvJat](https://github.com/OvJat/DeepSpeedTutorial) 、[afiaka87](https://github.com/afiaka87/latent-diffusion-deepspeed)、[liucongg](https://github.com/liucongg/ChatGLM-Finetuning)，在此感谢这些项目的付出！

在这些项目中，本项目的优势是：
1. 充分使用`Deepspeed`的能力
2. 与主流的`Diffusers`对齐
3. 流程完整，包含各种额外脚本
4. 等等，等待您的发掘~

## 最近更新 🔥 
- [2024/04/01] 支持`wandb`。
- [2024/03/31] 支持`lora`微调，修复了全量微调后的生成图片总是黑色图片的问题，增加了`slurm`脚本
- [2024/03/29] **`deepspeed-diffusers`** 发布了，支持`Unet`全量微调。

## 演示

1. 只训练unet
2. lora+unet


## 可能出现的问题
> [!NOTE]
> 1. 生成的图像都是黑色图片或者报错`RuntimeWarning: invalid value encountered in cast images = (images * 255).round().astype("uint8")`
>
> 请注意，本项目发布的`cfg.json`中关于优化器，学习率的参数仅对于宝可梦这个数据集而言。这个问题是因为优化器，学习率不适合训练集，而造成训练的损失一直是none。

## 性能展示