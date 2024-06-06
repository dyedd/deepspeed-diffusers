# 🧩 Deepspeed-diffusers

*Read this in [English](README_en.md).*

**`deepspeed-diffusers`** 是一个结合`Deepspeed`和`Diffusers`库来训练扩散模型（Diffusion Models）的项目。

`Diffusers`是目前最受欢迎的预训练扩散模型的首选集成库。然而由于产品的绑定，许多用于`Diffusers`
并行训练的脚本绝大部分都是通过`Huggingface`另一个产品`Accelerate`集成的。

> [!IMPORTANT]
> `Accelerate`同样也是一个统一API操作的库，但截止目前本项目的发布，`Accelerate`明确表示并不完全支持`Deepspeed`。
>
> 并且让人疑惑的是，`Deepspeed`仓库中用来示例`Stable Diffusion`的脚本竟然是`Accelerate`?🤔
>
> 为了充分发挥`Deepspeed`的能力，本项目就这么诞生了。
> 
> 据我所知，这是开源的第一个完全使用Deepspeed的扩散模型数据并行/Zero框架

此外，本项目也借鉴了[OvJat](https://github.com/OvJat/DeepSpeedTutorial) 、[afiaka87](https://github.com/afiaka87/latent-diffusion-deepspeed)、[liucongg](https://github.com/liucongg/ChatGLM-Finetuning)
，在此感谢这些项目的付出！

在这些项目中，本项目的优势是：

1. 充分使用`Deepspeed`的能力
2. 与主流的`Diffusers`对齐
3. 可以使用`Unet`和`Lora`两种模式训练 
4. 等等，等待您的发掘~

## 最近更新 🔥

- [2024/05/30] 类似`Accelerate`示例，不降stable diffusion类完全加入DeepSpeed，减少了大概3GB+的显存
- [2024/05/25] 支持数据集自定义，支持EMA
- [2024/04/18] 修复训练文本不匹配的问题，增加推理验证
- [2024/04/10] 支持`slurm`多节点和单节点训练。
- [2024/04/01] 支持`wandb`。
- [2024/03/31] 支持`lora`微调，修复了全量微调后的生成图片总是黑色图片的问题，增加了`slurm`单节点训练脚本
- [2024/03/29] **`deepspeed-diffusers`** 发布了，支持`Unet`全量微调。

## 演示

### 安装依赖

在运行脚本之前，请确定安装了所有的依赖：

- 请保证源码是最新的：`git clone https://github.com/dyedd/deepspeed-diffusers`
- 然后`cd`到文件夹并运行：`pip -r install requirements.txt`

### [宝可梦数据集](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)示例

> 推荐直接把数据集下载到本地，否则通过脚本自动下载的是缓存文件，每次运行都要去请求`huggingface`。

根据下载的目录，修改cfg.json的`dataset_dir`内容。

### 下载权重

接下来的示例结果都是在[stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)的权重下实验的。

注意：如果您使用 [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768
模型，请将cfg.json的`resolution`更改为 768。

在此，仍然强烈您自己`git clone`权重，不要通过huggingface自动下载~然后修改cfg.json的`pretrained_model_name_or_path`。

### 训练

本项目支持2种训练模式。

1. 全量微调unet，在混合精度下，迭代210次，显存大致在12到20GB。
2. lora+unet，在混合精度下，迭代210次，显存大致在4到9.63GB。

将`cfg/default`的`use_lora.action`修改成`true`即可支持模式2。

此外，模式2的权重模型保存在`use_lora.output_dir`，模式1保存在`output_dir`，名称都为`ckpt_name`。

cfg.json的配置其实都很清楚(key就是原意)，从`use_fp16`
开始都是与deespeed有关的配置，如果有添加，不要忘记同时修改`utils.py`的`deepspeed_config_from_args`函数。

> 如果你在本地，可以直接通过`bash scripts/train_text_to_image.sh`运行脚本；
>
>如果你在slurm系统，在修改部分信息后，可以通过`sbatch scripts/train_text_to_image.slurm`下提交。

### 推理/采样
同样支持本地和Slurm调度，命令分别为
```
bash scripts/test_text_to_image.sh
sbatch scripts/test_text_to_image.slurm
```

## 可能出现的问题

> [!NOTE]
> 1.
生成的图像都是黑色图片或者报错`RuntimeWarning: invalid value encountered in cast images = (images * 255).round().astype("uint8")`
>
> 请注意，本项目发布的`cfg.json`中关于优化器，学习率的参数仅对于宝可梦这个数据集而言。这个问题是因为优化器，学习率不适合训练集，而造成训练的损失一直是none。

2. 提交的`slurm`脚本运行时间为0秒就退出
> 这是因为`slurm`脚本里写了日志保存在`log`文件夹，而该文件夹目前不存在，就无法运行。

## 引用
如果您在论文和项目中使用了`Deepspeed-diffusers`，请使用以下`BibTeX`引用它。
```
@Misc{Deepspeed-diffusers,
  title =        {Deepspeed-diffusers: training diffusers with deepspeed.},
  author =       {Ximiao Dong},
  howpublished = {\url{https://github.com/dyedd/deepspeed-diffusers}},
  year =         {2024}
}
```