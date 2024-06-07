# 🧩 Deepspeed-diffusers

*Read this in [English](README_en.md).*

`deepspeed-diffusers` 是一个原生结合`Deepspeed`、`Diffusers`、`Peft`库来训练扩散模型（Diffusion Models）的项目。

## 动机
训练过`Diffusers`的开发者都知道目前已经有了`Huggingface`另一个产品`Accelerate`来并行训练扩散模型。

为什么我还开源了这个项目呢？
- 一个非常滑稽的事实是当初我没有仔细看到`diffusers`的训练示例文件是部分支持`Zero3`而不是不支持。本项目最后实现下来，结果`Zero3`实现思路与其不谋而和。
- 可以安慰自己的是，虽然现在集成框架很多，使用起来也很便捷，但是集成的东西太多了，就不容易二次开发了！
- `Slurm`调度系统的脚本起码我也提供了，使用和修复了`Deepspeed`本来就支持却又不太行的`Slurm`启动器。

> 为了充分单独发挥`Deepspeed`的能力，本项目就这么诞生了。

此外，本项目在原生结合的时候也借鉴了[OvJat](https://github.com/OvJat/DeepSpeedTutorial) 、[afiaka87](https://github.com/afiaka87/latent-diffusion-deepspeed)、[liucongg](https://github.com/liucongg/ChatGLM-Finetuning)的代码，在此非常感谢这些项目的付出！

在这些项目中，本项目的缺点是什么？

1. 我在实现的时候，`ZeRO-1,2`也只支持`UNet2DConditionModel`。目前我还没有发现我差别人具体在哪，还得充分测试。
2. 无法保存Lora训练时候的检测点，这和`Peft`库太过集成绑定有些关系，除非自己写个Lora微调逻辑。

除此之外，`Accelerate`有的功能本项目也有，本项目主打一个简洁实用🥹

## 最近更新 🔥

- [2024/05/30] 类似`Accelerate`示例，不把`stable diffusion`类完全加入DeepSpeed，仅加载`Unet`，这样减少了大概3GB+的显存
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

> 强烈推荐直接把数据集下载到本地，而不是通过脚本自动下载的是缓存文件。否则哪天断网也要断`Huggingface`的数据反馈以及对于国内环境不是非常友好。

根据下载的目录，修改cfg.json的`dataset_dir`内容。

### 下载权重

接下来的示例结果都是在[stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)的权重下实验的。

注意：如果您使用 [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768
模型，请将`cfg/default.json`的`resolution`更改为 768。

在此，仍然强烈自己下载`git clone`权重，不要通过huggingface自动下载~然后修改`cfg/default.json`的`pretrained_model_name_or_path`。


### 配置文件解读
项目所需的参数都写在了`cfg/default.json`：
以下是每个字段的解释：
- "num_epochs": 训练的总轮数。
- "validation_epochs": 每多少轮进行一次验证。
- "max_train_steps": 限制最大训练步数，0表示不限制。
- "lr_warmup_steps": 学习率预热的步数。
- "save_interval": 模型保存的间隔步数。
- "seed": 随机种子，用于确保实验的可重复性。
- "validation_prompts": 验证时使用的提示。
- "pretrained_model_name_or_path": 预训练模型的名称或路径。
- "dataset_dir": 数据集的目录。
- "imagefolder": 是否使用包含**图像**的文件夹作为数据源。
- "checkpoint_dir": 检查点（模型状态）保存的目录。
- "output_dir": 输出（如训练日志、生成的图像等）的目录。
- "use_lora": 是否使用LoRA（Long Range Attention）技术，以及相关的参数。
  - "action": 是否启用LoRA。
  - "rank": LoRA的秩。
  - "alpha": LoRA的α参数。
- "dataloader_num_workers": 数据加载器的工作线程数。
- "resume_from_checkpoint": 是否从检查点恢复训练。
- "use_ema": 是否使用指数移动平均（Exponential Moving Average）。
- "offline": 是否在`wandb`离线模式下运行。
- "resolution": 图像的分辨率。
- "center_crop": 是否对图像进行中心裁剪。
- "random_flip": 是否对图像进行随机翻转。
- 之后都是`DeepSpeed`常用的配置，如有不解，请查看官方文档。如果有添加，不要忘记同时修改`utils.py`的`deepspeed_config_from_args`函数。

### 训练

本项目支持2种训练模式。

- 全量微调unet，在混合精度FP16下，如果不开启Zero，每张卡的bacth_size为4, 显存大致在11.61到23.32GB。
- Lora+unet，在混合精度FP16下，如果不开启Zero，每张卡的bacth_size为4, 显存大致在2到13GB。

> 只要开启梯度累积和Zero，那么全量微调也能实现实现16GB以下的显卡训练了！


如果你在本地，可以直接通过`bash scripts/train_text_to_image.sh`运行脚本；

如果你在slurm系统，在修改部分信息后，可以通过`sbatch scripts/train_text_to_image.slurm`下提交。

### 推理/采样
同样支持本地和Slurm调度，命令分别为
```
bash scripts/test_text_to_image.sh
sbatch scripts/test_text_to_image.slurm
```

## 可能出现的问题

> [!NOTE]
> 1. 生成的图像都是黑色图片或者报错`RuntimeWarning: invalid value encountered in cast images = (images * 255).round().astype("uint8")`
>
> 请注意，本项目配置文件中关于优化器，学习率的参数仅对于宝可梦这个数据集而言。这个问题是因为优化器，学习率不适合训练集，而造成训练的损失一直是none。
> 2. 提交的`slurm`脚本运行时间为0秒就退出
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