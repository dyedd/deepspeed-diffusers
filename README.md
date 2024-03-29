# 🧩 Deepspeed-diffusers

**`deepspeed-diffusers`** 是一个结合Deepspeed和Diffusers库来训练扩散模型的仓库。

Diffusers是 Huggingface 出品的最集成的预训练扩散模型的首选库。然而由于产品的绑定，目前许多用于 Diffusers 并行训练的脚本很大部分都
是通过 Huggingface 另一个产品 Accelerate 集成的。
> [!IMPORTANT]
> 尽管 Accelerate 很强大了，但截止目前本项目的发布，Accelerate明确表示并不完全支持 Deepspeed。
> 
> 并且让人疑惑的是，Deepspeed仓库中用来示例 Stable Diffusion 的脚本竟然是 Accelerate?🤔
> 
> 因此，本项目就这么诞生了。
> 
> 但实际在本项目发布之前，借鉴了[中科院一大佬发布的deepspeed训练SD，但包含DeepSpeed训练内容、训练方式、流程不完整](https://github.com/OvJat/DeepSpeedTutorial) 、[使用deepspeed来训练源码版本](https://github.com/afiaka87/latent-diffusion-deepspeed)，感谢这些项目的付出！

## 最近更新 🔥 
- [2024/03/29] **`deepspeed-diffusers`** 发布了，支持 Unet 全量微调。

## 演示