# 🧩 Deepspeed-diffusers

*Read this in [中文](README.md).*

**`deepspeed-diffusers`** is a project that combines `Deepspeed` and `Diffusers` libraries to train Diffusion Models.

`Diffusers` is currently the most popular integration library of choice for pre-trained diffusion models. However, due to product bindings, the majority of scripts for parallel use with `Diffusers` are integrated through another `Huggingface` product, `Accelerate`.

> [!IMPORTANT]
> `Accelerate` is also a library for unified API operations, but as of the release of this project, `Accelerate` explicitly stated that it does not fully support `Deepspeed`.
>
> Moreover, it is puzzling that the script in the `Deepspeed` repository used to demonstrate `Stable Diffusion` is `Accelerate`?🤔
>
> To fully utilize the capabilities of `Deepspeed`, this project was born.

Additionally, this project also draws inspiration from [OvJat](https://github.com/OvJat/DeepSpeedTutorial), [afiaka87](https://github.com/afiaka87/latent-diffusion-deepspeed), and [liucongg](https://github.com/liucongg/ChatGLM-Finetuning), and we thank these projects for their contributions!

The advantages of this project include:
1. Full utilization of `Deepspeed` capabilities
2. Alignment with mainstream `Diffusers`
3. Complete process, including various additional scripts
4. And more, waiting for you to discover~

## Recent Updates 🔥 
- [2024/03/31] Support for `lora` fine-tuning, fixed the issue where the generated images were always black after full fine-tuning.
- [2024/03/29] **`deepspeed-diffusers`** was released, supporting full fine-tuning of `Unet`.

## Demonstrations

1. Train only unet
2. lora+unet

## Possible Issues
> [!NOTE]
> 1. The generated images are black or an error is reported `RuntimeWarning: invalid value encountered in cast images = (images * 255).round().astype("uint8")`
>
> Please note, the parameters about the optimizer and learning rate in the `cfg.json` released with this project are specifically for the Pokémon dataset. This issue is due to the optimizer and learning rate not being suitable for the training set, causing the training loss to always be none.

## Performance Showcase
