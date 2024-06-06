# 🧩 Deepspeed-diffusers

*Read this in [中文](README.md).*

**`deepspeed-diffusers`** is a project that combines `Deepspeed` and `Diffusers` libraries to train Diffusion Models.

`Diffusers` is currently the most popular integration library of choice for pre-trained diffusion models. However, due
to product bindings, the majority of scripts for Parallel training with `Diffusers` are integrated through
another `Huggingface` product, `Accelerate`.

> [!IMPORTANT]
> `Accelerate` is also a library for unified API operations, but as of the release of this project, `Accelerate`
> explicitly stated that it does not fully support `Deepspeed`.
>
> Moreover, it is puzzling that the script in the `Deepspeed` repository used to demonstrate `Stable Diffusion`
> is `Accelerate`?🤔
>
> To fully utilize the capabilities of `Deepspeed`, this project was born.
> 
> As far as I know, this is the first diffusion model data parallel/Zero framework in open source that uses Deepspeed exclusively.

Additionally, this project also draws inspiration
from [OvJat](https://github.com/OvJat/DeepSpeedTutorial), [afiaka87](https://github.com/afiaka87/latent-diffusion-deepspeed),
and [liucongg](https://github.com/liucongg/ChatGLM-Finetuning), and we thank these projects for their contributions!


The advantages of this project include:

1. Full utilization of `Deepspeed` capabilities
2. Alignment with mainstream `Diffusers`
3. Both `Unet' and `Lora' models can be used for training. 
4. And more, waiting for you to discover~

## Recent Updates 🔥

- [2024/05/30] Similar to the `Accelerate` example, not dropping the stable diffusion class to fully add DeepSpeed reduces the video memory by roughly 3GB+!
- [2024/05/25] Support for dataset customization, EMA support
- [2024/04/18] Fix training text mismatch, add inference validation
- [2024/04/01] Support for `wandb`.
- [2024/03/31] Support for `lora` fine-tuning, fixed the issue where the generated images were always black after full
  fine-tuning, Added `slurm` scripts
- [2024/03/29] **`deepspeed-diffusers`** was released, supporting full fine-tuning of `Unet`.

## Demo

### Installing Dependencies

Before running the scripts, please make sure all dependencies are installed:

- Ensure the source code is up to date: `git clone https://github.com/dyedd/deepspeed-diffusers`
- Then `cd` into the folder and run: `pip install -r requirements.txt`

### [Pokémon Dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) Example

> It's recommended to download the dataset directly to your local machine, as downloading through the script will fetch
> cached files, requiring requests to `huggingface` each time it runs.

Modify the `dataset_dir` in cfg.json according to your download directory.

### Downloading Weights

The following example results were experimented under the weights
of [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5).

Note: If you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model,
please change `resolution` in cfg.json to 768.

It's strongly recommended to `git clone` the weights yourself, instead of auto-downloading through huggingface~ Then
modify `pretrained_model_name_or_path` in cfg.json.

### Training

This project supports 2 training modes.

1. Full fine-tuning of unet, with 210 iterations at mixed precision, is roughly 12 to 20 GB of video memory.
2. lora+unet, with 210 iterations at mixed precision, is roughly 4 to 9.63 GB of video memory.

Modify `use_lora.action` in `cfg/default` to `true` to support mode 2.

Additionally, weights for mode 2 are saved in `use_lora.output_dir`, and for mode 1 in `output_dir`, both named
as `ckpt_name`.

The configurations in cfg.json are quite clear (the key is the literal meaning), starting from `use_fp16` are related to
deespeed configurations, don't forget to also modify the `deepspeed_config_from_args` function in `utils.py`.

> If you are local, you can run the script directly with `bash scripts/train_text_to_image.sh`;
>
> If you are on a slurm system, after modifying some information, you can submit it with `sbatch scripts/train_text_to_image.slurm`.

### Inference/Sample
Local and Slurm scheduling are also supported, with the commands being respectively
``
bash scripts/test_text_to_image.sh
sbatch scripts/test_text_to_image.slurm
```

## Possible Issues

> [!NOTE]
> 1. The generated images are black or an error is
     reported `RuntimeWarning: invalid value encountered in cast images = (images * 255).round().astype("uint8")`
>
> Please note, the parameters about the optimizer and learning rate in the `cfg.json` released with this project are
> specifically for the Pokémon dataset. This issue is due to the optimizer and learning rate not being suitable for the
> training set, causing the training loss to always be none.

## Citing Deepspeed-diffusers
If you use Deepspeed-diffusers in your publication, please cite it by using the following BibTeX entry.

@Misc{Deepspeed-diffusers,
  title =        {Deepspeed-diffusers: training diffusers with deepspeed.},
  author =       {Ximiao Dong},
  howpublished = {\url{https://github.com/dyedd/deepspeed-diffusers}},
  year =         {2024}
}
