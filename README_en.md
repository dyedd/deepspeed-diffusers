# 🧩 Deepspeed-diffusers

*Read this in [English](README_en.md).*

`deepspeed-diffusers` is a project that integrates the `Deepspeed`, `Diffusers`, and `Peft` libraries to train diffusion models.

## Motivation
Developers who have trained with `Diffusers` know that there is already another `Huggingface` product, `Accelerate`, for parallel training of diffusion models.

Why did I open-source this project then?
- A rather funny fact is that I didn't initially notice that the training example file of `diffusers` partially supports `Zero3` rather than not supporting it. The final implementation of this project ended up with a `Zero3` implementation approach that coincidentally aligns with it.
- The consolation is that, although there are now many integrated frameworks that are convenient to use, having too many integrations makes secondary development difficult!
- At least I provided a script for the `Slurm` scheduling system, using and fixing the `Deepspeed`'s originally supported but not quite functional `Slurm` launcher.

> To fully leverage the capabilities of `Deepspeed`, this project was born.

Additionally, the project also drew inspiration from the code of [OvJat](https://github.com/OvJat/DeepSpeedTutorial), [afiaka87](https://github.com/afiaka87/latent-diffusion-deepspeed), and [liucongg](https://github.com/liucongg/ChatGLM-Finetuning) during native integration. Special thanks to these projects!

What are the shortcomings of this project compared to these projects?

1. When I implemented it, `ZeRO-1,2` only supported `UNet2DConditionModel`. I still haven't figured out exactly where I fall short compared to others and need to test more.
2. Unable to save checkpoints during Lora training, which is somewhat related to the tight integration with the `Peft` library, unless you write your own Lora fine-tuning logic.

Apart from these, this project also has the features of `Accelerate`, focusing on simplicity and practicality 🥹

## Recent Updates 🔥

- [2024/05/30] Similar to the `Accelerate` example, only loading `Unet` instead of fully adding `stable diffusion` to DeepSpeed, reducing approximately 3GB+ of VRAM.
- [2024/05/25] Support for custom datasets and EMA.
- [2024/04/18] Fixed issues with mismatched training text and added inference validation.
- [2024/04/10] Support for `slurm` multi-node and single-node training.
- [2024/04/01] Support for `wandb`.
- [2024/03/31] Support for `lora` fine-tuning, fixed the issue where images generated after full fine-tuning were always black, and added `slurm` single-node training scripts.
- [2024/03/29] **`deepspeed-diffusers`** released, supporting full fine-tuning of `Unet`.

## Demo

### Installation of Dependencies

Before running the script, please ensure that all dependencies are installed:

- Ensure the source code is up-to-date: `git clone https://github.com/dyedd/deepspeed-diffusers`
- Then `cd` into the folder and run: `pip install -r requirements.txt`

### [Pokemon Dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) Example

> It is highly recommended to download the dataset locally instead of using cached files downloaded by the script automatically. Otherwise, if you lose internet connection, the data feedback from `Huggingface` might be lost, and it's not very friendly for the domestic environment.

Modify the `dataset_dir` field in `cfg.json` according to the downloaded directory.

### Download Weights

The following example results are all based on the [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) weights.

Note: If you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model, please change the `resolution` in `cfg/default.json` to 768.

It is still highly recommended to download the weights using `git clone` rather than through huggingface automatically, and then modify the `pretrained_model_name_or_path` in `cfg/default.json`.

### Configuration File Interpretation
The parameters required by the project are written in `cfg/default.json`:
Below is an explanation of each field:
- "num_epochs": The total number of training epochs.
- "validation_epochs": How often to perform validation.
- "max_train_steps": Limit on the maximum number of training steps, 0 means no limit.
- "lr_warmup_steps": The number of steps for learning rate warm-up.
- "save_interval": The interval of steps for saving the model.
- "seed": The random seed to ensure the reproducibility of the experiment.
- "validation_prompts": Prompts used during validation.
- "pretrained_model_name_or_path": The name or path of the pretrained model.
- "dataset_dir": The directory of the dataset.
- "imagefolder": Whether to use a folder containing **images** as the data source.
- "checkpoint_dir": The directory for saving checkpoints (model state).
- "output_dir": The directory for outputs (e.g., training logs, generated images).
- "use_lora": Whether to use LoRA (Long Range Attention) technology, and related parameters.
  - "action": Whether to enable LoRA.
  - "rank": The rank of LoRA.
  - "alpha": The α parameter of LoRA.
- "dataloader_num_workers": The number of worker threads for the data loader.
- "resume_from_checkpoint": Whether to resume training from a checkpoint.
- "use_ema": Whether to use Exponential Moving Average (EMA).
- "offline": Whether to run in `wandb` offline mode.
- "resolution": The resolution of the images.
- "center_crop": Whether to center-crop the images.
- "random_flip": Whether to randomly flip the images.
- Other configurations are commonly used for `DeepSpeed`. For any confusion, please refer to the official documentation. If you add any, don't forget to modify the `deepspeed_config_from_args` function in `utils.py` as well.

### Training

This project supports two training modes.

- Full fine-tuning of Unet. Under mixed precision FP16, without Zero enabled, the batch size per card is 4, with VRAM usage ranging from approximately 11.61 to 23.32GB.
- Lora + Unet. Under mixed precision FP16, without Zero enabled, the batch size per card is 4, with VRAM usage ranging from approximately 2 to 13GB.

> As long as gradient accumulation and Zero are enabled, full fine-tuning can also be achieved with a graphics card of less than 16GB!

If you are working locally, you can run the script directly via `bash scripts/train_text_to_image.sh`;

If you are on a slurm system, you can submit it via `sbatch scripts/train_text_to_image.slurm` after modifying some information.

### Inference/Sampling
Also supports local and Slurm scheduling, with commands:
```
bash scripts/test_text_to_image.sh
sbatch scripts/test_text_to_image.slurm
```

## Possible Issues

> [!NOTE]
> 1. Generated images are all black or an error `RuntimeWarning: invalid value encountered in cast images = (images * 255).round().astype("uint8")`
>
> Please note that the optimizer and learning rate parameters in the configuration file of this project are only applicable to the Pokémon dataset. This issue arises because the optimizer and learning rate are not suitable for the training set, causing the training loss to always be none.
> 2. The submitted `slurm` script exits after 0 seconds
> This is because the `slurm` script logs are saved in the `log` folder, and the folder does not exist, so it cannot run.

## Citation
If you use `Deepspeed-diffusers` in your paper or project, please use the following `BibTeX` to cite it.
```
@Misc{Deepspeed-diffusers,
  title =        {Deepspeed-diffusers: training diffusers with deepspeed.},
  author =       {Ximiao Dong},
  howpublished = {\url{https://github.com/dyedd/deepspeed-diffusers}},
  year =         {2024}
}
```