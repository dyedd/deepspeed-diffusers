import argparse
import logging
import math
import os

import deepspeed
import torch
import torch.nn.functional as F
from deepspeed import get_accelerator
from diffusers import (AutoencoderKL, DDPMScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from diffusers.training_utils import EMAModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from utils import (deepspeed_config_from_args, is_rank_0, load_custom_dataset,
                   load_training_config, log_validation, set_random_seed)


def main():
    parser = argparse.ArgumentParser(description='deepspeed训练SD脚本')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='传递给deepspeed的hook,除了多节点Slurm，其它启动会自动补充')
    parser.add_argument('--cfg', type=str, default="./default.json", help='配置文件路径')
    args = parser.parse_args()
    cfg_path = args.cfg
    cfg = load_training_config(cfg_path)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if cfg.offline:
        os.environ["WANDB_MODE"] = "offline"
    # 兼容单机，单机多卡没有这个变量
    if os.environ.get('SLURM_NTASKS'):
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        os.environ['MASTER_PORT'] = os.environ['MASTER_PORT']
        os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
        args.local_rank = int(os.environ['SLURM_LOCALID'])

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    if "seed" in cfg:
        # 设置随机数，保证结果可验证
        set_random_seed(cfg.seed)

    logging.info("模型初始化中😉")
    # 初始化模型
    if cfg.use_fp16.enabled:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    # 从预训练模型中加载模型
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="unet")
    # 冻结VAE和text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if cfg.use_lora.action:
        # 冻结Unet的权重
        unet.requires_grad_(False)
        unet_lora_config = LoraConfig(
            r=cfg.use_lora.rank,
            lora_alpha=cfg.use_lora.alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        cfg.checkpoint_dir = cfg.checkpoint_dir + "-lora"
        # 移动到GPU
        unet.to(device, dtype=weight_dtype)
        text_encoder.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        # 增加适配器
        unet.add_adapter(unet_lora_config)
        if weight_dtype == torch.float16:
            for param in unet.parameters():
                # 训练LoRA的参数只能是fp32
                if param.requires_grad:
                    param.data = param.to(torch.float32)
    else:
        # 移动到GPU
        text_encoder.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        if cfg.use_ema:
            ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
            ema_unet.to(device)

    logging.info("初始化数据集🍳")
    # 初始化数据集
    train_dataset, collate_fn = load_custom_dataset(cfg, tokenizer, imagefolder=cfg.imagefolder)
    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   collate_fn=collate_fn,
                                                   shuffle=(train_sampler is None),
                                                   batch_size=cfg.train_micro_batch_size_per_gpu,
                                                   sampler=train_sampler,
                                                   num_workers=cfg.dataloader_num_workers,
                                                   pin_memory=True)

    logging.info("DeepSpeed装载ing🛠️")
    # 初始化引擎
    deepspeed_config = deepspeed_config_from_args(cfg)
    if deepspeed_config["scheduler"]["type"].startswith("Warm"):
        deepspeed_config["scheduler"]["params"]["warmup_max_lr"] = deepspeed_config["optimizer"]["params"]["lr"]
        deepspeed_config["scheduler"]["params"]["warmup_num_steps"] = cfg.lr_warmup_steps * int(
            os.environ['WORLD_SIZE'])
    torch.distributed.barrier()
    parameters = filter(lambda p: p.requires_grad, unet.parameters())
    unet, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=unet,
        model_parameters=parameters,
        config=deepspeed_config,
    )

    # Train!
    total_batch_size = cfg.train_micro_batch_size_per_gpu * int(
        os.environ['WORLD_SIZE']) * cfg.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)

    if cfg.max_train_steps == 0:
        cfg.max_train_steps = math.ceil(cfg.num_epochs * num_update_steps_per_epoch)
    else:
        cfg.num_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    logging.info("***** 开始训练啦！🚀*****")
    logging.info(f"  数据样本数量 = {len(train_dataset)}")
    logging.info(f"  每个设备的数据样本数量 = {len(train_dataloader)}")
    logging.info(f"  训练轮数 = {cfg.num_epochs}")
    logging.info(f"  每个设备的Batch size = {cfg.train_micro_batch_size_per_gpu}")
    logging.info(f"  总的训练批处理大小（包括并行，分布式和累积） = {total_batch_size}")
    logging.info(f"  梯度累积步骤 = {cfg.gradient_accumulation_steps}")
    logging.info(f"  总优化步骤 = {cfg.max_train_steps}")

    global_step = 0
    first_epoch = 0
    if cfg.resume_from_checkpoint:
        # 加载检查点
        # 使用lora训练的时候不要加载原来的权重了，因为保存的权重并不是Unet权重，导入会出错
        if os.path.exists(cfg.checkpoint_dir) and not cfg.use_lora.action:
            unet.load_checkpoint(f"./{cfg.checkpoint_dir}/")
            latest_file_path = os.path.join(f"./{cfg.checkpoint_dir}", "latest")
            with open(latest_file_path, "r") as file:
                content = file.read().strip()
                step_str = content.split('step')[-1]
                global_step = int(step_str)
                initial_global_step = global_step
                first_epoch = int(global_step // num_update_steps_per_epoch)
        else:
            logging.info("未找到检查点文件，将开始新的训练过程。")
            cfg.resume_from_checkpoint = False
            initial_global_step = 0
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not is_rank_0(),
    )

    for epoch in range(first_epoch, cfg.num_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            if global_step >= cfg.max_train_steps:
                break
            with torch.no_grad():
                images, texts = batch['pixel_values'].to(unet.device, dtype=weight_dtype, non_blocking=True), batch[
                    'input_ids'].to(
                    unet.device, non_blocking=True)
                # 将一个批次的图像转换为潜空间表示
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                # 生成高斯噪声
                noise = torch.randn_like(latents)
                # 为批次里的每张图片随机选择一个时间步
                batch_size = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,),
                                          device=latents.device)
                timesteps = timesteps.long()
                # 前向过程
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 处理文本
                text_tokens = text_encoder(texts, return_dict=False)[0]

                # 目标噪声
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # 预测噪声残差并计算损失
            model_pred = unet(noisy_latents, timesteps, text_tokens, return_dict=False)[0]

            loss = F.mse_loss(model_pred.float(),
                              target.float(), reduction="mean")
            unet.backward(loss)
            unet.step()
            if not cfg.use_lora.action and cfg.use_ema:
                ema_unet.step(unet.module.parameters())
            if step % cfg.gradient_accumulation_steps == 0:
                logs = {"epoch": f"{epoch + 1 : d}", "loss": f"{loss.item():.6f}"}
                progress_bar.set_postfix(**logs)
                progress_bar.update(1)
                if step % cfg.save_interval == 0 and is_rank_0():
                    unet.save_checkpoint(f"{cfg.checkpoint_dir}")
                    logging.info("转换为SD权重...")
                    if not cfg.use_lora.action:
                        if cfg.use_ema:
                            ema_unet.copy_to(unet.module.parameters())
                            ema_unet.save_pretrained(
                                os.path.join(cfg.output_dir, f"global_step{global_step + 1}", "unet_ema"))
                        else:
                            unet.module.save_pretrained(
                                os.path.join(cfg.output_dir, f"global_step{global_step + 1}", "unet"))
                    else:
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unet.module.to(torch.float32)))
                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=os.path.join(cfg.output_dir, f"global_step{global_step + 1}"),
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True
                        )
                        if weight_dtype == torch.float16:
                            unet.module.to(torch.float16)
                    logging.info(f'权重转换完成____当前保存的是：{global_step + 1}__loss: {loss.item():.6f}')
                global_step += 1
        if global_step >= cfg.max_train_steps:
            break
        if is_rank_0() and cfg.validation_prompts is not None and cfg.validation_epochs > 0 and epoch % cfg.validation_epochs == 0:
            if not cfg.use_lora.action and cfg.use_ema:
                ema_unet.store(unet.module.parameters())
                ema_unet.copy_to(unet.module.parameters())
            log_validation(unet.module, cfg, device, weight_dtype)
            if not cfg.use_lora.action and cfg.use_ema:
                ema_unet.restore(unet.module.parameters())


if __name__ == '__main__':
    main()
