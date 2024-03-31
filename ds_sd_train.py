import deepspeed
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from model import Diffusion
import time
import os
import argparse
from loguru import logger as logging
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from deepspeed_config import deepspeed_config_from_args
from utils import load_training_config
from custom_dataset import load_custom_dataset


def main():
    parser = argparse.ArgumentParser(description='deepspeed训练SD脚本')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='传递给deepspeed的hook,程序启动会自动补充')
    parser.add_argument('--cfg', type=str, default="./cfg.json", help='配置文件路径')
    args = parser.parse_args()
    cfg_path = args.cfg
    cfg = load_training_config(cfg_path)

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # 初始化分布式训练
        deepspeed.init_distributed()

    # 初始化模型
    logging.debug("init model")
    if cfg.use_fp16:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    model = Diffusion(cfg.pretrained_model_name_or_path)

    if cfg.use_lora.action:
        # 冻结Unet的权重
        for param in model.unet.parameters():
            param.requires_grad_(False)
        unet_lora_config = LoraConfig(
            r=torch.cuda.device_count(),
            lora_alpha=torch.cuda.device_count(),
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        model = get_peft_model(model, unet_lora_config)
        cfg.output_dir = cfg.output_dir + "-lora"

    # 移动到GPU
    model.unet.to(device, dtype=weight_dtype)
    model.text_encoder.to(device, dtype=weight_dtype)
    model.vae.to(device, dtype=weight_dtype)

    if cfg.use_lora.action and weight_dtype == torch.float16:
        for param in model.unet.parameters():
            # 训练LoRA的参数只能是fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    # 初始化数据集
    logging.debug("init dataset")
    train_dataset, collate_fn = load_custom_dataset(cfg, model)

    # 初始化引擎
    logging.debug("init engine")
    deepspeed_config = deepspeed_config_from_args(cfg)
    parameters = filter(lambda p: p.requires_grad, model.unet.parameters())
    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters,
        training_data=train_dataset,
        collate_fn=collate_fn,
        config_params=deepspeed_config,
    )

    # Train!
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {cfg.num_epochs}")
    logging.info(f"  Instantaneous batch size per device = {cfg.train_micro_batch_size_per_gpu}")
    logging.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")

    # 加载检查点
    # 使用lora训练的时候不要加载原来的权重了，因为保存的权重并不是Unet权重，导入会出错
    if os.path.exists(cfg.output_dir) and not cfg.use_lora.action:
        model_engine.load_checkpoint(f"./{cfg.output_dir}/")

    if cfg.use_lora.action:
        model_engine.train()
    else:
        model_engine.unet.train()

    local_rank = model_engine.local_rank
    start_time = time.time()
    for epoch in range(cfg.num_epochs):
        # 记录训练开始时间
        last_time = time.time()
        running_loss = 0.0
        for step, batch in enumerate(training_dataloader):
            images, texts = batch['pixel_values'].to(model_engine.device, dtype=weight_dtype), batch['input_ids'].to(
                model_engine.device)
            loss = model_engine(images, texts)
            model_engine.backward(loss)
            model_engine.step()
            running_loss += loss.item()
            if step % cfg.log_interval == 0:
                if local_rank == 0:
                    used_time = time.time() - last_time
                    logging.info(
                        f"[epoch: {epoch + 1 : d}, step: {step + 1 : 5d}] Loss: {running_loss / cfg.log_interval : .3f} Time/Batch: {used_time / cfg.log_interval:6.4f}s")
                    last_time = time.time()
                running_loss = 0.0
            if step % cfg.save_interval == 0:
                # save checkpoint
                model_engine.save_checkpoint(f"{cfg.output_dir}")
    if local_rank == 0:
        logging.info(f"Total training time: {time.time() - start_time:6.4f}s")
        if not cfg.use_lora.action:
            pipeline = StableDiffusionPipeline.from_pretrained(
                cfg.pretrained_model_name_or_path,
                unet = model_engine.unet,
                text_encoder=model_engine.text_encoder,
                vae=model_engine.vae,
                torch_dtype=weight_dtype
            )
            pipeline.save_pretrained(cfg.output_dir + '/' + cfg.ckpt_name, torch_dtype=weight_dtype)
        else:
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(model_engine.to(torch.float32)))
            StableDiffusionPipeline.save_lora_weights(
                save_directory=cfg.use_lora.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
                weight_name=cfg.ckpt_name + '.safetensor'
            )


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
