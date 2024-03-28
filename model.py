import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig

class Diffusion(nn.Module):
    def __init__(self, pretrained_model_name_or_path, weight_dtype, is_lora=False, rank=None):
        super().__init__()

        # 初始化模型组件
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=True,
            torch_dtype=weight_dtype,
            use_safetensors=True,
            safety_checker=None,
        )
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.unet = pipe.unet
        # 冻结VAE和text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if is_lora:
            # 冻结Unet的权重
            for param in self.unet.parameters():
                param.requires_grad_(False)
            unet_lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet.add_adapter(unet_lora_config)
            if weight_dtype == torch.float16:
                for param in self.unet.parameters():
                    # 训练LoRA的参数只能是fp32
                    if param.requires_grad:
                        param.data = param.to(torch.float32)

    def forward(self, images, texts):
        # train step
        with torch.no_grad():
            # 将一个批次的图像转换为潜空间表示
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            # 生成高斯噪声
            noise = torch.randn_like(latents)
            # 为批次里的每张图片随机选择一个时间步
            batch_size = latents.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,),
                                      device=latents.device)
            timesteps = timesteps.long()
            # 前向过程
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # 处理文本
            text_tokens = self.text_encoder(texts)[0]

            # 目标噪声
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # 预测噪声残差并计算损失
        model_pred  = self.unet(noisy_latents, timesteps, text_tokens).sample

        loss = F.mse_loss(model_pred.float(),
                          target.float(), reduction="mean")
        return loss