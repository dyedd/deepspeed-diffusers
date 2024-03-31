import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel


class Diffusion(nn.Module):
    def __init__(self, pretrained_model_name_or_path):
        super().__init__()

        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

        # 冻结VAE和text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
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
            text_tokens = self.text_encoder(texts, return_dict=False)[0]

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
        model_pred  = self.unet(noisy_latents, timesteps, text_tokens, return_dict=False)[0]

        loss = F.mse_loss(model_pred.float(),
                          target.float(), reduction="mean")
        return loss