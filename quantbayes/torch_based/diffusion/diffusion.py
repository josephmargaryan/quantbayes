# diffusion_lib/diffusion.py

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Beta Schedules
###############################################################################


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).float()


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-(t * (end - start) + start) / tau).sigmoid()
    alphas_cumprod = (alphas_cumprod - v_start) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).float()


###############################################################################
# BetaSchedule Wrapper
###############################################################################


class BetaSchedule:
    def __init__(
        self,
        schedule_type="linear",
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        **kwargs,
    ):
        self.schedule_type = schedule_type
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        if schedule_type == "linear":
            self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule_type == "cosine":
            self.betas = cosine_beta_schedule(timesteps, **kwargs)
        elif schedule_type == "sigmoid":
            self.betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")


###############################################################################
# GaussianDiffusion Class
###############################################################################


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        beta_schedule: BetaSchedule,
        sampling_timesteps=None,
        use_ddim=False,
    ):
        """
        model: a noise-predicting network (can be a U-Net, DiT, etc.)
        beta_schedule: an instance of BetaSchedule
        sampling_timesteps: if specified and less than total timesteps, use DDIM sampling.
        use_ddim: flag to use DDIM even if sampling_timesteps equals training timesteps.
        """
        super().__init__()
        self.model = model
        self.betas = beta_schedule.betas  # 1-D tensor
        self.num_timesteps = len(self.betas)
        self.use_ddim = use_ddim

        # Sampling timesteps
        self.sampling_timesteps = (
            sampling_timesteps if sampling_timesteps is not None else self.num_timesteps
        )

        # Precompute alphas and related quantities
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer(
            "alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        )
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod - 1)
        )

    def forward_diffusion_sample(self, x0, t, device="cuda"):
        """
        Sample from q(x_t | x_0): x_t = sqrt(alpha_cumprod)*x0 + sqrt(1 - alpha_cumprod)*noise
        """
        batch_size = x0.shape[0]
        alphas_cumprod_t = self.alphas_cumprod[t].reshape(
            batch_size, *((1,) * (len(x0.shape) - 1))
        )
        noise = torch.randn_like(x0, device=device)
        return (
            torch.sqrt(alphas_cumprod_t) * x0
            + torch.sqrt(1 - alphas_cumprod_t) * noise,
            noise,
        )

    @torch.no_grad()
    def p_sample_step(self, x, t):
        """
        Perform one reverse diffusion step:
            x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_cumprod_t)*model(x_t, t))
                      + sigma_t * z
        """
        beta_t = self.betas[t]
        alpha_t = 1.0 - beta_t
        alpha_cumprod_t = self.alphas_cumprod[t]
        coef1 = 1.0 / math.sqrt(alpha_t)
        coef2 = beta_t / math.sqrt(1 - alpha_cumprod_t)

        # Create a tensor for timesteps
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)
        predicted_noise = self.model(x, t_tensor)
        mean = coef1 * (x - coef2 * predicted_noise)

        if t > 0:
            noise = torch.randn_like(x)
            # A simplified variance: you might refine this further.
            sigma = math.sqrt(beta_t)
            return mean + sigma * noise
        else:
            return mean

    @torch.no_grad()
    def sample(self, shape, device="cuda"):
        """
        Sample from the diffusion model by running reverse diffusion.
        shape: (B, C, H, W) or other shape as needed.
        """
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample_step(x, t)
        return x

    def diffusion_loss(self, x0, t, device="cuda"):
        """
        Compute the training loss:
            L = MSE( predicted_noise, true_noise )
        where x_t = sqrt(alpha_cumprod)*x0 + sqrt(1 - alpha_cumprod)*noise.
        """
        x_t, noise = self.forward_diffusion_sample(x0, t, device=device)
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(predicted_noise, noise)
