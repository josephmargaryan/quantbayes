# diffusion_lib/generate.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from quantbayes.torch_based.diffusion import GaussianDiffusion


def generate_images(
    diffusion: GaussianDiffusion, shape=(8, 3, 64, 64), device="cuda", show=True
):
    """
    Generate images using the diffusion model.
    shape: expected output shape (B, C, H, W)
    """
    diffusion.eval()
    with torch.no_grad():
        samples = diffusion.sample(shape, device=device)
        # Assume model output is in range [-1, 1]. Convert to [0, 1].
        samples = (samples + 1) / 2.0
        samples = samples.clamp(0, 1).cpu().numpy()

        if show:
            # Plot a grid of images.
            batch, c, h, w = samples.shape
            grid_size = int(np.sqrt(batch))
            fig, axes = plt.subplots(
                grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
            )
            for i in range(grid_size):
                for j in range(grid_size):
                    img = np.transpose(samples[i * grid_size + j], (1, 2, 0))
                    axes[i, j].imshow(img)
                    axes[i, j].axis("off")
            plt.tight_layout()
            plt.show()
    return samples


def generate_time_series(
    diffusion: GaussianDiffusion, shape=(8, 100, 1), device="cuda"
):
    """
    Generate time-series samples.
    """
    with torch.no_grad():
        return diffusion.sample(shape, device=device)


def generate_tabular_samples(
    diffusion: GaussianDiffusion, shape=(8, 16), device="cuda"
):
    """
    Generate tabular data samples.
    """
    with torch.no_grad():
        return diffusion.sample(shape, device=device)
