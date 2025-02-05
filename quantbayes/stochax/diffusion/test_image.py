# score_diffusion/tests/test_image_diffusion.py

import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import equinox as eqx
import einops

from score_diffusion.config import ImageConfig
from score_diffusion.data.dataloaders import generate_synthetic_image_dataset, dataloader
from score_diffusion.models.mixer_2d import Mixer2d
from score_diffusion.models.unet_2d import SimpleUNet
from score_diffusion.sde.sde_utils import int_beta_linear, weight_fn, single_sample_fn
from score_diffusion.training.trainer import train_model

def test_mixer2d_image_diffusion():
    cfg = ImageConfig()
    key = jr.PRNGKey(cfg.seed)
    data = generate_synthetic_image_dataset(num_samples=5000, shape=(1, 28, 28), key=key)
    data_mean = jnp.mean(data)
    data_std = jnp.std(data)
    data_min, data_max = jnp.min(data), jnp.max(data)

    model_key, train_key, sample_key = jr.split(key, 3)

    # Instantiate the model
    patch_size = cfg.patch_size
    hidden_size = cfg.hidden_size
    mix_patch_size = cfg.mix_patch_size
    mix_hidden_size = cfg.mix_hidden_size
    t1 = cfg.t1
    num_blocks = cfg.num_blocks

    mixer = Mixer2d(
        img_size=(1, 28, 28),
        patch_size=patch_size,
        hidden_size=hidden_size,
        mix_patch_size=mix_patch_size,
        mix_hidden_size=mix_hidden_size,
        num_blocks=num_blocks,
        t1=t1,
        key=model_key,
    )

    # Normalize data
    data = (data - data_mean) / data_std

    # Train
    trained_model = train_model(
        model=mixer,
        dataset=data,
        t1=cfg.t1,
        lr=cfg.lr,
        num_steps=cfg.num_steps,
        batch_size=cfg.batch_size,
        weight_fn=weight_fn,
        int_beta_fn=int_beta_linear,
        print_every=cfg.print_every,
        seed=cfg.seed,
        data_loader_func=dataloader,
    )

    # Sample
    sample_shape = (1, 28, 28)
    sample_size = 4
    keys = jr.split(sample_key, sample_size ** 2)
    def sample_fn(k):
        return single_sample_fn(trained_model, int_beta_linear, sample_shape, cfg.dt0, cfg.t1, k)

    samples = jax.vmap(sample_fn)(keys)
    # Denormalize
    samples = samples * data_std + data_mean
    samples = jnp.clip(samples, data_min, data_max)
    # Visualize
    grid = einops.rearrange(samples, "(n1 n2) c h w -> (n1 h) (n2 w)", n1=sample_size, n2=sample_size)
    plt.imshow(grid, cmap="gray")
    plt.title("Mixer2D Diffusion Samples")
    plt.axis("off")
    plt.show()

def test_unet2d_image_diffusion():
    """
    Similar to test_mixer2d_image_diffusion but using a simpler UNet architecture.
    """
    # Implementation is analogous. Just replace the model with SimpleUNet or a more advanced UNet.
    pass
