# score_diffusion/tests/test_image_diffusion.py

import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import einops

from quantbayes.stochax.diffusion.config import ImageConfig
from quantbayes.stochax.diffusion.dataloaders import (
    generate_synthetic_image_dataset,
    dataloader,
)
from quantbayes.stochax.diffusion.sde import (
    int_beta_linear,
    weight_fn,
    single_sample_fn,
)
from quantbayes.stochax.diffusion.trainer import train_model
from quantbayes.stochax.diffusion.models.unet_2d import UNet
from quantbayes.stochax.diffusion.models.mixer_2d import Mixer2d
from quantbayes.stochax.diffusion.models.transformer_2d import DiffusionTransformer2D


def test_mixer2d_image_diffusion():
    cfg = ImageConfig()
    key = jr.PRNGKey(cfg.seed)
    data = generate_synthetic_image_dataset(num_samples=10, shape=(1, 28, 28), key=key)
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
        num_steps=10,
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
    keys = jr.split(sample_key, sample_size**2)

    def sample_fn(k):
        return single_sample_fn(
            trained_model, int_beta_linear, sample_shape, cfg.dt0, cfg.t1, k
        )

    samples = jax.vmap(sample_fn)(keys)
    # Denormalize
    samples = samples * data_std + data_mean
    samples = jnp.clip(samples, data_min, data_max)
    # Visualize
    grid = einops.rearrange(
        samples, "(n1 n2) c h w -> c (n1 h) (n2 w)", n1=sample_size, n2=sample_size
    )
    grid = grid.squeeze(0)  # Remove the channel dimension for grayscale images.
    plt.imshow(grid, cmap="gray")
    plt.title("Mixer2D Diffusion Samples")
    plt.axis("off")
    plt.show()


def test_unet_diffusion():
    # Load configuration and initialize key
    cfg = ImageConfig()
    key = jr.PRNGKey(cfg.seed)

    # Generate a synthetic image dataset (e.g. MNIST-like)
    data = generate_synthetic_image_dataset(num_samples=10, shape=(1, 28, 28), key=key)
    data_mean = jnp.mean(data)
    data_std = jnp.std(data)
    data_min, data_max = jnp.min(data), jnp.max(data)

    # Split key into model, training, and sampling keys
    model_key, train_key, sample_key = jr.split(key, 3)

    # Instantiate the Advanced UNet.
    # Adjust the parameters as needed.
    advanced_unet = UNet(
        data_shape=(1, 28, 28),
        is_biggan=False,  # e.g., False if using transposed convolutions
        dim_mults=[2, 2],  # Example: hidden_size, hidden_size*2, hidden_size*2*?
        hidden_size=cfg.hidden_size,
        heads=4,
        dim_head=32,
        dropout_rate=0.1,
        num_res_blocks=2,
        attn_resolutions=[
            14,
            7,
        ],  # Apply attention when the resolution matches these numbers
        key=model_key,
    )

    # Normalize the data
    data = (data - data_mean) / data_std

    # Train the model
    trained_model = train_model(
        model=advanced_unet,
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

    # Sampling:
    sample_shape = (1, 28, 28)
    sample_size = 4
    # Split key for each sample in the grid (sample_size^2 total)
    sample_keys = jr.split(sample_key, sample_size**2)

    def sample_fn(k):
        return single_sample_fn(
            trained_model, int_beta_linear, sample_shape, cfg.dt0, cfg.t1, k
        )

    # Vectorize the sampling function over the keys
    samples = jax.vmap(sample_fn)(sample_keys)

    # Denormalize the samples
    samples = samples * data_std + data_mean
    samples = jnp.clip(samples, data_min, data_max)

    # Rearrange the samples into a grid for visualization
    grid = einops.rearrange(
        samples, "(n1 n2) c h w -> (n1 h) (n2 w)", n1=sample_size, n2=sample_size
    )

    plt.imshow(grid, cmap="gray")
    plt.title("Advanced UNet Diffusion Samples")
    plt.axis("off")
    plt.show()


def test_diffusion_transformer_2d():
    cfg = ImageConfig()
    key = jr.PRNGKey(cfg.seed)

    # Generate a small synthetic image dataset
    data = generate_synthetic_image_dataset(num_samples=8, shape=(1, 32, 32), key=key)
    data_mean = jnp.mean(data)
    data_std = jnp.std(data)
    data_min, data_max = jnp.min(data), jnp.max(data)

    model_key, train_key, sample_key = jr.split(key, 3)

    # Instantiate the DiffusionTransformer2D
    patch_size = 4
    embed_dim = 64
    depth = 4
    n_heads = 4
    mlp_ratio = 4.0
    dropout_rate = 0.1
    time_emb_dim = 32
    c, h, w = (1, 32, 32)

    model = DiffusionTransformer2D(
        img_size=(c, h, w),
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        dropout_rate=dropout_rate,
        time_emb_dim=time_emb_dim,
        key=model_key,
    )

    # Normalize data
    data = (data - data_mean) / data_std

    # Train
    trained_model = train_model(
        model=model,
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
    sample_size = 4
    sample_shape = (1, 32, 32)
    keys = jr.split(sample_key, sample_size**2)

    def sample_fn(k):
        return single_sample_fn(
            trained_model, int_beta_linear, sample_shape, cfg.dt0, cfg.t1, k
        )

    samples = jax.vmap(sample_fn)(keys)  # shape => (sample_size^2, 1, 32, 32)
    samples = samples * data_std + data_mean
    samples = jnp.clip(samples, data_min, data_max)

    # Visualize
    grid = einops.rearrange(
        samples, "(n1 n2) c h w -> (n1 h) (n2 w)", n1=sample_size, n2=sample_size
    )
    plt.imshow(grid, cmap="gray")
    plt.title("DiffusionTransformer2D Samples")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    test_unet_diffusion()
    # test_mixer2d_image_diffusion()
    # test_diffusion_transformer_2d()
