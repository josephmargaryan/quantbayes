# score_diffusion/tests/test_time_series_diffusion.py

import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt

from quantbayes.stochax.diffusion.config import TimeSeriesConfig
from quantbayes.stochax.diffusion.dataloaders import generate_synthetic_time_series, dataloader
from quantbayes.stochax.diffusion.sde import int_beta_linear, weight_fn, single_sample_fn 
from quantbayes.stochax.diffusion.trainer import train_model
from quantbayes.stochax.diffusion.models.times_series_1d import ConvTimeUNet
from quantbayes.stochax.diffusion.models.transformer_1d import DiffusionTransformer1D


def test_conv_time_unet_diffusion():
    cfg = TimeSeriesConfig()
    key = jr.PRNGKey(cfg.seed)

    # Generate time-series data: shape [num_samples, seq_length]
    data = generate_synthetic_time_series(num_samples=10, seq_length=cfg.seq_length, key=key)
    data_mean = jnp.mean(data)
    data_std = jnp.std(data)
    data_min, data_max = jnp.min(data), jnp.max(data)

    model_key, train_key, sample_key = jr.split(key, 3)

    # Instantiate the model
    hidden_dim = cfg.hidden_dim
    dim_mults = [2, 2]  # example
    num_res_blocks = 2
    model = ConvTimeUNet(
        seq_length=cfg.seq_length,
        in_channels=1,      # e.g. treat each TS as [1, seq_length]
        hidden_dim=hidden_dim,
        dim_mults=dim_mults,
        num_res_blocks=num_res_blocks,
        time_emb_dim=cfg.time_emb_dim,
        dropout=0.1,
        key=model_key,
    )

    # Normalize
    data = (data - data_mean) / data_std

    # Train
    trained_model = train_model(
        model=model,
        dataset=data,      # shape [N, seq_length]
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

    # Sampling: pick a sample_size=4 for demonstration
    sample_size = 4
    sample_keys = jr.split(sample_key, sample_size)
    def sample_fn(k):
        return single_sample_fn(trained_model, int_beta_linear, (cfg.seq_length,), cfg.dt0, cfg.t1, k)
    samples = jax.vmap(sample_fn)(sample_keys)  # shape [4, seq_length]

    # Denormalize
    samples = samples * data_std + data_mean
    samples = jnp.clip(samples, data_min, data_max)
    print("Sampled time-series shape:", samples.shape)

    plt.plot(samples[0])
    plt.show()


def test_diffusion_transformer_1d():
    cfg = TimeSeriesConfig()
    key = jr.PRNGKey(cfg.seed)

    # Generate synthetic data of shape [num_samples, seq_length]
    data = generate_synthetic_time_series(num_samples=16, seq_length=cfg.seq_length, key=key)
    data_mean = jnp.mean(data)
    data_std = jnp.std(data)
    data_min, data_max = jnp.min(data), jnp.max(data)

    model_key, train_key, sample_key = jr.split(key, 3)

    # Instantiate the 1D diffusion transformer
    embed_dim = 64
    depth = 4
    n_heads = 4
    mlp_ratio = 4.0
    dropout_rate = 0.1
    time_emb_dim = 32

    model = DiffusionTransformer1D(
        seq_length=cfg.seq_length,
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
        dataset=data,  # shape => (N, seq_length)
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
    sample_keys = jr.split(sample_key, sample_size)
    def sample_fn(k):
        # shape => (seq_length,)
        return single_sample_fn(trained_model, int_beta_linear, (cfg.seq_length,), cfg.dt0, cfg.t1, k)

    samples = jax.vmap(sample_fn)(sample_keys)  # (sample_size, seq_length)
    samples = samples * data_std + data_mean
    samples = jnp.clip(samples, data_min, data_max)

    print("Sampled time-series shape:", samples.shape)

    # Plot first sample
    plt.plot(samples[0])
    plt.title("DiffusionTransformer1D Sample")
    plt.show()

if __name__ == "__main__":
    # test_conv_time_unet_diffusion()
    test_diffusion_transformer_1d()