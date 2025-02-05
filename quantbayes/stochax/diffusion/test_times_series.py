# score_diffusion/tests/test_time_series_diffusion.py

import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import equinox as eqx

from score_diffusion.config import TimeSeriesConfig
from score_diffusion.data.dataloaders import generate_synthetic_time_series, dataloader
from score_diffusion.models.timeseries_1d import TimeSeriesScoreModel
from score_diffusion.sde.sde_utils import int_beta_linear, weight_fn, single_sample_fn
from score_diffusion.training.trainer import train_model

def test_time_series_diffusion():
    cfg = TimeSeriesConfig()
    key = jr.PRNGKey(cfg.seed)
    data = generate_synthetic_time_series(num_samples=10000, seq_length=cfg.seq_length, key=key)
    data_mean = jnp.mean(data)
    data_std = jnp.std(data)
    data_min, data_max = jnp.min(data), jnp.max(data)

    data = (data - data_mean) / data_std

    model_key, train_key, sample_key = jr.split(key, 3)

    # Instantiate the model
    model = TimeSeriesScoreModel(
        seq_length=cfg.seq_length,
        hidden_dim=cfg.hidden_dim,
        time_emb_dim=cfg.time_emb_dim,
        num_layers=cfg.num_layers,
        key=model_key,
    )

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
    sample_shape = (cfg.seq_length,)
    sample_size = 4
    keys = jr.split(sample_key, sample_size)
    def sample_fn(k):
        return single_sample_fn(trained_model, int_beta_linear, sample_shape, cfg.dt0, cfg.t1, k)

    samples = jax.vmap(sample_fn)(keys)
    # denormalize
    samples = samples * data_std + data_mean
    # Plot the samples
    for i in range(sample_size):
        plt.plot(samples[i], label=f"Sample {i}")
    plt.title("Time-Series Diffusion Samples")
    plt.legend()
    plt.show()
