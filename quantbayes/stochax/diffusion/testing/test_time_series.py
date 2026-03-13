from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.dataloaders import dataloader
from quantbayes.stochax.diffusion.trainer import train_model
from quantbayes.stochax.diffusion.edm import edm_batch_loss
from quantbayes.stochax.diffusion import sample_edm
from quantbayes.stochax.diffusion.generate import generate_with_sampler

from quantbayes.stochax.diffusion.models.timeseries_dit import TimeDiT1D
from quantbayes.stochax.diffusion.models.wrappers import UnconditionalWrapper


def generate_synthetic_timeseries_dataset(
    num_samples: int, length: int, channels: int, key: jr.PRNGKey
) -> jnp.ndarray:
    k_noise = jr.split(key, 1)[0]
    t = jnp.linspace(0.0, 2.0 * jnp.pi, length)
    freqs = jnp.arange(1, channels + 1)
    base = jnp.stack([jnp.sin(f * t) for f in freqs], axis=-1)
    data = base[None, ...] + 0.1 * jr.normal(k_noise, (num_samples, length, channels))
    return data


def train_edm(model, data, steps=80, seed=0, batch=64):
    return train_model(
        model,
        dataset=data,
        t1=1.0,
        lr=3e-4,
        num_steps=steps,
        batch_size=batch,
        weight_fn=None,
        int_beta_fn=None,
        print_every=max(steps // 2, 1),
        seed=seed,
        data_loader_func=dataloader,
        loss_impl="edm",
        custom_loss=lambda m, b, k: edm_batch_loss(
            m, b, k, sigma_data=0.5, rho_min=-1.2, rho_max=1.2, sample="uniform"
        ),
        checkpoint_dir=None,
    )


def make_ts_model(key, shape):
    L, C = shape
    return TimeDiT1D(
        seq_len=L,
        in_channels=C,
        patch_size=8,
        embed_dim=192,
        depth=2,
        n_heads=4,
        mlp_ratio=4.0,
        time_emb_dim=256,
        dropout_rate=0.0,
        learn_sigma=False,
        key=key,
    )


def run_one_model(name, factory_fn, data, seed=0):
    key = jr.PRNGKey(seed)
    sample_shape = data.shape[1:]  # (L, C)
    core = factory_fn(key, sample_shape)
    model = UnconditionalWrapper(model=core, time_mode="vp_t")

    ema = train_edm(model, data, steps=80, seed=seed, batch=64)

    samples = sample_edm(
        ema_model=ema,
        num_samples=8,
        sample_shape=sample_shape,
        key=jr.PRNGKey(seed + 1),
        steps=12,
        sigma_min=0.002,
        sigma_max=1.0,
        sigma_data=0.5,
        rho=7.0,
    )
    print(f"{name} (dpmpp_3m) samples shape:", samples.shape)

    ema_eval = eqx.tree_inference(ema, value=True)

    def denoise_fn(log_sigma, x):
        return ema_eval(log_sigma, x, key=None, train=False)

    uni = generate_with_sampler(
        denoise_fn,
        "unipc",
        sample_shape,
        key=jr.PRNGKey(seed + 2),
        num_samples=4,
        sampler_kwargs=dict(
            steps=12, sigma_min=0.002, sigma_max=1.0, sigma_data=0.5, rho=7.0
        ),
    )
    print(f"{name} (unipc) samples shape:", uni.shape)

    ip2 = generate_with_sampler(
        denoise_fn,
        "ipndm",
        sample_shape,
        key=jr.PRNGKey(seed + 3),
        num_samples=4,
        sampler_kwargs=dict(
            steps=12, sigma_min=0.002, sigma_max=1.0, sigma_data=0.5, rho=7.0
        ),
    )
    print(f"{name} (ipndm) samples shape:", ip2.shape)

    ip4 = generate_with_sampler(
        denoise_fn,
        "ipndm4",
        sample_shape,
        key=jr.PRNGKey(seed + 4),
        num_samples=4,
        sampler_kwargs=dict(
            steps=12, sigma_min=0.002, sigma_max=1.0, sigma_data=0.5, rho=7.0
        ),
    )
    print(f"{name} (ipndm4) samples shape:", ip4.shape)

    v3 = generate_with_sampler(
        denoise_fn,
        "dpmv3",
        sample_shape,
        key=jr.PRNGKey(seed + 5),
        num_samples=4,
        sampler_kwargs=dict(
            steps=12, sigma_min=0.002, sigma_max=1.0, sigma_data=0.5, rho=7.0, order=3
        ),
    )
    print(f"{name} (dpmv3-o3) samples shape:", v3.shape)


def main():
    key = jr.PRNGKey(0)
    data = generate_synthetic_timeseries_dataset(
        num_samples=512, length=64, channels=3, key=key
    )

    print("=== Time-Series DiT (1D) ===")
    run_one_model("TimeDiT1D", make_ts_model, data, seed=20)


if __name__ == "__main__":
    main()
