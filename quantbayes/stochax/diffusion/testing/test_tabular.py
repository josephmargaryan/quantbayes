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

from quantbayes.stochax.diffusion.models.tabular_dit import TabDiT
from quantbayes.stochax.diffusion.models.wrappers import UnconditionalWrapper


def generate_synthetic_tabular_dataset(
    num_samples: int, dim: int, key: jr.PRNGKey
) -> jnp.ndarray:
    k_mix, k0, k1, k_perm = jr.split(key, 4)
    n0 = num_samples // 2
    n1 = num_samples - n0
    mu0 = jnp.linspace(-1.0, 1.0, dim)
    mu1 = jnp.linspace(1.0, -1.0, dim)
    x0 = jr.normal(k0, (n0, dim)) * 0.5 + mu0
    x1 = jr.normal(k1, (n1, dim)) * 0.5 + mu1
    data = jnp.concatenate([x0, x1], axis=0)
    perm = jr.permutation(k_perm, data.shape[0])
    return data[perm]


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


def make_tabular(key, dim: int):
    return TabDiT(
        num_features=dim,
        embed_dim=192,
        depth=2,
        n_heads=4,
        mlp_ratio=4.0,
        time_emb_dim=128,
        dropout_rate=0.0,
        key=key,
    )


def run_one_model(name, factory_fn, data, seed=0):
    key = jr.PRNGKey(seed)
    dim = data.shape[1]
    core = factory_fn(key, dim)
    model = UnconditionalWrapper(model=core, time_mode="vp_t")

    ema = train_edm(model, data, steps=80, seed=seed, batch=64)

    samples = sample_edm(
        ema_model=ema,
        num_samples=8,
        sample_shape=(dim,),
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
        (dim,),
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
        (dim,),
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
        (dim,),
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
        (dim,),
        key=jr.PRNGKey(seed + 5),
        num_samples=4,
        sampler_kwargs=dict(
            steps=12, sigma_min=0.002, sigma_max=1.0, sigma_data=0.5, rho=7.0, order=3
        ),
    )
    print(f"{name} (dpmv3-o3) samples shape:", v3.shape)


def main():
    key = jr.PRNGKey(123)
    data = generate_synthetic_tabular_dataset(num_samples=512, dim=16, key=key)

    print("=== Tabular DiT ===")
    run_one_model("TabDiT", make_tabular, data, seed=10)


if __name__ == "__main__":
    main()
