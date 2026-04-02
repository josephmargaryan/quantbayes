# diffusion/testing/spectral_end_to_end_demo.py
from __future__ import annotations

import optax
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp

from quantbayes.stochax.diffusion.dataloaders import (
    generate_synthetic_image_dataset,
    dataloader,
)
from quantbayes.stochax.diffusion.trainer import train_model
from quantbayes.stochax.diffusion.edm import edm_batch_loss
from quantbayes.stochax.diffusion import sample_edm
from quantbayes.stochax.diffusion.models.adaptive_DiT import DiT
from quantbayes.stochax.diffusion.models.wrappers import DiTWrapper

# from the retrofit patch
from quantbayes.stochax.diffusion.workflows import retrofit_diffusion_model
from quantbayes.stochax.utils.linear_surgery import make_s_only_freeze_mask
from quantbayes.stochax.utils.optim_util import OptimizerConfig, build_optimizer


def build_dense_dit(key):
    core = DiT(
        img_size=(1, 28, 28),
        patch_size=4,
        in_channels=1,
        embed_dim=192,
        depth=2,
        n_heads=6,
        mlp_ratio=4.0,
        dropout_rate=0.0,
        time_emb_dim=192,
        num_classes=10,
        learn_sigma=False,
        key=key,
    )
    return DiTWrapper(
        model=core,
        num_classes=10,
        time_mode="vp_t",
        null_label_index=None,
        cfg_rescale=0.7,
    )


def train_edm(model, data, *, steps=200, seed=0, batch_size=32, optimizer=None):
    return train_model(
        model,
        dataset=data,
        t1=1.0,
        lr=3e-4,
        num_steps=steps,
        batch_size=batch_size,
        weight_fn=None,
        int_beta_fn=None,
        print_every=max(steps // 5, 1),
        seed=seed,
        data_loader_func=dataloader,
        loss_impl="edm",
        custom_loss=lambda m, b, k: edm_batch_loss(
            m, b, k, sigma_data=0.5, rho_min=-1.2, rho_max=1.2, sample="uniform"
        ),
        checkpoint_dir=None,
        optimizer=optimizer,
    )


def make_s_only_optimizer(model, lr=1e-4):
    freeze_mask = make_s_only_freeze_mask(
        model,
        train_bias=False,
        train_alpha=False,
    )
    tx, _, _ = build_optimizer(
        model,
        OptimizerConfig(
            algorithm="adamw",
            lr=lr,
            weight_decay=0.0,
            clip_global_norm=1.0,
        ),
        prepend=optax.masked(optax.set_to_zero(), freeze_mask),
    )
    return tx


def sample_some(ema_model, *, seed=0, n=8):
    return sample_edm(
        ema_model=ema_model,
        num_samples=n,
        sample_shape=(1, 28, 28),
        key=jr.PRNGKey(seed),
        steps=20,
        sigma_min=0.002,
        sigma_max=1.0,
        sigma_data=0.5,
        rho=7.0,
    )


def main():
    data = generate_synthetic_image_dataset(
        num_samples=256,
        shape=(1, 28, 28),
        key=jr.PRNGKey(0),
    )

    # ------------------------------------------------------------
    # 1) Dense baseline
    # ------------------------------------------------------------
    dense = build_dense_dit(jr.PRNGKey(1))
    ema_dense = train_edm(dense, data, steps=100, seed=10)
    dense_samples = sample_some(ema_dense, seed=100)
    print("dense_samples:", dense_samples.shape)

    # ------------------------------------------------------------
    # 2) SVD-from-scratch
    # ------------------------------------------------------------
    svd_model, svd_report = retrofit_diffusion_model(
        build_dense_dit(jr.PRNGKey(2)),
        variant="svd",
        mode="all_linear",
    )
    print("svd_report:", svd_report)
    ema_svd = train_edm(svd_model, data, steps=100, seed=20)
    svd_samples = sample_some(ema_svd, seed=200)
    print("svd_samples:", svd_samples.shape)

    # ------------------------------------------------------------
    # 3) Warm-start: dense -> exact SVD transplant -> fine-tune only s
    # ------------------------------------------------------------
    warm_svd_model, warm_report = retrofit_diffusion_model(
        ema_dense,
        variant="svd",
        mode="all_linear",
    )
    print("warm_svd_report:", warm_report)

    s_only_tx = make_s_only_optimizer(warm_svd_model, lr=1e-4)
    ema_warm_svd = train_edm(
        warm_svd_model,
        data,
        steps=50,
        seed=30,
        optimizer=s_only_tx,
    )
    warm_svd_samples = sample_some(ema_warm_svd, seed=300)
    print("warm_svd_samples:", warm_svd_samples.shape)

    # ------------------------------------------------------------
    # 4) RFFT retrofit for square linears only
    # ------------------------------------------------------------
    rfft_model, rfft_report = retrofit_diffusion_model(
        build_dense_dit(jr.PRNGKey(3)),
        variant="rfft",
        mode="all_linear",
        warmstart=True,
        key=jr.PRNGKey(4),
    )
    print("rfft_report:", rfft_report)
    ema_rfft = train_edm(rfft_model, data, steps=100, seed=40)
    rfft_samples = sample_some(ema_rfft, seed=400)
    print("rfft_samples:", rfft_samples.shape)


if __name__ == "__main__":
    main()
