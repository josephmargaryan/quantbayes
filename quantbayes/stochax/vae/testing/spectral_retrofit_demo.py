# vae/testing/spectral_end_to_end_demo.py
from __future__ import annotations

import optax
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp

from quantbayes.stochax.vae.components import ViT_VAE
from quantbayes.stochax.vae.train_vae import TrainConfig, train_vae

# from the retrofit patch
from quantbayes.stochax.vae.workflows import retrofit_vae_model
from quantbayes.stochax.utils.linear_surgery import make_s_only_freeze_mask
from quantbayes.stochax.utils.optim_util import OptimizerConfig, build_optimizer


def build_dense_vae(key):
    return ViT_VAE(
        image_size=28,
        channels=1,
        patch_size=4,
        embedding_dim=256,
        num_layers=2,
        num_heads=4,
        latent_dim=32,
        dropout_rate=0.1,
        key=key,
    )


def make_cfg():
    return TrainConfig(
        epochs=3,
        batch_size=16,
        learning_rate=1e-3,
        likelihood="gaussian",
        beta_warmup_steps=2000,
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


def reconstruct(model, x, *, seed=0):
    recon, mu, logvar = model(x, jr.PRNGKey(seed), train=False)
    return recon, mu, logvar


def main():
    X = jr.uniform(jr.PRNGKey(0), shape=(128, 1, 28, 28))

    # ------------------------------------------------------------
    # 1) Dense baseline
    # ------------------------------------------------------------
    dense = build_dense_vae(jr.PRNGKey(1))
    dense = train_vae(dense, X, make_cfg())
    recon, mu, logvar = reconstruct(dense, X[:8], seed=10)
    print("dense recon:", recon.shape, "mu:", mu.shape, "logvar:", logvar.shape)

    # ------------------------------------------------------------
    # 2) SVD-from-scratch
    # ------------------------------------------------------------
    svd_model, svd_report = retrofit_vae_model(
        build_dense_vae(jr.PRNGKey(2)),
        variant="svd",
        mode="all_linear",
    )
    print("svd_report:", svd_report)
    svd_model = train_vae(svd_model, X, make_cfg())
    recon, mu, logvar = reconstruct(svd_model, X[:8], seed=20)
    print("svd recon:", recon.shape, "mu:", mu.shape, "logvar:", logvar.shape)

    # ------------------------------------------------------------
    # 3) Warm-start: dense -> exact SVD transplant -> fine-tune only s
    # ------------------------------------------------------------
    warm_svd_model, warm_report = retrofit_vae_model(
        dense,
        variant="svd",
        mode="all_linear",
    )
    print("warm_svd_report:", warm_report)

    s_only_tx = make_s_only_optimizer(warm_svd_model, lr=1e-4)
    warm_cfg = make_cfg()
    warm_cfg.epochs = 2
    warm_svd_model = train_vae(
        warm_svd_model,
        X,
        warm_cfg,
        optimizer=s_only_tx,
    )
    recon, mu, logvar = reconstruct(warm_svd_model, X[:8], seed=30)
    print("warm_svd recon:", recon.shape, "mu:", mu.shape, "logvar:", logvar.shape)

    # ------------------------------------------------------------
    # 4) RFFT retrofit
    # ------------------------------------------------------------
    rfft_model, rfft_report = retrofit_vae_model(
        build_dense_vae(jr.PRNGKey(3)),
        variant="rfft",
        mode="all_linear",
        warmstart=True,
        key=jr.PRNGKey(4),
    )
    print("rfft_report:", rfft_report)
    rfft_model = train_vae(rfft_model, X, make_cfg())
    recon, mu, logvar = reconstruct(rfft_model, X[:8], seed=40)
    print("rfft recon:", recon.shape, "mu:", mu.shape, "logvar:", logvar.shape)


if __name__ == "__main__":
    main()
