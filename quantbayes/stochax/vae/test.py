#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np

from quantbayes.stochax.vae import (
    MLP_VAE,
    ConvVAE,
    MultiHeadAttentionVAE,
    ViT_VAE,
    TrainConfig,
    train_vae,
)


def gen_tabular(n=2000, d=2, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([[-2, 0], [2, 0], [0, 2], [0, -2]])
    X = means[rng.integers(0, len(means), size=n)] + rng.normal(scale=0.5, size=(n, d))
    return jnp.array(X, dtype=jnp.float32)


def gen_seq(n=1000, T=20, D=1, seed=0):
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, shape=(n, T, D))


def gen_images(n=512, C=1, H=28, W=28, seed=0):
    key = jax.random.PRNGKey(seed)
    # NCHW
    return jax.random.uniform(key, shape=(n, C, H, W))


def test_mlp():
    print("=== MLP_VAE (tabular, Gaussian) ===")
    X = gen_tabular(2000, d=2, seed=0)
    key = jax.random.PRNGKey(1)
    model = MLP_VAE(input_dim=2, hidden_dim=64, latent_dim=2, output_dim=2, key=key)
    cfg = TrainConfig(
        epochs=20, batch_size=128, likelihood="gaussian", beta_warmup_steps=5000
    )
    model = train_vae(model, X, cfg)
    # reconstruct a mini-batch
    rng = jax.random.PRNGKey(5)
    dec, mu, logvar = model(X[:8], rng, train=False)
    print("recon shape:", dec.shape)


def test_conv():
    print("=== ConvVAE (images, Bernoulli) ===")
    X = gen_images(512, C=1, H=28, W=28, seed=0)
    key = jax.random.PRNGKey(2)
    model = ConvVAE(
        image_size=28, channels=1, hidden_channels=32, latent_dim=16, key=key
    )
    cfg = TrainConfig(
        epochs=5, batch_size=64, likelihood="bernoulli", beta_warmup_steps=3000
    )
    model = train_vae(model, X, cfg)
    rng = jax.random.PRNGKey(6)
    dec, mu, logvar = model(X[:8], rng, train=False)
    print("recon logits shape:", dec.shape)


def test_attention_seq():
    print("=== MultiHeadAttentionVAE (sequence, Gaussian) ===")
    X = gen_seq(1000, T=20, D=1, seed=0)
    key = jax.random.PRNGKey(3)
    model = MultiHeadAttentionVAE(
        input_dim=1,
        latent_dim=8,
        hidden_dim=16,
        output_dim=1,
        seq_length=20,
        num_heads=1,
        key=key,
    )
    cfg = TrainConfig(
        epochs=10, batch_size=128, likelihood="gaussian", beta_warmup_steps=3000
    )
    model = train_vae(model, X, cfg)
    rng = jax.random.PRNGKey(7)
    dec, mu, logvar = model(X[:8], rng, train=False)
    print("recon shape:", dec.shape)


def test_vit():
    print("=== ViT_VAE (images, Gaussian) ===")
    X = gen_images(256, C=1, H=28, W=28, seed=0)
    key = jax.random.PRNGKey(4)
    model = ViT_VAE(
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
    cfg = TrainConfig(
        epochs=3, batch_size=16, likelihood="gaussian", beta_warmup_steps=2000
    )
    model = train_vae(model, X, cfg)
    rng = jax.random.PRNGKey(8)
    dec, mu, logvar = model(X[:4], rng, train=False)
    print("recon shape:", dec.shape)


if __name__ == "__main__":
    test_mlp()
    test_conv()
    test_attention_seq()
    test_vit()
