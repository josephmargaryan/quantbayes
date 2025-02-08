#!/usr/bin/env python3
import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from quantbayes.stochax.vae.components import (
    ConvVAE,
    ResidualVAE,
    MLP_VAE,
    GRU_VAE,
    LSTM_VAE,
    MultiHeadAttentionVAE,
    ViT_VAE,
)


def generate_sequence_data(rng_key, n=1000, seq_length=20, feature_dim=1):
    """
    Generate synthetic sequence data.
    Output shape: (n, seq_length, feature_dim)
    """
    rng = jax.random.PRNGKey(int(rng_key[0]))
    data = jax.random.uniform(rng, shape=(n, seq_length, feature_dim))
    return data


def generate_mixture_data(rng_key, n=1000, x_dim=2):
    rng = np.random.default_rng(int(rng_key[0]))
    means = np.array([[-2, 0], [2, 0], [0, 2], [0, -2]])
    n_clusters = len(means)
    X_list = []
    for _ in range(n):
        c = rng.integers(0, n_clusters)
        center = means[c]
        point = center + rng.normal(scale=0.5, size=(x_dim,))
        X_list.append(point)
    X = np.array(X_list)
    return jnp.array(X)


def generate_synthetic_images(rng_key, n=1000, image_size=28, channels=1):
    rng = jax.random.PRNGKey(int(rng_key[0]))
    images = jax.random.uniform(rng, shape=(n, channels, image_size, image_size))
    return images


def test_mlp_vae():
    print("=== Testing MLP_VAE on mixture data ===")
    rng_key = jax.random.PRNGKey(0)

    data = generate_mixture_data(rng_key, n=2000, x_dim=2)
    print("Data shape (MLPVAE):", data.shape)  # (2000, 2)

    input_dim = 2
    hidden_dim = 32
    latent_dim = 2
    output_dim = 2

    key = jax.random.PRNGKey(1)

    model = MLP_VAE(input_dim, hidden_dim, latent_dim, output_dim, key=key)
    model = model.fit(data, batch_size=64, n_epochs=50, lr=1e-3, seed=42)

    rng, subkey = jax.random.split(rng_key)
    recon, mu, logvar = model.reconstruct(data[:64], subkey, plot=True)
    print("MLPVAE: Reconstruction shape:", recon.shape)


def test_conv_vae():
    print("=== Testing ConvVAE on synthetic images ===")
    rng_key = jax.random.PRNGKey(10)

    image_size = 28
    channels = 1

    data = generate_synthetic_images(
        rng_key, n=2000, image_size=image_size, channels=channels
    )
    print("Data shape (ConvVAE):", data.shape)  # (2000, 28, 28, 1)

    hidden_channels = 32
    latent_dim = 20

    key = jax.random.PRNGKey(11)

    model = ConvVAE(image_size, channels, hidden_channels, latent_dim, key=key)
    model = model.fit(data, batch_size=64, n_epochs=50, lr=1e-3, seed=42)

    rng, subkey = jax.random.split(rng_key)
    recon, mu, logvar = model.reconstruct(data[:4], subkey, plot=True)
    print("ConvVAE: Reconstruction shape:", recon.shape)


def test_residual_vae():
    print("=== Testing ResidualVAE on mixture data ===")
    rng_key = jax.random.PRNGKey(20)

    data = generate_mixture_data(rng_key, n=2000, x_dim=2)
    print("Data shape (ResidualVAE):", data.shape)

    input_dim = 2
    hidden_dim = 32
    latent_dim = 2
    output_dim = 2
    num_layers = 3

    key = jax.random.PRNGKey(21)

    model = ResidualVAE(
        input_dim, hidden_dim, latent_dim, output_dim, num_layers, key=key
    )
    model = model.fit(data, batch_size=64, n_epochs=50, lr=1e-3, seed=42)

    rng, subkey = jax.random.split(rng_key)
    recon, mu, logvar = model.reconstruct(data[:64], subkey, plot=True)
    print("ResidualVAE: Reconstruction shape:", recon.shape)


def test_gru_vae():
    print("=== Testing GRU_VAE on sequence data ===")
    rng_key = jax.random.PRNGKey(30)

    seq_length = 20
    feature_dim = 1
    data = generate_sequence_data(
        rng_key, n=1000, seq_length=seq_length, feature_dim=feature_dim
    )
    print("Data shape (GRU_VAE):", data.shape)

    input_dim = feature_dim
    hidden_size = 16
    latent_dim = 8
    output_dim = feature_dim

    key = jax.random.PRNGKey(31)

    model = GRU_VAE(input_dim, hidden_size, latent_dim, output_dim, seq_length, key=key)
    model = model.fit(data, batch_size=64, n_epochs=30, lr=1e-3, seed=42)

    rng, subkey = jax.random.split(rng_key)
    recon, mu, logvar = model.reconstruct(data[:64], subkey, plot=True)
    print("GRU_VAE: Reconstruction shape:", recon.shape)


def test_lstm_vae():
    print("=== Testing LSTM_VAE on sequence data ===")
    rng_key = jax.random.PRNGKey(40)

    seq_length = 20
    feature_dim = 1
    data = generate_sequence_data(
        rng_key, n=1000, seq_length=seq_length, feature_dim=feature_dim
    )
    print("Data shape (LSTM_VAE):", data.shape)

    input_dim = feature_dim
    hidden_size = 16
    latent_dim = 8
    output_dim = feature_dim

    key = jax.random.PRNGKey(41)

    model = LSTM_VAE(
        input_dim, hidden_size, latent_dim, output_dim, seq_length, key=key
    )
    model = model.fit(data, batch_size=64, n_epochs=30, lr=1e-3, seed=42)

    rng, subkey = jax.random.split(rng_key)
    recon, mu, logvar = model.reconstruct(data[:64], subkey, plot=True)
    print("LSTM_VAE: Reconstruction shape:", recon.shape)


def test_attention_vae():
    print("=== Testing MultiHeadAttentionVAE on sequence data ===")
    rng_key = jax.random.PRNGKey(50)

    seq_length = 20
    feature_dim = 1
    data = generate_sequence_data(
        rng_key, n=1000, seq_length=seq_length, feature_dim=feature_dim
    )
    print("Data shape (MultiHeadAttentionVAE):", data.shape)

    input_dim = feature_dim
    latent_dim = 8
    hidden_dim = 16
    output_dim = feature_dim
    num_heads = 1

    key = jax.random.PRNGKey(51)

    model = MultiHeadAttentionVAE(
        input_dim, latent_dim, hidden_dim, output_dim, seq_length, num_heads, key=key
    )
    model = model.fit(data, batch_size=64, n_epochs=300, lr=1e-3, seed=42)

    rng, subkey = jax.random.split(rng_key)
    recon, mu, logvar = model.reconstruct(data[:64], subkey, plot=True)
    print("MultiHeadAttentionVAE: Reconstruction shape:", recon.shape)


def test_vit_vae():
    print("=== Testing ViT_VAE on synthetic images ===")
    rng_key = jax.random.PRNGKey(60)

    image_size = 28
    channels = 1
    data = generate_synthetic_images(rng_key)
    print("Data shape (ViT_VAE):", data.shape)

    # ViT configuration
    patch_size = 4
    embedding_dim = 512
    num_layers = 2
    num_heads = 4
    latent_dim = 62
    dropout_rate = 0.1

    key = jax.random.PRNGKey(61)
    model = ViT_VAE(
        image_size=image_size,
        channels=channels,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        key=key,
    )

    model = model.fit(data, batch_size=16, n_epochs=2, lr=1e-3, seed=42)

    rng, subkey = jax.random.split(rng_key)
    model = eqx.tree_inference(model, value=True)
    recon, mu, logvar = model.reconstruct(data[:16], subkey, plot=True)
    print("ViT_VAE: Reconstruction shape:", recon.shape)

    # Compute reconstruction error (MSE)
    recon_error = jnp.mean((recon - data[:16]) ** 2)
    # Compute KL divergence
    kl_div = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1))

    print(f"Reconstruction Error: {recon_error:.4f}")
    print(f"KL Divergence: {kl_div:.4f}")


def main():
    # test_mlp_vae()
    # test_conv_vae()
    # test_residual_vae()
    # test_gru_vae()
    # test_lstm_vae()
    # test_attention_vae()
    test_vit_vae()


if __name__ == "__main__":
    main()
