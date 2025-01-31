import jax
import jax.numpy as jnp
from jax import random

import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from numpyro.contrib.module import flax_module

########################################
# 1. Flax Modules for VAE
########################################

class Encoder(nn.Module):
    """
    A simple MLP encoder for q(z | x).
    x_dim: dimension of input
    hidden_dim: dimension of hidden layer
    z_dim: dimension of latent
    """
    x_dim: int
    hidden_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, x):
        # x: shape (batch, x_dim)
        relu = nn.relu
        # 2-layer MLP
        h = relu(nn.Dense(self.hidden_dim)(x))
        h = relu(nn.Dense(self.hidden_dim)(h))
        # output 2*z_dim => [mu, log_sigma]
        out = nn.Dense(2*self.z_dim)(h)
        mu, log_sigma = jnp.split(out, 2, axis=-1)
        return mu, log_sigma  # shape (batch, z_dim) each

class Decoder(nn.Module):
    """
    A simple MLP decoder for p(x | z).
    z_dim: dimension of latent
    hidden_dim: dimension of hidden layer
    x_dim: dimension of reconstructed data
    """
    z_dim: int
    hidden_dim: int
    x_dim: int

    @nn.compact
    def __call__(self, z):
        # z: shape (batch, z_dim)
        relu = nn.relu
        # 2-layer MLP
        h = relu(nn.Dense(self.hidden_dim)(z))
        h = relu(nn.Dense(self.hidden_dim)(h))
        # output 2*x_dim => [mu_x, log_sigma_x]
        out = nn.Dense(2*self.x_dim)(h)
        mu_x, log_sigma_x = jnp.split(out, 2, axis=-1)
        return mu_x, log_sigma_x

########################################
# 2. VAE Model / Guide in NumPyro
########################################

def vae_model(x, encoder_dim, decoder_dim, z_dim):
    """
    Model p(x, z):
      z ~ Normal(0, 1)
      x ~ Normal(decoder(z))
    """
    # 1) Register a decoder submodule
    #    We'll guess shapes (batch=1, z_dim) for init
    dec_mod = flax_module("decoder",
        Decoder(z_dim=z_dim, hidden_dim=decoder_dim, x_dim=x.shape[-1]),
        input_shape=(1, z_dim)
    )

    batch_size = x.shape[0]

    with numpyro.plate("data_plate", batch_size):
        # prior on z
        z = numpyro.sample("z",
            dist.Normal(jnp.zeros(z_dim), 1.0).to_event(1))

        # decode => mu_x, log_sigma_x
        mu_x, log_sigma_x = dec_mod(z)
        sigma_x = jnp.exp(log_sigma_x)

        numpyro.sample("obs",
            dist.Normal(mu_x, sigma_x).to_event(1),
            obs=x)

def vae_guide(x, encoder_dim, decoder_dim, z_dim):
    """
    Guide q(z | x):
      z ~ Normal(enc_mu, enc_sigma)
    """
    # 1) Register an encoder submodule
    enc_mod = flax_module("encoder",
        Encoder(x_dim=x.shape[-1], hidden_dim=encoder_dim, z_dim=z_dim),
        input_shape=(1, x.shape[-1])
    )

    batch_size = x.shape[0]

    with numpyro.plate("data_plate", batch_size):
        mu_q, log_sigma_q = enc_mod(x)
        sigma_q = jnp.exp(log_sigma_q)
        numpyro.sample("z",
            dist.Normal(mu_q, sigma_q).to_event(1))

########################################
# 3. Synthetic Data + SVI Training
########################################

def generate_synthetic_vae_data(
    rng_key,
    n=500,
    x_dim=2
):
    """
    We'll create random 'toy' data from a ground-truth 2D distribution.
    For example, let's do a mixture of a few gaussians.
    """
    rng = np.random.default_rng(int(rng_key[0]))  # Convert JAX array to Python int

    # Define cluster centers
    means = np.array([[-2, 0],
                      [ 2, 0],
                      [0, 2],
                      [0, -2]])
    n_clusters = len(means)

    X = []
    for i in range(n):
        c = rng.integers(0, n_clusters)  # Choose a cluster
        center = means[c]
        x_i = center + rng.normal(scale=0.5, size=(x_dim,))  # Add Gaussian noise
        X.append(x_i)
    return jnp.array(X)  # shape (n, x_dim)

def train_vae(
    x_data,
    z_dim=2,
    encoder_dim=64,
    decoder_dim=64,
    lr=1e-3,
    num_steps=2000,
    seed=0
):
    def model_fn(data):
        return vae_model(data, encoder_dim, decoder_dim, z_dim)
    def guide_fn(data):
        return vae_guide(data, encoder_dim, decoder_dim, z_dim)

    optimizer = Adam(lr)
    svi = SVI(model_fn, guide_fn, optimizer, Trace_ELBO())
    rng_key = random.PRNGKey(seed)
    svi_state = svi.init(rng_key, x_data)

    losses = []
    for step in range(1, num_steps + 1):
        svi_state, loss_val = svi.update(svi_state, x_data)
        losses.append(loss_val)
        if step % max(1, num_steps//5) == 0 or step == 1:
            print(f"[Step {step}] ELBO = {-loss_val:.4f}")
    trained_params = svi.get_params(svi_state)
    return trained_params, np.array(losses)

########################################
# 4. Posterior Means & Plot Helpers
########################################

def posterior_means(
    trained_params,
    x_data: jnp.ndarray,
    z_dim: int,
    encoder_dim: int,
    decoder_dim: int
):
    """
    Extracts the posterior means from the encoder.
    """
    enc_def = Encoder(x_dim=x_data.shape[-1], hidden_dim=encoder_dim, z_dim=z_dim)
    enc_params = trained_params["encoder$params"]

    # Compute posterior means
    mu_q, _ = enc_def.apply({"params": enc_params}, x_data)
    return mu_q  # shape (batch, z_dim)

def compute_reconstruction_mse(x_true, x_recon):
    mse = jnp.mean((x_true - x_recon) ** 2)
    print(f"Reconstruction MSE: {mse:.4f}")
    return mse

def compute_latent_mse(z_true, z_post):
    mse = jnp.mean((z_true - z_post) ** 2)
    print(f"Latent MSE: {mse:.4f}")
    return mse

def visualize_latents_all(z_true, z_post):
    # Flatten the batch dimension
    z_true_flat = z_true.reshape(-1, z_true.shape[-1])    # Shape: (n_samples, z_dim)
    z_post_flat = z_post.reshape(-1, z_post.shape[-1])    # Shape: (n_samples, z_dim)
    
    plt.figure(figsize=(8,8))
    plt.scatter(z_true_flat[:,0], z_true_flat[:,1], label='True Latent', alpha=0.5, color='blue')
    plt.scatter(z_post_flat[:,0], z_post_flat[:,1], label='Posterior Mean', alpha=0.5, color='red')
    plt.legend()
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent Space Comparison Across All Samples")
    plt.show()

def visualize_latents_multiple(z_true, z_post, num_sequences=5):
    for seq_idx in range(num_sequences):
        plt.figure(figsize=(6,6))
        plt.scatter(z_true[seq_idx,:,0], z_true[seq_idx,:,1], label='True Latent', color='blue')
        plt.scatter(z_post[seq_idx,:,0], z_post[seq_idx,:,1], label='Posterior Mean', alpha=0.6, color='red')
        plt.legend()
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.title(f"Latent Space Comparison for Sample {seq_idx}")
        plt.show()

########################################
# 5. Main Demo
########################################

def main():
    rng_key = jax.random.PRNGKey(0)
    x_dim = 2
    n_data = 500

    # Generate data
    data = generate_synthetic_vae_data(rng_key, n=n_data, x_dim=x_dim)
    print("Data shape:", data.shape)  # (500, 2)

    # Train VAE
    z_dim = 2
    trained_params, losses = train_vae(
        data,
        z_dim=z_dim,
        encoder_dim=64,
        decoder_dim=64,
        lr=1e-3,
        num_steps=2000,
        seed=42
    )

    # Plot training losses
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.title("VAE Training - Negative ELBO")
    plt.xlabel("Iteration")
    plt.ylabel("Negative ELBO")
    plt.show()

    # Extract posterior means
    z_post_means = posterior_means(trained_params, data, z_dim=z_dim, encoder_dim=64, decoder_dim=64)

    # Compare latents for a few samples
    sample_indices = [0, 1, 2, 3, 4]
    for idx in sample_indices:
        plt.figure(figsize=(6,4))
        plt.scatter(z_post_means[idx,0], z_post_means[idx,1], label='Posterior Mean', color='red')
        plt.scatter(0, 0, label='True Mean', color='blue')  # Since true z ~ N(0,1)
        plt.legend()
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.title(f"Latent Space for Sample {idx}")
        plt.show()

    # Visualize latent space across all samples
    visualize_latents_all(jnp.zeros_like(z_post_means), z_post_means)  # Assuming true z ~ N(0,1)

    # Reconstruct some samples
    decoder_def = Decoder(z_dim=z_dim, hidden_dim=64, x_dim=x_dim)
    decoder_params = trained_params["decoder$params"]
    mu_x, log_sigma_x = decoder_def.apply({"params": decoder_params}, z_post_means[:5])  # Reconstruct first 5 samples
    sigma_x = jnp.exp(log_sigma_x)

    # Plot reconstructions
    for i in range(5):
        plt.figure(figsize=(6,4))
        plt.scatter(data[i,0], data[i,1], label='True x', color='blue')
        plt.scatter(mu_x[i,0], mu_x[i,1], label='Reconstructed x', color='red')
        plt.legend()
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Reconstruction for Sample {i}")
        plt.show()

    # Compute quantitative metrics for first 5 samples
    for i in range(5):
        print(f"Sample {i}:")
        compute_reconstruction_mse(data[i], mu_x[i])
        compute_latent_mse(jnp.zeros(z_dim), z_post_means[i])  # Since true z ~ N(0,1), mean is 0

if __name__ == "__main__":
    main()
