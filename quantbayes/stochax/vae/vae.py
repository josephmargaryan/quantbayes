#!/usr/bin/env python3
"""
A simple MLP-based VAE in Equinox, training on random 2D data. 
- We use `equinox.nn.MLP` for the encoder/decoder. 
- Because MLP expects shape (in_size,), we apply `jax.vmap` over the batch dimension. 
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt


# ------------------
# 1. Data
# ------------------
def generate_mixture_data(rng_key, n=1000, x_dim=2):
    """
    Generate random points from a 2D mixture of Gaussians.
    """
    rng = np.random.default_rng(int(rng_key[0]))
    means = np.array([
        [-2, 0],
        [2, 0],
        [0, 2],
        [0, -2],
    ])
    n_clusters = len(means)

    X_list = []
    for _ in range(n):
        c = rng.integers(0, n_clusters)
        center = means[c]
        point = center + rng.normal(scale=0.5, size=(x_dim,))
        X_list.append(point)
    X = np.array(X_list)
    return jnp.array(X)


# ------------------
# 2. Encoder / Decoder
# ------------------
class Encoder(eqx.Module):
    """
    MLP encoder producing (mu, logvar) of q(z|x).
    We'll do: MLP(in_size= x_dim, out_size= 2*latent_dim).
    Then split the output into (mu, logvar).
    """
    net: eqx.nn.MLP
    latent_dim: int

    def __init__(self, x_dim, hidden_dim, latent_dim, *, key):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = eqx.nn.MLP(
            in_size=x_dim,
            out_size=2 * latent_dim,  # we want [mu, logvar]
            width_size=hidden_dim,
            depth=2,   # or however many layers you'd like
            key=key
        )

    def __call__(self, x_batch):
        """
        x_batch shape: (batch, x_dim)
        We apply MLP to each row => shape (batch, 2*latent_dim)
        """
        # MLP expects shape (x_dim,). We'll use jax.vmap to handle the batch.
        def forward_single(x):
            return self.net(x)  # shape (2*latent_dim,)
        out = jax.vmap(forward_single)(x_batch)
        mu, logvar = jnp.split(out, 2, axis=-1)
        return mu, logvar


class Decoder(eqx.Module):
    """
    MLP decoder producing recon_mu = p(x|z). We'll treat it as the mean of a Gaussian with fixed variance, for simplicity.
    """
    net: eqx.nn.MLP

    def __init__(self, latent_dim, hidden_dim, x_dim, *, key):
        super().__init__()
        self.net = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=x_dim,
            width_size=hidden_dim,
            depth=2,
            key=key
        )

    def __call__(self, z_batch):
        """
        z_batch shape: (batch, latent_dim)
        Return shape (batch, x_dim)
        """
        def forward_single(z):
            return self.net(z)
        return jax.vmap(forward_single)(z_batch)


class VAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    latent_dim: int

    def __init__(self, x_dim, hidden_dim, latent_dim, *, key):
        super().__init__()
        enc_key, dec_key = jax.random.split(key, 2)
        self.encoder = Encoder(x_dim, hidden_dim, latent_dim, key=enc_key)
        self.decoder = Decoder(latent_dim, hidden_dim, x_dim, key=dec_key)
        self.latent_dim = latent_dim

    def sample_z(self, rng, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * eps
        mu, logvar shape: (batch, latent_dim)
        """
        eps = jax.random.normal(rng, shape=mu.shape)
        sigma = jnp.exp(0.5 * logvar)
        return mu + sigma * eps

    def __call__(self, x_batch, rng):
        """
        Forward pass: return recon_mu, mu_z, logvar_z
        x_batch shape: (batch, x_dim)
        """
        mu_z, logvar_z = self.encoder(x_batch)  # each (batch, latent_dim)
        z = self.sample_z(rng, mu_z, logvar_z)  # (batch, latent_dim)
        recon_mu = self.decoder(z)              # (batch, x_dim)
        return recon_mu, mu_z, logvar_z


# ------------------
# 3. Loss and Train
# ------------------
@eqx.filter_jit
def loss_fn(model: VAE, x_batch, rng):
    """
    Negative ELBO
    """
    recon_mu, mu_z, logvar_z = model(x_batch, rng)
    # Reconstruction MSE (batch-averaged)
    recon_loss = jnp.mean(jnp.sum((x_batch - recon_mu) ** 2, axis=-1))
    # KL
    kl_div = 0.5 * jnp.mean(
        jnp.sum(jnp.exp(logvar_z) + (mu_z**2) - 1.0 - logvar_z, axis=-1)
    )
    return recon_loss + kl_div


@eqx.filter_jit
def make_step(model: VAE, x_batch, optimizer, opt_state, rng):
    loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)
    loss_val, grads = loss_and_grad_fn(model, x_batch, rng)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


def train_vae(model, data, batch_size=64, n_epochs=50, lr=1e-3, seed=42):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    rng = jax.random.PRNGKey(seed)
    n_data = data.shape[0]
    n_batches = int(np.ceil(n_data / batch_size))

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_data)
        data_shuf = data[perm]

        epoch_loss = 0.0
        for i in range(n_batches):
            batch = data_shuf[i * batch_size : (i + 1) * batch_size]
            rng, step_key = jax.random.split(rng)
            model, opt_state, loss_val = make_step(model, batch, optimizer, opt_state, step_key)
            epoch_loss += loss_val

        epoch_loss /= n_batches
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")

    return model


# ------------------
# 4. Main
# ------------------
def main():
    rng_key = jax.random.PRNGKey(0)
    data = generate_mixture_data(rng_key, n=2000, x_dim=2)
    print("Data shape:", data.shape)  # (2000,2)

    x_dim = 2
    hidden_dim = 32
    latent_dim = 2
    init_key = jax.random.PRNGKey(1)
    model = VAE(x_dim, hidden_dim, latent_dim, key=init_key)

    trained_model = train_vae(model, data, batch_size=64, n_epochs=50, lr=1e-3, seed=999)

    # Test reconstruction
    test_idx = np.random.choice(len(data), size=5, replace=False)
    test_batch = data[test_idx]
    rng_test = jax.random.PRNGKey(123)
    recon_mu, mu_z, logvar_z = trained_model(test_batch, rng_test)
    test_mse = jnp.mean(jnp.sum((test_batch - recon_mu) ** 2, axis=-1))
    print("Test MSE:", test_mse.item())

    # Visualization
    plt.figure(figsize=(6,6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.2, label="Data")
    plt.scatter(test_batch[:,0], test_batch[:,1], color="green", marker="x", label="Original")
    plt.scatter(recon_mu[:,0], recon_mu[:,1], color="red", label="Recon")
    plt.title("VAE with Equinox MLP")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
