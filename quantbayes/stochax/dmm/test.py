#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np

from quantbayes.stochax.dmm.components import DeepMarkovModel


def generate_synthetic_sequences(rng_key, n_sequences=64, T=15, x_dim=1, z_dim=2):
    """
    Generate synthetic sequences:
      - x_data: shape (n_sequences, T, x_dim)
      - z_data: shape (n_sequences, T, z_dim) (latent ground truth; not used for training)
    """
    rng = np.random.default_rng(int(rng_key[0]))
    A = 0.5 * rng.normal(size=(z_dim, z_dim))
    B = 0.7 * rng.normal(size=(x_dim, z_dim))
    z_list = []
    x_list = []
    for _ in range(n_sequences):
        z_prev = rng.normal(size=(z_dim,))
        seq_z = []
        seq_x = []
        for t in range(T):
            z_t = A @ z_prev + 0.1 * rng.normal(size=(z_dim,))
            x_t = B @ z_t + 0.1 * rng.normal(size=(x_dim,))
            seq_z.append(z_t)
            seq_x.append(x_t)
            z_prev = z_t
        z_list.append(np.stack(seq_z, axis=0))
        x_list.append(np.stack(seq_x, axis=0))
    z_data = jnp.array(np.stack(z_list, axis=0))  # (n_sequences, T, z_dim)
    x_data = jnp.array(np.stack(x_list, axis=0))  # (n_sequences, T, x_dim)
    return x_data, z_data


if __name__ == "__main__":
    # Generate synthetic data.
    rng_key = jax.random.PRNGKey(0)
    x_data, z_data = generate_synthetic_sequences(
        rng_key, n_sequences=64, T=15, x_dim=1, z_dim=2
    )
    print("x_data shape (DMM):", x_data.shape)  # Expected: (64, 15, 1)
    print("z_data shape (DMM):", z_data.shape)  # Expected: (64, 15, 2)

    observation_dim = 1
    latent_dim = 2
    hidden_dim = 16
    key = jax.random.PRNGKey(1)

    # Instantiate the enhanced DMM.
    model = DeepMarkovModel(
        observation_dim,
        latent_dim,
        hidden_dim,
        transition_type="mlp",  # or "mlp", "linear"
        key=key,
    )

    # Train the model.
    model = model.fit(x_data, n_epochs=100, batch_size=8, lr=1e-3, seed=42)

    # Reconstruct one sequence (assume batch size 1).
    test_seq = x_data[0:1]  # shape (1, T, observation_dim)
    rng, rec_key = jax.random.split(rng_key)
    recon_seq = model.reconstruct(test_seq, rec_key, plot=True)
    print("Reconstructed sequence shape:", recon_seq.shape)
