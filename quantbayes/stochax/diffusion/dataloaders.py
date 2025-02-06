# score_diffusion/data/dataloaders.py

import jax
import jax.numpy as jnp
import jax.random as jr


def dataloader(dataset, batch_size, *, key):
    """
    Yields batches of data from a JAX array dataset.
    """
    n = dataset.shape[0]
    indices = jnp.arange(n)
    while True:
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, indices)
        for i in range(0, n, batch_size):
            batch_indices = perm[i : i + batch_size]
            yield dataset[batch_indices]


def generate_synthetic_time_series(num_samples=10000, seq_length=128, key=None):
    """
    Creates a simple synthetic sinusoidal or random time-series dataset.
    """
    if key is None:
        key = jr.PRNGKey(0)
    key_sin, key_noise = jr.split(key)
    t = jnp.linspace(0, 2 * jnp.pi, seq_length)
    sin_freq = jr.uniform(key_sin, shape=[num_samples], minval=1, maxval=5)
    # shape: [num_samples, seq_length]
    data = jnp.stack([jnp.sin(freq * t) for freq in sin_freq], axis=0)
    # add some noise
    noise = 0.1 * jr.normal(key_noise, shape=data.shape)
    data = data + noise
    # data shape: [num_samples, seq_length]
    return data


def generate_synthetic_image_dataset(num_samples=60000, shape=(1, 28, 28), key=None):
    """
    Placeholder for a synthetic image dataset, or you can load real data (MNIST, CIFAR, etc.)
    """
    if key is None:
        key = jr.PRNGKey(0)
    data = jr.normal(key, shape=(num_samples, *shape)) * 0.1  # random images
    return data
