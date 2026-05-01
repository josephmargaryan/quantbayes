# diffusion/data/dataloaders.py

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


def generate_synthetic_image_dataset(num_samples=60000, shape=(1, 28, 28), key=None):
    """
    Placeholder for a synthetic image dataset, or you can load real data (MNIST, CIFAR, etc.)
    """
    if key is None:
        key = jr.PRNGKey(0)
    data = jr.normal(key, shape=(num_samples, *shape)) * 0.1  # random images
    return data
