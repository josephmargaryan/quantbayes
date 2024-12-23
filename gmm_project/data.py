import jax.numpy as jnp
from jax import random


def generate_synthetic_data(num_samples=500, K=2):
    """Generates synthetic data for Gaussian Mixture Model."""
    key = random.PRNGKey(0)
    weights = jnp.array([0.3, 0.7])
    means = jnp.array([-2.0, 2.0])
    std_devs = jnp.array([0.5, 1.0])
    components = random.choice(key, jnp.arange(K), p=weights, shape=(num_samples,))
    data = random.normal(key, (num_samples,)) * std_devs[components] + means[components]
    return data
