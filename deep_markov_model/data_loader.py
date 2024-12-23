import jax
import jax.numpy as jnp
from jax.random import PRNGKey, randint, normal


def _reverse_padded(padded, lengths):
    def _reverse_single(p, length):
        new = jnp.zeros_like(p)
        reverse = jnp.roll(p[::-1], length, axis=0)
        return new.at[:].set(reverse)

    return jax.vmap(_reverse_single)(padded, lengths)


def load_data(split="train", num_sequences=100, max_length=50, data_dim=88):
    """
    Generates synthetic data for testing.

    Args:
        split (str): Data split ('train' or 'valid'). Only used for distinction.
        num_sequences (int): Number of synthetic sequences to generate.
        max_length (int): Maximum length of the sequences.
        data_dim (int): Dimensionality of the data (e.g., pitch range for music).

    Returns:
        tuple: Synthetic sequences, reversed sequences, and sequence lengths.
    """
    rng_key = PRNGKey(0 if split == "train" else 1)

    lengths = randint(rng_key, (num_sequences,), minval=10, maxval=max_length)
    seqs = randint(
        rng_key, (num_sequences, max_length, data_dim), minval=0, maxval=2
    ) * normal(rng_key, (num_sequences, max_length, data_dim))
    seqs = (seqs > 0).astype(jnp.float32)

    seqs = jnp.where(
        jnp.arange(max_length)[None, :, None] < lengths[:, None, None],
        seqs,
        jnp.zeros_like(seqs),
    )

    return seqs, _reverse_padded(seqs, lengths), lengths
