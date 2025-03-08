import jax
import equinox as eqx
import jax.numpy as jnp
from quantbayes.stochax.diffusion.sde import single_sample_fn


def generate(
    model,  # the trained diffusion model
    int_beta_fn,  # the beta function (e.g., int_beta_linear)
    sample_shape,  # shape of a single sample (e.g., (seq_length,) for time series or (1, 28, 28) for images)
    dt0,  # the dt0 parameter
    t1,  # the t1 parameter
    sample_key,  # a JAX random key (or a key to be split)
    num_samples: int,  # number of samples to generate
    data_stats: dict = None,  # optional: dict with keys "mean", "std", "min", "max" for denormalization
):
    model = eqx.tree_inference(model, value=True)
    # Split the provided key into the desired number of keys.
    sample_keys = jax.random.split(sample_key, num_samples)

    def sample_fn(k):
        return single_sample_fn(model, int_beta_fn, sample_shape, dt0, t1, k)

    # Vectorize the sampling function across all keys.
    samples = jax.vmap(sample_fn)(sample_keys)

    # Optionally denormalize and clip samples.
    if data_stats is not None:
        samples = samples * data_stats["std"] + data_stats["mean"]
        samples = jnp.clip(samples, data_stats["min"], data_stats["max"])

    return samples
