from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_value
from numpyro import handlers
import jax.numpy as jnp
from jax import random


def initialize_guide(model, data, K=2):
    """Initializes the variational guide."""
    init_values = {
        "weights": jnp.ones(K) / K,
        "scale": jnp.sqrt(data.var() / 2),
        "locs": data[
            random.categorical(
                random.PRNGKey(0), jnp.ones(len(data)) / len(data), shape=(K,)
            )
        ],
    }

    def wrapped_model(*args, **kwargs):
        """Wrapper to ensure a PRNGKey is passed."""
        with handlers.seed(rng_seed=0):
            return model(*args, **kwargs)

    guide = AutoDelta(
        handlers.block(
            wrapped_model,
            hide_fn=lambda site: site["name"]
            not in ["weights", "scale", "locs", "components"],
        ),
        init_loc_fn=init_to_value(values=init_values),
    )
    return guide
