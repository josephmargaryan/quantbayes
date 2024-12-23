from collections import defaultdict
from jax import pure_callback
import optax
import jax.numpy as jnp


def hook_optax(optimizer):
    """Hooks optimizer for gradient norm tracking."""
    gradient_norms = defaultdict(list)

    def append_grad(grad):
        for name, g in grad.items():
            gradient_norms[name].append(float(jnp.linalg.norm(g)))
        return grad

    def update_fn(grads, state, params=None):
        grads = pure_callback(append_grad, grads, grads)
        return optimizer.update(grads, state, params=params)

    return optax.GradientTransformation(optimizer.init, update_fn), gradient_norms
