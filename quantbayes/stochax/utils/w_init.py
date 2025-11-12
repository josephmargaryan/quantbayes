import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = [
    "xavier_init",
    "he_init",
    "orthogonal_init",
    "uniform_init",
    "apply_custom_initialization",
]


def xavier_init(key, shape, dtype=jnp.float32):
    """Xavier (Glorot) normal initialization."""
    fan_in, fan_out = shape
    std = jnp.sqrt(2.0 / (fan_in + fan_out))
    return jax.random.normal(key, shape, dtype) * std


def he_init(key, shape, dtype=jnp.float32):
    """He (Kaiming) normal initialization, often used with ReLU activations."""
    fan_in, _ = shape
    std = jnp.sqrt(2.0 / fan_in)
    return jax.random.normal(key, shape, dtype) * std


def orthogonal_init(key, shape, dtype=jnp.float32):
    """Orthogonal initialization. Works for 2D matrices."""
    if len(shape) < 2:
        raise ValueError("Orthogonal initialization requires at least 2 dimensions")
    a = jax.random.normal(key, shape, dtype)
    # Compute the SVD of the random matrix.
    u, s, v = jnp.linalg.svd(a, full_matrices=False)
    # Use whichever of u or v matches the desired shape.
    q = u if u.shape == shape else v
    return q.astype(dtype)


def uniform_init(key, shape, dtype=jnp.float32, scale=0.05):
    """Uniform initialization with a given scale."""
    return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)


def apply_custom_initialization(model, init_fn, key):
    """
    Recursively traverse the Equinox pytree in `model` and reinitialize
    the weights of any eqx.nn.Linear module using `init_fn`.

    Args:
        model: An Equinox module (or pytree) to reinitialize.
        init_fn: A function taking (key, shape, dtype) that returns new weights.
        key: A JAX PRNGKey used for randomness.

    Returns:
        A new model with reinitialized weights.
    """
    # Flatten the model into leaves, while keeping track of the pytree structure.
    flat, treedef = jax.tree_util.tree_flatten(model)
    # Count how many Linear modules we have.
    linear_indices = [i for i, x in enumerate(flat) if isinstance(x, eqx.nn.Linear)]

    # Generate one key per linear layer.
    keys = jax.random.split(key, len(linear_indices))

    # We'll use an iterator over our keys.
    key_iter = iter(keys)

    def update_fn(x):
        # If x is an instance of eqx.nn.Linear, update its weight.
        if isinstance(x, eqx.nn.Linear):
            current_key = next(key_iter)
            new_weight = init_fn(current_key, x.weight.shape, x.weight.dtype)
            # Replace weight using eqx.tree_at.
            return eqx.tree_at(lambda mod: mod.weight, x, new_value=new_weight)
        else:
            return x

    # Map over the whole pytree.
    new_flat = [update_fn(x) for x in flat]
    # Reconstruct the model.
    return jax.tree_util.tree_unflatten(treedef, new_flat)


# Example usage:
if __name__ == "__main__":
    import equinox as eqx
    import jax.random as random

    class MLP(eqx.Module):
        l1: eqx.nn.Linear
        l2: eqx.nn.Linear

        def __init__(self, in_features: int, key):
            keys = random.split(key, 2)
            self.l1 = eqx.nn.Linear(in_features, 128, key=keys[0])
            self.l2 = eqx.nn.Linear(128, 1, key=keys[1])

        def __call__(self, x):
            x = jax.nn.relu(self.l1(x))
            return self.l2(x)

    key = random.PRNGKey(42)
    model = MLP(10, key)

    # Reinitialize all linear layers in the model using the xavier_init function.
    new_key = random.PRNGKey(123)
    model = apply_custom_initialization(model, xavier_init, new_key)
