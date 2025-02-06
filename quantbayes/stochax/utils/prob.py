import jax.numpy as jnp
from jax import tree_util
import equinox as eqx
import numpyro
import numpyro.distributions as dist


# ------------------------------------------------------------------------------
# Helper: Bayesianize an Equinox module
# ------------------------------------------------------------------------------
def bayesianize(module: eqx.Module, prior_fn):
    """
    Traverse the module's pytree and, for every array leaf, replace it with a
    NumPyro sample drawn from `prior_fn` (a function accepting a shape and
    returning a distribution). Each sample is tagged with a unique name.
    """
    leaves, treedef = tree_util.tree_flatten(module)
    new_leaves = []
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, jnp.ndarray):
            # Draw a sample for this parameter using the provided prior function.
            new_leaf = numpyro.sample(f"param_{i}", prior_fn(leaf.shape))
            new_leaves.append(new_leaf)
        else:
            new_leaves.append(leaf)
    return tree_util.tree_unflatten(treedef, new_leaves)


# ------------------------------------------------------------------------------
# Helper: Prior function defined as a normal distribution
# ------------------------------------------------------------------------------
def prior_fn(shape):
    return dist.Normal(0, 1).expand(shape).to_event(len(shape))
