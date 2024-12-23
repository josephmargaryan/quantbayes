import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def gmm_model(data=None, K=2):
    """Defines the Gaussian Mixture Model."""
    weights = numpyro.sample("weights", dist.Dirichlet(0.5 * jnp.ones(K)))
    scale = numpyro.sample("scale", dist.LogNormal(0.0, 2.0))
    with numpyro.plate("components", K):
        locs = numpyro.sample("locs", dist.Normal(0.0, 10.0))

    with numpyro.plate("data", len(data)):
        assignment = numpyro.sample(
            "assignment", dist.Categorical(weights), infer={"enumerate": "parallel"}
        )
        numpyro.sample("obs", dist.Normal(locs[assignment], scale), obs=data)
