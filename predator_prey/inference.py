import os
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS, Predictive


def run_mcmc(model, data, num_warmup, num_samples, num_chains):
    """Runs MCMC for the given model and data."""
    mcmc = MCMC(
        NUTS(model, dense_mass=True),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(PRNGKey(1), N=data.shape[0], y=data)
    mcmc.print_summary()
    return mcmc


def predict(model, mcmc_samples, N):
    """Generates posterior predictive samples."""
    predictive = Predictive(model, mcmc_samples)
    return predictive(PRNGKey(2), N)["y"]
