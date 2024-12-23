from numpyro.infer import SVI, TraceEnum_ELBO, MCMC, NUTS
from numpyro import handlers
from jax import random


def run_svi(model, guide, optimizer, data, num_iters=200):
    """Runs SVI to optimize the variational parameters."""
    svi = SVI(model, guide, optimizer, loss=TraceEnum_ELBO())
    result = svi.run(random.PRNGKey(0), num_iters, data)
    return result


def run_mcmc(model, data, num_samples=250, num_warmup=50):
    """Runs MCMC to infer posterior distributions."""
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(random.PRNGKey(1), data=data)
    return mcmc
