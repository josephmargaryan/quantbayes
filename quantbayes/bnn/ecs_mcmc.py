"""
mcmc_utils.py

A lightweight utility module for running NUTS and HMCECS (energy-conserving subsampling HMC)
with minimal boilerplate. Simply import `sample_hmcecs` and pass your model,
training data, and desired parameters.
"""

import time
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, HMCECS, SVI, Trace_ELBO, autoguide


def fit_reference(
    rng_key, model, guide, optimizer, num_svi_steps, X, y, subsample_size
):
    """
    Fit a reference point via SVI (AutoDelta) for use with HMCECS.
    Returns a dict of reference parameter values.
    """
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    result = svi.run(rng_key, num_svi_steps, X, y, subsample_size, progress_bar=False)
    # extract *_auto_loc entries and strip suffix
    ref_params = {
        name.replace("_auto_loc", ""): result.params[name] for name in result.params
    }
    return ref_params


def build_kernel_hmcecs(model_fn, ref_params, num_blocks=10):
    """Build an HMCECS kernel given reference parameters."""
    proxy = HMCECS.taylor_proxy(ref_params)
    # use NUTS as the inner integrator
    return HMCECS(NUTS(model_fn), num_blocks=num_blocks, proxy=proxy)


def run_mcmc(
    rng_key,
    model_fn,
    X,
    y,
    kernel,
    num_warmup,
    num_samples,
    subsample_size,
    progress_bar,
):
    """
    Generic MCMC runner. Returns the MCMC object after sampling.
    """
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, X, y, subsample_size)
    return mcmc


def sample_hmcecs(
    rng_key,
    model_fn,
    X,
    y,
    subsample_size,
    num_svi_steps=1000,
    num_warmup=500,
    num_samples=1000,
    num_blocks=10,
    progress_bar=False,
):
    """
    Run HMCECS: performs one SVI fit for reference, then energy-conserving HMC
    with data subsampling. Returns a dict of posterior samples.
    """
    # 1) Fit reference via SVI
    guide = autoguide.AutoDelta(model_fn)
    optimizer = numpyro.optim.Adam(1e-3)
    ref_params = fit_reference(
        rng_key, model_fn, guide, optimizer, num_svi_steps, X, y, subsample_size
    )
    # 2) Build HMCECS kernel
    kernel = build_kernel_hmcecs(model_fn, ref_params, num_blocks)
    # 3) Run MCMC with subsampling
    mcmc = run_mcmc(
        rng_key,
        model_fn,
        X,
        y,
        kernel,
        num_warmup,
        num_samples,
        subsample_size=subsample_size,
        progress_bar=progress_bar,
    )
    return mcmc.get_samples()


# Example usage for ECSS HMC
if __name__ == "__main__":
    import numpy as np
    from jax import random

    # Define your model; must accept (X, y, subsample_size)
    def logistic_model(X, y=None, subsample_size=None):
        D = X.shape[1]
        W = numpyro.sample("W", dist.Normal(0, 1).expand([D]).to_event(1))
        b = numpyro.sample("b", dist.Normal(0, 1))
        if subsample_size:
            with numpyro.plate("data", X.shape[0], subsample_size=subsample_size):
                batch_X = numpyro.subsample(X, event_dim=1)
                batch_y = numpyro.subsample(y, event_dim=0) if y is not None else None
                logits = jnp.dot(batch_X, W) + b
                numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=batch_y)
        else:
            logits = jnp.dot(X, W) + b
            with numpyro.plate("data", X.shape[0]):
                numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

    # Load or generate your training data
    N, D = 5000, 10
    X_train = np.random.randn(N, D)
    y_train = (X_train @ np.ones(D) + np.random.randn(N) > 0).astype(int)

    rng_key = random.PRNGKey(42)

    # Run energy-conserving subsampling HMC (HMCECS)
    samples = sample_hmcecs(
        rng_key,
        logistic_model,
        X_train,
        y_train,
        subsample_size=500,
        num_svi_steps=2000,
        num_warmup=1000,
        num_samples=2000,
        num_blocks=20,
        progress_bar=True,
    )

    print(
        "ECSS HMC completed; obtained",
        len(samples[list(samples.keys())[0]]),
        "samples.",
    )
