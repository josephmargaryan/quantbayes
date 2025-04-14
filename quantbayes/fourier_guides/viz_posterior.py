"""
diagnostic_tool.py

A modular diagnostic tool for comparing the posterior approximations
obtained via NUTS (HMC) and SVI for Bayesian models in NumPyro.
This example demonstrates the tool for two types of regression models:
one using a spectral linear layer and one using a standard linear layer.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal, AutoLowRankMultivariateNormal, AutoGuideList
import numpyro.optim as optim
import matplotlib.pyplot as plt

# =============================================================================
# Models
# =============================================================================

# Example spectral model: uses the SpectralCirculantLayer from quantbayes.bnn.
from quantbayes.bnn import SpectralCirculantLayer  # your custom spectral layer
from fourier_family import AutoFourier
from guides import SpectralImagGuide, SpectralRealGuide

from quantbayes.fake_data import generate_regression_data

def spectral_model(X, y=None):
    """
    A regression model that uses the spectral layer.
    
    Parameters:
        X: Input data with shape (N, D).
        y: (Optional) Regression targets.
    """
    N, D = X.shape
    # Apply the spectral layer to inputs
    X_trans = SpectralCirculantLayer(D)(X)
    
    # Bayesian linear layer on the transformed features
    W = numpyro.sample("w", dist.Normal(0, 1).expand([D, 1]).to_event(2))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([1]).to_event(1))
    
    # Activation and prediction
    X_act = jax.nn.tanh(X_trans)
    out = jnp.dot(X_act, W) + b
    mu = jnp.squeeze(out)
    
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("likelihood", dist.Normal(mu, sigma), obs=y)

def linear_model(X, y=None):
    """
    A standard Bayesian linear regression model.
    
    Parameters:
        X: Input data with shape (N, D).
        y: (Optional) Regression targets.
    """
    N, D = X.shape
    # Bayesian linear layer directly on inputs
    W = numpyro.sample("w", dist.Normal(0, 1).expand([D, 1]).to_event(2))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([1]).to_event(1))
    
    # Activation and prediction (you can remove the tanh activation if you wish)
    X_act = jax.nn.tanh(X)
    out = jnp.dot(X_act, W) + b
    mu = jnp.squeeze(out)
    
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("likelihood", dist.Normal(mu, sigma), obs=y)

# =============================================================================
# Data Generation
# =============================================================================

def generate_synthetic_data(N=100, D=5, noise_std=0.1, seed=0):
    """
    Generate synthetic regression data.
    
    Returns:
        X: Feature matrix of shape (N, D)
        y: Target vector of shape (N,)
    """
    key = jax.random.PRNGKey(seed)
    X = jax.random.normal(key, (N, D))
    true_W = jnp.ones((D, 1))
    true_b = jnp.array([0.5])
    mu = jnp.squeeze(jnp.dot(jax.nn.tanh(X), true_W) + true_b)
    y = mu + noise_std * jax.random.normal(key, mu.shape)
    return X, y

# =============================================================================
# Inference Functions
# =============================================================================

def run_nuts_inference(model, X, y, num_warmup=500, num_samples=1000, rng_key=jax.random.PRNGKey(1)):
    """
    Run NUTS (MCMC) inference.
    
    Returns:
        mcmc_samples: Posterior samples obtained via MCMC.
    """
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, X, y)
    return mcmc.get_samples()

def run_svi_inference(model, X, y, num_steps=5000, step_size=0.01, rng_key=jax.random.PRNGKey(1)):
    """
    Run SVI inference with AutoNormal guide.
    
    Returns:
        svi_samples: Posterior samples drawn from the variational approximation.
        svi_params: The optimized variational parameters.
    """
    optimizer = optim.Adam(step_size=step_size)
    K = 3

    # Create custom guides for the spectral sites.
    spectral_real_guide = SpectralRealGuide(spectral_model, K=K)
    spectral_imag_guide = SpectralImagGuide(spectral_model, K=K)

    # For the remaining sites, use AutoNormal.  To ensure it does not interfere
    # with the spectral sites, block (hide) them.
    other_guide = AutoNormal(numpyro.handlers.block(spectral_model, hide=["spectral_circ_jvp_real", "spectral_circ_jvp_imag"]))

    # Now combine them using AutoGuideList.
    guide = AutoGuideList(spectral_model)
    guide.append(spectral_real_guide)
    guide.append(spectral_imag_guide)
    guide.append(other_guide)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(rng_key, num_steps=num_steps, X=X, y=y)
    svi_params = svi_result.params
    svi_samples = guide.sample_posterior(rng_key, svi_params, sample_shape=(1000,))
    return svi_samples, svi_params

# =============================================================================
# Diagnostic Visualization
# =============================================================================

def plot_comparison(param_name, mcmc_samples, svi_samples, flatten=True):
    """
    Plot histograms comparing the posterior distributions for a given parameter.
    
    Parameters:
        param_name: Name of the parameter to plot.
        mcmc_samples: Dictionary of samples from MCMC.
        svi_samples: Dictionary of samples from SVI.
        flatten: If True, flatten arrays before plotting.
    """
    mcmc_vals = mcmc_samples[param_name]
    svi_vals = svi_samples[param_name]
    
    if flatten:
        mcmc_vals = mcmc_vals.ravel()
        svi_vals = svi_vals.ravel()
    
    plt.figure(figsize=(8, 4))
    plt.hist(mcmc_vals, bins=30, alpha=0.5, label="NUTS", density=True)
    plt.hist(svi_vals, bins=30, alpha=0.5, label="SVI", density=True)
    plt.xlabel(param_name)
    plt.ylabel("Density")
    plt.title(f"Posterior comparison for {param_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_diagnostics(model, X, y, param_names=["w", "b", "sigma"], rng_key=jax.random.PRNGKey(1)):
    """
    Run both NUTS and SVI inference for a given model and dataset,
    and produce diagnostic plots comparing the posterior distributions.
    
    Parameters:
        model: A function defining the NumPyro model.
        X: Input data.
        y: Observations.
        param_names: List of parameter names to diagnose.
        rng_key: PRNGKey for reproducibility.
    """
    print("Running NUTS inference...")
    mcmc_samples = run_nuts_inference(model, X, y, rng_key=rng_key)
    
    print("Running SVI inference...")
    svi_samples, _ = run_svi_inference(model, X, y, rng_key=rng_key)
    
    for param in param_names:
        print(f"Plotting parameter: {param}")
        plot_comparison(param, mcmc_samples, svi_samples, flatten=True)

# =============================================================================
# Main Routine for Demonstration
# =============================================================================

def main():
    # Generate synthetic data
    X, y = generate_synthetic_data(N=100, D=5, noise_std=0.1, seed=0)

    print("Diagnostics for Spectral Model:")
    run_diagnostics(spectral_model, X, y, param_names=["spectral_circ_jvp_imag", "spectral_circ_jvp_real", "sigma"], rng_key=jax.random.PRNGKey(1))

    print("Diagnostics for Linear Model:")
    run_diagnostics(linear_model, X, y, param_names=["w", "b", "sigma"], rng_key=jax.random.PRNGKey(2))

if __name__ == '__main__':
    main()
