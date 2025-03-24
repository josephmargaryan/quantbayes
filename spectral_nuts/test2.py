import os
import math
import warnings
from collections import OrderedDict, namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import tree_flatten

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.hmc_util import euclidean_kinetic_energy

from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# Import your CirculantProcess and data generator.
from quantbayes.bnn import CirculantProcess
from quantbayes.fake_data import generate_regression_data

# ------------------------------------------------------------------------------
# Custom kinetic energy function for the spectral domain.
# Here, we assume that the momentum (r) is a pytree (dictionary) with key "spectral".
def spectral_kinetic_energy(mass_inv, r, spectral_mass):
    energy = 0.0
    for key, value in r.items():
        if key == "spectral":
            energy += 0.5 * jnp.sum((value ** 2) / spectral_mass)
        else:
            energy += 0.5 * jnp.sum(value ** 2)
    return energy

# ------------------------------------------------------------------------------
# Custom momentum generator for the spectral domain.
def spectral_momentum_generator(prototype_r, spectral_mass, rng_key):
    new_r = {}
    keys = random.split(rng_key, len(prototype_r))
    for i, (key, value) in enumerate(prototype_r.items()):
        if key == "spectral":
            new_r[key] = random.normal(keys[i], shape=jnp.shape(value)) * jnp.sqrt(spectral_mass)
        else:
            new_r[key] = random.normal(keys[i], shape=jnp.shape(value))
    return new_r

# ------------------------------------------------------------------------------
# Minimal adaptive spectral NUTS Kernel.
# This subclass makes a small change: it uses custom spectral kinetic energy and momentum generation,
# and provides a method to update the spectral mass from the collected Fourier coefficients.
class AdaptiveSpectralNUTS(NUTS):
    def __init__(self, model, init_spectral_mass, adapt_rate=0.05, **kwargs):
        """
        :param model: The probabilistic model that uses the CirculantProcess.
        :param init_spectral_mass: Initial 1D jnp.array for the variances of the half-spectrum.
                                  For example, if padded_dim = D then k_half = (D//2)+1.
        :param adapt_rate: Adaptation rate for the exponential moving average of the spectral mass.
        :param kwargs: Additional keyword arguments for the standard NUTS kernel.
        """
        self.spectral_mass = init_spectral_mass  # shape: (k_half,)
        self.adapt_rate = adapt_rate
        # Define the spectral kinetic energy function using our current spectral_mass.
        def kinetic_fn(mass_inv, r):
            return spectral_kinetic_energy(mass_inv, r, self.spectral_mass)
        self.custom_kinetic_fn = kinetic_fn
        super().__init__(model, kinetic_fn=kinetic_fn, **kwargs)
    
    def momentum_generator(self, prototype_r, rng_key):
        return spectral_momentum_generator(prototype_r, self.spectral_mass, rng_key)
    
    def update_spectral_mass(self, fourier_sample):
        """
        Update the spectral mass using an exponential moving average.
        Assumes fourier_sample is the full Fourier transform (shape: (padded_dim,)).
        We extract the half-spectrum (first k_half coefficients) so that the update matches the shape.
        """
        k_half = self.spectral_mass.shape[0]
        fourier_half = fourier_sample[:k_half]
        new_estimate = jnp.abs(fourier_half) ** 2
        self.spectral_mass = (1 - self.adapt_rate) * self.spectral_mass + self.adapt_rate * new_estimate

# ------------------------------------------------------------------------------
# Define a spectral model that uses the CirculantProcess.
def spectral_model(X, y=None):
    # X has shape (N, D); treat D as the input dimension.
    D = X.shape[1]
    cp_layer = CirculantProcess(in_features=D, use_bias=False)
    X_transformed = cp_layer(X)
    # Register Fourier coefficients as a deterministic site.
    fourier = cp_layer.get_fourier_coeffs()
    numpyro.deterministic("spectral", fourier)
    
    # Apply a nonlinearity.
    X_transformed = jax.nn.tanh(X_transformed)
    
    # Simple linear mapping.
    W = numpyro.sample("W", dist.Normal(jnp.zeros((D, 1)), jnp.ones((D, 1))).to_event(2))
    b = numpyro.sample("b", dist.Normal(0, 1))
    mu = jnp.dot(X_transformed, W) + b
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("obs", dist.Normal(mu.squeeze(), sigma), obs=y)

# ------------------------------------------------------------------------------
# Test code: Run adaptive spectral NUTS on regression data.
def main():
    # Generate regression data.
    df = generate_regression_data(n_continuous=12, random_seed=24)
    X, y = df.drop("target", axis=1), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    
    rng_key = random.PRNGKey(24)
    
    # Set up initial spectral mass.
    D = X_train.shape[1]
    padded_dim = D  # For simplicity, we use D as the padded dimension.
    k_half = (padded_dim // 2) + 1
    init_spectral_mass = jnp.ones(k_half)
    
    # Instantiate the adaptive spectral NUTS kernel with minimal modifications.
    adaptive_nuts_kernel = AdaptiveSpectralNUTS(
        spectral_model,
        init_spectral_mass=init_spectral_mass,
        adapt_rate=0.05  # A modest adaptation rate.
    )
    
    mcmc = MCMC(adaptive_nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc.run(rng_key, X=X_train, y=y_train)
    samples = mcmc.get_samples()
    
    # Update spectral mass based on the average Fourier coefficients.
    fourier_samples = samples.get("spectral")
    if fourier_samples is not None:
        avg_fourier = jnp.mean(fourier_samples, axis=0)
        adaptive_nuts_kernel.update_spectral_mass(avg_fourier)
        print("Updated spectral mass:", adaptive_nuts_kernel.spectral_mass)
    
    predictive = Predictive(spectral_model, samples)
    preds = predictive(rng_key, X=X_test)
    mean_preds = preds["obs"].mean(axis=0)
    
    loss = mean_squared_error(np.array(y_test), np.array(mean_preds))
    print("Test MSE:", loss)

if __name__ == "__main__":
    main()
