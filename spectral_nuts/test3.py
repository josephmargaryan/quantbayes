"""
A full example that:
  1. Defines an FFT‐based circulant layer with custom JVP.
  2. Implements a CirculantProcess layer that returns Fourier coefficients and a kernel.
  3. Defines a spectral model that uses the layer.
  4. Implements a slight modification of the NUTS kernel (“SpectralNUTS”) where we only change
     the momentum generation and kinetic energy for a site named "spectral" (using a fixed spectral mass),
     and an adaptive version (AdaptiveSpectralNUTS) that updates the spectral mass.
  5. Runs inference with default NUTS, fixed SpectralNUTS, and AdaptiveSpectralNUTS so that the test MSE can be compared.
  
In our modified kernels the spectral site is sampled as usual but when generating momentum
for that site, we scale by a provided spectral mass and compute its kinetic energy accordingly.
If spectral_mass is set to 1, the behavior is identical to standard NUTS.
AdaptiveSpectralNUTS updates the spectral mass based on the Fourier coefficients, which may help
improve performance.
"""

import os
import math
import warnings
from collections import OrderedDict, namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.flatten_util import ravel_pytree

import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.hmc_util import euclidean_kinetic_energy, IntegratorState, velocity_verlet, warmup_adapter

# =============================================================================
# FFT-based circulant multiplication with custom JVP.
# =============================================================================

@jax.custom_jvp
def spectral_circulant_matmul(x: jnp.ndarray, fft_full: jnp.ndarray) -> jnp.ndarray:
    padded_dim = fft_full.shape[0]
    single_example = (x.ndim == 1)
    if single_example:
        x = x[None, :]
    d_in = x.shape[-1]
    if d_in < padded_dim:
        pad_len = padded_dim - d_in
        x_pad = jnp.pad(x, ((0, 0), (0, pad_len)))
    elif d_in > padded_dim:
        x_pad = x[..., :padded_dim]
    else:
        x_pad = x
    X_fft = jnp.fft.fft(x_pad, axis=-1)
    y_fft = X_fft * fft_full[None, :]
    y = jnp.fft.ifft(y_fft, axis=-1).real
    if single_example:
        return y[0]
    return y

@spectral_circulant_matmul.defjvp
def spectral_circulant_matmul_jvp(primals, tangents):
    x, fft_full = primals
    dx, dfft = tangents
    padded_dim = fft_full.shape[0]
    
    single_example = (x.ndim == 1)
    if single_example:
        x = x[None, :]
    if dx is not None and dx.ndim == 1:
        dx = dx[None, :]
    d_in = x.shape[-1]
    if d_in < padded_dim:
        pad_len = padded_dim - d_in
        x_pad = jnp.pad(x, ((0, 0), (0, pad_len)))
        dx_pad = jnp.pad(dx, ((0, 0), (0, pad_len))) if dx is not None else None
    elif d_in > padded_dim:
        x_pad = x[..., :padded_dim]
        dx_pad = dx[..., :padded_dim] if dx is not None else None
    else:
        x_pad = x
        dx_pad = dx
    X_fft = jnp.fft.fft(x_pad, axis=-1)
    primal_y_fft = X_fft * fft_full[None, :]
    primal_y = jnp.fft.ifft(primal_y_fft, axis=-1).real

    if dx_pad is None:
        dX_fft = 0.0
    else:
        dX_fft = jnp.fft.fft(dx_pad, axis=-1)
    if dfft is None:
        term2 = 0.0
    else:
        term2 = X_fft * dfft[None, :]
    dY_fft = dX_fft * fft_full[None, :] + term2
    dY = jnp.fft.ifft(dY_fft, axis=-1).real
    if single_example:
        return primal_y[0], dY[0]
    return primal_y, dY

# =============================================================================
# CirculantProcess: a spectral layer using circulant covariance.
# =============================================================================

class CirculantProcess:
    def __init__(
        self,
        in_features: int,
        padded_dim: int = None,
        alpha: float = None,
        alpha_prior=dist.HalfNormal(1.0),
        K: int = None,
        name: str = "spectral_circ_jvp",
        prior_fn=None,
        use_bias: bool = True,
    ):
        """
        :param in_features: Input dimension.
        :param padded_dim: If provided, pad/truncate inputs to this dimension.
                           Default is in_features.
        :param alpha: Fixed value for the decay exponent; if None, a hyperprior is used.
        :param alpha_prior: Prior distribution for alpha if it is not fixed.
        :param K: Number of active frequencies to keep; if None, use full half-spectrum.
        :param name: Base name for sample sites.
        :param prior_fn: Function mapping a scale to a distribution (default: Normal(0, scale)).
        :param use_bias: If True, add a simple bias term; if False, sample a latent function.
        """
        self.in_features = in_features
        self.padded_dim = padded_dim if padded_dim is not None else in_features
        self.alpha = alpha
        self.alpha_prior = alpha_prior
        self.name = name
        self.use_bias = use_bias

        self.k_half = (self.padded_dim // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        if prior_fn is None:
            self.prior_fn = lambda scale: dist.Normal(0.0, scale)
        else:
            self.prior_fn = prior_fn

        self._last_fft_full = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Sample or fix the decay exponent.
        if self.alpha is None:
            alpha = numpyro.sample(f"{self.name}_alpha", self.alpha_prior)
        else:
            alpha = self.alpha

        freq_idx = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_idx**alpha)

        active_idx = jnp.arange(self.K)
        active_real = numpyro.sample(
            f"{self.name}_real",
            self.prior_fn(prior_std[active_idx]).expand([self.K]).to_event(1),
        )
        active_imag = numpyro.sample(
            f"{self.name}_imag",
            self.prior_fn(prior_std[active_idx]).expand([self.K]).to_event(1),
        )

        full_real = jnp.zeros((self.k_half,))
        full_imag = jnp.zeros((self.k_half,))
        full_real = full_real.at[active_idx].set(active_real)
        full_imag = full_imag.at[active_idx].set(active_imag)

        # Enforce zero imaginary part for the DC (and Nyquist, if even).
        full_imag = full_imag.at[0].set(0.0)
        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            full_imag = full_imag.at[-1].set(0.0)

        half_complex = full_real + 1j * full_imag

        if (self.padded_dim % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )

        # Store Fourier coefficients for later diagnostics.
        self._last_fft_full = jax.lax.stop_gradient(fft_full)

        # Compute the circulant covariance kernel (first row) via the inverse FFT of the PSD.
        cov_row = jnp.fft.ifft(jnp.abs(fft_full) ** 2).real

        # Either add a learned bias term or sample a latent function.
        if self.use_bias:
            additional = numpyro.sample(
                f"{self.name}_bias_spectral",
                dist.Normal(0.0, 1.0).expand([self.padded_dim]).to_event(1),
            )
        else:
            scale = jnp.sqrt(jnp.maximum(cov_row, 1e-6))
            circ_dist = dist.Normal(jnp.zeros(self.padded_dim), scale).to_event(1)
            additional = numpyro.sample(f"{self.name}_latent_function", circ_dist)

        out = spectral_circulant_matmul(x, fft_full)
        if out.ndim == 2:
            return out + additional[None, :]
        else:
            return out + additional

    def get_fourier_coeffs(self) -> jnp.ndarray:
        if self._last_fft_full is None:
            raise ValueError("No Fourier coefficients available; call the layer first.")
        return self._last_fft_full

    def get_kernel(self) -> jnp.ndarray:
        """
        Returns the covariance kernel (impulse response) computed from the PSD,
        without any additional bias or latent function.
        """
        impulse = jnp.zeros(self.padded_dim)
        impulse = impulse.at[0].set(1.0)
        fft_full = self.get_fourier_coeffs()
        X_fft = jnp.fft.fft(impulse)
        PSD = jnp.abs(fft_full) ** 2
        kernel_fft = X_fft * PSD
        kernel = jnp.fft.ifft(kernel_fft).real
        return kernel

# =============================================================================
# Spectral model that uses the CirculantProcess.
# =============================================================================

def spectral_model(X, y=None):
    """
    A simple regression model that first applies a circulant (spectral) layer,
    then a nonlinearity and finally a linear map to produce predictions.
    """
    D = X.shape[1]
    fft_layer = CirculantProcess(in_features=D, use_bias=False)
    spectral_model.fft_layer = fft_layer  # attach for later diagnostics

    X_transformed = fft_layer(X)
    numpyro.deterministic("spectral", fft_layer.get_fourier_coeffs())

    X_transformed = jax.nn.tanh(X_transformed)

    W = numpyro.sample("W", dist.Normal(jnp.zeros((D, 1)), jnp.ones((D, 1))).to_event(2))
    b = numpyro.sample("b", dist.Normal(0, 1))
    mu = jnp.dot(X_transformed, W) + b
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("obs", dist.Normal(mu.squeeze(), sigma), obs=y)
    return mu

# =============================================================================
# SpectralNUTS: slight modification with fixed spectral mass.
# =============================================================================

class SpectralNUTS(NUTS):
    def __init__(self, model, init_spectral_mass, param_names=('_real', '_imag'), **kwargs):
        self.spectral_mass = init_spectral_mass
        self.param_names = param_names  # Names of real/imag parameters
        
        def spectral_kinetic_energy(mass_inv, r):
            energy = 0.0
            for key in self.param_names:
                if key in r:
                    energy += 0.5 * jnp.sum((r[key] ** 2) / self.spectral_mass)
            # Handle other parameters with identity mass
            other_energy = 0.5 * sum(jnp.sum(v**2) for k,v in r.items() if k not in self.param_names)
            return energy + other_energy
        
        super().__init__(model, kinetic_fn=spectral_kinetic_energy, **kwargs)

    def momentum_generator(self, prototype_r, rng_key):
        new_r = {}
        keys = random.split(rng_key, len(prototype_r))
        for i, (key, value) in enumerate(prototype_r.items()):
            if key in self.param_names:
                new_r[key] = random.normal(keys[i], shape=value.shape) * jnp.sqrt(self.spectral_mass)
            else:
                new_r[key] = random.normal(keys[i], shape=value.shape)
        return new_r

# =============================================================================
# AdaptiveSpectralNUTS: adaptive update of the spectral mass.
# =============================================================================

class AdaptiveSpectralNUTS(SpectralNUTS):
    """
    Extends SpectralNUTS by adaptively updating the spectral mass using an exponential moving average.
    """
    def __init__(self, model, init_spectral_mass, adapt_rate=0.05, **kwargs):
        self.adapt_rate = adapt_rate
        super().__init__(model, init_spectral_mass, **kwargs)
    
    def update_spectral_mass(self, spectral_sample):
        # spectral_sample is expected to be the Fourier coefficients from the "spectral" site.
        k_half = self.spectral_mass.shape[0]
        # Extract the first k_half coefficients (the half-spectrum)
        fourier_half = spectral_sample[:k_half]
        new_estimate = jnp.abs(fourier_half) ** 2
        self.spectral_mass = (1 - self.adapt_rate) * self.spectral_mass + self.adapt_rate * new_estimate

# =============================================================================
# Helper: generate synthetic regression data.
# =============================================================================

def generate_regression_data(n_continuous=12, random_seed=24):
    np.random.seed(random_seed)
    N = 200
    D = n_continuous
    X = np.random.randn(N, D)
    y = X @ np.random.randn(D) + np.random.randn(N) * 0.5
    import pandas as pd
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(D)])
    df["target"] = y
    return df

# =============================================================================
# Helper: get kernel for a given parameter set.
# =============================================================================

def get_kernel_for_params(model, X, param_dict, rng_key=random.PRNGKey(0)):
    with handlers.seed(rng_seed=rng_key):
        with handlers.substitute(data=param_dict):
            _ = model(X)
    kernel = model.fft_layer.get_kernel()
    return jax.device_get(kernel)

# =============================================================================
# Main: run experiments with default NUTS, fixed SpectralNUTS, and AdaptiveSpectralNUTS.
# =============================================================================

def main():
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    df = generate_regression_data(n_continuous=12, random_seed=24)
    X = df.drop("target", axis=1).values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)

    rng_key = random.PRNGKey(24)

    # Common hyperparameters.
    step_size = 0.5
    trajectory_length = 2 * math.pi
    adapt_step_size = True
    adapt_mass_matrix = True
    dense_mass = False
    target_accept_prob = 0.8

    # -----------------------------
    # Experiment 1: Default NUTS
    # -----------------------------
    default_nuts_kernel = NUTS(
        spectral_model,
        step_size=step_size,
        trajectory_length=trajectory_length,
        adapt_step_size=adapt_step_size,
        adapt_mass_matrix=adapt_mass_matrix,
        dense_mass=dense_mass,
        target_accept_prob=target_accept_prob,
    )
    mcmc_default = MCMC(default_nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc_default.run(rng_key, X=X_train, y=y_train)
    samples_default = mcmc_default.get_samples()
    predictive_default = Predictive(spectral_model, samples_default)
    preds_default = predictive_default(rng_key, X=X_test)
    mean_preds_default = preds_default["obs"].mean(axis=0)
    mse_default = mean_squared_error(np.array(y_test), np.array(mean_preds_default))
    print("Default NUTS Test MSE:", mse_default)

    # -----------------------------
    # Experiment 2: Fixed SpectralNUTS
    # -----------------------------
    D = X_train.shape[1]
    padded_dim = D
    k_half = (padded_dim // 2) + 1
    init_spectral_mass = jnp.ones(k_half)  # fixed spectral mass (all ones)
    spectral_nuts_kernel = SpectralNUTS(
        spectral_model,
        init_spectral_mass=init_spectral_mass,
        step_size=step_size,
        trajectory_length=trajectory_length,
        adapt_step_size=adapt_step_size,
        adapt_mass_matrix=adapt_mass_matrix,
        dense_mass=dense_mass,
        target_accept_prob=target_accept_prob,
    )
    mcmc_spectral = MCMC(spectral_nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc_spectral.run(rng_key, X=X_train, y=y_train)
    samples_spectral = mcmc_spectral.get_samples()
    predictive_spectral = Predictive(spectral_model, samples_spectral)
    preds_spectral = predictive_spectral(rng_key, X=X_test)
    mean_preds_spectral = preds_spectral["obs"].mean(axis=0)
    mse_spectral = mean_squared_error(np.array(y_test), np.array(mean_preds_spectral))
    print("Fixed SpectralNUTS Test MSE:", mse_spectral)

    # -----------------------------
    # Experiment 3: AdaptiveSpectralNUTS
    # -----------------------------
    init_spectral_mass = jnp.ones(k_half)
    adaptive_spectral_nuts_kernel = AdaptiveSpectralNUTS(
        spectral_model,
        init_spectral_mass=init_spectral_mass,
        adapt_rate=0.05,
        step_size=step_size,
        trajectory_length=trajectory_length,
        adapt_step_size=adapt_step_size,
        adapt_mass_matrix=adapt_mass_matrix,
        dense_mass=dense_mass,
        target_accept_prob=target_accept_prob,
    )
    mcmc_adaptive = MCMC(adaptive_spectral_nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc_adaptive.run(rng_key, X=X_train, y=y_train)
    samples_adaptive = mcmc_adaptive.get_samples()
    # Update the spectral mass using the average Fourier coefficients from the adaptive run.
    avg_fourier = jnp.mean(samples_adaptive.get("spectral"), axis=0)
    adaptive_spectral_nuts_kernel.update_spectral_mass(avg_fourier)
    print("Updated spectral mass:", adaptive_spectral_nuts_kernel.spectral_mass)
    predictive_adaptive = Predictive(spectral_model, samples_adaptive)
    preds_adaptive = predictive_adaptive(rng_key, X=X_test)
    mean_preds_adaptive = preds_adaptive["obs"].mean(axis=0)
    mse_adaptive = mean_squared_error(np.array(y_test), np.array(mean_preds_adaptive))
    print("AdaptiveSpectralNUTS Test MSE:", mse_adaptive)

if __name__ == "__main__":
    main()
