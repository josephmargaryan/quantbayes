import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoGuide
from numpyro.distributions import CirculantNormal
from numpyro.distributions import constraints

__all__ = [
    "SpectralRealGuide",
    "SpectralImagGuide",
    "SpectralRealCirculantGuide",
    "SpectralImagCirculantGuide"
]

# Custom guide for the real part of the Fourier coefficients.
class SpectralRealGuide(AutoGuide):
    def __init__(self, model, K):
        """
        Args:
            model: the model function.
            K: the dimensionality of the real-valued spectral coefficients.
        """
        self.K = K
        super().__init__(model)

    def __call__(self, *args, **kwargs):
        # Set up variational parameters for the real part.
        mean_real = numpyro.param("spectral_circ_jvp_real_mean", jnp.zeros((self.K,)))
        log_std_real = numpyro.param("spectral_circ_jvp_real_log_std", -3.0 * jnp.ones((self.K,)))
        sample_real = numpyro.sample(
            "spectral_circ_jvp_real",
            dist.Normal(mean_real, jnp.exp(log_std_real)).to_event(1)
        )
        return {"spectral_circ_jvp_real": sample_real}

    def sample_posterior(self, rng_key, params, *args, sample_shape=()):
        mean_real = params["spectral_circ_jvp_real_mean"]
        log_std_real = params["spectral_circ_jvp_real_log_std"]
        sample_real = dist.Normal(mean_real, jnp.exp(log_std_real)).sample(rng_key, sample_shape)
        return {"spectral_circ_jvp_real": sample_real}


# Custom guide for the imaginary part of the Fourier coefficients.
class SpectralImagGuide(AutoGuide):
    def __init__(self, model, K):
        """
        Args:
            model: the model function.
            K: the dimensionality of the imaginary-valued spectral coefficients.
        """
        self.K = K
        super().__init__(model)

    def __call__(self, *args, **kwargs):
        # Set up variational parameters for the imaginary part.
        mean_imag = numpyro.param("spectral_circ_jvp_imag_mean", jnp.zeros((self.K,)))
        log_std_imag = numpyro.param("spectral_circ_jvp_imag_log_std", -3.0 * jnp.ones((self.K,)))
        sample_imag = numpyro.sample(
            "spectral_circ_jvp_imag",
            dist.Normal(mean_imag, jnp.exp(log_std_imag)).to_event(1)
        )
        return {"spectral_circ_jvp_imag": sample_imag}

    def sample_posterior(self, rng_key, params, *args, sample_shape=()):
        mean_imag = params["spectral_circ_jvp_imag_mean"]
        log_std_imag = params["spectral_circ_jvp_imag_log_std"]
        sample_imag = dist.Normal(mean_imag, jnp.exp(log_std_imag)).sample(rng_key, sample_shape)
        return {"spectral_circ_jvp_imag": sample_imag}


#############################################
# Testing
#############################################

class SpectralRealCirculantGuide(AutoGuide):
    """
    AutoGuide for the real part of Fourier coefficients using a CirculantNormal.
    It parameterizes the mean and the first row of the circulant covariance with 
    safe initialization for numerical stability.
    """
    def __init__(self, model, K, init_scale=0.1, eps=1e-3):
        """
        Args:
            model: The model function.
            K: Dimensionality of the real-valued spectral coefficients.
            init_scale: Initial scale for the covariance row.
            eps: A small constant added for numerical stability.
        """
        self.K = K
        self.init_scale = init_scale
        self.eps = eps
        super().__init__(model)

    def __call__(self, *args, **kwargs):
        # Variational parameter for the mean (initialized at zero).
        mean_real = numpyro.param("spectral_circ_jvp_real_mean", jnp.zeros((self.K,)))
        # Variational parameter for the first row of the circulant covariance.
        # Initialize with a small scale (e.g., 0.1) for stability.
        cov_row_real = numpyro.param(
            "spectral_circ_jvp_real_cov_row",
            jnp.ones((self.K,)) * self.init_scale,
            constraint=constraints.positive
        )
        # Add epsilon to guard against very small values.
        cov_row_real = cov_row_real + self.eps

        # Sample using the CirculantNormal (which uses the circulant covariance structure).
        sample_real = numpyro.sample(
            "spectral_circ_jvp_real",
            dist.CirculantNormal(loc=mean_real, covariance_row=cov_row_real)
        )
        return {"spectral_circ_jvp_real": sample_real}

    def sample_posterior(self, rng_key, params, *args, sample_shape=()):
        mean_real = params["spectral_circ_jvp_real_mean"]
        cov_row_real = params["spectral_circ_jvp_real_cov_row"] + self.eps
        sample_real = dist.CirculantNormal(loc=mean_real, covariance_row=cov_row_real).sample(rng_key, sample_shape)
        return {"spectral_circ_jvp_real": sample_real}


class SpectralImagCirculantGuide(AutoGuide):
    """
    AutoGuide for the imaginary part of Fourier coefficients using a CirculantNormal.
    This guide also enforces that the DC component (and Nyquist frequency for even K)
    remains zero.
    """
    def __init__(self, model, K, init_scale=0.1, eps=1e-3):
        """
        Args:
            model: The model function.
            K: Dimensionality of the imaginary-valued spectral coefficients.
            init_scale: Initial scale for the covariance row.
            eps: A small constant added for numerical stability.
        """
        self.K = K
        self.init_scale = init_scale
        self.eps = eps
        super().__init__(model)

    def __call__(self, *args, **kwargs):
        # Variational parameter for the mean, initialized at zero.
        mean_imag = numpyro.param("spectral_circ_jvp_imag_mean", jnp.zeros((self.K,)))
        # Variational parameter for the first row of the circulant covariance.
        cov_row_imag = numpyro.param(
            "spectral_circ_jvp_imag_cov_row",
            jnp.ones((self.K,)) * self.init_scale,
            constraint=constraints.positive
        )
        # Add epsilon for stability.
        cov_row_imag = cov_row_imag + self.eps
        
        # Sample from the CirculantNormal.
        sample_imag = numpyro.sample(
            "spectral_circ_jvp_imag",
            dist.CirculantNormal(loc=mean_imag, covariance_row=cov_row_imag)
        )
        # Enforce that the DC component is zero.
        sample_imag = sample_imag.at[0].set(0.0)
        # For even K, also enforce the Nyquist frequency to be zero.
        if self.K % 2 == 0:
            sample_imag = sample_imag.at[-1].set(0.0)
        return {"spectral_circ_jvp_imag": sample_imag}

    def sample_posterior(self, rng_key, params, *args, sample_shape=()):
        mean_imag = params["spectral_circ_jvp_imag_mean"]
        cov_row_imag = params["spectral_circ_jvp_imag_cov_row"] + self.eps
        sample_imag = dist.CirculantNormal(loc=mean_imag, covariance_row=cov_row_imag).sample(rng_key, sample_shape)
        sample_imag = sample_imag.at[0].set(0.0)
        if self.K % 2 == 0:
            sample_imag = sample_imag.at[-1].set(0.0)
        return {"spectral_circ_jvp_imag": sample_imag}
