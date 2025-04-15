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

