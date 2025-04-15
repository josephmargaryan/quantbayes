from typing import Callable
import jax
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoGuide
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform, ComposeTransform

__all__ = [
    "SpectralRealGuide",
    "SpectralImagGuide",
    "LowRankSpectralGuide",
    "FlowSpectralGuide",
    "MixtureSpectralGuide",
    "MultiScaleSpectralRealGuide",
    "MultiScaleSpectralImagGuide",
    "LowRankMultiScaleSpectralGuide",
    "BayesianConv2DLowRankGuide"

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



class LowRankSpectralGuide(AutoGuide):
    """
    Low-rank covariance guide for the spectral coefficients (real or imaginary).
    We define a rank-r factorization: Cov = V @ V^T + diag(eps).
    """
    def __init__(self,
                 model: Callable,
                 K: int,
                 rank: int = 2,
                 site_name: str = "spectral_circ_jvp_real"):
        """
        :param model: The model function.
        :param K: Number of frequencies to approximate.
        :param rank: The rank of the low-rank approximation.
        :param site_name: 'spectral_circ_jvp_real' or 'spectral_circ_jvp_imag'.
        """
        super().__init__(model)
        self.K = K
        self.rank = rank
        self.site_name = site_name

    def __call__(self, *args, **kwargs):
        # Mean vector
        mean = numpyro.param(f"{self.site_name}_mean", jnp.zeros((self.K,)))
        # Low-rank factor
        V_init = jnp.zeros((self.K, self.rank))
        V = numpyro.param(f"{self.site_name}_V", V_init)
        # Diagonal term
        diag_init = -3.0 * jnp.ones((self.K,))
        log_diag = numpyro.param(f"{self.site_name}_log_diag", diag_init)

        # Reconstruct covariance: Cov = V @ V^T + diag(exp(log_diag))
        # For reparameterization, we often use a standard normal and then transform it:
        eps = jnp.exp(log_diag)

        z = numpyro.sample(
            self.site_name,
            dist.MultivariateNormal(loc=mean,
                covariance_matrix=(V @ V.T) + jnp.diag(eps))
        )
        return {self.site_name: z}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        """
        Draw posterior samples given the final learned parameters.
        """
        mean = params[f"{self.site_name}_mean"]
        V = params[f"{self.site_name}_V"]
        log_diag = params[f"{self.site_name}_log_diag"]
        eps = jnp.exp(log_diag)
        cov = (V @ V.T) + jnp.diag(eps)

        # We can sample from MultivariateNormal
        dist_mv = dist.MultivariateNormal(mean, cov)
        samples = dist_mv.sample(rng_key, sample_shape=sample_shape)
        return {self.site_name: samples}
    
class FlowSpectralGuide(AutoGuide):
    def __init__(self, model, K, site_name, num_flows=2):
        self.K = K
        self.site_name = site_name  # e.g., "spectral_circ_jvp_real" or "spectral_circ_jvp_imag"
        self.num_flows = num_flows
        super().__init__(model)

    def __call__(self, *args, **kwargs):
        # Define the base distribution parameters.
        base_mean = numpyro.param(f"{self.site_name}_flow_mean", jnp.zeros((self.K,)))
        base_log_std = numpyro.param(f"{self.site_name}_flow_log_std", -3.0 * jnp.ones((self.K,)))
        base_dist = dist.Normal(base_mean, jnp.exp(base_log_std)).to_event(1)
        
        # Create a chain of affine transforms.
        transforms = []
        for i in range(self.num_flows):
            shift = numpyro.param(f"{self.site_name}_flow_shift_{i}", jnp.zeros((self.K,)))
            scale = numpyro.param(
                f"{self.site_name}_flow_scale_{i}", jnp.ones((self.K,)),
                constraint=constraints.positive
            )
            transforms.append(AffineTransform(loc=shift, scale=scale))
            
        flow_transform = ComposeTransform(transforms)
        flow_dist = dist.TransformedDistribution(base_dist, flow_transform)
        
        # Sample from the variational distribution.
        spectral_sample = numpyro.sample(self.site_name, flow_dist)
        return {self.site_name: spectral_sample}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        # Use the learned parameters stored in params.
        base_mean = params[f"{self.site_name}_flow_mean"]
        base_log_std = params[f"{self.site_name}_flow_log_std"]
        base_std = jnp.exp(base_log_std)
        base_dist = dist.Normal(base_mean, base_std).to_event(1)
        
        # Reconstruct the chain of affine transforms using the learned parameters.
        transforms = []
        for i in range(self.num_flows):
            shift = params[f"{self.site_name}_flow_shift_{i}"]
            scale = params[f"{self.site_name}_flow_scale_{i}"]
            transforms.append(AffineTransform(loc=shift, scale=scale))
            
        flow_transform = ComposeTransform(transforms)
        flow_dist = dist.TransformedDistribution(base_dist, flow_transform)
        
        spectral_sample = flow_dist.sample(rng_key, sample_shape)
        return {self.site_name: spectral_sample}

class MixtureSpectralGuide(AutoGuide):
    def __init__(self, model, K, site_name, num_components=3):
        """
        Mixture of Gaussians guide for spectral coefficients.
        
        Args:
            model: the model function.
            K: the dimensionality of the spectral coefficients (e.g., length of half spectrum).
            site_name: the name of the sample site to target (e.g., "spectral_circ_jvp_real" 
                       or "spectral_circ_jvp_imag").
            num_components: number of mixture components.
        """
        self.K = K
        self.site_name = site_name
        self.num_components = num_components
        super().__init__(model)

    def __call__(self, *args, **kwargs):
        # Mixture weight parameters (logits)
        logits = numpyro.param(f"{self.site_name}_mixture_logits", jnp.zeros((self.num_components,)))
        # Convert logits to probabilities via softmax.
        weights = jax.nn.softmax(logits)
        
        # Parameters for each Gaussian component.
        means = numpyro.param(f"{self.site_name}_mixture_means", jnp.zeros((self.num_components, self.K)))
        log_stds = numpyro.param(f"{self.site_name}_mixture_log_stds", -3.0 * jnp.ones((self.num_components, self.K)))
        
        # Build a component distribution: an independent Normal for each component.
        component_dist = dist.Independent(dist.Normal(means, jnp.exp(log_stds)), 1)
        
        # Create a MixtureSameFamily distribution using positional arguments.
        mixture_dist = dist.MixtureSameFamily(dist.Categorical(probs=weights), component_dist)
        
        # Sample from the variational distribution using the target site_name.
        spectral_sample = numpyro.sample(self.site_name, mixture_dist)
        return {self.site_name: spectral_sample}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        # Retrieve learned parameters.
        logits = params[f"{self.site_name}_mixture_logits"]
        weights = jax.nn.softmax(logits)
        means = params[f"{self.site_name}_mixture_means"]
        log_stds = params[f"{self.site_name}_mixture_log_stds"]
        
        component_dist = dist.Independent(dist.Normal(means, jnp.exp(log_stds)), 1)
        mixture_dist = dist.MixtureSameFamily(dist.Categorical(probs=weights), component_dist)
        
        spectral_sample = mixture_dist.sample(rng_key, sample_shape)
        return {self.site_name: spectral_sample}
    

#############################################
# Custom Guide for MultiScaleSpectralCirculantLayer:
#############################################

class MultiScaleSpectralRealGuide(AutoGuide):
    """
    Custom guide for the real parts of the multi-scale spectral layer.
    This guide sets up variational parameters for both the coarse and fine real components.
    """
    def __init__(self, model: Callable, coarse_K: int, fine_K: int, name: str = "multi_scale_spectral"):
        """
        Args:
            model: the model function.
            coarse_K: dimensionality for the coarse real coefficients.
            fine_K: dimensionality for the fine real coefficients.
            name: base name used in the model for sampling sites.
        """
        self.coarse_K = coarse_K
        self.fine_K = fine_K
        self.name = name
        super().__init__(model)

    def __call__(self, *args, **kwargs):
        # Coarse real part:
        mean_coarse = numpyro.param(f"{self.name}_coarse_real_mean", jnp.zeros((self.coarse_K,)))
        log_std_coarse = numpyro.param(f"{self.name}_coarse_real_log_std", -3.0 * jnp.ones((self.coarse_K,)))
        coarse_sample = numpyro.sample(
            f"{self.name}_coarse_real",
            dist.Normal(mean_coarse, jnp.exp(log_std_coarse)).to_event(1)
        )
        # Fine real part:
        mean_fine = numpyro.param(f"{self.name}_fine_real_mean", jnp.zeros((self.fine_K,)))
        log_std_fine = numpyro.param(f"{self.name}_fine_real_log_std", -3.0 * jnp.ones((self.fine_K,)))
        fine_sample = numpyro.sample(
            f"{self.name}_fine_real",
            dist.Normal(mean_fine, jnp.exp(log_std_fine)).to_event(1)
        )
        return {
            f"{self.name}_coarse_real": coarse_sample,
            f"{self.name}_fine_real": fine_sample,
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        mean_coarse = params[f"{self.name}_coarse_real_mean"]
        log_std_coarse = params[f"{self.name}_coarse_real_log_std"]
        coarse_sample = dist.Normal(mean_coarse, jnp.exp(log_std_coarse)).sample(rng_key, sample_shape)
        mean_fine = params[f"{self.name}_fine_real_mean"]
        log_std_fine = params[f"{self.name}_fine_real_log_std"]
        fine_sample = dist.Normal(mean_fine, jnp.exp(log_std_fine)).sample(rng_key, sample_shape)
        return {
            f"{self.name}_coarse_real": coarse_sample,
            f"{self.name}_fine_real": fine_sample,
        }

#############################################
# Custom Guide for the Imaginary Parts
#############################################

class MultiScaleSpectralImagGuide(AutoGuide):
    """
    Custom guide for the imaginary parts of the multi-scale spectral layer.
    This guide sets up variational parameters for both the coarse and fine imaginary components.
    """
    def __init__(self, model: Callable, coarse_K: int, fine_K: int, name: str = "multi_scale_spectral"):
        """
        Args:
            model: the model function.
            coarse_K: dimensionality for the coarse imaginary coefficients.
            fine_K: dimensionality for the fine imaginary coefficients.
            name: base name used in the model for sampling sites.
        """
        self.coarse_K = coarse_K
        self.fine_K = fine_K
        self.name = name
        super().__init__(model)

    def __call__(self, *args, **kwargs):
        # Coarse imaginary part:
        mean_coarse = numpyro.param(f"{self.name}_coarse_imag_mean", jnp.zeros((self.coarse_K,)))
        log_std_coarse = numpyro.param(f"{self.name}_coarse_imag_log_std", -3.0 * jnp.ones((self.coarse_K,)))
        coarse_sample = numpyro.sample(
            f"{self.name}_coarse_imag",
            dist.Normal(mean_coarse, jnp.exp(log_std_coarse)).to_event(1)
        )
        # Fine imaginary part:
        mean_fine = numpyro.param(f"{self.name}_fine_imag_mean", jnp.zeros((self.fine_K,)))
        log_std_fine = numpyro.param(f"{self.name}_fine_imag_log_std", -3.0 * jnp.ones((self.fine_K,)))
        fine_sample = numpyro.sample(
            f"{self.name}_fine_imag",
            dist.Normal(mean_fine, jnp.exp(log_std_fine)).to_event(1)
        )
        return {
            f"{self.name}_coarse_imag": coarse_sample,
            f"{self.name}_fine_imag": fine_sample,
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        mean_coarse = params[f"{self.name}_coarse_imag_mean"]
        log_std_coarse = params[f"{self.name}_coarse_imag_log_std"]
        coarse_sample = dist.Normal(mean_coarse, jnp.exp(log_std_coarse)).sample(rng_key, sample_shape)
        mean_fine = params[f"{self.name}_fine_imag_mean"]
        log_std_fine = params[f"{self.name}_fine_imag_log_std"]
        fine_sample = dist.Normal(mean_fine, jnp.exp(log_std_fine)).sample(rng_key, sample_shape)
        return {
            f"{self.name}_coarse_imag": coarse_sample,
            f"{self.name}_fine_imag": fine_sample,
        }

#############################################
# Low-Rank Variant of the Multi-Scale Guides
#############################################

class LowRankMultiScaleSpectralGuide(AutoGuide):
    """
    Low-rank covariance guide for a multi-scale spectral site (either real or imaginary).
    This guide uses a low-rank factorization for the covariance of the selected site.
    
    Args:
      model: the model function.
      K: number of frequencies for this site.
      rank: desired rank for the low-rank approximation.
      site_name: should be one of the following:
         - "multi_scale_spectral_coarse_real"
         - "multi_scale_spectral_fine_real"
         - "multi_scale_spectral_coarse_imag"
         - "multi_scale_spectral_fine_imag"
    """
    def __init__(self, model: Callable, K: int, rank: int = 2, site_name: str = "multi_scale_spectral_coarse_real"):
        super().__init__(model)
        self.K = K
        self.rank = rank
        self.site_name = site_name

    def __call__(self, *args, **kwargs):
        mean = numpyro.param(f"{self.site_name}_mean", jnp.zeros((self.K,)))
        V_init = jnp.zeros((self.K, self.rank))
        V = numpyro.param(f"{self.site_name}_V", V_init)
        diag_init = -3.0 * jnp.ones((self.K,))
        log_diag = numpyro.param(f"{self.site_name}_log_diag", diag_init)

        cov = (V @ V.T) + jnp.diag(jnp.exp(log_diag))
        z = numpyro.sample(self.site_name, dist.MultivariateNormal(loc=mean, covariance_matrix=cov))
        return {self.site_name: z}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        mean = params[f"{self.site_name}_mean"]
        V = params[f"{self.site_name}_V"]
        log_diag = params[f"{self.site_name}_log_diag"]
        cov = (V @ V.T) + jnp.diag(jnp.exp(log_diag))
        samples = dist.MultivariateNormal(mean, cov).sample(rng_key, sample_shape)
        return {self.site_name: samples}
    

class BayesianConv2DLowRankGuide(AutoGuide):
    """
    A low-rank covariance guide for the convolutional layer's weights plus
    a diagonal guide for the bias.

    Assumes the model sample sites for the convolutional layer look like:
    
        W = numpyro.sample(
            f"{name}_W",
            dist.Normal(0, prior_std).expand(kernel_shape).to_event(len(kernel_shape))
        )
        b = numpyro.sample(
            f"{name}_b",
            dist.Normal(0, prior_std).expand([out_channels]).to_event(1)
        )

    We replace that with:
      W ~ MultivariateNormal(w_mean, cov_w)
      b ~ Normal(b_mean, exp(b_log_std))
    where W is flattened internally to shape (num_w,).
    """
    def __init__(self, model, kernel_shape, out_channels, rank=2, name="bayes_conv2d"):
        """
        :param model: The model function that contains the sample sites.
        :param kernel_shape: Tuple of (kernel_h, kernel_w, in_channels, out_channels).
        :param out_channels: The number of output channels (for b).
        :param rank: The rank of the low-rank approximation for W.
        :param name: Common prefix for the W and b sample sites.
        """
        super().__init__(model)
        self.kernel_shape = kernel_shape
        self.out_channels = out_channels
        self.rank = rank
        self.name = name
        
        # Flatten the kernel for weights.
        self.num_w = np.prod(kernel_shape)
        self.num_b = out_channels

    def __call__(self, *args, **kwargs):
        # -------------------------------
        # Guide for W (flattened)
        # -------------------------------
        w_mean = numpyro.param(
            f"{self.name}_W_mean",
            jnp.zeros((self.num_w,))
        )
        # Low-rank factor V of shape (num_w, rank)
        w_V = numpyro.param(
            f"{self.name}_W_V",
            jnp.zeros((self.num_w, self.rank))
        )
        # Diagonal term (log scale) for W (vector of length num_w)
        w_log_diag = numpyro.param(
            f"{self.name}_W_log_diag",
            -3.0 * jnp.ones((self.num_w,))
        )
        # Construct covariance: cov_w = V @ V^T + diag(exp(w_log_diag))
        cov_w = (w_V @ w_V.T) + jnp.diag(jnp.exp(w_log_diag))
        
        # Sample the flattened weights from a MultivariateNormal.
        w_flat = numpyro.sample(
            f"{self.name}_W",
            dist.MultivariateNormal(loc=w_mean, covariance_matrix=cov_w)
        )
        
        # -------------------------------
        # Guide for b (bias, diagonal Normal)
        # -------------------------------
        b_mean = numpyro.param(
            f"{self.name}_b_mean",
            jnp.zeros((self.num_b,))
        )
        b_log_std = numpyro.param(
            f"{self.name}_b_log_std",
            -3.0 * jnp.ones((self.num_b,))
        )
        b = numpyro.sample(
            f"{self.name}_b",
            dist.Normal(b_mean, jnp.exp(b_log_std)).to_event(1)
        )
        
        # Reshape flattened weights back to the kernel shape.
        W_reshaped = jnp.reshape(w_flat, self.kernel_shape)
        return {
            f"{self.name}_W": W_reshaped,
            f"{self.name}_b": b
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        """
        Draw posterior samples given the final learned parameters.
        Returns a dictionary with keys corresponding to the sample sites.
        """
        w_mean = params[f"{self.name}_W_mean"]
        w_V = params[f"{self.name}_W_V"]
        w_log_diag = params[f"{self.name}_W_log_diag"]
        b_mean = params[f"{self.name}_b_mean"]
        b_log_std = params[f"{self.name}_b_log_std"]
        
        cov_w = (w_V @ w_V.T) + jnp.diag(jnp.exp(w_log_diag))
        
        # Define the distributions.
        dist_w = dist.MultivariateNormal(w_mean, cov_w)
        dist_b = dist.Normal(b_mean, jnp.exp(b_log_std))
        
        w_samples_flat = dist_w.sample(rng_key, sample_shape=sample_shape)
        b_samples = dist_b.sample(rng_key, sample_shape=sample_shape)
        
        # Reshape w_samples_flat from sample_shape + (num_w,)
        # to sample_shape + kernel_shape.
        w_samples_reshaped = jnp.reshape(w_samples_flat, sample_shape + self.kernel_shape)
        
        return {
            f"{self.name}_W": w_samples_reshaped,
            f"{self.name}_b": b_samples
        }