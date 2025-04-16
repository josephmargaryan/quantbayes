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

class FlowMultiScaleCoarseRealGuide(AutoGuide):
    def __init__(self, model, coarse_K, num_flows=2, name="multi_scale_spectral_coarse_real"):
        self.coarse_K = coarse_K
        self.num_flows = num_flows
        self.name = name
        super().__init__(model)
        
    def __call__(self, *args, **kwargs):
        # Base distribution parameters for coarse real part.
        base_mean = numpyro.param(f"{self.name}_flow_mean", jnp.zeros((self.coarse_K,)))
        base_log_std = numpyro.param(f"{self.name}_flow_log_std", -3.0 * jnp.ones((self.coarse_K,)))
        base_dist = dist.Normal(base_mean, jnp.exp(base_log_std)).to_event(1)
        
        # Build a chain of flow transformations.
        transforms = []
        for i in range(self.num_flows):
            shift = numpyro.param(f"{self.name}_flow_shift_{i}", jnp.zeros((self.coarse_K,)))
            scale = numpyro.param(f"{self.name}_flow_scale_{i}", jnp.ones((self.coarse_K,)),
                                  constraint=constraints.positive)
            transforms.append(AffineTransform(loc=shift, scale=scale))
            
        flow_transform = ComposeTransform(transforms)
        flow_dist = dist.TransformedDistribution(base_dist, flow_transform)
        
        sample = numpyro.sample(self.name, flow_dist)
        return {self.name: sample}
    
    def sample_posterior(self, rng_key, params, sample_shape=()):
        base_mean = params[f"{self.name}_flow_mean"]
        base_log_std = params[f"{self.name}_flow_log_std"]
        base_std = jnp.exp(base_log_std)
        base_dist = dist.Normal(base_mean, base_std).to_event(1)
        
        transforms = []
        for i in range(self.num_flows):
            shift = params[f"{self.name}_flow_shift_{i}"]
            scale = params[f"{self.name}_flow_scale_{i}"]
            transforms.append(AffineTransform(loc=shift, scale=scale))
            
        flow_transform = ComposeTransform(transforms)
        flow_dist = dist.TransformedDistribution(base_dist, flow_transform)
        
        sample = flow_dist.sample(rng_key, sample_shape)
        return {self.name: sample}


class FlowMultiScaleCoarseImagGuide(AutoGuide):
    def __init__(self, model, coarse_K, num_flows=2, name="multi_scale_spectral_coarse_imag"):
        self.coarse_K = coarse_K
        self.num_flows = num_flows
        self.name = name
        super().__init__(model)
        
    def __call__(self, *args, **kwargs):
        base_mean = numpyro.param(f"{self.name}_flow_mean", jnp.zeros((self.coarse_K,)))
        base_log_std = numpyro.param(f"{self.name}_flow_log_std", -3.0 * jnp.ones((self.coarse_K,)))
        base_dist = dist.Normal(base_mean, jnp.exp(base_log_std)).to_event(1)
        
        transforms = []
        for i in range(self.num_flows):
            shift = numpyro.param(f"{self.name}_flow_shift_{i}", jnp.zeros((self.coarse_K,)))
            scale = numpyro.param(f"{self.name}_flow_scale_{i}", jnp.ones((self.coarse_K,)),
                                  constraint=constraints.positive)
            transforms.append(AffineTransform(loc=shift, scale=scale))
            
        flow_transform = ComposeTransform(transforms)
        flow_dist = dist.TransformedDistribution(base_dist, flow_transform)
        
        sample = numpyro.sample(self.name, flow_dist)
        return {self.name: sample}
    
    def sample_posterior(self, rng_key, params, sample_shape=()):
        base_mean = params[f"{self.name}_flow_mean"]
        base_log_std = params[f"{self.name}_flow_log_std"]
        base_std = jnp.exp(base_log_std)
        base_dist = dist.Normal(base_mean, base_std).to_event(1)
        
        transforms = []
        for i in range(self.num_flows):
            shift = params[f"{self.name}_flow_shift_{i}"]
            scale = params[f"{self.name}_flow_scale_{i}"]
            transforms.append(AffineTransform(loc=shift, scale=scale))
            
        flow_transform = ComposeTransform(transforms)
        flow_dist = dist.TransformedDistribution(base_dist, flow_transform)
        
        sample = flow_dist.sample(rng_key, sample_shape)
        return {self.name: sample}

class FlowMultiScaleFineRealGuide(AutoGuide):
    """
    Flow guide for the fine real components of the multi-scale spectral layer.
    Targets the site "multi_scale_spectral_fine_real".
    """
    def __init__(self, model, fine_K, num_flows=2, name="multi_scale_spectral_fine_real"):
        self.fine_K = fine_K
        self.num_flows = num_flows
        self.name = name
        super().__init__(model)
        
    def __call__(self, *args, **kwargs):
        # Base distribution parameters.
        base_mean = numpyro.param(f"{self.name}_flow_mean", jnp.zeros((self.fine_K,)))
        base_log_std = numpyro.param(f"{self.name}_flow_log_std", -3.0 * jnp.ones((self.fine_K,)))
        base_dist = dist.Normal(base_mean, jnp.exp(base_log_std)).to_event(1)
        
        # Build a chain of affine flows.
        transforms = []
        for i in range(self.num_flows):
            shift = numpyro.param(f"{self.name}_flow_shift_{i}", jnp.zeros((self.fine_K,)))
            scale = numpyro.param(f"{self.name}_flow_scale_{i}", jnp.ones((self.fine_K,)),
                                  constraint=constraints.positive)
            transforms.append(AffineTransform(loc=shift, scale=scale))
            
        flow_transform = ComposeTransform(transforms)
        flow_dist = dist.TransformedDistribution(base_dist, flow_transform)
        
        sample = numpyro.sample(self.name, flow_dist)
        return {self.name: sample}
    
    def sample_posterior(self, rng_key, params, sample_shape=()):
        base_mean = params[f"{self.name}_flow_mean"]
        base_log_std = params[f"{self.name}_flow_log_std"]
        base_std = jnp.exp(base_log_std)
        base_dist = dist.Normal(base_mean, base_std).to_event(1)
        
        transforms = []
        for i in range(self.num_flows):
            shift = params[f"{self.name}_flow_shift_{i}"]
            scale = params[f"{self.name}_flow_scale_{i}"]
            transforms.append(AffineTransform(loc=shift, scale=scale))
            
        flow_transform = ComposeTransform(transforms)
        flow_dist = dist.TransformedDistribution(base_dist, flow_transform)
        
        sample = flow_dist.sample(rng_key, sample_shape)
        return {self.name: sample}


class FlowMultiScaleFineImagGuide(AutoGuide):
    """
    Flow guide for the fine imaginary components of the multi-scale spectral layer.
    Targets the site "multi_scale_spectral_fine_imag".
    """
    def __init__(self, model, fine_K, num_flows=2, name="multi_scale_spectral_fine_imag"):
        self.fine_K = fine_K
        self.num_flows = num_flows
        self.name = name
        super().__init__(model)
        
    def __call__(self, *args, **kwargs):
        base_mean = numpyro.param(f"{self.name}_flow_mean", jnp.zeros((self.fine_K,)))
        base_log_std = numpyro.param(f"{self.name}_flow_log_std", -3.0 * jnp.ones((self.fine_K,)))
        base_dist = dist.Normal(base_mean, jnp.exp(base_log_std)).to_event(1)
        
        transforms = []
        for i in range(self.num_flows):
            shift = numpyro.param(f"{self.name}_flow_shift_{i}", jnp.zeros((self.fine_K,)))
            scale = numpyro.param(f"{self.name}_flow_scale_{i}", jnp.ones((self.fine_K,)),
                                  constraint=constraints.positive)
            transforms.append(AffineTransform(loc=shift, scale=scale))
            
        flow_transform = ComposeTransform(transforms)
        flow_dist = dist.TransformedDistribution(base_dist, flow_transform)
        
        sample = numpyro.sample(self.name, flow_dist)
        return {self.name: sample}
    
    def sample_posterior(self, rng_key, params, sample_shape=()):
        base_mean = params[f"{self.name}_flow_mean"]
        base_log_std = params[f"{self.name}_flow_log_std"]
        base_std = jnp.exp(base_log_std)
        base_dist = dist.Normal(base_mean, base_std).to_event(1)
        
        transforms = []
        for i in range(self.num_flows):
            shift = params[f"{self.name}_flow_shift_{i}"]
            scale = params[f"{self.name}_flow_scale_{i}"]
            transforms.append(AffineTransform(loc=shift, scale=scale))
            
        flow_transform = ComposeTransform(transforms)
        flow_dist = dist.TransformedDistribution(base_dist, flow_transform)
        
        sample = flow_dist.sample(rng_key, sample_shape)
        return {self.name: sample}


class MixtureMultiScaleFineRealGuide(AutoGuide):
    def __init__(self, model, fine_K, num_components=3, name="multi_scale_spectral_fine_real"):
        self.fine_K = fine_K
        self.num_components = num_components
        self.name = name
        super().__init__(model)
    
    def __call__(self, *args, **kwargs):
        # Mixture logits for the components
        logits = numpyro.param(f"{self.name}_mixture_logits", jnp.zeros((self.num_components,)))
        weights = jax.nn.softmax(logits)
        
        # Each component gets its own mean and log_std
        means = numpyro.param(f"{self.name}_mixture_means", jnp.zeros((self.num_components, self.fine_K)))
        log_stds = numpyro.param(f"{self.name}_mixture_log_stds", -3.0 * jnp.ones((self.num_components, self.fine_K)))
        
        component_dist = dist.Independent(dist.Normal(means, jnp.exp(log_stds)), 1)
        mixture_dist = dist.MixtureSameFamily(dist.Categorical(probs=weights), component_dist)
        
        sample = numpyro.sample(self.name, mixture_dist)
        return {self.name: sample}
    
    def sample_posterior(self, rng_key, params, sample_shape=()):
        logits = params[f"{self.name}_mixture_logits"]
        weights = jax.nn.softmax(logits)
        means = params[f"{self.name}_mixture_means"]
        log_stds = params[f"{self.name}_mixture_log_stds"]
        
        component_dist = dist.Independent(dist.Normal(means, jnp.exp(log_stds)), 1)
        mixture_dist = dist.MixtureSameFamily(dist.Categorical(probs=weights), component_dist)
        sample = mixture_dist.sample(rng_key, sample_shape)
        return {self.name: sample}


class MixtureMultiScaleFineImagGuide(AutoGuide):
    def __init__(self, model, fine_K, num_components=3, name="multi_scale_spectral_fine_imag"):
        self.fine_K = fine_K
        self.num_components = num_components
        self.name = name
        super().__init__(model)
    
    def __call__(self, *args, **kwargs):
        logits = numpyro.param(f"{self.name}_mixture_logits", jnp.zeros((self.num_components,)))
        weights = jax.nn.softmax(logits)
        
        means = numpyro.param(f"{self.name}_mixture_means", jnp.zeros((self.num_components, self.fine_K)))
        log_stds = numpyro.param(f"{self.name}_mixture_log_stds", -3.0 * jnp.ones((self.num_components, self.fine_K)))
        
        component_dist = dist.Independent(dist.Normal(means, jnp.exp(log_stds)), 1)
        mixture_dist = dist.MixtureSameFamily(dist.Categorical(probs=weights), component_dist)
        
        sample = numpyro.sample(self.name, mixture_dist)
        return {self.name: sample}
    
    def sample_posterior(self, rng_key, params, sample_shape=()):
        logits = params[f"{self.name}_mixture_logits"]
        weights = jax.nn.softmax(logits)
        means = params[f"{self.name}_mixture_means"]
        log_stds = params[f"{self.name}_mixture_log_stds"]
        
        component_dist = dist.Independent(dist.Normal(means, jnp.exp(log_stds)), 1)
        mixture_dist = dist.MixtureSameFamily(dist.Categorical(probs=weights), component_dist)
        sample = mixture_dist.sample(rng_key, sample_shape)
        return {self.name: sample}

class MixtureMultiScaleCoarseRealGuide(AutoGuide):
    """
    Mixture of Gaussians guide for the coarse real components of the multi-scale spectral layer.
    Targets the site "multi_scale_spectral_coarse_real".
    """
    def __init__(self, model, coarse_K, num_components=3, name="multi_scale_spectral_coarse_real"):
        self.coarse_K = coarse_K
        self.num_components = num_components
        self.name = name
        super().__init__(model)
    
    def __call__(self, *args, **kwargs):
        # Mixture logits for the components.
        logits = numpyro.param(f"{self.name}_mixture_logits", jnp.zeros((self.num_components,)))
        weights = jax.nn.softmax(logits)
        
        # Mean and log_std for each Gaussian component.
        means = numpyro.param(f"{self.name}_mixture_means", jnp.zeros((self.num_components, self.coarse_K)))
        log_stds = numpyro.param(f"{self.name}_mixture_log_stds", -3.0 * jnp.ones((self.num_components, self.coarse_K)))
        
        component_dist = dist.Independent(dist.Normal(means, jnp.exp(log_stds)), 1)
        mixture_dist = dist.MixtureSameFamily(dist.Categorical(probs=weights), component_dist)
        
        sample = numpyro.sample(self.name, mixture_dist)
        return {self.name: sample}
    
    def sample_posterior(self, rng_key, params, sample_shape=()):
        logits = params[f"{self.name}_mixture_logits"]
        weights = jax.nn.softmax(logits)
        means = params[f"{self.name}_mixture_means"]
        log_stds = params[f"{self.name}_mixture_log_stds"]
        
        component_dist = dist.Independent(dist.Normal(means, jnp.exp(log_stds)), 1)
        mixture_dist = dist.MixtureSameFamily(dist.Categorical(probs=weights), component_dist)
        
        sample = mixture_dist.sample(rng_key, sample_shape)
        return {self.name: sample}


class MixtureMultiScaleCoarseImagGuide(AutoGuide):
    """
    Mixture of Gaussians guide for the coarse imaginary components of the multi-scale spectral layer.
    Targets the site "multi_scale_spectral_coarse_imag".
    """
    def __init__(self, model, coarse_K, num_components=3, name="multi_scale_spectral_coarse_imag"):
        self.coarse_K = coarse_K
        self.num_components = num_components
        self.name = name
        super().__init__(model)
    
    def __call__(self, *args, **kwargs):
        logits = numpyro.param(f"{self.name}_mixture_logits", jnp.zeros((self.num_components,)))
        weights = jax.nn.softmax(logits)
        
        means = numpyro.param(f"{self.name}_mixture_means", jnp.zeros((self.num_components, self.coarse_K)))
        log_stds = numpyro.param(f"{self.name}_mixture_log_stds", -3.0 * jnp.ones((self.num_components, self.coarse_K)))
        
        component_dist = dist.Independent(dist.Normal(means, jnp.exp(log_stds)), 1)
        mixture_dist = dist.MixtureSameFamily(dist.Categorical(probs=weights), component_dist)
        
        sample = numpyro.sample(self.name, mixture_dist)
        return {self.name: sample}
    
    def sample_posterior(self, rng_key, params, sample_shape=()):
        logits = params[f"{self.name}_mixture_logits"]
        weights = jax.nn.softmax(logits)
        means = params[f"{self.name}_mixture_means"]
        log_stds = params[f"{self.name}_mixture_log_stds"]
        
        component_dist = dist.Independent(dist.Normal(means, jnp.exp(log_stds)), 1)
        mixture_dist = dist.MixtureSameFamily(dist.Categorical(probs=weights), component_dist)
        
        sample = mixture_dist.sample(rng_key, sample_shape)
        return {self.name: sample}



class LowRankMultiScaleRealGuide(AutoGuide):
    """
    Low-rank covariance guide for the multi-scale real Fourier coefficients.
    
    This guide jointly approximates both the coarse and fine real parts.
    It defines a low-rank factorization for the joint covariance over a vector of length
    (coarse_K + fine_K) and then splits the resulting vector into coarse and fine components.
    
    Args:
        model: The model function.
        coarse_K: Number of coarse frequencies.
        fine_K: Number of fine frequencies.
        rank: The rank for the low-rank approximation.
        name: Base name for the guide. The resulting sample sites are expected to be
              "multi_scale_spectral_coarse_real" and "multi_scale_spectral_fine_real".
    """
    def __init__(self, model: Callable, coarse_K: int, fine_K: int, rank: int = 2,
                 name: str = "multi_scale_spectral_real"):
        super().__init__(model)
        self.coarse_K = coarse_K
        self.fine_K = fine_K
        self.total_K = coarse_K + fine_K
        self.rank = rank
        self.name = name

    def __call__(self, *args, **kwargs):
        # Mean vector for the joint real coefficients
        mean = numpyro.param(f"{self.name}_mean", jnp.zeros((self.total_K,)))
        # Low-rank factor: V with shape (total_K, rank)
        V_init = jnp.zeros((self.total_K, self.rank))
        V = numpyro.param(f"{self.name}_V", V_init)
        # Log diagonal entries for the diagonal factor of the covariance.
        diag_init = -3.0 * jnp.ones((self.total_K,))
        log_diag = numpyro.param(f"{self.name}_log_diag", diag_init)
        
        # Reconstruct covariance: Cov = V @ V^T + diag(exp(log_diag))
        cov = (V @ V.T) + jnp.diag(jnp.exp(log_diag))
        
        # Sample joint real vector
        z = numpyro.sample(
            self.name,
            dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
        )
        
        # Split into coarse and fine parts.
        coarse_sample = z[:self.coarse_K]
        fine_sample = z[self.coarse_K:]
        return {
            "multi_scale_spectral_coarse_real": coarse_sample,
            "multi_scale_spectral_fine_real": fine_sample
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        mean = params[f"{self.name}_mean"]
        V = params[f"{self.name}_V"]
        log_diag = params[f"{self.name}_log_diag"]
        cov = (V @ V.T) + jnp.diag(jnp.exp(log_diag))
        dist_mv = dist.MultivariateNormal(mean, cov)
        z = dist_mv.sample(rng_key, sample_shape=sample_shape)
        coarse_sample = z[..., :self.coarse_K]  # support batch sampling
        fine_sample = z[..., self.coarse_K:]
        return {
            "multi_scale_spectral_coarse_real": coarse_sample,
            "multi_scale_spectral_fine_real": fine_sample
        }


class LowRankMultiScaleImagGuide(AutoGuide):
    """
    Low-rank covariance guide for the multi-scale imaginary Fourier coefficients.
    
    This guide jointly approximates both the coarse and fine imaginary parts.
    It uses the same idea as the real guide: a joint low-rank parameterization
    and splitting the sample into coarse and fine components.
    
    Args:
        model: The model function.
        coarse_K: Number of coarse frequencies.
        fine_K: Number of fine frequencies.
        rank: The rank for the low-rank approximation.
        name: Base name for the guide. The resulting sample sites are expected to be
              "multi_scale_spectral_coarse_imag" and "multi_scale_spectral_fine_imag".
    """
    def __init__(self, model: Callable, coarse_K: int, fine_K: int, rank: int = 2,
                 name: str = "multi_scale_spectral_imag"):
        super().__init__(model)
        self.coarse_K = coarse_K
        self.fine_K = fine_K
        self.total_K = coarse_K + fine_K
        self.rank = rank
        self.name = name

    def __call__(self, *args, **kwargs):
        mean = numpyro.param(f"{self.name}_mean", jnp.zeros((self.total_K,)))
        V_init = jnp.zeros((self.total_K, self.rank))
        V = numpyro.param(f"{self.name}_V", V_init)
        diag_init = -3.0 * jnp.ones((self.total_K,))
        log_diag = numpyro.param(f"{self.name}_log_diag", diag_init)
        
        cov = (V @ V.T) + jnp.diag(jnp.exp(log_diag))
        
        z = numpyro.sample(
            self.name,
            dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
        )
        
        # Split the joint sample into coarse and fine imaginary parts
        coarse_sample = z[:self.coarse_K]
        fine_sample = z[self.coarse_K:]
        return {
            "multi_scale_spectral_coarse_imag": coarse_sample,
            "multi_scale_spectral_fine_imag": fine_sample
        }

    def sample_posterior(self, rng_key, params, sample_shape=()):
        mean = params[f"{self.name}_mean"]
        V = params[f"{self.name}_V"]
        log_diag = params[f"{self.name}_log_diag"]
        cov = (V @ V.T) + jnp.diag(jnp.exp(log_diag))
        dist_mv = dist.MultivariateNormal(mean, cov)
        z = dist_mv.sample(rng_key, sample_shape=sample_shape)
        coarse_sample = z[..., :self.coarse_K]
        fine_sample = z[..., self.coarse_K:]
        return {
            "multi_scale_spectral_coarse_imag": coarse_sample,
            "multi_scale_spectral_fine_imag": fine_sample
        }

    

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