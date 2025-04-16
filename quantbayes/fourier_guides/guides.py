from typing import Callable, Tuple
import jax
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoGuide
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform, ComposeTransform

__all__ = [
    "LowRankSpectralGuide",
    "FlowSpectralGuide",
    "FlowMultiScaleCoarseRealGuide",
    "FlowMultiScaleCoarseImagGuide",
    "LowRankMultiScaleRealGuide",
    "LowRankMultiScaleImagGuide",
    "LowRankSpectral2DGuide",
    "LowRankSpectral1DGuide",
]


# -----------------------------------------------------------------------------#
#  Utility: numerically stable low‑rank‑plus‑diagonal MVN draw
# -----------------------------------------------------------------------------#
def _build_lowrank_mvn(name: str, dim: int, rank: int, jitter: float = 1e-5):
    """
    Draw z ~ 𝒩(loc, V Vᵀ + diag(d)) while registering the variational parameters.

    loc        : free mean     (shape [dim])
    V          : low‑rank mat  (shape [dim, rank])
    d_raw      : unconstrained diagonal parameters
    d          : softplus(d_raw) + jitter  ➜ strictly positive
    """
    loc = numpyro.param(f"{name}_loc", jnp.zeros(dim))
    V = numpyro.param(f"{name}_V", jnp.zeros((dim, rank)))
    d_raw = numpyro.param(f"{name}_d_raw", -3.0 * jnp.ones(dim))

    diag = jax.nn.softplus(d_raw) + jitter
    cov = V @ V.T + jnp.diag(diag)

    return numpyro.sample(name, dist.MultivariateNormal(loc, covariance_matrix=cov))


# -----------------------------------------------------------------------------#
#  Low‑rank Gaussian guide
# -----------------------------------------------------------------------------#
class LowRankSpectralGuide(AutoGuide):
    """
    Low‑rank multivariate Gaussian for *one* spectral site.

    Parameters
    ----------
    model      : underlying NumPyro model
    K          : number of active frequencies (dimensionality)
    rank       : rank of the low‑rank factor (≈ 2–8 is typical)
    site_name  : \"spectral_circ_jvp_real\"  *or*  \"spectral_circ_jvp_imag\"
    """

    def __init__(self, model: Callable, K: int, rank: int, site_name: str):
        super().__init__(model)
        self.K = K
        self.rank = rank
        self.site = site_name

    def __call__(self, *args, **kwargs):
        z = _build_lowrank_mvn(self.site, self.K, self.rank)
        return {self.site: z}

    # convenient posterior sampler for diagnostics / visualisation
    def sample_posterior(self, rng_key, params, sample_shape=()):
        loc = params[f"{self.site}_loc"]
        V = params[f"{self.site}_V"]
        d_raw = params[f"{self.site}_d_raw"]
        diag = jax.nn.softplus(d_raw) + 1e-5
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)
        return {self.site: z}


# -----------------------------------------------------------------------------#
#  Affine‑flow guide (location‑scale chain) for skew / heavy tails
# -----------------------------------------------------------------------------#
class FlowSpectralGuide(AutoGuide):
    """
    Shallow affine flow (scale‑shift) guide for one spectral site.

    Parameters
    ----------
    model     : underlying NumPyro model
    K         : dimensionality of the site
    site_name : \"spectral_circ_jvp_real\"  *or*  \"spectral_circ_jvp_imag\"
    num_flows : length of the scale‑shift chain (default 2)
    """

    def __init__(self, model: Callable, K: int, site_name: str, num_flows: int = 2):
        super().__init__(model)
        self.K = K
        self.site = site_name
        self.num_flows = num_flows

    # ---------------------------------------------------------------- guide call
    def __call__(self, *args, **kwargs):
        # base diagonal Gaussian
        mu = numpyro.param(f"{self.site}_flow_mu", jnp.zeros(self.K))
        log = numpyro.param(f"{self.site}_flow_log_sigma", -3.0 * jnp.ones(self.K))
        base = dist.Normal(mu, jnp.exp(log)).to_event(1)

        # chain of affine transforms
        transforms = []
        for i in range(self.num_flows):
            shift = numpyro.param(f"{self.site}_flow_shift_{i}", jnp.zeros(self.K))
            scale = numpyro.param(
                f"{self.site}_flow_scale_{i}",
                jnp.ones(self.K),
                constraint=constraints.positive,
            )
            transforms.append(AffineTransform(loc=shift, scale=scale))

        flow = dist.TransformedDistribution(base, ComposeTransform(transforms))
        z = numpyro.sample(self.site, flow)
        return {self.site: z}

    # ------------------------------------------------ posterior sampling helper
    def sample_posterior(self, rng_key, params, sample_shape=()):
        mu = params[f"{self.site}_flow_mu"]
        log = params[f"{self.site}_flow_log_sigma"]
        base = dist.Normal(mu, jnp.exp(log)).to_event(1)

        transforms = []
        for i in range(self.num_flows):
            shift = params[f"{self.site}_flow_shift_{i}"]
            scale = params[f"{self.site}_flow_scale_{i}"]
            transforms.append(AffineTransform(loc=shift, scale=scale))

        flow = dist.TransformedDistribution(base, ComposeTransform(transforms))
        return {self.site: flow.sample(rng_key, sample_shape)}


class FlowMultiScaleCoarseRealGuide(AutoGuide):
    def __init__(
        self, model, coarse_K, num_flows=2, name="multi_scale_spectral_coarse_real"
    ):
        self.coarse_K = coarse_K
        self.num_flows = num_flows
        self.name = name
        super().__init__(model)

    def __call__(self, *args, **kwargs):
        # Base distribution parameters for coarse real part.
        base_mean = numpyro.param(f"{self.name}_flow_mean", jnp.zeros((self.coarse_K,)))
        base_log_std = numpyro.param(
            f"{self.name}_flow_log_std", -3.0 * jnp.ones((self.coarse_K,))
        )
        base_dist = dist.Normal(base_mean, jnp.exp(base_log_std)).to_event(1)

        # Build a chain of flow transformations.
        transforms = []
        for i in range(self.num_flows):
            shift = numpyro.param(
                f"{self.name}_flow_shift_{i}", jnp.zeros((self.coarse_K,))
            )
            scale = numpyro.param(
                f"{self.name}_flow_scale_{i}",
                jnp.ones((self.coarse_K,)),
                constraint=constraints.positive,
            )
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
    def __init__(
        self, model, coarse_K, num_flows=2, name="multi_scale_spectral_coarse_imag"
    ):
        self.coarse_K = coarse_K
        self.num_flows = num_flows
        self.name = name
        super().__init__(model)

    def __call__(self, *args, **kwargs):
        base_mean = numpyro.param(f"{self.name}_flow_mean", jnp.zeros((self.coarse_K,)))
        base_log_std = numpyro.param(
            f"{self.name}_flow_log_std", -3.0 * jnp.ones((self.coarse_K,))
        )
        base_dist = dist.Normal(base_mean, jnp.exp(base_log_std)).to_event(1)

        transforms = []
        for i in range(self.num_flows):
            shift = numpyro.param(
                f"{self.name}_flow_shift_{i}", jnp.zeros((self.coarse_K,))
            )
            scale = numpyro.param(
                f"{self.name}_flow_scale_{i}",
                jnp.ones((self.coarse_K,)),
                constraint=constraints.positive,
            )
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


def _build_lowrank_mvn(
    name: str, dim: int, rank: int, jitter: float = 1e-5
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Utility that returns a single draw `z ~ N(loc,  VVᵀ + diag(d))` where

        • loc      : free location parameter  (shape: [dim])
        • V        : free low‑rank factor      (shape: [dim, rank])
        • d_raw    : free unconstrained diag   (shape: [dim])
        • d        : softplus(d_raw) + jitter  (guaranteed positive)

    All parameters are registered with `numpyro.param`.
    """
    loc = numpyro.param(f"{name}_loc", jnp.zeros(dim))
    V = numpyro.param(f"{name}_V", jnp.zeros((dim, rank)))
    draw = numpyro.param(f"{name}_d_raw", -3.0 * jnp.ones(dim))

    # positive diagonal (softplus keeps gradients smooth at 0)
    diag = jax.nn.softplus(draw) + jitter
    cov = V @ V.T + jnp.diag(diag)

    return numpyro.sample(
        name,
        dist.MultivariateNormal(loc, covariance_matrix=cov),
        infer={"is_auxiliary": True},  # ← keep this site out of the model
    )


class LowRankMultiScaleRealGuide(AutoGuide):
    """
    Joint low‑rank guide for coarse & fine **real** Fourier coefficients.

    Model sites covered
    -------------------
    • 'multi_scale_spectral_coarse_real'
    • 'multi_scale_spectral_fine_real'
    """

    def __init__(
        self,
        model: Callable,
        coarse_K: int,
        fine_K: int,
        rank: int = 4,  # a sensible default
        name_joint: str = "mss_real_joint",
    ):
        super().__init__(model)
        self.coarse_K, self.fine_K = coarse_K, fine_K
        self.total_K, self.rank = coarse_K + fine_K, rank
        self.name_joint = name_joint

        # names that *also* exist inside the model
        self.name_coarse = "multi_scale_spectral_coarse_real"
        self.name_fine = "multi_scale_spectral_fine_real"

    # -------------------------------------------------------------------- guide
    def __call__(self, *args, **kwargs):
        z = _build_lowrank_mvn(self.name_joint, self.total_K, self.rank)

        # split the joint draw into the two model sites
        coarse, fine = z[..., : self.coarse_K], z[..., self.coarse_K :]

        numpyro.sample(self.name_coarse, dist.Delta(coarse).to_event(1))
        numpyro.sample(self.name_fine, dist.Delta(fine).to_event(1))

        # (return value is ignored by SVI)
        return {self.name_coarse: coarse, self.name_fine: fine}

    # -------------------------------------------------------- posterior sampler
    def sample_posterior(self, rng_key, params, sample_shape=()):
        loc = params[f"{self.name_joint}_loc"]
        V = params[f"{self.name_joint}_V"]
        d_raw = params[f"{self.name_joint}_d_raw"]
        diag = jax.nn.softplus(d_raw) + 1e-5
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)

        return {
            self.name_coarse: z[..., : self.coarse_K],
            self.name_fine: z[..., self.coarse_K :],
        }


class LowRankMultiScaleImagGuide(AutoGuide):
    """
    Joint low‑rank guide for coarse & fine **imaginary** Fourier coefficients.

    Model sites covered
    -------------------
    • 'multi_scale_spectral_coarse_imag'
    • 'multi_scale_spectral_fine_imag'
    """

    def __init__(
        self,
        model: Callable,
        coarse_K: int,
        fine_K: int,
        rank: int = 4,
        name_joint: str = "mss_imag_joint",
    ):
        super().__init__(model)
        self.coarse_K, self.fine_K = coarse_K, fine_K
        self.total_K, self.rank = coarse_K + fine_K, rank
        self.name_joint = name_joint

        self.name_coarse = "multi_scale_spectral_coarse_imag"
        self.name_fine = "multi_scale_spectral_fine_imag"

    def __call__(self, *args, **kwargs):
        z = _build_lowrank_mvn(self.name_joint, self.total_K, self.rank)

        coarse, fine = z[..., : self.coarse_K], z[..., self.coarse_K :]

        numpyro.sample(self.name_coarse, dist.Delta(coarse).to_event(1))
        numpyro.sample(self.name_fine, dist.Delta(fine).to_event(1))

        return {self.name_coarse: coarse, self.name_fine: fine}

    def sample_posterior(self, rng_key, params, sample_shape=()):
        loc = params[f"{self.name_joint}_loc"]
        V = params[f"{self.name_joint}_V"]
        d_raw = params[f"{self.name_joint}_d_raw"]
        diag = jax.nn.softplus(d_raw) + 1e-5
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)

        return {
            self.name_coarse: z[..., : self.coarse_K],
            self.name_fine: z[..., self.coarse_K :],
        }


# -----------------------------------------------------------------------------#
#  Shared helper: low‑rank‑plus‑diag MVN draw with auxiliary flag
# -----------------------------------------------------------------------------#
def _lowrank_aux(name: str, dim: int, rank: int, jitter: float = 1e-5):
    """Return a draw z and register (loc, V, diag) variational parameters."""
    loc = numpyro.param(f"{name}_loc", jnp.zeros(dim))
    V = numpyro.param(f"{name}_V", jnp.zeros((dim, rank)))
    d_raw = numpyro.param(f"{name}_d_raw", -3.0 * jnp.ones(dim))

    diag = jax.nn.softplus(d_raw) + jitter
    cov = V @ V.T + jnp.diag(diag)

    return numpyro.sample(
        name,
        dist.MultivariateNormal(loc, covariance_matrix=cov),
        infer={"is_auxiliary": True},  # ← keep this site out of the model
    )


# =============================================================================
#  1‑D  :  LowRankSpectral1DGuide
# =============================================================================
class LowRankSpectral1DGuide(AutoGuide):
    """
    Joint low‑rank Gaussian over *all* complex Fourier coeffs of a 1‑D kernel.

    ── covers the two model sites ───────────────────────────────────────────────
    • f\"{name}_real\"   – real part  (length K)
    • f\"{name}_imag\"   – imag part  (length K)
    """

    def __init__(
        self, model: Callable, K: int, rank: int = 4, name: str = "spectral_conv1d"
    ):
        super().__init__(model)
        self.K = K
        self.rank = rank
        self.name = name
        self.joint = f"{name}_joint"  # auxiliary site

    # ------------------------------------------------------------------ guide
    def __call__(self, *args, **kwargs):
        z = _lowrank_aux(self.joint, dim=2 * self.K, rank=self.rank)  # [real | imag]
        real, imag = jnp.split(z, 2, axis=-1)

        # DC is strictly real
        imag = imag.at[..., 0].set(0.0)

        numpyro.sample(f"{self.name}_real", dist.Delta(real).to_event(1))
        numpyro.sample(f"{self.name}_imag", dist.Delta(imag).to_event(1))

        return {f"{self.name}_real": real, f"{self.name}_imag": imag}

    # ------------------------------------------------ posterior sampler helper
    def sample_posterior(self, rng_key, params, sample_shape=()):
        loc = params[f"{self.joint}_loc"]
        V = params[f"{self.joint}_V"]
        d_raw = params[f"{self.joint}_d_raw"]
        diag = jax.nn.softplus(d_raw) + 1e-5
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)  # shape (..., 2K)
        real, imag = jnp.split(z, 2, axis=-1)
        imag = imag.at[..., 0].set(0.0)
        return {f"{self.name}_real": real, f"{self.name}_imag": imag}


# =============================================================================
#  2‑D  :  LowRankSpectral2DGuide
# =============================================================================
class LowRankSpectral2DGuide(AutoGuide):
    """
    Joint low‑rank Gaussian over *all unique* complex Fourier coeffs of a 2‑D
    kernel (half‑plane + DC/Nyquist lines).

    Parameters
    ----------
    H_pad, W_pad : padded kernel height & width (must match layer)
    rank         : low‑rank factor size
    name         : layer's base name (defaults to 'spectral_conv2d')
    """

    def __init__(
        self,
        model: Callable,
        H_pad: int,
        W_pad: int,
        rank: int = 8,
        name: str = "spectral_conv2d",
    ):
        super().__init__(model)
        self.H_pad, self.W_pad = H_pad, W_pad
        # number of unique frequencies in the upper half‑plane including DC row
        self.M = (H_pad * W_pad) // 2 + 1
        self.rank = rank
        self.name = name
        self.joint = f"{name}_joint"

    # ------------------------------------------------------------------ guide
    def __call__(self, *args, **kwargs):
        z = _lowrank_aux(self.joint, dim=2 * self.M, rank=self.rank)  # [real|imag]
        real_half, imag_half = jnp.split(z, 2, axis=-1)

        # enforce purely real DC (0,0)
        imag_half = imag_half.at[..., 0].set(0.0)

        numpyro.sample(f"{self.name}_real", dist.Delta(real_half).to_event(1))
        numpyro.sample(f"{self.name}_imag", dist.Delta(imag_half).to_event(1))

        return {f"{self.name}_real": real_half, f"{self.name}_imag": imag_half}

    # ------------------------------------------------ posterior sampler helper
    def sample_posterior(self, rng_key, params, sample_shape=()):
        loc = params[f"{self.joint}_loc"]
        V = params[f"{self.joint}_V"]
        d_raw = params[f"{self.joint}_d_raw"]
        diag = jax.nn.softplus(d_raw) + 1e-5
        cov = V @ V.T + jnp.diag(diag)

        mvn = dist.MultivariateNormal(loc, covariance_matrix=cov)
        z = mvn.sample(rng_key, sample_shape)  # shape (..., 2M)
        real_half, imag_half = jnp.split(z, 2, axis=-1)
        imag_half = imag_half.at[..., 0].set(0.0)
        return {f"{self.name}_real": real_half, f"{self.name}_imag": imag_half}
