# quantbayes/pkstruct/toy/vrw_components.py
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr

from ..typing import Array, PRNGKey
from ..core.pk import PKPosterior
from ..core.protocols import PKComponents, Prior, LogDensity
from ..utils.stats import log_vonmises, log_scaled_beta_pdf
from .vrw import vrw_r, stephens_logpdf_r, StephensConfig


@dataclass(frozen=True)
class VRWPKConfig:
    """
    Configuration matching the paper Section 5.
    """

    N: int = 5
    mu: float = 0.0
    kappa: float = 10.0
    alpha: float = 10.0
    beta: float = 10.0


@dataclass(frozen=True)
class VonMisesIIDPrior(Prior):
    """
    π(θ) = ∏_i vM(θ_i | μ, κ)
    """

    N: int
    mu: float
    kappa: float

    def sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        key = (
            jr.PRNGKey(int(jax.random.randint(key, (), 0, 2**31 - 1)))
            if not isinstance(key, jax.Array)
            else key
        )
        # Sample by inversion via numpyro? We stay JAX-only: use jax.random.von_mises if available.
        # JAX provides random.von_mises in recent versions. If yours doesn't, replace with numpyro sampling in user code.
        if hasattr(jr, "von_mises"):
            return jr.von_mises(
                key,
                loc=self.mu,
                concentration=self.kappa,
                shape=sample_shape + (self.N,),
            )
        # Fallback: raise a clear error
        raise RuntimeError(
            "jax.random.von_mises not available. Upgrade JAX or use NumPyro sampler backend."
        )

    def log_prob(self, theta: Array) -> Array:
        theta = jnp.asarray(theta)
        return jnp.sum(log_vonmises(theta, self.mu, self.kappa))


@dataclass(frozen=True)
class ScaledBetaEvidence(LogDensity):
    """
    p(r) where r in (0,N), induced by Beta(alpha,beta) on r/N.
    """

    alpha: float
    beta: float
    N: int

    def log_prob(self, r: Array) -> Array:
        return log_scaled_beta_pdf(r, self.alpha, self.beta, self.N)


@dataclass(frozen=True)
class StephensReference(LogDensity):
    """
    π(r) via Stephens approximation induced by the VRW prior.
    """

    cfg: StephensConfig

    def log_prob(self, r: Array) -> Array:
        return stephens_logpdf_r(r, cfg=self.cfg)


def build_vrw_pk_posterior(cfg: VRWPKConfig) -> PKPosterior:
    """
    Builds the VRW PK posterior corresponding to paper Eq. (31):

      p(θ) ∝ π(θ) * ScaledBeta(r(θ)) / Stephens(r(θ))
    """
    prior = VonMisesIIDPrior(N=cfg.N, mu=cfg.mu, kappa=cfg.kappa)
    evidence = ScaledBetaEvidence(alpha=cfg.alpha, beta=cfg.beta, N=cfg.N)
    reference = StephensReference(cfg=StephensConfig(kappa=cfg.kappa, N=cfg.N))

    comps = PKComponents(
        prior=prior,
        coarse_map=vrw_r,
        evidence=evidence,
        reference=reference,
    )
    return PKPosterior(comps)
