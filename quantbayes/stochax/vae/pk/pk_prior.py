# quantbayes/stochax/vae/pk/pk_prior.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from .features import FeatureMap
from .score_model import LatentScoreNet


@dataclass(frozen=True)
class PKPriorConfig:
    lambda_strength: float = 1.0
    sigma_max: Optional[float] = None  # apply correction only when sigma <= sigma_max

    # sigma-weight dial: w(sigma) = (sigma_ref^2 / (sigma^2 + sigma_ref^2))^gamma
    sigma_ref: float = 1.0
    sigma_weight_gamma: float = 0.0

    max_correction_norm: float = 50.0
    eps: float = 1e-12


class PKLatentPrior(eqx.Module):
    """
    PK-updated latent prior score:
      score(z) = score_base(z) + lambda * gate * w(sigma) * J_F(z)^T (s_p(u) - s_pi(u))
    where base prior is N(0,I) => score_base(z) = -z.
    """

    feature_map: FeatureMap
    evidence_score: LatentScoreNet
    reference_score: Optional[LatentScoreNet] = None
    cfg: PKPriorConfig = eqx.field(static=True)

    def __init__(
        self,
        *,
        feature_map: FeatureMap,
        evidence_score: LatentScoreNet,
        reference_score: Optional[LatentScoreNet],
        cfg: PKPriorConfig,
    ):
        self.feature_map = feature_map
        self.evidence_score = evidence_score
        self.reference_score = reference_score
        self.cfg = cfg

    def _as_batch(self, z: jnp.ndarray):
        z = jnp.asarray(z)
        if z.ndim == 1:
            return z[None, :], True
        return z, False

    def __call__(self, log_sigma: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        z, squeeze = self._as_batch(z)
        b = z.shape[0]

        sigma = jnp.exp(jnp.asarray(log_sigma).reshape(()))

        # base prior score for N(0,I)
        score_base = -z

        u = self.feature_map(z)  # (B,M)

        # evidence score s_p(u | sigma)
        s_p = self.evidence_score(jnp.full((b,), jnp.log(sigma)), u)

        # reference score s_pi(u | sigma)
        s_pi = None
        if self.reference_score is not None:
            s_pi = self.reference_score(jnp.full((b,), jnp.log(sigma)), u)
        else:
            s_pi = self.feature_map.prior_score_u(u)
            if s_pi is None:
                raise ValueError(
                    "No reference_score provided and feature_map has no analytic prior_score_u(u). "
                    "Provide reference_score or use Identity/Linear feature map."
                )

        delta_u = s_p - s_pi  # (B,M)
        delta_z = self.feature_map.vjp(z, delta_u)  # (B,D)

        # per-sample clip
        n = jnp.sqrt(jnp.sum(delta_z * delta_z, axis=-1, keepdims=True) + self.cfg.eps)
        clip = jnp.minimum(1.0, self.cfg.max_correction_norm / n)
        delta_z = delta_z * clip

        # gate
        gate = 1.0
        if self.cfg.sigma_max is not None:
            gate = (sigma <= float(self.cfg.sigma_max)).astype(z.dtype)

        # sigma-weight
        gamma = float(self.cfg.sigma_weight_gamma)
        if gamma != 0.0:
            sr = float(self.cfg.sigma_ref)
            base = (sr * sr) / (sigma * sigma + sr * sr)
            w_sigma = jnp.power(base, gamma).astype(z.dtype)
        else:
            w_sigma = jnp.asarray(1.0, dtype=z.dtype)

        total = (
            score_base + (float(self.cfg.lambda_strength) * gate * w_sigma) * delta_z
        )

        if squeeze:
            return total[0]
        return total
