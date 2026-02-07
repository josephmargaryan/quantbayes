# quantbayes/stochax/vae/pk/pk_prior.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax.numpy as jnp

from .features import FeatureMap
from .score_model import LatentScoreNet


@dataclass(frozen=True)
class PKPriorConfig:
    lambda_strength: float = 1.0

    # NEW: band-pass gate for correction
    sigma_max: Optional[float] = None
    sigma_min: Optional[float] = None

    # sigma weighting
    sigma_ref: float = 1.0
    sigma_weight_gamma: float = 0.0
    sigma_weight_mode: str = "edm"  # "edm" | "high" | "flat"

    max_correction_norm: float = 50.0
    eps: float = 1e-12


class PKLatentPrior(eqx.Module):
    """
    PK-updated latent prior score:
      score(z) = score_base(z) + λ * gate(σ) * w(σ) * J_F(z)^T [ s_p(u) - s_pi(u) ]

    Base prior: z~N(0,I) => score_base(z) = -z.

    evidence_score: DSM score net for u=F(z) under evidence distribution (aggregate posterior)
    reference_score:
        - if provided: DSM score net for u under pushforward of N(0,I)
        - else: requires feature_map.prior_score_u(u) analytic (Identity/Linear)
    """

    feature_map: FeatureMap
    evidence_score: LatentScoreNet
    reference_score: Optional[LatentScoreNet] = None
    cfg: PKPriorConfig = eqx.field(static=True)

    def _as_batch(self, z: jnp.ndarray):
        z = jnp.asarray(z)
        if z.ndim == 1:
            return z[None, :], True
        return z, False

    def _gate(self, sigma: jnp.ndarray, dtype) -> jnp.ndarray:
        gate = jnp.asarray(1.0, dtype=dtype)
        if self.cfg.sigma_max is not None:
            gate = gate * (sigma <= float(self.cfg.sigma_max)).astype(dtype)
        if self.cfg.sigma_min is not None:
            gate = gate * (sigma >= float(self.cfg.sigma_min)).astype(dtype)
        return gate

    def _w_sigma(self, sigma: jnp.ndarray, dtype) -> jnp.ndarray:
        mode = (self.cfg.sigma_weight_mode or "edm").lower().strip()
        if mode == "flat" or float(self.cfg.sigma_weight_gamma) == 0.0:
            return jnp.asarray(1.0, dtype=dtype)

        sr2 = float(self.cfg.sigma_ref) ** 2
        s2 = sigma**2

        if mode == "edm":
            base = sr2 / (s2 + sr2)
        elif mode == "high":
            base = s2 / (s2 + sr2)
        else:
            raise ValueError(
                f"Unknown sigma_weight_mode={self.cfg.sigma_weight_mode!r}"
            )

        return jnp.power(base, float(self.cfg.sigma_weight_gamma)).astype(dtype)

    def __call__(self, log_sigma: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        z, squeeze = self._as_batch(z)
        b = z.shape[0]
        dtype = z.dtype

        sigma = jnp.exp(jnp.asarray(log_sigma).reshape(()))

        # base prior score
        score_base = -z

        u = self.feature_map(z)  # (B,M)

        # evidence score s_p(u | sigma)
        s_p = self.evidence_score(jnp.full((b,), jnp.log(sigma)), u)

        # reference score s_pi(u | sigma)
        if self.reference_score is not None:
            s_pi = self.reference_score(jnp.full((b,), jnp.log(sigma)), u)
        else:
            s_pi = self.feature_map.prior_score_u(u)
            if s_pi is None:
                raise ValueError(
                    "No reference_score provided and feature_map has no analytic prior_score_u(u). "
                    "Provide reference_score or use Identity/Linear feature map."
                )

        delta_u = s_p - s_pi
        delta_z = self.feature_map.vjp(z, delta_u)

        # clip correction per sample
        n = jnp.sqrt(jnp.sum(delta_z * delta_z, axis=-1, keepdims=True) + self.cfg.eps)
        clip = jnp.minimum(1.0, float(self.cfg.max_correction_norm) / n)
        delta_z = delta_z * clip

        gate = self._gate(sigma, dtype)
        w = self._w_sigma(sigma, dtype)

        total = score_base + (float(self.cfg.lambda_strength) * gate * w) * delta_z
        return total[0] if squeeze else total
