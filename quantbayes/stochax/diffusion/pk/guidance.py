from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp

from quantbayes.stochax.diffusion.edm import edm_precond_scalars
from quantbayes.stochax.diffusion.parameterizations import edm_denoise_to_x0
from .observables import CoarseObservable

PKMode = Literal["none", "evidence", "pk"]


@dataclass(frozen=True)
class PKGuidanceConfig:
    """
    λ + 𝟙[σ <= sigma_max] style guidance:
      delta_x0 = (sigma^2) * λ * gate(σ) * clip( (score_term(d)) * ∇d )
    """

    strength: float = 1.0  # λ
    sigma_max: Optional[float] = None  # apply only when σ <= sigma_max
    max_norm: float = 10.0
    eps: float = 1e-12
    x0_clip_min: float = -1.0
    x0_clip_max: float = 1.0
    scale_by_sigma2: bool = True


def _as_batch(z: jnp.ndarray) -> jnp.ndarray:
    z = jnp.asarray(z)
    if z.ndim == 0:
        return z[None]
    return z


class PKGuidance:
    """
    Implements PK correction in score form in a modular way.

    You provide:
      - observable: d=T(x), with analytic grad wrt x
      - standardization: z=(d-mean_d)/std_d
      - evidence_score_z: score of target p(z) (or any score fn in z-space)
      - reference_score_z: score of prior pushforward π(z)
    """

    def __init__(
        self,
        *,
        observable: CoarseObservable,
        mean_d: float,
        std_d: float,
        evidence_score_z: Callable[[jnp.ndarray], jnp.ndarray],
        reference_score_z: Callable[[jnp.ndarray], jnp.ndarray],
        cfg: PKGuidanceConfig,
    ):
        self.observable = observable
        self.mean_d = float(mean_d)
        self.std_d = float(std_d)
        self.evidence_score_z = evidence_score_z
        self.reference_score_z = reference_score_z
        self.cfg = cfg

    def delta_x0(
        self, x0: jnp.ndarray, *, sigma: jnp.ndarray, mode: PKMode
    ) -> jnp.ndarray:
        """
        Compute delta in x0-space to be applied as:
          x0 <- x0 + delta_x0

        x0: (B, ...)
        sigma: scalar array
        """
        if mode == "none":
            return jnp.zeros_like(x0)

        x0c = jnp.clip(x0, self.cfg.x0_clip_min, self.cfg.x0_clip_max)

        d, grad_d = self.observable.value_and_grad(x0c)  # d: (B,), grad_d: (B,...)

        # standardize
        z = (d - self.mean_d) / (self.std_d + self.cfg.eps)  # (B,)
        z = _as_batch(z)

        s_p = _as_batch(self.evidence_score_z(z))  # (B,)

        if mode == "evidence":
            score_z = s_p
        else:
            s_pi = _as_batch(self.reference_score_z(z))  # (B,)
            score_z = s_p - s_pi

        # convert score wrt z -> score wrt d
        score_d = score_z / (self.std_d + self.cfg.eps)  # (B,)

        # broadcast to x shape
        b = x0.shape[0]
        reshape = (b,) + (1,) * (x0.ndim - 1)
        score_d_b = score_d.reshape(reshape)

        g = score_d_b * grad_d  # (B,...)

        # per-sample norm clip
        axes = tuple(range(1, g.ndim))
        g_norm = jnp.sqrt(jnp.sum(g * g, axis=axes, keepdims=True) + self.cfg.eps)
        clip = jnp.minimum(1.0, self.cfg.max_norm / g_norm)
        g = g * clip

        # gate by sigma
        gate = 1.0
        if self.cfg.sigma_max is not None:
            gate = (sigma <= self.cfg.sigma_max).astype(x0.dtype)

        scale = jnp.asarray(self.cfg.strength, dtype=x0.dtype) * gate
        if self.cfg.scale_by_sigma2:
            scale = scale * (sigma**2)

        return scale * g


def wrap_edm_denoise_fn_with_pk(
    base_denoise_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    *,
    sigma_data: float,
    guidance: PKGuidance,
    mode: PKMode,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Wrap an EDM denoise_fn (log_sigma, x) -> D with PK guidance.

    Key property for your cancellation test:
      If delta_x0 == 0, then this returns *exactly* the original D (bit-for-bit),
      because we update as: D' = D + delta_x0 / c_out.
    """

    sd = float(sigma_data)

    def guided(log_sigma: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.exp(log_sigma)
        D = base_denoise_fn(log_sigma, x)

        if mode == "none":
            return D

        # x0 from D
        x0 = edm_denoise_to_x0(x, D, sigma, sigma_data=sd)

        # delta in x0 space
        delta = guidance.delta_x0(x0, sigma=sigma, mode=mode)

        # Convert x0-delta into D-delta:
        # x0 = c_skip*x + c_out*D  => D' = D + delta/c_out
        _, _, c_out = edm_precond_scalars(sigma, sd)
        c_out = jnp.maximum(c_out, 1e-12)

        return D + (delta / c_out)

    return guided
