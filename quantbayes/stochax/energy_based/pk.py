# quantbayes/stochax/energy_based/pk.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Tuple, Literal, Optional

import jax
import jax.numpy as jnp


PKMode = Literal["none", "evidence", "pk"]


class CoarseObservable(Protocol):
    def value_and_grad(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns d(x): (B,) and grad wrt x: (B, ...)"""
        ...


@dataclass(frozen=True)
class InkFractionObservable01:
    """
    Ink fraction observable for x in [0,1], for MNIST-like images.
    x: (B,1,H,W) or (1,H,W) single sample.

    d = mean(sigmoid((x - thr)/temp)).
    """

    thr: float = 0.35
    temp: float = 0.08
    eps: float = 1e-12

    def value_and_grad(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.asarray(x)
        if x.ndim == 3:
            x = x[None, ...]
        temp = jnp.maximum(jnp.asarray(self.temp, dtype=x.dtype), self.eps)

        u = (x - self.thr) / temp
        s = jax.nn.sigmoid(u)

        d = jnp.mean(s, axis=(1, 2, 3))  # (B,)

        n_pix = x.shape[1] * x.shape[2] * x.shape[3]
        grad = s * (1.0 - s) * (1.0 / temp) * (1.0 / n_pix)
        return d, grad


def gaussian_score(
    z: jnp.ndarray, mu: float, tau: float, eps: float = 1e-12
) -> jnp.ndarray:
    tau2 = (tau * tau) + eps
    return -(z - mu) / tau2


@dataclass(frozen=True)
class PKGuidanceConfig:
    strength: float = 1.0
    sigma_max: Optional[float] = None  # gate on sigma <= sigma_max

    # sigma-weight dial: (sigma_ref^2 / (sigma^2 + sigma_ref^2))^gamma
    sigma_ref: float = 1.0
    gamma: float = 0.0

    max_norm: float = 50.0
    eps: float = 1e-12
    scale_by_sigma2: bool = True


@dataclass(frozen=True)
class PKEvidence:
    mean_d: float
    std_d: float
    mu_z: float
    tau_z: float


class PKGuidance:
    """
    Adds a correction term in x-space score:
      delta_score = λ * gate(sigma) * w(sigma) * clip( score_d(d) * ∇d(x) )
    where score_d(d) comes from (score_p(z) - score_pi(z)) / std_d.
    """

    def __init__(
        self,
        *,
        observable: CoarseObservable,
        evidence: PKEvidence,
        reference_score_z: Callable[[jnp.ndarray], jnp.ndarray],  # s_pi(z): (B,)
        cfg: PKGuidanceConfig,
    ):
        self.observable = observable
        self.evidence = evidence
        self.reference_score_z = reference_score_z
        self.cfg = cfg

    def delta_score(
        self, x: jnp.ndarray, *, sigma: jnp.ndarray, mode: PKMode
    ) -> jnp.ndarray:
        if mode == "none":
            return jnp.zeros_like(x)

        sigma = jnp.asarray(sigma).reshape(())
        d, grad_d = self.observable.value_and_grad(x)  # (B,), (B,...)

        mean_d = float(self.evidence.mean_d)
        std_d = float(self.evidence.std_d)
        z = (d - mean_d) / (std_d + self.cfg.eps)  # (B,)

        # evidence score in z-space
        s_p = gaussian_score(
            z,
            mu=float(self.evidence.mu_z),
            tau=float(self.evidence.tau_z),
            eps=self.cfg.eps,
        )

        if mode == "evidence":
            delta_z = s_p
        else:
            s_pi = self.reference_score_z(z)
            delta_z = s_p - s_pi

        # to d-space
        score_d = delta_z / (std_d + self.cfg.eps)  # (B,)

        reshape = (x.shape[0],) + (1,) * (x.ndim - 1)
        g = score_d.reshape(reshape) * grad_d  # (B,...)

        # clip
        axes = tuple(range(1, g.ndim))
        n = jnp.sqrt(jnp.sum(g * g, axis=axes, keepdims=True) + self.cfg.eps)
        clip = jnp.minimum(1.0, float(self.cfg.max_norm) / n)
        g = g * clip

        # gate
        gate = 1.0
        if self.cfg.sigma_max is not None:
            gate = (sigma <= float(self.cfg.sigma_max)).astype(x.dtype)

        # sigma weight
        if float(self.cfg.gamma) != 0.0:
            sr = float(self.cfg.sigma_ref)
            base = (sr * sr) / (sigma * sigma + sr * sr)
            w = jnp.power(base, float(self.cfg.gamma)).astype(x.dtype)
        else:
            w = jnp.asarray(1.0, dtype=x.dtype)

        scale = float(self.cfg.strength) * gate * w
        if self.cfg.scale_by_sigma2:
            scale = scale * (sigma**2)

        return scale * g


def wrap_score_fn_with_pk(
    base_score_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    *,
    guidance: PKGuidance,
    mode: PKMode,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    def score_fn(log_sigma: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.exp(log_sigma)
        return base_score_fn(log_sigma, x) + guidance.delta_score(
            x, sigma=sigma, mode=mode
        )

    return score_fn
