# compute_pac_bound.py
"""
Utilities to evaluate the Normal-Layer PAC-Bayes bound derived in
Theorem 1 of `theory.tex`.

Requires:
    • Equinox ≥ 0.10
    • JAX ≥ 0.4
    • The `spectral_frobenius_lists` helper from norms.py.

Public API
----------
compute_pac_bound(
    model: eqx.Module,
    state: Any,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    *,
    gamma: float = 1.0,
    delta: float = 0.05,
    B: float | None = None,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> float
    Computes L̂_γ (empirical γ-margin loss on the train set)
    plus the PAC-Bayes excess term R, and returns
        bound = min(1.0, L̂_γ + R).
"""
from __future__ import annotations
import math
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jr

from norms import spectral_frobenius_lists
from quantbayes.stochax.trainer.train import predict


def _margin_loss(logits: jnp.ndarray, y: jnp.ndarray, gamma: float) -> float:
    """
    Empirical γ-margin loss:
      • multiclass: fraction of i s.t. f(x_i)_y_i ≤ max_{j≠y_i} f(x_i)_j + γ
      • binary:     fraction of i s.t. (2y_i−1)·logit_i ≤ γ

    Accepts either
      - shape (N,)  → binary logits,
      - shape (N,1) → binary logits,
      - shape (N,K) → multiclass with K>1.
    """
    # flatten any (N,1) to (N,)
    if logits.ndim == 2 and logits.shape[-1] == 1:
        logits = logits.reshape(-1)

    # multiclass case
    if logits.ndim == 2 and logits.shape[-1] > 1:
        correct = logits[jnp.arange(y.shape[0]), y]
        # mask out the correct class
        other = jnp.where(
            jnp.arange(logits.shape[-1])[None, :] == y[:, None],
            -jnp.inf,
            logits,
        ).max(axis=-1)
        vio = correct <= (other + gamma)

    # binary case (now any 1-D array)
    elif logits.ndim == 1:
        signed = jnp.where(y == 1, logits, -logits)
        vio = signed <= gamma

    else:
        raise ValueError(f"Unsupported logits shape {logits.shape}")

    return float(jnp.mean(vio))


def _pac_bayes_excess(
    sigmas: np.ndarray,
    frobs: np.ndarray,
    *,
    B: float,
    d: int,
    gamma: float,
    m: int,
    delta: float,
) -> float:
    """
    The PAC-Bayes excess term
      R = sqrt( [ B²·d²·∏σ_i² · ∑(‖W_i‖_F² / σ_i²) + ln(m/δ) ] / (γ² m) ).
    """
    prod_sigma_sq = float(np.prod(sigmas**2))
    sum_ratio = float(np.sum((frobs**2) / (sigmas**2)))
    numer = (B**2) * (d**2) * prod_sigma_sq * sum_ratio + math.log(m / delta)
    denom = (gamma**2) * m
    return math.sqrt(numer / denom)


def compute_pac_bound(
    model: eqx.Module,
    state: Any,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    *,
    gamma: float = 1.0,
    delta: float = 0.05,
    B: Optional[float] = None,
    key: jr.PRNGKey = jr.PRNGKey(0),
) -> float:
    """
    Compute the PAC-Bayes bound on the TRAIN set:
      bound = min(1, L̂_γ + R),
    where L̂_γ is the empirical margin loss, and R the excess term.
    """
    # 1) empirical margin loss
    logits = predict(model, state, X_train, key=key)
    L_hat = _margin_loss(logits, y_train, gamma)

    # 2) spectral & Frobenius norms
    sigmas, frobs, _ = spectral_frobenius_lists(model, n_iter=50, key=key)
    sigmas = np.asarray(sigmas, dtype=float)
    frobs = np.asarray(frobs, dtype=float)

    # 3) data radius
    if B is None:
        B = float(jnp.max(jnp.linalg.norm(X_train, axis=-1)))

    # 4) PAC-Bayes excess
    R = _pac_bayes_excess(
        sigmas, frobs, B=B, d=len(sigmas), gamma=gamma, m=len(X_train), delta=delta
    )

    return float(min(1.0, L_hat + R))
