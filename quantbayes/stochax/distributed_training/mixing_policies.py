# quantbayes/stochax/distributed_training/mixing_policies.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

# Use your existing helpers
from .dgd import _tree_mix, laplacian_from_edges  # already in your repo

Pytree = Any


# ---------- small list-of-pytrees algebra ----------
def _plist_add(A: List[Pytree], B: List[Pytree]) -> List[Pytree]:
    return [jax.tree_util.tree_map(lambda x, y: x + y, a, b) for a, b in zip(A, B)]


def _plist_scale(A: List[Pytree], s: float) -> List[Pytree]:
    return [jax.tree_util.tree_map(lambda x: s * x, a) for a in A]


# ---------- simple baseline: repeat-K ----------
def repeat_mix(W: jnp.ndarray, params: List[Pytree], K: int) -> List[Pytree]:
    """Apply W K times to the list of parameter pytrees."""
    out = params
    for _ in range(max(1, int(K))):
        out = _tree_mix(W, out)
    return out


# ---------- Chebyshev polynomial acceleration ----------
def _cheb_T_scalar(K: int, x: float) -> float:
    if K == 0:
        return 1.0
    if K == 1:
        return x
    Tkm2, Tkm1 = 1.0, x
    for _ in range(2, K + 1):
        Tk = 2.0 * x * Tkm1 - Tkm2
        Tkm2, Tkm1 = Tkm1, Tk
    return Tkm1


def chebyshev_mix(
    W: jnp.ndarray,
    params: List[Pytree],
    K: int,
    lam_min: float,
    lam_max: float,
) -> List[Pytree]:
    """
    Apply degree-K Chebyshev polynomial p_K(W) to `params`.
    Average-preserving: normalized so p_K(1) = 1.
    Requires the disagreement-spectrum interval [lam_min, lam_max] of W.
    """
    assert K >= 1 and lam_max >= lam_min
    n = W.shape[0]
    I = jnp.eye(n, dtype=W.dtype)

    # Affine map W -> W_hat in [-1,1] on disagreement spectral interval
    W_hat = (2.0 * W - (lam_max + lam_min) * I) / (lam_max - lam_min)

    # Chebyshev recurrence on lists-of-pytrees:
    y0 = params
    y1 = _tree_mix(W_hat, params)  # T1(W_hat) * params
    if K == 1:
        num = y1
    else:
        ykm2, ykm1 = y0, y1
        for _ in range(2, K + 1):
            tmp = _tree_mix(W_hat, ykm1)  # W_hat * y_{k-1}
            yk = _plist_add(_plist_scale(tmp, 2.0), _plist_scale(ykm2, -1.0))
            ykm2, ykm1 = ykm1, yk
        num = ykm1  # T_K(W_hat) * params

    # Normalize so that p_K(1)=1
    xi = (2.0 - (lam_max + lam_min)) / (lam_max - lam_min)
    denom = _cheb_T_scalar(K, float(xi))
    return _plist_scale(num, 1.0 / float(denom))


# ---------- helper: disagreement interval for DS W = I - α L ----------
def disagreement_interval_from_L(L: jnp.ndarray, alpha: float) -> Tuple[float, float]:
    """
    For W = I - α L (undirected), the disagreement eigenvalues lie in:
      [1 - α λ_max(L), 1 - α λ_2(L)].
    """
    eigs = jnp.linalg.eigvalsh(L)  # sorted ascending
    lam2 = float(eigs[1])
    lammax = float(eigs[-1])
    lam_min = 1.0 - alpha * lammax
    lam_max = 1.0 - alpha * lam2
    return lam_min, lam_max


@dataclass
class AdaptiveChebyConfig:
    """Config for adaptive star rounds."""

    p_participation: float = 0.5  # Bernoulli per client
    rho_target: float = 0.05  # target per-star contraction on disagreement
    K_max: int = 5  # cap on chebyshev degree
    ensure_nonempty: bool = True  # always activate >=1 client


def min_K_for_target_rho(xi: float, rho_target: float) -> int:
    """Smallest integer K with 1/T_K(xi)^2 <= rho_target (xi>1)."""
    xi = float(xi)
    rho_target = float(rho_target)
    if not np.isfinite(xi) or xi <= 1.0 or rho_target <= 0.0 or rho_target >= 1.0:
        return 1
    num = np.arccosh(1.0 / max(1e-12, np.sqrt(rho_target)))
    den = np.arccosh(xi)
    if not np.isfinite(den) or den <= 0.0:
        return 1
    return int(np.ceil(num / den))


def sample_active_clients(
    key: jax.Array, n_clients: int, p: float, *, ensure_nonempty: bool = True
) -> Tuple[List[int], jax.Array]:
    """Bernoulli(p) per client; optionally force at least one active."""
    key, sub = jr.split(key)
    if p >= 1.0:
        active = list(range(n_clients))
        return active, key
    mask = jr.uniform(sub, (n_clients,)) < float(p)
    active = [i for i in range(n_clients) if bool(mask[i])]
    if ensure_nonempty and not active:
        key, sub2 = jr.split(key)
        fallback = int(jr.randint(sub2, shape=(), minval=0, maxval=n_clients))
        active = [fallback]
    return active, key


def build_partial_star_W(
    n_total: int, server_id: int, active_clients_global: List[int], alpha: float
) -> jnp.ndarray:
    """
    Row-stochastic, symmetric W for a star that connects the server only to the
    *active* clients; all inactive clients are identity rows.
    """
    W = jnp.eye(n_total, dtype=jnp.float32)
    deg = float(len(active_clients_global))
    if deg <= 0:
        return W
    a = float(alpha)
    # Server row:
    W = W.at[server_id, server_id].set(1.0 - a * deg)
    for c in active_clients_global:
        W = W.at[server_id, c].set(a)
        W = W.at[c, server_id].set(a)
        W = W.at[c, c].set(1.0 - a)
    return W
