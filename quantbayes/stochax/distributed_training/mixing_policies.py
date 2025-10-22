# quantbayes/stochax/distributed_training/mixing_policies.py
from __future__ import annotations
from typing import List, Any, Tuple
import jax
import jax.numpy as jnp

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
