# quantbayes/stochax/distributed_training/spectral.py
from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray
Pytree = Any

# ---------- Graph spectrum & bounds ----------


def disagreement_interval_from_W(W: Array) -> Tuple[float, float]:
    """Return (lambda_min, lambda_max) over the disagreement spectrum of W.

    Assumes W is symmetric, row-stochastic; the largest eigenvalue is ~1 and
    corresponds to the consensus eigenvector. Then:
      lambda_min = smallest eigenvalue (possibly negative)
      lambda_max = second-largest eigenvalue (just below 1 in connected graphs)
    """
    ds_checks(W)
    eigW = jnp.linalg.eigvalsh(W)  # ascending
    # Largest eigenvalue should be 1 up to small numerical error.
    assert jnp.isclose(eigW[-1], 1.0, atol=1e-5), "Largest eigenvalue of W must be 1."
    lam_min = float(eigW[0])
    lam_max = float(eigW[-2]) if eigW.shape[0] >= 2 else float(eigW[-1])
    return lam_min, lam_max


def ds_checks(W: Array, atol: float = 1e-6) -> None:
    rs = jnp.sum(W, axis=1)
    assert jnp.allclose(W, W.T, atol=atol), "W must be symmetric."
    assert jnp.allclose(rs, jnp.ones_like(rs), atol=atol), "Rows must sum to 1."


def disagreement_interval_from_L(L: Array, alpha: float) -> Tuple[float, float]:
    eigL = jnp.linalg.eigvalsh(L)  # ascending
    lam2 = float(eigL[1]) if eigL.shape[0] >= 2 else 0.0
    lammax = float(eigL[-1]) if eigL.shape[0] >= 1 else 0.0
    lam_min = 1.0 - float(alpha) * lammax
    lam_max = 1.0 - float(alpha) * lam2
    return lam_min, lam_max


def xi_from_interval(lam_min: float, lam_max: float) -> float:
    width = float(lam_max - lam_min)
    if not jnp.isfinite(width) or width <= 0.0:
        return 1.0
    return float((2.0 - (lam_max + lam_min)) / width)


def T_K_scalar(K: int, x: float) -> float:
    if K == 0:
        return 1.0
    if K == 1:
        return float(x)
    if x > 1.0:
        return float(math.cosh(K * math.acosh(float(x))))
    t0, t1 = 1.0, float(x)
    for _ in range(2, K + 1):
        t0, t1 = t1, 2.0 * x * t1 - t0
    return float(t1)


def rho_bound_sq(K: int, xi: float) -> float:
    if not jnp.isfinite(xi) or xi <= 1.0 or K <= 0:
        return 1.0
    return float(1.0 / (math.cosh(K * math.acosh(float(xi))) ** 2))


def spectrum_report(
    W: Array, L: Array | None = None, alpha: float | None = None
) -> Dict[str, float]:
    """Report interval and derived quantities for the *actual* W provided.
    L and alpha are ignored (kept only for backward-compatibility).
    """
    lam_min, lam_max = disagreement_interval_from_W(W)
    xi = xi_from_interval(lam_min, lam_max)
    slem = max(abs(lam_min), abs(lam_max))
    return {
        "lam_min": float(lam_min),
        "lam_max": float(lam_max),
        "xi": float(xi),
        "slem": float(slem),
    }


# ---------- Pytree mixing ----------


def tree_mean(param_list: List[Pytree]) -> Pytree:
    n = len(param_list)
    w = jnp.ones((n,), dtype=jnp.float32) / max(1, n)

    def combine(*leaves):
        stacked = jnp.stack(leaves, axis=0)
        return jnp.tensordot(w, stacked, axes=(0, 0))

    return jax.tree_util.tree_map(combine, *param_list)


def tree_mix(W: Array, param_list: List[Pytree]) -> List[Pytree]:
    def weighted_sum(weights, plist):
        def combine(*leaves):
            stacked = jnp.stack(leaves, axis=0)
            return jnp.tensordot(weights, stacked, axes=(0, 0))

        return jax.tree_util.tree_map(combine, *plist)

    return [weighted_sum(W[i], param_list) for i in range(W.shape[0])]


def repeat_mix(W: Array, params: List[Pytree], K: int) -> List[Pytree]:
    out = params
    for _ in range(max(1, int(K))):
        out = tree_mix(W, out)
    return out


def chebyshev_mix(
    W: Array, params: List[Pytree], K: int, lam_min: float, lam_max: float
) -> List[Pytree]:
    ds_checks(W)
    if K <= 0:
        return params
    K = int(K)
    width = float(lam_max - lam_min)
    if width <= 1e-12 or not jnp.isfinite(width):
        return params

    mu = tree_mean(params)
    dev0 = [jax.tree_util.tree_map(lambda p, m: p - m, p, mu) for p in params]

    a = 2.0 / width
    b = (lam_max + lam_min) / width

    def apply_Z(vs: List[Pytree]) -> List[Pytree]:
        Wv = tree_mix(W, vs)
        return [
            jax.tree_util.tree_map(lambda wv, v: a * wv - b * v, Wv[i], vs[i])
            for i in range(len(vs))
        ]

    if K == 1:
        Tk = apply_Z(dev0)
    else:
        Tkm2 = dev0
        Tkm1 = apply_Z(Tkm2)
        for _ in range(2, K + 1):
            ZTkm1 = apply_Z(Tkm1)
            Tk = [
                jax.tree_util.tree_map(lambda z, t: 2.0 * z - t, z1, t2)
                for z1, t2 in zip(ZTkm1, Tkm2)
            ]
            Tkm2, Tkm1 = Tkm1, Tk
        Tk = Tkm1

    xi = xi_from_interval(lam_min, lam_max)
    scale = 1.0 / max(1e-20, T_K_scalar(K, xi))
    devK = [jax.tree_util.tree_map(lambda v: scale * v, v) for v in Tk]
    return [jax.tree_util.tree_map(lambda d, m: d + m, d, mu) for d in devK]


def flatten_params_l2(pytree: Pytree) -> Array:
    leaves = jax.tree_util.tree_leaves(pytree)
    flat = [jnp.ravel(x) for x in leaves if x is not None]
    return jnp.concatenate(flat) if flat else jnp.zeros((1,), dtype=jnp.float32)


def consensus_gamma(params: List[Pytree]) -> float:
    vecs = [flatten_params_l2(p) for p in params]
    stack = jnp.stack(vecs, axis=0)
    mu = jnp.mean(stack, axis=0, keepdims=True)
    sq = jnp.sum((stack - mu) ** 2, axis=1)
    return float(jnp.mean(sq))


def consensus_gamma_subset(params: List[Pytree], idxs: List[int]) -> float:
    vecs = [flatten_params_l2(params[i]) for i in idxs]
    stack = jnp.stack(vecs, axis=0)
    mu = jnp.mean(stack, axis=0, keepdims=True)
    sq = jnp.sum((stack - mu) ** 2, axis=1)
    return float(jnp.mean(sq))


# ------------------------------- MAIN ---------------------------------
if __name__ == "__main__":
    """
    Minimal demo:
      • Build a ring graph
      • Report spectrum + Chebyshev bound
      • Apply Chebyshev mixing once and print empirical contraction vs bound
    """
    import jax.random as jr
    from quantbayes.stochax.distributed_training.helpers import (
        ring_edges,
        laplacian_from_edges,
        mixing_matrix,
        safe_alpha,
    )

    N = 5
    edges = ring_edges(N)
    alpha = safe_alpha(edges, N)
    L = laplacian_from_edges(N, edges)
    W = mixing_matrix(N, edges, alpha, lazy=False)

    rep = spectrum_report(W, L, alpha)
    print("Spectrum report:", rep)
    K = 3
    print(f"Theory bound (K={K}): rho_bound_sq =", rho_bound_sq(K, rep["xi"]))

    # synthetic parameters (one array per node)
    key = jr.PRNGKey(0)
    keys = jr.split(key, N)
    params = [jr.normal(keys[i], (32,)) for i in range(N)]
    g0 = consensus_gamma(params)
    params2 = chebyshev_mix(
        W, params, K=K, lam_min=rep["lam_min"], lam_max=rep["lam_max"]
    )
    g1 = consensus_gamma(params2)
    rho_hat = g1 / max(g0, 1e-12)
    print(
        f"Empirical rho_hat={rho_hat:.4e} vs. theory ≤ {rho_bound_sq(K, rep['xi']):.4e}"
    )
