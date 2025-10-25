from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from quantbayes.stochax.distributed_training.helpers import tree_mean, tree_mix

Pytree = Any
Array = jnp.ndarray


# ---------- small list-of-pytrees algebra (kept for back-compat) ----------
def _plist_add(A: List[Pytree], B: List[Pytree]) -> List[Pytree]:
    return [jax.tree_util.tree_map(lambda x, y: x + y, a, b) for a, b in zip(A, B)]


def _plist_scale(A: List[Pytree], s: float) -> List[Pytree]:
    return [jax.tree_util.tree_map(lambda x: s * x, a) for a in A]


# ---------- simple baseline: repeat-K ----------
def repeat_mix(W: jnp.ndarray, params: List[Pytree], K: int) -> List[Pytree]:
    """Apply W K times to the list of parameter pytrees."""
    out = params
    for _ in range(max(1, int(K))):
        out = tree_mix(W, out)
    return out


# ---------- Chebyshev helpers ----------
def _cheb_T_scalar(K: int, x: float) -> float:
    """Numerically stable Chebyshev T_K(x)."""
    if K == 0:
        return 1.0
    if K == 1:
        return float(x)
    if x > 1.0:
        a = math.acosh(float(x))
        return float(math.cosh(K * a))
    # x in [-1,1]
    Tkm2, Tkm1 = 1.0, float(x)
    for _ in range(2, K + 1):
        Tk = 2.0 * x * Tkm1 - Tkm2
        Tkm2, Tkm1 = Tkm1, Tk
    return float(Tkm1)


def xi_from_interval(lam_min: float, lam_max: float) -> float:
    """ξ = (2 - (λmax+λmin)) / (λmax - λmin); returns 1.0 if interval degenerate."""
    width = float(lam_max - lam_min)
    if not np.isfinite(width) or width <= 0.0:
        return 1.0
    return float((2.0 - (lam_max + lam_min)) / width)


def _assert_symmetric_ds(W: Array, atol: float = 1e-6) -> None:
    """Guarantees used here require symmetric, row-stochastic W."""
    if not jnp.allclose(W, W.T, atol=atol):
        raise ValueError("Chebyshev requires symmetric W (W == W.T).")
    rs = jnp.sum(W, axis=1)
    if not jnp.allclose(rs, jnp.ones_like(rs), atol=atol):
        raise ValueError("Chebyshev requires row sums ≈ 1 (row-stochastic).")


def should_mix(t: int, mix_every: int) -> bool:
    """Mix when (t+1) is divisible by mix_every (τ)."""
    mix_every = max(1, int(mix_every))
    return ((t + 1) % mix_every) == 0


def chebyshev_mix(
    W: Array, param_list: List[Any], K: int, lam_min: float, lam_max: float
) -> List[Any]:
    """
    Degree-K Chebyshev mixing on the disagreement component with average preserved.
    Implements p_K(W) = T_K(Z) / T_K(ξ) on deviations, where Z is W normalized
    from [lam_min, lam_max] to [-1,1]; then adds the mean back.

    Note: if Γ_t = (1/n) Σ ||θ_i - θ̄||^2, then a single linear mixing step obeys
      Γ_{t+1/2} / Γ_t ≤ μ^2, where μ = max_{i≥2} |λ_i(W)| (SLEM). For Chebyshev, the
      worst-case K-step ratio satisfies Γ_after / Γ_before ≤ 1 / T_K(ξ)^2.
    """
    K = int(max(1, K))
    width = float(lam_max - lam_min)
    if not np.isfinite(width) or width <= 1e-12:
        return param_list  # degenerate
    _assert_symmetric_ds(W)

    # separate mean
    mu = tree_mean(param_list)
    dev0 = [jax.tree_util.tree_map(lambda pi, m: pi - m, p, mu) for p in param_list]

    # normalized operator Z acting on pytrees:
    def _apply_Z(v_list: List[Any]) -> List[Any]:
        a = 2.0 / width
        b = (lam_max + lam_min) / width
        Wv = tree_mix(W, v_list)
        out: List[Any] = []
        for i in range(len(v_list)):
            out.append(
                jax.tree_util.tree_map(
                    lambda wvi, vi: a * wvi - b * vi, Wv[i], v_list[i]
                )
            )
        return out

    # Chebyshev recurrence on deviations
    if K == 1:
        num = _apply_Z(dev0)  # T1(Z)*dev0
    else:
        Tkm2 = dev0  # T0
        Tkm1 = _apply_Z(Tkm2)  # T1
        for _ in range(2, K + 1):
            ZTkm1 = _apply_Z(Tkm1)
            Tk = [
                jax.tree_util.tree_map(lambda z, t: 2.0 * z - t, ztk1, tk2)
                for ztk1, tk2 in zip(ZTkm1, Tkm2)
            ]  # T_{k+1} = 2 Z T_k - T_{k-1}
            Tkm2, Tkm1 = Tkm1, Tk
        num = Tkm1  # T_K(Z) * dev0

    # normalize by T_K(ξ) to get minimax contraction
    xi = xi_from_interval(lam_min, lam_max)
    denom = float(_cheb_T_scalar(K, xi))
    scale = 1.0 / max(1e-20, denom)
    devK = [jax.tree_util.tree_map(lambda v: scale * v, v) for v in num]

    # reconstruct parameters
    out = [jax.tree_util.tree_map(lambda di, m: di + m, d, mu) for d in devK]
    return out


# ---------- helper: disagreement interval for DS W = I - α L ----------
def disagreement_interval_from_L(L: jnp.ndarray, alpha: float) -> Tuple[float, float]:
    """
    For W = I - α L (undirected), the disagreement eigenvalues lie in:
      [1 - α λ_max(L), 1 - α λ_2(L)].
    """
    eigs = jnp.linalg.eigvalsh(L)  # sorted ascending
    lam2 = float(eigs[1]) if eigs.shape[0] >= 2 else 0.0
    lammax = float(eigs[-1]) if eigs.shape[0] >= 1 else 0.0
    lam_min = 1.0 - float(alpha) * lammax
    lam_max = 1.0 - float(alpha) * lam2
    return lam_min, lam_max


def rho_phase_sq(K: int, xi: float) -> float:
    """Worst-case squared-contraction on Γ for degree-K Chebyshev: 1 / T_K(ξ)^2."""
    if K <= 0 or xi <= 1.0 or not np.isfinite(xi):
        return 1.0
    return float(1.0 / (math.cosh(K * math.acosh(float(xi))) ** 2))


def K_from_rho_target_sq(xi: float, rho_target_sq: float) -> int:
    """
    Smallest K with 1/T_K(ξ)^2 <= rho_target_sq (i.e., Γ_after / Γ_before <= rho_target_sq).
    """
    xi = float(xi)
    rho_target_sq = float(rho_target_sq)
    if not (0.0 < rho_target_sq < 1.0) or xi <= 1.0 or not np.isfinite(xi):
        return 1
    return int(math.ceil(math.acosh(1.0 / math.sqrt(rho_target_sq)) / math.acosh(xi)))


@dataclass
class AdaptiveChebyController:
    """
    Global-aware Chebyshev controller.

    Targets a GLOBAL contraction on Γ at mix events, accounting for partial
    participation via the heuristic decomposition:
        E[ rho_global ] ≈ (1 - p) + p * rho_local,  with  p = active / total
    where rho denotes the squared ratio Γ_after / Γ_before.
    """

    rho_target_sq_global: float = (
        0.20  # desired GLOBAL Γ_after / Γ_before at mix events
    )
    K_max: int = 5
    last_post_gamma: float = 1.0  # Γ right after previous mix
    last_K: int = 1

    # ---- New: global-aware chooser
    def choose_K_global(
        self, xi: float, gamma_pre: float, active: int, total: int
    ) -> int:
        """
        Choose K to meet a GLOBAL target on Γ, given only 'active' of 'total' nodes participate.
        Uses E[rho_global] ≈ (1-p) + p * rho_local and budgets observed growth since last mix.
        """
        p = float(max(1, active)) / float(max(1, total))

        # Growth since last mix (≥ 1.0)
        growth = float(gamma_pre / max(1e-20, self.last_post_gamma))
        growth = float(np.clip(growth, 1.0, 1e9))

        # Per-mix global budget, after accounting for growth
        rho_goal_sq_global = float(
            np.clip(self.rho_target_sq_global / growth, 1e-6, 0.999)
        )

        # Inactivity floor (cannot beat 1-p globally in a single event)
        floor = 1.0 - p
        if rho_goal_sq_global <= floor + 1e-12:
            rho_local_req_sq = 1e-6  # saturate locally
        else:
            rho_local_req_sq = (rho_goal_sq_global - floor) / max(1e-6, p)
            rho_local_req_sq = float(np.clip(rho_local_req_sq, 1e-6, 0.999))

        K_req = K_from_rho_target_sq(float(xi), float(rho_local_req_sq))
        return int(max(1, min(self.K_max, K_req)))

    # ---- Legacy: per-mix local target (kept for back-compat)
    def choose_K(self, xi: float, gamma_pre: float) -> int:
        """
        Legacy local-only budgeter: ignores participation; preserved for back-compat.
        """
        growth = float(gamma_pre / max(1e-20, self.last_post_gamma))
        growth = float(np.clip(growth, 1.0, 1e6))
        # Interpret stored target as "local" for legacy call-sites
        rho_target_sq_local = float(
            np.clip(self.rho_target_sq_global / growth, 1e-6, 0.999)
        )
        K_req = K_from_rho_target_sq(float(xi), rho_target_sq_local)
        return int(max(1, min(self.K_max, K_req)))

    def update_after_mix(self, hat_rho_sq: float, K_used: int, gamma_post: float):
        self.last_post_gamma = float(max(1e-20, gamma_post))
        self.last_K = int(K_used)


# Back-compat helper (same semantics you used)
def min_K_for_target_rho(xi: float, rho_target: float) -> int:
    """Smallest integer K with 1/T_K(ξ)^2 <= rho_target (ξ>1)."""
    xi = float(xi)
    rho_target = float(rho_target)
    if not np.isfinite(xi) or xi <= 1.0 or not (0.0 < rho_target < 1.0):
        return 1
    num = np.arccosh(1.0 / max(1e-12, np.sqrt(rho_target)))
    den = np.arccosh(xi)
    if not np.isfinite(den) or den <= 0.0:
        return 1
    return int(np.ceil(num / den))


# ---------- partial participation utilities ----------
def sample_active_clients(
    key: jax.Array, n_clients: int, p: float, *, ensure_nonempty: bool = True
) -> Tuple[List[int], jax.Array]:
    """Bernoulli(p) per client; optionally force at least one active."""
    import jax.random as jr

    key, sub = jr.split(key)
    if p >= 1.0:
        return list(range(n_clients)), key
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
