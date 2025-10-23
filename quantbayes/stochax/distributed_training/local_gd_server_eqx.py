# Local GD with a server (coordinator) and periodic communication (topology switching).
# The server has no data and never performs a local gradient step.
# At iterations that are multiples of τ, we use topology A (star: server connected to all clients).
# Otherwise, topology B (no edges) => identity mixing.
#
# Public API:
#   - LocalGDServerEqx
#   - make_star_with_server_edges
#   - make_mixing_with_per_node_alphas
#   - plot_server_loss, plot_consensus_localgd
#   - make_constant_lr, make_polynomial_decay (simple LR schedules)

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, Any
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt

from quantbayes.stochax.trainer.train import eval_step, binary_loss
from quantbayes.stochax.privacy.dp import DPSGDConfig, DPPrivacyEngine

# Reuse helpers from your DGD module
from quantbayes.stochax.distributed_training.dgd import (
    mixing_matrix,  # W = I - alpha * L (symmetric, row-stochastic)
    safe_alpha as _safe_alpha_ha4,
    _partition_params,
    _combine_params,
    _tree_mix,
    laplacian_from_edges,
)

# NEW: spectral star policies
from quantbayes.stochax.distributed_training.mixing_policies import (
    repeat_mix,
    chebyshev_mix,
    disagreement_interval_from_L,
    AdaptiveChebyConfig,  # NEW
    min_K_for_target_rho,  # NEW
    sample_active_clients,  # NEW
    build_partial_star_W,  # NEW
)

Array = jnp.ndarray
PRNG = jax.Array
BatchSchedule = Callable[[int, int, int], int]  # B = schedule(t, client_index, n_i)

__all__ = [
    "LocalGDServerEqx",
    "make_star_with_server_edges",
    "make_mixing_with_per_node_alphas",
    "plot_server_loss",
    "plot_consensus_localgd",
    "safe_alpha",
    "make_constant_lr",
    "make_polynomial_decay",
]


# ---- small utilities ----


def safe_alpha(edges: List[Tuple[int, int]], n_nodes: int) -> float:
    """Conservative α < 1/deg_max, re-exported from HA4."""
    return _safe_alpha_ha4(edges, n_nodes)


def _flatten_params_l2(pytree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(pytree)
    flat = [jnp.ravel(x) for x in leaves if x is not None]
    return jnp.concatenate(flat) if flat else jnp.zeros((1,), dtype=jnp.float32)


def _tree_weighted_sum(weights: Array, param_list: List[Any]) -> Any:
    def combine(*leaves):
        stacked = jnp.stack(leaves, axis=0)  # (n_nodes, ...)
        return jnp.tensordot(weights, stacked, axes=(0, 0))

    return jax.tree_util.tree_map(combine, *param_list)


def _tree_mix_local(W: Array, param_list: List[Any]) -> List[Any]:
    n = len(param_list)
    mixed: List[Any] = []
    for i in range(n):
        mixed_i = _tree_weighted_sum(W[i], param_list)
        mixed.append(mixed_i)
    return mixed


def make_star_with_server_edges(
    n_clients: int, server_id: int | None = None
) -> List[Tuple[int, int]]:
    """
    Build star edges for n_clients + 1 total nodes. Server is by default the last index.
    Returns undirected edge list: [(server, 0), (server, 1), ...].
    """
    if server_id is None:
        server_id = n_clients
    edges = [(server_id, i) for i in range(n_clients)]
    return edges


def make_mixing_with_per_node_alphas(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    alphas: List[float],
) -> Array:
    """
    Optional: Build a row-stochastic mixing matrix using node-specific rates α_i.
    For undirected edges (i,j), for each row i:
        W_ii = 1 - α_i * deg(i),   W_ij = α_i if j in N(i), else 0.
    Row-stochastic by construction; not generally symmetric.
    """
    assert len(alphas) == n_nodes
    deg = np.zeros(n_nodes, dtype=np.int32)
    N = [set() for _ in range(n_nodes)]
    for i, j in edges:
        if i == j:
            continue
        N[i].add(j)
        N[j].add(i)
        deg[i] += 1
        deg[j] += 1

    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        ai = float(alphas[i])
        for j in N[i]:
            W[i, j] = ai
        W[i, i] = 1.0 - ai * deg[i]
    rs = np.sum(W, axis=1)
    if not np.allclose(rs, np.ones_like(rs), atol=1e-6):
        raise ValueError("Row sums of W are not 1; check alphas.")
    return jnp.array(W)


def plot_server_loss(
    histories: Dict[str, Dict[str, List[float]]],
    title: str = "Server global training loss",
    save: Optional[str] = None,
):
    plt.figure(figsize=(7.6, 4.4))
    for name, h in histories.items():
        plt.plot(h["loss_server"], label=name, linewidth=2)
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("global training loss (server model)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=180)
    plt.show()


def plot_consensus_localgd(
    histories: Dict[str, Dict[str, List[float]]],
    title: str = "Consensus distance",
    save: Optional[str] = None,
):
    plt.figure(figsize=(7.2, 4.2))
    for name, h in histories.items():
        if "consensus_all" in h:
            plt.plot(h["consensus_all"], label=f"{name} — all (incl. S)", linewidth=2)
        if "consensus_clients" in h:
            plt.plot(
                h["consensus_clients"],
                label=f"{name} — clients only",
                linestyle="--",
                linewidth=2,
            )
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("squared consensus distance")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=180)
    plt.show()


# ---- main trainer ----
class LocalGDServerEqx:
    """
    Local GD with a coordinator (server) and periodic communication.

    Nodes: n_clients clients + 1 server (index = server_id).
    At iterations t = τ, 2τ, 3τ, ... we use topology A (e.g., star with server),
    so a gossip step with W_A is applied. At all the other iterations we use topology B
    (no edges) → W_B = I (no mixing).
    After gossip, clients do a local gradient step (GD or SGD). The server never
    does a local step: θ_{t+1}^S = θ_{t+1/2}^S.

    Star policy options (self.star_policy["name"]):
      - "single":   apply W_A once
      - "repeat":   apply W_A K times (self.star_policy["K"])
      - "cheby":    apply degree-K Chebyshev p_K(W_A) using the full-star spectrum
      - "adaptive": per-star partial participation + adaptive Chebyshev
    Logged histories:
      - loss_server, loss_node1, loss_node4
      - consensus_all, consensus_clients
      - rho_star, K_hist, active_hist (when applicable)
    """

    def __init__(
        self,
        model_init_fn: Callable[[PRNG], eqx.Module],
        n_clients: int,
        tau: int,
        edges_A: List[Tuple[int, int]],
        *,
        edges_B: Optional[List[Tuple[int, int]]] = None,
        alpha_A: Optional[float] = None,
        gamma: float | Callable[[int], float] = 0.1,  # schedule support
        T: int = 400,
        loss_fn: Callable = binary_loss,
        key: Optional[PRNG] = None,
        eval_inference_mode: bool = True,
        server_id: Optional[int] = None,
        make_W_A_custom: Optional[Callable[[int, List[Tuple[int, int]]], Array]] = None,
        dp_config: Optional[DPSGDConfig] = None,
        star_policy: Optional[Dict[str, Any]] = None,
        batch_schedule: BatchSchedule | None = None,  # NEW: None => GD; else => SGD
    ):
        self.model_init_fn = model_init_fn
        self.n_clients = int(n_clients)
        self.server_id = self.n_clients if server_id is None else int(server_id)
        self.n_total = self.n_clients + 1
        assert 0 <= self.server_id < self.n_total
        self.client_ids = [i for i in range(self.n_total) if i != self.server_id]

        self.tau = int(tau)
        self.edges_A = edges_A
        self.edges_B = edges_B if edges_B is not None else []
        self.alpha_A = (
            float(alpha_A) if alpha_A is not None else safe_alpha(edges_A, self.n_total)
        )
        self.gamma = gamma  # float or Callable
        self.T = int(T)
        self.loss_fn = loss_fn
        self.key = jr.PRNGKey(0) if key is None else key
        self.eval_inference_mode = bool(eval_inference_mode)
        self.dp_config = dp_config
        self._dp_engine = DPPrivacyEngine(dp_config) if dp_config else None

        # Mixing matrices
        if make_W_A_custom is None:
            self.W_A = mixing_matrix(self.n_total, self.edges_A, self.alpha_A)
            rs = jnp.sum(self.W_A, axis=1)
            assert jnp.allclose(self.W_A, self.W_A.T, atol=1e-6)
            assert jnp.allclose(rs, jnp.ones_like(rs), atol=1e-6)
        else:
            self.W_A = make_W_A_custom(self.n_total, self.edges_A)
            rs = jnp.sum(self.W_A, axis=1)
            # May be non-symmetric row-stochastic; still require row sums 1
            assert jnp.allclose(rs, jnp.ones_like(rs), atol=1e-6)

        self.W_B = jnp.eye(self.n_total, dtype=jnp.float32)
        self.models: List[eqx.Module] = []
        self.states: List[Any] = []

        # Star policy
        self.star_policy: Dict[str, Any] = star_policy or {"name": "single"}
        if self.star_policy.get("name") == "adaptive":
            self.star_policy.setdefault("p_participation", 0.5)
            self.star_policy.setdefault("rho_target", 0.05)
            self.star_policy.setdefault("K_max", 5)
            self.star_policy.setdefault("ensure_nonempty", True)

        # NEW: minibatch control
        self.batch_schedule = batch_schedule  # None => GD; else => DSGD

    # ---- helpers ----

    def _gamma_t(self, t: int) -> float:
        return float(self.gamma(t)) if callable(self.gamma) else float(self.gamma)

    def _local_grad_step(
        self, model: eqx.Module, state: Any, X: Array, y: Array, key: PRNG
    ):
        # returns (grad, new_state)
        def loss(m, s, xb, yb, k):
            return self.loss_fn(m, s, xb, yb, k)

        if self._dp_engine is None:
            (_, new_s), grad = eqx.filter_value_and_grad(loss, has_aux=True)(
                model, state, X, y, key
            )
            return grad, new_s
        else:
            noisy_grad, new_s = self._dp_engine.noisy_grad(
                loss, model, state, X, y, key=key
            )
            return noisy_grad, new_s

    def _init_models(self) -> None:
        models, states = [], []
        k = self.key
        for _ in range(self.n_total):
            k, sub = jr.split(k)
            m, s = eqx.nn.make_with_state(self.model_init_fn)(sub)
            models.append(m)
            states.append(s)
        self.key = k
        self.models = models
        self.states = states

    def _consensus_distance(self, params_list: List[Any], idxs: List[int]) -> float:
        def _flatten_params_l2(pytree: Any) -> Array:
            leaves = jax.tree_util.tree_leaves(pytree)
            flat = [jnp.ravel(x) for x in leaves if x is not None]
            return jnp.concatenate(flat) if flat else jnp.zeros((1,), dtype=jnp.float32)

        vecs = [_flatten_params_l2(params_list[i]) for i in idxs]
        stack = jnp.stack(vecs, axis=0)
        mean = jnp.mean(stack, axis=0, keepdims=True)
        sq = jnp.sum((stack - mean) ** 2, axis=1)
        return float(jnp.mean(sq))

    # ---- main fit ----

    def fit(
        self,
        parts: List[Tuple[Array, Array]],
        X_full: Array,
        y_full: Array,
        *,
        eval_key: Optional[PRNG] = None,
        log: bool = True,
    ) -> Dict[str, List[float]]:
        if not self.models:
            self._init_models()

        params_list, static_list = [], []
        for m in self.models:
            p, s = _partition_params(m)
            params_list.append(p)
            static_list.append(s)

        rng = self.key if eval_key is None else eval_key
        log_interval = max(1, self.T // 10)

        # NEW: client dataset sizes + sample accounting (if SGD)
        sizes = {i: int(parts[i][0].shape[0]) for i in self.client_ids}
        total_samples = {i: 0 for i in self.client_ids} if self.batch_schedule else None

        hist_loss_server: List[float] = []
        hist_loss_node1: List[float] = []
        hist_loss_node4: List[float] = []
        hist_cons_all: List[float] = []
        hist_cons_clients: List[float] = []
        hist_star_rho: List[float] = []  # per-star Γ_after / Γ_before
        hist_K: List[int] = []  # degree used per star (all policies)
        hist_active: List[int] = (
            []
        )  # #active clients per star (adaptive; else = n_clients)

        # Precompute Chebyshev interval lazily for full DS star (for "cheby")
        def _ensure_star_interval():
            li = getattr(self, "_star_lam_interval", None)
            if li is None:
                L_star = laplacian_from_edges(self.n_total, self.edges_A)
                self._star_lam_interval = disagreement_interval_from_L(
                    L_star, self.alpha_A
                )
            return self._star_lam_interval

        for t in range(self.T):
            is_star = (t + 1) % self.tau == 0

            # Per-star contraction: measure before star
            if is_star:
                cons_before = self._consensus_distance(
                    params_list, list(range(self.n_total))
                )

            # --- Gossip (with policy on star rounds) ---
            if is_star:
                policy = self.star_policy.get("name", "single")
                if policy == "single":
                    params_half = _tree_mix(self.W_A, params_list)
                    K_used = 1
                    active_count = len(self.client_ids)
                elif policy == "repeat":
                    K_used = int(self.star_policy.get("K", 2))
                    params_half = repeat_mix(self.W_A, params_list, K_used)
                    active_count = len(self.client_ids)
                elif policy == "cheby":
                    K_used = int(self.star_policy.get("K", 3))
                    lam_min, lam_max = _ensure_star_interval()
                    params_half = chebyshev_mix(
                        self.W_A, params_list, K_used, lam_min, lam_max
                    )
                    active_count = len(self.client_ids)
                elif policy == "adaptive":
                    p_part = float(self.star_policy.get("p_participation", 0.5))
                    ensure_nonempty = bool(
                        self.star_policy.get("ensure_nonempty", True)
                    )
                    active_local, rng = sample_active_clients(
                        rng,
                        len(self.client_ids),
                        p_part,
                        ensure_nonempty=ensure_nonempty,
                    )
                    active_clients = [self.client_ids[i] for i in active_local]
                    active_count = len(active_clients)

                    if active_count >= 1:
                        edges_small = [(0, j + 1) for j in range(active_count)]
                        L_sub = laplacian_from_edges(1 + active_count, edges_small)
                        lam_min, lam_max = disagreement_interval_from_L(
                            L_sub, self.alpha_A
                        )
                        denom = float(lam_max - lam_min)
                    else:
                        lam_min, lam_max, denom = 1.0, 1.0, 0.0

                    rho_target = float(self.star_policy.get("rho_target", 0.05))
                    K_cap = int(self.star_policy.get("K_max", 5))
                    if denom <= 1e-12 or not np.isfinite(denom) or active_count <= 1:
                        K_used = 1
                        params_half = params_list
                    else:
                        xi = (2.0 - (lam_max + lam_min)) / denom
                        K_used = max(
                            1, min(K_cap, min_K_for_target_rho(float(xi), rho_target))
                        )
                        W_A_act = build_partial_star_W(
                            self.n_total, self.server_id, active_clients, self.alpha_A
                        )
                        params_half = chebyshev_mix(
                            W_A_act, params_list, K_used, lam_min, lam_max
                        )
                else:
                    raise ValueError(f"Unknown star_policy {policy}")

                # Per-star contraction: measure after star and log ratio
                cons_after = self._consensus_distance(
                    params_half, list(range(self.n_total))
                )
                rho_hat = float(cons_after / cons_before) if cons_before > 0 else 1.0
                hist_star_rho.append(rho_hat)
                hist_K.append(int(K_used))
                hist_active.append(int(active_count))
            else:
                params_half = _tree_mix(self.W_B, params_list)  # identity mixing

            models_half = [
                _combine_params(ph, s) for ph, s in zip(params_half, static_list)
            ]

            # --- Local update at clients (GD or SGD); server: no local step ---
            new_params_list = []
            gamma_t = self._gamma_t(t)
            for i in range(self.n_total):
                if i == self.server_id:
                    new_params_list.append(params_half[i])
                    continue

                Xi, yi = parts[i]
                # batch select
                if self.batch_schedule is None:
                    Xb, yb = Xi, yi
                else:
                    n_i = sizes[i]
                    B = int(self.batch_schedule(t, i, n_i))
                    B = max(1, min(B, n_i))
                    rng, sub = jr.split(rng)
                    idx = (
                        jnp.arange(n_i)
                        if B == n_i
                        else jr.choice(sub, n_i, shape=(B,), replace=False)
                    )
                    Xb, yb = Xi[idx], yi[idx]
                    total_samples[i] += int(B)

                rng, sub = jr.split(rng)
                grad_i, new_state_i = self._local_grad_step(
                    models_half[i], self.states[i], Xb, yb, sub
                )
                updated = jax.tree_util.tree_map(
                    lambda p, g: p - gamma_t * g, params_half[i], grad_i
                )
                new_params_list.append(updated)
                self.states[i] = new_state_i

            params_list = new_params_list

            # --- Logging: server loss, nodes 1 & 4 losses, consensus ---
            mS = _combine_params(
                params_list[self.server_id], static_list[self.server_id]
            )
            if self.eval_inference_mode:
                mS = eqx.nn.inference_mode(mS, value=True)
            rng, kS = jr.split(rng)
            lossS = float(
                eval_step(
                    mS, self.states[self.server_id], X_full, y_full, kS, self.loss_fn
                )
            )
            hist_loss_server.append(lossS)

            i1 = self.client_ids[0]
            i4 = self.client_ids[-1]
            m1 = _combine_params(params_list[i1], static_list[i1])
            m4 = _combine_params(params_list[i4], static_list[i4])
            if self.eval_inference_mode:
                m1 = eqx.nn.inference_mode(m1, value=True)
                m4 = eqx.nn.inference_mode(m4, value=True)
            rng, k1 = jr.split(rng)
            loss1 = float(
                eval_step(m1, self.states[i1], X_full, y_full, k1, self.loss_fn)
            )
            rng, k4 = jr.split(rng)
            loss4 = float(
                eval_step(m4, self.states[i4], X_full, y_full, k4, self.loss_fn)
            )
            hist_loss_node1.append(loss1)
            hist_loss_node4.append(loss4)

            hist_cons_all.append(
                self._consensus_distance(params_list, list(range(self.n_total)))
            )
            hist_cons_clients.append(
                self._consensus_distance(params_list, self.client_ids)
            )

            if log and (((t + 1) % log_interval == 0) or (t + 1 == self.T)):
                msg = (
                    f"[{t+1}/{self.T}] server loss={lossS:.4f}  "
                    f"cons_all={hist_cons_all[-1]:.3e}  cons_cli={hist_cons_clients[-1]:.3e}"
                )
                if is_star:
                    msg += f"  rho_star={hist_star_rho[-1]:.4f}  K={hist_K[-1]}  act={hist_active[-1]}"
                print(msg, flush=True)

        # finalize
        self.models = [_combine_params(p, s) for p, s in zip(params_list, static_list)]
        self.key = rng

        out = {
            "loss_server": hist_loss_server,
            "loss_node1": hist_loss_node1,
            "loss_node4": hist_loss_node4,
            "consensus_all": hist_cons_all,
            "consensus_clients": hist_cons_clients,
            "rho_star": hist_star_rho,
            "K_hist": hist_K,
            "active_hist": hist_active,
        }
        if total_samples is not None:
            out["total_samples_per_client"] = total_samples
        return out


# ---- simple LR schedules you can import right away ----


def make_constant_lr(gamma: float) -> Callable[[int], float]:
    """Return a constant LR schedule γ_t ≡ gamma."""
    g = float(gamma)

    def sched(t: int) -> float:
        return g

    return sched


def make_polynomial_decay(
    gamma0: float, power: float = 1.0, t0: float = 1.0
) -> Callable[[int], float]:
    """γ_t = gamma0 / (t + t0)^power  (default: 1/t decay)."""
    g0 = float(gamma0)
    p = float(power)
    tt = float(t0)

    def sched(t: int) -> float:
        return g0 / ((t + tt) ** p)

    return sched


if __name__ == "__main__":
    """
    Demo: synthetic logistic regression, n_clients=4 + 1 server.
    Compares GD vs SGD under periodic star mixing (τ=10).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    # ---- synthetic data ----
    def make_synth_logistic(n=8000, d=30, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, d).astype(np.float32)
        w_true = (rng.randn(d) / np.sqrt(d)).astype(np.float32)
        logits = X @ w_true
        p = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.rand(n) < p).astype(np.float32)
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
        X = (X - mu) / sd
        return jnp.asarray(X), jnp.asarray(y)

    def shard_clients_uniform(X, y, n_clients):
        N = int(X.shape[0])
        base, rem = divmod(N, n_clients)
        parts, start = [], 0
        for i in range(n_clients):
            sz = base + (1 if i < rem else 0)
            parts.append((X[start : start + sz], y[start : start + sz]))
            start += sz
        return parts

    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, d: int, key: jax.Array):
            self.lin = eqx.nn.Linear(d, 1, key=key)

        def __call__(self, x, key, state):
            return self.lin(x).squeeze(-1), state

    def model_init_fn(key: jax.Array) -> eqx.Module:
        return LR(30, key)  # d=30 as above

    # --- star topology ---
    n_clients = 4
    server_id = n_clients
    edges_A = make_star_with_server_edges(n_clients, server_id=server_id)
    n_total = n_clients + 1
    alpha_A = safe_alpha(edges_A, n_total)

    # --- data & parts (server has empty set) ---
    X, y = make_synth_logistic(n=8000, d=30, seed=0)
    client_parts = shard_clients_uniform(X, y, n_clients)
    server_part = (
        jnp.zeros((0, X.shape[1]), dtype=X.dtype),
        jnp.zeros((0,), dtype=y.dtype),
    )
    parts = client_parts + [server_part]

    # --- batch schedules ---
    def powerlaw(b0=16, p=0.7, bmax=256):
        b0 = int(max(1, b0))
        bmax = int(max(b0, bmax))
        p = float(p)

        def sched(t, i, n_i):
            import math

            b = int(math.ceil(b0 * ((t + 1) ** p)))
            return min(b, bmax, n_i)

        return sched

    # --- run GD (batch_schedule=None) ---
    T = 250
    trainer_gd = LocalGDServerEqx(
        model_init_fn=model_init_fn,
        n_clients=n_clients,
        tau=10,
        edges_A=edges_A,
        alpha_A=alpha_A,
        gamma=0.1,
        T=T,
        loss_fn=binary_loss,
        key=jr.PRNGKey(0),
        eval_inference_mode=True,
        server_id=server_id,
        star_policy={"name": "single"},
        batch_schedule=None,  # GD
    )
    hist_gd = trainer_gd.fit(parts, X, y, eval_key=jr.PRNGKey(123), log=False)
    print(f"[LocalGDServer GD] last server loss={hist_gd['loss_server'][-1]:.4f}")

    # --- run SGD (minibatch) ---
    trainer_sgd = LocalGDServerEqx(
        model_init_fn=model_init_fn,
        n_clients=n_clients,
        tau=10,
        edges_A=edges_A,
        alpha_A=alpha_A,
        gamma=lambda t: 0.05,
        T=T,
        loss_fn=binary_loss,
        key=jr.PRNGKey(1),
        eval_inference_mode=True,
        server_id=server_id,
        star_policy={"name": "repeat", "K": 2},  # example: repeat star mixing
        batch_schedule=powerlaw(b0=32, p=0.6, bmax=256),  # SGD
    )
    hist_sgd = trainer_sgd.fit(parts, X, y, eval_key=jr.PRNGKey(456), log=False)
    print(f"[LocalGDServer SGD] last server loss={hist_sgd['loss_server'][-1]:.4f}")
    if "total_samples_per_client" in hist_sgd:
        print(
            "[LocalGDServer SGD] total samples per client:",
            hist_sgd["total_samples_per_client"],
        )

    # --- plots ---
    plt.figure(figsize=(7, 4))
    plt.plot(hist_gd["loss_server"], label="GD")
    plt.plot(hist_sgd["loss_server"], label="SGD")
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("server global loss")
    plt.title("LocalGDServerEqx: GD vs SGD (τ=10)")
    plt.legend()
    plt.tight_layout()
    plt.show()
