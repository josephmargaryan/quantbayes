# peer_gossip_trainer_eqx.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, Any
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.trainer.train import binary_loss, eval_step
from quantbayes.stochax.privacy.dp import DPSGDConfig, DPPrivacyEngine
from quantbayes.stochax.distributed_training.mixing_policies import (
    chebyshev_mix,
    disagreement_interval_from_L,
    min_K_for_target_rho,
)

Array = jnp.ndarray
PRNG = jax.Array
BatchSchedule = Callable[[int, int, int], int]  # B = schedule(t, node, n_i)


# ---------------- graph helpers ----------------
def laplacian_from_edges(n_nodes: int, edges: List[Tuple[int, int]]) -> Array:
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i, j in edges:
        if i == j:
            continue
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.diag(A.sum(axis=1))
    L = D - A
    return jnp.array(L)


def mixing_matrix(n_nodes: int, edges: List[Tuple[int, int]], alpha: float) -> Array:
    L = laplacian_from_edges(n_nodes, edges)
    I = jnp.eye(n_nodes, dtype=jnp.float32)
    return I - float(alpha) * L


def safe_alpha(edges: List[Tuple[int, int]], n_nodes: int) -> float:
    deg = np.zeros(n_nodes, dtype=np.int32)
    for i, j in edges:
        if i != j:
            deg[i] += 1
            deg[j] += 1
    deg_max = int(deg.max()) if n_nodes > 0 else 1
    return 0.49 / max(1, deg_max)


# --------------- pytree helpers ----------------
def _partition_params(model: eqx.Module):
    return eqx.partition(model, eqx.is_inexact_array)


def _combine_params(params: Any, static: Any) -> eqx.Module:
    return eqx.combine(params, static)


def _tree_weighted_sum(weights: Array, param_list: List[Any]) -> Any:
    def combine(*leaves):
        stacked = jnp.stack(leaves, axis=0)
        return jnp.tensordot(weights, stacked, axes=(0, 0))

    return jax.tree_util.tree_map(combine, *param_list)


def _tree_mix(W: Array, param_list: List[Any]) -> List[Any]:
    return [_tree_weighted_sum(W[i], param_list) for i in range(len(param_list))]


def _powerK_mix(W: Array, param_list: List[Any], K: int) -> List[Any]:
    out = param_list
    for _ in range(max(1, int(K))):
        out = _tree_mix(W, out)
    return out


# --------------- class ----------------
class PeerGossipTrainerEqx:
    """
    Unified P2P D(G)D/DSGD on undirected graphs with DS mixing W = I - αL.

      θ_{t+1/2} = M_t(θ_t)         # mixing policy (single, powerK, cheby, adaptive_cheby)
      θ_{t+1}^i = θ_{t+1/2}^i - γ_t * g_i(.; batch)

    Graph modes:
      - fixed: one graph (edges_fixed)
      - odd_even: alternating edges_odd / edges_even

    Mix policies (set via mix_policy dict):
      - {"name": "single"}
      - {"name": "powerK", "K": int}
      - {"name": "cheby",  "K": int}
      - {"name": "adaptive_cheby",
         "p_participation": float in (0,1],
         "rho_target": float in (0,1),
         "K_max": int,
         "ensure_nonempty": bool}
        → per-step partial participation, K_t chosen from active subgraph spectrum,
          embedded as block-diagonal mixing (active block W_small, identity on inactive).

    GD vs SGD:
      - batch_schedule is None → full-batch GD
      - batch_schedule is callable → mini-batch DSGD

    Logs:
      - "loss_node1": global loss evaluated at node 0 each iteration
      - "consensus_sq": 1/n ∑ ||θ_i - θ̄||² each iteration
      - "rho_hist": consensus ratio after/before mixing each iteration
      - "K_hist": degree used (1 for single, K for others)
      - "active_hist": #active nodes (adaptive_cheby); else n_nodes
      - "total_samples_per_node": for DSGD
    """

    def __init__(
        self,
        model_init_fn: Callable[[PRNG], eqx.Module],
        n_nodes: int,
        *,
        edges_fixed: Optional[List[Tuple[int, int]]] = None,
        edges_odd: Optional[List[Tuple[int, int]]] = None,
        edges_even: Optional[List[Tuple[int, int]]] = None,
        alpha_fixed: Optional[float] = None,
        alpha_odd: Optional[float] = None,
        alpha_even: Optional[float] = None,
        mix_policy: Optional[Dict[str, Any]] = None,
        gamma: float | Callable[[int], float] = 0.1,
        T: int = 400,
        loss_fn: Callable = binary_loss,
        batch_schedule: BatchSchedule | None = None,
        key: Optional[PRNG] = None,
        eval_inference_mode: bool = True,
        dp_config: Optional[DPSGDConfig] = None,
    ):
        self.model_init_fn = model_init_fn
        self.n_nodes = int(n_nodes)
        self.gamma = gamma
        self.T = int(T)
        self.loss_fn = loss_fn
        self.batch_schedule = batch_schedule
        self.key = jr.PRNGKey(0) if key is None else key
        self.eval_inference_mode = bool(eval_inference_mode)
        self.mix_policy = mix_policy or {"name": "single"}

        # Graph mode
        self.graph_mode = "fixed" if edges_fixed is not None else "odd_even"
        if self.graph_mode == "fixed":
            assert edges_fixed is not None
            self.edges_fixed = edges_fixed
            self.alpha_fixed = (
                float(alpha_fixed)
                if alpha_fixed is not None
                else safe_alpha(edges_fixed, n_nodes)
            )
            self.L_fixed = laplacian_from_edges(n_nodes, edges_fixed)
            self.W_fixed = mixing_matrix(n_nodes, edges_fixed, self.alpha_fixed)
            # cache disagreement interval for cheby
            self._lam_interval_fixed = disagreement_interval_from_L(
                self.L_fixed, self.alpha_fixed
            )
        else:
            assert edges_odd is not None and edges_even is not None
            self.edges_odd = edges_odd
            self.edges_even = edges_even
            self.alpha_odd = (
                float(alpha_odd)
                if alpha_odd is not None
                else safe_alpha(edges_odd, n_nodes)
            )
            self.alpha_even = (
                float(alpha_even)
                if alpha_even is not None
                else safe_alpha(edges_even, n_nodes)
            )
            self.L_odd = laplacian_from_edges(n_nodes, edges_odd)
            self.L_even = laplacian_from_edges(n_nodes, edges_even)
            self.W_odd = mixing_matrix(n_nodes, edges_odd, self.alpha_odd)
            self.W_even = mixing_matrix(n_nodes, edges_even, self.alpha_even)
            self._lam_interval_odd = disagreement_interval_from_L(
                self.L_odd, self.alpha_odd
            )
            self._lam_interval_even = disagreement_interval_from_L(
                self.L_even, self.alpha_even
            )

        self.dp_config = dp_config
        self._dp_engine = DPPrivacyEngine(dp_config) if dp_config else None

        self.models: List[eqx.Module] = []
        self.states: List[Any] = []

    # -------- utilities --------
    def _init_models(self):
        models, states = [], []
        k = self.key
        for _ in range(self.n_nodes):
            k, sub = jr.split(k)
            m, s = eqx.nn.make_with_state(self.model_init_fn)(sub)
            models.append(m)
            states.append(s)
        self.key = k
        self.models = models
        self.states = states

    def _gamma_t(self, t: int) -> float:
        return float(self.gamma(t)) if callable(self.gamma) else float(self.gamma)

    def _consensus_distance(self, params_list: List[Any]) -> float:
        def flatten(pytree: Any) -> Array:
            leaves = jax.tree_util.tree_leaves(pytree)
            flat = [jnp.ravel(x) for x in leaves if x is not None]
            return jnp.concatenate(flat) if flat else jnp.zeros((1,), dtype=jnp.float32)

        vecs = [flatten(p) for p in params_list]
        stack = jnp.stack(vecs, axis=0)
        mean = jnp.mean(stack, axis=0, keepdims=True)
        sq = jnp.sum((stack - mean) ** 2, axis=1)
        return float(jnp.mean(sq))

    def _sample_active_nodes(
        self, rng: PRNG, n_nodes: int, p: float, ensure_nonempty: bool = True
    ) -> Tuple[List[int], PRNG]:
        rng, sub = jr.split(rng)
        if p >= 1.0:
            return list(range(n_nodes)), rng
        mask = jr.uniform(sub, (n_nodes,)) < float(p)
        active = [i for i in range(n_nodes) if bool(mask[i])]
        if ensure_nonempty and not active:
            rng, sub2 = jr.split(rng)
            fallback = int(jr.randint(sub2, shape=(), minval=0, maxval=n_nodes))
            active = [fallback]
        return active, rng

    def _build_partial_W_and_interval(
        self,
        base_edges: List[Tuple[int, int]],
        alpha: float,
        active_idx: List[int],
    ) -> Tuple[Array, Tuple[float, float]]:
        """
        Build a full-size W that mixes only the active induced subgraph by W_small = I - α L_small,
        and identity elsewhere. Also return the disagreement interval for W_small.
        """
        n = self.n_nodes
        # Map active nodes to 0..m-1
        idx_map = {u: i for i, u in enumerate(active_idx)}
        # Induced edge list on active nodes (undirected, no self loops)
        edges_sub = [
            (idx_map[i], idx_map[j])
            for (i, j) in base_edges
            if i in idx_map and j in idx_map and i != j
        ]

        m = len(active_idx)
        if m <= 1 or len(edges_sub) == 0:
            # Degenerate: no mixing possible
            W_full = jnp.eye(n, dtype=jnp.float32)
            # Choose any valid interval; we will force K=1 upstream in this case
            return W_full, (1.0, 1.0)

        # Small Laplacian and mixing
        L_small = laplacian_from_edges(m, edges_sub)
        lam_min, lam_max = disagreement_interval_from_L(L_small, alpha)
        W_small = jnp.eye(m, dtype=jnp.float32) - float(alpha) * L_small

        # Embed into full W
        W_full = jnp.eye(n, dtype=jnp.float32)
        for i, u in enumerate(active_idx):
            for j, v in enumerate(active_idx):
                W_full = W_full.at[u, v].set(W_small[i, j])

        return W_full, (lam_min, lam_max)

    # -------- training loop --------
    def fit(
        self,
        parts: List[Tuple[Array, Array]],
        X_full: Array,
        y_full: Array,
        *,
        eval_key: Optional[PRNG] = None,
    ) -> Dict[str, Any]:
        if not self.models:
            self._init_models()

        params_list, static_list = [], []
        for m in self.models:
            p, s = _partition_params(m)
            params_list.append(p)
            static_list.append(s)

        rng = self.key if eval_key is None else eval_key
        sizes = [int(parts[i][0].shape[0]) for i in range(self.n_nodes)]
        total_samples = (
            {i: 0 for i in range(self.n_nodes)} if self.batch_schedule else None
        )

        loss_hist, cons_hist = [], []
        rho_hist, K_hist, active_hist = [], [], []

        for t in range(self.T):
            # choose base graph at t
            if self.graph_mode == "fixed":
                W_base = self.W_fixed
                L_interval = self._lam_interval_fixed
                edges_t = self.edges_fixed
                alpha_t = self.alpha_fixed
            else:
                if (t + 1) % 2 == 1:
                    W_base = self.W_odd
                    L_interval = self._lam_interval_odd
                    edges_t = self.edges_odd
                    alpha_t = self.alpha_odd
                else:
                    W_base = self.W_even
                    L_interval = self._lam_interval_even
                    edges_t = self.edges_even
                    alpha_t = self.alpha_even

            # consensus before mixing (for rho)
            cons_before = self._consensus_distance(params_list)

            # mixing policy
            policy = self.mix_policy.get("name", "single")
            if policy == "single":
                params_half = _tree_mix(W_base, params_list)
                K_used = 1
                active_count = self.n_nodes
            elif policy == "powerK":
                K_used = int(max(1, self.mix_policy.get("K", 2)))
                params_half = _powerK_mix(W_base, params_list, K_used)
                active_count = self.n_nodes
            elif policy == "cheby":
                K_used = int(max(1, self.mix_policy.get("K", 2)))
                lam_min, lam_max = L_interval
                params_half = chebyshev_mix(
                    W_base, params_list, K_used, lam_min, lam_max
                )
                active_count = self.n_nodes
            elif policy == "adaptive_cheby":
                p_part = float(self.mix_policy.get("p_participation", 0.5))
                ensure_nonempty = bool(self.mix_policy.get("ensure_nonempty", True))
                active_idx, rng = self._sample_active_nodes(
                    rng, self.n_nodes, p_part, ensure_nonempty
                )
                active_count = len(active_idx)

                # build partial W and spectrum for the active induced subgraph
                W_partial, (lam_min, lam_max) = self._build_partial_W_and_interval(
                    edges_t, alpha_t, active_idx
                )
                denom = float(lam_max - lam_min)
                rho_target = float(self.mix_policy.get("rho_target", 0.05))
                K_cap = int(self.mix_policy.get("K_max", 5))

                if active_count <= 1 or not np.isfinite(denom) or denom <= 1e-12:
                    K_used = 1
                    params_half = params_list  # no mixing possible
                else:
                    xi = (2.0 - (lam_max + lam_min)) / denom
                    K_used = max(
                        1, min(K_cap, min_K_for_target_rho(float(xi), rho_target))
                    )
                    params_half = chebyshev_mix(
                        W_partial, params_list, K_used, lam_min, lam_max
                    )
            else:
                raise ValueError(f"Unknown mix_policy '{policy}'")

            # consensus after mixing & ratio
            cons_after = self._consensus_distance(params_half)
            rho_hat = float(cons_after / (cons_before + 1e-12))
            rho_hist.append(rho_hat)
            K_hist.append(int(K_used))
            active_hist.append(int(active_count))

            # local updates
            step = self._gamma_t(t)
            new_params = []
            for i in range(self.n_nodes):
                Xi, yi = parts[i]
                n_i = sizes[i]
                # batch selection
                if self.batch_schedule is None:
                    Xb, yb = Xi, yi
                else:
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

                def loss_local(m, s, xb, yb, k):
                    return self.loss_fn(m, s, xb, yb, k)

                rng, sub = jr.split(rng)
                if self._dp_engine is None:
                    (_, new_state_i), grad_i = eqx.filter_value_and_grad(
                        loss_local, has_aux=True
                    )(
                        _combine_params(params_half[i], static_list[i]),
                        self.states[i],
                        Xb,
                        yb,
                        sub,
                    )
                else:
                    grad_i, new_state_i = self._dp_engine.noisy_grad(
                        loss_local,
                        _combine_params(params_half[i], static_list[i]),
                        self.states[i],
                        Xb,
                        yb,
                        key=sub,
                    )

                upd = jax.tree_util.tree_map(
                    lambda p, g: p - step * g, params_half[i], grad_i
                )
                new_params.append(upd)
                self.states[i] = new_state_i

            params_list = new_params

            # log global loss at node 0 and consensus
            m0 = _combine_params(params_list[0], static_list[0])
            if self.eval_inference_mode:
                m0 = eqx.nn.inference_mode(m0, value=True)
            rng, k0 = jr.split(rng)
            loss0 = float(
                eval_step(m0, self.states[0], X_full, y_full, k0, self.loss_fn)
            )
            loss_hist.append(loss0)
            cons_hist.append(self._consensus_distance(params_list))

        # finalize
        self.models = [_combine_params(p, s) for p, s in zip(params_list, static_list)]
        self.key = rng

        out: Dict[str, Any] = {
            "loss_node1": loss_hist,
            "consensus_sq": cons_hist,
            "rho_hist": rho_hist,
            "K_hist": K_hist,
            "active_hist": active_hist,
        }
        if total_samples is not None:
            out["total_samples_per_node"] = total_samples
        return out


# ------------------- demo -------------------
if __name__ == "__main__":
    """
    Demo for PeerGossipTrainerEqx:
      • fixed ring vs odd/even switching
      • GD (full-batch) vs SGD (minibatch)
      • mix policy 'single' vs 'powerK' vs 'cheby' vs 'adaptive_cheby'
    """
    import matplotlib.pyplot as plt

    # --- synthetic logistic data ---
    def make_synth_logistic(n=5000, d=32, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, d).astype(np.float32)
        w_true = (rng.randn(d) / np.sqrt(d)).astype(np.float32)
        logits = X @ w_true
        p = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.rand(n) < p).astype(np.float32)
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
        X = (X - mu) / sd
        return jnp.asarray(X), jnp.asarray(y)

    def uniform_partition(X, y, n_nodes=4):
        N = int(X.shape[0])
        base, rem = divmod(N, n_nodes)
        parts, start = [], 0
        for i in range(n_nodes):
            sz = base + (1 if i < rem else 0)
            parts.append((X[start : start + sz], y[start : start + sz]))
            start += sz
        return parts

    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, d: int, key: jax.Array):
            self.lin = eqx.nn.Linear(d, 1, key=key)

        def __call__(self, x, key=None, state=None):
            return self.lin(x).squeeze(-1), state

    # --- graphs ---
    edges_fixed = [(0, 1), (1, 2), (2, 3), (3, 0)]  # ring
    edges_odd = edges_fixed
    edges_even = [(0, 1), (2, 3)]  # two disjoint edges

    # --- data ---
    X, y = make_synth_logistic(n=5000, d=32, seed=0)
    parts = uniform_partition(X, y, n_nodes=4)

    T = 200

    # Fixed graph, GD, single vs powerK vs cheby vs adaptive_cheby
    trainer_single = PeerGossipTrainerEqx(
        model_init_fn=lambda k: LR(32, k),
        n_nodes=4,
        edges_fixed=edges_fixed,
        mix_policy={"name": "single"},
        gamma=0.1,
        T=T,
        batch_schedule=None,  # GD
        key=jr.PRNGKey(0),
    )
    out_single = trainer_single.fit(parts, X, y)

    trainer_power = PeerGossipTrainerEqx(
        model_init_fn=lambda k: LR(32, k),
        n_nodes=4,
        edges_fixed=edges_fixed,
        mix_policy={"name": "powerK", "K": 2},
        gamma=0.1,
        T=T,
        batch_schedule=None,
        key=jr.PRNGKey(1),
    )
    out_power = trainer_power.fit(parts, X, y)

    trainer_cheby = PeerGossipTrainerEqx(
        model_init_fn=lambda k: LR(32, k),
        n_nodes=4,
        edges_fixed=edges_fixed,
        mix_policy={"name": "cheby", "K": 2},
        gamma=0.1,
        T=T,
        batch_schedule=None,
        key=jr.PRNGKey(2),
    )
    out_cheby = trainer_cheby.fit(parts, X, y)

    trainer_adapt = PeerGossipTrainerEqx(
        model_init_fn=lambda k: LR(32, k),
        n_nodes=4,
        edges_fixed=edges_fixed,
        mix_policy={
            "name": "adaptive_cheby",
            "p_participation": 0.5,
            "rho_target": 0.05,
            "K_max": 5,
            "ensure_nonempty": True,
        },
        gamma=0.1,
        T=T,
        batch_schedule=None,
        key=jr.PRNGKey(3),
    )
    out_adapt = trainer_adapt.fit(parts, X, y)

    # --- plots ---
    plt.figure(figsize=(7.6, 4.4))
    plt.plot(out_single["loss_node1"], label="single")
    plt.plot(out_power["loss_node1"], label="powerK=2")
    plt.plot(out_cheby["loss_node1"], label="cheby K=2")
    plt.plot(out_adapt["loss_node1"], label="adaptive_cheby")
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("global loss @ node 0")
    plt.title("PeerGossipTrainerEqx (fixed ring)")
    plt.legend()
    plt.tight_layout()
    plt.show()
