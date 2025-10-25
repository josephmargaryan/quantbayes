# quantbayes/stochax/distributed_training/peer_gossip_trainer_eqx.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, Any
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.trainer.train import binary_loss, eval_step
from quantbayes.stochax.privacy.dp import DPSGDConfig, DPPrivacyEngine

# === core helpers (do not redefine them here) ===
from quantbayes.stochax.distributed_training.helpers import (
    laplacian_from_edges,
    mixing_matrix,
    safe_alpha,
    _partition_params,
    _combine_params,
    tree_mix,
    tree_mean,
    is_weight_array,
    weights_only_l2_penalty,
    ring_edges,
    load_mnist_38,
    make_hetero_3v8_parts_no_server,
    estimate_gamma_logistic,
    make_polynomial_decay,
    plot_global_loss_q3,
    plot_consensus,
    summarize_histories,
    print_publication_summary,
    latex_table_from_summary,
)

# === spectral / adaptive utilities ===
from quantbayes.stochax.distributed_training.mixing_policies import (
    repeat_mix,
    chebyshev_mix,
    disagreement_interval_from_L,
    AdaptiveChebyController,  # GLOBAL-aware
    min_K_for_target_rho,
)

Array = jnp.ndarray
PRNG = jax.Array
BatchSchedule = Callable[[int, int, int], int]  # B = schedule(t, node, n_i)


# -------------------------------------------------------------------------
# Small local utilities (specific to this trainer)
# -------------------------------------------------------------------------
def _flatten_params_l2(pytree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(pytree)
    flat = [jnp.ravel(x) for x in leaves if x is not None]
    return jnp.concatenate(flat) if flat else jnp.zeros((1,), dtype=jnp.float32)


def _consensus_distance(params_list: List[Any]) -> float:
    vecs = [_flatten_params_l2(p) for p in params_list]
    stack = jnp.stack(vecs, axis=0)
    mean = jnp.mean(stack, axis=0, keepdims=True)
    sq = jnp.sum((stack - mean) ** 2, axis=1)
    return float(jnp.mean(sq))


def _is_connected(m: int, edges_sub: List[Tuple[int, int]]) -> bool:
    """Connectivity guard for the active induced subgraph (P2P adaptive)."""
    if m <= 1:
        return False
    g = {i: set() for i in range(m)}
    for u, v in edges_sub:
        if u == v:
            continue
        g[u].add(v)
        g[v].add(u)
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in g[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == m


# -------------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------------
class PeerGossipTrainerEqx:
    """
    Unified P2P D(G)D/DSGD/GT on undirected graphs with DS mixing W = I - αL.

    Mix policies (set via mix_policy dict):
      - {"name": "single"}
      - {"name": "powerK", "K": int}
      - {"name": "cheby",  "K": int}
      - {"name": "adaptive_cheby",
         "p_participation": float in (0,1],
         "rho_target": float in (0,1),            # GLOBAL target on Γ (squared ratio)
         "K_max": int,
         "ensure_nonempty": bool,
         "adapt_p_on_sat": bool,                  # optional: autotune p
         "p_cooldown": int,                       # stars between p changes
         "p_min": float, "p_max": float           # clamps
        }

      - {"name": "gt", "kernel": "single"|"cheby", "K": int (if cheby)}
        Gradient tracking (GT) with mixing every 'mix_every' steps:
          x_{t+1} = M_t x_t - γ y_t
          y_{t+1} = M_t y_t + ∇f(x_{t+1}) - ∇f(x_t)
        where M_t is I off-mix steps and W or p_K(W) on mix steps.

    Logs:
      - "loss_node1": global loss at node 0 each iteration
      - "consensus_sq": 1/n ∑ ||θ_i - θ̄||² each iteration
      - "rho_hist": Γ_after / Γ_before at mix events (1 otherwise)
      - "K_hist": degree used (0 if no mix applied on that event)
      - "active_hist": #active nodes on mix events (adaptive); else n_nodes or 0
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
        lam_l2: Optional[float] = None,
        weight_decay: float = 0.0,
        mix_every: int = 1,
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
        self.lam_l2 = float(lam_l2) if lam_l2 is not None else 0.0
        self.weight_decay = float(weight_decay)
        self.mix_every = int(max(1, mix_every))
        assert not (
            self.lam_l2 > 0.0 and self.weight_decay > 0.0
        ), "Choose either lam_l2 (loss-level L2) OR weight_decay (decoupled), not both."

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

        # GLOBAL-aware controller for adaptive_cheby
        self._cheby_ctl = None
        if self.mix_policy.get("name") == "adaptive_cheby":
            self._cheby_ctl = AdaptiveChebyController(
                rho_target_sq_global=float(self.mix_policy.get("rho_target", 0.20)),
                K_max=int(self.mix_policy.get("K_max", 5)),
            )
        # cooldown for p auto-tuning
        self._p_cooldown_p2p: int = 0

    # -------- internal utilities --------
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

    def _loss_with_reg(self, m, s, xb, yb, k):
        base, new_s = self.loss_fn(m, s, xb, yb, k)
        if self.lam_l2 > 0.0:
            base = base + self.lam_l2 * weights_only_l2_penalty(
                m, lam=self.lam_l2
            ) / max(
                self.lam_l2, 1e-12
            )  # reuse helper signature
        return base, new_s

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

    def _build_partial_W_edges_interval(
        self,
        base_edges: List[Tuple[int, int]],
        alpha: float,
        active_idx: List[int],
    ) -> Tuple[Array, Tuple[float, float], List[Tuple[int, int]]]:
        """
        Full-size W that mixes only the active induced subgraph by W_small = I - α L_small,
        identity elsewhere. Also return the disagreement interval for W_small and the subgraph edges.
        """
        n = self.n_nodes
        idx_map = {u: i for i, u in enumerate(active_idx)}
        edges_sub = [
            (idx_map[i], idx_map[j])
            for (i, j) in base_edges
            if i in idx_map and j in idx_map and i != j
        ]
        m = len(active_idx)
        if m <= 1 or len(edges_sub) == 0:
            W_full = jnp.eye(n, dtype=jnp.float32)
            return W_full, (1.0, 1.0), edges_sub
        L_small = laplacian_from_edges(m, edges_sub)
        lam_min, lam_max = disagreement_interval_from_L(L_small, alpha)
        W_small = jnp.eye(m, dtype=jnp.float32) - float(alpha) * L_small
        W_full = jnp.eye(n, dtype=jnp.float32)
        for i, u in enumerate(active_idx):
            for j, v in enumerate(active_idx):
                W_full = W_full.at[u, v].set(W_small[i, j])
        return W_full, (lam_min, lam_max), edges_sub

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

        # --- GT state (allocated when/if needed) ---
        policy = self.mix_policy.get("name", "single")
        use_gt = policy == "gt"
        y_list: Optional[List[Any]] = None
        g_prev: Optional[List[Any]] = None
        if use_gt:
            y_list, g_prev = [], []
            # initial gradients at x^0
            for i in range(self.n_nodes):
                Xi, yi = parts[i]
                n_i = sizes[i]
                if self.batch_schedule is None:
                    Xb, yb = Xi, yi
                else:
                    B = int(self.batch_schedule(0, i, n_i))
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
                grad_i0, _ = self._gt_local_grad_only(
                    _combine_params(params_list[i], static_list[i]),
                    self.states[i],
                    Xb,
                    yb,
                    sub,
                )
                y_list.append(grad_i0)
                g_prev.append(grad_i0)

        for t in range(self.T):
            # choose current base graph
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

            do_mix = (t + 1) % self.mix_every == 0
            cons_before = _consensus_distance(params_list)

            # ------------------------------------------------------------------
            # NON-GT policies (single/powerK/cheby/adaptive_cheby)
            # ------------------------------------------------------------------
            if policy in ("single", "powerK", "cheby", "adaptive_cheby"):
                if do_mix:
                    if policy == "single":
                        params_half = tree_mix(W_base, params_list)
                        K_used = 1
                        active_count = self.n_nodes

                    elif policy == "powerK":
                        K_used = int(max(1, self.mix_policy.get("K", 2)))
                        params_half = repeat_mix(W_base, params_list, K_used)
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
                        ensure_nonempty = bool(
                            self.mix_policy.get("ensure_nonempty", True)
                        )
                        p_min = float(self.mix_policy.get("p_min", 0.2))
                        p_max = float(self.mix_policy.get("p_max", 1.0))
                        p_part = float(np.clip(p_part, p_min, p_max))

                        active_idx, rng = self._sample_active_nodes(
                            rng, self.n_nodes, p_part, ensure_nonempty
                        )
                        active_count = len(active_idx)

                        W_partial, (lam_min, lam_max), edges_sub = (
                            self._build_partial_W_edges_interval(
                                edges_t, alpha_t, active_idx
                            )
                        )
                        denom = float(lam_max - lam_min)
                        connected = _is_connected(len(active_idx), edges_sub)

                        if (
                            (active_count <= 1)
                            or (not connected)
                            or (not np.isfinite(denom))
                            or (denom <= 1e-12)
                        ):
                            # no reliable spectral design possible this round → skip mixing
                            params_half = params_list
                            K_used = 0  # mark "no mix applied"
                        else:
                            xi = (2.0 - (lam_max + lam_min)) / denom
                            if self._cheby_ctl is not None:
                                K_req = self._cheby_ctl.choose_K_global(
                                    float(xi),
                                    gamma_pre=cons_before,
                                    active=active_count,
                                    total=self.n_nodes,
                                )
                            else:
                                K_req = min_K_for_target_rho(
                                    float(xi),
                                    float(self.mix_policy.get("rho_target", 0.20)),
                                )
                            K_cap = int(self.mix_policy.get("K_max", 5))
                            K_used = int(max(1, min(K_cap, K_req)))
                            params_half = chebyshev_mix(
                                W_partial, params_list, K_used, lam_min, lam_max
                            )

                    # measure contraction due to mixing
                    cons_after_mix = _consensus_distance(params_half)
                    rho_hat = float(cons_after_mix / (cons_before + 1e-12))
                    rho_hist.append(rho_hat)
                    K_hist.append(int(K_used))
                    active_hist.append(int(active_count))

                    # optional p auto-tuning with cooldown (only if adaptive_cheby)
                    if (
                        policy == "adaptive_cheby"
                        and self._cheby_ctl is not None
                        and bool(self.mix_policy.get("adapt_p_on_sat", False))
                    ):
                        target = float(self._cheby_ctl.rho_target_sq_global)
                        p_cool = int(self.mix_policy.get("p_cooldown", 3))
                        if self._p_cooldown_p2p > 0:
                            self._p_cooldown_p2p -= 1
                        else:
                            if (rho_hat > 1.5 * target) and (
                                K_used >= self._cheby_ctl.K_max
                            ):
                                self.mix_policy["p_participation"] = float(
                                    np.clip(1.25 * p_part, p_min, p_max)
                                )
                                self._p_cooldown_p2p = p_cool
                            elif (rho_hat < 0.4 * target) and (K_used >= 1):
                                self.mix_policy["p_participation"] = float(
                                    np.clip(0.8 * p_part, p_min, p_max)
                                )
                                self._p_cooldown_p2p = p_cool

                    if self._cheby_ctl is not None and policy == "adaptive_cheby":
                        self._cheby_ctl.update_after_mix(
                            rho_hat, K_used, cons_after_mix
                        )

                else:
                    # no communication
                    params_half = params_list
                    rho_hist.append(1.0)
                    K_hist.append(0)
                    active_hist.append(0)

                # local updates
                step = self._gamma_t(t)
                new_params = []
                rng_loop = rng
                for i in range(self.n_nodes):
                    Xi, yi = parts[i]
                    n_i = sizes[i]
                    # batch selection
                    if self.batch_schedule is None:
                        Xb, yb = Xi, yi
                    else:
                        B = int(self.batch_schedule(t, i, n_i))
                        B = max(1, min(B, n_i))
                        rng_loop, sub = jr.split(rng_loop)
                        idx = (
                            jnp.arange(n_i)
                            if B == n_i
                            else jr.choice(sub, n_i, shape=(B,), replace=False)
                        )
                        Xb, yb = Xi[idx], yi[idx]
                        total_samples[i] += int(B)

                    def loss_local(m, s, xb, yb, k):
                        return self._loss_with_reg(m, s, xb, yb, k)

                    rng_loop, sub = jr.split(rng_loop)
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

                    # optional decoupled WD
                    if self.weight_decay > 0.0:
                        decay = 1.0 - step * self.weight_decay
                        params_half_i = jax.tree_util.tree_map(
                            lambda p: p * decay if is_weight_array(p) else p,
                            params_half[i],
                        )
                    else:
                        params_half_i = params_half[i]

                    updated = jax.tree_util.tree_map(
                        lambda p, g: p - step * g, params_half_i, grad_i
                    )
                    new_params.append(updated)
                    self.states[i] = new_state_i

                rng = rng_loop
                params_list = new_params

            # ------------------------------------------------------------------
            # GT policy
            # ------------------------------------------------------------------
            elif policy == "gt":
                kernel = str(self.mix_policy.get("kernel", "single")).lower()
                K_fixed = int(self.mix_policy.get("K", 2)) if kernel == "cheby" else 1

                # measure rho from mixing on x (pre-update)
                if do_mix:
                    if kernel == "single":
                        x_half = tree_mix(W_base, params_list)
                        K_used = 1
                    elif kernel == "cheby":
                        lam_min, lam_max = L_interval
                        x_half = chebyshev_mix(
                            W_base, params_list, K_fixed, lam_min, lam_max
                        )
                        K_used = K_fixed
                    else:
                        raise ValueError(f"Unknown GT kernel '{kernel}'")
                else:
                    x_half = params_list
                    K_used = 0

                cons_after_mix = _consensus_distance(x_half)
                rho_hat = float(cons_after_mix / (cons_before + 1e-12))
                rho_hist.append(rho_hat)
                K_hist.append(int(K_used))
                active_hist.append(self.n_nodes if do_mix else 0)

                # reuse one batch for g_t and g_{t+1}
                batches = []
                rng_loop = rng
                for i in range(self.n_nodes):
                    Xi, yi = parts[i]
                    n_i = sizes[i]
                    if self.batch_schedule is None:
                        Xb, yb = Xi, yi
                    else:
                        B = int(self.batch_schedule(t, i, n_i))
                        B = max(1, min(B, n_i))
                        rng_loop, sub = jr.split(rng_loop)
                        idx = (
                            jnp.arange(n_i)
                            if B == n_i
                            else jr.choice(sub, n_i, shape=(B,), replace=False)
                        )
                        Xb, yb = Xi[idx], yi[idx]
                        total_samples[i] += int(B)
                    batches.append((Xb, yb))

                # x_{t+1} = M_t x_t - γ y_t  (WD optional)
                step = self._gamma_t(t)
                x_new = []
                for i in range(self.n_nodes):
                    if self.weight_decay > 0.0:
                        decay = 1.0 - step * self.weight_decay
                        x_half_i = jax.tree_util.tree_map(
                            lambda p: p * decay if is_weight_array(p) else p, x_half[i]
                        )
                    else:
                        x_half_i = x_half[i]
                    x_new_i = jax.tree_util.tree_map(
                        lambda p, y: p - step * y, x_half_i, y_list[i]
                    )
                    x_new.append(x_new_i)

                # g_{t+1}
                g_next = []
                for i in range(self.n_nodes):
                    Xb, yb = batches[i]
                    rng_loop, sub = jr.split(rng_loop)
                    grad_i_next, new_state_i = self._gt_local_grad_only(
                        _combine_params(x_new[i], static_list[i]),
                        self.states[i],
                        Xb,
                        yb,
                        sub,
                    )
                    g_next.append(grad_i_next)
                    self.states[i] = new_state_i

                # y mixing on mix steps
                if do_mix:
                    if kernel == "single":
                        y_mix = tree_mix(W_base, y_list)
                    else:
                        lam_min, lam_max = L_interval
                        y_mix = chebyshev_mix(W_base, y_list, K_fixed, lam_min, lam_max)
                else:
                    y_mix = y_list

                # y_{t+1} = y_mix + g_{t+1} - g_t
                y_new = []
                for i in range(self.n_nodes):
                    y_new_i = jax.tree_util.tree_map(
                        lambda ym, gn, gp: ym + (gn - gp),
                        y_mix[i],
                        g_next[i],
                        g_prev[i],
                    )
                    y_new.append(y_new_i)

                params_list = x_new
                y_list = y_new
                g_prev = g_next
                rng = rng_loop

            else:
                raise ValueError(f"Unknown mix_policy '{policy}'")

            # --- logging: loss at node 0 + consensus ---
            m0 = _combine_params(params_list[0], static_list[0])
            if self.eval_inference_mode:
                m0 = eqx.nn.inference_mode(m0, value=True)
            rng, k0 = jr.split(rng)
            loss0 = float(
                eval_step(m0, self.states[0], X_full, y_full, k0, self.loss_fn)
            )
            loss_hist.append(loss0)
            cons_hist.append(_consensus_distance(params_list))

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

    # --- GT grad-only (DP-aware) ---
    def _gt_local_grad_only(
        self, model: eqx.Module, state: Any, X: Array, y: Array, key: PRNG
    ):
        def loss(m, s, xb, yb, k):
            base, new_s = self.loss_fn(m, s, xb, yb, k)
            if self.lam_l2 > 0.0:
                base = base + self.lam_l2 * weights_only_l2_penalty(
                    m, lam=self.lam_l2
                ) / max(self.lam_l2, 1e-12)
            return base, new_s

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


# -------------------------------------------------------------------------
# Publication-ready demo
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    DEFAULT_PLOT_STYLE = "accessible"

    # ----- data -----
    try:
        X, y = load_mnist_38(seed=0, flatten=True, standardize=True)
    except Exception:
        rng = np.random.RandomState(0)
        n, d = 6000, 50
        X = rng.randn(n, d).astype(np.float32)
        w = (rng.randn(d) / np.sqrt(d)).astype(np.float32)
        logits = X @ w
        p = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.rand(n) < p).astype(np.float32)
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
        X = (X - mu) / sd
        X, y = jnp.asarray(X), jnp.asarray(y)

    N_NODES = 4
    parts = make_hetero_3v8_parts_no_server(X, y, n_nodes=N_NODES)

    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, d: int, key):
            self.lin = eqx.nn.Linear(d, 1, key=key)

        def __call__(self, x, key=None, state=None):
            return self.lin(x).squeeze(-1), state

    d = int(X.shape[1])
    model_init = lambda k: LR(d, k)
    gamma0 = estimate_gamma_logistic(X, lam_l2=1e-3)
    lr_sched = make_polynomial_decay(gamma0, power=0.0, t0=1.0)

    edges = ring_edges(N_NODES)
    T = 250
    histories: Dict[str, Dict[str, List[float]]] = {}

    # A) mix every step, single
    trainer_A = PeerGossipTrainerEqx(
        model_init_fn=model_init,
        n_nodes=N_NODES,
        edges_fixed=edges,
        mix_policy={"name": "single"},
        mix_every=1,
        gamma=lr_sched,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        key=jr.PRNGKey(0),
        eval_inference_mode=True,
    )
    hA = trainer_A.fit(parts, X, y, eval_key=jr.PRNGKey(101))
    histories["mix1-single"] = hA

    # B) mix every 10 steps, Chebyshev K=2 (full participation)
    trainer_B = PeerGossipTrainerEqx(
        model_init_fn=model_init,
        n_nodes=N_NODES,
        edges_fixed=edges,
        mix_policy={"name": "cheby", "K": 2},
        mix_every=10,
        gamma=lr_sched,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        key=jr.PRNGKey(1),
        eval_inference_mode=True,
    )
    hB = trainer_B.fit(parts, X, y, eval_key=jr.PRNGKey(102))
    histories["mix10-chebyK2"] = hB

    # C) mix every 10 steps, adaptive Chebyshev (GLOBAL-aware + connectivity guard)
    trainer_C = PeerGossipTrainerEqx(
        model_init_fn=model_init,
        n_nodes=N_NODES,
        edges_fixed=edges,
        mix_policy={
            "name": "adaptive_cheby",
            "p_participation": 0.5,
            "rho_target": 0.20,  # GLOBAL target on Γ (squared ratio)
            "K_max": 6,
            "ensure_nonempty": True,
            "adapt_p_on_sat": True,
            "p_cooldown": 3,
            "p_min": 0.2,
            "p_max": 1.0,
        },
        mix_every=10,
        gamma=lr_sched,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        key=jr.PRNGKey(2),
        eval_inference_mode=True,
    )
    hC = trainer_C.fit(parts, X, y, eval_key=jr.PRNGKey(103))
    histories["mix10-adaptive-global"] = hC

    # D) GT with mixing each step (single)
    trainer_D = PeerGossipTrainerEqx(
        model_init_fn=model_init,
        n_nodes=N_NODES,
        edges_fixed=edges,
        mix_policy={"name": "gt", "kernel": "single"},
        mix_every=1,
        gamma=lr_sched,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        key=jr.PRNGKey(3),
        eval_inference_mode=True,
    )
    hD = trainer_D.fit(parts, X, y, eval_key=jr.PRNGKey(104))
    histories["mix1-gt-single"] = hD

    # E) GT with Chebyshev K=2, mixing every 5
    trainer_E = PeerGossipTrainerEqx(
        model_init_fn=model_init,
        n_nodes=N_NODES,
        edges_fixed=edges,
        mix_policy={"name": "gt", "kernel": "cheby", "K": 2},
        mix_every=5,
        gamma=lr_sched,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        key=jr.PRNGKey(4),
        eval_inference_mode=True,
    )
    hE = trainer_E.fit(parts, X, y, eval_key=jr.PRNGKey(105))
    histories["mix5-gt-cheby2"] = hE

    # ---- plots ----
    os.makedirs("figs_peer_gossip", exist_ok=True)
    plot_global_loss_q3(
        histories,
        title="Peer2Peer — global loss",
        save="figs_peer_gossip/loss.png",
        style=DEFAULT_PLOT_STYLE,
    )
    plot_consensus(
        histories,
        title="Peer2Peer — consensus (Γ)",
        save="figs_peer_gossip/consensus.png",
        style=DEFAULT_PLOT_STYLE,
    )
    print("Saved figs_peer_gossip/{loss,consensus}.png")

    # ---- numeric summary + LaTeX ----
    os.makedirs("tables", exist_ok=True)
    summary = summarize_histories(histories)
    print("\nNumeric summary (peer-to-peer):")
    print_publication_summary(summary, decimals=4)
    latex = latex_table_from_summary(
        summary,
        decimals=3,
        caption="Peer-to-peer: single vs Chebyshev vs adaptive (GLOBAL-aware) vs GT.",
        label="tab:peer",
    )
    with open("tables/peer_gossip_summary.tex", "w") as f:
        f.write(latex)
    print("Wrote tables/peer_gossip_summary.tex")
