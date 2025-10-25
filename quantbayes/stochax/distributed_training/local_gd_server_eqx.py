# quantbayes/stochax/distributed_training/local_gd_server_eqx.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, Any
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.trainer.train import eval_step, binary_loss
from quantbayes.stochax.privacy.dp import DPSGDConfig, DPPrivacyEngine

# Reuse helpers from your DGD module
from quantbayes.stochax.distributed_training.helpers import (
    mixing_matrix,  # W = I - alpha * L (symmetric, row-stochastic)
    safe_alpha as _safe_alpha_ha4,
    _partition_params,
    tree_mix,
    _combine_params,
    laplacian_from_edges,
)

# NEW: spectral star policies (safe Chebyshev + controller)
from quantbayes.stochax.distributed_training.mixing_policies import (
    repeat_mix,
    chebyshev_mix,  # SAFE wrapper (symmetry/DS + interval checks)
    disagreement_interval_from_L,
    AdaptiveChebyController,  # NEW
    min_K_for_target_rho,
    sample_active_clients,
    build_partial_star_W,
    xi_from_interval,  # NEW (optional)
)

Array = jnp.ndarray
PRNG = jax.Array
BatchSchedule = Callable[[int, int, int], int]  # B = schedule(t, client_index, n_i)


# ---- small utilities ----
def safe_alpha(edges: List[Tuple[int, int]], n_nodes: int) -> float:
    """Conservative α < 1/deg_max, re-exported from HA4."""
    return _safe_alpha_ha4(edges, n_nodes)


def _flatten_params_l2(pytree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(pytree)
    flat = [jnp.ravel(x) for x in leaves if x is not None]
    return jnp.concatenate(flat) if flat else jnp.zeros((1,), dtype=jnp.float32)


def _is_weight_array(x):
    return eqx.is_inexact_array(x) and (getattr(x, "ndim", 0) >= 2)


def _weights_only_l2_penalty(params: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(eqx.filter(params, _is_weight_array))
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in leaves)


def make_star_with_server_edges(
    n_clients: int, server_id: int | None = None
) -> List[Tuple[int, int]]:
    if server_id is None:
        server_id = n_clients
    edges = [(server_id, i) for i in range(n_clients)]
    return edges


def make_mixing_with_per_node_alphas(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    alphas: List[float],
) -> Array:
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


# ---- main trainer ----
class LocalGDServerEqx:
    """
    Local GD with a coordinator (server) and periodic communication.

    Star policy options (self.star_policy["name"]):
      - "single":   apply W_A once
      - "repeat":   apply W_A K times (self.star_policy["K"])
      - "cheby":    apply degree-K Chebyshev p_K(W_A) using the full-star spectrum
      - "adaptive": per-star partial participation + phase-aware adaptive Chebyshev
                    (uses AdaptiveChebyController to budget contraction when τ>1)
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
        batch_schedule: BatchSchedule | None = None,
        lam_l2: Optional[float] = None,
        weight_decay: float = 0.0,
    ):
        self.model_init_fn = model_init_fn
        self.n_clients = int(n_clients)
        self.server_id = self.n_clients if server_id is None else int(server_id)
        self.n_total = self.n_clients + 1
        assert 0 <= self.server_id < self.n_total
        self.client_ids = [i for i in range(self.n_total) if i != self.server_id]

        self.tau = int(max(1, tau))
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
        self.lam_l2 = float(lam_l2) if lam_l2 is not None else 0.0
        self.weight_decay = float(weight_decay)
        assert not (
            self.lam_l2 > 0.0 and self.weight_decay > 0.0
        ), "Choose either lam_l2 (loss-level L2) OR weight_decay (decoupled), not both."

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

        # NEW: phase-aware controller (for τ > 1)
        self._cheby_ctl = None
        if self.star_policy.get("name") == "adaptive":
            self._cheby_ctl = AdaptiveChebyController(
                rho_target_sq=float(self.star_policy.get("rho_target", 0.05)),
                K_max=int(self.star_policy.get("K_max", 5)),
            )

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
            base, new_s = self.loss_fn(m, s, xb, yb, k)
            if self.lam_l2 > 0.0:
                base = base + self.lam_l2 * _weights_only_l2_penalty(m)
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
        )  # #active clients per star (adaptive; else n_clients)

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
                    params_half = tree_mix(self.W_A, params_list)
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

                    if active_count >= 2:
                        # compute star interval cheaply (star of size 1+active_count)
                        edges_small = [(0, j + 1) for j in range(active_count)]
                        L_sub = laplacian_from_edges(1 + active_count, edges_small)
                        lam_min, lam_max = disagreement_interval_from_L(
                            L_sub, self.alpha_A
                        )
                        xi = xi_from_interval(lam_min, lam_max)
                        W_A_act = build_partial_star_W(
                            self.n_total, self.server_id, active_clients, self.alpha_A
                        )

                        # choose K using controller (phase-aware) or fallback
                        if self._cheby_ctl is not None:
                            K_req = self._cheby_ctl.choose_K(xi, gamma_pre=cons_before)
                        else:
                            K_req = min_K_for_target_rho(
                                xi, float(self.star_policy.get("rho_target", 0.05))
                            )
                        K_cap = int(self.star_policy.get("K_max", 5))
                        K_used = int(max(1, min(K_cap, K_req)))
                        params_half = chebyshev_mix(
                            W_A_act, params_list, K_used, lam_min, lam_max
                        )
                    else:
                        K_used = 1
                        params_half = params_list
                else:
                    raise ValueError(f"Unknown star_policy {policy}")

                # Per-star contraction: measure after star and log ratio
                cons_after = self._consensus_distance(
                    params_half, list(range(self.n_total))
                )
                rho_hat = float(cons_after / (cons_before + 1e-12))
                hist_star_rho.append(rho_hat)
                hist_K.append(int(K_used))
                hist_active.append(int(active_count))
                if self._cheby_ctl is not None:
                    self._cheby_ctl.update_after_mix(rho_hat, K_used, cons_after)
            else:
                params_half = tree_mix(self.W_B, params_list)  # identity mixing

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

                # Optional decoupled WD for clients only (exclude server)
                if self.weight_decay > 0.0:
                    decay = 1.0 - gamma_t * self.weight_decay

                    def _shrink(p):
                        return p * decay if _is_weight_array(p) else p

                    params_half_i = jax.tree_util.tree_map(_shrink, params_half[i])
                else:
                    params_half_i = params_half[i]

                updated = jax.tree_util.tree_map(
                    lambda p, g: p - gamma_t * g, params_half_i, grad_i
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


if __name__ == "__main__":
    import os
    import numpy as np
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    from quantbayes.stochax.distributed_training.helpers import (
        load_mnist_38,
        make_hetero_3v8_parts_with_server,
        make_star_with_server_edges,
        safe_alpha,
        estimate_gamma_logistic,
        make_polynomial_decay,
        plot_server_loss,
        plot_consensus_localgd,
        summarize_histories,
        print_publication_summary,
        latex_table_from_summary,
    )
    from quantbayes.stochax.distributed_training.local_gd_server_eqx import (
        LocalGDServerEqx,
    )

    DEFAULT_PLOT_STYLE = "accessible"

    # ----- data -----
    try:
        X, y = load_mnist_38(seed=0, flatten=True, standardize=True)
    except Exception:
        rng = np.random.RandomState(0)
        n, d = 8000, 30
        X = rng.randn(n, d).astype(np.float32)
        w = (rng.randn(d) / np.sqrt(d)).astype(np.float32)
        logits = X @ w
        p = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.rand(n) < p).astype(np.float32)
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
        X = (X - mu) / sd
        X, y = jnp.asarray(X), jnp.asarray(y)

    n_clients = 4
    parts = make_hetero_3v8_parts_with_server(X, y, n_clients=n_clients)
    d = int(X.shape[1])
    server_id = n_clients
    edges_A = make_star_with_server_edges(n_clients, server_id=server_id)
    alpha_A = safe_alpha(edges_A, n_clients + 1)

    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, d: int, key):
            self.lin = eqx.nn.Linear(d, 1, key=key)

        def __call__(self, x, key=None, state=None):
            return self.lin(x).squeeze(-1), state

    model_init = lambda k: LR(d, k)
    gamma0 = estimate_gamma_logistic(X, lam_l2=1e-3)
    lr_sched = make_polynomial_decay(gamma0, power=0.0, t0=1.0)

    T = 300
    histories = {}

    trainer_A = LocalGDServerEqx(
        model_init_fn=model_init,
        n_clients=n_clients,
        tau=1,
        edges_A=edges_A,
        alpha_A=alpha_A,
        gamma=lr_sched,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        key=jr.PRNGKey(0),
        eval_inference_mode=True,
        server_id=server_id,
        star_policy={"name": "single"},
        batch_schedule=None,
    )
    hA = trainer_A.fit(parts, X, y, eval_key=jr.PRNGKey(301), log=False)
    histories["tau1-single"] = hA

    trainer_B = LocalGDServerEqx(
        model_init_fn=model_init,
        n_clients=n_clients,
        tau=10,
        edges_A=edges_A,
        alpha_A=alpha_A,
        gamma=lr_sched,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        key=jr.PRNGKey(1),
        eval_inference_mode=True,
        server_id=server_id,
        star_policy={"name": "cheby", "K": 3},
        batch_schedule=None,
    )
    hB = trainer_B.fit(parts, X, y, eval_key=jr.PRNGKey(302), log=False)
    histories["tau10-cheby3"] = hB

    trainer_C = LocalGDServerEqx(
        model_init_fn=model_init,
        n_clients=n_clients,
        tau=10,
        edges_A=edges_A,
        alpha_A=alpha_A,
        gamma=lr_sched,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        key=jr.PRNGKey(2),
        eval_inference_mode=True,
        server_id=server_id,
        star_policy={
            "name": "adaptive",
            "p_participation": 0.5,
            "rho_target": 0.05,
            "K_max": 5,
            "ensure_nonempty": True,
        },
        batch_schedule=None,
    )
    hC = trainer_C.fit(parts, X, y, eval_key=jr.PRNGKey(303), log=False)
    histories["tau10-adaptive"] = hC

    # ---- plots ----
    os.makedirs("figs_local_server", exist_ok=True)
    plot_server_loss(
        histories,
        title="LocalGDServer — server loss",
        save="figs_local_server/server_loss.png",
        style=DEFAULT_PLOT_STYLE,
    )
    plot_consensus_localgd(
        histories,
        title="LocalGDServer — consensus",
        save="figs_local_server/consensus.png",
        style=DEFAULT_PLOT_STYLE,
    )
    print("Saved figs_local_server/{server_loss,consensus}.png")

    # ---- numeric summary + LaTeX ----
    os.makedirs("tables", exist_ok=True)
    summary = summarize_histories(histories)
    print("\nNumeric summary (server stars):")
    print_publication_summary(summary, decimals=4)
    latex = latex_table_from_summary(
        summary,
        decimals=3,
        caption="Server star rounds: single, Chebyshev, and adaptive.",
        label="tab:server",
    )
    with open("tables/local_server_summary.tex", "w") as f:
        f.write(latex)
    print("Wrote tables/local_server_summary.tex")
