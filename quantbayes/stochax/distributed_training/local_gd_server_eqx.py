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

# === core helpers (do not redefine) ===
from quantbayes.stochax.distributed_training.helpers import (
    mixing_matrix,  # W = I - α L
    safe_alpha as _safe_alpha,  # from helpers: slightly bold 0.95/deg_max
    _partition_params,
    _combine_params,
    tree_mix,
    laplacian_from_edges,
    make_star_with_server_edges,
    load_mnist_38,
    make_hetero_3v8_parts_with_server,
    estimate_gamma_logistic,
    make_polynomial_decay,
    plot_server_loss,
    plot_consensus_localgd,
    summarize_histories,
    print_publication_summary,
    latex_table_from_summary,
    is_weight_array,
    weights_only_l2_penalty,
)

# === spectral / adaptive utilities ===
from quantbayes.stochax.distributed_training.mixing_policies import (
    repeat_mix,
    chebyshev_mix,
    disagreement_interval_from_L,
    AdaptiveChebyController,
    min_K_for_target_rho,
    sample_active_clients,
    build_partial_star_W,
    xi_from_interval,
)

Array = jnp.ndarray
PRNG = jax.Array
BatchSchedule = Callable[[int, int, int], int]


def _flatten_params_l2(pytree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(pytree)
    flat = [jnp.ravel(x) for x in leaves if x is not None]
    return jnp.concatenate(flat) if flat else jnp.zeros((1,), dtype=jnp.float32)


class LocalGDServerEqx:
    """
    Local GD with a coordinator (server) and periodic communication.

    Star policy options (self.star_policy["name"]):
      - "single":   apply W_A once
      - "repeat":   apply W_A K times
      - "cheby":    degree-K Chebyshev p_K(W_A)
      - "adaptive": partial participation + GLOBAL-aware Chebyshev controller
                    Guardrails:
                      * min_active clients enforced
                      * periodic full coverage every 'coverage_stride' stars
                      * p-participation auto-tuner with cooldown + clamp
      - "gt":       gradient tracking at star cadence:
                      {"name":"gt", "kernel":"single"|"cheby", "K":int (if cheby)}
                    M_t is W_A or p_K(W_A) on star rounds, identity otherwise.
                    Server's gradient is identically zero.
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
            float(alpha_A)
            if alpha_A is not None
            else _safe_alpha(edges_A, self.n_total)
        )
        self.gamma = gamma
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

        # Mixing matrix for star rounds
        if make_W_A_custom is None:
            self.W_A = mixing_matrix(self.n_total, self.edges_A, self.alpha_A)
            rs = jnp.sum(self.W_A, axis=1)
            assert jnp.allclose(self.W_A, self.W_A.T, atol=1e-6)
            assert jnp.allclose(rs, jnp.ones_like(rs), atol=1e-6)
        else:
            self.W_A = make_W_A_custom(self.n_total, self.edges_A)
            rs = jnp.sum(self.W_A, axis=1)
            assert jnp.allclose(rs, jnp.ones_like(rs), atol=1e-6)

        # Identity mixing off-star rounds
        self.W_B = jnp.eye(self.n_total, dtype=jnp.float32)
        self.models: List[eqx.Module] = []
        self.states: List[Any] = []

        # Star policy + defaults
        self.star_policy: Dict[str, Any] = star_policy or {"name": "single"}
        if self.star_policy.get("name") == "adaptive":
            self.star_policy.setdefault("p_participation", 0.5)
            self.star_policy.setdefault(
                "rho_target", 0.20
            )  # GLOBAL target on Γ (sq ratio)
            self.star_policy.setdefault("K_max", 6)
            self.star_policy.setdefault("ensure_nonempty", True)
            self.star_policy.setdefault("adapt_p_on_sat", True)
            self.star_policy.setdefault("p_cooldown", 3)
            self.star_policy.setdefault("p_min", 0.2)
            self.star_policy.setdefault("p_max", 1.0)
            self.star_policy.setdefault("min_active", 2)
            self.star_policy.setdefault(
                "coverage_stride", 3
            )  # force full coverage every 3rd star

        # GLOBAL-aware controller (for τ > 1)
        self._cheby_ctl = None
        if self.star_policy.get("name") == "adaptive":
            self._cheby_ctl = AdaptiveChebyController(
                rho_target_sq_global=float(self.star_policy.get("rho_target", 0.20)),
                K_max=int(self.star_policy.get("K_max", 6)),
            )

        # p tuner cooldown
        self._p_cooldown_star: int = 0

        self.batch_schedule = batch_schedule

    def _gamma_t(self, t: int) -> float:
        return float(self.gamma(t)) if callable(self.gamma) else float(self.gamma)

    def _local_grad_step(
        self, model: eqx.Module, state: Any, X: Array, y: Array, key: PRNG
    ):
        # returns (grad, new_state)
        def loss(m, s, xb, yb, k):
            base, new_s = self.loss_fn(m, s, xb, yb, k)
            if self.lam_l2 > 0.0:
                base = base + weights_only_l2_penalty(m, lam=self.lam_l2)
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
        vecs = [_flatten_params_l2(params_list[i]) for i in idxs]
        stack = jnp.stack(vecs, axis=0)
        mean = jnp.mean(stack, axis=0, keepdims=True)
        sq = jnp.sum((stack - mean) ** 2, axis=1)
        return float(jnp.mean(sq))

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
        hist_star_rho: List[float] = []
        hist_K: List[int] = []
        hist_active: List[int] = []

        # Precompute Chebyshev interval lazily for full DS star (for "cheby")
        def _ensure_star_interval():
            li = getattr(self, "_star_lam_interval", None)
            if li is None:
                L_star = laplacian_from_edges(self.n_total, self.edges_A)
                self._star_lam_interval = disagreement_interval_from_L(
                    L_star, self.alpha_A
                )
            return self._star_lam_interval

        policy = self.star_policy.get("name", "single")
        use_gt = policy == "gt"

        # GT state
        if use_gt:
            y_list = []
            g_prev = []
            # server grad is zero
            for i in range(self.n_total):
                if i == self.server_id:
                    zero = jax.tree_util.tree_map(
                        lambda x: jnp.zeros_like(x), params_list[i]
                    )
                    y_list.append(zero)
                    g_prev.append(zero)
                else:
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
                    grad_i0, _ = self._local_grad_step(
                        _combine_params(params_list[i], static_list[i]),
                        self.states[i],
                        Xb,
                        yb,
                        sub,
                    )
                    y_list.append(grad_i0)
                    g_prev.append(grad_i0)

        for t in range(self.T):
            is_star = (t + 1) % self.tau == 0
            if is_star:
                cons_before = self._consensus_distance(
                    params_list, list(range(self.n_total))
                )

            if not use_gt:
                # ---------------- existing non-GT policies ----------------
                if is_star:
                    pol = policy
                    if pol == "single":
                        params_half = tree_mix(self.W_A, params_list)
                        K_used = 1
                        active = len(self.client_ids)

                    elif pol == "repeat":
                        K_used = int(self.star_policy.get("K", 2))
                        params_half = repeat_mix(self.W_A, params_list, K_used)
                        active = len(self.client_ids)

                    elif pol == "cheby":
                        K_used = int(self.star_policy.get("K", 3))
                        lam_min, lam_max = _ensure_star_interval()
                        params_half = chebyshev_mix(
                            self.W_A, params_list, K_used, lam_min, lam_max
                        )
                        active = len(self.client_ids)

                    elif pol == "adaptive":
                        p_part = float(self.star_policy.get("p_participation", 0.5))
                        p_min = float(self.star_policy.get("p_min", 0.2))
                        p_max = float(self.star_policy.get("p_max", 1.0))
                        p_part = float(np.clip(p_part, p_min, p_max))
                        ensure_nonempty = bool(
                            self.star_policy.get("ensure_nonempty", True)
                        )
                        min_active = int(self.star_policy.get("min_active", 2))
                        stride = int(self.star_policy.get("coverage_stride", 3))

                        # sample active
                        active_local, rng = sample_active_clients(
                            rng,
                            len(self.client_ids),
                            p_part,
                            ensure_nonempty=ensure_nonempty,
                        )
                        active_clients = [self.client_ids[i] for i in active_local]
                        active = len(active_clients)

                        # enforce min_active
                        if active < min_active:
                            # promote deterministically (first k clients)
                            need = min_active - active
                            promote = [
                                c for c in self.client_ids if c not in active_clients
                            ][:need]
                            active_clients.extend(promote)
                            active = len(active_clients)

                        # periodic full coverage
                        star_idx = (t + 1) // self.tau
                        if stride > 0 and (star_idx % stride == 0):
                            active_clients = self.client_ids[:]
                            active = len(active_clients)

                        if active >= 1:
                            # small star spectrum
                            edges_small = [(0, j + 1) for j in range(active)]
                            L_sub = laplacian_from_edges(1 + active, edges_small)
                            lam_min, lam_max = disagreement_interval_from_L(
                                L_sub, self.alpha_A
                            )
                            xi = xi_from_interval(lam_min, lam_max)
                            W_A_act = build_partial_star_W(
                                self.n_total,
                                self.server_id,
                                active_clients,
                                self.alpha_A,
                            )
                            if self._cheby_ctl is not None:
                                K_req = self._cheby_ctl.choose_K_global(
                                    xi,
                                    gamma_pre=cons_before,
                                    active=active + 1,
                                    total=self.n_total,
                                )
                            else:
                                K_req = min_K_for_target_rho(
                                    xi, float(self.star_policy.get("rho_target", 0.20))
                                )
                            K_cap = int(self.star_policy.get("K_max", 6))
                            K_used = int(max(1, min(K_cap, K_req)))
                            params_half = chebyshev_mix(
                                W_A_act, params_list, K_used, lam_min, lam_max
                            )
                        else:
                            K_used = 0
                            params_half = params_list
                    else:
                        raise ValueError(f"Unknown star_policy {pol}")

                    cons_after = self._consensus_distance(
                        params_half, list(range(self.n_total))
                    )
                    rho_hat = float(cons_after / (cons_before + 1e-12))
                    hist_star_rho.append(rho_hat)
                    hist_K.append(int(K_used))
                    hist_active.append(int(active))

                    # p tuner (cooldown + clamp)
                    if (
                        (pol == "adaptive")
                        and (self._cheby_ctl is not None)
                        and bool(self.star_policy.get("adapt_p_on_sat", True))
                    ):
                        target = float(self._cheby_ctl.rho_target_sq_global)
                        p_cool = int(self.star_policy.get("p_cooldown", 3))
                        if self._p_cooldown_star > 0:
                            self._p_cooldown_star -= 1
                        else:
                            if (rho_hat > 1.5 * target) and (
                                K_used >= self._cheby_ctl.K_max
                            ):
                                self.star_policy["p_participation"] = float(
                                    np.clip(1.25 * p_part, p_min, p_max)
                                )
                                self._p_cooldown_star = p_cool
                            elif (rho_hat < 0.4 * target) and (K_used >= 1):
                                self.star_policy["p_participation"] = float(
                                    np.clip(0.8 * p_part, p_min, p_max)
                                )
                                self._p_cooldown_star = p_cool

                    if self._cheby_ctl is not None and pol == "adaptive":
                        self._cheby_ctl.update_after_mix(rho_hat, K_used, cons_after)

                else:
                    params_half = tree_mix(
                        self.W_B, params_list
                    )  # identity mixing off-star

                # --- Local client updates (server: no local step) ---
                new_params_list = []
                gamma_t = self._gamma_t(t)
                rng_loop = rng
                for i in range(self.n_total):
                    if i == self.server_id:
                        new_params_list.append(params_half[i])
                        continue

                    Xi, yi = parts[i]
                    if self.batch_schedule is None:
                        Xb, yb = Xi, yi
                    else:
                        n_i = sizes[i]
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

                    rng_loop, sub = jr.split(rng_loop)
                    grad_i, new_state_i = self._local_grad_step(
                        _combine_params(params_half[i], static_list[i]),
                        self.states[i],
                        Xb,
                        yb,
                        sub,
                    )

                    if self.weight_decay > 0.0:
                        decay = 1.0 - gamma_t * self.weight_decay
                        params_half_i = jax.tree_util.tree_map(
                            lambda p: p * decay if is_weight_array(p) else p,
                            params_half[i],
                        )
                    else:
                        params_half_i = params_half[i]

                    updated = jax.tree_util.tree_map(
                        lambda p, g: p - gamma_t * g, params_half_i, grad_i
                    )
                    new_params_list.append(updated)
                    self.states[i] = new_state_i

                rng = rng_loop
                params_list = new_params_list

            else:
                # ---------------- GT policy at star cadence ----------------
                kernel = str(self.star_policy.get("kernel", "single")).lower()
                K_fixed = int(self.star_policy.get("K", 3)) if kernel == "cheby" else 1

                if is_star:
                    if kernel == "single":
                        x_half = tree_mix(self.W_A, params_list)
                        K_used = 1
                    elif kernel == "cheby":
                        lam_min, lam_max = _ensure_star_interval()
                        x_half = chebyshev_mix(
                            self.W_A, params_list, K_fixed, lam_min, lam_max
                        )
                        K_used = K_fixed
                    else:
                        raise ValueError(f"Unknown GT kernel '{kernel}'")
                    cons_after = self._consensus_distance(
                        x_half, list(range(self.n_total))
                    )
                    rho_hat = float(cons_after / (cons_before + 1e-12))
                    hist_star_rho.append(rho_hat)
                    hist_K.append(int(K_used))
                    hist_active.append(len(self.client_ids))
                else:
                    x_half = params_list
                    K_used = 0

                # one batch per client for both g_t and g_{t+1}
                rng_loop = rng
                batches = {}
                for i in self.client_ids:
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
                    batches[i] = (Xb, yb)

                # x_{t+1} = M_t x_t - γ y_t (server excluded)
                gamma_t = self._gamma_t(t)
                x_new = []
                for i in range(self.n_total):
                    if self.weight_decay > 0.0:
                        decay = 1.0 - gamma_t * self.weight_decay
                        x_half_i = jax.tree_util.tree_map(
                            lambda p: p * decay if is_weight_array(p) else p, x_half[i]
                        )
                    else:
                        x_half_i = x_half[i]
                    x_new_i = jax.tree_util.tree_map(
                        lambda p, y: p - gamma_t * y, x_half_i, y_list[i]
                    )
                    x_new.append(x_new_i)

                # g_{t+1}
                g_next = []
                for i in range(self.n_total):
                    if i == self.server_id:
                        g_next.append(
                            jax.tree_util.tree_map(
                                lambda x: jnp.zeros_like(x), x_new[i]
                            )
                        )
                        continue
                    Xb, yb = batches[i]
                    rng_loop, sub = jr.split(rng_loop)
                    grad_i_next, new_state_i = self._local_grad_step(
                        _combine_params(x_new[i], static_list[i]),
                        self.states[i],
                        Xb,
                        yb,
                        sub,
                    )
                    g_next.append(grad_i_next)
                    self.states[i] = new_state_i

                # mix y on star rounds
                if is_star:
                    if kernel == "single":
                        y_mix = tree_mix(self.W_A, y_list)
                    else:
                        lam_min, lam_max = _ensure_star_interval()
                        y_mix = chebyshev_mix(
                            self.W_A, y_list, K_fixed, lam_min, lam_max
                        )
                else:
                    y_mix = y_list

                # y_{t+1} = y_mix + g_{t+1} - g_t
                y_new = []
                for i in range(self.n_total):
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

            # --- Logging: server + sample clients
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
                if (
                    is_star
                    and (policy in ("single", "repeat", "cheby", "adaptive", "gt"))
                    and hist_star_rho
                ):
                    msg += f"  rho_star={hist_star_rho[-1]:.4f}  K={hist_K[-1]}  act={ (len(self.client_ids) if policy!='adaptive' else hist_active[-1]) }"
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
    alpha_A = _safe_alpha(edges_A, n_clients + 1)

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
    histories: Dict[str, Dict[str, List[float]]] = {}

    # A) tau=1 single
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

    # B) tau=10, Chebyshev K=3
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

    # C) tau=10, adaptive (GLOBAL-aware) + guardrails
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
            "rho_target": 0.20,  # GLOBAL target on Γ (squared ratio)
            "K_max": 6,
            "ensure_nonempty": True,
            "adapt_p_on_sat": True,
            "p_cooldown": 3,
            "p_min": 0.2,
            "p_max": 1.0,
            "min_active": 2,
            "coverage_stride": 3,
        },
        batch_schedule=None,
    )
    hC = trainer_C.fit(parts, X, y, eval_key=jr.PRNGKey(303), log=False)
    histories["tau10-adaptive-global"] = hC

    # D) tau=10, GT @ star cadence, Chebyshev K=3
    trainer_D = LocalGDServerEqx(
        model_init_fn=model_init,
        n_clients=n_clients,
        tau=10,
        edges_A=edges_A,
        alpha_A=alpha_A,
        gamma=lr_sched,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        key=jr.PRNGKey(4),
        eval_inference_mode=True,
        server_id=server_id,
        star_policy={"name": "gt", "kernel": "cheby", "K": 3},
        batch_schedule=None,
    )
    hD = trainer_D.fit(parts, X, y, eval_key=jr.PRNGKey(304), log=False)
    histories["tau10-gt-cheby3"] = hD

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
        caption="Server stars: single, Chebyshev, adaptive (GLOBAL-aware), and GT.",
        label="tab:server",
    )
    with open("tables/local_server_summary.tex", "w") as f:
        f.write(latex)
    print("Wrote tables/local_server_summary.tex")

    # ---- simple comm budget print (approx.) ----
    def comm_units(h: Dict[str, List[float]]) -> Dict[str, float]:
        K = np.asarray(h.get("K_hist", []), dtype=float)
        A = np.asarray(h.get("active_hist", []), dtype=float)  # active clients
        if K.size == 0 or A.size == 0:
            return {"sum_K": float(np.sum(K)), "sum_edge_ops": 0.0, "avg_active": 0.0}
        edges_used = np.where(A > 0, A, 0.0)  # star hub degree
        sum_edge_ops = float(np.sum(K * edges_used))
        return {
            "sum_K": float(np.sum(K)),
            "sum_edge_ops": sum_edge_ops,
            "avg_active": float(np.mean(A)),
        }

    print("\nCommunication budget (approx.):")
    for name, h in histories.items():
        c = comm_units(h)
        print(
            f"{name:>22s} | sum_K={c['sum_K']:.1f}  edge_ops≈{c['sum_edge_ops']:.1f}  avg_active={c['avg_active']:.2f}"
        )
