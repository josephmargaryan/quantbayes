# quantbayes/stochax/distributed_training/star_theory_trainer_eqx.py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.trainer.train import binary_loss, eval_step
from quantbayes.stochax.distributed_training.helpers import (
    laplacian_from_edges,
    mixing_matrix,
    safe_alpha,
    _partition_params,
    _combine_params,
    is_weight_array,
    weights_only_l2_penalty,
)
from quantbayes.stochax.distributed_training.spectral import (
    tree_mix,
    repeat_mix,
    chebyshev_mix,
    consensus_gamma,
    consensus_gamma_subset,
    spectrum_report,
)

Array = jnp.ndarray
PRNG = jax.Array
BatchSchedule = Callable[[int, int, int], int]


def make_star_edges(
    n_clients: int, server_id: Optional[int] = None
) -> List[Tuple[int, int]]:
    server = n_clients if server_id is None else int(server_id)
    return [(server, i) for i in range(n_clients)]


class StarTheoryTrainerEqx:
    """
    Client–Server (star) distributed (S)GD with theory order:
      (i) Star mix on rounds where (t+1)%tau==0: θ_{t+1/2} = KERNEL(W_star) θ_t
     (ii) Local updates on each node (server grad is zero if server has no data)

    Kernels: {"name":"single"} | {"name":"powerK","K":int} | {"name":"cheby","K":int}
    """

    def __init__(
        self,
        model_init_fn: Callable[[PRNG], eqx.Module],
        n_clients: int,
        *,
        tau: int = 10,  # inner steps between star mixes
        server_id: Optional[int] = None,
        edges_star: Optional[List[Tuple[int, int]]] = None,
        alpha: Optional[float] = None,
        lazy: bool = False,
        kernel: Dict[str, Any] = {"name": "single"},
        gamma: float | Callable[[int], float] = 0.1,
        T: int = 400,
        loss_fn: Callable = binary_loss,
        batch_schedule: BatchSchedule | None = None,
        lam_l2: float = 0.0,
        weight_decay: float = 0.0,
        server_has_data: bool = False,
        key: Optional[PRNG] = None,
        eval_inference_mode: bool = True,
    ):
        self.model_init_fn = model_init_fn
        self.n_clients = int(n_clients)
        self.server_id = self.n_clients if server_id is None else int(server_id)
        self.n_total = self.n_clients + 1
        assert 0 <= self.server_id < self.n_total

        self.client_ids = [i for i in range(self.n_total) if i != self.server_id]
        self.tau = int(max(1, tau))

        self.edges_star = (
            edges_star
            if edges_star is not None
            else [(self.server_id, i) for i in self.client_ids]
        )
        self.alpha = (
            float(alpha)
            if alpha is not None
            else safe_alpha(self.edges_star, self.n_total)
        )
        self.lazy = bool(lazy)
        self.W = mixing_matrix(
            self.n_total, self.edges_star, self.alpha, lazy=self.lazy
        )
        self.L = laplacian_from_edges(self.n_total, self.edges_star)

        self.kernel = dict(kernel)
        assert self.kernel["name"] in ("single", "powerK", "cheby")
        self.gamma = gamma
        self.T = int(T)
        self.loss_fn = loss_fn
        self.batch_schedule = batch_schedule
        self.lam_l2 = float(lam_l2)
        self.weight_decay = float(weight_decay)
        assert not (
            self.lam_l2 > 0.0 and self.weight_decay > 0.0
        ), "Use either lam_l2 (loss penalty) OR decoupled weight_decay."
        self.server_has_data = bool(server_has_data)

        self.eval_inference_mode = bool(eval_inference_mode)
        self.key = jr.PRNGKey(0) if key is None else key

        self.models: List[eqx.Module] = []
        self.states: List[Any] = []
        self._init_models()

        rep = spectrum_report(self.W)
        self._lam_min, self._lam_max = rep["lam_min"], rep["lam_max"]
        self._xi, self._slem = rep["xi"], rep["slem"]

    def _init_models(self):
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

    def _gamma_t(self, t: int) -> float:
        return float(self.gamma(t)) if callable(self.gamma) else float(self.gamma)

    def _loss_with_reg(self, m, s, xb, yb, k):
        base, new_s = self.loss_fn(m, s, xb, yb, k)
        if self.lam_l2 > 0.0:
            base = base + weights_only_l2_penalty(m, lam=self.lam_l2)
        return base, new_s

    def _select_batch(
        self, rng: PRNG, Xi: Array, yi: Array, t: int, i: int
    ) -> Tuple[PRNG, Array, Array]:
        if self.batch_schedule is None:
            return rng, Xi, yi
        n_i = int(Xi.shape[0])
        B = int(self.batch_schedule(t, i, n_i))
        B = max(1, min(B, n_i))
        rng, sub = jr.split(rng)
        idx = (
            jnp.arange(n_i)
            if B == n_i
            else jr.choice(sub, n_i, shape=(B,), replace=False)
        )
        return rng, Xi[idx], yi[idx]

    def fit(
        self,
        parts: List[Tuple[Array, Array]],
        X_full: Array,
        y_full: Array,
        *,
        eval_key: Optional[PRNG] = None,
    ) -> Dict[str, List[float]]:
        params_list, static_list = [], []
        for m in self.models:
            p, s = _partition_params(m)
            params_list.append(p)
            static_list.append(s)

        rng = self.key if eval_key is None else eval_key
        hist_loss_server: List[float] = []
        hist_loss_node1: List[float] = []
        hist_loss_nodel: List[float] = []
        hist_cons_all: List[float] = []
        hist_cons_cli: List[float] = []
        rho_hist: List[float] = []
        K_hist: List[int] = []

        for t in range(self.T):
            is_star = ((t + 1) % self.tau) == 0
            cons_before = consensus_gamma(params_list)

            # --- Star gossip ---
            if is_star:
                name = self.kernel["name"]
                if name == "single":
                    params_half, K_used = tree_mix(self.W, params_list), 1
                elif name == "powerK":
                    K = int(max(1, self.kernel.get("K", 2)))
                    params_half, K_used = repeat_mix(self.W, params_list, K), K
                elif name == "cheby":
                    K = int(max(1, self.kernel.get("K", 3)))
                    params_half = chebyshev_mix(
                        self.W, params_list, K, self._lam_min, self._lam_max
                    )
                    K_used = K
                cons_after = consensus_gamma(params_half)
                rho_hist.append(float(cons_after / (cons_before + 1e-12)))
                K_hist.append(int(K_used))
            else:
                params_half = params_list
                rho_hist.append(1.0)
                K_hist.append(0)

            # --- Local steps (server: optional zero grad) ---
            step = self._gamma_t(t)
            new_params: List[Any] = []
            rng_loop = rng
            for i in range(self.n_total):
                Xi, yi = parts[i]
                if (i == self.server_id) and (not self.server_has_data):
                    # Optional decoupled WD only
                    if self.weight_decay > 0.0:
                        decay = 1.0 - step * self.weight_decay
                        params_half_i = jax.tree_util.tree_map(
                            lambda p: p * decay if is_weight_array(p) else p,
                            params_half[i],
                        )
                    else:
                        params_half_i = params_half[i]
                    new_params.append(params_half_i)
                    continue

                rng_loop, Xb, yb = self._select_batch(rng_loop, Xi, yi, t, i)

                def loss_local(m, s, xb, yb, k):
                    return self._loss_with_reg(m, s, xb, yb, k)

                rng_loop, sub = jr.split(rng_loop)
                (_, new_state_i), grad_i = eqx.filter_value_and_grad(
                    loss_local, has_aux=True
                )(
                    _combine_params(params_half[i], static_list[i]),
                    self.states[i],
                    Xb,
                    yb,
                    sub,
                )

                if self.weight_decay > 0.0:
                    decay = 1.0 - step * self.weight_decay
                    params_half_i = jax.tree_util.tree_map(
                        lambda p: p * decay if is_weight_array(p) else p, params_half[i]
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

            # --- Logging (server + extremes) ---
            server = self.server_id
            c0 = self.client_ids[0]
            cL = self.client_ids[-1]
            mS = _combine_params(params_list[server], static_list[server])
            m1 = _combine_params(params_list[c0], static_list[c0])
            mL = _combine_params(params_list[cL], static_list[cL])
            if self.eval_inference_mode:
                mS = eqx.nn.inference_mode(mS, value=True)
                m1 = eqx.nn.inference_mode(m1, value=True)
                mL = eqx.nn.inference_mode(mL, value=True)
            rng, kS = jr.split(rng)
            lossS = float(
                eval_step(mS, self.states[server], X_full, y_full, kS, self.loss_fn)
            )
            rng, k1 = jr.split(rng)
            loss1 = float(
                eval_step(m1, self.states[c0], X_full, y_full, k1, self.loss_fn)
            )
            rng, kL = jr.split(rng)
            lossL = float(
                eval_step(mL, self.states[cL], X_full, y_full, kL, self.loss_fn)
            )
            hist_loss_server.append(lossS)
            hist_loss_node1.append(loss1)
            hist_loss_nodel.append(lossL)

            hist_cons_all.append(consensus_gamma(params_list))
            hist_cons_cli.append(consensus_gamma_subset(params_list, self.client_ids))

        self.models = [_combine_params(p, s) for p, s in zip(params_list, static_list)]
        self.key = rng
        return {
            "loss_server": hist_loss_server,
            "loss_node1": hist_loss_node1,
            "loss_node_last": hist_loss_nodel,
            "consensus_all": hist_cons_all,
            "consensus_clients": hist_cons_cli,
            "rho_hist": rho_hist,
            "K_hist": K_hist,
            "lam_min": [self._lam_min],
            "lam_max": [self._lam_max],
            "xi": [self._xi],
            "slem": [self._slem],
        }


# ------------------------------- MAIN ---------------------------------
if __name__ == "__main__":
    """
    Minimal demo of StarTheoryTrainerEqx:
      • Star with 4 clients + 1 server (server has no data)
      • Compare τ=1 single vs τ=10 Chebyshev K=3
      • Save server-loss/consensus plots and a small summary
    """
    import os
    import numpy as np
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    from quantbayes.stochax.distributed_training.helpers import (
        load_mnist_38,
        make_hetero_3v8_parts_with_server,
        estimate_gamma_logistic,
        make_polynomial_decay,
        plot_server_loss,
        plot_consensus_localgd,
        summarize_histories,
        print_publication_summary,
        latex_table_from_summary,
    )

    # Data
    try:
        X, y = load_mnist_38(seed=0, flatten=True, standardize=True)
    except Exception:
        rng = np.random.default_rng(0)
        n, d = 4000, 50
        X = rng.standard_normal((n, d)).astype(np.float32)
        w = (rng.standard_normal(d) / np.sqrt(d)).astype(np.float32)
        p = 1 / (1 + np.exp(-(X @ w)))
        y = (rng.random(n) < p).astype(np.float32)
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
        X = (X - mu) / sd
        X, y = jnp.asarray(X), jnp.asarray(y)

    # Model
    d = int(X.shape[1])

    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, d: int, key):
            self.lin = eqx.nn.Linear(d, 1, key=key)

        def __call__(self, x, key=None, state=None):
            return self.lin(x).squeeze(-1), state

    model_init = lambda k: LR(d, k)

    # Parts (clients + empty server)
    N = 4
    parts = make_hetero_3v8_parts_with_server(X, y, n_clients=N)

    # LR schedule
    gamma0 = estimate_gamma_logistic(X, lam_l2=1e-3)
    lr_const = make_polynomial_decay(gamma0, power=0.0, t0=1.0)

    os.makedirs("figs_theory", exist_ok=True)
    os.makedirs("tables", exist_ok=True)

    T = 300
    out1 = StarTheoryTrainerEqx(
        model_init_fn=model_init,
        n_clients=N,
        tau=1,
        server_id=N,
        kernel={"name": "single"},
        gamma=lr_const,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        server_has_data=False,
        key=jr.PRNGKey(10),
    ).fit(parts, X, y, eval_key=jr.PRNGKey(2101))

    out2 = StarTheoryTrainerEqx(
        model_init_fn=model_init,
        n_clients=N,
        tau=10,
        server_id=N,
        kernel={"name": "cheby", "K": 3},
        gamma=lr_const,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        server_has_data=False,
        key=jr.PRNGKey(11),
    ).fit(parts, X, y, eval_key=jr.PRNGKey(2102))

    group = {"tau1-single": out1, "tau10-cheby3": out2}
    plot_server_loss(group, title="Star: server loss", save="figs_theory/star_loss.png")
    plot_consensus_localgd(
        group, title="Star: consensus", save="figs_theory/star_consensus.png"
    )

    summary = summarize_histories(
        {
            k: {
                kk: vv
                for kk, vv in v.items()
                if kk
                in (
                    "loss_server",
                    "consensus_all",
                    "consensus_clients",
                    "rho_hist",
                    "K_hist",
                )
            }
            for k, v in group.items()
        }
    )
    print_publication_summary(summary, decimals=4)
    with open("tables/star_demo.tex", "w") as f:
        f.write(
            latex_table_from_summary(
                summary, decimals=3, caption="Star demo.", label="tab:star_demo"
            )
        )
