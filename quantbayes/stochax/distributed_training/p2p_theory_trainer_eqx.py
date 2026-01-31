# quantbayes/stochax/distributed_training/p2p_theory_trainer_eqx.py
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
    spectrum_report,
)

Array = jnp.ndarray
PRNG = jax.Array
BatchSchedule = Callable[[int, int, int], int]


class P2PTheoryTrainerEqx:
    """
    P2P distributed (S)GD with exact theory order:
      (i) Gossip: θ_{t+1/2} = KERNEL(W) θ_t  (when (t+1)%mix_every==0; else identity)
     (ii) Local:  θ_{t+1}^i  = θ_{t+1/2}^i - γ_t g_i(θ_{t+1/2}^i)

    Kernels: {"name":"single"} | {"name":"powerK","K":int} | {"name":"cheby","K":int}
    """

    def __init__(
        self,
        model_init_fn: Callable[[PRNG], eqx.Module],
        n_nodes: int,
        edges: List[Tuple[int, int]],
        *,
        alpha: Optional[float] = None,
        lazy: bool = False,
        kernel: Dict[str, Any] = {"name": "single"},
        mix_every: int = 1,  # τ inner steps before each mix
        gamma: float | Callable[[int], float] = 0.1,
        T: int = 400,
        loss_fn: Callable = binary_loss,
        batch_schedule: BatchSchedule | None = None,
        lam_l2: float = 0.0,
        weight_decay: float = 0.0,
        key: Optional[PRNG] = None,
        eval_inference_mode: bool = True,
    ):
        self.model_init_fn = model_init_fn
        self.n_nodes = int(n_nodes)
        self.edges = edges
        self.alpha = float(alpha) if alpha is not None else safe_alpha(edges, n_nodes)
        self.lazy = bool(lazy)
        self.W = mixing_matrix(n_nodes, edges, self.alpha, lazy=self.lazy)
        self.L = laplacian_from_edges(n_nodes, edges)

        self.kernel = dict(kernel)
        assert self.kernel["name"] in ("single", "powerK", "cheby")
        self.mix_every = int(max(1, mix_every))
        self.gamma = gamma
        self.T = int(T)
        self.loss_fn = loss_fn
        self.batch_schedule = batch_schedule
        self.lam_l2 = float(lam_l2)
        self.weight_decay = float(weight_decay)
        assert not (
            self.lam_l2 > 0.0 and self.weight_decay > 0.0
        ), "Use either lam_l2 (loss penalty) OR decoupled weight_decay."

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
        loss_hist: List[float] = []
        cons_hist: List[float] = []
        rho_hist: List[float] = []
        K_hist: List[int] = []

        for t in range(self.T):
            do_mix = ((t + 1) % self.mix_every) == 0
            cons_before = consensus_gamma(params_list)

            # --- Gossip ---
            if do_mix:
                name = self.kernel["name"]
                if name == "single":
                    params_half, K_used = tree_mix(self.W, params_list), 1
                elif name == "powerK":
                    K = int(max(1, self.kernel.get("K", 2)))
                    params_half, K_used = repeat_mix(self.W, params_list, K), K
                elif name == "cheby":
                    K = int(max(1, self.kernel.get("K", 2)))
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

            # --- Local step ---
            step = self._gamma_t(t)
            new_params: List[Any] = []
            rng_loop = rng
            for i in range(self.n_nodes):
                Xi, yi = parts[i]
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

            # --- Logging ---
            m0 = _combine_params(params_list[0], static_list[0])
            if self.eval_inference_mode:
                m0 = eqx.nn.inference_mode(m0, value=True)
            rng, k0 = jr.split(rng)
            loss0 = float(
                eval_step(m0, self.states[0], X_full, y_full, k0, self.loss_fn)
            )
            loss_hist.append(loss0)
            cons_hist.append(consensus_gamma(params_list))

        self.models = [_combine_params(p, s) for p, s in zip(params_list, static_list)]
        self.key = rng
        return {
            "loss_node1": loss_hist,
            "consensus_sq": cons_hist,
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
    Minimal demo of P2PTheoryTrainerEqx:
      • Ring(4) graph, logistic regression
      • Compare τ=1 single-step vs τ=10 Chebyshev K=3
      • Save loss/consensus plots and print a small summary
    """
    import os
    import numpy as np
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    from quantbayes.stochax.distributed_training.helpers import (
        load_mnist_38,
        make_hetero_3v8_parts_no_server,
        ring_edges,
        estimate_gamma_logistic,
        make_polynomial_decay,
        plot_global_loss_q3,
        plot_consensus,
        summarize_histories,
        print_publication_summary,
        latex_table_from_summary,
        safe_alpha,
        laplacian_from_edges,
        mixing_matrix,
    )
    from quantbayes.stochax.distributed_training.spectral import (
        spectrum_report,
        rho_bound_sq,
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

    # Parts & graph
    N = 4
    parts = make_hetero_3v8_parts_no_server(X, y, n_nodes=N)
    edges = ring_edges(N)
    alpha = safe_alpha(edges, N)
    L = laplacian_from_edges(N, edges)
    W = mixing_matrix(N, edges, alpha, lazy=False)
    rep = spectrum_report(W, L, alpha)
    print("Ring spectrum:", rep)
    print("Chebyshev K=3 bound:", rho_bound_sq(3, rep["xi"]))

    # LR schedule
    gamma0 = estimate_gamma_logistic(X, lam_l2=1e-3)
    lr_const = make_polynomial_decay(gamma0, power=0.0, t0=1.0)

    # Runs
    os.makedirs("figs_theory", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    T = 250
    outA = P2PTheoryTrainerEqx(
        model_init_fn=model_init,
        n_nodes=N,
        edges=edges,
        alpha=alpha,
        kernel={"name": "single"},
        mix_every=1,
        gamma=lr_const,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        key=jr.PRNGKey(0),
    ).fit(parts, X, y, eval_key=jr.PRNGKey(1001))
    outB = P2PTheoryTrainerEqx(
        model_init_fn=model_init,
        n_nodes=N,
        edges=edges,
        alpha=alpha,
        kernel={"name": "cheby", "K": 3},
        mix_every=10,
        gamma=lr_const,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        key=jr.PRNGKey(1),
    ).fit(parts, X, y, eval_key=jr.PRNGKey(1002))

    # Plots + summary
    hist = {
        "tau1-single": {
            "loss_node1": outA["loss_node1"],
            "consensus_sq": outA["consensus_sq"],
        },
        "tau10-cheby3": {
            "loss_node1": outB["loss_node1"],
            "consensus_sq": outB["consensus_sq"],
        },
    }
    plot_global_loss_q3(hist, title="P2P: loss", save="figs_theory/p2p_loss.png")
    plot_consensus(hist, title="P2P: consensus", save="figs_theory/p2p_consensus.png")
    summary = summarize_histories(hist)
    print_publication_summary(summary, decimals=4)
    with open("tables/p2p_demo.tex", "w") as f:
        f.write(
            latex_table_from_summary(
                summary, decimals=3, caption="P2P demo.", label="tab:p2p_demo"
            )
        )
