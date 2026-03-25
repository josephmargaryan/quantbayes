# quantbayes/stochax/distributed_training/async_ps_trainer_eqx.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, Any
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.trainer.train import binary_loss, eval_step
from quantbayes.stochax.distributed_training.helpers import l2_penalty
from quantbayes.stochax.privacy.dp import DPSGDConfig, DPPrivacyEngine

Array = jnp.ndarray
PRNG = jax.Array

__all__ = [
    "AsyncParameterServerEqx",
    "uniform_partition",
    "uniform_delay_sampler",
    "poisson_delay_sampler",
]


# ---------- utils ----------
def _partition_params(model: eqx.Module):
    return eqx.partition(model, eqx.is_inexact_array)  # (params, static)


def _combine_params(params: Any, static: Any) -> eqx.Module:
    return eqx.combine(params, static)


def uniform_partition(X: Array, y: Array, n_nodes: int) -> List[Tuple[Array, Array]]:
    N = int(X.shape[0])
    base, rem = divmod(N, n_nodes)
    out, start = [], 0
    for i in range(n_nodes):
        size = base + (1 if i < rem else 0)
        out.append((X[start : start + size], y[start : start + size]))
        start += size
    return out


def uniform_delay_sampler(
    tau_min: int = 1, tau_max: int = 5
) -> Callable[[int, int], int]:
    rng = np.random.RandomState(0)
    tau_min, tau_max = int(tau_min), int(tau_max)
    assert 1 <= tau_min <= tau_max

    def sample(worker_id: int, t: int) -> int:
        return int(rng.randint(tau_min, tau_max + 1))

    return sample


def poisson_delay_sampler(
    lmbda: float = 2.0, cap: int = 10
) -> Callable[[int, int], int]:
    rng = np.random.RandomState(1)
    cap = int(cap)

    def sample(worker_id: int, t: int) -> int:
        d = 1 + rng.poisson(lam=max(1e-8, float(lmbda)))
        return int(min(d, cap))

    return sample


# ---------- server optimizer (FedOpt) ----------
class _ServerOpt:
    """
    Stateless interface:
      apply(params, update) performs: params <- params - lr * OPT(update)
    where OPT is identity/momentum/Adam transformation of 'update'.
    """

    def __init__(
        self, name="sgd", lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, momentum=0.9
    ):
        self.name = str(name).lower()
        self.lr = float(lr)
        self.b1 = float(beta1)
        self.b2 = float(beta2)
        self.eps = float(eps)
        self.mu = float(momentum)
        self.m = None
        self.v = None
        self.t = 0

    @staticmethod
    def _zeros_like(p):
        return jax.tree_util.tree_map(jnp.zeros_like, p)

    @staticmethod
    def _add(a, b):
        return jax.tree_util.tree_map(lambda x, y: x + y, a, b)

    @staticmethod
    def _scale(a, s):
        return jax.tree_util.tree_map(lambda x: s * x, a)

    @staticmethod
    def _sub(a, b):
        return jax.tree_util.tree_map(lambda x, y: x - y, a, b)

    def apply(self, params, update):
        if self.name in {"sgd", "none"}:
            return self._sub(params, self._scale(update, self.lr))

        if self.name == "momentum":
            if self.m is None:
                self.m = self._zeros_like(update)
            self.m = self._add(self._scale(self.m, self.mu), update)
            return self._sub(params, self._scale(self.m, self.lr))

        if self.name in {"adam", "adamw"}:
            if self.m is None:
                self.m = self._zeros_like(update)
            if self.v is None:
                self.v = self._zeros_like(update)
            self.t += 1
            self.m = jax.tree_util.tree_map(
                lambda m, g: self.b1 * m + (1 - self.b1) * g, self.m, update
            )
            self.v = jax.tree_util.tree_map(
                lambda v, g: self.b2 * v + (1 - self.b2) * (g * g), self.v, update
            )
            mhat = self._scale(self.m, 1.0 / (1 - self.b1**self.t))
            vhat = self._scale(self.v, 1.0 / (1 - self.b2**self.t))
            step = jax.tree_util.tree_map(
                lambda m, v: self.lr * m / (jnp.sqrt(v) + self.eps), mhat, vhat
            )
            return self._sub(params, step)

        raise ValueError(f"Unknown server_opt '{self.name}'")


# ---------- main trainer ----------
class AsyncParameterServerEqx:
    """
    Event-driven Asynchronous Parameter Server (production-grade):

    • Aggregation mode:
        - 'grad': one minibatch gradient from the worker snapshot (FedSGD)
        - 'delta': S local steps with client LR (FedAvg-style); send δ = θ_loc - θ_snap
    • Staleness-aware scaling ϕ(s): 'none' | 'power' (alpha) | 'exp' (lambda)
    • Server optimizer (FedOpt): 'sgd' | 'momentum' | 'adam' | 'adamw'
    • Optional global-norm clipping, DP on worker gradients, EF-Sign compression.

    Update on arrival with scale s_i = w_i * ϕ(staleness):
        params <- OPT.apply(params,  s_i * u_i )           if aggregation='grad'
        params <- OPT.apply(params, -s_i * δ_i )           if aggregation='delta'
    (Minus sign on δ_i because OPT.apply subtracts its argument times lr.)
    """

    def __init__(
        self,
        model_init_fn: Callable[[PRNG], eqx.Module],
        *,
        aggregation: str = "grad",  # 'grad' or 'delta'
        server_opt: Optional[Dict[str, float | str]] = None,  # {"name":"adam","lr":...}
        delay_sampler: Callable[[int, int], int] = poisson_delay_sampler(2.0, cap=10),
        loss_fn: Callable = binary_loss,
        lam_l2: Optional[float] = None,  # loss-level L2 on worker objective
        dp_config: Optional[
            DPSGDConfig
        ] = None,  # DP on worker gradients (in 'grad' mode and inside local steps)
        weight_by_data: bool = True,  # scale by worker's data fraction
        staleness: Optional[
            Dict[str, float | str]
        ] = None,  # {"mode":"power|exp|none","alpha":0.6,"lambda":0.1}
        clip_norm: Optional[float] = None,  # global norm clipping on worker update
        compress: Optional[
            Dict[str, Any]
        ] = None,  # {"name":"none|sign","error_feedback":True}
        # client local training (used in 'delta' mode; optional in 'grad' mode to form larger minibatch)
        client: Optional[
            Dict[str, Any]
        ] = None,  # {"local_epochs":S, "lr":η_c, "batch":B}
        key: Optional[PRNG] = None,
        eval_inference_mode: bool = True,
        log_every: int = 20,
    ):
        self.model_init_fn = model_init_fn
        self.aggregation = str(aggregation).lower()
        assert self.aggregation in {"grad", "delta"}

        self.loss_fn = loss_fn
        self.lam_l2 = lam_l2
        self.delay_sampler = delay_sampler
        self.weight_by_data = bool(weight_by_data)
        self.eval_inference_mode = bool(eval_inference_mode)
        self.log_every = int(max(1, log_every))
        self.key = key if key is not None else jr.PRNGKey(0)

        # DP engine
        self.dp_config = dp_config
        self._dp_engine = DPPrivacyEngine(dp_config) if dp_config else None

        # server optimizer
        so = server_opt or {
            "name": "adam",
            "lr": 3e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        }
        self._opt = _ServerOpt(**so)

        # staleness decay
        st = staleness or {"mode": "power", "alpha": 0.6}
        self.st_mode = str(st.get("mode", "power")).lower()
        self.st_alpha = float(st.get("alpha", 0.6))
        self.st_lambda = float(st.get("lambda", 0.1))

        # clipping & compression
        self.clip_norm = None if clip_norm is None else float(clip_norm)
        c = compress or {"name": "none", "error_feedback": True}
        self.compress_name = str(c.get("name", "none")).lower()
        self.error_feedback = bool(c.get("error_feedback", True))

        # client local training defaults
        cli = client or {}
        self.local_epochs = int(cli.get("local_epochs", 1))
        self.client_lr = float(cli.get("lr", 1e-1))
        self.client_batch = (
            None if cli.get("batch", None) is None else int(cli["batch"])
        )

        # server state & residuals
        self.model: Optional[eqx.Module] = None
        self.state: Optional[Any] = None
        self._residuals: Optional[List[Any]] = None  # for EF-Sign

    # ---- helpers ----
    def _phi(self, staleness: int) -> float:
        s = float(max(0, int(staleness)))
        if self.st_mode == "power":
            return (1.0 + s) ** (-self.st_alpha)
        if self.st_mode == "exp":
            return float(np.exp(-self.st_lambda * s))
        return 1.0

    @staticmethod
    def _global_norm(pytree) -> Array:
        leaves = jax.tree_util.tree_leaves(pytree)
        return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))

    def _clip_by_global_norm(self, tree):
        if self.clip_norm is None or self.clip_norm <= 0:
            return tree
        gnorm = self._global_norm(tree)
        scale = jnp.minimum(1.0, self.clip_norm / (gnorm + 1e-12))
        return jax.tree_util.tree_map(lambda g: g * scale, tree)

    def _compress_with_residual(self, i: int, payload):
        if self.compress_name == "none":
            return payload
        # error feedback residuals
        if self.error_feedback:
            r = self._residuals[i]
            payload = jax.tree_util.tree_map(lambda g, ri: g + ri, payload, r)
        if self.compress_name == "sign":
            comp = jax.tree_util.tree_map(jnp.sign, payload)
        else:
            raise ValueError(f"Unknown compressor '{self.compress_name}'")
        if self.error_feedback:
            self._residuals[i] = jax.tree_util.tree_map(
                lambda ge, c: ge - c, payload, comp
            )
        return comp

    def _worker_grad(
        self, model: eqx.Module, state: Any, X: Array, y: Array, key: PRNG
    ):
        # Loss with optional L2 on weights
        def loss_with_reg(m, s, xb, yb, k):
            base, new_s = self.loss_fn(m, s, xb, yb, k)
            p = eqx.filter(m, eqx.is_inexact_array)
            if self.lam_l2 is not None and self.lam_l2 > 0.0:
                base = base + l2_penalty(p, self.lam_l2)
            return base, new_s

        if self._dp_engine is None:
            (_, _), grads = eqx.filter_value_and_grad(loss_with_reg, has_aux=True)(
                model, state, X, y, key
            )
            return grads
        else:
            # If DP is enabled, the engine performs clipping+noise as configured
            grads, _ = self._dp_engine.noisy_grad(
                loss_with_reg, model, state, X, y, key=key
            )
            return grads

    def _sample_batch(
        self, rng: PRNG, X: Array, y: Array, B: Optional[int]
    ) -> Tuple[PRNG, Array, Array]:
        n = int(X.shape[0])
        if B is None or B >= n:
            return rng, X, y
        rng, sub = jr.split(rng)
        idx = jr.choice(sub, n, shape=(int(B),), replace=False)
        return rng, X[idx], y[idx]

    # ---- main ----
    def fit(
        self,
        parts: List[Tuple[Array, Array]],  # per-worker (X_i, y_i)
        X_full: Array,
        y_full: Array,
        *,
        max_time: int = 1000,  # number of event ticks
        seed: int = 0,
    ) -> Dict[str, List[float] | List[int]]:
        n_workers = len(parts)
        sizes = [int(parts[i][0].shape[0]) for i in range(n_workers)]
        total = float(sum(sizes))
        weights = (
            [s / (total + 1e-12) for s in sizes]
            if self.weight_by_data
            else [1.0 / n_workers] * n_workers
        )

        # init model & state
        model, state = eqx.nn.make_with_state(self.model_init_fn)(self.key)
        self.model, self.state = model, state
        params, static = _partition_params(model)

        # EF residuals
        self._residuals = [
            jax.tree_util.tree_map(jnp.zeros_like, params) for _ in range(n_workers)
        ]

        # inflight jobs per worker: snapshot params/state, due time, etc.
        inflight: List[Optional[dict]] = [None] * n_workers

        rng = jr.PRNGKey(seed)
        hist_loss: List[float] = []
        hist_staleness: List[int] = []
        hist_updates: List[int] = []
        num_updates = 0

        def maybe_launch(worker_id: int, tnow: int):
            nonlocal inflight, params, static, state
            if inflight[worker_id] is not None:
                return
            theta_snap = params  # snapshot at launch
            state_snap = state
            delay = int(self.delay_sampler(worker_id, tnow))
            due = tnow + max(1, delay)
            inflight[worker_id] = {
                "params_snapshot": theta_snap,
                "state_snapshot": state_snap,
                "due_time": due,
                "wid": worker_id,
                "t_launch": tnow,
            }

        for t in range(max_time):
            # launch idle workers
            for i in range(n_workers):
                maybe_launch(i, t)

            # complete arrivals
            completed_idxs = [
                i
                for i, job in enumerate(inflight)
                if (job is not None and job["due_time"] == t)
            ]
            for i in completed_idxs:
                job = inflight[i]
                wid = job["wid"]
                t_launch = job["t_launch"]
                staleness = t - t_launch
                hist_staleness.append(int(staleness))
                model_snap = _combine_params(job["params_snapshot"], static)
                state_snap = job["state_snapshot"]
                Xi, yi = parts[wid]

                rng, sub = jr.split(rng)

                if self.aggregation == "grad":
                    # one minibatch gradient from snapshot
                    rng, Xb, yb = self._sample_batch(sub, Xi, yi, self.client_batch)
                    grads = self._worker_grad(model_snap, state_snap, Xb, yb, sub)

                    # optional post-DP global clipping
                    grads = self._clip_by_global_norm(grads)

                    # compression with EF
                    payload = self._compress_with_residual(
                        wid, grads
                    )  # still a gradient-like payload
                    scale = float(weights[wid]) * float(self._phi(staleness))
                    scaled = jax.tree_util.tree_map(lambda g: scale * g, payload)
                    params = self._opt.apply(params, scaled)

                else:  # aggregation == "delta" (local steps, send delta)
                    theta_loc = job["params_snapshot"]
                    # S local steps of SGD starting from snapshot (on the *snapshot* model)
                    for s in range(max(1, self.local_epochs)):
                        rng, sub = jr.split(rng)
                        rng, Xb, yb = self._sample_batch(sub, Xi, yi, self.client_batch)
                        model_tmp = _combine_params(theta_loc, static)
                        grads = self._worker_grad(model_tmp, state_snap, Xb, yb, sub)
                        # (optional) clip local grad; note DP engine already clips if configured
                        grads = self._clip_by_global_norm(grads)
                        theta_loc = jax.tree_util.tree_map(
                            lambda p, g: p - self.client_lr * g, theta_loc, grads
                        )

                    delta = jax.tree_util.tree_map(
                        lambda p_new, p_old: p_new - p_old,
                        theta_loc,
                        job["params_snapshot"],
                    )
                    delta = self._clip_by_global_norm(delta)
                    payload = self._compress_with_residual(wid, delta)
                    scale = float(weights[wid]) * float(self._phi(staleness))
                    # OPT.apply subtracts; to *add* delta, pass negative payload
                    scaled = jax.tree_util.tree_map(lambda d: -scale * d, payload)
                    params = self._opt.apply(params, scaled)

                inflight[i] = None
                num_updates += 1

            # periodic logging
            if (t + 1) % self.log_every == 0 or (t + 1) == max_time:
                self.model = _combine_params(params, static)
                model_eval = (
                    eqx.nn.inference_mode(self.model, value=True)
                    if self.eval_inference_mode
                    else self.model
                )
                rng, sub = jr.split(rng)
                loss_val = float(
                    eval_step(model_eval, self.state, X_full, y_full, sub, self.loss_fn)
                )
                hist_loss.append(loss_val)
                hist_updates.append(num_updates)

        # finalize
        self.model = _combine_params(params, static)
        self.key = rng
        return {"loss": hist_loss, "updates": hist_updates, "staleness": hist_staleness}


# ------------------------------- MAIN ---------------------------------
if __name__ == "__main__":
    import os
    import numpy as np
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    from quantbayes.stochax.distributed_training.helpers import (
        load_mnist_38,
        plot_global_loss_q3,
        plot_async_loss_vs_updates,
        plot_staleness_hist,
        summarize_histories,
        print_publication_summary,
        latex_table_from_summary,
    )

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

    # split
    n = int(X.shape[0])
    ntr = int(0.8 * n)
    Xtr, Xte = X[:ntr], X[ntr:]
    ytr, yte = y[:ntr], y[ntr:]

    # model
    d = int(X.shape[1])

    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, d: int, key):
            self.lin = eqx.nn.Linear(d, 1, key=key)

        def __call__(self, x, key=None, state=None):
            return self.lin(x).squeeze(-1), state

    model_init = lambda k: LR(d, k)

    # partition across workers
    N_WORKERS = 8
    parts = uniform_partition(Xtr, ytr, N_WORKERS)

    # ----- runs -----
    runs_full = {}
    histories_loss = {}

    # A) baseline async (no staleness, no server opt, no compression)
    trainer_A = AsyncParameterServerEqx(
        model_init_fn=model_init,
        delay_sampler=poisson_delay_sampler(2.0, cap=10),
        server_opt={"name": "none", "lr": 1e-3},
        staleness={"mode": "none"},
        compress={"name": "none"},
        clip_norm=None,
        key=jr.PRNGKey(0),
        log_every=20,
    )
    hA = trainer_A.fit(parts, Xte, yte, max_time=1000, seed=0)
    runs_full["baseline"] = {
        "loss": hA["loss"],
        "updates": hA["updates"],
        "staleness": hA["staleness"],
    }
    histories_loss["baseline"] = {"loss_node1": hA["loss"]}

    # B) staleness-aware (power alpha=0.6)
    trainer_B = AsyncParameterServerEqx(
        model_init_fn=model_init,
        delay_sampler=poisson_delay_sampler(2.0, cap=10),
        server_opt={"name": "none", "lr": 1e-3},
        staleness={"mode": "power", "alpha": 0.6},
        compress={"name": "none"},
        clip_norm=1.0,
        key=jr.PRNGKey(1),
        log_every=20,
    )
    hB = trainer_B.fit(parts, Xte, yte, max_time=1000, seed=1)
    runs_full["staleness"] = {
        "loss": hB["loss"],
        "updates": hB["updates"],
        "staleness": hB["staleness"],
    }
    histories_loss["staleness"] = {"loss_node1": hB["loss"]}

    # C) server Adam + staleness
    trainer_C = AsyncParameterServerEqx(
        model_init_fn=model_init,
        delay_sampler=poisson_delay_sampler(2.0, cap=10),
        server_opt={
            "name": "adam",
            "lr": 5e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        staleness={"mode": "power", "alpha": 0.6},
        compress={"name": "none"},
        clip_norm=1.0,
        key=jr.PRNGKey(2),
        log_every=20,
    )
    hC = trainer_C.fit(parts, Xte, yte, max_time=1000, seed=2)
    runs_full["adam+staleness"] = {
        "loss": hC["loss"],
        "updates": hC["updates"],
        "staleness": hC["staleness"],
    }
    histories_loss["adam+staleness"] = {"loss_node1": hC["loss"]}

    # D) server Adam + staleness + sign compression (error-feedback)
    trainer_D = AsyncParameterServerEqx(
        model_init_fn=model_init,
        delay_sampler=poisson_delay_sampler(2.0, cap=10),
        server_opt={
            "name": "adam",
            "lr": 5e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        staleness={"mode": "power", "alpha": 0.6},
        compress={"name": "sign", "error_feedback": True},
        clip_norm=1.0,
        key=jr.PRNGKey(3),
        log_every=20,
    )
    hD = trainer_D.fit(parts, Xte, yte, max_time=1000, seed=3)
    runs_full["adam+staleness+signEF"] = {
        "loss": hD["loss"],
        "updates": hD["updates"],
        "staleness": hD["staleness"],
    }
    histories_loss["adam+staleness+signEF"] = {"loss_node1": hD["loss"]}

    # ----- plots -----
    os.makedirs("figs_async", exist_ok=True)

    # (1) loss vs event-time (ticks) — same as before
    plot_global_loss_q3(
        histories_loss,
        title="Async PS — loss vs time (ticks)",
        save="figs_async/loss_vs_time.png",
        style="accessible",
    )

    # (2) loss vs updates (fair comparison across variants)
    plot_async_loss_vs_updates(
        {
            "baseline": runs_full["baseline"],
            "staleness": runs_full["staleness"],
            "adam+staleness": runs_full["adam+staleness"],
            "adam+staleness+signEF": runs_full["adam+staleness+signEF"],
        },
        save="figs_async/loss_vs_updates.png",
        style="accessible",
    )

    # (3) staleness histogram
    plot_staleness_hist(
        {
            "baseline": runs_full["baseline"]["staleness"],
            "staleness": runs_full["staleness"]["staleness"],
            "adam+staleness": runs_full["adam+staleness"]["staleness"],
            "adam+staleness+signEF": runs_full["adam+staleness+signEF"]["staleness"],
        },
        save="figs_async/staleness_hist.png",
    )
    print("Saved figs_async/{loss_vs_time,loss_vs_updates,staleness_hist}.png")

    # ----- numeric summary + LaTeX (using loss-vs-time curves for consistency) -----
    os.makedirs("tables", exist_ok=True)
    summary = summarize_histories(histories_loss)
    print("\nNumeric summary (Async PS):")
    print_publication_summary(summary, decimals=4)
    latex = latex_table_from_summary(
        summary,
        decimals=3,
        caption="Async PS variants (loss vs time).",
        label="tab:async",
    )
    with open("tables/async_summary.tex", "w") as f:
        f.write(latex)
    print("Wrote tables/async_summary.tex")

# ------------------------------- MAIN ---------------------------------
# ------------------------------- MAIN ---------------------------------
if __name__ == "__main__":
    """
    Demo: Asynchronous Parameter Server with multiple production knobs.
      • Aggregation='grad'  (FedSGD-like, one minibatch per arrival)
      • Aggregation='delta' (FedAvg-style local training, sends parameter delta)
      • Staleness decay ϕ(s), server Adam (FedOpt), optional EF-Sign compression
    Outputs: figs_async/{loss_vs_time,loss_vs_updates,staleness_hist}.png
    """
    import os
    import numpy as np
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    from quantbayes.stochax.distributed_training.helpers import (
        load_mnist_38,
        plot_global_loss_q3,
        plot_async_loss_vs_updates,
        plot_staleness_hist,
        summarize_histories,
        print_publication_summary,
        latex_table_from_summary,
    )

    # ----- data -----
    try:
        X, y = load_mnist_38(seed=0, flatten=True, standardize=True)
    except Exception:
        rng = np.random.default_rng(0)
        n, d = 6000, 50
        X = rng.standard_normal((n, d)).astype(np.float32)
        w = (rng.standard_normal(d) / np.sqrt(d)).astype(np.float32)
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        y = (rng.random(n) < p).astype(np.float32)
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
        X = (X - mu) / sd
        X, y = jnp.asarray(X), jnp.asarray(y)

    # split
    n = int(X.shape[0])
    ntr = int(0.8 * n)
    Xtr, Xte = X[:ntr], X[ntr:]
    ytr, yte = y[:ntr], y[ntr:]

    # model
    d = int(X.shape[1])

    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, d: int, key):
            self.lin = eqx.nn.Linear(d, 1, key=key)

        def __call__(self, x, key=None, state=None):
            return self.lin(x).squeeze(-1), state

    model_init = lambda k: LR(d, k)

    # partition across workers
    N_WORKERS = 8
    parts = uniform_partition(Xtr, ytr, N_WORKERS)

    # runs to collect
    os.makedirs("figs_async", exist_ok=True)
    histories_loss = {}
    runs = {}

    # A) FedSGD-style (aggregation='grad'), server SGD, no staleness/comp
    trainer_A = AsyncParameterServerEqx(
        model_init_fn=model_init,
        aggregation="grad",
        server_opt={"name": "sgd", "lr": 1e-3},
        delay_sampler=poisson_delay_sampler(2.0, cap=10),
        staleness={"mode": "none"},
        compress={"name": "none"},
        clip_norm=None,
        key=jr.PRNGKey(0),
        log_every=20,
    )
    hA = trainer_A.fit(parts, Xte, yte, max_time=1000, seed=0)
    histories_loss["FedSGD+SGD"] = {"loss_node1": hA["loss"]}
    runs["FedSGD+SGD"] = hA

    # B) FedSGD + server Adam + staleness decay
    trainer_B = AsyncParameterServerEqx(
        model_init_fn=model_init,
        aggregation="grad",
        server_opt={
            "name": "adam",
            "lr": 3e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        delay_sampler=poisson_delay_sampler(2.0, cap=10),
        staleness={"mode": "power", "alpha": 0.6},
        compress={"name": "none"},
        clip_norm=1.0,
        key=jr.PRNGKey(1),
        log_every=20,
    )
    hB = trainer_B.fit(parts, Xte, yte, max_time=1000, seed=1)
    histories_loss["FedSGD+Adam+ϕ(s)"] = {"loss_node1": hB["loss"]}
    runs["FedSGD+Adam+ϕ(s)"] = hB

    # C) FedAvg-style (aggregation='delta'), local S=3 steps, server Adam + ϕ(s)
    trainer_C = AsyncParameterServerEqx(
        model_init_fn=model_init,
        aggregation="delta",
        client={"local_epochs": 3, "lr": 1e-1, "batch": 128},
        server_opt={"name": "adam", "lr": 3e-3},
        delay_sampler=poisson_delay_sampler(2.0, cap=10),
        staleness={"mode": "power", "alpha": 0.6},
        compress={"name": "none"},
        clip_norm=1.0,
        key=jr.PRNGKey(2),
        log_every=20,
    )
    hC = trainer_C.fit(parts, Xte, yte, max_time=1000, seed=2)
    histories_loss["FedAvg(S=3)+Adam+ϕ(s)"] = {"loss_node1": hC["loss"]}
    runs["FedAvg(S=3)+Adam+ϕ(s)"] = hC

    # D) FedAvg + Adam + ϕ(s) + EF-Sign compression
    trainer_D = AsyncParameterServerEqx(
        model_init_fn=model_init,
        aggregation="delta",
        client={"local_epochs": 3, "lr": 1e-1, "batch": 128},
        server_opt={"name": "adam", "lr": 3e-3},
        delay_sampler=poisson_delay_sampler(2.0, cap=10),
        staleness={"mode": "power", "alpha": 0.6},
        compress={"name": "sign", "error_feedback": True},
        clip_norm=1.0,
        key=jr.PRNGKey(3),
        log_every=20,
    )
    hD = trainer_D.fit(parts, Xte, yte, max_time=1000, seed=3)
    histories_loss["FedAvg+Adam+ϕ(s)+SignEF"] = {"loss_node1": hD["loss"]}
    runs["FedAvg+Adam+ϕ(s)+SignEF"] = hD

    # plots
    plot_global_loss_q3(
        histories_loss,
        title="Async PS — loss vs time",
        save="figs_async/loss_vs_time.png",
    )
    plot_async_loss_vs_updates(
        runs, title="Async PS — loss vs updates", save="figs_async/loss_vs_updates.png"
    )
    plot_staleness_hist(
        {k: v["staleness"] for k, v in runs.items()},
        save="figs_async/staleness_hist.png",
    )
    print("Saved figs_async/{loss_vs_time,loss_vs_updates,staleness_hist}.png")

    # numeric summary
    summary = summarize_histories(histories_loss)
    print("\nNumeric summary (Async PS):")
    print_publication_summary(summary, decimals=4)
    latex = latex_table_from_summary(
        summary,
        decimals=3,
        caption="Async PS variants (loss vs time).",
        label="tab:async",
    )
    os.makedirs("tables", exist_ok=True)
    with open("tables/async_summary.tex", "w") as f:
        f.write(latex)
    print("Wrote tables/async_summary.tex")
