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
    def __init__(
        self, name="none", lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, momentum=0.9
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
    def _sub(a, b):
        return jax.tree_util.tree_map(lambda x, y: x - y, a, b)

    @staticmethod
    def _scale(a, s):
        return jax.tree_util.tree_map(lambda x: s * x, a)

    def apply(self, params, grad):
        if self.name == "none":
            return self._sub(params, self._scale(grad, self.lr))

        if self.name == "momentum":
            if self.m is None:
                self.m = self._zeros_like(grad)
            self.m = self._add(self._scale(self.m, self.mu), grad)
            return self._sub(params, self._scale(self.m, self.lr))

        if self.name in {"adam"}:
            if self.m is None:
                self.m = self._zeros_like(grad)
            if self.v is None:
                self.v = self._zeros_like(grad)
            self.t += 1
            self.m = jax.tree_util.tree_map(
                lambda m, g: self.b1 * m + (1 - self.b1) * g, self.m, grad
            )
            self.v = jax.tree_util.tree_map(
                lambda v, g: self.b2 * v + (1 - self.b2) * (g * g), self.v, grad
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
    Event-driven Asynchronous Parameter Server (SOTA-ready).

    Additions vs baseline:
      • staleness-aware decay ϕ(s) (power or exp)
      • server FedOpt (none | momentum | adam)
      • gradient clipping (global norm)
      • optional compression 'sign' with error-feedback per worker
      • optional DP on worker gradients (as before)

    Update rule on arrivals:
        θ ← θ - γ_t * w_i * ϕ(staleness) * C( g_i + r_i ) ;   r_i ← (g_i + r_i) - C(...)
    where C is compressor (identity or sign), and r_i residuals (error feedback).
    """

    def __init__(
        self,
        model_init_fn: Callable[[PRNG], eqx.Module],
        gamma: float | Callable[[int], float],
        *,
        loss_fn: Callable = binary_loss,
        lam_l2: Optional[float] = None,
        delay_sampler: Callable[[int, int], int] = uniform_delay_sampler(1, 5),
        batch_size_per_worker: Optional[List[int]] = None,
        key: Optional[PRNG] = None,
        eval_inference_mode: bool = True,
        log_every: int = 20,
        dp_config: Optional[DPSGDConfig] = None,  # optional DP on worker gradients
        # NEW knobs:
        server_opt: Optional[
            Dict[str, float | str]
        ] = None,  # {"name":"none|momentum|adam", "lr":..., "beta1":..., "beta2":..., "eps":..., "momentum":...}
        staleness: Optional[
            Dict[str, float | str]
        ] = None,  # {"mode":"power|exp|none", "alpha":0.6, "lambda":0.1}
        clip_norm: Optional[
            float
        ] = None,  # global norm clipping on worker gradients (before compression)
        compress: Optional[
            Dict[str, Any]
        ] = None,  # {"name":"none|sign", "error_feedback":True}
    ):
        self.model_init_fn = model_init_fn
        self.gamma = gamma
        self.loss_fn = loss_fn
        self.lam_l2 = lam_l2
        self.delay_sampler = delay_sampler
        self.batch_size_per_worker = batch_size_per_worker
        self.key = key if key is not None else jr.PRNGKey(0)
        self.eval_inference_mode = bool(eval_inference_mode)
        self.log_every = int(max(1, log_every))

        # DP engine (shared). If you want per-worker ε accounting, instantiate one per worker.
        self.dp_config = dp_config
        self._dp_engine = DPPrivacyEngine(dp_config) if dp_config else None

        # server optimizer & knobs
        self.server_opt = server_opt or {
            "name": "momentum",
            "lr": 1e-3,
            "momentum": 0.9,
        }
        self._opt = _ServerOpt(**self.server_opt)

        # staleness decay
        st = staleness or {"mode": "power", "alpha": 0.6}
        self.st_mode = str(st.get("mode", "power")).lower()
        self.st_alpha = float(st.get("alpha", 0.6))
        self.st_lambda = float(st.get("lambda", 0.1))

        # grad clipping & compression
        self.clip_norm = None if clip_norm is None else float(clip_norm)
        c = compress or {"name": "none", "error_feedback": True}
        self.compress_name = str(c.get("name", "none")).lower()
        self.error_feedback = bool(c.get("error_feedback", True))

        # server state
        self.model: Optional[eqx.Module] = None
        self.state: Optional[Any] = None

        # error feedback residuals per worker (initialized in fit)
        self._residuals: Optional[List[Any]] = None

    def _gamma_t(self, t: int) -> float:
        return float(self.gamma(t)) if callable(self.gamma) else float(self.gamma)

    def _grad_at(self, model: eqx.Module, state: Any, X: Array, y: Array, key: PRNG):
        """Return gradient wrt model params (state is passed-through aux)."""

        def loss_with_reg(m, s, xb, yb, k):
            base, new_s = self.loss_fn(m, s, xb, yb, k)
            p = eqx.filter(m, eqx.is_inexact_array)
            if self.lam_l2 is not None and self.lam_l2 > 0:
                base = base + l2_penalty(p, self.lam_l2)
            return base, new_s

        (loss_val, _new_state), grads = eqx.filter_value_and_grad(
            loss_with_reg, has_aux=True
        )(model, state, X, y, key)
        return grads

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

    def _clip_by_global_norm(self, grads):
        if self.clip_norm is None or self.clip_norm <= 0:
            return grads
        gnorm = self._global_norm(grads)
        scale = jnp.minimum(1.0, self.clip_norm / (gnorm + 1e-12))
        return jax.tree_util.tree_map(lambda g: g * scale, grads)

    def _compress(self, i: int, grads):
        """Return compressed gradients and update residuals if EF is on."""
        if self.compress_name == "none":
            return grads
        # error feedback: g_eff = g + r_i
        if self.error_feedback:
            r = self._residuals[i]
            g_eff = jax.tree_util.tree_map(lambda g, ri: g + ri, grads, r)
        else:
            g_eff = grads
        if self.compress_name == "sign":
            comp = jax.tree_util.tree_map(jnp.sign, g_eff)
        else:
            raise ValueError(f"Unknown compressor '{self.compress_name}'")
        if self.error_feedback:
            self._residuals[i] = jax.tree_util.tree_map(
                lambda ge, c: ge - c, g_eff, comp
            )
        return comp

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
        weights = [s / (total + 1e-12) for s in sizes]

        # init model & state
        model, state = eqx.nn.make_with_state(self.model_init_fn)(self.key)
        self.model, self.state = model, state
        params, static = _partition_params(model)

        # per-worker batch sizes
        if self.batch_size_per_worker is None:
            B = [sizes[i] for i in range(n_workers)]
        else:
            assert len(self.batch_size_per_worker) == n_workers
            B = [
                min(sizes[i], int(self.batch_size_per_worker[i]))
                for i in range(n_workers)
            ]

        # error-feedback residuals
        self._residuals = [
            jax.tree_util.tree_map(jnp.zeros_like, params) for _ in range(n_workers)
        ]

        # inflight jobs: None or dict with {params_snapshot, state_snapshot, due_time, wid, t_launch}
        inflight: List[Optional[dict]] = [None] * n_workers

        rng = jr.PRNGKey(seed)
        hist_loss: List[float] = []
        hist_staleness: List[int] = []
        hist_updates: List[int] = []
        num_updates = 0

        def maybe_launch(worker_id: int, tnow: int):
            nonlocal inflight, params, static, state, rng
            if inflight[worker_id] is not None:
                return
            theta_snap = params
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

            # apply all completions at time t
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
                n_i = Xi.shape[0]
                rng, sub = jr.split(rng)
                if B[wid] >= n_i:
                    idx = jnp.arange(n_i)
                else:
                    idx = jr.choice(sub, n_i, shape=(B[wid],), replace=False)
                Xb, yb = Xi[idx], yi[idx]

                rng, sub = jr.split(rng)
                # gradient (DP optional)
                if self._dp_engine is None:
                    g_i = self._grad_at(model_snap, state_snap, Xb, yb, sub)
                else:

                    def loss_with_reg(m, s, xb, yb, k):
                        base, new_s = self.loss_fn(m, s, xb, yb, k)
                        if self.lam_l2 is not None and self.lam_l2 > 0:
                            p = eqx.filter(m, eqx.is_inexact_array)
                            base = base + l2_penalty(p, self.lam_l2)
                        return base, new_s

                    g_i, _ = self._dp_engine.noisy_grad(
                        loss_with_reg, model_snap, state_snap, Xb, yb, key=sub
                    )

                # optional clip and compression
                g_i = self._clip_by_global_norm(g_i)
                g_i = self._compress(wid, g_i)

                # apply update with weight, staleness decay, and server optimizer
                phi = self._phi(staleness)
                step = self._gamma_t(t) * float(weights[wid]) * float(phi)
                scaled_grad = jax.tree_util.tree_map(lambda g: step * g, g_i)
                params = self._opt.apply(params, scaled_grad)

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

        out: Dict[str, List[float] | List[int]] = {
            "loss": hist_loss,
            "updates": hist_updates,
            "staleness": hist_staleness,
        }
        return out


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
        gamma=1e-3,
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
        gamma=1e-3,
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
        gamma=1e-3,
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
        gamma=1e-3,
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
