# quantbayes/stochax/distributed_training/async_ps_trainer_eqx.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, Any
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.trainer.train import binary_loss, eval_step
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


def _maybe_l2_penalty(pyparams, lam: Optional[float]) -> Array:
    if lam is None or lam <= 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return (
        0.5
        * lam
        * sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(pyparams))
    )


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
    tau_min, tau_max = int(tau_min), int(tau_max)
    assert 1 <= tau_min <= tau_max
    rng = np.random.RandomState(0)

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


# ---------- main trainer ----------


class AsyncParameterServerEqx:
    """
    Event-driven Asynchronous Parameter Server.

    • One global parameter vector θ (server).
    • Worker i repeatedly:
         pull θ_snapshot, compute gradient on local data (possibly mini-batch),
         wait a random/bounded delay, push gradient computed at θ_snapshot.
      The server applies updates immediately on arrival:
         θ  ←  θ - γ_t * w_i * ∇f_i(θ_snapshot).

    • Supports:
        - full-batch or mini-batch per worker (batch_size_i)
        - data-size weights w_i = |D_i| / ∑_j |D_j|
        - optional L2 regularization
        - constant or scheduled step size γ_t (float or Callable[t] -> float)
        - arbitrary delay sampler (bounded staleness)
        - optional DP on worker gradients (per-example clip + Gaussian noise)
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
        dp_config: Optional[DPSGDConfig] = None,  # optional DP
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

        # server state
        self.model: Optional[eqx.Module] = None
        self.state: Optional[Any] = None

        # DP engine (shared). If you want per-worker ε accounting, instantiate one per worker.
        self.dp_config = dp_config
        self._dp_engine = DPPrivacyEngine(dp_config) if dp_config else None

    def _gamma_t(self, t: int) -> float:
        return float(self.gamma(t)) if callable(self.gamma) else float(self.gamma)

    def _grad_at(self, model: eqx.Module, state: Any, X: Array, y: Array, key: PRNG):
        """Return gradient wrt model params (state is passed-through aux)."""

        def loss_with_reg(m, s, xb, yb, k):
            base, new_s = self.loss_fn(m, s, xb, yb, k)
            p = eqx.filter(m, eqx.is_inexact_array)
            if self.lam_l2 is not None and self.lam_l2 > 0:
                base = base + _maybe_l2_penalty(p, self.lam_l2)
            return base, new_s

        (loss_val, _new_state), grads = eqx.filter_value_and_grad(
            loss_with_reg, has_aux=True
        )(model, state, X, y, key)
        return grads

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

        # inflight jobs: None or dict with {params_snapshot, state_snapshot, due_time, wid, t_launch}
        inflight: List[Optional[dict]] = [None] * n_workers

        rng = jr.PRNGKey(seed)
        hist_loss: List[float] = []
        hist_staleness: List[int] = []
        hist_updates: List[int] = []
        eps_hist: List[float] = []  # DP epsilon over time (optional)

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

        num_updates = 0
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
                # gradient (DP if requested)
                if self._dp_engine is None:
                    g_i = self._grad_at(model_snap, state_snap, Xb, yb, sub)
                else:

                    def loss_with_reg(m, s, xb, yb, k):
                        base, new_s = self.loss_fn(m, s, xb, yb, k)
                        if self.lam_l2 is not None and self.lam_l2 > 0:
                            p = eqx.filter(m, eqx.is_inexact_array)
                            base = base + _maybe_l2_penalty(p, self.lam_l2)
                        return base, new_s

                    g_i, _ = self._dp_engine.noisy_grad(
                        loss_with_reg, model_snap, state_snap, Xb, yb, key=sub
                    )

                w_i = float(weights[wid])
                step = self._gamma_t(t)
                params = jax.tree_util.tree_map(
                    lambda p, g: p - step * w_i * g, params, g_i
                )

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
                if self._dp_engine is not None:
                    eps_hist.append(self._dp_engine.epsilon())

        # finalize
        self.model = _combine_params(params, static)
        self.key = rng  # store RNG back for reproducibility

        out: Dict[str, List[float] | List[int]] = {
            "loss": hist_loss,
            "updates": hist_updates,
            "staleness": hist_staleness,
        }
        if self._dp_engine is not None:
            out["eps"] = eps_hist  # optional DP epsilon history
        return out


if __name__ == "__main__":
    import numpy as onp
    import numpy as np
    import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    # synthetic binary logistic data
    rng = onp.random.RandomState(0)
    n_total, d = 5000, 30
    X = rng.randn(n_total, d).astype(onp.float32)
    w_true = (rng.randn(d) / np.sqrt(d)).astype(onp.float32)
    logits = X @ w_true
    p = 1.0 / (1.0 + onp.exp(-logits))
    y = (rng.rand(n_total) < p).astype(onp.float32)

    # train/test split + standardize on train
    idx = rng.permutation(n_total)
    X, y = X[idx], y[idx]
    n_train = int(0.8 * n_total)
    X_tr_np, X_te_np = X[:n_train], X[n_train:]
    y_tr_np, y_te_np = y[:n_train], y[n_train:]

    mu = X_tr_np.mean(axis=0, keepdims=True)
    sd = X_tr_np.std(axis=0, keepdims=True) + 1e-8
    X_tr_np = (X_tr_np - mu) / sd
    X_te_np = (X_te_np - mu) / sd

    X_tr = jnp.array(X_tr_np)
    y_tr = jnp.array(y_tr_np)
    X_te = jnp.array(X_te_np)
    y_te = jnp.array(y_te_np)

    # model
    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, key):
            self.lin = eqx.nn.Linear(d, 1, key=key)

        def __call__(self, x, key=None, state=None):
            return self.lin(x), state

    def model_init_fn(key: jax.Array) -> eqx.Module:
        return LR(key)

    # workers and delays
    n_workers = 6
    parts = uniform_partition(X_tr, y_tr, n_workers)

    # bounded staleness τ∈[1,8]
    delay = uniform_delay_sampler(1, 8)

    # step size heuristic for logistic: L ≈ 0.25 * λ_max((X^T X)/n)
    def estimate_gamma(X: jnp.ndarray, lam_l2: float = 0.0, tau0: int = 8) -> float:
        n = X.shape[0]
        XtX = (X.T @ X) / max(1, n)
        v = jnp.ones((XtX.shape[0],), dtype=X.dtype)
        for _ in range(25):
            v = XtX @ v
            v = v / (jnp.linalg.norm(v) + 1e-12)
        lam_max = float(v @ (XtX @ v))
        L = 0.25 * lam_max + lam_l2
        return 0.9 / max(L * (tau0 + 1), 1e-8)  # conservative scaling for staleness

    lam_l2 = 5e-4
    gamma = estimate_gamma(X_tr, lam_l2, tau0=8)

    # Optional DP toggle
    USE_DP = False
    dp_cfg = (
        DPSGDConfig(clipping_norm=1.0, noise_multiplier=1.0, delta=1e-5)
        if USE_DP
        else None
    )

    trainer = AsyncParameterServerEqx(
        model_init_fn=model_init_fn,
        gamma=gamma,
        loss_fn=binary_loss,
        lam_l2=lam_l2,
        delay_sampler=delay,
        batch_size_per_worker=[256] * n_workers,  # SGD; None => full-batch
        key=jr.PRNGKey(0),
        eval_inference_mode=True,
        log_every=10,
        dp_config=dp_cfg,
    )

    hist = trainer.fit(parts, X_tr, y_tr, max_time=600)

    # ---- visuals ----
    updates = np.array(hist["updates"])
    loss = np.array(hist["loss"])
    staleness = np.array(hist["staleness"])

    plt.figure(figsize=(7.6, 4.4))
    plt.plot(updates, loss, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Cumulative async updates applied")
    plt.ylabel("Global training loss (PS)")
    plt.title("Async PS — Loss vs updates")
    plt.tight_layout()
    plt.show()

    if len(staleness) > 0:
        window = max(1, len(staleness) // 30)
        ker = np.ones(window) / window
        smooth = np.convolve(staleness, ker, mode="same")

        plt.figure(figsize=(7.6, 3.8))
        plt.plot(smooth, linewidth=2)
        plt.xlabel("Completion index (events)")
        plt.ylabel("Staleness (moving average)")
        plt.title("Async PS — Staleness over completions")
        plt.tight_layout()
        plt.show()

    if USE_DP and "eps" in hist:
        eps = np.array(hist["eps"])
        plt.figure(figsize=(7.0, 3.8))
        plt.plot(eps, linewidth=2)
        plt.xlabel("log step index")
        plt.ylabel("epsilon (RDP→(ε,δ))")
        plt.title("Privacy ε over time (approx.)")
        plt.tight_layout()
        plt.show()
        print(f"[Async PS + DP] final ε ≈ {eps[-1]:.3f} (δ={dp_cfg.delta})")
