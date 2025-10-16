# quantbayes/stochax/distributed_training/dsgd_trainer_eqx.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, Any
import math
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.distributed_training.dgd import (
    mixing_matrix,
    laplacian_from_edges,
    safe_alpha as _safe_alpha_ha4,
    _partition_params,
    _combine_params,
)

from quantbayes.stochax.trainer.train import binary_loss, eval_step
from quantbayes.stochax.privacy.dp import DPSGDConfig, DPPrivacyEngine


Array = jnp.ndarray
PRNG = jax.Array

__all__ = [
    "safe_alpha",
    "make_batch_schedule_powerlaw",
    "make_batch_schedule_piecewise",
    "DSGDTrainerEqx",
    "DGDTrainerSwitchingEqx",
    "plot_dsgd_global_losses",
    "plot_consensus",
]


# ---------- public re-export ----------
def safe_alpha(edges: List[Tuple[int, int]], n_nodes: int) -> float:
    """Conservative α < 1/deg_max."""
    return _safe_alpha_ha4(edges, n_nodes)


BatchSchedule = Callable[[int, int, int], int]
# signature: B = schedule(t, i, n_i) where t is 0-based iteration,
# i is node id (0,..,n-1), and n_i is local dataset size.


def make_batch_schedule_powerlaw(
    b0: int = 8, p: float = 0.7, bmax: int = 256
) -> BatchSchedule:
    """
    Power-law growth: B_t ≈ min(bmax, ceil(b0 * (t+1)^p)).
    Small noisy batches early; reduce variance later.
    """
    b0 = max(1, int(b0))
    bmax = max(b0, int(bmax))
    p = float(p)

    def schedule(t: int, i: int, n_i: int) -> int:
        b = int(math.ceil(b0 * ((t + 1) ** p)))
        return int(min(b, bmax, n_i))

    return schedule


def make_batch_schedule_piecewise(segments: List[Tuple[int, int]]) -> BatchSchedule:
    """
    Piecewise-constant schedule given as [(T1, B1), (T2, B2), ...].
    For 0<=t<T1 use B1; for T1<=t<T2 use B2; ...; t>=T_last use last B.
    """
    segs = [(max(0, int(T)), max(1, int(B))) for (T, B) in segments]
    segs.sort(key=lambda z: z[0])

    def schedule(t: int, i: int, n_i: int) -> int:
        B = segs[-1][1]
        for Tcut, Bb in segs:
            if t < Tcut:
                B = Bb
                break
        return int(min(B, n_i))

    return schedule


import matplotlib.pyplot as plt


def plot_dsgd_global_losses(
    histories: Dict[str, Dict[str, List[float]]],
    *,
    title: str = "Global training loss (nodes 1 & 4)",
    show_legend: bool = True,
    save: Optional[str] = None,
):
    plt.figure(figsize=(7.6, 4.4))
    for name, h in histories.items():
        plt.plot(h["loss_node1"], label=f"{name} — node 1", linewidth=2)
        plt.plot(h["loss_node4"], label=f"{name} — node 4", linestyle="--", linewidth=2)
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("global training loss")
    plt.title(title)
    if show_legend:
        plt.legend(ncol=2)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=180)
    plt.show()


def plot_consensus(
    histories: Dict[str, Dict[str, List[float]]],
    *,
    title: str = "Squared consensus distance",
    save: Optional[str] = None,
):
    plt.figure(figsize=(7.2, 4.2))
    for name, h in histories.items():
        plt.plot(h["consensus_sq"], label=name, linewidth=2)
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("1/n ∑ ||θ_i - θ̄||²")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=180)
    plt.show()


def _flatten_params_l2(pytree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(pytree)
    flat = [jnp.ravel(x) for x in leaves if x is not None]
    return (
        jnp.concatenate(flat) if len(flat) > 0 else jnp.zeros((1,), dtype=jnp.float32)
    )


def _tree_weighted_sum(weights: Array, param_list: List[Any]) -> Any:
    def combine(*leaves):
        stacked = jnp.stack(leaves, axis=0)  # (n, ...)
        return jnp.tensordot(weights, stacked, axes=(0, 0))

    return jax.tree_util.tree_map(combine, *param_list)


def _tree_mix(W: Array, param_list: List[Any]) -> List[Any]:
    n = len(param_list)
    mixed: List[Any] = []
    for i in range(n):
        mixed_i = _tree_weighted_sum(W[i], param_list)
        mixed.append(mixed_i)
    return mixed


class DSGDTrainerEqx:
    """
    P2P DSGD:
    ```math
        θ_{t+1/2}  = W θ_t              (gossip with W = I - \alpha L)
        θ_{t+1}^i  = θ_{t+1/2}^i - γ_t * g_i(θ_{t+1/2}^i; B_t^i)
    ```

    where g_i is the stochastic gradient over a mini-batch B_t^i (uniform without replacement).

    * Accepts an arbitrary per-node batch-size schedule: B_t^i = schedule(t, i, |S_i|).
    * Tracks total samples ∑_t |B_t^i| used by each node.
    * Logs global training loss for nodes 1 and 4 (0- and 3-index in code).
    * Optionally allow learning-rate schedule \ghamma_t (callable), else constant γ.
    """

    def __init__(
        self,
        model_init_fn: Callable[[PRNG], eqx.Module],
        n_nodes: int,
        edges: List[Tuple[int, int]],
        alpha: float,
        gamma: float | Callable[[int], float],
        T: int,
        *,
        loss_fn: Callable = binary_loss,
        batch_schedule: BatchSchedule | None = None,
        key: Optional[PRNG] = None,
        eval_inference_mode: bool = True,
        dp_config: Optional[DPSGDConfig] = None,  # NEW
    ):
        self.model_init_fn = model_init_fn
        self.n_nodes = int(n_nodes)
        self.edges = edges
        self.alpha = float(alpha)
        self.gamma = gamma
        self.T = int(T)
        self.loss_fn = loss_fn
        self.batch_schedule = batch_schedule or make_batch_schedule_powerlaw()
        self.key = key if key is not None else jr.PRNGKey(0)
        self.eval_inference_mode = bool(eval_inference_mode)
        self.W = mixing_matrix(self.n_nodes, self.edges, self.alpha)
        rs = jnp.sum(self.W, axis=1)
        assert jnp.allclose(self.W, self.W.T, atol=1e-6)
        assert jnp.allclose(rs, jnp.ones_like(rs), atol=1e-6)
        self.models: List[eqx.Module] = []
        self.states: List[Any] = []
        self.dp_config = dp_config  # NEW
        self._dp_engine = DPPrivacyEngine(dp_config) if dp_config else None  # NEW

    def _init_models(self) -> None:
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

    def _consensus_distance(self, params_list: List[Any]) -> float:
        vecs = [_flatten_params_l2(p) for p in params_list]
        stack = jnp.stack(vecs, axis=0)
        mean = jnp.mean(stack, axis=0, keepdims=True)
        sq = jnp.sum((stack - mean) ** 2, axis=1)
        return float(jnp.mean(sq))

    def _gamma_t(self, t: int) -> float:
        if callable(self.gamma):
            return float(self.gamma(t))
        return float(self.gamma)

    def fit(
        self,
        parts: List[Tuple[Array, Array]],
        X_full: Array,
        y_full: Array,
        *,
        eval_key: Optional[PRNG] = None,
    ) -> Dict[str, List[float] | Dict[int, int]]:
        if not self.models:
            self._init_models()

        params_list, static_list = [], []
        for m in self.models:
            p, s = _partition_params(m)
            params_list.append(p)
            static_list.append(s)

        rng = self.key if eval_key is None else eval_key
        total_samples = {i: 0 for i in range(self.n_nodes)}
        hist_loss_node1, hist_loss_node4, hist_cons = [], [], []

        sizes = [int(parts[i][0].shape[0]) for i in range(self.n_nodes)]

        def _gamma_t(t: int) -> float:
            return float(self.gamma(t)) if callable(self.gamma) else float(self.gamma)

        for t in range(self.T):
            params_half = _tree_mix(self.W, params_list)
            models_half = [
                _combine_params(ph, s) for ph, s in zip(params_half, static_list)
            ]

            new_params_list = []
            for i in range(self.n_nodes):
                Xi, yi = parts[i]
                n_i = sizes[i]
                B = self.batch_schedule(t, i, n_i)
                total_samples[i] += B
                rng, sub = jr.split(rng)
                idx = (
                    jnp.arange(n_i)
                    if B >= n_i
                    else jr.choice(sub, n_i, shape=(B,), replace=False)
                )
                Xb, yb = Xi[idx], yi[idx]

                def loss(m, s, xb, yb, k):
                    return self.loss_fn(m, s, xb, yb, k)

                if self._dp_engine is None:
                    (loss_val, new_state_i), grad_i = eqx.filter_value_and_grad(
                        loss, has_aux=True
                    )(models_half[i], self.states[i], Xb, yb, sub)
                else:
                    grad_i, new_state_i = self._dp_engine.noisy_grad(
                        loss, models_half[i], self.states[i], Xb, yb, key=sub
                    )

                step = _gamma_t(t)
                updated = jax.tree_util.tree_map(
                    lambda p, gg: p - step * gg, params_half[i], grad_i
                )
                new_params_list.append(updated)
                self.states[i] = new_state_i

            params_list = new_params_list

            m1 = _combine_params(params_list[0], static_list[0])
            m4 = _combine_params(params_list[3], static_list[3])
            if self.eval_inference_mode:
                m1 = eqx.nn.inference_mode(m1, value=True)
                m4 = eqx.nn.inference_mode(m4, value=True)

            rng, k1 = jr.split(rng)
            loss1 = float(
                eval_step(m1, self.states[0], X_full, y_full, k1, self.loss_fn)
            )
            rng, k4 = jr.split(rng)
            loss4 = float(
                eval_step(m4, self.states[3], X_full, y_full, k4, self.loss_fn)
            )
            hist_loss_node1.append(loss1)
            hist_loss_node4.append(loss4)
            hist_cons.append(self._consensus_distance(params_list))

        self.models = [_combine_params(p, s) for p, s in zip(params_list, static_list)]
        self.key = rng

        return {
            "loss_node1": hist_loss_node1,
            "loss_node4": hist_loss_node4,
            "consensus_sq": hist_cons,
            "total_samples_per_node": total_samples,
        }


class DGDTrainerSwitchingEqx:
    """
    Full-gradient P2P DGD with topology switching.
    Example (HA5 Q4): at odd t use edges_D, at even t use edges_E.
    ```math
    \theta_{t+1/2} = W_t \theta_t,   where W_t corresponds to edges at time t,
    \theta_{t+1}^i = \theta_{t+1/2}^i - \gamma \nabla L_i(\theta_{t+1/2}^i)   (full local gradient).
    ```
    Can pass per-topology α values (alpha_D, alpha_E). If omitted, uses safe_alpha for each.
    """

    def __init__(
        self,
        model_init_fn: Callable[[PRNG], eqx.Module],
        n_nodes: int,
        edges_odd: List[Tuple[int, int]],
        edges_even: List[Tuple[int, int]],
        gamma: float,
        T: int,
        *,
        alpha_odd: Optional[float] = None,
        alpha_even: Optional[float] = None,
        loss_fn: Callable = binary_loss,
        key: Optional[PRNG] = None,
        eval_inference_mode: bool = True,
        dp_config: Optional[DPSGDConfig] = None,  # NEW
    ):
        self.model_init_fn = model_init_fn
        self.n_nodes = int(n_nodes)
        self.edges_odd = edges_odd
        self.edges_even = edges_even
        self.gamma = float(gamma)
        self.T = int(T)
        self.loss_fn = loss_fn
        self.key = key if key is not None else jr.PRNGKey(0)
        self.eval_inference_mode = bool(eval_inference_mode)

        self.alpha_odd = (
            float(alpha_odd)
            if alpha_odd is not None
            else _safe_alpha_ha4(edges_odd, n_nodes)
        )
        self.alpha_even = (
            float(alpha_even)
            if alpha_even is not None
            else _safe_alpha_ha4(edges_even, n_nodes)
        )
        self.W_odd = mixing_matrix(n_nodes, edges_odd, self.alpha_odd)
        self.W_even = mixing_matrix(n_nodes, edges_even, self.alpha_even)
        for W in (self.W_odd, self.W_even):
            rs = jnp.sum(W, axis=1)
            assert jnp.allclose(W, W.T, atol=1e-6)
            assert jnp.allclose(rs, jnp.ones_like(rs), atol=1e-6)

        self.models: List[eqx.Module] = []
        self.states: List[Any] = []
        self.dp_config = dp_config  # NEW
        self._dp_engine = DPPrivacyEngine(dp_config) if dp_config else None  # NEW

    def _init_models(self) -> None:
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

    def _consensus_distance(self, params_list: List[Any]) -> float:
        vecs = [_flatten_params_l2(p) for p in params_list]
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
    ) -> Dict[str, List[float]]:
        if not self.models:
            self._init_models()

        params_list, static_list = [], []
        for m in self.models:
            p, s = _partition_params(m)
            params_list.append(p)
            static_list.append(s)

        rng = self.key if eval_key is None else eval_key
        hist_loss_node1, hist_cons = [], []

        for t in range(self.T):
            W = self.W_odd if ((t + 1) % 2 == 1) else self.W_even

            params_half = _tree_mix(W, params_list)
            models_half = [
                _combine_params(ph, s) for ph, s in zip(params_half, static_list)
            ]

            new_params_list = []
            for i in range(self.n_nodes):
                Xi, yi = parts[i]
                rng, sub = jr.split(rng)

                def loss(m, s, xb, yb, k):
                    return self.loss_fn(m, s, xb, yb, k)

                if self._dp_engine is None:
                    (_, new_state_i), grad_i = eqx.filter_value_and_grad(
                        loss, has_aux=True
                    )(models_half[i], self.states[i], Xi, yi, sub)
                else:
                    grad_i, new_state_i = self._dp_engine.noisy_grad(
                        loss, models_half[i], self.states[i], Xi, yi, key=sub
                    )

                updated = jax.tree_util.tree_map(
                    lambda p, g: p - self.gamma * g, params_half[i], grad_i
                )
                new_params_list.append(updated)
                self.states[i] = new_state_i
            params_list = new_params_list

            m1 = _combine_params(params_list[0], static_list[0])
            if self.eval_inference_mode:
                m1 = eqx.nn.inference_mode(m1, value=True)
            rng, k1 = jr.split(rng)
            loss1 = float(
                eval_step(m1, self.states[0], X_full, y_full, k1, self.loss_fn)
            )
            hist_loss_node1.append(loss1)
            hist_cons.append(self._consensus_distance(params_list))

        self.models = [_combine_params(p, s) for p, s in zip(params_list, static_list)]
        self.key = rng
        return {"loss_node1": hist_loss_node1, "consensus_sq": hist_cons}


if __name__ == "__main__":

    import numpy as np
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    from quantbayes.stochax.trainer.train import binary_loss, eval_step

    rng = np.random.RandomState(0)
    n_total, d = 4000, 50
    X = rng.randn(n_total, d).astype(np.float32)
    w_true = (rng.randn(d) / np.sqrt(d)).astype(np.float32)
    logits = X @ w_true
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(n_total) < p).astype(np.float32)

    # Shuffle, split, standardize on train
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

    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, key):
            self.lin = eqx.nn.Linear(d, 1, key=key)

        def __call__(self, x, key, state):
            return self.lin(x), state

    def model_init_fn(key: jax.Array) -> eqx.Module:

        return LR(key)

    def uniform_partition(X: jnp.ndarray, y: jnp.ndarray, n_nodes: int):
        N = X.shape[0]
        base, rem = divmod(N, n_nodes)
        parts = []
        start = 0
        for i in range(n_nodes):
            size = base + (1 if i < rem else 0)
            parts.append((X[start : start + size], y[start : start + size]))
            start += size
        return parts

    n_nodes = 4
    parts = uniform_partition(X_tr, y_tr, n_nodes)

    edges_ring = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edges_star0 = [(0, 1), (0, 2), (0, 3)]
    alpha = safe_alpha(edges_ring, n_nodes)

    def estimate_gamma_full(X: jnp.ndarray) -> float:
        """Heuristic for logistic loss: L ≈ 0.25 * λ_max((X^T X)/n)."""
        n = X.shape[0]
        XtX = (X.T @ X) / max(1, n)
        v = jnp.ones((XtX.shape[0],), dtype=X.dtype)
        for _ in range(25):
            v = XtX @ v
            v = v / (jnp.linalg.norm(v) + 1e-12)
        lam_max = float(v @ (XtX @ v))
        L_smooth = 0.25 * lam_max
        return 0.9 / max(L_smooth, 1e-8)

    gamma_full = estimate_gamma_full(X_tr)
    gamma_sgd = 0.5 * gamma_full
    T = 200

    dsgd = DSGDTrainerEqx(
        model_init_fn=model_init_fn,
        n_nodes=n_nodes,
        edges=edges_ring,
        alpha=alpha,
        gamma=gamma_sgd,
        T=T,
        loss_fn=binary_loss,
        batch_schedule=make_batch_schedule_powerlaw(b0=16, p=0.7, bmax=128),
        key=jr.PRNGKey(0),
    )

    hist_dsgd = dsgd.fit(parts, X_tr, y_tr, eval_key=jr.PRNGKey(10))

    assert len(hist_dsgd["loss_node1"]) == T
    assert len(hist_dsgd["loss_node4"]) == T
    assert len(hist_dsgd["consensus_sq"]) == T
    import numpy as _np

    assert _np.isfinite(hist_dsgd["loss_node1"][-1])
    assert _np.isfinite(hist_dsgd["loss_node4"][-1])
    assert _np.isfinite(hist_dsgd["consensus_sq"][-1])

    print(
        f"[DSGD] last loss node1={hist_dsgd['loss_node1'][-1]:.4f} | "
        f"node4={hist_dsgd['loss_node4'][-1]:.4f} | "
        f"consensus={hist_dsgd['consensus_sq'][-1]:.3e}"
    )
    print("[DSGD] total samples per node:", hist_dsgd["total_samples_per_node"])

    dgd_sw = DGDTrainerSwitchingEqx(
        model_init_fn=model_init_fn,
        n_nodes=n_nodes,
        edges_odd=edges_ring,
        edges_even=edges_star0,
        gamma=gamma_full,
        T=T,
        loss_fn=binary_loss,
        key=jr.PRNGKey(1),
    )

    hist_sw = dgd_sw.fit(parts, X_tr, y_tr, eval_key=jr.PRNGKey(11))

    assert len(hist_sw["loss_node1"]) == T
    assert len(hist_sw["consensus_sq"]) == T
    assert _np.isfinite(hist_sw["loss_node1"][-1])
    assert _np.isfinite(hist_sw["consensus_sq"][-1])

    print(
        f"[Switching DGD] last loss node1={hist_sw['loss_node1'][-1]:.4f} | "
        f"consensus={hist_sw['consensus_sq'][-1]:.3e}"
    )

    try:
        plot_dsgd_global_losses(
            {
                "DSGD": {
                    "loss_node1": hist_dsgd["loss_node1"],
                    "loss_node4": hist_dsgd["loss_node4"],
                }
            },
            title="DSGD — Global training loss (nodes 1 & 4)",
        )
        plot_consensus(
            {"DSGD": hist_dsgd, "Switching DGD": hist_sw},
            title="Consensus distance (DSGD vs Switching DGD)",
        )
    except Exception as e:
        print("Plotting skipped:", e)

    m1 = dsgd.models[0]
    s1 = dsgd.states[0]
    m1_eval = eqx.nn.inference_mode(m1, value=True)
    _, test_loss = (
        eval_step(m1_eval, s1, X_te, y_te, jr.PRNGKey(1234), binary_loss),
        None,
    )
    print("[DSGD] test-set eval completed.")
