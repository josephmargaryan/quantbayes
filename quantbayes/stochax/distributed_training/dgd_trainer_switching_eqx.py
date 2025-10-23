# dgd_trainer_switching_eqx.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, Any
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.trainer.train import binary_loss, eval_step
from quantbayes.stochax.privacy.dp import DPSGDConfig, DPPrivacyEngine

from quantbayes.stochax.distributed_training.dgd import (
    mixing_matrix,  # W = I - αL (symmetric if α is scalar and graph undirected)
    safe_alpha as _safe_alpha_ha4,
    _partition_params,
    _combine_params,
)

Array = jnp.ndarray
PRNG = jax.Array
BatchSchedule = Callable[[int, int, int], int]  # B = sched(t, node_index, n_i)

__all__ = ["DGDTrainerSwitchingEqx"]


def _flatten_params_l2(pytree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(pytree)
    flat = [jnp.ravel(x) for x in leaves if x is not None]
    return jnp.concatenate(flat) if flat else jnp.zeros((1,), dtype=jnp.float32)


def _tree_weighted_sum(weights: Array, param_list: List[Any]) -> Any:
    def combine(*leaves):
        stacked = jnp.stack(leaves, axis=0)  # (n, ...)
        return jnp.tensordot(weights, stacked, axes=(0, 0))

    return jax.tree_util.tree_map(combine, *param_list)


def _tree_mix(W: Array, param_list: List[Any]) -> List[Any]:
    n = len(param_list)
    return [_tree_weighted_sum(W[i], param_list) for i in range(n)]


class DGDTrainerSwitchingEqx:
    """
    P2P DGD with switching topologies (odd/even) and optional SGD:
        θ_{t+1/2} = W_t θ_t,
        θ_{t+1}^i = θ_{t+1/2}^i - γ_t * g_i(θ_{t+1/2}^i; B_t^i)

    • If batch_schedule is None → full-batch GD (g_i = ∇L_i on full set).
    • Else → DSGD with per-node mini-batches B_t^i.
    • Returns loss (node 1) and consensus distance history; also sample counts if SGD.
    """

    def __init__(
        self,
        model_init_fn: Callable[[PRNG], eqx.Module],
        n_nodes: int,
        edges_odd: List[Tuple[int, int]],
        edges_even: List[Tuple[int, int]],
        gamma: float | Callable[[int], float],
        T: int,
        *,
        alpha_odd: Optional[float] = None,
        alpha_even: Optional[float] = None,
        loss_fn: Callable = binary_loss,
        batch_schedule: BatchSchedule | None = None,  # NEW (None => GD)
        key: Optional[PRNG] = None,
        eval_inference_mode: bool = True,
        dp_config: Optional[DPSGDConfig] = None,  # optional DP
    ):
        self.model_init_fn = model_init_fn
        self.n_nodes = int(n_nodes)
        self.edges_odd = edges_odd
        self.edges_even = edges_even
        self.gamma = gamma
        self.T = int(T)
        self.loss_fn = loss_fn
        self.batch_schedule = batch_schedule
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
            # Undirected + scalar α ⇒ W symmetric; keep row-stochastic regardless.
            assert jnp.allclose(rs, jnp.ones_like(rs), atol=1e-6)

        self.models: List[eqx.Module] = []
        self.states: List[Any] = []

        self.dp_config = dp_config
        self._dp_engine = DPPrivacyEngine(dp_config) if dp_config else None

    # --- utils ---
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

    def _gamma_t(self, t: int) -> float:
        return float(self.gamma(t)) if callable(self.gamma) else float(self.gamma)

    def _consensus_distance(self, params_list: List[Any]) -> float:
        vecs = [_flatten_params_l2(p) for p in params_list]
        stack = jnp.stack(vecs, axis=0)
        mean = jnp.mean(stack, axis=0, keepdims=True)
        sq = jnp.sum((stack - mean) ** 2, axis=1)
        return float(jnp.mean(sq))

    # --- main ---
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

        hist_loss_node1, hist_cons = [], []

        for t in range(self.T):
            W = self.W_odd if ((t + 1) % 2 == 1) else self.W_even
            params_half = _tree_mix(W, params_list)
            models_half = [
                _combine_params(ph, s) for ph, s in zip(params_half, static_list)
            ]

            step = self._gamma_t(t)
            new_params_list = []
            for i in range(self.n_nodes):
                Xi, yi = parts[i]
                n_i = sizes[i]

                # --- choose full batch vs minibatch ---
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

                # grad (DP optional)
                def loss_fn_local(m, s, xb, yb, k):
                    return self.loss_fn(m, s, xb, yb, k)

                if self._dp_engine is None:
                    (_, new_state_i), grad_i = eqx.filter_value_and_grad(
                        loss_fn_local, has_aux=True
                    )(models_half[i], self.states[i], Xb, yb, rng)
                else:
                    grad_i, new_state_i = self._dp_engine.noisy_grad(
                        loss_fn_local, models_half[i], self.states[i], Xb, yb, key=rng
                    )
                rng, _ = jr.split(rng)

                updated = jax.tree_util.tree_map(
                    lambda p, g: p - step * g, params_half[i], grad_i
                )
                new_params_list.append(updated)
                self.states[i] = new_state_i

            params_list = new_params_list

            # logging: global loss at node 1, consensus
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

        out: Dict[str, Any] = {"loss_node1": hist_loss_node1, "consensus_sq": hist_cons}
        if total_samples is not None:
            out["total_samples_per_node"] = total_samples
        return out


if __name__ == "__main__":
    """
    Demo:, 4 nodes, topology switches every step:
      - odd t: ring (1-2-3-4-1)
      - even t: two disjoint links (1-2 and 3-4)

    We run:
      (A) full-batch GD
      (B) minibatch SGD with a power-law batch schedule
    and print the final losses and show quick plots.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import datasets, transforms

    rng = np.random.RandomState(0)
    n_total, d = 4000, 50
    X = rng.randn(n_total, d).astype(np.float32)
    w_true = (rng.randn(d) / np.sqrt(d)).astype(np.float32)
    logits = X @ w_true
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(n_total) < p).astype(np.float32)

    idx = rng.permutation(n_total)
    X, y = X[idx], y[idx]

    def uniform_partition(X, y, n_nodes=4):
        N = int(X.shape[0])
        base, rem = divmod(N, n_nodes)
        parts = []
        start = 0
        for i in range(n_nodes):
            sz = base + (1 if i < rem else 0)
            parts.append((X[start : start + sz], y[start : start + sz]))
            start += sz
        return parts

    # ---- simple Equinox logistic regression ----
    class Logistic(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, in_dim: int, key: jax.Array):
            self.lin = eqx.nn.Linear(in_dim, 1, key=key)

        def __call__(self, x, key, state):
            # x: [B, d] -> logits: [B]
            logits = self.lin(x)  # [B,1]
            return logits.squeeze(-1), state  # [B]

    def model_init_fn(key: jax.Array) -> eqx.Module:
        return Logistic(X.shape[1], key)

    # ---- batch schedule (power-law growth) ----
    def batch_schedule_powerlaw(b0=8, p=0.7, bmax=256):
        b0 = int(max(1, b0))
        bmax = int(max(b0, bmax))
        p = float(p)

        def sched(t: int, i: int, n_i: int) -> int:
            b = int(np.ceil(b0 * ((t + 1) ** p)))
            return min(b, bmax, n_i)

        return sched

    # ---- build odd/even topologies ----
    # nodes are 0..3
    edges_odd = [(0, 1), (1, 2), (2, 3), (3, 0)]  # ring
    edges_even = [(0, 1), (2, 3)]  # two disjoint links

    # ---- load data & split ----

    parts = uniform_partition(X, y, n_nodes=4)

    # ---- run GD (full batch) ----
    T = 300
    trainer_gd = DGDTrainerSwitchingEqx(
        model_init_fn=model_init_fn,
        n_nodes=4,
        edges_odd=edges_odd,
        edges_even=edges_even,
        gamma=0.1,  # full-batch step size
        T=T,
        batch_schedule=None,  # <-- GD
        key=jr.PRNGKey(0),
    )
    out_gd = trainer_gd.fit(parts, X, y)
    print(f"[GD] final loss(node1) = {out_gd['loss_node1'][-1]:.4f}")

    # ---- run SGD (minibatch) ----
    trainer_sgd = DGDTrainerSwitchingEqx(
        model_init_fn=model_init_fn,
        n_nodes=4,
        edges_odd=edges_odd,
        edges_even=edges_even,
        gamma=lambda t: 0.05,  # a bit smaller for SGD
        T=T,
        batch_schedule=batch_schedule_powerlaw(b0=16, p=0.6, bmax=256),  # <-- SGD
        key=jr.PRNGKey(1),
    )
    out_sgd = trainer_sgd.fit(parts, X, y)
    print(f"[SGD] final loss(node1) = {out_sgd['loss_node1'][-1]:.4f}")
    if "total_samples_per_node" in out_sgd:
        print("[SGD] total samples per node:", out_sgd["total_samples_per_node"])

    # ---- quick plots ----
    plt.figure(figsize=(7, 4))
    plt.plot(out_gd["loss_node1"], label="GD (full-batch)")
    plt.plot(out_sgd["loss_node1"], label="SGD (minibatch)")
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("global train loss @ node 1")
    plt.title("Switching-topology DGD: GD vs SGD")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(out_gd["consensus_sq"], label="GD consensus")
    plt.plot(out_sgd["consensus_sq"], label="SGD consensus")
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("1/n Σ ||θ_i - θ̄||²")
    plt.title("Consensus error over time")
    plt.legend()
    plt.tight_layout()
    plt.show()
