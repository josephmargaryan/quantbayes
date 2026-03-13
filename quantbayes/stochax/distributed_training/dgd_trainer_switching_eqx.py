# quantbayes/stochax/distributed_training/dgd_trainer_switching_eqx.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, Any
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.trainer.train import binary_loss, eval_step
from quantbayes.stochax.privacy.dp import DPSGDConfig, DPPrivacyEngine

from quantbayes.stochax.distributed_training.helpers import (
    mixing_matrix,  # W = I - αL (symmetric if α scalar and graph undirected)
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


def _is_weight_array(x):
    return eqx.is_inexact_array(x) and (getattr(x, "ndim", 0) >= 2)


def _weights_only_l2_penalty(params: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(eqx.filter(params, _is_weight_array))
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in leaves)


class DGDTrainerSwitchingEqx:
    """
    P2P D(G)D with switching topologies and optional SGD.

    Topology selector toggles every `switch_every` steps:
        phase = ((t+1) // switch_every) % 2
        phase==1 → use W_odd, else W_even

    Communication happens every `mix_every` steps. If no communication, identity is applied.

    If `batch_schedule` is None → full-batch GD; else → DSGD.
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
        batch_schedule: BatchSchedule | None = None,  # (None => GD)
        key: Optional[PRNG] = None,
        eval_inference_mode: bool = True,
        dp_config: Optional[DPSGDConfig] = None,  # optional DP
        lam_l2: Optional[float] = None,  # loss-level L2 (weights-only)
        weight_decay: float = 0.0,  # decoupled WD (weights-only)
        mix_every: int = 1,  # communicate every `mix_every` steps
        switch_every: int = 1,  # toggle odd/even every `switch_every` steps
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
            assert jnp.allclose(rs, jnp.ones_like(rs), atol=1e-6)

        self.models: List[eqx.Module] = []
        self.states: List[Any] = []

        self.dp_config = dp_config
        self._dp_engine = DPPrivacyEngine(dp_config) if dp_config else None
        self.lam_l2 = float(lam_l2) if lam_l2 is not None else 0.0
        self.weight_decay = float(weight_decay)
        self.mix_every = int(max(1, mix_every))
        self.switch_every = int(max(1, switch_every))
        assert not (
            self.lam_l2 > 0.0 and self.weight_decay > 0.0
        ), "Choose either lam_l2 (loss-level L2) OR weight_decay (decoupled), not both."

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
            # communicate only when scheduled
            do_mix = (t + 1) % self.mix_every == 0

            if do_mix:
                phase = ((t + 1) // self.switch_every) % 2
                W = self.W_odd if (phase == 1) else self.W_even
                params_half = _tree_mix(W, params_list)
            else:
                params_half = params_list  # identity mixing

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
                    base, new_s = self.loss_fn(m, s, xb, yb, k)
                    if self.lam_l2 > 0.0:
                        base = base + self.lam_l2 * _weights_only_l2_penalty(m)
                    return base, new_s

                rng, gkey = jr.split(rng)
                if self._dp_engine is None:
                    (_, new_state_i), grad_i = eqx.filter_value_and_grad(
                        loss_fn_local, has_aux=True
                    )(models_half[i], self.states[i], Xb, yb, gkey)
                else:
                    grad_i, new_state_i = self._dp_engine.noisy_grad(
                        loss_fn_local, models_half[i], self.states[i], Xb, yb, key=gkey
                    )

                # Optional decoupled WD (weights-only), applied outside the DP gradient
                if self.weight_decay > 0.0:
                    decay = 1.0 - step * self.weight_decay

                    def _shrink(p):
                        return p * decay if _is_weight_array(p) else p

                    params_half_i = jax.tree_util.tree_map(_shrink, params_half[i])
                else:
                    params_half_i = params_half[i]

                updated = jax.tree_util.tree_map(
                    lambda p, g: p - step * g, params_half_i, grad_i
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


# ------------------------------- MAIN ---------------------------------
if __name__ == "__main__":
    import os
    import numpy as np
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    from quantbayes.stochax.distributed_training.helpers import (
        load_mnist_38,
        make_hetero_3v8_parts_no_server,
        ring_edges,
        switching_edges,
        estimate_gamma_logistic,
        make_polynomial_decay,
        plot_global_loss_q3,
        plot_consensus,
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

    # hetero P2P split (no server)
    N_NODES = 4
    parts = make_hetero_3v8_parts_no_server(X, y, n_nodes=N_NODES)
    d = int(X.shape[1])

    # model
    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, d: int, key):
            self.lin = eqx.nn.Linear(d, 1, key=key)

        def __call__(self, x, key=None, state=None):
            return self.lin(x).squeeze(-1), state

    model_init = lambda k: LR(d, k)

    # step-size (constant; you can swap in polynomial decay)
    gamma0 = estimate_gamma_logistic(X, lam_l2=1e-3)
    lr_sched = make_polynomial_decay(gamma0, power=0.0, t0=1.0)

    # odd/even switching topologies
    edges_odd, edges_even = switching_edges(N_NODES)  # helper: (ring, disjoint pairs)

    T = 250
    histories = {}

    # A) switch_every=1, mix_every=1 (switch each step; communicate each step)
    trainer_A = DGDTrainerSwitchingEqx(
        model_init_fn=model_init,
        n_nodes=N_NODES,
        edges_odd=edges_odd,
        edges_even=edges_even,
        gamma=lr_sched,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        mix_every=1,
        switch_every=1,
        key=jr.PRNGKey(0),
        eval_inference_mode=True,
    )
    hA = trainer_A.fit(parts, X, y, eval_key=jr.PRNGKey(101))
    histories["mix1_switch1"] = {
        "loss_node1": hA["loss_node1"],
        "consensus_sq": hA["consensus_sq"],
    }

    # B) switch_every=5, mix_every=1 (hold topology for 5 steps; communicate each step)
    trainer_B = DGDTrainerSwitchingEqx(
        model_init_fn=model_init,
        n_nodes=N_NODES,
        edges_odd=edges_odd,
        edges_even=edges_even,
        gamma=lr_sched,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        mix_every=1,
        switch_every=5,
        key=jr.PRNGKey(1),
        eval_inference_mode=True,
    )
    hB = trainer_B.fit(parts, X, y, eval_key=jr.PRNGKey(102))
    histories["mix1_switch5"] = {
        "loss_node1": hB["loss_node1"],
        "consensus_sq": hB["consensus_sq"],
    }

    # C) switch_every=5, mix_every=5 (hold topology 5 steps; communicate every 5th step)
    trainer_C = DGDTrainerSwitchingEqx(
        model_init_fn=model_init,
        n_nodes=N_NODES,
        edges_odd=edges_odd,
        edges_even=edges_even,
        gamma=lr_sched,
        T=T,
        lam_l2=1e-3,
        weight_decay=0.0,
        mix_every=5,
        switch_every=5,
        key=jr.PRNGKey(2),
        eval_inference_mode=True,
    )
    hC = trainer_C.fit(parts, X, y, eval_key=jr.PRNGKey(103))
    histories["mix5_switch5"] = {
        "loss_node1": hC["loss_node1"],
        "consensus_sq": hC["consensus_sq"],
    }

    # ----- plots -----
    os.makedirs("figs_dgd_switching", exist_ok=True)
    plot_global_loss_q3(
        histories,
        title="DGD Switching — loss",
        save="figs_dgd_switching/loss.png",
        style="accessible",
    )
    plot_consensus(
        histories,
        title="DGD Switching — consensus",
        save="figs_dgd_switching/consensus.png",
        style="accessible",
    )
    print("Saved figs_dgd_switching/{loss,consensus}.png")

    # ----- numeric summary + LaTeX -----
    os.makedirs("tables", exist_ok=True)
    summary = summarize_histories(histories)
    print("\nNumeric summary (DGD switching):")
    print_publication_summary(summary, decimals=4)
    latex = latex_table_from_summary(
        summary,
        decimals=3,
        caption="Switching topologies with separate mix and switch frequencies.",
        label="tab:dgd_switching",
    )
    with open("tables/dgd_switching_summary.tex", "w") as f:
        f.write(latex)
    print("Wrote tables/dgd_switching_summary.tex")
