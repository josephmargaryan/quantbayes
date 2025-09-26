# dgd_trainer_eqx.py
from __future__ import annotations
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Callable, Any
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from quantbayes.stochax.trainer.train import binary_loss, eval_step, predict

Array = jnp.ndarray
PRNG = jax.Array

__all__ = [
    "DGDTrainerEqx",
    "centralized_gd_eqx",
    "plot_global_loss_q3",
    "plot_consensus_q3",
    "plot_q4_cases",  # add
    "plot_link_replacement",  # add
    "safe_alpha",  # add
]


def _maybe_l2_penalty(pyparams, lam: Optional[float]) -> Array:
    """0.5 * lam * ||params||^2 if lam>0 else 0."""
    if lam is None or lam <= 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return (
        0.5
        * lam
        * sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(pyparams))
    )


def laplacian_from_edges(n_nodes: int, edges: List[Tuple[int, int]]) -> Array:
    """Undirected graph Laplacian L = D - A (nodes indexed 0..n-1)."""
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i, j in edges:
        if i == j:
            continue
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.diag(A.sum(axis=1))
    L = D - A
    return jnp.array(L)


def mixing_matrix(n_nodes: int, edges: List[Tuple[int, int]], alpha: float) -> Array:
    """W = I - alpha * L, used for gossip step. Pick alpha < 1/deg_max."""
    L = laplacian_from_edges(n_nodes, edges)
    I = jnp.eye(n_nodes, dtype=jnp.float32)
    return I - alpha * L


def safe_alpha(edges: List[Tuple[int, int]], n_nodes: int) -> float:
    """Conservative alpha < 1/deg_max."""
    deg = np.zeros(n_nodes, dtype=np.int32)
    for i, j in edges:
        if i != j:
            deg[i] += 1
            deg[j] += 1
    deg_max = int(deg.max()) if n_nodes > 0 else 1
    return 0.49 / max(1, deg_max)


def _partition_params(model: eqx.Module):
    """Split model into (params, static) trees."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    return params, static


def _combine_params(params: Any, static: Any) -> eqx.Module:
    """Recombine (params, static) into a model."""
    return eqx.combine(params, static)


def _flatten_params_l2(pytree: Any) -> Array:
    """Flatten params pytree to a single 1D vector for consensus metrics."""
    leaves = jax.tree_util.tree_leaves(pytree)
    flat = [jnp.ravel(x) for x in leaves if x is not None]
    return (
        jnp.concatenate(flat) if len(flat) > 0 else jnp.zeros((1,), dtype=jnp.float32)
    )


def _tree_weighted_sum(weights: Array, param_list: List[Any]) -> Any:
    """
    Given weights w (n_nodes,) and param_list (length n_nodes), return sum_i w_i * params_i
    """

    def combine(*leaves):
        # leaves is a tuple (leaf_0, leaf_1, ..., leaf_{n-1})
        stacked = jnp.stack(leaves, axis=0)  # (n_nodes, ...)
        return jnp.tensordot(weights, stacked, axes=(0, 0))  # (...)

    return jax.tree_util.tree_map(combine, *param_list)


def _tree_mix(W: Array, param_list: List[Any]) -> List[Any]:
    """
    Apply mixing params' -> W * params (per leaf, linear in node dimension).
    param_list: list of length n_nodes.
    """
    n = len(param_list)
    mixed: List[Any] = []
    for i in range(n):
        mixed_i = _tree_weighted_sum(W[i], param_list)
        mixed.append(mixed_i)
    return mixed


class DGDTrainerEqx:
    """
    P2P DGD:
      θ_{t+1/2} = (I - α L) θ_t
      θ_{t+1}^i  = θ_{t+1/2}^i - γ ∇ L_i(θ_{t+1/2}^i)

    • Mix only PARAMETERS (not BN state).
    • Local step is one full-batch GD update (plain gradient descent).
    • Optional L2 via _maybe_l2_penalty(m, lam) implements 0.5*lam*||θ||^2.
    """

    def __init__(
        self,
        model_init_fn: Callable[[PRNG], eqx.Module],
        n_nodes: int,
        edges: List[Tuple[int, int]],
        lam: Optional[float],
        alpha: float,
        gamma: float,
        T: int,
        loss_fn: Callable,  # e.g., binary_loss
        key: Optional[PRNG] = None,
        *,
        eval_inference_mode: bool = True,  # BN-safe eval
    ):
        self.model_init_fn = model_init_fn
        self.n_nodes = n_nodes
        self.edges = edges
        self.lam = lam if (lam is not None and lam > 0.0) else None
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.T = int(T)
        self.loss_fn = loss_fn
        self.key = key if key is not None else jr.PRNGKey(0)

        # Mixing matrix (gossip)
        self.W = mixing_matrix(n_nodes, edges, self.alpha)
        # (Optional sanity) ensure symmetric & row-stochastic
        rs = jnp.sum(self.W, axis=1)
        assert jnp.allclose(self.W, self.W.T, atol=1e-6), "W must be symmetric"
        assert jnp.allclose(rs, jnp.ones_like(rs), atol=1e-6), "Rows of W must sum to 1"

        self.eval_inference_mode = eval_inference_mode

        # Filled by fit(...)
        self.models: List[eqx.Module] = []
        self.states: List[Any] = []

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

    def _local_fullbatch_grad_and_state(
        self, model: eqx.Module, state: Any, X: Array, y: Array, key: PRNG
    ):
        """Plain full-batch GD (with optional 0.5*lam*||θ||^2)."""

        def loss_with_reg(m: eqx.Module, s: Any, xb: Array, yb: Array, k: PRNG):
            base, new_s = self.loss_fn(m, s, xb, yb, k)
            if self.lam is not None:
                p = eqx.filter(m, eqx.is_inexact_array)
                base = base + _maybe_l2_penalty(p, self.lam)
            return base, new_s

        (loss_val, new_s), grad = eqx.filter_value_and_grad(
            loss_with_reg, has_aux=True
        )(model, state, X, y, key)
        return loss_val, grad, new_s

    def _consensus_distance(self, params_list: List[Any]) -> float:
        vecs = [_flatten_params_l2(p) for p in params_list]
        stack = jnp.stack(vecs, axis=0)
        mean = jnp.mean(stack, axis=0, keepdims=True)
        sq = jnp.sum((stack - mean) ** 2, axis=1)
        return float(jnp.mean(sq))

    def fit(
        self,
        parts: List[Tuple[Array, Array]],  # [(X_i, y_i)] per node
        X_full: Array,
        y_full: Array,
        eval_key: Optional[PRNG] = None,
    ) -> Dict[str, List[float]]:
        if not self.models:
            self._init_models()

        # Work on (params, static) to mix params only
        params_list, static_list = [], []
        for m in self.models:
            p, s = _partition_params(m)
            params_list.append(p)
            static_list.append(s)

        hist_loss_node1: List[float] = []
        hist_consensus: List[float] = []

        rng = self.key if eval_key is None else eval_key
        log_interval = max(1, self.T // 10)

        for t in range(self.T):
            # 1) Gossip
            params_half = _tree_mix(self.W, params_list)
            models_half = [
                _combine_params(ph, s) for ph, s in zip(params_half, static_list)
            ]

            # 2) Local full-batch GD (exactly one step)
            new_params_list = []
            for i in range(self.n_nodes):
                Xi, yi = parts[i]
                rng, sub = jr.split(rng)
                _, gi, new_state_i = self._local_fullbatch_grad_and_state(
                    models_half[i], self.states[i], Xi, yi, sub
                )
                # θ_{t+1}^i = θ_{t+1/2}^i − γ ∇L_i(θ_{t+1/2}^i)
                updated = jax.tree_util.tree_map(
                    lambda p, g: p - self.gamma * g, params_half[i], gi
                )
                new_params_list.append(updated)
                self.states[i] = new_state_i  # keep BN/etc state local to node i

            params_list = new_params_list

            # Logging (global loss of node 1; consensus distance)
            model_node1 = _combine_params(params_list[0], static_list[0])
            model_eval = (
                eqx.nn.inference_mode(model_node1, value=True)
                if self.eval_inference_mode
                else model_node1
            )
            rng, eval_sub = jr.split(rng)
            loss_node1 = float(
                eval_step(
                    model_eval, self.states[0], X_full, y_full, eval_sub, self.loss_fn
                )
            )
            hist_loss_node1.append(loss_node1)
            hist_consensus.append(self._consensus_distance(params_list))

            if ((t + 1) % log_interval == 0) or ((t + 1) == self.T):
                print(
                    f"[{t+1}/{self.T}] loss(node1)={hist_loss_node1[-1]:.4f} cons={hist_consensus[-1]:.3e}",
                    flush=True,
                )

        # Recombine final models
        self.models = [_combine_params(p, s) for p, s in zip(params_list, static_list)]
        self.key = rng
        return {"loss_node1": hist_loss_node1, "consensus_sq": hist_consensus}


# ---------- centralized GD baseline (same model class) ----------


def centralized_gd_eqx(
    model_init_fn: Callable[[PRNG], eqx.Module],
    X: Array,
    y: Array,
    lam: Optional[float],
    gamma: float,
    T: int,
    loss_fn: Callable,
    key: Optional[PRNG] = None,
) -> Tuple[eqx.Module, List[float]]:
    rng = jr.PRNGKey(0) if key is None else key
    model, state = eqx.nn.make_with_state(model_init_fn)(rng)

    def loss_with_reg(m: eqx.Module, s: Any, xb: Array, yb: Array, k: PRNG):
        base, new_s = loss_fn(m, s, xb, yb, k)
        if lam is not None and lam > 0.0:
            p = eqx.filter(m, eqx.is_inexact_array)
            base = base + _maybe_l2_penalty(p, lam)  # 0.5*lam*||θ||^2
        return base, new_s

    losses: List[float] = []
    for _ in range(T):
        rng, sub = jr.split(rng)
        (tot, new_state), grads = eqx.filter_value_and_grad(
            loss_with_reg, has_aux=True
        )(model, state, X, y, sub)
        params, static = _partition_params(model)
        new_params = jax.tree_util.tree_map(lambda p, g: p - gamma * g, params, grads)
        model = _combine_params(new_params, static)
        state = new_state

        # Report base loss (no penalty) for comparability
        rng, esub = jr.split(rng)
        loss_val = float(
            eval_step(
                eqx.nn.inference_mode(model, value=True), state, X, y, esub, loss_fn
            )
        )
        losses.append(loss_val)

    return model, losses


def plot_global_loss_q3(
    histories: dict,
    centralized: list[float] | None = None,
    title="Global training loss (node 1)",
    save: str | None = None,
):
    """
    histories: {"line": {"loss_node1": [...]}, "ring": {...}, "star": {...}}
    centralized: list of global losses from centralized GD (optional)
    """
    plt.figure(figsize=(7.2, 4.2))
    if centralized is not None:
        plt.plot(centralized, label="centralized GD", linewidth=3)
    for name, hist in histories.items():
        plt.plot(hist["loss_node1"], label=f"DGD - {name}", linewidth=2)
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("global training loss (node 1)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=180)
    plt.show()


def plot_consensus_q3(
    histories: dict, title="Consensus distance", save: str | None = None
):
    """
    histories: {"line": {"consensus_sq": [...]}, "ring": {...}, "star": {...}}
    """
    plt.figure(figsize=(7.2, 4.2))
    for name, hist in histories.items():
        plt.plot(hist["consensus_sq"], label=name, linewidth=2)
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("squared consensus distance")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=180)
    plt.show()


def plot_q4_cases(
    hist_C: dict,
    hist_D: dict,
    title_prefix="Global training loss (node 1)",
    save: str | None = None,
):
    """Compare Case C vs Case D for a fixed topology."""
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(hist_C["loss_node1"], label="Case C", linewidth=2)
    plt.plot(hist_D["loss_node1"], label="Case D", linewidth=2)
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("global training loss (node 1)")
    plt.title(f"{title_prefix}: Case C vs Case D")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=180)
    plt.show()


def plot_link_replacement(
    best_hist: dict,
    best_edges: list[tuple[int, int]],
    title="Case C — best link replacement",
    save: str | None = None,
):
    """Show the best-performing link replacement run (already selected)."""
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(best_hist["loss_node1"], label=f"{best_edges}", linewidth=2)
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("global training loss (node 1)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=180)
    plt.show()


if __name__ == "__main__":
    import numpy as np
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    # If you placed the trainer in a module, uncomment this import:
    # from dgd_trainer_eqx import DGDTrainerEqx, centralized_gd_eqx, safe_alpha
    from quantbayes.stochax.trainer.train import binary_loss, eval_step

    # ------------------------------
    # 1) Synthetic binary dataset
    # ------------------------------
    rng = np.random.RandomState(0)
    n_total, d = 4000, 50
    X = rng.randn(n_total, d).astype(np.float32)
    w_true = (rng.randn(d) / np.sqrt(d)).astype(np.float32)
    logits = X @ w_true
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(n_total) < p).astype(np.float32)

    # Shuffle + split 80/20
    idx = rng.permutation(n_total)
    X, y = X[idx], y[idx]
    n_train = int(0.8 * n_total)
    X_tr_np, X_te_np = X[:n_train], X[n_train:]
    y_tr_np, y_te_np = y[:n_train], y[n_train:]

    # Standardize using train stats
    mu = X_tr_np.mean(axis=0, keepdims=True)
    sd = X_tr_np.std(axis=0, keepdims=True) + 1e-8
    X_tr_np = (X_tr_np - mu) / sd
    X_te_np = (X_te_np - mu) / sd

    X_tr = jnp.array(X_tr_np)
    y_tr = jnp.array(y_tr_np)
    X_te = jnp.array(X_te_np)
    y_te = jnp.array(y_te_np)

    # ------------------------------
    # 2) Simple Equinox logistic model
    # ------------------------------
    class LR(eqx.Module):
        lin: eqx.nn.Linear

        def __init__(self, key):
            self.lin = eqx.nn.Linear(d, 1, key=key)  # bias=True by default

        def __call__(self, x, key, state):
            return self.lin(x), state  # (logits, state)

    def model_init_fn(key: jax.Array) -> eqx.Module:
        return LR(key)

    # ------------------------------
    # 3) Uniform partition across 4 nodes
    # ------------------------------
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

    parts = uniform_partition(X_tr, y_tr, 4)

    # ------------------------------
    # 4) Topologies + hyperparameters
    # ------------------------------
    edges_line = [(0, 1), (1, 2), (2, 3)]
    edges_ring = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edges_star = [(3, 0), (3, 1), (3, 2)]

    lam = 1e-3
    alpha_line = safe_alpha(edges_line, 4)
    alpha_ring = safe_alpha(edges_ring, 4)
    alpha_star = safe_alpha(edges_star, 4)

    # Rough gamma from smoothness: L_smooth ≈ 0.25 * λ_max(X^T X / n) + lam
    def estimate_gamma(X: jnp.ndarray, lam: float) -> float:
        n = X.shape[0]
        XtX = (X.T @ X) / n
        v = jnp.ones((XtX.shape[0],), dtype=X.dtype)
        for _ in range(25):
            v = XtX @ v
            v = v / (jnp.linalg.norm(v) + 1e-12)
        lam_max = float(v @ (XtX @ v))
        L_smooth = 0.25 * lam_max + lam
        return 0.9 / L_smooth

    gamma = float(estimate_gamma(X_tr, lam))
    T = 300

    # ------------------------------
    # 5) Train DGD per topology (node 1 = index 0) + centralized baseline
    # ------------------------------
    trainer_line = DGDTrainerEqx(
        model_init_fn,
        4,
        edges_line,
        lam,
        alpha_line,
        gamma,
        T,
        binary_loss,
        jr.PRNGKey(0),
    )
    trainer_ring = DGDTrainerEqx(
        model_init_fn,
        4,
        edges_ring,
        lam,
        alpha_ring,
        gamma,
        T,
        binary_loss,
        jr.PRNGKey(1),
    )
    trainer_star = DGDTrainerEqx(
        model_init_fn,
        4,
        edges_star,
        lam,
        alpha_star,
        gamma,
        T,
        binary_loss,
        jr.PRNGKey(2),
    )

    hist_line = trainer_line.fit(parts, X_tr, y_tr, eval_key=jr.PRNGKey(10))
    hist_ring = trainer_ring.fit(parts, X_tr, y_tr, eval_key=jr.PRNGKey(11))
    hist_star = trainer_star.fit(parts, X_tr, y_tr, eval_key=jr.PRNGKey(12))

    _, loss_cgd = centralized_gd_eqx(
        model_init_fn, X_tr, y_tr, lam, gamma, T, binary_loss, jr.PRNGKey(123)
    )

    # ------------------------------
    # 6) Automatic plots
    # ------------------------------
    histories = {"line": hist_line, "ring": hist_ring, "star": hist_star}
    plot_global_loss_q3(
        histories,
        centralized=loss_cgd,
        title="Global training loss vs iteration (synthetic)",
    )
    plot_consensus_q3(histories, title="Consensus distance vs iteration (synthetic)")
