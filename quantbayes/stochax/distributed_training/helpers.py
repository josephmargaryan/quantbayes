# helpers.py
from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional, Callable, Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt

Array = jnp.ndarray
PRNG = jax.Array
BatchSchedule = Callable[[int, int, int], int]  # B = schedule(t, node_index, n_i)
LRSchedule = Callable[[int], float]  # γ_t = lr(t)

__all__ = [
    "make_batch_schedule_powerlaw",
    "make_batch_schedule_piecewise",
    "make_constant_lr",
    "make_polynomial_decay",
    "is_weight_array",
    "weights_only_l2_penalty",
    "l2_penalty",
    "load_mnist_38",
    "make_hetero_3v8_parts_no_server",
    "make_hetero_3v8_parts_with_server",
    "laplacian_from_edges",
    "mixing_matrix",
    "safe_alpha",
    "ring_edges",
    "switching_edges",
    "make_star_with_server_edges",
    "build_partial_star_W",
    "partition_params",
    "combine_params",
    "tree_weighted_sum",
    "tree_mix",
    "tree_mean",
    "flatten_params_l2",
    "consensus_distance",
    "estimate_gamma_logistic",
    "plot_global_loss_q3",
    "plot_dsgd_global_losses",
    "plot_consensus",
    "plot_server_loss",
    "plot_consensus_localgd",
    "plot_q4_cases",
    "plot_link_replacement",
    "plot_async_loss_vs_updates",
    "plot_staleness_hist",
    "plot_rho_vs_bound",
    "summarize_histories",
    "print_publication_summary",
    "_partition_params",
    "_combine_params",
]


def _partition_params(model: eqx.Module):
    return eqx.partition(model, eqx.is_inexact_array)


def _combine_params(params: Any, static: Any) -> eqx.Module:
    return eqx.combine(params, static)


# =========================
# Batch / LR schedules
# =========================


def make_batch_schedule_powerlaw(
    b0: int = 8, p: float = 0.7, bmax: int = 256
) -> BatchSchedule:
    """
    Power-law growth: B_t = min(bmax, ceil(b0 * (t+1)^p)).
    Handy for starting noisy then reducing variance over time.
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
    Piecewise-constant batch schedule specified as [(T1, B1), (T2, B2), ...].
    For 0 <= t < T1 use B1; for T1 <= t < T2 use B2; ...; t >= T_last use last B.
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


def make_constant_lr(gamma: float) -> LRSchedule:
    g = float(gamma)

    def sched(t: int) -> float:
        return g

    return sched


def make_polynomial_decay(
    gamma0: float, power: float = 1.0, t0: float = 1.0
) -> LRSchedule:
    """
    γ_t = gamma0 / (t + t0)^power  (default: 1/t decay).
    """
    g0 = float(gamma0)
    p = float(power)
    tt = float(t0)

    def sched(t: int) -> float:
        return g0 / ((t + tt) ** p)

    return sched


# =========================
# L2 (strong convexity) utilities
# =========================


def is_weight_array(x: Any) -> bool:
    """
    True for floating/complex arrays with ndim >= 2 (e.g., Linear/Conv kernels).
    Excludes 1D biases and norm scale/offset params by construction.
    """
    return eqx.is_inexact_array(x) and (getattr(x, "ndim", 0) >= 2)


def weights_only_l2_penalty(params: Any, lam: float) -> jnp.ndarray:
    """
    0.5 * lam * ||θ_W||^2 over WEIGHTS ONLY (ndim >= 2).
    """
    lam = float(lam)
    if lam <= 0.0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    leaves = jax.tree_util.tree_leaves(eqx.filter(params, is_weight_array))
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return 0.5 * lam * sum(jnp.sum(jnp.square(p)) for p in leaves)


def l2_penalty(params: Any, lam: float, exclude_1d_bias: bool = True) -> jnp.ndarray:
    """
    0.5 * lam * Σ ||leaf||^2, optionally excluding 1D leaves (typical biases).
    Provided for backwards compatibility; prefer weights_only_l2_penalty for theory.
    """
    lam = float(lam)
    if lam <= 0.0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    leaves = [x for x in jax.tree_util.tree_leaves(params) if x is not None]
    if exclude_1d_bias:
        leaves = [x for x in leaves if getattr(x, "ndim", 0) > 1]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return 0.5 * lam * sum(jnp.sum(jnp.square(x)) for x in leaves)


# =========================
# Optional data helpers (MNIST 3 vs 8)
# =========================


def load_mnist_38(
    seed: int = 0, flatten: bool = True, standardize: bool = True
) -> Tuple[Array, Array]:
    """
    Loads MNIST digits {3,8} as a binary task (label 1 for digit 8).
    Requires torchvision to be available.
    """
    try:
        from torchvision import datasets  # type: ignore
    except Exception as e:
        raise ImportError("torchvision is required for load_mnist_38") from e

    ds = datasets.MNIST(root="./data", train=True, download=True, transform=None)
    X = ds.data.numpy().astype(np.float32) / 255.0
    y = ds.targets.numpy().astype(np.int64)
    mask = np.isin(y, [3, 8])
    X = X[mask]
    y = (y[mask] == 8).astype(np.float32)
    if flatten:
        X = X.reshape(X.shape[0], -1)
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    if standardize:
        mu = jnp.mean(X, axis=0, keepdims=True)
        sd = jnp.std(X, axis=0, keepdims=True) + 1e-6
        X = (X - mu) / sd
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_hetero_3v8_parts_no_server(
    X: Array, y: Array, n_nodes: int = 4
) -> List[Tuple[Array, Array]]:
    """
    Two nodes get only digit 3, two nodes only digit 8. No server partition.
    """
    assert n_nodes == 4, "Helper assumes 4 nodes."
    X3, y3 = X[y == 0.0], y[y == 0.0]
    X8, y8 = X[y == 1.0], y[y == 1.0]
    m3 = int(X3.shape[0]) // 2
    m8 = int(X8.shape[0]) // 2
    return [
        (X3[:m3], y3[:m3]),
        (X8[:m8], y8[:m8]),
        (X3[m3:], y3[m3:]),
        (X8[m8:], y8[m8:]),
    ]


def make_hetero_3v8_parts_with_server(
    X: Array, y: Array, n_clients: int = 4
) -> List[Tuple[Array, Array]]:
    """
    Two clients get only digit 3, two only digit 8; server has empty set.
    Returns client_parts + [server_part].
    """
    X3, y3 = X[y == 0.0], y[y == 0.0]
    X8, y8 = X[y == 1.0], y[y == 1.0]
    m3 = int(X3.shape[0]) // 2
    m8 = int(X8.shape[0]) // 2
    parts_cli = [
        (X3[:m3], y3[:m3]),
        (X8[:m8], y8[:m8]),
        (X3[m3:], y3[m3:]),
        (X8[m8:], y8[m8:]),
    ]
    d = int(X.shape[1])
    parts_srv = (jnp.zeros((0, d), X.dtype), jnp.zeros((0,), y.dtype))
    return parts_cli + [parts_srv]


# =========================
# Graphs & mixing
# =========================


def laplacian_from_edges(n_nodes: int, edges: List[Tuple[int, int]]) -> Array:
    """Undirected graph Laplacian L = D - A."""
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i, j in edges:
        if i == j:
            continue
        A[i, j] = 1.0
        A[j, i] = 1.0
    D = np.diag(A.sum(axis=1))
    L = D - A
    return jnp.array(L)


def mixing_matrix(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    alpha: float,
    *,
    lazy: bool = False,
) -> Array:
    """
    Row-stochastic, symmetric DS gossip matrix W = I - αL (undirected).
    If lazy=True, returns 0.5*(I+W) which keeps DS and pushes spectrum into [0,1].
    """
    L = laplacian_from_edges(n_nodes, edges)
    I = jnp.eye(n_nodes, dtype=jnp.float32)
    W = I - float(alpha) * L
    if lazy:
        W = 0.5 * (I + W)
    rs = jnp.sum(W, axis=1)
    assert jnp.allclose(W, W.T, atol=1e-6), "W must be symmetric (undirected)."
    assert jnp.allclose(rs, jnp.ones_like(rs), atol=1e-6), "W must be row-stochastic."
    return W


def safe_alpha(edges: List[Tuple[int, int]], n_nodes: int) -> float:
    """Conservative α < 1/deg_max (slightly bolder for faster mixing while keeping W ≥ 0)."""
    deg = np.zeros(n_nodes, dtype=np.int32)
    for i, j in edges:
        if i != j:
            deg[i] += 1
            deg[j] += 1
    deg_max = int(deg.max()) if n_nodes > 0 else 1
    return 0.95 / max(1, deg_max)


def ring_edges(n: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def switching_edges(n: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    odd = ring_edges(n)
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    return odd, even


def make_star_with_server_edges(
    n_clients: int, server_id: int | None = None
) -> List[Tuple[int, int]]:
    """
    Build star edges for n_clients + 1 total nodes. Server defaults to last index.
    Returns undirected edges: [(server, 0), ..., (server, n_clients-1)].
    """
    if server_id is None:
        server_id = n_clients
    return [(server_id, i) for i in range(n_clients)]


def build_partial_star_W(
    n_total: int, server_id: int, active_clients: List[int], alpha: float
) -> Array:
    """
    Build full-size W = I - α L for a star that connects the server to the given active clients only.
    Inactive clients keep self-identity rows/cols.
    """
    edges = [(server_id, c) for c in active_clients]
    return mixing_matrix(n_total, edges, float(alpha))


# =========================
# Pytree utilities
# =========================


def partition_params(model: eqx.Module) -> Tuple[Any, Any]:
    """Split model into (params, static) trees."""
    return eqx.partition(model, eqx.is_inexact_array)


def combine_params(params: Any, static: Any) -> eqx.Module:
    """Recombine (params, static) into a model."""
    return eqx.combine(params, static)


def tree_weighted_sum(weights: Array, param_list: List[Any]) -> Any:
    """
    Given weights w (n_nodes,) and param_list (length n_nodes), return Σ_i w_i * params_i (leafwise).
    """

    def combine(*leaves):
        stacked = jnp.stack(leaves, axis=0)  # (n_nodes, ...)
        return jnp.tensordot(weights, stacked, axes=(0, 0))

    return jax.tree_util.tree_map(combine, *param_list)


def tree_mix(W: Array, param_list: List[Any]) -> List[Any]:
    """
    Apply linear mixing: params' = W * params (leafwise).
    """
    n = len(param_list)
    return [tree_weighted_sum(W[i], param_list) for i in range(n)]


def tree_mean(param_list: List[Any]) -> Any:
    """
    Leafwise mean across nodes.
    """
    n = len(param_list)
    w = jnp.ones((n,), dtype=jnp.float32) / max(1, n)
    return tree_weighted_sum(w, param_list)


def flatten_params_l2(pytree: Any) -> Array:
    """
    Flatten params pytree to a single 1D vector (for consensus metrics).
    """
    leaves = jax.tree_util.tree_leaves(pytree)
    flat = [jnp.ravel(x) for x in leaves if x is not None]
    return jnp.concatenate(flat) if flat else jnp.zeros((1,), dtype=jnp.float32)


def consensus_distance(
    params_list: List[Any], idxs: Optional[List[int]] = None
) -> float:
    """
    1/|S| Σ_{i∈S} ||θ_i - θ̄||² over selected nodes S (defaults to all).
    """
    if idxs is None:
        idxs = list(range(len(params_list)))
    vecs = [flatten_params_l2(params_list[i]) for i in idxs]
    stack = jnp.stack(vecs, axis=0)
    mean = jnp.mean(stack, axis=0, keepdims=True)
    sq = jnp.sum((stack - mean) ** 2, axis=1)
    return float(jnp.mean(sq))


# =========================
# Step-size estimator (logistic)
# =========================


def estimate_gamma_logistic(
    X: Array, lam_l2: float = 0.0, safety: float = 0.9
) -> float:
    """
    Rough L-smoothness for logistic: L ≈ 0.25 * λ_max((X^T X)/n) + lam_l2.
    Return γ ≈ safety / L.
    """
    n = int(X.shape[0])
    XtX = (X.T @ X) / max(1, n)
    v = jnp.ones((XtX.shape[0],), dtype=X.dtype)
    for _ in range(25):
        v = XtX @ v
        v = v / (jnp.linalg.norm(v) + 1e-12)
    lam_max = float(v @ (XtX @ v))
    L_smooth = 0.25 * lam_max + float(lam_l2)
    return float(safety / max(L_smooth, 1e-8))


# =========================
# Plotting helpers
# =========================

DEFAULT_PLOT_STYLE = "accessible"


def _gen_line_styles(n: int, mode: str = DEFAULT_PLOT_STYLE):
    """
    Generate a list of dict(style kwargs) for n series.
    mode='mono' → black lines with varied dashes & markers (safest).
    mode='accessible' → limited color set avoiding green/orange/purple + varied dashes/markers.
    """
    LINES = ["-", "--", "-.", ":"]
    MARKS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "+"]
    if mode == "mono":
        COLORS = ["#000000"]  # black only
    else:
        # modified Okabe–Ito palette avoiding green/orange/purple
        COLORS = [
            "#000000",
            "#0072B2",
            "#56B4E9",
            "#999999",
            "#F0E442",
        ]  # black, blue, sky, gray, yellow
    styles = []
    for k in range(n):
        styles.append(
            {
                "color": COLORS[k % len(COLORS)],
                "linestyle": LINES[k % len(LINES)],
                "marker": MARKS[k % len(MARKS)],
                "markevery": 12,
                "linewidth": 2.0,
            }
        )
    return styles


def _finalize(ax, title: str, xlabel: str, ylabel: str, legend: bool = True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    if legend:
        ax.legend()
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    plt.tight_layout()


def plot_global_loss_q3(
    histories: Dict[str, Dict[str, List[float]]],
    centralized: Optional[List[float]] = None,
    title: str = "Global training loss (node 1)",
    save: Optional[str] = None,
    style: str = DEFAULT_PLOT_STYLE,
):
    """
    histories: {name: {"loss_node1": [...]}}
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    n_series = sum(1 for h in histories.values() if "loss_node1" in h) + (
        1 if centralized is not None else 0
    )
    styles = _gen_line_styles(n_series, style)
    s = 0
    if centralized is not None:
        ax.plot(centralized, label="centralized GD", **styles[s])
        s += 1
    for name, h in histories.items():
        y = h.get("loss_node1")
        if y is not None:
            ax.plot(y, label=name, **styles[s])
            s += 1
    _finalize(ax, title, xlabel="iteration t", ylabel="global training loss (node 1)")
    if save:
        plt.savefig(save, dpi=240)
    plt.show()


def plot_dsgd_global_losses(
    histories: Dict[str, Dict[str, List[float]]],
    *,
    title: str = "Global training loss (nodes 1 & 4)",
    show_legend: bool = True,
    save: Optional[str] = None,
    style: str = DEFAULT_PLOT_STYLE,
):
    """
    histories: {name: {"loss_node1": [...], "loss_node4": [...]} }
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    series = []
    for name, h in histories.items():
        if "loss_node1" in h:
            series.append((f"{name} — node 1", h["loss_node1"], {}))
        if "loss_node4" in h:
            series.append((f"{name} — node 4", h["loss_node4"], {"linestyle": "--"}))
    styles = _gen_line_styles(len(series), style)
    for (label, y, extra), sty in zip(series, styles):
        sty2 = {**sty, **extra}
        ax.plot(y, label=label, **sty2)
    _finalize(
        ax,
        title,
        xlabel="iteration t",
        ylabel="global training loss",
        legend=show_legend,
    )
    if save:
        plt.savefig(save, dpi=240)
    plt.show()


def plot_consensus(
    histories: Dict[str, Dict[str, List[float]]],
    *,
    title: str = "Squared consensus distance",
    save: Optional[str] = None,
    style: str = DEFAULT_PLOT_STYLE,
):
    """
    histories: {name: {"consensus_sq": [...]} }
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    series = [
        (name, h["consensus_sq"])
        for name, h in histories.items()
        if "consensus_sq" in h
    ]
    styles = _gen_line_styles(len(series), style)
    for (name, y), sty in zip(series, styles):
        ax.plot(y, label=name, **sty)
    _finalize(
        ax,
        title,
        xlabel="iteration t",
        ylabel=r"$\frac{1}{n}\sum_i\|\theta_i-\bar\theta\|^2$",
    )
    if save:
        plt.savefig(save, dpi=240)
    plt.show()


def plot_server_loss(
    histories: Dict[str, Dict[str, List[float]]],
    title: str = "Server global training loss",
    save: Optional[str] = None,
    style: str = DEFAULT_PLOT_STYLE,
):
    """
    histories: {name: {"loss_server": [...]}}
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    series = [
        (name, h["loss_server"]) for name, h in histories.items() if "loss_server" in h
    ]
    styles = _gen_line_styles(len(series), style)
    for (name, y), sty in zip(series, styles):
        ax.plot(y, label=name, **sty)
    _finalize(
        ax, title, xlabel="iteration t", ylabel="global training loss (server model)"
    )
    if save:
        plt.savefig(save, dpi=240)
    plt.show()


def plot_consensus_localgd(
    histories: Dict[str, Dict[str, List[float]]],
    title: str = "Consensus distance",
    save: Optional[str] = None,
    style: str = DEFAULT_PLOT_STYLE,
):
    """
    histories: {name: {"consensus_all": [...], "consensus_clients": [...]} }
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    series = []
    for name, h in histories.items():
        if "consensus_all" in h:
            series.append((f"{name} — all (incl. server)", h["consensus_all"], {}))
        if "consensus_clients" in h:
            series.append(
                (f"{name} — clients only", h["consensus_clients"], {"linestyle": "--"})
            )
    styles = _gen_line_styles(len(series), style)
    for (label, y), sty in zip([(lbl, y) for lbl, y, _ in series], styles):
        # need the matching extra for this (label,y)
        extra = next(ex for (lbl, yy, ex) in series if lbl == label and yy is y)
        sty2 = {**sty, **extra}
        ax.plot(y, label=label, **sty2)
    _finalize(ax, title, xlabel="iteration t", ylabel="squared consensus distance")
    if save:
        plt.savefig(save, dpi=240)
    plt.show()


def plot_q4_cases(
    hist_C: Dict[str, List[float]],
    hist_D: Dict[str, List[float]],
    title_prefix: str = "Global training loss (node 1)",
    save: Optional[str] = None,
    style: str = DEFAULT_PLOT_STYLE,
):
    """Compare Case C vs Case D for a fixed topology."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    series = []
    if "loss_node1" in hist_C:
        series.append(("Case C", hist_C["loss_node1"]))
    if "loss_node1" in hist_D:
        series.append(("Case D", hist_D["loss_node1"]))
    styles = _gen_line_styles(len(series), style)
    for (label, y), sty in zip(series, styles):
        ax.plot(y, label=label, **sty)
    _finalize(
        ax,
        f"{title_prefix}: Case C vs Case D",
        xlabel="iteration t",
        ylabel="global training loss (node 1)",
    )
    if save:
        plt.savefig(save, dpi=240)
    plt.show()


def plot_link_replacement(
    best_hist: Dict[str, List[float]],
    best_edges: List[Tuple[int, int]],
    title: str = "Case C — best link replacement",
    save: Optional[str] = None,
    style: str = DEFAULT_PLOT_STYLE,
):
    """Show the best-performing link replacement run (already selected)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = best_hist.get("loss_node1")
    if y is not None:
        sty = _gen_line_styles(1, style)[0]
        ax.plot(y, label=f"{best_edges}", **sty)
    _finalize(ax, title, xlabel="iteration t", ylabel="global training loss (node 1)")
    if save:
        plt.savefig(save, dpi=240)
    plt.show()


def plot_async_loss_vs_updates(
    runs: Dict[str, Dict[str, List[float]]],
    *,
    title: str = "Async: Loss vs Updates",
    save: Optional[str] = None,
    style: str = "accessible",
):
    """runs[name] expects {"loss": [...], "updates": [...]} from async fit()."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    series = [
        (name, r["updates"], r["loss"])
        for name, r in runs.items()
        if "loss" in r and "updates" in r
    ]
    styles = _gen_line_styles(len(series), style)
    for (name, upd, loss), sty in zip(series, styles):
        ax.plot(upd, loss, label=name, **sty)
    _finalize(ax, title, xlabel="cumulative updates", ylabel="loss")
    if save:
        plt.savefig(save, dpi=240)
    plt.show()


def plot_staleness_hist(
    staleness: Dict[str, List[int]],
    *,
    title: str = "Async: Staleness distribution",
    bins: int = 20,
    save: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, s in staleness.items():
        ax.hist(s, bins=bins, alpha=0.55, label=name)
    ax.set_title(title)
    ax.set_xlabel("staleness (ticks)")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=240)
    plt.show()


def plot_rho_vs_bound(
    rho_hist, xi: float, K: int, *, title="Per-mix contraction", save=None
):
    import matplotlib.pyplot as plt, math

    def T(K, x):
        if x > 1.0:
            return math.cosh(K * math.acosh(x))
        t0, t1 = 1.0, x
        for _ in range(2, K + 1):
            t0, t1 = t1, 2 * x * t1 - t0
        return t1

    bound = 1.0 / max(1e-20, T(K, float(xi))) ** 2
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(rho_hist, label=r"$\hat\rho_t$", linewidth=2)
    ax.axhline(bound, linestyle="--", label=rf"bound $1/T_K(\xi)^2$ (K={K})")
    ax.set_yscale("log")
    ax.set_xlabel("mix events")
    ax.set_ylabel(r"contraction $\Gamma_{t+1/2}/\Gamma_t$")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend()
    if save:
        plt.savefig(save, dpi=240)
    plt.show()


# =========================
# Publication summaries (numeric printouts + LaTeX)
# =========================


def _fmt(x: float, decimals: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    ax = abs(x)
    if (ax > 1e4) or (0 < ax < 1e-4):
        return f"{x:.{decimals}e}"
    return f"{x:.{decimals}f}"


def _summarize_one(h: Dict[str, List[float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # losses
    if "loss_server" in h:
        losses = np.asarray(h.get("loss_server", []), dtype=float)
    else:
        losses = np.asarray(h.get("loss_node1", []), dtype=float)
    if losses.size:
        out["final_loss"] = float(losses[-1])
        tmin = int(np.argmin(losses))
        out["min_loss"] = float(losses[tmin])
        out["t_min_loss"] = tmin + 1

    # consensus (prefers "consensus_sq"; also support all/clients for server)
    if "consensus_sq" in h:
        cons = np.asarray(h["consensus_sq"], dtype=float)
        if cons.size:
            out["final_consensus"] = float(cons[-1])
            tminc = int(np.argmin(cons))
            out["min_consensus"] = float(cons[tminc])
            out["t_min_consensus"] = tminc + 1

    if "consensus_all" in h:
        cons_all = np.asarray(h["consensus_all"], dtype=float)
        if cons_all.size:
            out["final_cons_all"] = float(cons_all[-1])
            out["min_cons_all"] = float(np.min(cons_all))
    if "consensus_clients" in h:
        cons_cli = np.asarray(h["consensus_clients"], dtype=float)
        if cons_cli.size:
            out["final_cons_clients"] = float(cons_cli[-1])
            out["min_cons_clients"] = float(np.min(cons_cli))

    # mixing metrics (rho, K, active)
    rho_seq = None
    if "rho_hist" in h:
        rho_seq = np.asarray(h["rho_hist"], dtype=float)
    elif "rho_star" in h:
        rho_seq = np.asarray(h["rho_star"], dtype=float)

    K = np.asarray(h.get("K_hist", []), dtype=float)
    A = np.asarray(h.get("active_hist", []), dtype=float)

    if K.size:
        mix_mask = K > 0
        out["avg_K_all_steps"] = float(np.mean(K))
        if mix_mask.any():
            out["avg_K_mix_only"] = float(np.mean(K[mix_mask]))
            out["num_mix_steps"] = int(np.sum(mix_mask))

    if A.size:
        out["avg_active_all_steps"] = float(np.mean(A))
        if K.size and (A.size == K.size):
            mix_mask = K > 0
            if mix_mask.any():
                out["avg_active_mix_only"] = float(np.mean(A[mix_mask]))

    if rho_seq is not None and rho_seq.size:
        if K.size == rho_seq.size:
            mask = (K > 0) & np.isfinite(rho_seq) & (rho_seq > 0)
        else:
            mask = np.isfinite(rho_seq) & (rho_seq > 0)
        if np.any(mask):
            r = rho_seq[mask]
            out["geomean_rho"] = float(np.exp(np.mean(np.log(r))))
            out["median_rho"] = float(np.median(r))
            out["last_rho"] = float(r[-1])

    return out


def summarize_histories(
    histories: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Dict[str, float]]:
    return {name: _summarize_one(h) for name, h in histories.items()}


def print_publication_summary(
    summary: Dict[str, Dict[str, float]],
    *,
    decimals: int = 4,
    columns: Optional[List[str]] = None,
) -> None:
    """
    Pretty-print numeric results for publications.
    """
    # default column order (only show those that exist in any row)
    preferred = [
        "final_loss",
        "min_loss",
        "t_min_loss",
        "final_consensus",
        "min_consensus",
        "t_min_consensus",
        "final_cons_all",
        "final_cons_clients",
        "geomean_rho",
        "median_rho",
        "last_rho",
        "avg_K_mix_only",
        "avg_K_all_steps",
        "avg_active_mix_only",
        "num_mix_steps",
    ]
    keys_present = set()
    for row in summary.values():
        keys_present.update(row.keys())
    show = (
        columns if columns is not None else [k for k in preferred if k in keys_present]
    )

    # header
    header = ["Policy"] + show
    rows = []
    for name, row in summary.items():
        vals = [_fmt(row.get(k, float("nan")), decimals) for k in show]
        rows.append([name] + vals)

    # column widths
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]

    def _fmt_row(cells):
        return " | ".join(s.ljust(widths[i]) for i, s in enumerate(cells))

    print(_fmt_row(header))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(_fmt_row(r))


def latex_table_from_summary(
    summary: Dict[str, Dict[str, float]],
    *,
    decimals: int = 3,
    caption: str = "Summary of runs.",
    label: str = "tab:summary",
    columns: Optional[List[str]] = None,
) -> str:
    preferred = [
        "final_loss",
        "min_loss",
        "t_min_loss",
        "final_consensus",
        "min_consensus",
        "t_min_consensus",
        "final_cons_all",
        "final_cons_clients",
        "geomean_rho",
        "median_rho",
        "last_rho",
        "avg_K_mix_only",
        "avg_K_all_steps",
        "avg_active_mix_only",
        "num_mix_steps",
    ]
    keys_present = set()
    for row in summary.values():
        keys_present.update(row.keys())
    show = (
        columns if columns is not None else [k for k in preferred if k in keys_present]
    )

    nice = {
        "final_loss": "Final loss",
        "min_loss": "Min loss",
        "t_min_loss": "$t^*_{\\text{loss}}$",
        "final_consensus": "Final cons.",
        "min_consensus": "Min cons.",
        "t_min_consensus": "$t^*_{\\text{cons}}$",
        "final_cons_all": "Final cons. (all)",
        "final_cons_clients": "Final cons. (clients)",
        "geomean_rho": "Geo. mean $\\widehat{\\rho}$",
        "median_rho": "Median $\\widehat{\\rho}$",
        "last_rho": "Last $\\widehat{\\rho}$",
        "avg_K_mix_only": "$\\overline{K}$ (mix)",
        "avg_K_all_steps": "$\\overline{K}$ (all)",
        "avg_active_mix_only": "$\\overline{|\\mathcal A|}$",
        "num_mix_steps": "#mix",
    }

    def _f(x):
        return _fmt(x, decimals)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    colspec = "l" + "c" * len(show)
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    header = ["Policy"] + [nice.get(k, k) for k in show]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for name, row in summary.items():
        cells = [name] + [_f(row.get(k, float("nan"))) for k in show]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)
