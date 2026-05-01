"""Shared utilities for the Paper 3 decentralized Ball-DP experiments.

The Paper 3 experiments are deliberately theorem-facing.  They use the linear
Gaussian observer model from ``quantbayes.ball_dp.decentralized`` and construct
attacks that are exact MAP/Bayes rules under the same release model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from quantbayes.ball_dp.accountants.rdp import gaussian_rdp_curve, rdp_to_dp
from quantbayes.ball_dp.decentralized import (
    account_linear_gaussian_observer,
    apply_linear_gaussian_view,
    gossip_transfer_matrix,
    run_linear_gaussian_finite_prior_attack,
    selector_matrix,
)
from quantbayes.ball_dp.serialization import save_dataframe
from quantbayes.ball_dp.types import Record


DEFAULT_ORDERS = tuple(list(range(2, 65)) + [80, 96, 128, 160, 192, 256])
DEFAULT_RADIUS_TAGS = ("q50", "q80", "q95")
RADIUS_TAG_TO_QUANTILE = {"q50": 0.50, "q80": 0.80, "q95": 0.95}
DEFAULT_GRAPH_LIST = ("path", "cycle", "star", "two_cluster", "complete")
DEFAULT_OBSERVER_MODES = ("self", "nearest", "farthest", "all")
DEFAULT_MECHANISMS = ("ball", "standard")
_NORMAL = NormalDist()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def slugify(text: str) -> str:
    out = []
    for ch in str(text).lower():
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def savefig_stem(fig: Any, stem: str | Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.6),
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
        }
    )


def add_embedding_loader_args(parser: argparse.ArgumentParser) -> None:
    """Add the loader flags expected by the existing embedding loaders."""
    parser.add_argument("--data-root", type=str, default="/content/quantbayes/data")
    parser.add_argument("--embedding-cache-path", type=str, default=None)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--encoder-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hf-cache-dir", type=str, default=None)


def load_embedding_dataset(args: argparse.Namespace, dataset_name: str):
    """Use the official Paper 1/2 embedding loader surface."""
    from quantbayes.ball_dp.experiments.convex.run_attack_experiment import (
        load_embeddings,
        resolve_dataset,
    )

    spec = resolve_dataset(dataset_name)
    return load_embeddings(args, spec)


def embedding_radius_report(
    X: np.ndarray, *, seed: int = 0, max_pairs: int = 20000
) -> dict[str, float]:
    """Estimate useful feature-space radii without pulling in the Paper 1 selector."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must have shape (n, d) with n >= 2.")
    rng = np.random.default_rng(int(seed))
    n = int(X.shape[0])
    m = min(int(max_pairs), max(1, n * min(n - 1, 32)))
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n - 1, size=m)
    j = j + (j >= i)
    dist = np.linalg.norm(X[i] - X[j], axis=1)
    return {
        "q50": float(np.quantile(dist, 0.50)),
        "q80": float(np.quantile(dist, 0.80)),
        "q95": float(np.quantile(dist, 0.95)),
        "mean": float(np.mean(dist)),
        "max_sampled": float(np.max(dist)),
    }


def parse_radius_grid(
    values: Sequence[str | float], X: np.ndarray, *, seed: int = 0
) -> tuple[list[float], dict[str, float]]:
    report = embedding_radius_report(X, seed=seed)
    radii: list[float] = []
    for value in values:
        key = str(value).strip().lower()
        if key in report:
            radii.append(float(report[key]))
        else:
            radii.append(float(value))
    return radii, report


def graph_adjacency(kind: str, num_nodes: int) -> np.ndarray:
    kind = str(kind).strip().lower()
    m = int(num_nodes)
    if m <= 1:
        raise ValueError("num_nodes must be at least 2.")
    A = np.zeros((m, m), dtype=np.float64)
    if kind in {"path", "line"}:
        for i in range(m - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
    elif kind in {"cycle", "ring"}:
        for i in range(m):
            A[i, (i + 1) % m] = A[(i + 1) % m, i] = 1.0
    elif kind == "star":
        for i in range(1, m):
            A[0, i] = A[i, 0] = 1.0
    elif kind == "complete":
        A[:] = 1.0
        np.fill_diagonal(A, 0.0)
    elif kind in {"two_cluster", "two-cluster", "two_clusters"}:
        if m < 4:
            raise ValueError("two_cluster requires num_nodes >= 4.")
        cut = m // 2
        for lo, hi in ((0, cut), (cut, m)):
            for i in range(lo, hi):
                for j in range(i + 1, hi):
                    A[i, j] = A[j, i] = 1.0
        A[cut - 1, cut] = A[cut, cut - 1] = 1.0
    elif kind.startswith("erdos") or kind.startswith("random"):
        # Deterministic sparse random graph for reproducibility.  The common
        # experiments prefer named topologies; this path is mainly for notebook use.
        rng = np.random.default_rng(0)
        p = min(0.75, max(0.15, 3.0 / float(max(2, m - 1))))
        upper = rng.random((m, m)) < p
        upper = np.triu(upper, k=1)
        A = upper.astype(np.float64) + upper.T.astype(np.float64)
        # Force connectivity by adding a path backbone.
        for i in range(m - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
    else:
        raise ValueError(
            "Unsupported graph. Use one of {'path', 'cycle', 'star', 'two_cluster', 'complete', 'random'}."
        )
    return A


def metropolis_mixing_matrix(adjacency: np.ndarray, *, lazy: float = 0.0) -> np.ndarray:
    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square.")
    if not np.allclose(A, A.T, atol=1e-12):
        raise ValueError("Only undirected graphs are supported.")
    if np.any(A < 0.0):
        raise ValueError("adjacency must be nonnegative.")
    m = int(A.shape[0])
    deg = np.sum(A > 0.0, axis=1).astype(float)
    W = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i != j and A[i, j] > 0.0:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - float(np.sum(W[i]))
    lazy = float(lazy)
    if not (0.0 <= lazy < 1.0):
        raise ValueError("lazy must lie in [0,1).")
    if lazy > 0.0:
        W = lazy * np.eye(m, dtype=np.float64) + (1.0 - lazy) * W
    return W


def constant_mixing_schedule(W: np.ndarray, rounds: int) -> list[np.ndarray]:
    T = int(rounds)
    if T <= 0:
        raise ValueError("rounds must be positive.")
    W = np.asarray(W, dtype=np.float64)
    return [W.copy() for _ in range(T)]


def graph_distances(adjacency: np.ndarray) -> np.ndarray:
    A = np.asarray(adjacency, dtype=np.float64)
    m = int(A.shape[0])
    dist = np.full((m, m), np.inf, dtype=np.float64)
    for src in range(m):
        dist[src, src] = 0.0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in np.flatnonzero(A[u] > 0.0).tolist():
                if not np.isfinite(dist[src, v]):
                    dist[src, v] = dist[src, u] + 1.0
                    queue.append(int(v))
    return dist


def observer_nodes_from_mode(
    mode: str,
    *,
    attacked_node: int,
    num_nodes: int,
    distances: np.ndarray,
) -> tuple[int, ...]:
    mode = str(mode).strip().lower()
    m = int(num_nodes)
    j = int(attacked_node)
    if mode == "self":
        return (j,)
    if mode in {"all", "public"}:
        return tuple(range(m))
    if mode in {"all_except_self", "all_except_attacked", "coalition"}:
        return tuple(i for i in range(m) if i != j)
    finite_others = [i for i in range(m) if i != j and np.isfinite(distances[j, i])]
    if not finite_others:
        return (j,)
    if mode in {"nearest", "neighbor", "near"}:
        best = min(finite_others, key=lambda i: (float(distances[j, i]), i))
        return (int(best),)
    if mode in {"farthest", "far", "distant"}:
        best = max(finite_others, key=lambda i: (float(distances[j, i]), -i))
        return (int(best),)
    if mode in {"hub", "node0"}:
        return (0,)
    if mode.startswith("node"):
        node = int(mode.replace("node", ""))
        if node < 0 or node >= m:
            raise ValueError(f"Observer mode {mode!r} refers to an invalid node.")
        return (node,)
    raise ValueError(
        "Unsupported observer mode. Use self, nearest, farthest, all, all_except_self, hub, or node<i>."
    )


def build_transfer_for_observer(
    *,
    W: np.ndarray,
    rounds: int,
    observer_nodes: Sequence[int],
    attacked_node: int,
) -> np.ndarray:
    S = selector_matrix(observer_nodes, num_nodes=int(W.shape[0]))
    return gossip_transfer_matrix(
        constant_mixing_schedule(W, int(rounds)),
        observer_selector=S,
        attacked_node=int(attacked_node),
    )


def block_weights(rounds: int, *, decay: float = 1.0) -> np.ndarray:
    T = int(rounds)
    decay = float(decay)
    if T <= 0:
        raise ValueError("rounds must be positive.")
    if decay <= 0.0:
        raise ValueError("decay must be positive.")
    return np.asarray([decay**t for t in range(T)], dtype=np.float64)


def mechanism_step_sensitivities(
    *,
    mechanism: str,
    radius: float,
    clip_norm: float,
    rounds: int,
    feature_lipschitz: float = 1.0,
    decay: float = 1.0,
) -> np.ndarray:
    mechanism = str(mechanism).lower()
    C = float(clip_norm)
    r = float(radius)
    L = float(feature_lipschitz)
    if C <= 0.0:
        raise ValueError("clip_norm must be positive.")
    if r < 0.0:
        raise ValueError("radius must be nonnegative.")
    if L < 0.0:
        raise ValueError("feature_lipschitz must be nonnegative.")
    if mechanism == "ball":
        base = min(L * r, 2.0 * C)
    elif mechanism in {"standard", "replacement", "dp"}:
        base = 2.0 * C
    else:
        raise ValueError("mechanism must be 'ball' or 'standard'.")
    return block_weights(rounds, decay=decay) * float(base)


def covariance_time(num_view_blocks: int, noise_std: float) -> np.ndarray:
    sigma = float(noise_std)
    if sigma <= 0.0:
        raise ValueError("noise_std must be positive.")
    d = int(num_view_blocks)
    if d <= 0:
        raise ValueError("num_view_blocks must be positive.")
    return (sigma * sigma) * np.eye(d, dtype=np.float64)


def normal_cdf(x: float) -> float:
    return float(_NORMAL.cdf(float(x)))


def normal_ppf(p: float) -> float:
    p = float(p)
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -float("inf")
        if p == 1.0:
            return float("inf")
        raise ValueError("p must lie in [0,1].")
    return float(_NORMAL.inv_cdf(p))


def gaussian_dp_tradeoff_value(*, type_i_error: float, mu: float) -> float:
    """GDP/f-DP tradeoff value G_mu(alpha) for a Gaussian mechanism.

    Here ``mu`` is the whitened sensitivity of the observer-specific Gaussian
    experiment.  The value returned is
    ``Phi(Phi^{-1}(1-alpha) - mu)``, the canonical Gaussian differential
    privacy tradeoff function at type-I error budget ``alpha``.
    """

    alpha = float(type_i_error)
    mu_f = float(mu)
    if alpha <= 0.0:
        return 1.0
    if alpha >= 1.0:
        return 0.0
    return float(_NORMAL.cdf(_NORMAL.inv_cdf(1.0 - alpha) - mu_f))


def direct_gaussian_rero_bound(
    *, kappa: float, transferred_sensitivity: float
) -> float:
    k = float(kappa)
    c = float(transferred_sensitivity)
    if k <= 0.0:
        return 0.0
    if k >= 1.0:
        return 1.0
    if c < 0.0:
        raise ValueError("transferred_sensitivity must be nonnegative.")
    return float(min(1.0, max(0.0, normal_cdf(normal_ppf(k) + c))))


def gaussian_rdp_rero_bound(
    *,
    orders: Sequence[int | float],
    sensitivity_sq: float,
    kappa: float,
) -> tuple[float, Optional[float]]:
    """Generic RDP-to-ReRo bound for the same Gaussian view."""
    from quantbayes.ball_dp.decentralized.rero import ball_pn_rdp_success_bound
    from quantbayes.ball_dp.types import RdpCurve

    eps = [0.5 * float(a) * float(sensitivity_sq) for a in orders]
    curve = RdpCurve(
        orders=tuple(float(a) for a in orders),
        epsilons=tuple(float(v) for v in eps),
        source="linear_gaussian_observer",
        radius=None,
    )
    return ball_pn_rdp_success_bound(curve, kappa=float(kappa))


def observer_accounting_row(
    *,
    dataset_tag: str,
    graph: str,
    W: np.ndarray,
    distances: np.ndarray,
    rounds: int,
    observer_mode: str,
    attacked_node: int,
    radius: float,
    clip_norm: float,
    noise_std: float,
    feature_dim: int,
    prior_size: int,
    mechanism: str,
    orders: Sequence[int | float] = DEFAULT_ORDERS,
    dp_delta: Optional[float] = None,
    feature_lipschitz: float = 1.0,
    block_decay: float = 1.0,
) -> dict[str, Any]:
    observer_nodes = observer_nodes_from_mode(
        observer_mode,
        attacked_node=int(attacked_node),
        num_nodes=int(W.shape[0]),
        distances=distances,
    )
    H = build_transfer_for_observer(
        W=W,
        rounds=int(rounds),
        observer_nodes=observer_nodes,
        attacked_node=int(attacked_node),
    )
    cov = covariance_time(H.shape[0], float(noise_std))
    deltas = mechanism_step_sensitivities(
        mechanism=mechanism,
        radius=float(radius),
        clip_norm=float(clip_norm),
        rounds=int(rounds),
        feature_lipschitz=float(feature_lipschitz),
        decay=float(block_decay),
    )
    acct = account_linear_gaussian_observer(
        transfer_matrix=H,
        covariance=cov,
        block_sensitivities=deltas,
        parameter_dim=int(feature_dim),
        orders=tuple(float(a) for a in orders),
        radius=float(radius),
        dp_delta=None if dp_delta is None else float(dp_delta),
        attacked_node=int(attacked_node),
        observer=tuple(int(v) for v in observer_nodes),
        covariance_mode="kron_eye",
        method="auto",
    )
    sensitivity_sq = float(acct.sensitivity_sq)
    transferred = math.sqrt(max(0.0, sensitivity_sq))
    kappa = 1.0 / float(prior_size)
    direct_bound = direct_gaussian_rero_bound(
        kappa=kappa,
        transferred_sensitivity=transferred,
    )
    rdp_bound, rdp_alpha = gaussian_rdp_rero_bound(
        orders=orders,
        sensitivity_sq=sensitivity_sq,
        kappa=kappa,
    )
    distance_to_observer = float("nan")
    if len(observer_nodes) == 1:
        distance_to_observer = float(
            distances[int(attacked_node), int(observer_nodes[0])]
        )
    elif len(observer_nodes) > 1:
        distance_to_observer = float(
            np.nanmin([distances[int(attacked_node), int(v)] for v in observer_nodes])
        )
    dp_epsilon = float("nan")
    dp_order = float("nan")
    if acct.dp_certificate is not None:
        dp_epsilon = float(acct.dp_certificate.epsilon)
        dp_order = float(acct.dp_certificate.order_opt or float("nan"))
    order_to_eps = acct.rdp_curve.as_dict()
    return {
        "dataset_tag": str(dataset_tag),
        "graph": str(graph),
        "num_nodes": int(W.shape[0]),
        "rounds": int(rounds),
        "attacked_node": int(attacked_node),
        "observer_mode": str(observer_mode),
        "observer_nodes": ",".join(str(int(v)) for v in observer_nodes),
        "observer_size": int(len(observer_nodes)),
        "graph_distance": distance_to_observer,
        "radius": float(radius),
        "clip_norm": float(clip_norm),
        "noise_std": float(noise_std),
        "feature_dim": int(feature_dim),
        "prior_size": int(prior_size),
        "mechanism": str(mechanism),
        "feature_lipschitz": float(feature_lipschitz),
        "block_decay": float(block_decay),
        "step_sensitivity_base": float(deltas[0]),
        "transferred_sensitivity_sq": sensitivity_sq,
        "transferred_sensitivity": transferred,
        "gaussian_dp_mu": transferred,
        "fdp_tradeoff_alpha_0_10": gaussian_dp_tradeoff_value(
            type_i_error=0.10, mu=transferred
        ),
        "fdp_tradeoff_alpha_0_25": gaussian_dp_tradeoff_value(
            type_i_error=0.25, mu=transferred
        ),
        "fdp_tradeoff_alpha_0_50": gaussian_dp_tradeoff_value(
            type_i_error=0.50, mu=transferred
        ),
        "exact_transferred_sensitivity": bool(acct.exact),
        "transferred_sensitivity_method": str(acct.method),
        "kappa_exact_id": kappa,
        "direct_gaussian_rero_bound": float(direct_bound),
        "rdp_rero_bound": float(rdp_bound),
        "rdp_rero_alpha_opt": float("nan") if rdp_alpha is None else float(rdp_alpha),
        "dp_delta": float("nan") if dp_delta is None else float(dp_delta),
        "dp_epsilon": dp_epsilon,
        "dp_order_opt": dp_order,
        "rdp_eps_alpha_8": float(order_to_eps.get(8.0, float("nan"))),
        "rdp_eps_alpha_16": float(order_to_eps.get(16.0, float("nan"))),
        "rdp_eps_alpha_32": float(order_to_eps.get(32.0, float("nan"))),
    }


def clip_l2_rows(X: np.ndarray, clip_norm: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    C = float(clip_norm)
    if C <= 0.0:
        raise ValueError("clip_norm must be positive.")
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    scale = np.minimum(1.0, C / np.maximum(norm, 1e-12))
    return (X * scale).astype(np.float32, copy=False)


def synthetic_finite_prior_ball(
    *,
    center: np.ndarray,
    radius: float,
    prior_size: int,
    rng: np.random.Generator,
    clip_norm: Optional[float] = None,
) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    m = int(prior_size)
    if m <= 0:
        raise ValueError("prior_size must be positive.")
    r = float(radius)
    if r < 0.0:
        raise ValueError("radius must be nonnegative.")
    d = int(center.size)
    if r == 0.0 or d == 0:
        X = np.repeat(center[None, :], m, axis=0).astype(np.float32)
    else:
        G = rng.normal(size=(m, d)).astype(np.float32)
        G /= np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
        U = rng.random(m).astype(np.float32) ** (1.0 / float(d))
        X = center[None, :] + float(r) * U[:, None] * G
    if clip_norm is not None:
        X = clip_l2_rows(X, float(clip_norm))
    return np.asarray(X, dtype=np.float32)


def make_sensitive_blocks_fn(
    *,
    rounds: int,
    clip_norm: float,
    block_decay: float = 1.0,
):
    weights = block_weights(int(rounds), decay=float(block_decay)).astype(np.float32)

    def sensitive_blocks(x: np.ndarray, y: int) -> np.ndarray:
        del y  # Paper 3 finite-prior runs use a known-label feature-only release.
        x_arr = np.asarray(x, dtype=np.float32).reshape(1, -1)
        x_clip = clip_l2_rows(x_arr, float(clip_norm))[0]
        return (weights[:, None] * x_clip[None, :]).astype(np.float32)

    return sensitive_blocks


def simulate_linear_gaussian_view(
    *,
    H: np.ndarray,
    candidate_x: np.ndarray,
    label: int,
    sensitive_blocks_fn: Any,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = np.asarray(
        apply_linear_gaussian_view(
            H,
            sensitive_blocks_fn(np.asarray(candidate_x, dtype=np.float32), int(label)),
        ),
        dtype=np.float64,
    )
    noise = rng.normal(0.0, float(noise_std), size=mean.shape)
    return (mean + noise).astype(np.float32)


def run_exact_finite_prior_trial(
    *,
    W: np.ndarray,
    distances: np.ndarray,
    rounds: int,
    observer_mode: str,
    attacked_node: int,
    radius: float,
    clip_norm: float,
    noise_std: float,
    prior_size: int,
    feature_dim: int,
    center: np.ndarray,
    label: int,
    rng: np.random.Generator,
    orders: Sequence[int | float] = DEFAULT_ORDERS,
    dp_delta: Optional[float] = None,
    feature_lipschitz: float = 1.0,
    block_decay: float = 1.0,
) -> dict[str, Any]:
    observer_nodes = observer_nodes_from_mode(
        observer_mode,
        attacked_node=int(attacked_node),
        num_nodes=int(W.shape[0]),
        distances=distances,
    )
    H = build_transfer_for_observer(
        W=W,
        rounds=int(rounds),
        observer_nodes=observer_nodes,
        attacked_node=int(attacked_node),
    )
    sensitive_blocks_fn = make_sensitive_blocks_fn(
        rounds=int(rounds), clip_norm=float(clip_norm), block_decay=float(block_decay)
    )
    candidates = synthetic_finite_prior_ball(
        center=np.asarray(center, dtype=np.float32).reshape(-1),
        radius=float(radius),
        prior_size=int(prior_size),
        rng=rng,
        clip_norm=None,
    )
    true_index = int(rng.integers(0, int(prior_size)))
    labels = np.full((int(prior_size),), int(label), dtype=np.int32)
    observed = simulate_linear_gaussian_view(
        H=H,
        candidate_x=candidates[true_index],
        label=int(label),
        sensitive_blocks_fn=sensitive_blocks_fn,
        noise_std=float(noise_std),
        rng=rng,
    )
    cov = covariance_time(H.shape[0], float(noise_std))

    def mean_fn(x: np.ndarray, y: int) -> np.ndarray:
        return apply_linear_gaussian_view(H, sensitive_blocks_fn(x, int(y)))

    true_record = Record(features=candidates[true_index], label=int(label))
    attack = run_linear_gaussian_finite_prior_attack(
        observed_view=observed,
        candidate_features=candidates,
        candidate_labels=labels,
        mean_fn=mean_fn,
        covariance=cov,
        prior_weights=None,
        true_record=true_record,
        eta_grid=(0.0, float(radius) / 10.0, float(radius) / 2.0, float(radius)),
        covariance_mode="kron_eye",
    )
    acct_row = observer_accounting_row(
        dataset_tag="_trial_",
        graph="_trial_",
        W=W,
        distances=distances,
        rounds=int(rounds),
        observer_mode=observer_mode,
        attacked_node=int(attacked_node),
        radius=float(radius),
        clip_norm=float(clip_norm),
        noise_std=float(noise_std),
        feature_dim=int(feature_dim),
        prior_size=int(prior_size),
        mechanism="ball",
        orders=orders,
        dp_delta=dp_delta,
        feature_lipschitz=feature_lipschitz,
        block_decay=block_decay,
    )
    predicted_idx = int(attack.diagnostics.get("predicted_prior_index", -1))
    return {
        "true_prior_index": int(true_index),
        "predicted_prior_index": int(predicted_idx),
        "exact_identification_success": float(predicted_idx == true_index),
        "prior_rank": float(attack.metrics.get("prior_rank", float("nan"))),
        "prior_hit_at_1": float(
            attack.metrics.get("prior_hit@1", predicted_idx == true_index)
        ),
        "prior_hit_at_5": float(attack.metrics.get("prior_hit@5", float("nan"))),
        "mse": float(attack.metrics.get("mse", float("nan"))),
        "l2_error": float(
            np.linalg.norm(
                np.asarray(attack.z_hat, dtype=np.float32).reshape(-1)
                - np.asarray(candidates[true_index], dtype=np.float32).reshape(-1)
            )
        ),
        "attack_status": str(attack.status),
        "attack_family": str(attack.attack_family),
        "observed_view_dim": int(observed.size),
        **{k: v for k, v in acct_row.items() if k not in {"dataset_tag", "graph"}},
    }


def summarize_numeric(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    existing = [c for c in value_cols if c in df.columns]
    if not existing:
        return df.groupby(list(group_cols), dropna=False).size().reset_index(name="n")
    agg = {c: ["mean", "std", "min", "max"] for c in existing}
    out = df.groupby(list(group_cols), dropna=False).agg(agg)
    out.columns = ["_".join(str(x) for x in col if str(x)) for col in out.columns]
    out = out.reset_index()
    counts = df.groupby(list(group_cols), dropna=False).size().reset_index(name="n")
    return counts.merge(out, on=list(group_cols), how="left")


def mechanism_noise_for_target_dp(
    *,
    target_epsilon: float,
    sensitivity: float,
    orders: Sequence[int | float] = DEFAULT_ORDERS,
    delta: float = 1e-6,
    lower: float = 1e-6,
    upper: float = 1.0,
    max_upper: float = 1e4,
    steps: int = 50,
) -> dict[str, float]:
    """Calibrate scalar Gaussian noise for one vector release with sensitivity Δ."""
    target = float(target_epsilon)
    sens = float(sensitivity)
    if target <= 0.0:
        raise ValueError("target_epsilon must be positive.")
    if sens < 0.0:
        raise ValueError("sensitivity must be nonnegative.")
    if sens == 0.0:
        return {"noise_std": 0.0, "epsilon": 0.0, "order_opt": float("nan")}

    def eps_for(sigma: float) -> tuple[float, float]:
        curve = gaussian_rdp_curve(
            orders=tuple(float(a) for a in orders),
            sensitivity=sens,
            sigma=float(sigma),
            source="gaussian_vector_release_calibration",
        )
        cert = rdp_to_dp(curve, delta=float(delta), source="rdp_to_dp")
        return float(cert.epsilon), float(cert.order_opt or float("nan"))

    lo = float(lower)
    hi = float(upper)
    eps_hi, _ = eps_for(hi)
    while eps_hi > target:
        hi *= 2.0
        if hi > float(max_upper):
            raise RuntimeError(
                f"Could not calibrate Gaussian noise up to max_upper={max_upper}. Last epsilon={eps_hi}."
            )
        eps_hi, _ = eps_for(hi)
    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        eps_mid, _ = eps_for(mid)
        if eps_mid <= target:
            hi = mid
        else:
            lo = mid
    eps_final, order_final = eps_for(hi)
    return {
        "noise_std": float(hi),
        "epsilon": float(eps_final),
        "order_opt": float(order_final),
    }


def partition_indices_iid(
    y: np.ndarray, *, num_nodes: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(y), dtype=np.int64)
    rng.shuffle(idx)
    return [
        np.asarray(part, dtype=np.int64) for part in np.array_split(idx, int(num_nodes))
    ]


def local_class_sum_states(
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_nodes: int,
    num_classes: int,
    clip_norm: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    Xc = clip_l2_rows(np.asarray(X, dtype=np.float32), float(clip_norm))
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    p = int(Xc.shape[1])
    K = int(num_classes)
    parts = partition_indices_iid(y, num_nodes=int(num_nodes), seed=int(seed))
    states = np.zeros((int(num_nodes), K, p), dtype=np.float32)
    counts = np.zeros((K,), dtype=np.float64)
    for node, idx in enumerate(parts):
        for k in range(K):
            mask = idx[y[idx] == k]
            if mask.size:
                states[node, k] = np.sum(Xc[mask], axis=0)
                counts[k] += float(mask.size)
    return states.reshape((int(num_nodes), K * p)), counts


def nearest_prototype_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    prototypes: np.ndarray,
    *,
    clip_norm: Optional[float] = None,
) -> float:
    X = np.asarray(X, dtype=np.float32)
    if clip_norm is not None:
        X = clip_l2_rows(X, float(clip_norm))
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    P = np.asarray(prototypes, dtype=np.float32)
    diff = X[:, None, :] - P[None, :, :]
    pred = np.argmin(np.sum(diff * diff, axis=2), axis=1).astype(np.int32)
    return float(np.mean(pred == y))


def run_noisy_prototype_gossip(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    W: np.ndarray,
    rounds: int,
    num_classes: int,
    clip_norm: float,
    noise_std: float,
    seed: int,
) -> dict[str, float]:
    """A simple decentralized noisy prototype learner.

    Each node computes clipped class-sum vectors, adds Gaussian noise once to its local
    vector, and then runs deterministic gossip.  Counts are treated as public in this
    controlled benchmark, so the protected object is the feature vector conditional on
    label.  This gives a lightweight utility/privacy tradeoff without changing the
    theorem-facing observer experiments.
    """
    m = int(W.shape[0])
    rng = np.random.default_rng(int(seed))
    x0, counts = local_class_sum_states(
        X_train,
        y_train,
        num_nodes=m,
        num_classes=int(num_classes),
        clip_norm=float(clip_norm),
        seed=int(seed),
    )
    if float(noise_std) > 0.0:
        x = x0 + rng.normal(0.0, float(noise_std), size=x0.shape).astype(np.float32)
    else:
        x = x0.copy()
    for _ in range(int(rounds)):
        x = np.asarray(W @ x, dtype=np.float32)
    K = int(num_classes)
    p = int(X_train.shape[1])
    counts_safe = np.maximum(counts, 1.0)
    accs = []
    for node in range(m):
        # W is doubly stochastic for the supported graphs, so consensus approaches
        # average local sums; multiply by m to recover the global sum scale.
        sums = (float(m) * x[node]).reshape((K, p))
        prototypes = sums / counts_safe[:, None]
        accs.append(
            nearest_prototype_accuracy(
                X_test,
                y_test,
                prototypes,
                clip_norm=float(clip_norm),
            )
        )
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_min_node": float(np.min(accs)),
        "accuracy_max_node": float(np.max(accs)),
        "accuracy_std_node": float(np.std(accs)),
        "consensus_state_disagreement": float(np.mean(np.var(x, axis=0))),
    }


def write_rows_and_summary(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
    rows_name: str,
    summary_name: str,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> tuple[Path, Path]:
    if not rows:
        raise RuntimeError("No rows were generated.")
    df = pd.DataFrame(rows)
    rows_path = out_dir / rows_name
    save_dataframe(df, rows_path, save_parquet_if_possible=False)
    summary = summarize_numeric(df, group_cols=group_cols, value_cols=value_cols)
    summary_path = out_dir / summary_name
    save_dataframe(summary, summary_path, save_parquet_if_possible=False)
    return rows_path, summary_path
