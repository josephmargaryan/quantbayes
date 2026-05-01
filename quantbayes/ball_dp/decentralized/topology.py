"""Topology utilities for decentralized Ball-DP experiments.

The functions in this module are intentionally NumPy-only.  They are used by the
Paper 3 experiments and by the demo notebook to make the observer geometry
explicit: a graph determines a mixing matrix, and powers of that matrix determine
how a node-local perturbation is seen by each observer over a public transcript.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import numpy as np


def graph_adjacency(name: str, num_nodes: int) -> np.ndarray:
    """Return an undirected adjacency matrix for a small benchmark topology.

    Parameters
    ----------
    name:
        One of ``path``, ``cycle``, ``star``, ``complete``, ``ring`` or
        ``line``.  ``ring`` is an alias for ``cycle`` and ``line`` is an alias
        for ``path``.
    num_nodes:
        Number of vertices. Must be positive.
    """

    m = int(num_nodes)
    if m <= 0:
        raise ValueError("num_nodes must be positive")
    key = str(name).lower().replace("-", "_")
    if key == "line":
        key = "path"
    if key == "ring":
        key = "cycle"

    A = np.zeros((m, m), dtype=float)
    if key == "path":
        for i in range(m - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
    elif key == "cycle":
        if m == 1:
            return A
        for i in range(m):
            A[i, (i + 1) % m] = A[(i + 1) % m, i] = 1.0
    elif key == "star":
        for i in range(1, m):
            A[0, i] = A[i, 0] = 1.0
    elif key == "complete":
        A[:] = 1.0
        np.fill_diagonal(A, 0.0)
    else:
        raise ValueError(f"unknown graph topology {name!r}")
    return A


def graph_distances(adjacency: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path distances for an unweighted undirected graph."""

    A = np.asarray(adjacency, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    m = A.shape[0]
    out = np.full((m, m), np.inf, dtype=float)
    for src in range(m):
        out[src, src] = 0.0
        q: deque[int] = deque([src])
        while q:
            u = q.popleft()
            for v in np.flatnonzero(A[u] > 0):
                if not np.isfinite(out[src, v]):
                    out[src, v] = out[src, u] + 1.0
                    q.append(int(v))
    return out


def metropolis_mixing_matrix(adjacency: np.ndarray, *, lazy: float = 0.0) -> np.ndarray:
    """Construct a symmetric Metropolis mixing matrix.

    For adjacent nodes ``i`` and ``j``, the off-diagonal weight is
    ``1 / (1 + max(deg(i), deg(j)))``.  Diagonal entries are chosen so each row
    sums to one.  ``lazy`` interpolates with the identity matrix, which is useful
    for slower communication ablations.
    """

    A = np.asarray(adjacency, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    lazy_f = float(lazy)
    if lazy_f < 0.0 or lazy_f >= 1.0:
        raise ValueError("lazy must lie in [0, 1)")
    m = A.shape[0]
    deg = A.sum(axis=1)
    W = np.zeros_like(A, dtype=float)
    for i in range(m):
        for j in range(m):
            if i != j and A[i, j] > 0:
                W[i, j] = 1.0 / (1.0 + max(float(deg[i]), float(deg[j])))
    np.fill_diagonal(W, 1.0 - W.sum(axis=1))
    if lazy_f > 0.0:
        W = lazy_f * np.eye(m) + (1.0 - lazy_f) * W
    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError("mixing matrix rows do not sum to one")
    return W


def observer_nodes_from_mode(
    *,
    mode: str,
    attacked_node: int,
    distances: np.ndarray,
    num_nodes: int | None = None,
) -> tuple[int, ...]:
    """Map a named observer policy to a tuple of observing node indices."""

    D = np.asarray(distances, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distances must be square")
    m = int(num_nodes if num_nodes is not None else D.shape[0])
    j = int(attacked_node)
    if j < 0 or j >= m:
        raise ValueError("attacked_node out of range")
    key = str(mode).lower().replace("-", "_")
    if key == "self":
        return (j,)
    if key in {"all", "global"}:
        return tuple(range(m))
    if key in {"all_except_self", "others"}:
        return tuple(i for i in range(m) if i != j)
    finite = np.asarray(D[:, j], dtype=float)
    finite[j] = np.inf
    if key == "nearest":
        return (int(np.argmin(finite)),)
    if key == "farthest":
        vals = np.asarray(D[:, j], dtype=float)
        vals[j] = -np.inf
        vals[~np.isfinite(vals)] = -np.inf
        return (int(np.argmax(vals)),)
    if key.startswith("node"):
        return (int(key.replace("node", "")),)
    raise ValueError(f"unknown observer mode {mode!r}")


def gossip_transfer_rows(
    W: np.ndarray,
    *,
    rounds: int,
    observer_nodes: Sequence[int],
    attacked_node: int,
    include_initial: bool = False,
) -> np.ndarray:
    """Return scalar transfer coefficients from one node to an observer transcript.

    Row ``t`` is the coefficient multiplying the attacked node's initial local
    update in the observation made at a specific observer and time.  When model
    parameters are vector-valued, the full linear map is this coefficient matrix
    Kronecker the identity over parameter coordinates.
    """

    M = np.asarray(W, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("W must be square")
    T = int(rounds)
    if T <= 0:
        raise ValueError("rounds must be positive")
    obs = tuple(int(o) for o in observer_nodes)
    j = int(attacked_node)
    if j < 0 or j >= M.shape[0]:
        raise ValueError("attacked_node out of range")
    powers: list[np.ndarray] = []
    cur = np.eye(M.shape[0])
    if include_initial:
        powers.append(cur.copy())
    for _ in range(T):
        cur = M @ cur
        powers.append(cur.copy())
    rows = []
    for P in powers:
        for o in obs:
            rows.append(float(P[o, j]))
    return np.asarray(rows, dtype=float).reshape(-1, 1)


def scalar_transferred_sensitivity(
    W: np.ndarray,
    *,
    rounds: int,
    observer_nodes: Sequence[int],
    attacked_node: int,
    per_round_sensitivity: Sequence[float] | float = 1.0,
    noise_std: float = 1.0,
) -> float:
    """Whitened scalar sensitivity of an observer's transcript.

    This is the square root of ``sum_t (h_t delta_t / sigma)^2``.  Multiplying by
    ``sqrt(parameter_dim)`` gives the isotropic vector-coordinate analogue when
    every coordinate has the same sensitivity profile.
    """

    h = gossip_transfer_rows(
        W,
        rounds=rounds,
        observer_nodes=tuple(observer_nodes),
        attacked_node=attacked_node,
    ).reshape(-1)
    if np.isscalar(per_round_sensitivity):
        delta = np.full_like(h, float(per_round_sensitivity), dtype=float)
    else:
        arr = np.asarray(
            tuple(float(x) for x in per_round_sensitivity), dtype=float
        ).reshape(-1)
        if arr.size == int(rounds):
            if len(h) % int(rounds) != 0:
                raise ValueError(
                    "observer transcript length is incompatible with rounds"
                )
            delta = np.tile(arr, len(h) // int(rounds))
        elif arr.size == h.size:
            delta = arr
        else:
            raise ValueError(
                "per_round_sensitivity must be scalar, length rounds, or length transcript"
            )
    sigma = float(noise_std)
    if sigma <= 0.0:
        raise ValueError("noise_std must be positive")
    return float(np.linalg.norm(h * delta / sigma))
