from __future__ import annotations

from collections import deque
from typing import Literal, Sequence

import numpy as np


GraphName = Literal[
    "path", "cycle", "ring", "star", "complete", "erdos_renyi", "random", "two_cluster"
]


def selector_matrix(
    observer_nodes: Sequence[int],
    *,
    num_nodes: int,
) -> np.ndarray:
    """Build the row-selector S_A for an observer set A."""
    m = int(num_nodes)
    if m <= 0:
        raise ValueError("num_nodes must be positive.")
    obs = [int(v) for v in observer_nodes]
    if not obs:
        raise ValueError("observer_nodes must be non-empty.")
    if len(set(obs)) != len(obs):
        raise ValueError("observer_nodes must not contain duplicates.")
    if any(v < 0 or v >= m for v in obs):
        raise ValueError("observer_nodes contains an index outside [0, num_nodes).")
    S = np.zeros((len(obs), m), dtype=np.float64)
    for row, node in enumerate(obs):
        S[row, node] = 1.0
    return S


def graph_distances(adjacency: np.ndarray) -> np.ndarray:
    """All-pairs unweighted shortest-path distances; disconnected pairs are inf."""
    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square.")
    m = int(A.shape[0])
    dist = np.full((m, m), np.inf, dtype=np.float64)
    for src in range(m):
        dist[src, src] = 0.0
        q: deque[int] = deque([src])
        while q:
            u = q.popleft()
            for v_raw in np.flatnonzero(A[u] > 0.0):
                v = int(v_raw)
                if not np.isfinite(dist[src, v]):
                    dist[src, v] = dist[src, u] + 1.0
                    q.append(v)
    return dist


def make_graph_adjacency(
    graph: GraphName,
    *,
    num_nodes: int,
    erdos_p: float = 0.35,
    seed: int = 0,
) -> np.ndarray:
    """Construct an undirected graph adjacency matrix for topology experiments.

    The returned matrix has zeros on the diagonal and is symmetric.  The random
    graph branch resamples until connected, because observer-distance comparisons
    are hard to interpret on disconnected graphs.
    """
    m = int(num_nodes)
    if m <= 1:
        raise ValueError("num_nodes must be at least 2.")
    key = str(graph).lower()
    if key == "ring":
        key = "cycle"
    if key == "random":
        key = "erdos_renyi"

    A = np.zeros((m, m), dtype=np.float64)
    if key == "path":
        for i in range(m - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
    elif key == "cycle":
        if m < 3:
            raise ValueError("cycle/ring requires num_nodes >= 3.")
        for i in range(m):
            A[i, (i + 1) % m] = A[(i + 1) % m, i] = 1.0
    elif key == "star":
        if m < 3:
            raise ValueError("star requires num_nodes >= 3.")
        for i in range(1, m):
            A[0, i] = A[i, 0] = 1.0
    elif key == "complete":
        A[:] = 1.0
        np.fill_diagonal(A, 0.0)
    elif key == "two_cluster":
        if m < 4:
            raise ValueError("two_cluster requires num_nodes >= 4.")
        cut = m // 2
        for lo, hi in ((0, cut), (cut, m)):
            for i in range(lo, hi):
                for j in range(i + 1, hi):
                    A[i, j] = A[j, i] = 1.0
        A[cut - 1, cut] = A[cut, cut - 1] = 1.0
    elif key == "erdos_renyi":
        p = float(erdos_p)
        if not (0.0 < p <= 1.0):
            raise ValueError("erdos_p must lie in (0,1].")
        rng = np.random.default_rng(int(seed))
        for _ in range(10_000):
            U = rng.random((m, m))
            A = ((U < p) | (U.T < p)).astype(np.float64)
            np.fill_diagonal(A, 0.0)
            if np.all(np.isfinite(graph_distances(A))):
                break
        else:
            raise RuntimeError("Failed to sample a connected Erdos--Renyi graph.")
    else:
        raise ValueError(
            "graph must be one of {'path','cycle','ring','star','complete','two_cluster','erdos_renyi','random'}."
        )
    return A


def metropolis_mixing_matrix(
    adjacency: np.ndarray,
    *,
    lazy: float | None = None,
    laziness: float | None = None,
) -> np.ndarray:
    """Build a symmetric row-stochastic lazy Metropolis gossip matrix."""
    if lazy is None and laziness is None:
        lazy_value = 0.0
    elif lazy is None:
        lazy_value = float(laziness)
    elif laziness is None:
        lazy_value = float(lazy)
    else:
        if abs(float(lazy) - float(laziness)) > 1e-12:
            raise ValueError(
                "Provide at most one of lazy and laziness, or equal values."
            )
        lazy_value = float(lazy)

    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square.")
    if not np.allclose(A, A.T, atol=1e-12, rtol=0.0):
        raise ValueError("adjacency must be symmetric.")
    if np.any(A < 0.0) or not np.all(np.isfinite(A)):
        raise ValueError("adjacency must have finite nonnegative entries.")
    m = int(A.shape[0])
    deg = np.sum(A > 0.0, axis=1).astype(np.float64)
    if np.any(deg <= 0.0):
        raise ValueError("all nodes must have positive graph degree.")
    W = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i != j and A[i, j] > 0.0:
                W[i, j] = 1.0 / (1.0 + max(float(deg[i]), float(deg[j])))
        W[i, i] = 1.0 - float(np.sum(W[i, :]))
    if not (0.0 <= lazy_value < 1.0):
        raise ValueError("lazy/laziness must lie in [0,1).")
    if lazy_value > 0.0:
        W = lazy_value * np.eye(m, dtype=np.float64) + (1.0 - lazy_value) * W
    if not np.allclose(W @ np.ones((m,)), np.ones((m,)), atol=1e-10):
        raise RuntimeError(
            "Metropolis construction failed to produce row-stochastic W."
        )
    return W


def _validate_mixing_matrices(
    mixing_matrices: Sequence[np.ndarray],
) -> list[np.ndarray]:
    Ws = [np.asarray(W, dtype=np.float64) for W in mixing_matrices]
    if not Ws:
        raise ValueError("mixing_matrices must be non-empty.")
    if Ws[0].ndim != 2 or Ws[0].shape[0] != Ws[0].shape[1]:
        raise ValueError("Every mixing matrix must be square.")
    m = int(Ws[0].shape[0])
    for t, W in enumerate(Ws):
        if W.shape != (m, m):
            raise ValueError(
                f"mixing_matrices[{t}] has shape {W.shape}, expected {(m, m)}."
            )
        if not np.all(np.isfinite(W)):
            raise ValueError(f"mixing_matrices[{t}] contains non-finite entries.")
    return Ws


def _validate_selector(observer_selector: np.ndarray, *, num_nodes: int) -> np.ndarray:
    S_A = np.asarray(observer_selector, dtype=np.float64)
    if S_A.ndim != 2 or S_A.shape[1] != int(num_nodes):
        raise ValueError("observer_selector must have shape (d_A, num_nodes).")
    if S_A.shape[0] <= 0:
        raise ValueError("observer_selector must select at least one observer row.")
    if not np.all(np.isfinite(S_A)):
        raise ValueError("observer_selector contains non-finite entries.")
    return S_A


def gossip_transfer_matrix(
    mixing_matrices: Sequence[np.ndarray],
    *,
    observer_selector: np.ndarray,
    attacked_node: int,
) -> np.ndarray:
    r"""Build H_{A<-j} for x_{t+1}=W_t x_t+u_t+xi_t, y_{t+1}=S_A x_{t+1}.

    Returned shape is (T*|A|, T).  Entry block (r,c) is
    S_A W_r W_{r-1} ... W_{c+1} e_j for c <= r, and zero otherwise.

    The matrix product order matters when the W_t do not commute.  This function
    is validated against a direct unroll of the linear recursion.
    """
    Ws = _validate_mixing_matrices(mixing_matrices)
    m = int(Ws[0].shape[0])
    S_A = _validate_selector(observer_selector, num_nodes=m)
    j = int(attacked_node)
    if j < 0 or j >= m:
        raise ValueError("attacked_node is outside [0, num_nodes).")
    T = len(Ws)
    d_A = int(S_A.shape[0])
    H = np.zeros((T * d_A, T), dtype=np.float64)
    e_j = np.zeros((m,), dtype=np.float64)
    e_j[j] = 1.0
    for r in range(T):
        P = np.eye(m, dtype=np.float64)
        for c in range(r, -1, -1):
            if c < r:
                P = P @ Ws[c + 1]
            H[r * d_A : (r + 1) * d_A, c] = S_A @ (P @ e_j)
    return H


def gossip_noise_transfer_matrix(
    mixing_matrices: Sequence[np.ndarray],
    *,
    observer_selector: np.ndarray,
) -> np.ndarray:
    r"""Transfer matrix from per-round process noise to the stacked observer view."""
    Ws = _validate_mixing_matrices(mixing_matrices)
    m = int(Ws[0].shape[0])
    S_A = _validate_selector(observer_selector, num_nodes=m)
    T = len(Ws)
    d_A = int(S_A.shape[0])
    G = np.zeros((T * d_A, T * m), dtype=np.float64)
    for r in range(T):
        P = np.eye(m, dtype=np.float64)
        for c in range(r, -1, -1):
            if c < r:
                P = P @ Ws[c + 1]
            G[r * d_A : (r + 1) * d_A, c * m : (c + 1) * m] = S_A @ P
    return G


def gossip_observer_noise_covariance(
    mixing_matrices: Sequence[np.ndarray],
    *,
    observer_selector: np.ndarray,
    state_noise_stds: Sequence[float] | float,
    observation_noise_std: float = 0.0,
    jitter: float = 1e-9,
) -> np.ndarray:
    r"""Covariance K for stacked observer noise, so full covariance is K \otimes I_p."""
    Ws = _validate_mixing_matrices(mixing_matrices)
    T = len(Ws)
    S_A = _validate_selector(observer_selector, num_nodes=int(Ws[0].shape[0]))
    d_A = int(S_A.shape[0])
    if isinstance(state_noise_stds, (int, float)):
        sigmas = np.full((T,), float(state_noise_stds), dtype=np.float64)
    else:
        sigmas = np.asarray(tuple(float(v) for v in state_noise_stds), dtype=np.float64)
        if sigmas.shape != (T,):
            raise ValueError(f"state_noise_stds must be scalar or have length {T}.")
    if np.any(~np.isfinite(sigmas)) or np.any(sigmas < 0.0):
        raise ValueError("state_noise_stds must be finite and >= 0.")
    obs_sigma = float(observation_noise_std)
    if not np.isfinite(obs_sigma) or obs_sigma < 0.0:
        raise ValueError("observation_noise_std must be finite and >= 0.")
    jit = float(jitter)
    if not np.isfinite(jit) or jit < 0.0:
        raise ValueError("jitter must be finite and >= 0.")

    G = gossip_noise_transfer_matrix(Ws, observer_selector=S_A)
    m = int(Ws[0].shape[0])
    D = np.zeros((T * m, T * m), dtype=np.float64)
    for t, sigma in enumerate(sigmas):
        sl = slice(t * m, (t + 1) * m)
        D[sl, sl] = (float(sigma) ** 2) * np.eye(m, dtype=np.float64)
    K = G @ D @ G.T
    if obs_sigma > 0.0:
        K += (obs_sigma**2) * np.eye(T * d_A, dtype=np.float64)
    if jit > 0.0:
        K += jit * np.eye(T * d_A, dtype=np.float64)
    return (0.5 * (K + K.T)).astype(np.float64, copy=False)


# Backwards-compatible alias used by earlier drafts.
def gossip_observation_covariance(
    mixing_matrices: Sequence[np.ndarray],
    *,
    observer_selector: np.ndarray,
    process_noise_std: float,
    observation_noise_std: float = 0.0,
    jitter: float = 1e-9,
) -> np.ndarray:
    return gossip_observer_noise_covariance(
        mixing_matrices,
        observer_selector=observer_selector,
        state_noise_stds=float(process_noise_std),
        observation_noise_std=float(observation_noise_std),
        jitter=float(jitter),
    )


def constant_mixing_matrices(
    mixing_matrix: np.ndarray, *, num_rounds: int
) -> list[np.ndarray]:
    T = int(num_rounds)
    if T <= 0:
        raise ValueError("num_rounds must be positive.")
    W = np.asarray(mixing_matrix, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("mixing_matrix must be square.")
    return [W.copy() for _ in range(T)]
