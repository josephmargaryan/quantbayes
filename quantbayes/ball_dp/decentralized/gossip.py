from __future__ import annotations

from typing import Sequence

import numpy as np


def selector_matrix(
    observer_nodes: Sequence[int],
    *,
    num_nodes: int,
) -> np.ndarray:
    """Build the row-selector S_A for an observer set A.

    Returns an array of shape (|A|, num_nodes) with one-hot rows selecting the
    coordinates visible to the observer set.
    """
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


def gossip_transfer_matrix(
    mixing_matrices: Sequence[np.ndarray],
    *,
    observer_selector: np.ndarray,
    attacked_node: int,
) -> np.ndarray:
    r"""Build the theorem-side transfer matrix H_{A<-j} for linear gossip.

    Model
    -----
    We assume the linear recursion
        x_{t+1} = W_t x_t + u_t + xi_t,
    and the observer view
        y_t = (S_A \otimes I_p) x_t.

    Indexing convention
    -------------------
    - `mixing_matrices[t]` is W_t for t = 0, ..., T-1.
    - The returned matrix H has shape (T * d_A, T), where d_A = S_A.shape[0].
    - Row block `r` (0-based) corresponds to the observed output y_{r+1}.
    - Column `c` corresponds to the attacked node's sensitive block u_c.

    Therefore the row-block / column entry is
        H[r, c] = S_A W_r W_{r-1} ... W_{c+1} e_j   if c <= r,
                 = 0                                 otherwise.

    This is exactly the 0-based version of the paper's block formula
        [H_{A<-j}]_{t, tau} = S_A P_{tau+1:t-1} e_j,    tau < t,
    when the stacked observer view is (y_1, ..., y_T).
    """
    Ws = [np.asarray(W, dtype=np.float64) for W in mixing_matrices]
    if not Ws:
        raise ValueError("mixing_matrices must be non-empty.")

    m = Ws[0].shape[0]
    if Ws[0].ndim != 2 or Ws[0].shape[1] != m:
        raise ValueError("Every mixing matrix must be square.")
    for t, W in enumerate(Ws):
        if W.shape != (m, m):
            raise ValueError(
                f"mixing_matrices[{t}] has shape {W.shape}, expected {(m, m)}."
            )

    S_A = np.asarray(observer_selector, dtype=np.float64)
    if S_A.ndim != 2 or S_A.shape[1] != m:
        raise ValueError(
            "observer_selector must have shape (d_A, num_nodes) and match the mixing matrices."
        )

    j = int(attacked_node)
    if j < 0 or j >= m:
        raise ValueError("attacked_node is outside [0, num_nodes).")

    T = len(Ws)
    d_A = int(S_A.shape[0])
    H = np.zeros((T * d_A, T), dtype=np.float64)

    e_j = np.zeros((m,), dtype=np.float64)
    e_j[j] = 1.0

    for r in range(T):
        coeff = e_j.copy()
        for c in range(r, -1, -1):
            if c < r:
                coeff = Ws[c + 1] @ coeff
            H[r * d_A : (r + 1) * d_A, c] = S_A @ coeff

    return H
