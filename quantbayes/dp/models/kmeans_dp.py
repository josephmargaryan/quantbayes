from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ..mechanisms.gaussian import gaussian_noise


@dataclass
class DPkMeansTranscript:
    f_list: List[np.ndarray]  # list over ell: (k, d) cluster sums with noise added
    g_list: List[np.ndarray]  # list over ell: (k,) counts with noise added


def _euclidean_assignments(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Compute nearest-center assignments with tie-break by lower index.
    Returns int array of shape (n,) with values in [0, k-1].
    """
    # distances: n x k
    dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    # argmin with stable tie-break: numpy argmin returns first occurrence (lowest index)
    return np.argmin(dists, axis=1)


def dp_kmeans_rho(t: int, sigma: float, sigma_prime: float) -> float:
    """
    zCDP parameter for the M_means' transcript mechanism (per assignment analysis):
      rho = 2t / sigma^2 + t / sigma_prime^2
    """
    if t <= 0:
        raise ValueError("t must be >= 1")
    if sigma <= 0 or sigma_prime <= 0:
        raise ValueError("sigma and sigma_prime must be > 0")
    return (2.0 * t) / (sigma**2) + (t / (sigma_prime**2))


class DPkMeans:
    """
    Private k-means variant from the assignment.

    Assumes input rows are L2-clipped to norm <= 1 (caller responsibility).
    Parameters
    ----------
    k : int
        Number of clusters.
    t : int
        Number of iterations.
    sigma : float
        Std dev for Gaussian noise added to cluster sums (per cluster, per iteration).
    sigma_prime : float
        Std dev for Gaussian noise added to counts (per cluster, per iteration).
    seed : Optional[int]
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        k: int,
        t: int,
        sigma: float,
        sigma_prime: float,
        seed: Optional[int] = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be >= 1")
        if t <= 0:
            raise ValueError("t must be >= 1")
        if sigma <= 0 or sigma_prime <= 0:
            raise ValueError("sigma and sigma_prime must be > 0")
        self.k = k
        self.t = t
        self.sigma = float(sigma)
        self.sigma_prime = float(sigma_prime)
        self.rng = np.random.RandomState(seed)
        self._centers: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        return_transcript: bool = False,
    ) -> Tuple[np.ndarray, Optional[DPkMeansTranscript]]:
        """
        Run t rounds starting from a random partition of [n] into k disjoint sets.
        Implements:
          c_i^{(ell)} = (1 / max(1, n_i^{(ell-1)})) * ( z_{ell,i} + sum_{j in S_i^{(ell-1)}} x_j )
          S_i^{(ell)} = nearest to c_i^{(ell)}
          n_i^{(ell)} = |S_i^{(ell)}| + z'_{ell,i}
        Returns final centers c^{(t)} and optional transcript (f_ell + z_ell, g_ell + z'_ell).
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = self.k
        # Random disjoint sets: random permutation then chunk
        perm = self.rng.permutation(n)
        chunks = np.array_split(perm, k)
        S_prev = [np.array(chunk, dtype=int) for chunk in chunks]
        n_prev = [len(s) for s in S_prev]

        f_list: List[np.ndarray] = []
        g_list: List[np.ndarray] = []

        centers = np.zeros((k, d), dtype=float)

        for ell in range(1, self.t + 1):
            # (a) Noisy cluster sums -> centers
            noisy_sums = np.zeros((k, d), dtype=float)
            for i in range(k):
                s_idx = S_prev[i]
                sum_vec = X[s_idx].sum(axis=0) if s_idx.size > 0 else np.zeros(d)
                z = self.rng.normal(0.0, self.sigma, size=d)
                noisy_sums[i] = sum_vec + z
                denom = max(1, n_prev[i])
                centers[i] = noisy_sums[i] / float(denom)

            # (b) Reassign using c^{(ell)}
            assignments = _euclidean_assignments(X, centers)
            S_curr = [np.where(assignments == i)[0] for i in range(k)]

            # (c) Noisy counts
            noisy_counts = np.zeros(k, dtype=float)
            for i in range(k):
                zc = self.rng.normal(0.0, self.sigma_prime)
                noisy_counts[i] = float(len(S_curr[i])) + zc

            if return_transcript:
                f_list.append(noisy_sums.copy())
                g_list.append(noisy_counts.copy())

            # Prepare next round
            S_prev = S_curr
            n_prev = [len(s) for s in S_prev]

        self._centers = centers.copy()
        transcript = (
            DPkMeansTranscript(f_list=f_list, g_list=g_list)
            if return_transcript
            else None
        )
        return centers, transcript

    @property
    def centers_(self) -> np.ndarray:
        if self._centers is None:
            raise ValueError("Model not yet fitted.")
        return self._centers
