# cohort_dp/clustering.py
from __future__ import annotations
from typing import Tuple
import numpy as np


def kmeans_lloyd(
    X: np.ndarray,
    n_clusters: int,
    n_iters: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple numpy k-means (Lloyd).
    Returns:
      centers: (k, d)
      labels: (n,)
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    k = int(n_clusters)
    if k <= 0 or k > n:
        raise ValueError("n_clusters must be in [1, n].")

    centers = X[rng.choice(n, size=k, replace=False)].copy()

    for _ in range(int(n_iters)):
        # dist^2 = ||x||^2 + ||c||^2 - 2 x·c
        X2 = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
        C2 = np.sum(centers * centers, axis=1, keepdims=True).T  # (1,k)
        d2 = np.maximum(X2 + C2 - 2.0 * (X @ centers.T), 0.0)
        labels = np.argmin(d2, axis=1)

        new_centers = np.zeros_like(centers)
        counts = np.zeros((k,), dtype=int)

        for i in range(n):
            c = int(labels[i])
            new_centers[c] += X[i]
            counts[c] += 1

        for c in range(k):
            if counts[c] == 0:
                new_centers[c] = X[rng.integers(0, n)]
                counts[c] = 1
            else:
                new_centers[c] /= float(counts[c])

        if np.allclose(new_centers, centers, atol=1e-6, rtol=0.0):
            centers = new_centers
            break
        centers = new_centers

    return centers, labels.astype(int)


def kmeans_centers(
    X: np.ndarray,
    n_clusters: int,
    n_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Convenience wrapper: centers only."""
    centers, _ = kmeans_lloyd(X, n_clusters=n_clusters, n_iters=n_iters, rng=rng)
    return centers
