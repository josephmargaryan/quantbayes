# cohort_dp/synthetic.py
from __future__ import annotations
from typing import Tuple
import numpy as np


def standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)


def make_synthetic_patients(
    n: int,
    d: int,
    n_clusters: int,
    rng: np.random.Generator,
    class_sep: float = 3.0,
    noise_std: float = 1.0,
) -> np.ndarray:
    centers = rng.normal(size=(n_clusters, d)) * class_sep
    cid = rng.integers(0, n_clusters, size=n)
    X = centers[cid] + rng.normal(scale=noise_std, size=(n, d))
    return standardize(X).astype(float)


def make_synthetic_embeddings_with_labels(
    n: int,
    d: int,
    n_classes: int,
    rng: np.random.Generator,
    class_sep: float = 3.0,
    noise_std: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    centers = rng.normal(size=(n_classes, d)) * class_sep
    y = rng.integers(0, n_classes, size=n)
    X = centers[y] + rng.normal(scale=noise_std, size=(n, d))
    return standardize(X).astype(float), y.astype(int)


def make_synthetic_multihospital(
    n_total: int,
    d: int,
    n_classes: int,
    n_hospitals: int,
    rng: np.random.Generator,
    class_sep: float,
    noise_std: float,
    hospital_shift_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = rng.normal(size=(n_classes, d)) * class_sep
    y = rng.integers(0, n_classes, size=n_total)
    base = centers[y] + rng.normal(scale=noise_std, size=(n_total, d))

    h = rng.integers(0, n_hospitals, size=n_total)
    shifts = rng.normal(scale=hospital_shift_std, size=(n_hospitals, d))
    X = base + shifts[h]
    X = standardize(X)
    return X.astype(float), y.astype(int), h.astype(int)
