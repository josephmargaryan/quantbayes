"""Decentralized noisy-prototype utilities for Paper 3 experiments.

This module implements a deliberately simple utility benchmark: each node forms
class-sum prototypes from clipped embeddings, releases noisy local sums, and then
communicates by deterministic gossip.  Labels/counts are treated as public in this
benchmark; the private object is the feature vector contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PrototypeGossipResult:
    """Summary of a decentralized prototype run."""

    node_accuracies: np.ndarray
    accuracy_mean: float
    accuracy_min: float
    consensus_disagreement: float
    prototypes_by_node: np.ndarray


def clip_l2_rows(X: np.ndarray, clip_norm: float) -> np.ndarray:
    """Clip each row of ``X`` to Euclidean norm at most ``clip_norm``."""

    C = float(clip_norm)
    if C <= 0.0:
        raise ValueError("clip_norm must be positive")
    Xf = np.asarray(X, dtype=float)
    norms = np.linalg.norm(Xf, axis=1, keepdims=True)
    return Xf * np.minimum(1.0, C / np.maximum(norms, 1e-12))


def partition_indices_iid(
    num_examples: int, num_nodes: int, *, seed: int = 0
) -> list[np.ndarray]:
    """Randomly partition examples into approximately equal node shards."""

    n = int(num_examples)
    m = int(num_nodes)
    if n < 0 or m <= 0:
        raise ValueError("num_examples must be nonnegative and num_nodes positive")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    return [np.asarray(x, dtype=np.int64) for x in np.array_split(perm, m)]


def local_class_sums(
    X: np.ndarray,
    y: np.ndarray,
    shards: Sequence[np.ndarray],
    *,
    num_classes: int,
    clip_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return node-local class sums and counts.

    Returns
    -------
    sums:
        Array of shape ``(num_nodes, num_classes, feature_dim)``.
    counts:
        Array of shape ``(num_nodes, num_classes)``.
    """

    Xc = clip_l2_rows(X, clip_norm)
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    K = int(num_classes)
    if K <= 0:
        raise ValueError("num_classes must be positive")
    if len(Xc) != len(labels):
        raise ValueError("X and y must have the same number of rows")
    m = len(shards)
    p = int(Xc.shape[1])
    sums = np.zeros((m, K, p), dtype=float)
    counts = np.zeros((m, K), dtype=float)
    for i, idx in enumerate(shards):
        idx_arr = np.asarray(idx, dtype=np.int64)
        Xi = Xc[idx_arr]
        yi = labels[idx_arr]
        for k in range(K):
            mask = yi == k
            if np.any(mask):
                sums[i, k] = Xi[mask].sum(axis=0)
                counts[i, k] = float(np.sum(mask))
    return sums, counts


def gossip_array(values: np.ndarray, W: np.ndarray, rounds: int) -> np.ndarray:
    """Apply ``rounds`` of linear gossip to the first axis of an array."""

    out = np.asarray(values, dtype=float).copy()
    M = np.asarray(W, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] != out.shape[0]:
        raise ValueError("W must be square with size matching values.shape[0]")
    for _ in range(int(rounds)):
        out = np.tensordot(M, out, axes=(1, 0))
    return out


def nearest_prototype_predict(X: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Predict labels by nearest Euclidean prototype."""

    Xf = np.asarray(X, dtype=float)
    P = np.asarray(prototypes, dtype=float)
    d2 = np.sum((Xf[:, None, :] - P[None, :, :]) ** 2, axis=2)
    return np.argmin(d2, axis=1).astype(np.int64)


def run_noisy_prototype_gossip(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    W: np.ndarray,
    num_classes: int,
    rounds: int,
    clip_norm: float,
    noise_std: float,
    seed: int = 0,
    shards: Sequence[np.ndarray] | None = None,
) -> PrototypeGossipResult:
    """Run the decentralized noisy-prototype benchmark.

    Counts are gossiped without noise because this benchmark conditions on labels
    and treats class counts as public metadata.  Feature sums receive iid Gaussian
    noise before gossip.
    """

    Xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=np.int64).reshape(-1)
    Xte = np.asarray(X_test, dtype=float)
    yte = np.asarray(y_test, dtype=np.int64).reshape(-1)
    m = int(np.asarray(W).shape[0])
    if shards is None:
        shards = partition_indices_iid(len(Xtr), m, seed=seed)
    sums, counts = local_class_sums(
        Xtr, ytr, shards, num_classes=int(num_classes), clip_norm=float(clip_norm)
    )
    rng = np.random.default_rng(int(seed))
    noisy_sums = sums + float(noise_std) * rng.normal(size=sums.shape)
    mixed_sums = gossip_array(noisy_sums, W, int(rounds))
    mixed_counts = gossip_array(counts, W, int(rounds))
    protos = mixed_sums / np.maximum(mixed_counts[..., None], 1e-12)
    accs = []
    for node in range(m):
        pred = nearest_prototype_predict(Xte, protos[node])
        accs.append(float(np.mean(pred == yte)))
    acc_arr = np.asarray(accs, dtype=float)
    consensus = float(
        np.mean(np.linalg.norm(protos - protos.mean(axis=0, keepdims=True), axis=-1))
    )
    return PrototypeGossipResult(
        node_accuracies=acc_arr,
        accuracy_mean=float(np.mean(acc_arr)),
        accuracy_min=float(np.min(acc_arr)),
        consensus_disagreement=consensus,
        prototypes_by_node=protos,
    )
