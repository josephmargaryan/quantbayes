# quantbayes/ball_dp/metrics.py
from __future__ import annotations

import numpy as np


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def l2_norms(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z)
    return np.linalg.norm(Z, axis=1)


def maybe_l2_normalize(Z: np.ndarray, enabled: bool, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization: z <- z / (||z|| + eps).
    """
    Z = np.asarray(Z)
    if not enabled:
        return Z
    n = np.linalg.norm(Z, axis=1, keepdims=True) + float(eps)
    return (Z / n).astype(Z.dtype, copy=False)


def clip_l2_rows(Z: np.ndarray, max_norm: float, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 clipping: z <- z * min(1, max_norm / (||z|| + eps)).

    This is important for DP baselines that assume a PUBLIC bound ||z|| <= B.
    If you claim B is a public bound, you should enforce it via clipping/normalization.
    """
    Z = np.asarray(Z)
    max_norm = float(max_norm)
    if max_norm <= 0:
        raise ValueError("max_norm must be > 0")

    n = np.linalg.norm(Z, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (n + float(eps)))
    return (Z * scale).astype(Z.dtype, copy=False)


def make_random_orthonormal_projection(
    d_in: int, d_out: int, *, seed: int = 0
) -> np.ndarray:
    """
    Create a random projection matrix P with orthonormal columns (d_in, d_out).
    Use the SAME P to project both e and e' so the metric is consistent.

    Note: uses QR; d_out should be <= d_in.
    """
    d_in = int(d_in)
    d_out = int(d_out)
    if d_in <= 0 or d_out <= 0:
        raise ValueError("d_in and d_out must be positive.")
    if d_out > d_in:
        raise ValueError("d_out must be <= d_in.")
    rng = np.random.default_rng(int(seed))
    R = rng.normal(size=(d_in, d_out)).astype(np.float64)
    Q, _ = np.linalg.qr(R, mode="reduced")  # (d_in, d_out)
    return Q.astype(np.float32, copy=False)


def apply_linear_projection(
    Z: np.ndarray,
    P: np.ndarray,
    *,
    l2_normalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply a fixed projection matrix P to embeddings:
      Zp = Z @ P

    Optionally L2-normalize rows afterward (recommended if you want a public bound B=1).
    """
    Z = np.asarray(Z, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    Zp = Z @ P
    if l2_normalize:
        n = np.linalg.norm(Zp, axis=1, keepdims=True) + float(eps)
        Zp = Zp / n
    return Zp.astype(np.float32, copy=False)


def make_random_coordinate_subset(
    d_in: int, d_keep: int, *, seed: int = 0
) -> np.ndarray:
    """
    Choose a random coordinate subset of size d_keep from {0,...,d_in-1}.
    Returns an int64 index array you can reuse.
    """
    d_in = int(d_in)
    d_keep = int(d_keep)
    if d_in <= 0 or d_keep <= 0:
        raise ValueError("d_in and d_keep must be positive.")
    if d_keep > d_in:
        raise ValueError("d_keep must be <= d_in.")
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(d_in, size=d_keep, replace=False)
    idx.sort()
    return idx.astype(np.int64)


def apply_coordinate_subset(
    Z: np.ndarray,
    idx: np.ndarray,
    *,
    l2_normalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply a fixed coordinate subset to embeddings: Zp = Z[:, idx].
    Optionally renormalize rows.
    """
    Z = np.asarray(Z, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    Zp = Z[:, idx]
    if l2_normalize:
        n = np.linalg.norm(Zp, axis=1, keepdims=True) + float(eps)
        Zp = Zp / n
    return Zp.astype(np.float32, copy=False)


def apply_isotropic_noise_then_normalize(
    Z: np.ndarray,
    *,
    noise_std: float,
    seed: int = 0,
    l2_normalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    A deliberately "bad" metric transform:
      Zp = normalize(Z + N(0, noise_std^2 I))

    This can destroy semantic neighborhoods while still preserving the public bound.
    """
    Z = np.asarray(Z, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    Zp = Z + rng.normal(0.0, float(noise_std), size=Z.shape)
    if l2_normalize:
        n = np.linalg.norm(Zp, axis=1, keepdims=True) + float(eps)
        Zp = Zp / n
    return Zp.astype(np.float32, copy=False)
