# quantbayes/ball_dp/reconstruction/vectorizers.py
from __future__ import annotations

from typing import Any, Optional

import numpy as np

Array = np.ndarray


def l2_distance(u: Array, v: Array) -> float:
    u = np.asarray(u).reshape(-1)
    v = np.asarray(v).reshape(-1)
    return float(np.linalg.norm(u - v))


# -------------------------
# Convex-head vectorizers
# -------------------------


def vectorize_prototypes(mus: Array) -> Array:
    """mus: (K,d) -> (K*d,)"""
    return np.asarray(mus, dtype=np.float64).reshape(-1)


def vectorize_softmax(W: Array, b: Array) -> Array:
    """W: (K,d), b: (K,) -> vec([W,b])"""
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return np.concatenate([W.reshape(-1), b], axis=0)


def vectorize_binary_linear(w: Array, b: float | Array) -> Array:
    """w: (d,), b: scalar -> (d+1,)"""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if b.size != 1:
        raise ValueError("b must be scalar-like.")
    return np.concatenate([w, b], axis=0)


# -------------------------
# Eqx model vectorizer
# -------------------------


def vectorize_eqx_model(
    model: Any,
    *,
    state: Optional[Any] = None,
    include_state: bool = False,
) -> Array:
    """
    Flattens an Equinox model (and optionally its state) into a single 1D numpy vector.
    Useful for shadow-model identification distance computations.

    Notes:
      - We only include inexact arrays (float params) to avoid pulling in ints/bools.
      - If include_state=True, we also flatten float arrays inside `state` if it is a pytree.
    """
    import equinox as eqx
    import jax
    import numpy as np

    params = eqx.filter(model, eqx.is_inexact_array)
    leaves = jax.tree_util.tree_leaves(params)
    flat = [np.ravel(np.asarray(x, dtype=np.float64)) for x in leaves]
    vec = np.concatenate(flat, axis=0) if flat else np.zeros((0,), dtype=np.float64)

    if include_state and (state is not None):
        st_leaves = jax.tree_util.tree_leaves(state)
        st_flat = []
        for x in st_leaves:
            # only numeric arrays; ignore ints/bools/None
            try:
                arr = np.asarray(x)
            except Exception:
                continue
            if arr.dtype.kind in ("f", "c"):  # float/complex
                st_flat.append(np.ravel(arr.astype(np.float64)))
        if st_flat:
            vec = np.concatenate([vec] + st_flat, axis=0)

    return vec
