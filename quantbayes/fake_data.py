"""
synthetic_data.py

Quick synthetic dataset generators for debugging models:
- regression (tabular)
- binary classification (tabular)
- multiclass classification (tabular)
- time series (sequence -> next-step target)
- image classification (NCHW)
- image segmentation (image + mask, both NCHW)

Notes:
- Uses local RNGs (np.random.default_rng) to avoid global seed side-effects.
- Image/mask outputs are always [N, C, H, W].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from sklearn.datasets import make_blobs, make_classification


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _np_rng(random_seed: int | None) -> np.random.Generator:
    return np.random.default_rng(random_seed)


def _to_backend(x, as_jax: bool):
    return jnp.asarray(x) if as_jax else np.asarray(x)


def df_to_arrays(
    df: pd.DataFrame,
    target_col: str = "target",
    categorical: str = "integer",  # "integer" or "onehot"
    float_dtype=np.float32,
    as_jax: bool = False,
):
    """
    Convert a DataFrame produced by the generators into (X, y) arrays.

    categorical="integer": categorical columns -> integer codes
    categorical="onehot": categorical columns -> one-hot expansion via get_dummies

    Returns
    -------
    X : (N, D) float array
    y : (N,) array
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' not found in df columns.")

    y = df[target_col].to_numpy()
    X_df = df.drop(columns=[target_col])

    cat_cols = [c for c in X_df.columns if str(X_df[c].dtype) == "category"]

    if categorical not in {"integer", "onehot"}:
        raise ValueError("categorical must be 'integer' or 'onehot'.")

    if categorical == "integer":
        X_df2 = X_df.copy()
        for c in cat_cols:
            X_df2[c] = X_df2[c].cat.codes.astype(np.int32)
        X = X_df2.to_numpy(dtype=float_dtype)
    else:
        X = pd.get_dummies(X_df, columns=cat_cols, drop_first=False).to_numpy(
            dtype=float_dtype
        )

    return _to_backend(X, as_jax), _to_backend(y, as_jax)


# -----------------------------------------------------------------------------
# Tabular regression
# -----------------------------------------------------------------------------
def generate_regression_data(
    n_samples: int = 1000,
    n_continuous: int = 1,
    n_categorical: int = 0,
    n_categories: int = 3,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Synthetic regression data with continuous + categorical features.

    Returns
    -------
    pd.DataFrame with columns:
      cont_feature_*, cat_feature_*, target
    """
    rng = _np_rng(random_seed)
    df = pd.DataFrame()

    # Continuous features: sinusoidal waves with noise
    t = np.linspace(0, 2 * np.pi * 10, n_samples, dtype=np.float32)
    for i in range(n_continuous):
        freq = 0.1 * (i + 1)
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = 1.0 + 0.5 * i
        noise = rng.normal(scale=0.1, size=n_samples).astype(np.float32)
        df[f"cont_feature_{i+1}"] = (
            amplitude * np.sin(freq * t + phase).astype(np.float32) + noise
        )

    # Categorical features
    for i in range(n_categorical):
        vals = rng.integers(0, n_categories, size=n_samples, dtype=np.int32)
        col = f"cat_feature_{i+1}"
        df[col] = pd.Series(vals).astype("category")

    # Target: average of continuous features + noise
    if n_continuous > 0:
        target = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_continuous):
            target += df[f"cont_feature_{i+1}"].to_numpy(dtype=np.float32)
        target /= float(n_continuous)
    else:
        target = np.zeros(n_samples, dtype=np.float32)

    target += rng.normal(scale=0.2, size=n_samples).astype(np.float32)
    df["target"] = target

    return df


# -----------------------------------------------------------------------------
# Tabular binary classification
# -----------------------------------------------------------------------------
def generate_binary_classification_data(
    n_samples: int = 500,
    n_continuous: int = 3,
    n_categorical: int = 2,
    class_sep: float = 1.0,
    random_seed: int = 42,
    n_categories: int = 3,
    n_clusters_per_class: int | None = None,
) -> pd.DataFrame:
    """
    Synthetic binary classification data with continuous + categorical features.

    Returns
    -------
    pd.DataFrame with columns:
      cont_feature_*, cat_feature_*, target
    """
    if n_continuous < 1:
        raise ValueError("n_continuous must be >= 1 for make_classification.")

    rng = _np_rng(random_seed)

    # Informative features
    n_informative = n_continuous if n_continuous < 4 else max(1, n_continuous // 2)

    # Clusters
    if n_clusters_per_class is None:
        n_clusters_per_class = 1 if n_continuous == 1 else 2

    X_cont, y = make_classification(
        n_samples=n_samples,
        n_features=n_continuous,
        n_informative=n_informative,
        n_redundant=max(0, n_continuous // 4),
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=0.1,
        random_state=random_seed,
    )

    df_cont = pd.DataFrame(
        X_cont.astype(np.float32),
        columns=[f"cont_feature_{i+1}" for i in range(n_continuous)],
    )

    if n_categorical > 0:
        X_cat = rng.integers(
            0, n_categories, size=(n_samples, n_categorical), dtype=np.int32
        )
        df_cat = pd.DataFrame(
            X_cat, columns=[f"cat_feature_{i+1}" for i in range(n_categorical)]
        )
        for col in df_cat.columns:
            df_cat[col] = df_cat[col].astype("category")
        df = pd.concat([df_cont, df_cat], axis=1)
    else:
        df = df_cont

    df["target"] = y.astype(np.int32)
    return df


# -----------------------------------------------------------------------------
# Tabular multiclass classification
# -----------------------------------------------------------------------------
def generate_multiclass_classification_data(
    n_samples: int = 500,
    n_continuous: int = 2,
    n_categorical: int = 2,
    n_classes: int = 3,
    cluster_std: float = 2.0,
    random_seed: int = 42,
    n_categories: int = 3,
) -> pd.DataFrame:
    """
    Synthetic multiclass classification data with continuous + categorical features.

    Returns
    -------
    pd.DataFrame with columns:
      cont_feature_*, cat_feature_*, target
    """
    if n_continuous < 1:
        raise ValueError("n_continuous must be >= 1 for make_blobs.")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2.")

    rng = _np_rng(random_seed)

    X_cont, y = make_blobs(
        n_samples=n_samples,
        n_features=n_continuous,
        centers=n_classes,
        cluster_std=cluster_std,
        random_state=random_seed,
    )

    # Add a bit of extra noise
    X_cont = (X_cont + rng.normal(0, 0.5, size=X_cont.shape)).astype(np.float32)

    df_cont = pd.DataFrame(
        X_cont, columns=[f"cont_feature_{i+1}" for i in range(n_continuous)]
    )

    if n_categorical > 0:
        X_cat = rng.integers(
            0, n_categories, size=(n_samples, n_categorical), dtype=np.int32
        )
        df_cat = pd.DataFrame(
            X_cat, columns=[f"cat_feature_{i+1}" for i in range(n_categorical)]
        )
        for col in df_cat.columns:
            df_cat[col] = df_cat[col].astype("category")
        df = pd.concat([df_cont, df_cat], axis=1)
    else:
        df = df_cont

    df["target"] = y.astype(np.int32)
    return df


# -----------------------------------------------------------------------------
# Time series (JAX)
# -----------------------------------------------------------------------------
def create_synthetic_time_series(
    n_samples: int = 1000,
    seq_len: int = 10,
    num_features: int = 1,
    train_split: float = 0.8,
    random_seed: int = 0,
    noise_std: float = 0.1,
):
    """
    Create synthetic time series data (next-step prediction).

    Returns
    -------
    X_train, X_val, y_train, y_val
      X_*: (N, seq_len, num_features)
      y_*: (N,)
    """
    if seq_len >= n_samples:
        raise ValueError("seq_len must be < n_samples.")
    if not (0.0 < train_split < 1.0):
        raise ValueError("train_split must be in (0, 1).")

    key = jax.random.PRNGKey(random_seed)
    t = jnp.linspace(0.0, 10.0, n_samples)

    key, noise_key = jax.random.split(key)
    y = jnp.sin(t) + noise_std * jax.random.normal(noise_key, shape=t.shape)

    # Features: noisy copies of y
    keys = jax.random.split(key, num_features)
    X = jnp.stack(
        [y + noise_std * jax.random.normal(k, shape=y.shape) for k in keys], axis=-1
    )  # (n_samples, num_features)

    # Build sequences (vectorized)
    idx = jnp.arange(n_samples - seq_len)[:, None] + jnp.arange(seq_len)[None, :]
    sequences_X = X[idx]  # (n_samples-seq_len, seq_len, num_features)
    sequences_y = y[seq_len:]  # (n_samples-seq_len,)

    split_idx = int(train_split * sequences_X.shape[0])
    X_train, X_val = sequences_X[:split_idx], sequences_X[split_idx:]
    y_train, y_val = sequences_y[:split_idx], sequences_y[split_idx:]

    return X_train, X_val, y_train, y_val


# -----------------------------------------------------------------------------
# Image classification (NCHW)
# -----------------------------------------------------------------------------
def generate_image_classification_data(
    n_samples: int = 256,
    n_classes: int = 4,
    channels: int = 1,
    height: int = 32,
    width: int = 32,
    noise_std: float = 0.15,
    signal: float = 1.0,
    random_seed: int | None = 0,
    as_jax: bool = False,
):
    """
    Synthetic image classification dataset.

    Returns
    -------
    X : (N, C, H, W) float32
    y : (N,) int64
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2.")
    if channels < 1:
        raise ValueError("channels must be >= 1.")

    rng = _np_rng(random_seed)

    # Coordinate grid for patterns
    yy, xx = np.meshgrid(
        np.linspace(0, 2 * np.pi, height, dtype=np.float32),
        np.linspace(0, 2 * np.pi, width, dtype=np.float32),
        indexing="ij",
    )

    X = rng.normal(0.0, noise_std, size=(n_samples, channels, height, width)).astype(
        np.float32
    )
    y = rng.integers(0, n_classes, size=(n_samples,), dtype=np.int64)

    for i in range(n_samples):
        k = int(y[i])
        # class-dependent sinusoidal pattern (learnable but simple)
        phase1, phase2 = rng.uniform(0.0, 2 * np.pi, size=2).astype(np.float32)
        freq = float(k + 1)

        pattern = np.sin(freq * xx + phase1) * np.cos(freq * yy + phase2)
        # normalize-ish to roughly [-0.5, 0.5]
        pattern = pattern.astype(np.float32)
        pattern = pattern / (np.max(np.abs(pattern)) + 1e-8)
        pattern = signal * pattern  # roughly [-signal, +signal]

        X[i] += pattern[None, :, :]  # broadcast to all channels

    return _to_backend(X, as_jax), _to_backend(y, as_jax)


# -----------------------------------------------------------------------------
# Image segmentation (image + mask, both NCHW)
# -----------------------------------------------------------------------------
def generate_image_and_mask_data(
    n_samples: int = 128,
    channels: int = 1,
    height: int = 64,
    width: int = 64,
    n_classes: int = 2,
    objects_per_image: tuple[int, int] = (1, 3),
    shape_types: tuple[str, ...] = ("circle", "rectangle"),
    noise_std: float = 0.10,
    signal: float = 1.0,
    random_seed: int | None = 0,
    mask_mode: str = "label",  # "label" -> (N,1,H,W) int32, "one_hot" -> (N,K,H,W) float32
    as_jax: bool = False,
):
    """
    Synthetic segmentation dataset.

    Outputs are always NCHW:
      X    : (N, C, H, W) float32
      mask : (N, 1, H, W) int32   if mask_mode="label"
          or (N, K, H, W) float32 if mask_mode="one_hot"

    Notes
    -----
    - Background is class 0.
    - Foreground objects use classes 1..K-1 (if K>1).
    - Objects may overlap; later objects overwrite earlier ones.
    """
    if channels < 1:
        raise ValueError("channels must be >= 1.")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2 (include background).")
    if objects_per_image[0] < 0 or objects_per_image[1] < objects_per_image[0]:
        raise ValueError("objects_per_image must be (min,max) with 0 <= min <= max.")
    if mask_mode not in {"label", "one_hot"}:
        raise ValueError("mask_mode must be 'label' or 'one_hot'.")

    rng = _np_rng(random_seed)

    X = rng.normal(0.0, noise_std, size=(n_samples, channels, height, width)).astype(
        np.float32
    )
    labels = np.zeros((n_samples, height, width), dtype=np.int32)

    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.int32),
        np.arange(width, dtype=np.int32),
        indexing="ij",
    )

    min_obj = objects_per_image[0]
    max_obj = objects_per_image[1]

    # Object size ranges (rough heuristics)
    min_r = max(2, min(height, width) // 10)
    max_r = max(min_r + 1, min(height, width) // 4)
    min_side = max(3, min(height, width) // 10)
    max_side = max(min_side + 1, min(height, width) // 3)

    for i in range(n_samples):
        n_obj = int(rng.integers(min_obj, max_obj + 1)) if max_obj > 0 else 0
        for _ in range(n_obj):
            cls = int(rng.integers(1, n_classes))  # 1..K-1
            shape = rng.choice(shape_types)

            if shape == "circle":
                r = int(rng.integers(min_r, max_r + 1))
                # ensure valid center range
                cy_low, cy_high = r, max(r + 1, height - r)
                cx_low, cx_high = r, max(r + 1, width - r)
                cy = int(rng.integers(cy_low, cy_high))
                cx = int(rng.integers(cx_low, cx_high))

                m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

            elif shape == "rectangle":
                h = int(rng.integers(min_side, max_side + 1))
                w = int(rng.integers(min_side, max_side + 1))
                y0 = int(rng.integers(0, max(1, height - h + 1)))
                x0 = int(rng.integers(0, max(1, width - w + 1)))
                y1 = min(height, y0 + h)
                x1 = min(width, x0 + w)

                m = np.zeros((height, width), dtype=bool)
                m[y0:y1, x0:x1] = True

            else:
                raise ValueError(f"Unknown shape type: {shape}")

            labels[i][m] = cls

        # Inject class-dependent signal into the image
        # background=0 -> 0 signal, class k -> (k/(K-1))*signal
        scale = (labels[i].astype(np.float32) / float(n_classes - 1)) * signal
        X[i] += scale[None, :, :]  # broadcast to channels

    if mask_mode == "label":
        mask = labels[:, None, :, :].astype(np.int32)  # (N,1,H,W)
    else:
        mask = np.zeros((n_samples, n_classes, height, width), dtype=np.float32)
        for k in range(n_classes):
            mask[:, k] = (labels == k).astype(np.float32)

    return _to_backend(X, as_jax), _to_backend(mask, as_jax)
