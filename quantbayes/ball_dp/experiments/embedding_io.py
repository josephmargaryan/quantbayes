#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

EMBEDDING_KEYS = ("X_train", "y_train", "X_test", "y_test")


def get_jax_device(require_gpu: bool):
    gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
    if gpu_devices:
        return gpu_devices[0]
    if require_gpu:
        raise RuntimeError(
            "JAX does not see a GPU. Install a GPU-enabled JAX build and verify "
            "`jax.devices()` reports a GPU."
        )
    return jax.devices()[0]


def default_cache_dir(root: str | Path = "./data") -> Path:
    return Path(root) / "embeddings"


def save_embedding_npz(
    output_path: str | Path,
    X_train,
    y_train,
    X_test,
    y_test,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X_train=np.asarray(jax.device_get(X_train), dtype=np.float32),
        y_train=np.asarray(jax.device_get(y_train), dtype=np.int32),
        X_test=np.asarray(jax.device_get(X_test), dtype=np.float32),
        y_test=np.asarray(jax.device_get(y_test), dtype=np.int32),
    )
    return output_path


def load_embedding_npz(
    input_path: str | Path,
    require_jax_gpu: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Embedding file does not exist: {input_path}")

    with np.load(input_path) as data:
        missing = [k for k in EMBEDDING_KEYS if k not in data.files]
        if missing:
            raise KeyError(
                f"Missing keys in {input_path}: {missing}. "
                f"Expected keys: {EMBEDDING_KEYS}"
            )

        X_train_np = np.asarray(data["X_train"], dtype=np.float32)
        y_train_np = np.asarray(data["y_train"], dtype=np.int32)
        X_test_np = np.asarray(data["X_test"], dtype=np.float32)
        y_test_np = np.asarray(data["y_test"], dtype=np.int32)

    jax_device = get_jax_device(require_gpu=require_jax_gpu)

    X_train = jax.device_put(X_train_np, device=jax_device)
    y_train = jax.device_put(y_train_np, device=jax_device)
    X_test = jax.device_put(X_test_np, device=jax_device)
    y_test = jax.device_put(y_test_np, device=jax_device)
    return X_train, y_train, X_test, y_test
