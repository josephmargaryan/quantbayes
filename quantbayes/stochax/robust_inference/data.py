# quantbayes/stochax/robust_inference/data.py
from __future__ import annotations
import jax
import jax.random as jr
from typing import List, Tuple
import numpy as np
import jax.numpy as jnp
from torchvision import datasets

from quantbayes.stochax.trainer.train import predict as _predict


def _standardize(Xtr: np.ndarray, Xte: np.ndarray):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def load_mnist(seed=0, limit_train=None, limit_test=None):
    tr = datasets.MNIST(root="./data", train=True, download=True)
    te = datasets.MNIST(root="./data", train=False, download=True)
    Xtr = tr.data.numpy().astype(np.float32) / 255.0
    ytr = tr.targets.numpy().astype(np.int64)
    Xte = te.data.numpy().astype(np.float32) / 255.0
    yte = te.targets.numpy().astype(np.int64)
    Xtr = Xtr.reshape(-1, 28 * 28)
    Xte = Xte.reshape(-1, 28 * 28)
    Xtr, Xte = _standardize(Xtr, Xte)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    Xtr, ytr = Xtr[idx], ytr[idx]
    if limit_train is not None:
        Xtr, ytr = Xtr[:limit_train], ytr[:limit_train]
    if limit_test is not None:
        Xte, yte = Xte[:limit_test], yte[:limit_test]
    return jnp.asarray(Xtr), jnp.asarray(ytr), jnp.asarray(Xte), jnp.asarray(yte)


def load_cifar10(seed=0, limit_train=None, limit_test=None):
    tr = datasets.CIFAR10(root="./data", train=True, download=True)
    te = datasets.CIFAR10(root="./data", train=False, download=True)
    Xtr = tr.data.astype(np.float32) / 255.0
    ytr = np.array(tr.targets, dtype=np.int64)
    Xte = te.data.astype(np.float32) / 255.0
    yte = np.array(te.targets, dtype=np.int64)
    Xtr = Xtr.reshape(-1, 32 * 32 * 3)
    Xte = Xte.reshape(-1, 32 * 32 * 3)
    Xtr, Xte = _standardize(Xtr, Xte)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    Xtr, ytr = Xtr[idx], ytr[idx]
    if limit_train is not None:
        Xtr, ytr = Xtr[:limit_train], ytr[:limit_train]
    if limit_test is not None:
        Xte, yte = Xte[:limit_test], yte[:limit_test]
    return jnp.asarray(Xtr), jnp.asarray(ytr), jnp.asarray(Xte), jnp.asarray(yte)


def load_cifar100(seed=0, limit_train=None, limit_test=None):
    tr = datasets.CIFAR100(root="./data", train=True, download=True)
    te = datasets.CIFAR100(root="./data", train=False, download=True)
    Xtr = tr.data.astype(np.float32) / 255.0
    ytr = np.array(tr.targets, dtype=np.int64)
    Xte = te.data.astype(np.float32) / 255.0
    yte = np.array(te.targets, dtype=np.int64)
    Xtr = Xtr.reshape(-1, 32 * 32 * 3)
    Xte = Xte.reshape(-1, 32 * 32 * 3)
    Xtr, Xte = _standardize(Xtr, Xte)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    Xtr, ytr = Xtr[idx], ytr[idx]
    if limit_train is not None:
        Xtr, ytr = Xtr[:limit_train], ytr[:limit_train]
    if limit_test is not None:
        Xte, yte = Xte[:limit_test], yte[:limit_test]
    return jnp.asarray(Xtr), jnp.asarray(ytr), jnp.asarray(Xte), jnp.asarray(yte)


def load_synthetic(
    n_train: int = 8000,
    n_test: int = 2000,
    d: int = 64,
    k: int = 6,
    seed: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Synthetic multiclass: X ~ N(0, I_d), teacher W ~ N(0, 1/sqrt(d)),
    y ~ softmax(W^T x). Returns standardized X (train mean/var).
    """
    rng = np.random.default_rng(seed)

    # features
    Xtr = rng.standard_normal((n_train, d)).astype(np.float32)
    Xte = rng.standard_normal((n_test, d)).astype(np.float32)

    # teacher
    W = (rng.standard_normal((d, k)) / np.sqrt(d)).astype(np.float32)

    # labels from softmax
    logits_tr = Xtr @ W
    P_tr = np.exp(logits_tr - logits_tr.max(axis=1, keepdims=True))
    P_tr /= P_tr.sum(axis=1, keepdims=True)
    ytr = (
        (rng.random(n_train)[:, None] < P_tr.cumsum(axis=1))
        .argmax(axis=1)
        .astype(np.int64)
    )

    logits_te = Xte @ W
    P_te = np.exp(logits_te - logits_te.max(axis=1, keepdims=True))
    P_te /= P_te.sum(axis=1, keepdims=True)
    yte = (
        (rng.random(n_test)[:, None] < P_te.cumsum(axis=1))
        .argmax(axis=1)
        .astype(np.int64)
    )

    # standardize by train
    Xtr, Xte = _standardize(Xtr, Xte)

    return jnp.asarray(Xtr), jnp.asarray(ytr), jnp.asarray(Xte), jnp.asarray(yte)


def load_dataset(
    name: str,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    name = name.lower()
    if name == "mnist":
        Xtr, ytr, Xte, yte = load_mnist(seed=0)
        K = 10
    elif name == "cifar-10":
        Xtr, ytr, Xte, yte = load_cifar10(seed=0)
        K = 10
    elif name == "cifar-100":
        Xtr, ytr, Xte, yte = load_cifar100(seed=0)
        K = 100
    elif name == "synthetic":
        Xtr, ytr, Xte, yte = load_synthetic(k=6, seed=0)
        K = int(jnp.max(ytr)) + 1
    else:
        raise ValueError(f"Unknown dataset {name}")
    return Xtr, ytr, Xte, yte, K


def dirichlet_label_split(
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_clients: int,
    n_classes: int,
    alpha: float,
    *,
    seed=0,
    equalize_sizes=False,
    min_per_client=0,
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    rng = np.random.default_rng(seed)
    idx_by_c = [np.where(np.asarray(y) == c)[0] for c in range(n_classes)]
    idx_by_c = [rng.permutation(ix) for ix in idx_by_c]
    parts_per_class = []
    for c in range(n_classes):
        m = len(idx_by_c[c])
        if m == 0:
            parts_per_class.append(
                [np.asarray([], dtype=int) for _ in range(n_clients)]
            )
            continue
        probs = rng.dirichlet([alpha] * n_clients)
        counts = rng.multinomial(m, probs)
        if min_per_client > 0:
            while np.any(counts < min_per_client):
                hi, lo = counts.argmax(), counts.argmin()
                if counts[hi] <= min_per_client:
                    break
                counts[hi] -= 1
                counts[lo] += 1
        parts = []
        s = 0
        for cc in counts:
            parts.append(idx_by_c[c][s : s + cc])
            s += cc
        parts_per_class.append(parts)
    out = []
    for i in range(n_clients):
        ids = (
            np.concatenate([parts_per_class[c][i] for c in range(n_classes)])
            if n_classes > 0
            else np.asarray([], dtype=int)
        )
        rng.shuffle(ids)
        Xi, yi = X[ids], y[ids]
        out.append((Xi, yi))
    if equalize_sizes and out:
        mmin = min(len(yi) for _, yi in out)
        out = [(Xi[:mmin], yi[:mmin]) for Xi, yi in out]
    return out


def collect_probits_dataset(models, states, X, *, batch_size: int = 256, key=None):
    """
    Collect (N, n, K) probits by running each (single-sample) Equinox client model
    over X using your trainer.predict (vmapped internally).
    """
    n = len(models)
    N = int(X.shape[0])
    Ps = []
    for start in range(0, N, batch_size):
        xb = X[start : start + batch_size]  # (B, d)
        per_client = []
        for i, m in enumerate(models):
            k_i = jr.fold_in(
                jr.PRNGKey(0) if key is None else key, i * 1_000_003 + start
            )
            logits = _predict(m, states[i], xb, k_i)  # (B, K)
            per_client.append(jax.nn.softmax(logits, axis=-1))  # (B, K)
        Ps.append(jnp.stack(per_client, axis=1))  # (B, n, K)
    return jnp.concatenate(Ps, axis=0)  # (N, n, K)
