# quantbayes/ball_dp/reconstruction/experiments/mnist/common.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from quantbayes.ball_dp.analytical_gaussian_mechanism import calibrate_analytic_gaussian
from quantbayes.ball_dp.lz import lz_softmax_linear_bound
from quantbayes.ball_dp.reconstruction.types import Candidate
from quantbayes.ball_dp.reconstruction.priors import PoolBallPrior, l2_metric_batch
from quantbayes.ball_dp.reconstruction.reporting import mse, psnr_from_mse
from quantbayes.ball_dp.reconstruction.vision.mnist_autoencoder_torch import (
    load_or_train_mnist_autoencoder,
    encode_numpy,
    decode_numpy,
    MNISTConvAutoencoder,
)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def load_mnist_numpy(
    *,
    n_train: int,
    n_test: int,
    seed: int,
    digits: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Xtr: (N,1,28,28) float32 in [0,1]
      ytr: (N,) int64
      Xte: (M,1,28,28)
      yte: (M,)
    """
    tfm = T.Compose([T.ToTensor()])
    tr = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=tfm
    )
    te = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=tfm
    )

    rng = np.random.default_rng(seed)

    def collect(ds, n):
        idx_all = np.arange(len(ds))
        rng.shuffle(idx_all)
        X, y = [], []
        for i in idx_all:
            xi, yi = ds[int(i)]
            yi = int(yi)
            if digits is not None and yi not in set(int(d) for d in digits):
                continue
            X.append(xi.numpy())
            y.append(yi)
            if len(X) >= n:
                break
        X = np.stack(X, axis=0).astype(np.float32)
        y = np.array(y, dtype=np.int64)
        return X, y

    Xtr, ytr = collect(tr, int(n_train))
    Xte, yte = collect(te, int(n_test))
    return Xtr, ytr, Xte, yte


def flatten_pixels(X: np.ndarray) -> np.ndarray:
    return X.reshape((X.shape[0], -1)).astype(np.float64)


def l2_clip_rows(X: np.ndarray, B: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1) + 1e-12
    scale = np.minimum(1.0, float(B) / norms)
    return X * scale[:, None]


# -------------------------------
# Autoencoder: train/load + embed
# -------------------------------


@dataclass
class AEArtifacts:
    ae: MNISTConvAutoencoder
    embed_dim: int
    B_embed: float  # public bound used for DP calibration (we clip to it)
    ae_hist: dict


def get_or_train_ae(
    *,
    save_dir: str | Path,
    Xtr: np.ndarray,
    Xte: np.ndarray,
    embed_dim: int,
    ae_epochs: int,
    ae_batch_size: int,
    ae_lr: float,
    seed: int,
    B_embed: Optional[float] = None,
) -> AEArtifacts:
    device = get_device()
    save_dir = Path(save_dir)
    ckpt = save_dir / f"mnist_ae_embed{int(embed_dim)}.pt"

    # small val split for AE
    n = Xtr.shape[0]
    n_val = max(1, int(0.1 * n))
    X_train, X_val = Xtr[:-n_val], Xtr[-n_val:]

    ae, hist = load_or_train_mnist_autoencoder(
        ckpt_path=ckpt,
        X_train=X_train,
        X_val=X_val,
        device=device,
        embed_dim=int(embed_dim),
        epochs=int(ae_epochs),
        batch_size=int(ae_batch_size),
        lr=float(ae_lr),
        seed=int(seed),
    )

    # compute embeddings + pick a public bound
    Etr = encode_numpy(ae, Xtr, device=device, batch_size=512)  # float64
    norms = np.linalg.norm(Etr, axis=1)
    B_data = float(np.max(norms))

    if B_embed is None:
        # since MNIST is public, using max-norm is acceptable for experiments
        B_embed_use = B_data
    else:
        B_embed_use = float(B_embed)

    return AEArtifacts(
        ae=ae,
        embed_dim=int(embed_dim),
        B_embed=float(B_embed_use),
        ae_hist=hist,
    )


# -------------------------------
# Convex softmax training (torch)
# -------------------------------


def train_softmax_lbfgs_torch(
    *,
    X: np.ndarray,  # (N,d) float64
    y: np.ndarray,  # (N,) int64
    lam: float,
    max_iter: int = 800,
    tol_grad: float = 1e-11,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Deterministic-ish torch L-BFGS in float64.
    Returns (W, b, stationarity_norm).
    """
    import torch.optim as optim

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    K = int(np.max(y) + 1)
    d = int(X.shape[1])

    W = torch.zeros((K, d), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((K,), dtype=torch.float64, requires_grad=True)

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = X_t @ W.t() + b
        loss = torch.nn.functional.cross_entropy(logits, y_t, reduction="mean")
        reg = 0.5 * float(lam) * (W.pow(2).sum() + b.pow(2).sum())
        obj = loss + reg
        obj.backward()
        return obj

    opt = optim.LBFGS(
        [W, b],
        lr=1.0,
        max_iter=int(max_iter),
        tolerance_grad=float(tol_grad),
        tolerance_change=1e-18,
        line_search_fn="strong_wolfe",
        history_size=50,
    )
    opt.step(closure)

    # stationarity check (sum-grad + lam*n*theta)
    with torch.no_grad():
        logits = X_t @ W.t() + b
        p = torch.softmax(logits, dim=1)
        diff = p.clone()
        diff[torch.arange(X_t.shape[0]), y_t] -= 1.0
        sum_gW = diff.t() @ X_t
        sum_gb = diff.sum(dim=0)
        n = X_t.shape[0]
        stat = torch.norm(sum_gW + float(lam) * float(n) * W, p="fro") + torch.norm(
            sum_gb + float(lam) * float(n) * b, p=2
        )

    return W.detach().numpy(), b.detach().numpy(), float(stat.item())


def softmax_predict(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    logits = X @ W.T + b[None, :]
    return np.argmax(logits, axis=1).astype(np.int64)


def softmax_accuracy(
    W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray
) -> float:
    yhat = softmax_predict(W, b, X)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    return float(np.mean(yhat == y))


@dataclass
class SoftmaxPrecomp:
    Xt: np.ndarray  # (n,d+1)
    Wt: np.ndarray  # (K,d+1)
    diff: np.ndarray  # (n,K)
    sum_grad: np.ndarray  # (K,d+1)
    n: int
    K: int
    d1: int


def softmax_precompute(
    W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray
) -> SoftmaxPrecomp:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    n = X.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    Xt = np.concatenate([X, ones], axis=1)  # (n,d+1)

    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    Wt = np.concatenate([W, b[:, None]], axis=1)

    logits = Xt @ Wt.T
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p = p / p.sum(axis=1, keepdims=True)

    diff = p
    diff[np.arange(n), y] -= 1.0
    sum_grad = diff.T @ Xt

    return SoftmaxPrecomp(
        Xt=Xt,
        Wt=Wt,
        diff=diff,
        sum_grad=sum_grad,
        n=int(n),
        K=int(Wt.shape[0]),
        d1=int(Wt.shape[1]),
    )


def softmax_stationarity_norm(pre: SoftmaxPrecomp, lam: float) -> float:
    # ||sum_grad + lam*n*Wt||_F
    R = pre.sum_grad + float(lam) * float(pre.n) * pre.Wt
    return float(np.linalg.norm(R))


def softmax_pseudo_missing_gradient(
    pre: SoftmaxPrecomp, *, target_idx: int, lam: float
) -> np.ndarray:
    """
    G_missing(j) = -lam*n*Wt - sum_{i != j} grad_i
    where grad_i = outer(diff[i], Xt[i])
    """
    j = int(target_idx)
    grad_j = np.outer(pre.diff[j], pre.Xt[j])  # (K,d+1)
    sum_minus = pre.sum_grad - grad_j  # sum_{i != j}
    Gm = -float(lam) * float(pre.n) * pre.Wt - sum_minus
    return Gm


def add_noise_softmax(
    W: np.ndarray, b: np.ndarray, *, sigma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    Wn = W + rng.normal(0.0, float(sigma), size=W.shape)
    bn = b + rng.normal(0.0, float(sigma), size=b.shape)
    return Wn, bn


@dataclass
class SoftmaxDPCalibration:
    sigma: float
    delta2: float
    Lz: float
    B: float
    r: float


def calibrate_softmax_output_perturbation(
    *,
    epsilon: float,
    delta: float,
    lam: float,
    n: int,
    B: float,
    r: float,
    include_bias: bool = True,
) -> SoftmaxDPCalibration:
    # Lz for softmax head
    Lz = float(
        lz_softmax_linear_bound(
            B=float(B), lam=float(lam), include_bias=bool(include_bias)
        )
    )

    # ERM sensitivity: Δ = (Lz/(lam*n)) * r
    delta2 = (Lz / (float(lam) * float(n))) * float(r)

    sigma = float(
        calibrate_analytic_gaussian(
            epsilon=float(epsilon), delta=float(delta), GS=float(delta2)
        )
    )

    return SoftmaxDPCalibration(
        sigma=sigma, delta2=float(delta2), Lz=Lz, B=float(B), r=float(r)
    )


# -------------------------------
# Candidate sets for shadow attacks
# -------------------------------


def make_candidates_including_target(
    *,
    pool_X: np.ndarray,
    pool_y: np.ndarray,
    target_idx: int,
    radius: float,
    m: int,
    rng: np.random.Generator,
    label_preserving: bool = True,
) -> List[Candidate]:
    pool_X = np.asarray(pool_X, dtype=np.float64)
    pool_y = np.asarray(pool_y, dtype=np.int64).reshape(-1)
    j = int(target_idx)
    yj = int(pool_y[j])
    xj = pool_X[j].copy()

    # always include the true target as candidate 0
    cands: List[Candidate] = [
        Candidate(record=xj, label=yj, meta={"pool_index": j, "id": "true"})
    ]

    prior = PoolBallPrior(
        pool_X=pool_X,
        pool_y=pool_y,
        radius=float(radius),
        metric_batch=l2_metric_batch,
        label_fixed=yj if label_preserving else None,
    )

    idx = prior.candidate_indices(center=xj)
    # exclude the true one
    idx = idx[idx != j]
    if idx.size == 0:
        return cands

    k = min(int(m) - 1, idx.size)
    pick = rng.choice(idx, size=k, replace=False)
    for p in pick:
        p = int(p)
        cands.append(
            Candidate(
                record=pool_X[p].copy(), label=int(pool_y[p]), meta={"pool_index": p}
            )
        )
    return cands


# -------------------------------
# Pixel metrics for MNIST images
# -------------------------------


def mnist_img_mse_psnr(
    xhat_1x28x28: np.ndarray, xtrue_1x28x28: np.ndarray
) -> tuple[float, float]:
    a = np.asarray(xhat_1x28x28, dtype=np.float64).reshape(1, 28, 28)
    b = np.asarray(xtrue_1x28x28, dtype=np.float64).reshape(1, 28, 28)
    m = mse(a, b)
    p = psnr_from_mse(m, max_val=1.0)
    return m, p
