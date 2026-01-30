# quantbayes/ball_dp/api.py

"""
1) I have my own dataset + encoder (Torch)
from torch.utils.data import DataLoader
from quantbayes.ball_dp.api import embed_torch_dataloaders, compute_radius_policy

bundle = embed_torch_dataloaders(
    train_loader=DataLoader(train_ds, batch_size=256, shuffle=False),
    test_loader=DataLoader(test_ds, batch_size=256, shuffle=False),
    encoder=my_encoder,        # any torch.nn.Module
    device="cuda",
    l2_normalize=True,
)

policy = compute_radius_policy(bundle.Ztr, bundle.ytr, percentiles=(10,25,50,75,90))
print(policy.r_values, policy.r_std)

2) I have embeddings already
from quantbayes.ball_dp.api import compute_radius_policy
policy = compute_radius_policy(Ztr, ytr)

3) I trained any strongly-convex ERM head and want a private release
from quantbayes.ball_dp.api import dp_release_erm_params_gaussian

out = dp_release_erm_params_gaussian(
    params=w,            # 1D numpy vector
    lz=Lz,
    r=policy.r_values[50.0],
    lam=1e-2,
    n=len(Ztr),
    eps=1.0,
    delta=1e-5,
    sigma_method="analytic",
)
w_noisy = out["params_noisy"]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np

# Optional torch path (only imported inside functions) to keep base import light.

from quantbayes.ball_dp.metrics import l2_norms
from quantbayes.ball_dp.radius import within_class_nn_distances, radii_from_percentiles
from quantbayes.ball_dp.sensitivity import erm_sensitivity_l2
from quantbayes.ball_dp.mechanisms import add_gaussian_noise, gaussian_sigma


@dataclass(frozen=True)
class EmbeddingBundle:
    """Standard container that makes downstream code dataset/encoder-agnostic."""

    Ztr: np.ndarray
    ytr: np.ndarray
    Zte: np.ndarray
    yte: np.ndarray
    meta: Dict[str, object]


@dataclass(frozen=True)
class RadiusPolicy:
    """Ball radii (percentile-based) + bounded-replacement baseline r_std=2B."""

    r_values: Dict[float, float]  # percentile -> radius
    percentiles: Tuple[float, ...]
    nn_sample_per_class: int
    B_quantile: float
    B: float
    r_std: float
    nn_dists_summary: Dict[str, float]


def _l2_normalize_rows(Z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float32)
    n = np.linalg.norm(Z, axis=1, keepdims=True) + float(eps)
    return Z / n


def embed_torch_dataloaders(
    *,
    train_loader,
    test_loader,
    encoder,
    device: str = "cuda",
    l2_normalize: bool = False,
    encode_fn: Optional[Callable] = None,
) -> EmbeddingBundle:
    """
    Compute embeddings for arbitrary Torch datasets/loaders and encoder.

    train_loader/test_loader must yield (x, y).
    encoder is typically a torch.nn.Module. If your encoder needs special calling
    logic, pass encode_fn(x)->embedding_tensor.

    Returns EmbeddingBundle with Ztr,ytr,Zte,yte as numpy arrays.
    """
    import torch

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(dev).eval()

    if encode_fn is None:

        def encode_fn(x):
            return encoder(x)

    @torch.no_grad()
    def _embed(loader) -> Tuple[np.ndarray, np.ndarray]:
        Zs, Ys = [], []
        for x, y in loader:
            x = x.to(dev)
            z = encode_fn(x)
            if isinstance(z, (tuple, list)):
                z = z[0]
            z = z.detach().float().cpu().numpy().astype(np.float32)
            y = y.detach().cpu().numpy().astype(np.int64)

            if l2_normalize:
                z = _l2_normalize_rows(z)

            Zs.append(z)
            Ys.append(y)

        Z = np.concatenate(Zs, axis=0)
        Y = np.concatenate(Ys, axis=0)
        return Z, Y

    Ztr, ytr = _embed(train_loader)
    Zte, yte = _embed(test_loader)

    meta = {
        "l2_normalize": bool(l2_normalize),
        "device": str(dev),
        "Ztr_shape": tuple(Ztr.shape),
        "Zte_shape": tuple(Zte.shape),
    }
    return EmbeddingBundle(Ztr=Ztr, ytr=ytr, Zte=Zte, yte=yte, meta=meta)


def save_embeddings_npz(bundle: EmbeddingBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        Ztr=bundle.Ztr,
        ytr=bundle.ytr,
        Zte=bundle.Zte,
        yte=bundle.yte,
    )


def load_embeddings_npz(path: str | Path) -> EmbeddingBundle:
    path = Path(path)
    d = np.load(path)
    return EmbeddingBundle(
        Ztr=d["Ztr"].astype(np.float32),
        ytr=d["ytr"].astype(np.int64),
        Zte=d["Zte"].astype(np.float32),
        yte=d["yte"].astype(np.int64),
        meta={"source": str(path)},
    )


def compute_radius_policy(
    Ztr: np.ndarray,
    ytr: np.ndarray,
    *,
    percentiles: Sequence[float] = (10, 25, 50, 75, 90),
    nn_sample_per_class: int = 400,
    seed: int = 0,
    B_quantile: float = 0.999,
) -> RadiusPolicy:
    """
    Dataset-agnostic policy helper:
      - ball radii from within-class NN distances
      - r_std = 2B from a norm quantile (bounded replacement baseline)
    """
    percentiles = tuple(float(p) for p in percentiles)

    nn_dists = within_class_nn_distances(
        np.asarray(Ztr, dtype=np.float32),
        np.asarray(ytr, dtype=np.int64),
        num_classes=int(np.max(ytr) + 1),
        per_class=int(nn_sample_per_class),
        seed=int(seed),
    )
    r_vals = radii_from_percentiles(nn_dists, list(percentiles))

    norms = l2_norms(np.asarray(Ztr, dtype=np.float32))
    B = float(np.quantile(norms, float(B_quantile)))
    r_std = 2.0 * B

    summary = {
        "nn_min": float(np.min(nn_dists)),
        "nn_med": float(np.median(nn_dists)),
        "nn_p95": float(np.quantile(nn_dists, 0.95)),
        "nn_max": float(np.max(nn_dists)),
        "n_nn": int(nn_dists.size),
    }

    return RadiusPolicy(
        r_values={float(p): float(r_vals[float(p)]) for p in percentiles},
        percentiles=percentiles,
        nn_sample_per_class=int(nn_sample_per_class),
        B_quantile=float(B_quantile),
        B=float(B),
        r_std=float(r_std),
        nn_dists_summary=summary,
    )


def dp_release_erm_params_gaussian(
    params: np.ndarray,
    *,
    lz: float,
    r: float,
    lam: float,
    n: int,
    eps: float,
    delta: float,
    sigma_method: Literal["classic", "analytic"] = "classic",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """
    Head-agnostic helper: given a trained ERM parameter vector `params`,
    apply Gaussian output perturbation calibrated from the Ball-DP ERM sensitivity.

    Returns a dict with:
      - params_noisy
      - Delta (sensitivity)
      - sigma
      - bookkeeping fields
    """
    params = np.asarray(params, dtype=np.float32).reshape(-1)
    if rng is None:
        rng = np.random.default_rng()

    sens = erm_sensitivity_l2(lz=float(lz), r=float(r), lam=float(lam), n=int(n))
    Delta = float(sens.delta_l2)

    sigma = float(
        gaussian_sigma(Delta, float(eps), float(delta), method=str(sigma_method))
    )
    params_noisy = add_gaussian_noise(params, sigma, rng=rng)

    return {
        "params_noisy": params_noisy.astype(np.float32, copy=False),
        "Delta": Delta,
        "sigma": sigma,
        "eps": float(eps),
        "delta": float(delta),
        "sigma_method": str(sigma_method),
        "lz": float(lz),
        "r": float(r),
        "lam": float(lam),
        "n": int(n),
    }
