# quantbayes/ball_dp/api.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple

import numpy as np

from quantbayes.ball_dp.metrics import clip_l2_rows, l2_norms, maybe_l2_normalize
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
    """
    Ball radii (percentile-based) + bounded-replacement baseline r_std=2B.

    IMPORTANT:
      The baseline r_std=2B is DP-valid ONLY if B is a PUBLIC bound that is ENFORCED
      by preprocessing (e.g., L2-normalization or L2-clipping).

    This class stores bookkeeping so you can report the baseline correctly.
    """

    r_values: Dict[float, float]  # percentile -> radius
    percentiles: Tuple[float, ...]
    nn_sample_per_class: int

    # B/baseline definition
    B_mode: str  # "public" | "max" | "quantile"
    B_public: Optional[float]
    B_quantile: float
    B: float
    r_std: float

    # diagnostics
    max_norm: float
    bound_tol: float
    bound_check_passed: bool

    nn_dists_summary: Dict[str, float]


def embed_torch_dataloaders(
    *,
    train_loader,
    test_loader,
    encoder,
    device: str = "cuda",
    l2_normalize: bool = False,
    l2_clip_norm: Optional[float] = None,
    encode_fn: Optional[Callable] = None,
) -> EmbeddingBundle:
    """
    Compute embeddings for arbitrary Torch datasets/loaders and encoder.

    train_loader/test_loader must yield (x, y).
    encoder is typically a torch.nn.Module. If your encoder needs special calling
    logic, pass encode_fn(x)->embedding_tensor.

    DP note:
      - If you want a DP-valid bounded replacement baseline (r_std = 2B),
        you must enforce a public bound B via L2-normalization or L2-clipping.
      - Use l2_normalize=True (=> B=1) OR l2_clip_norm=B (=> enforce ||z||<=B).

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

            # Enforce record definition (normalization / clipping) BEFORE DP logic.
            z = maybe_l2_normalize(z, bool(l2_normalize))
            if l2_clip_norm is not None:
                z = clip_l2_rows(z, float(l2_clip_norm))

            Zs.append(z)
            Ys.append(y)

        Z = np.concatenate(Zs, axis=0)
        Y = np.concatenate(Ys, axis=0)
        return Z, Y

    Ztr, ytr = _embed(train_loader)
    Zte, yte = _embed(test_loader)

    meta = {
        "l2_normalize": bool(l2_normalize),
        "l2_clip_norm": None if l2_clip_norm is None else float(l2_clip_norm),
        "device": str(dev),
        "Ztr_shape": tuple(Ztr.shape),
        "Zte_shape": tuple(Zte.shape),
        "Ztr_max_norm": float(np.max(l2_norms(Ztr))),
        "Zte_max_norm": float(np.max(l2_norms(Zte))),
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
        meta=np.array([bundle.meta], dtype=object),
    )


def load_embeddings_npz(path: str | Path) -> EmbeddingBundle:
    path = Path(path)
    d = np.load(path, allow_pickle=True)
    meta = {"source": str(path)}
    if "meta" in d:
        try:
            meta = dict(d["meta"][0])
        except Exception:
            pass
    return EmbeddingBundle(
        Ztr=d["Ztr"].astype(np.float32),
        ytr=d["ytr"].astype(np.int64),
        Zte=d["Zte"].astype(np.float32),
        yte=d["yte"].astype(np.int64),
        meta=meta,
    )


def compute_radius_policy(
    Ztr: np.ndarray,
    ytr: np.ndarray,
    *,
    percentiles: Sequence[float] = (10, 25, 50, 75, 90),
    nn_sample_per_class: int = 400,
    seed: int = 0,
    # Baseline bound mode:
    #   - "public": require B_public and optionally check that max_norm <= B_public*(1+tol)
    #   - "max":    set B = max ||z|| on this dataset (NOT "public", but at least a true bound on THIS data)
    #   - "quantile": set B = quantile ||z|| (NOT a bound; NOT DP-valid baseline unless you also clip)
    B_mode: Literal["public", "max", "quantile"] = "public",
    B_public: Optional[float] = None,
    B_quantile: float = 0.999,
    check_public_bound: bool = True,
    bound_tol: float = 1e-3,
) -> RadiusPolicy:
    """
    Dataset-agnostic policy helper:
      - ball radii from within-class NN distances
      - baseline r_std = 2B

    IMPORTANT (DP correctness):
      If you plan to report a standard bounded-replacement baseline r_std=2B as a DP baseline,
      use B_mode="public" and provide B_public, and ensure embeddings satisfy ||z|| <= B_public
      via L2-normalization or L2-clipping.

    If you use B_mode="quantile", then B is NOT a public bound and the resulting "baseline"
    is NOT DP-correct unless you also enforce clipping to that B as part of the record definition.
    """
    percentiles = tuple(float(p) for p in percentiles)

    Ztr = np.asarray(Ztr, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)

    # Ball radii from within-class NN distances (replacement model / audit proxy)
    nn_dists = within_class_nn_distances(
        Ztr,
        ytr,
        num_classes=int(np.max(ytr) + 1),
        per_class=int(nn_sample_per_class),
        seed=int(seed),
    )
    r_vals = radii_from_percentiles(nn_dists, list(percentiles))

    # Baseline bound B
    norms = l2_norms(Ztr)
    max_norm = float(np.max(norms))

    bound_ok = True
    if B_mode == "public":
        if B_public is None or float(B_public) <= 0:
            raise ValueError("B_mode='public' requires a positive B_public.")
        B = float(B_public)
        if check_public_bound:
            bound_ok = max_norm <= B * (1.0 + float(bound_tol))
            if not bound_ok:
                raise ValueError(
                    f"Public bound check failed: max ||z|| = {max_norm:.6f} exceeds "
                    f"B_public*(1+tol) = {B*(1.0+float(bound_tol)):.6f}. "
                    "Fix by using l2_normalize=True or l2_clip_norm=B_public in embedding extraction."
                )
    elif B_mode == "max":
        # Not public, but is a true bound on THIS Ztr.
        B = float(max_norm)
        bound_ok = False
    elif B_mode == "quantile":
        # Not a bound, hence not DP-correct as bounded replacement baseline unless enforced by clipping.
        B = float(np.quantile(norms, float(B_quantile)))
        bound_ok = False
    else:
        raise ValueError("Unknown B_mode. Use 'public', 'max', or 'quantile'.")

    r_std = 2.0 * float(B)

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
        B_mode=str(B_mode),
        B_public=None if B_public is None else float(B_public),
        B_quantile=float(B_quantile),
        B=float(B),
        r_std=float(r_std),
        max_norm=float(max_norm),
        bound_tol=float(bound_tol),
        bound_check_passed=bool(bound_ok) if B_mode == "public" else False,
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

    THEORY NOTE:
      This helper assumes `params` corresponds to the deterministic ERM minimizer
      \hat{theta}(D) (or a sufficiently accurate solution). The DP guarantee is
      for the mechanism M(D) = \hat{theta}(D) + N(0, sigma^2 I).

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


def dp_release_ridge_prototypes_gaussian(
    mus: np.ndarray,
    counts: np.ndarray,
    *,
    r: float,
    lam: float,
    eps: float,
    delta: float,
    sigma_method: Literal["classic", "analytic"] = "classic",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """
    DP release helper for the CLOSED-FORM ridge prototypes head in heads/prototypes.py.

    Uses the EXACT sensitivity bound from prototypes_sensitivity_l2(), not the generic ERM bound.
    """
    from quantbayes.ball_dp.heads.prototypes import prototypes_sensitivity_l2

    mus = np.asarray(mus, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)

    if mus.ndim != 2:
        raise ValueError("mus must have shape (K,d)")
    if counts.ndim != 1 or counts.size != mus.shape[0]:
        raise ValueError("counts must have shape (K,) matching mus.shape[0].")

    if rng is None:
        rng = np.random.default_rng()

    n_total = int(np.sum(counts))
    if n_total <= 0:
        raise ValueError("counts sum must be >= 1")

    pos = counts[counts > 0]
    if pos.size == 0:
        raise ValueError("all classes have zero count")
    n_min = int(np.min(pos))

    Delta = float(
        prototypes_sensitivity_l2(
            r=float(r), n_min=int(n_min), n_total=int(n_total), lam=float(lam)
        )
    )
    sigma = float(
        gaussian_sigma(float(Delta), float(eps), float(delta), method=str(sigma_method))
    )

    mus_noisy = add_gaussian_noise(mus.reshape(-1), sigma, rng=rng).reshape(mus.shape)

    return {
        "mus_noisy": mus_noisy.astype(np.float32, copy=False),
        "Delta": float(Delta),
        "sigma": float(sigma),
        "eps": float(eps),
        "delta": float(delta),
        "sigma_method": str(sigma_method),
        "r": float(r),
        "lam": float(lam),
        "n_total": int(n_total),
        "n_min": int(n_min),
    }
