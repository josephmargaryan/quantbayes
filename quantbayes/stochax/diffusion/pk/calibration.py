# quantbayes/stochax/diffusion/pk/calibration.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Tuple, Optional

import csv
import numpy as np

import jax.numpy as jnp
import jax.random as jr


def gaussian_pdf(x: np.ndarray, mu: float, tau: float) -> np.ndarray:
    tau = float(max(tau, 1e-12))
    return (1.0 / (np.sqrt(2.0 * np.pi) * tau)) * np.exp(-0.5 * ((x - mu) / tau) ** 2)


def w1_to_gaussian(
    z_samples: np.ndarray, mu: float, tau: float, *, seed: int = 0
) -> float:
    """
    Approximate 1D Wasserstein-1 distance by matching sorted samples to sorted Gaussian draws.
    """
    z = np.sort(np.asarray(z_samples).reshape(-1))
    rng = np.random.default_rng(int(seed))
    t = np.sort(rng.normal(loc=float(mu), scale=float(tau), size=z.shape[0]))
    return float(np.mean(np.abs(z - t)))


def save_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to save.")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


@dataclass(frozen=True)
class PKCalibrationGrids:
    w_gamma_grid: list[float]
    guide_strength_grid: list[float]
    guide_sigma_max_grid: list[float]


@dataclass(frozen=True)
class PKCalibrationWeights:
    alpha_clip: float = 0.25  # penalize frequent clipping
    beta_w1: float = 0.10  # tie-breaker shape matching
    eps: float = 1e-12


def calibrate_pk_gaussian_1d(
    sampler_pk,  # jitted fn: (key, mean_d, std_d, mu_z, tau_z, guide_strength, guide_sigma_max, max_guide_norm, w_gamma) -> (x, clip_avg)
    coarse_value_fn: Callable[[jnp.ndarray], jnp.ndarray],  # x -> d (B,)
    *,
    mean_d: float,
    std_d: float,
    mu_z: float,
    tau_z: float,
    max_guide_norm: float,
    grids: PKCalibrationGrids,
    weights: PKCalibrationWeights = PKCalibrationWeights(),
    seed: int = 0,
    key_base: int = 9000,
    w1_seed_base: int = 100000,
) -> Tuple[dict, list[dict]]:
    """
    Calibrate on PK samples: grid search over (gamma, guide_strength, guide_sigma_max).

    Loss = mean_err^2 + std_err^2 + alpha_clip*clip_rate + beta_w1*(W1/tau)
    """
    mean_d_a = jnp.asarray(mean_d, dtype=jnp.float32)
    std_d_a = jnp.asarray(std_d, dtype=jnp.float32)
    mu_z_a = jnp.asarray(mu_z, dtype=jnp.float32)
    tau_z_a = jnp.asarray(tau_z, dtype=jnp.float32)
    max_norm_a = jnp.asarray(max_guide_norm, dtype=jnp.float32)

    rows: list[dict] = []
    best: Optional[dict] = None
    idx = 0

    base_key = jr.PRNGKey(int(seed + key_base))

    for w_gamma in grids.w_gamma_grid:
        for gs in grids.guide_strength_grid:
            for smax in grids.guide_sigma_max_grid:
                idx += 1
                k = jr.fold_in(base_key, idx)

                x_pk, clip_avg = sampler_pk(
                    k,
                    mean_d_a,
                    std_d_a,
                    mu_z_a,
                    tau_z_a,
                    jnp.asarray(gs, dtype=jnp.float32),
                    jnp.asarray(smax, dtype=jnp.float32),
                    max_norm_a,
                    jnp.asarray(w_gamma, dtype=jnp.float32),
                )

                d = coarse_value_fn(x_pk)  # (B,)
                z = (np.asarray(d) - float(mean_d)) / float(std_d)

                mean_z = float(np.mean(z))
                std_z = float(np.std(z) + weights.eps)
                w1 = float(
                    w1_to_gaussian(
                        z, mu=mu_z, tau=tau_z, seed=seed + w1_seed_base + idx
                    )
                )
                clip_f = float(np.asarray(clip_avg))

                denom = float(tau_z + weights.eps)
                mean_err = (mean_z - float(mu_z)) / denom
                std_err = (std_z - float(tau_z)) / denom

                loss = float(
                    mean_err**2
                    + std_err**2
                    + weights.alpha_clip * clip_f
                    + weights.beta_w1 * (w1 / denom)
                )

                rec = dict(
                    w_gamma=float(w_gamma),
                    guide_strength=float(gs),
                    guide_sigma_max=float(smax),
                    mean_z=mean_z,
                    std_z=std_z,
                    w1=w1,
                    clip_rate=clip_f,
                    calib_loss=loss,
                )
                rows.append(rec)

                if (best is None) or (loss < best["calib_loss"]):
                    best = rec

    assert best is not None
    return best, rows
