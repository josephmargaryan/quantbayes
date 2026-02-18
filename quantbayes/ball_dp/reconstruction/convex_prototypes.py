# quantbayes/ball_dp/reconstruction/convex_prototypes.py
from __future__ import annotations

from typing import Tuple

import numpy as np

from quantbayes.ball_dp.heads.prototypes import fit_ridge_prototypes
from quantbayes.ball_dp.api import dp_release_ridge_prototypes_gaussian


def class_sums_counts_np(
    Z: np.ndarray, y: np.ndarray, *, num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    Z = np.asarray(Z, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    K = int(num_classes)
    sums = np.zeros((K, Z.shape[1]), dtype=np.float64)
    counts = np.zeros((K,), dtype=np.int64)
    for c in range(K):
        idx = np.where(y == c)[0]
        counts[c] = int(idx.size)
        if idx.size:
            sums[c] = Z[idx].sum(axis=0, dtype=np.float64)
    return sums.astype(np.float32), counts


def reconstruct_missing_from_prototypes_given_label(
    *,
    mu_y_release: np.ndarray,
    sum_y_minus: np.ndarray,
    n_y_minus: int,
    n_total_full: int,
    lam: float,
) -> np.ndarray:
    mu_y = np.asarray(mu_y_release, dtype=np.float64).reshape(-1)
    sum_y_minus = np.asarray(sum_y_minus, dtype=np.float64).reshape(-1)
    lam = float(lam)
    den = 2.0 * float(int(n_y_minus) + 1) + lam * float(int(n_total_full))
    z_hat = (den / 2.0) * mu_y - sum_y_minus
    return z_hat.astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, d = 3, 5
    n_per = 20
    Z = rng.normal(size=(K * n_per, d)).astype(np.float32)
    y = np.repeat(np.arange(K), n_per).astype(np.int64)

    fixed = np.arange(K * n_per - 1)
    target = K * n_per - 1
    Zm, ym = Z[fixed], y[fixed]
    z, yc = Z[target], int(y[target])

    lam = 0.1
    Zfull = np.concatenate([Zm, z[None, :]], axis=0)
    yfull = np.concatenate([ym, np.array([yc], dtype=np.int64)], axis=0)
    mus, counts = fit_ridge_prototypes(Zfull, yfull, num_classes=K, lam=lam)

    sums_minus, counts_minus = class_sums_counts_np(Zm, ym, num_classes=K)

    # noiseless recon
    z_hat = reconstruct_missing_from_prototypes_given_label(
        mu_y_release=mus[yc],
        sum_y_minus=sums_minus[yc],
        n_y_minus=int(counts_minus[yc]),
        n_total_full=int(Zm.shape[0] + 1),
        lam=lam,
    )
    err = float(np.linalg.norm(z_hat - z))
    print("prototype recon err (noiseless):", err)
    assert err < 1e-5

    # DP recon quick check
    eps, delta = 1.0, 1e-5
    r = 1.0
    dp = dp_release_ridge_prototypes_gaussian(
        mus=mus,
        counts=counts,
        r=r,
        lam=lam,
        eps=eps,
        delta=delta,
        sigma_method="analytic",
        rng=np.random.default_rng(123),
    )
    mus_noisy = dp["mus_noisy"]
    z_hat_dp = reconstruct_missing_from_prototypes_given_label(
        mu_y_release=mus_noisy[yc],
        sum_y_minus=sums_minus[yc],
        n_y_minus=int(counts_minus[yc]),
        n_total_full=int(Zm.shape[0] + 1),
        lam=lam,
    )
    err_dp = float(np.linalg.norm(z_hat_dp - z))
    print(f"prototype recon err (DP): {err_dp:.4f} | sigma={dp['sigma']:.6g}")
    print("[OK] prototypes tests done.")
