# quantbayes/ball_dp/experiments/exp_cifar10_attacks.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from quantbayes.ball_dp.experiments.cifar10_embed_cache import (
    CIFAR10EmbedConfig,
    get_or_compute_cifar10_embeddings,
)
from quantbayes.ball_dp.radius import within_class_nn_distances
from quantbayes.ball_dp.heads.prototypes import prototypes_sensitivity_l2
from quantbayes.ball_dp.mechanisms import gaussian_sigma
from quantbayes.ball_dp.attacks.audit import (
    gaussian_expected_llr_attack_acc,
    gaussian_dp_slack_closed_form,
)
from quantbayes.ball_dp.utils import (
    ensure_dir,
    write_csv_rows,
    save_line_plot,
    set_global_seed,
)


def build_pairs_in_distance_range(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    num_classes: int,
    t_lo: float,
    t_hi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a pool of within-class pairs (i,j) with ||Z[i]-Z[j]|| in [t_lo, t_hi].

    Returns:
      pair_i: (M,)
      pair_j: (M,)
      pair_class: (M,)
    """
    t_lo = float(t_lo)
    t_hi = float(t_hi)
    if t_lo < 0 or t_hi < 0 or t_hi < t_lo:
        raise ValueError("Require 0 <= t_lo <= t_hi.")

    pair_i: List[int] = []
    pair_j: List[int] = []
    pair_c: List[int] = []

    for c in range(int(num_classes)):
        idx = np.where(y == c)[0]
        if idx.size < 2:
            continue

        A = Z[idx].astype(np.float64, copy=False)
        AA = (A * A).sum(axis=1, keepdims=True)
        D2 = AA + AA.T - 2.0 * (A @ A.T)
        np.fill_diagonal(D2, np.inf)
        D = np.sqrt(np.maximum(D2, 0.0))

        mask = (D <= t_hi) & (D >= t_lo)
        ii, jj = np.where(np.triu(mask, k=1))

        if ii.size == 0:
            continue

        pair_i.extend(idx[ii].tolist())
        pair_j.extend(idx[jj].tolist())
        pair_c.extend([c] * int(ii.size))

    if len(pair_i) == 0:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
        )

    return (
        np.asarray(pair_i, dtype=int),
        np.asarray(pair_j, dtype=int),
        np.asarray(pair_c, dtype=int),
    )


def _ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    x.sort()
    if x.size == 0:
        return np.asarray([]), np.asarray([])
    y = np.linspace(0.0, 1.0, num=x.size, endpoint=True)
    return x, y


def binned_curve(
    x: np.ndarray, y: np.ndarray, *, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple binning: returns bin centers and mean(y) per bin (NaNs dropped).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    idx = np.digitize(x, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = np.full_like(centers, np.nan, dtype=np.float64)
    for b in range(len(centers)):
        m = y[idx == b]
        if m.size:
            means[b] = float(np.mean(m))
    ok = np.isfinite(means)
    return centers[ok], means[ok]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, default="./runs/cifar10_attacks_ball_dp")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument(
        "--cache_npz", type=str, default="./cache/cifar10_resnet18_embeds.npz"
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--weights", type=str, default="DEFAULT")
    ap.add_argument("--l2_normalize", action="store_true")

    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument(
        "--sigma_method", type=str, default="analytic", choices=["classic", "analytic"]
    )

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nn_sample_per_class", type=int, default=400)
    ap.add_argument("--r_percentile", type=float, default=50.0)
    ap.add_argument("--eps_list", type=str, default="0.1,0.5,1,5")

    # audit construction
    ap.add_argument("--n_per_class", type=int, default=200)
    ap.add_argument("--n_pairs", type=int, default=5000)

    # If band>0, only pairs with ||z-z'|| in [(1-band)r, r] are used (hard neighbors).
    ap.add_argument("--band", type=float, default=0.10)

    # NEW: allow sampling beyond r for profile validation plots.
    ap.add_argument(
        "--max_mult",
        type=float,
        default=1.0,
        help="Build an extended pair pool up to (max_mult * r). Use >1 to include out-of-radius pairs.",
    )

    # Optional stress-test figure: accuracy vs multiplier t/r for a chosen eps
    ap.add_argument("--make_multiplier_plot", action="store_true")
    ap.add_argument("--mult_eps", type=float, default=1.0)

    args = ap.parse_args()
    set_global_seed(args.seed)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    # embeddings
    emb_cfg = CIFAR10EmbedConfig(
        data_dir=args.data_dir,
        cache_npz=args.cache_npz,
        batch_size=args.batch_size,
        device=args.device,
        weights=args.weights,
        l2_normalize=bool(args.l2_normalize),
        seed=args.seed,
    )
    Ztr, ytr, _, _ = get_or_compute_cifar10_embeddings(emb_cfg)

    # balanced subset (for audit neighbor sampling)
    rng = np.random.default_rng(args.seed)
    idx_all = []
    for c in range(10):
        idx = np.where(ytr == c)[0]
        sub = rng.choice(idx, size=int(args.n_per_class), replace=False)
        idx_all.append(sub)
    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)

    Z = Ztr[idx_all]
    y = ytr[idx_all]

    num_classes = 10
    counts = np.array([(y == c).sum() for c in range(num_classes)], dtype=int)
    n_min = int(counts.min())
    n_total = int(Z.shape[0])

    # radius computed from full train set NN distances (as in your paper)
    nn_dists = within_class_nn_distances(
        Ztr,
        ytr,
        num_classes=num_classes,
        per_class=int(args.nn_sample_per_class),
        seed=int(args.seed),
    )
    r_val = float(np.percentile(nn_dists, float(args.r_percentile)))

    # Build VALID ball-neighbor pool in-band near r (for the actual DP slack audit)
    lo_in = (1.0 - float(args.band)) * r_val if float(args.band) > 0 else 0.0
    hi_in = r_val
    pair_i, pair_j, pair_c = build_pairs_in_distance_range(
        Z, y, num_classes=num_classes, t_lo=lo_in, t_hi=hi_in
    )
    if pair_i.size == 0:
        # fall back to full in-radius range
        pair_i, pair_j, pair_c = build_pairs_in_distance_range(
            Z, y, num_classes=num_classes, t_lo=0.0, t_hi=r_val
        )
    if pair_i.size == 0:
        raise RuntimeError(
            "No within-class pairs found with ||z-z'|| <= r. "
            "Increase n_per_class or choose a larger r_percentile."
        )

    # Sample in-radius pairs (with replacement)
    M = int(pair_i.size)
    take = int(args.n_pairs)
    samp = rng.integers(0, M, size=take)
    ii = pair_i[samp]
    jj = pair_j[samp]
    cc = pair_c[samp]

    # distances and multipliers
    diffs = Z[ii] - Z[jj]
    t = np.linalg.norm(diffs, axis=1).astype(np.float64)  # ||z-z'||
    mult = t / max(r_val, 1e-12)  # t/r

    # prototype parameter shift magnitude for each sampled pair:
    # ||mu0-mu1|| = 2||z-z'|| / (2 n_c + lam*n_total)
    denom = 2.0 * counts[cc].astype(np.float64) + float(args.lam) * float(n_total)
    dmu = (2.0 * t) / denom  # scalar magnitude

    # global sensitivity bound for calibration (worst class)
    Delta_global = prototypes_sensitivity_l2(
        r=r_val, n_min=n_min, n_total=n_total, lam=float(args.lam)
    )

    # NEW: tightness diagnostic gamma = ||f(D)-f(D')|| / Delta_global
    gamma = dmu / max(float(Delta_global), 1e-18)
    gx, gy = _ecdf(gamma)
    if gx.size:
        save_line_plot(
            gx,
            gy,
            title="Tightness diagnostic: ECDF of gamma = ||Δf|| / Δ_r (valid Ball neighbors)",
            xlabel="gamma",
            ylabel="CDF",
            out_path=fig_dir / "audit_gamma_cdf.png",
        )

    eps_list = [float(s.strip()) for s in args.eps_list.split(",") if s.strip()]

    rows: List[Dict[str, object]] = []
    acc_curve = []

    for eps in eps_list:
        sigma = gaussian_sigma(
            float(Delta_global),
            float(eps),
            float(args.delta),
            method=str(args.sigma_method),
        )

        # expected LLR accuracy per sampled pair
        accs = [
            gaussian_expected_llr_attack_acc(np.zeros(1), np.array([d]), float(sigma))
            for d in dmu
        ]
        acc_mean = float(np.mean(accs))
        acc_std = float(np.std(accs))

        # exact per-pair DP slack; report max over sampled VALID ball-neighbor pairs
        ddir_max = 0.0
        drev_max = 0.0
        for d in dmu:
            dd, rr = gaussian_dp_slack_closed_form(
                np.zeros(1), np.array([d]), float(sigma), float(eps)
            )
            ddir_max = max(ddir_max, dd)
            drev_max = max(drev_max, rr)

        rows.append(
            {
                "head": "prototypes",
                "sigma_method": str(args.sigma_method),
                "r_percentile": float(args.r_percentile),
                "r_value": float(r_val),
                "lam": float(args.lam),
                "n_total": int(n_total),
                "delta_target": float(args.delta),
                "eps": float(eps),
                "Delta": float(Delta_global),
                "sigma": float(sigma),
                "n_pairs": int(take),
                "attack_acc": float(acc_mean),
                "attack_acc_std": float(acc_std),
                "delta_hat_D_to_Dprime": float(ddir_max),
                "delta_hat_Dprime_to_D": float(drev_max),
                "mult_mean": float(np.mean(mult)),
                "mult_q95": float(np.quantile(mult, 0.95)),
                "mult_q99": float(np.quantile(mult, 0.99)),
                "gamma_mean": float(np.mean(gamma)),
                "gamma_q95": float(np.quantile(gamma, 0.95)),
                "gamma_q99": float(np.quantile(gamma, 0.99)),
                "gamma_max": float(np.max(gamma)),
            }
        )
        acc_curve.append(acc_mean)

    write_csv_rows(out_dir / "audit_results.csv", rows, fieldnames=list(rows[0].keys()))

    # Paper figure (same filename your LaTeX already uses)
    save_line_plot(
        np.asarray(eps_list, dtype=np.float64),
        np.asarray(acc_curve, dtype=np.float64),
        title=f"LLR Expected Attack Accuracy vs epsilon (closed-form; p{int(args.r_percentile)})",
        xlabel="epsilon",
        ylabel="expected attack accuracy",
        out_path=fig_dir / "attack_audit_vs_eps.png",
    )

    # Optional stress-test: accuracy vs multiplier t/r for a chosen eps.
    # If max_mult>1, sample pairs up to max_mult*r to include out-of-radius behavior.
    if bool(args.make_multiplier_plot):
        max_mult = float(args.max_mult)
        t_hi_ext = max_mult * r_val
        pair_i2, pair_j2, pair_c2 = build_pairs_in_distance_range(
            Z, y, num_classes=num_classes, t_lo=0.0, t_hi=t_hi_ext
        )
        if pair_i2.size == 0:
            raise RuntimeError("No pairs found for multiplier plot pool.")

        M2 = int(pair_i2.size)
        samp2 = rng.integers(0, M2, size=take)
        ii2 = pair_i2[samp2]
        jj2 = pair_j2[samp2]
        cc2 = pair_c2[samp2]

        t2 = np.linalg.norm(Z[ii2] - Z[jj2], axis=1).astype(np.float64)
        mult2 = t2 / max(r_val, 1e-12)
        denom2 = 2.0 * counts[cc2].astype(np.float64) + float(args.lam) * float(n_total)
        dmu2 = (2.0 * t2) / denom2

        eps0 = float(args.mult_eps)
        sigma0 = gaussian_sigma(
            float(Delta_global), eps0, float(args.delta), method=str(args.sigma_method)
        )
        acc0 = np.asarray(
            [
                gaussian_expected_llr_attack_acc(
                    np.zeros(1), np.array([d]), float(sigma0)
                )
                for d in dmu2
            ],
            dtype=np.float64,
        )
        bins = np.linspace(0.0, max(max_mult, float(np.max(mult2))), 40)
        xs, ys = binned_curve(mult2, acc0, bins=bins)
        save_line_plot(
            xs,
            ys,
            title=f"Expected LLR accuracy vs multiplier t/r (eps={eps0:g}, max_mult={max_mult:g})",
            xlabel="multiplier t/r",
            ylabel="expected attack accuracy",
            out_path=fig_dir / f"attack_acc_vs_multiplier_eps{eps0:g}.png",
        )


if __name__ == "__main__":
    main()
