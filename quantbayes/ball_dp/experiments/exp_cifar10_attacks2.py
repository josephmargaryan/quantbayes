# quantbayes/ball_dp/experiments/exp_cifar10_attacks2.py
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from quantbayes.ball_dp.experiments.cifar10_embed_cache import (
    CIFAR10EmbedConfig,
    get_or_compute_cifar10_embeddings,
)
from quantbayes.ball_dp.heads.prototypes import prototypes_sensitivity_l2
from quantbayes.ball_dp.radius import within_class_nn_distances
from quantbayes.ball_dp.mechanisms import gaussian_sigma
from quantbayes.ball_dp.utils import (
    ensure_dir,
    write_csv_rows,
    save_line_plot,
    set_global_seed,
)

from quantbayes.ball_dp.attacks.audit import (
    gaussian_expected_llr_attack_acc,
    gaussian_dp_slack_closed_form,
)


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

    # Pair sampling controls
    ap.add_argument(
        "--n_per_class", type=int, default=200
    )  # balanced subset size for neighbor sampling
    ap.add_argument(
        "--n_pairs", type=int, default=2000
    )  # number of neighbor pairs to sample (data-level MC)
    ap.add_argument(
        "--hard_band", type=float, default=0.0
    )  # if >0, enforce ||z-z'|| in [ (1-hard_band)r, r ]

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

    # balanced subset (used for neighbor sampling / counts)
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

    counts = np.array([(y == c).sum() for c in range(10)], dtype=int)
    n_min = int(counts.min())

    # radius from full train set NN distances (as in your other experiments)
    nn_dists = within_class_nn_distances(
        Ztr,
        ytr,
        num_classes=10,
        per_class=int(args.nn_sample_per_class),
        seed=int(args.seed),
    )
    r_val = float(np.percentile(nn_dists, float(args.r_percentile)))

    eps_list = [float(s.strip()) for s in args.eps_list.split(",") if s.strip()]

    # Pre-sample neighbor pairs (z1,z2) within class, optionally near the boundary
    pairs: List[Tuple[int, np.ndarray, np.ndarray]] = []
    hard_band = float(args.hard_band)

    for _ in range(int(args.n_pairs)):
        c = int(rng.integers(0, 10))
        idx = np.where(y == c)[0]
        if idx.size < 2:
            continue

        # rejection sampling
        z1 = z2 = None
        for _tries in range(300):
            i1, i2 = rng.choice(idx, size=2, replace=False)
            a = Z[i1]
            b = Z[i2]
            d = float(np.linalg.norm(a - b))
            if d <= r_val:
                if hard_band > 0.0:
                    if d >= (1.0 - hard_band) * r_val:
                        z1, z2 = a, b
                        break
                else:
                    z1, z2 = a, b
                    break

        if z1 is None:
            # fallback: accept closest pair in a small subset
            sub = rng.choice(idx, size=min(60, idx.size), replace=False)
            A = Z[sub].astype(np.float64, copy=False)
            AA = (A * A).sum(axis=1, keepdims=True)
            D2 = AA + AA.T - 2.0 * (A @ A.T)
            np.fill_diagonal(D2, np.inf)
            ii = int(np.argmin(D2))
            i = ii // D2.shape[1]
            j = ii % D2.shape[1]
            z1 = A[i].astype(np.float32)
            z2 = A[j].astype(np.float32)

        pairs.append((c, z1, z2))

    # Precompute d_mu = ||mu0-mu1|| for each pair using the closed-form prototype update
    # mu shift only in class c block: delta = 2(z2-z1)/(2 n_c + lam)
    dmus = []
    mults = []
    for c, z1, z2 in pairs:
        denom = 2.0 * float(counts[c]) + float(args.lam)
        delta = 2.0 * (z2 - z1) / denom
        dmus.append(float(np.linalg.norm(delta)))
        mults.append(float(np.linalg.norm(z2 - z1) / max(r_val, 1e-12)))

    dmus = np.asarray(dmus, dtype=np.float64)

    rows: List[Dict[str, object]] = []
    acc_curve = []

    # Global sensitivity used for sigma calibration (worst-case over Ball neighbors)
    Delta_global = prototypes_sensitivity_l2(r=r_val, n_min=n_min, lam=float(args.lam))

    for eps in eps_list:
        sigma = gaussian_sigma(
            float(Delta_global),
            float(eps),
            float(args.delta),
            method=str(args.sigma_method),
        )

        # expected attack acc per pair (LLR closed-form)
        accs = [
            gaussian_expected_llr_attack_acc(np.zeros(1), np.array([d]), float(sigma))
            for d in dmus
        ]
        # trick above is silly; better: use formula directly:
        accs = [
            float(0.5 * (1.0 + math.erf((d / (2.0 * sigma)) / math.sqrt(2.0))))
            for d in dmus
        ]

        # worst-case (over sampled pairs) DP slack at this eps (closed-form)
        deltas_dir = []
        deltas_rev = []
        for d in dmus:
            mu0 = np.zeros(1, dtype=np.float64)
            mu1 = np.array(
                [d], dtype=np.float64
            )  # same norm difference as full vector for isotropic case
            dd, rr = gaussian_dp_slack_closed_form(mu0, mu1, float(sigma), float(eps))
            deltas_dir.append(dd)
            deltas_rev.append(rr)

        acc_mean = float(np.mean(accs))
        acc_std = float(np.std(accs))
        delta_dir_max = float(np.max(deltas_dir)) if deltas_dir else 0.0
        delta_rev_max = float(np.max(deltas_rev)) if deltas_rev else 0.0

        rows.append(
            {
                "head": "prototypes",
                "sigma_method": str(args.sigma_method),
                "r_percentile": float(args.r_percentile),
                "r_value": float(r_val),
                "lam": float(args.lam),
                "delta_target": float(args.delta),
                "eps": float(eps),
                "Delta_global": float(Delta_global),
                "sigma": float(sigma),
                "n_pairs": int(len(pairs)),
                "acc_expected_mean": acc_mean,
                "acc_expected_std": acc_std,
                "delta_dir_max_over_pairs": delta_dir_max,
                "delta_rev_max_over_pairs": delta_rev_max,
            }
        )
        acc_curve.append(acc_mean)

    write_csv_rows(
        out_dir / "audit_results_closed_form.csv", rows, fieldnames=list(rows[0].keys())
    )

    # Plot (keep same filename as your paper uses, so no LaTeX change)
    save_line_plot(
        np.asarray(eps_list, dtype=np.float64),
        np.asarray(acc_curve, dtype=np.float64),
        title=f"LLR Expected Attack Accuracy vs epsilon (closed-form; p{int(args.r_percentile)})",
        xlabel="epsilon",
        ylabel="expected attack accuracy",
        out_path=fig_dir / "attack_audit_vs_eps.png",
    )

    # Also save multiplier stats so you can cite hardness of sampled pairs
    np.savez_compressed(
        out_dir / "neighbor_pair_stats.npz", dmu=dmus, mult=np.asarray(mults)
    )


if __name__ == "__main__":
    main()
