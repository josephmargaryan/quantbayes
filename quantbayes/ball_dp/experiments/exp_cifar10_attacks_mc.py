# quantbayes/ball_dp/experiments/exp_cifar10_attacks_mc.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from quantbayes.ball_dp.experiments.cifar10_embed_cache import (
    CIFAR10EmbedConfig,
    get_or_compute_cifar10_embeddings,
)
from quantbayes.ball_dp.heads.prototypes import (
    fit_ridge_prototypes,
    prototypes_sensitivity_l2,
)
from quantbayes.ball_dp.radius import within_class_nn_distances
from quantbayes.ball_dp.mechanisms import gaussian_sigma
from quantbayes.ball_dp.attacks.audit import run_llr_audit_trials
from quantbayes.ball_dp.utils import (
    ensure_dir,
    write_csv_rows,
    save_line_plot,
    set_global_seed,
)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, default="./runs/cifar10_prototypes_ball_dp")
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
        "--sigma_method", type=str, default="classic", choices=["classic", "analytic"]
    )

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--nn_sample_per_class", type=int, default=400)
    ap.add_argument(
        "--r_percentile", type=float, default=50.0
    )  # pick one radius for audits

    ap.add_argument("--eps_list", type=str, default="0.05,0.1,0.2,0.5,1,2,5,10")
    ap.add_argument("--n_trials", type=int, default=2000)

    # choose a subset size per class for the audit (faster, cleaner control)
    ap.add_argument("--n_per_class", type=int, default=200)

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

    # small balanced subset for auditing
    rng = np.random.default_rng(args.seed)
    idx_all = []
    for c in range(10):
        idx = np.where(ytr == c)[0]
        sub = rng.choice(idx, size=int(args.n_per_class), replace=False)
        idx_all.append(sub)
    idx_all = np.concatenate(idx_all, axis=0)
    rng.shuffle(idx_all)
    Z = Ztr[idx_all]
    y = ytr[idx_all]

    # radius selection from within-class NN distances computed on the (full) training set
    nn_dists = within_class_nn_distances(
        Ztr,
        ytr,
        num_classes=10,
        per_class=int(args.nn_sample_per_class),
        seed=int(args.seed),
    )
    r_val = float(np.percentile(nn_dists, float(args.r_percentile)))

    # Fit prototypes (deterministic query f(D))
    mus, counts = fit_ridge_prototypes(Z, y, num_classes=10, lam=float(args.lam))
    n_min = int(counts.min())

    # Deterministic query (flattened prototypes)
    def f(dataset: np.ndarray) -> np.ndarray:
        # Kept for generality; in this prototype audit, we use closed-over mus.
        return mus.reshape(-1).astype(np.float32)

    # Ball-neighbor constructor: swap one within-class record with another within radius r_val.
    # For prototypes, we can compute mu1 from mu0 by a rank-one update.
    def make_neighbor(_D: np.ndarray, rng_local: np.random.Generator):
        c = int(rng_local.integers(0, 10))
        idx = np.where(y == c)[0]
        if idx.size < 2:
            c = int(np.argmax(counts))
            idx = np.where(y == c)[0]

        # find close pair (rejection sampling)
        for _ in range(200):
            i1, i2 = rng_local.choice(idx, size=2, replace=False)
            z1 = Z[i1]
            z2 = Z[i2]
            if float(np.linalg.norm(z1 - z2)) <= r_val:
                break
        else:
            # fallback: closest pair in a small subset
            sub = rng_local.choice(idx, size=min(50, idx.size), replace=False)
            A = Z[sub]
            AA = (A * A).sum(axis=1, keepdims=True)
            D2 = AA + AA.T - 2.0 * (A @ A.T)
            np.fill_diagonal(D2, np.inf)
            ii = int(np.argmin(D2))
            i = ii // D2.shape[1]
            j = ii % D2.shape[1]
            z1, z2 = A[i], A[j]
            i1, i2 = sub[i], sub[j]

        bit = int(rng_local.integers(0, 2))  # 0 => use D, 1 => use D'

        mu0 = mus.copy()
        mu1 = mus.copy()
        denom = 2.0 * float(counts[c]) + float(args.lam) * float(Z.shape[0])
        mu1[c] = mu0[c] + (2.0 * (z2 - z1) / denom).astype(np.float32)

        return (
            _D,
            bit,
            mu0.reshape(-1).astype(np.float32),
            mu1.reshape(-1).astype(np.float32),
        )

    eps_list = [float(s.strip()) for s in args.eps_list.split(",") if s.strip()]
    rows: List[Dict[str, object]] = []

    attack_accs = []

    for eps in eps_list:
        Delta = prototypes_sensitivity_l2(
            r=r_val,
            n_min=n_min,
            n_total=int(Z.shape[0]),
            lam=float(args.lam),
        )
        sigma = gaussian_sigma(
            Delta, float(eps), float(args.delta), method=str(args.sigma_method)
        )

        res = run_llr_audit_trials(
            f=f,
            make_neighbor=make_neighbor,
            D=Z,
            sigma=float(sigma),
            eps=float(eps),
            n_trials=int(args.n_trials),
            seed=int(args.seed) + 1234,
        )

        rows.append(
            {
                "head": "prototypes",
                "r_percentile": float(args.r_percentile),
                "r_value": float(r_val),
                "lam": float(args.lam),
                "delta_target": float(args.delta),
                "sigma_method": str(args.sigma_method),
                "eps": float(eps),
                "Delta": float(Delta),
                "sigma": float(sigma),
                "attack_acc": float(res.attack_acc),
                "llr_mean_under_D": float(res.llr_mean_under_D),
                "llr_std_under_D": float(res.llr_std_under_D),
                "llr_mean_under_Dprime": float(res.llr_mean_under_Dprime),
                "llr_std_under_Dprime": float(res.llr_std_under_Dprime),
                "p_L_gt_eps_under_D": float(res.p_L_gt_eps_under_D),
                "p_L_gt_eps_under_Dprime": float(res.p_L_gt_eps_under_Dprime),
                "delta_hat_D_to_Dprime": float(res.delta_hat_D_to_Dprime),
                "p_L_lt_minus_eps_under_D": float(res.p_L_lt_minus_eps_under_D),
                "p_L_lt_minus_eps_under_Dprime": float(
                    res.p_L_lt_minus_eps_under_Dprime
                ),
                "delta_hat_Dprime_to_D": float(res.delta_hat_Dprime_to_D),
                "n_trials": int(res.n_trials),
                "n_D": int(res.n_D),
                "n_Dprime": int(res.n_Dprime),
            }
        )
        attack_accs.append(float(res.attack_acc))

    write_csv_rows(
        out_dir / "audit_results_mc.csv", rows, fieldnames=list(rows[0].keys())
    )

    # Plot attack accuracy vs eps (paper figure)
    save_line_plot(
        np.asarray(eps_list, dtype=np.float64),
        np.asarray(attack_accs, dtype=np.float64),
        title=f"LLR Audit Attack Accuracy vs epsilon (Ball r=p{int(args.r_percentile)})",
        xlabel="epsilon",
        ylabel="attack accuracy",
        out_path=fig_dir / "attack_audit_vs_eps.png",
    )


if __name__ == "__main__":
    main()
