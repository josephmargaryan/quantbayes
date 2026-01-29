# quantbayes/ball_dp/experiments/exp_cifar10_prototypes.py
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
from quantbayes.ball_dp.heads.prototypes import (
    fit_ridge_prototypes,
    predict_nearest_prototype,
    prototypes_sensitivity_l2,
)
from quantbayes.ball_dp.metrics import l2_norms
from quantbayes.ball_dp.radius import within_class_nn_distances, radii_from_percentiles
from quantbayes.ball_dp.mechanisms import gaussian_sigma
from quantbayes.ball_dp.utils import (
    ensure_dir,
    write_json,
    write_csv_rows,
    save_errorbar_plot,
    set_global_seed,
)


def accuracy(yhat: np.ndarray, y: np.ndarray) -> float:
    return float((yhat == y).mean())


def subsample_per_class(
    Z: np.ndarray, y: np.ndarray, n_per_class: int, *, num_classes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_all = []
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        if idx.size < n_per_class:
            raise RuntimeError(
                f"class {c} has only {idx.size} samples; need {n_per_class}"
            )
        sub = rng.choice(idx, size=n_per_class, replace=False)
        idx_all.append(sub)
    idx_all = np.concatenate(idx_all, axis=0)
    rng.shuffle(idx_all)
    return Z[idx_all], y[idx_all]


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
    ap.add_argument("--trials", type=int, default=10)

    ap.add_argument("--nn_sample_per_class", type=int, default=400)
    ap.add_argument("--r_percentiles", type=str, default="10,25,50,75,90")
    ap.add_argument("--B_quantile", type=float, default=0.999)

    ap.add_argument(
        "--eps_list", type=str, default="0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10"
    )
    ap.add_argument("--n_per_class_list", type=str, default="100,2000,5000")

    args = ap.parse_args()
    set_global_seed(args.seed)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    # 1) Embeddings
    emb_cfg = CIFAR10EmbedConfig(
        data_dir=args.data_dir,
        cache_npz=args.cache_npz,
        batch_size=args.batch_size,
        device=args.device,
        weights=args.weights,
        l2_normalize=bool(args.l2_normalize),
        seed=args.seed,
    )
    Ztr, ytr, Zte, yte = get_or_compute_cifar10_embeddings(emb_cfg)

    # 2) Radii: ball adjacency from within-class NN distances
    nn_dists = within_class_nn_distances(
        Ztr,
        ytr,
        num_classes=10,
        per_class=int(args.nn_sample_per_class),
        seed=int(args.seed),
    )
    r_percentiles = [
        float(s.strip()) for s in args.r_percentiles.split(",") if s.strip()
    ]
    r_vals = radii_from_percentiles(nn_dists, r_percentiles)

    # 3) Standard bounded replacement baseline r_std = 2B (B from norms quantile)
    norms = l2_norms(Ztr)
    B = float(np.quantile(norms, float(args.B_quantile)))
    r_std = 2.0 * B

    write_json(
        out_dir / "radii.json",
        {
            "l2_normalize": bool(args.l2_normalize),
            "nn_sample_per_class": int(args.nn_sample_per_class),
            "r_percentiles": r_percentiles,
            "r_values": r_vals,
            "B_quantile": float(args.B_quantile),
            "B": float(B),
            "r_std": float(r_std),
        },
    )

    eps_list = [float(s.strip()) for s in args.eps_list.split(",") if s.strip()]
    n_list = [int(s.strip()) for s in args.n_per_class_list.split(",") if s.strip()]

    results_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(args.seed)

    # Optional: build a single combined plot for paper (aggregate over n)
    # We'll collect a representative n (the first entry) and plot that as the paper figure.
    paper_n = n_list[0]
    paper_curves = None

    for n_per_class in n_list:
        Zsub, ysub = subsample_per_class(
            Ztr, ytr, n_per_class=n_per_class, num_classes=10, seed=args.seed
        )

        mus, counts = fit_ridge_prototypes(
            Zsub, ysub, num_classes=10, lam=float(args.lam)
        )
        n_min = int(counts.min())
        base_acc = accuracy(predict_nearest_prototype(Zte, mus), yte)

        curves: Dict[str, Tuple[List[float], List[float]]] = {}

        def eval_for_radius(
            r_kind: str, r_val: float
        ) -> Tuple[List[float], List[float]]:
            means, stds = [], []
            Delta = prototypes_sensitivity_l2(
                r=float(r_val), n_min=n_min, lam=float(args.lam)
            )

            for eps in eps_list:
                sig = gaussian_sigma(
                    Delta, eps, float(args.delta), method=args.sigma_method
                )
                acc_trials = []
                for _ in range(int(args.trials)):
                    noise = rng.normal(0.0, sig, size=mus.shape).astype(np.float32)
                    mus_noisy = mus + noise
                    yhat = predict_nearest_prototype(Zte, mus_noisy)
                    acc_trials.append(accuracy(yhat, yte))

                m = float(np.mean(acc_trials))
                s = float(np.std(acc_trials))
                means.append(m)
                stds.append(s)

                results_rows.append(
                    {
                        "head": "prototypes",
                        "n_per_class": int(n_per_class),
                        "n_min": int(n_min),
                        "lam": float(args.lam),
                        "delta": float(args.delta),
                        "sigma_method": str(args.sigma_method),
                        "r_kind": str(r_kind),
                        "r_value": float(r_val),
                        "eps": float(eps),
                        "Delta": float(Delta),
                        "sigma": float(sig),
                        "acc_mean": float(m),
                        "acc_std": float(s),
                        "acc_non_private": float(base_acc),
                        "l2_normalize": int(bool(args.l2_normalize)),
                        "B_quantile": float(args.B_quantile),
                        "B": float(B),
                        "r_std": float(r_std),
                    }
                )
            return means, stds

        # Ball radii
        for p in r_percentiles:
            r_val = float(r_vals[float(p)])
            label = f"Ball r=p{int(p)} ({r_val:.3g})"
            curves[label] = eval_for_radius(r_kind=f"ball_p{int(p)}", r_val=r_val)

        # Baseline
        curves[f"Bounded r_std=2B ({r_std:.3g})"] = eval_for_radius(
            r_kind="std_2B", r_val=r_std
        )

        # Save per-n plot
        out_png = fig_dir / f"acc_vs_eps_n{n_per_class}.png"
        save_errorbar_plot(
            eps_list,
            curves,
            title=f"CIFAR-10 | ResNet18 embeds | Prototypes | n/class={n_per_class} (base={base_acc:.3f})",
            xlabel="epsilon (log scale)",
            ylabel="test accuracy",
            out_path=out_png,
            xscale_log=True,
        )

        # Paper plot: use first n as representative (or change policy as you like)
        if n_per_class == paper_n:
            paper_curves = curves

    # Save CSV
    cols = [
        "head",
        "n_per_class",
        "n_min",
        "lam",
        "delta",
        "sigma_method",
        "r_kind",
        "r_value",
        "eps",
        "Delta",
        "sigma",
        "acc_mean",
        "acc_std",
        "acc_non_private",
        "l2_normalize",
        "B_quantile",
        "B",
        "r_std",
    ]
    write_csv_rows(out_dir / "results.csv", results_rows, fieldnames=cols)

    # Save a single canonical figure for the paper
    if paper_curves is not None:
        save_errorbar_plot(
            eps_list,
            paper_curves,
            title=f"CIFAR-10 | Prototypes | n/class={paper_n}",
            xlabel="epsilon (log scale)",
            ylabel="test accuracy",
            out_path=fig_dir / "acc_vs_eps_prototypes.png",  # <-- used by LaTeX
            xscale_log=True,
        )


if __name__ == "__main__":
    main()
