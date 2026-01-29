# quantbayes/ball_dp/experiments/exp_cifar10_logreg.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from quantbayes.ball_dp.experiments.cifar10_embed_cache import (
    CIFAR10EmbedConfig,
    get_or_compute_cifar10_embeddings,
)
from quantbayes.ball_dp.radius import within_class_nn_distances, radii_from_percentiles
from quantbayes.ball_dp.metrics import l2_norms
from quantbayes.ball_dp.lz import lz_softmax_linear_bound
from quantbayes.ball_dp.sensitivity import erm_sensitivity_l2
from quantbayes.ball_dp.mechanisms import gaussian_sigma, add_gaussian_noise
from quantbayes.ball_dp.heads.logreg_torch import (
    LogRegTorchConfig,
    train_softmax_logreg_torch,
    predict_softmax_logreg_torch,
)
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
    """
    Subsample exactly n_per_class points per class (without replacement).
    """
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
    ap.add_argument("--out_dir", type=str, default="./runs/cifar10_logreg_ball_dp")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument(
        "--cache_npz", type=str, default="./cache/cifar10_resnet18_embeds.npz"
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--weights", type=str, default="DEFAULT")
    ap.add_argument("--l2_normalize", action="store_true")

    ap.add_argument("--lam", type=float, default=1e-2)  # ERM strong convexity (L2 reg)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument(
        "--sigma_method", type=str, default="classic", choices=["classic", "analytic"]
    )

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=10)

    ap.add_argument("--nn_sample_per_class", type=int, default=400)
    ap.add_argument("--r_percentiles", type=str, default="10,25,50,75,90")
    ap.add_argument("--B_quantile", type=float, default=0.999)

    ap.add_argument("--eps_list", type=str, default="0.05,0.1,0.2,0.5,1,2,5,10")

    # logreg training knobs
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--logreg_batch", type=int, default=2048)

    # NEW: produce multiple plots by training-set size per class (like the prototype exp)
    ap.add_argument("--n_per_class_list", type=str, default="5000")

    args = ap.parse_args()
    set_global_seed(args.seed)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

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

    # Radii (computed on full training set for consistency across n_per_class)
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

    # Baseline radius 2B (B from norms quantile on full training set)
    norms = l2_norms(Ztr)
    B = float(np.quantile(norms, float(args.B_quantile)))
    r_std = 2.0 * B

    write_json(
        out_dir / "radii.json",
        {
            "l2_normalize": bool(args.l2_normalize),
            "B_quantile": float(args.B_quantile),
            "B": float(B),
            "r_std": float(r_std),
            "r_percentiles": r_percentiles,
            "r_values": r_vals,
        },
    )

    # L_z bound (conservative) for softmax linear head
    # In theory, B should be a hard bound (e.g. via L2-normalization or clipping).
    Lz = lz_softmax_linear_bound(B=B, lam=float(args.lam))

    eps_list = [float(s.strip()) for s in args.eps_list.split(",") if s.strip()]
    n_list = [int(s.strip()) for s in args.n_per_class_list.split(",") if s.strip()]

    all_results: List[Dict[str, object]] = []

    for n_per_class in n_list:
        # Subsample train set to n_per_class per class (still evaluate on full test set)
        Zsub, ysub = subsample_per_class(
            Ztr, ytr, n_per_class=int(n_per_class), num_classes=10, seed=args.seed
        )

        # Train non-private logreg on subset
        log_cfg = LogRegTorchConfig(
            num_classes=10,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            epochs=int(args.epochs),
            batch_size=int(args.logreg_batch),
            device=str(args.device),
            seed=int(args.seed),
        )
        model, tr_metrics = train_softmax_logreg_torch(
            Zsub, ysub, None, None, cfg=log_cfg
        )
        base_acc = accuracy(predict_softmax_logreg_torch(model, Zte), yte)

        # Flatten parameters (W,b) into one vector for output perturbation
        W = model.linear.weight.detach().cpu().numpy().astype(np.float32)  # (K,d)
        bvec = model.linear.bias.detach().cpu().numpy().astype(np.float32)  # (K,)
        params = np.concatenate([W.reshape(-1), bvec.reshape(-1)], axis=0)

        rng = np.random.default_rng(args.seed)

        curves: Dict[str, Tuple[List[float], List[float]]] = {}

        def eval_for_radius(
            r_kind: str, r_val: float
        ) -> Tuple[List[float], List[float]]:
            means, stds = [], []
            n = int(Zsub.shape[0])

            sens = erm_sensitivity_l2(lz=Lz, r=float(r_val), lam=float(args.lam), n=n)
            Delta = sens.delta_l2

            for eps in eps_list:
                sig = gaussian_sigma(
                    Delta, float(eps), float(args.delta), method=args.sigma_method
                )
                acc_trials = []
                for _ in range(int(args.trials)):
                    noisy = add_gaussian_noise(params, sig, rng=rng)
                    Wn = noisy[: W.size].reshape(W.shape)
                    bn = noisy[W.size :].reshape(bvec.shape)

                    # plug into model (fast)
                    model.linear.weight.data[:] = 0.0
                    model.linear.bias.data[:] = 0.0
                    model.linear.weight.data += torch.from_numpy(Wn)
                    model.linear.bias.data += torch.from_numpy(bn)

                    yhat = predict_softmax_logreg_torch(model, Zte)
                    acc_trials.append(accuracy(yhat, yte))

                m = float(np.mean(acc_trials))
                s = float(np.std(acc_trials))
                means.append(m)
                stds.append(s)

                all_results.append(
                    {
                        "head": "logreg_softmax",
                        "n_per_class": int(n_per_class),
                        "lam": float(args.lam),
                        "delta": float(args.delta),
                        "sigma_method": str(args.sigma_method),
                        "Lz_bound": float(Lz),
                        "B": float(B),
                        "r_kind": str(r_kind),
                        "r_value": float(r_val),
                        "n": int(n),
                        "eps": float(eps),
                        "Delta": float(Delta),
                        "sigma": float(sig),
                        "acc_mean": float(m),
                        "acc_std": float(s),
                        "acc_non_private": float(base_acc),
                        "l2_normalize": int(bool(args.l2_normalize)),
                        "train_acc": float(tr_metrics.get("train_acc", float("nan"))),
                    }
                )

            return means, stds

        # Ball radii
        for p in r_percentiles:
            rv = float(r_vals[float(p)])
            label = f"Ball r=p{int(p)} ({rv:.3g})"
            curves[label] = eval_for_radius(r_kind=f"ball_p{int(p)}", r_val=rv)

        # Baseline radius
        curves[f"Bounded r_std=2B ({r_std:.3g})"] = eval_for_radius(
            r_kind="std_2B", r_val=r_std
        )

        # Save per-n plot (for your 2x3 grid)
        save_errorbar_plot(
            eps_list,
            curves,
            title=f"CIFAR-10 | Softmax LogReg | n/class={n_per_class} (base={base_acc:.3f})",
            xlabel="epsilon (log scale)",
            ylabel="test accuracy",
            out_path=fig_dir / f"acc_vs_eps_logreg_n{int(n_per_class)}.png",
            xscale_log=True,
        )

    # Write CSV once (now includes n_per_class)
    cols = [
        "head",
        "n_per_class",
        "lam",
        "delta",
        "sigma_method",
        "Lz_bound",
        "B",
        "r_kind",
        "r_value",
        "n",
        "eps",
        "Delta",
        "sigma",
        "acc_mean",
        "acc_std",
        "acc_non_private",
        "l2_normalize",
        "train_acc",
    ]
    write_csv_rows(out_dir / "results.csv", all_results, fieldnames=cols)


if __name__ == "__main__":
    main()
