# quantbayes/ball_dp/experiments/exp_privacy_profile.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import matplotlib.pyplot as plt

from quantbayes.ball_dp.experiments.cifar10_embed_cache import (
    CIFAR10EmbedConfig,
    get_or_compute_cifar10_embeddings,
)
from quantbayes.ball_dp.utils.io import ensure_dir, write_json, write_csv_rows
from quantbayes.ball_dp.utils.seeding import set_global_seed

from quantbayes.retrieval_dp.radius import within_class_nn_distances
from quantbayes.retrieval_dp.metrics import l2_norm_rows
from quantbayes.retrieval_dp.sensitivity import bounded_replacement_radius


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    x.sort()
    if x.size == 0:
        return np.asarray([]), np.asarray([])
    y = np.linspace(0.0, 1.0, num=x.size, endpoint=True)
    return x, y


def _quantiles(x: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(100*q):02d}": float("nan") for q in qs}
    vals = np.quantile(x, np.array(qs, dtype=np.float64))
    return {f"q{int(100*q):02d}": float(v) for q, v in zip(qs, vals)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./runs/cifar10_privacy_profile")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument(
        "--cache_npz", type=str, default="./cache/cifar10_resnet18_embeds.npz"
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--weights", type=str, default="DEFAULT")
    ap.add_argument("--l2_normalize", action="store_true")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nn_sample_per_class", type=int, default=400)
    ap.add_argument("--r_percentiles", type=str, default="10,25,50,75,90")
    ap.add_argument("--B_quantile", type=float, default=0.999)

    # plotting controls
    ap.add_argument("--bins", type=int, default=200)

    args = ap.parse_args()
    set_global_seed(args.seed)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    # Load embeddings
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

    # Replacement-model distances: within-class NN distances (approx via subsampling)
    nn_dists = within_class_nn_distances(
        Ztr,
        ytr,
        num_classes=10,
        per_class=int(args.nn_sample_per_class),
        seed=int(args.seed),
    )

    r_percentiles: List[float] = [
        float(s.strip()) for s in args.r_percentiles.split(",") if s.strip()
    ]
    r_vals = {p: float(np.percentile(nn_dists, p)) for p in r_percentiles}

    # bounded-replacement baseline r_std = 2B (B from public quantile)
    norms = l2_norm_rows(Ztr)
    B = float(np.quantile(norms, float(args.B_quantile)))
    r_std = float(bounded_replacement_radius(B))

    # Compute multiplier distributions t/r
    curves: Dict[str, np.ndarray] = {}
    for p in r_percentiles:
        r = r_vals[p]
        curves[f"Ball r=p{int(p)} ({r:.3g})"] = nn_dists / max(r, 1e-12)

    curves[f"Bounded r_std=2B ({r_std:.3g})"] = nn_dists / max(r_std, 1e-12)

    # Save summary stats
    rows = []
    for name, mult in curves.items():
        qs = _quantiles(mult, qs=(0.50, 0.90, 0.95, 0.99))
        rows.append(
            {
                "curve": name,
                "mult_mean": float(np.mean(mult)),
                "mult_std": float(np.std(mult)),
                **qs,
                "mult_max": float(np.max(mult)),
            }
        )
    write_csv_rows(
        out_dir / "privacy_profile_multiplier_summary.csv",
        rows,
        fieldnames=list(rows[0].keys()),
    )

    write_json(
        out_dir / "privacy_profile_setup.json",
        {
            "seed": int(args.seed),
            "l2_normalize": int(bool(args.l2_normalize)),
            "nn_sample_per_class": int(args.nn_sample_per_class),
            "r_percentiles": r_percentiles,
            "r_values": r_vals,
            "B_quantile": float(args.B_quantile),
            "B": float(B),
            "r_std": float(r_std),
            "n_nn_dists": int(nn_dists.size),
        },
    )

    # Plot CDF overlays of t/r
    plt.figure(figsize=(7.5, 4.8))
    for name, mult in curves.items():
        x, y = _ecdf(mult)
        if x.size:
            plt.plot(x, y, label=name, linewidth=1.75)

    plt.axvline(1.0, linestyle="--", linewidth=1.0)
    plt.xlim(left=0.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("privacy-loss multiplier  t / r")
    plt.ylabel("CDF")
    plt.title("Privacy profile multiplier under replacement model (within-class NN)")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path = fig_dir / "privacy_profile_multiplier_cdf.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"[OK] wrote: {out_path}")
    print(f"[OK] wrote: {out_dir / 'privacy_profile_multiplier_summary.csv'}")


if __name__ == "__main__":
    main()
