# quantbayes/ball_dp/experiments/exp_radius_coverage.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quantbayes.ball_dp.experiments.cifar10_embed_cache import (
    CIFAR10EmbedConfig,
    get_or_compute_cifar10_embeddings,
)
from quantbayes.ball_dp.radius import within_class_nn_distances
from quantbayes.ball_dp.utils import ensure_dir, save_line_plot, set_global_seed


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nn_sample_per_class", type=int, default=400)
    ap.add_argument("--grid_points", type=int, default=25)
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
    Ztr, ytr, _, _ = get_or_compute_cifar10_embeddings(emb_cfg)

    nn = within_class_nn_distances(
        Ztr,
        ytr,
        num_classes=10,
        per_class=int(args.nn_sample_per_class),
        seed=int(args.seed),
    )
    # grid radii across observed range
    lo, hi = float(np.min(nn)), float(np.max(nn))
    rs = np.linspace(lo, hi, int(args.grid_points))
    cov = np.array([(nn <= r).mean() for r in rs], dtype=np.float64)

    save_line_plot(
        rs,
        cov,
        title="Coverage curve Cov(r) from within-class NN distances",
        xlabel="radius r",
        ylabel="Cov(r) = P(nn_dist <= r)",
        out_path=fig_dir / "coverage_curve.png",  # <-- used by LaTeX
    )


if __name__ == "__main__":
    main()
