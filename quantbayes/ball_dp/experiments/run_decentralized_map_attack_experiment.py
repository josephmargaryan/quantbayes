#!/usr/bin/env python3
"""Official Paper 3 theorem-aligned decentralized MAP attack experiment.

The attack in this script is not a generic reconstruction heuristic.  It is the exact
finite-prior Bayes/MAP rule for the same linear Gaussian observer model used by the
Paper 3 theorem:

    Y_A = (H_{A<-j} \\otimes I_p) s_j(z) + Gaussian noise.

For each trial we build a finite Ball-local prior, sample the true candidate, release
an observer-specific Gaussian view, score every candidate under the exact likelihood,
and report exact-identification success against the direct Gaussian ReRo bound.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from quantbayes.ball_dp.experiments.paper3_decentralized_common import (
    DEFAULT_OBSERVER_MODES,
    DEFAULT_ORDERS,
    add_embedding_loader_args,
    configure_matplotlib,
    embedding_radius_report,
    ensure_dir,
    graph_adjacency,
    graph_distances,
    load_embedding_dataset,
    metropolis_mixing_matrix,
    observer_accounting_row,
    parse_radius_grid,
    run_exact_finite_prior_trial,
    savefig_stem,
    slugify,
    write_json,
    write_rows_and_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=str, default="results_paper3")
    parser.add_argument("--dataset", type=str, default="MNIST-embeddings")
    parser.add_argument("--synthetic-feature-dim", type=int, default=0)
    parser.add_argument("--synthetic-num-examples", type=int, default=4096)
    parser.add_argument("--graphs", nargs="+", default=["path", "star", "complete"])
    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--lazy", type=float, default=0.0)
    parser.add_argument("--radius-grid", nargs="+", default=["q80"])
    parser.add_argument("--noise-grid", nargs="+", type=float, default=[4.0, 8.0, 16.0])
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--feature-lipschitz", type=float, default=1.0)
    parser.add_argument("--block-decay", type=float, default=1.0)
    parser.add_argument("--prior-size", type=int, default=8)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--release-seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--observer-modes", nargs="+", default=["nearest", "farthest", "all"]
    )
    parser.add_argument(
        "--attacked-nodes",
        nargs="+",
        type=int,
        default=None,
        help="Default: two representative nodes, 0 and floor(num_nodes/2).",
    )
    parser.add_argument("--orders", nargs="+", type=float, default=list(DEFAULT_ORDERS))
    parser.add_argument("--dp-delta", type=float, default=1e-6)
    parser.add_argument("--radius-estimation-seed", type=int, default=0)
    parser.add_argument(
        "--make-plots", action=argparse.BooleanOptionalAction, default=True
    )
    add_embedding_loader_args(parser)
    return parser.parse_args()


def load_reference(args: argparse.Namespace):
    if int(args.synthetic_feature_dim) > 0:
        rng = np.random.default_rng(123)
        X = rng.normal(
            size=(int(args.synthetic_num_examples), int(args.synthetic_feature_dim))
        ).astype(np.float32)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        y = rng.integers(0, 10, size=int(args.synthetic_num_examples), dtype=np.int32)
        tag = f"synthetic_d{int(args.synthetic_feature_dim)}"
        report = embedding_radius_report(X, seed=int(args.radius_estimation_seed))
        return tag, tag, X, y, int(args.synthetic_feature_dim), report
    data = load_embedding_dataset(args, args.dataset)
    X = np.asarray(data.X_train, dtype=np.float32)
    y = np.asarray(data.y_train, dtype=np.int32)
    report = embedding_radius_report(X, seed=int(args.radius_estimation_seed))
    return data.spec.tag, data.spec.display_name, X, y, int(data.feature_dim), report


def make_attack_plots(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return
    grouped = (
        df.groupby(["graph", "observer_mode", "noise_std"], dropna=False)
        .agg(
            success=("exact_identification_success", "mean"),
            bound=("direct_gaussian_rero_bound", "mean"),
            standard_bound=("standard_direct_gaussian_rero_bound", "mean"),
            mse=("mse", "mean"),
        )
        .reset_index()
        .sort_values("noise_std")
    )
    for graph in sorted(grouped["graph"].unique()):
        sub = grouped[grouped["graph"] == graph]
        fig, ax = plt.subplots()
        for obs, g in sub.groupby("observer_mode"):
            ax.plot(g["noise_std"], g["success"], marker="o", label=f"attack/{obs}")
            ax.plot(
                g["noise_std"],
                g["bound"],
                marker="x",
                linestyle="--",
                label=f"bound/{obs}",
            )
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Gaussian noise std sigma")
        ax.set_ylabel("success probability")
        ax.set_title(f"Exact finite-prior MAP attack vs bound: {graph}")
        ax.legend(ncol=2, fontsize=8)
        savefig_stem(fig, out_dir / "figures" / f"map_attack_vs_bound_{graph}")
        plt.close(fig)

    # Bound comparison against standard replacement adjacency at the same sigma.
    for graph in sorted(grouped["graph"].unique()):
        sub = grouped[grouped["graph"] == graph]
        fig, ax = plt.subplots()
        for obs, g in sub.groupby("observer_mode"):
            ax.plot(g["noise_std"], g["bound"], marker="o", label=f"Ball/{obs}")
            ax.plot(
                g["noise_std"],
                g["standard_bound"],
                marker="x",
                linestyle="--",
                label=f"standard/{obs}",
            )
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Gaussian noise std sigma")
        ax.set_ylabel("direct ReRo bound")
        ax.set_title(f"Ball vs standard bound at same release noise: {graph}")
        ax.legend(ncol=2, fontsize=8)
        savefig_stem(fig, out_dir / "figures" / f"ball_vs_standard_bound_{graph}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    dataset_tag, display_name, X_ref, y_ref, feature_dim, radius_report = (
        load_reference(args)
    )
    radii, radius_report = parse_radius_grid(
        args.radius_grid, X_ref, seed=int(args.radius_estimation_seed)
    )
    out_dir = ensure_dir(
        Path(args.results_root)
        / "paper3"
        / "decentralized_map_attack"
        / slugify(dataset_tag)
    )
    if args.attacked_nodes is None:
        attacked_nodes = sorted(set([0, int(args.num_nodes) // 2]))
    else:
        attacked_nodes = [int(v) for v in args.attacked_nodes]

    rows: list[dict] = []
    graph_metadata: dict[str, dict] = {}
    center_rng = np.random.default_rng(2026)
    center_indices = center_rng.choice(
        np.arange(len(X_ref)),
        size=max(1, int(args.trials)),
        replace=len(X_ref) < int(args.trials),
    )

    for graph in args.graphs:
        A = graph_adjacency(graph, int(args.num_nodes))
        W = metropolis_mixing_matrix(A, lazy=float(args.lazy))
        D = graph_distances(A)
        graph_metadata[str(graph)] = {
            "adjacency": A.astype(int).tolist(),
            "mixing_matrix": W.astype(float).tolist(),
            "graph_distances": D.astype(float).tolist(),
        }
        for radius in radii:
            for noise_std in args.noise_grid:
                for attacked_node in attacked_nodes:
                    for observer_mode in args.observer_modes:
                        # Standard replacement-adjacent bound for the same release model/noise.
                        standard_row = observer_accounting_row(
                            dataset_tag=dataset_tag,
                            graph=str(graph),
                            W=W,
                            distances=D,
                            rounds=int(args.rounds),
                            observer_mode=str(observer_mode),
                            attacked_node=int(attacked_node),
                            radius=float(radius),
                            clip_norm=float(args.clip_norm),
                            noise_std=float(noise_std),
                            feature_dim=int(feature_dim),
                            prior_size=int(args.prior_size),
                            mechanism="standard",
                            orders=tuple(float(a) for a in args.orders),
                            dp_delta=float(args.dp_delta),
                            feature_lipschitz=float(args.feature_lipschitz),
                            block_decay=float(args.block_decay),
                        )
                        for release_seed in args.release_seeds:
                            for trial in range(int(args.trials)):
                                idx = int(center_indices[trial % len(center_indices)])
                                graph_seed = (
                                    sum(
                                        (i + 1) * ord(ch)
                                        for i, ch in enumerate(str(graph))
                                    )
                                    % 1009
                                )
                                rng = np.random.default_rng(
                                    int(release_seed) * 1_000_003
                                    + int(trial) * 9176
                                    + int(attacked_node) * 131
                                    + int(graph_seed)
                                )
                                row = run_exact_finite_prior_trial(
                                    W=W,
                                    distances=D,
                                    rounds=int(args.rounds),
                                    observer_mode=str(observer_mode),
                                    attacked_node=int(attacked_node),
                                    radius=float(radius),
                                    clip_norm=float(args.clip_norm),
                                    noise_std=float(noise_std),
                                    prior_size=int(args.prior_size),
                                    feature_dim=int(feature_dim),
                                    center=np.asarray(X_ref[idx], dtype=np.float32),
                                    label=int(y_ref[idx]),
                                    rng=rng,
                                    orders=tuple(float(a) for a in args.orders),
                                    dp_delta=float(args.dp_delta),
                                    feature_lipschitz=float(args.feature_lipschitz),
                                    block_decay=float(args.block_decay),
                                )
                                row.update(
                                    {
                                        "dataset_tag": dataset_tag,
                                        "graph": str(graph),
                                        "radius": float(radius),
                                        "noise_std": float(noise_std),
                                        "release_seed": int(release_seed),
                                        "trial": int(trial),
                                        "center_index": int(idx),
                                        "center_label": int(y_ref[idx]),
                                        "standard_transferred_sensitivity": float(
                                            standard_row["transferred_sensitivity"]
                                        ),
                                        "standard_direct_gaussian_rero_bound": float(
                                            standard_row["direct_gaussian_rero_bound"]
                                        ),
                                        "standard_dp_epsilon": float(
                                            standard_row["dp_epsilon"]
                                        ),
                                    }
                                )
                                rows.append(row)

    rows_path, summary_path = write_rows_and_summary(
        rows=rows,
        out_dir=out_dir,
        rows_name="map_attack_rows.csv",
        summary_name="map_attack_summary.csv",
        group_cols=[
            "dataset_tag",
            "graph",
            "observer_mode",
            "attacked_node",
            "radius",
            "noise_std",
        ],
        value_cols=[
            "exact_identification_success",
            "prior_rank",
            "prior_hit_at_5",
            "mse",
            "l2_error",
            "direct_gaussian_rero_bound",
            "standard_direct_gaussian_rero_bound",
            "gaussian_dp_mu",
            "fdp_tradeoff_alpha_0_10",
            "fdp_tradeoff_alpha_0_25",
            "fdp_tradeoff_alpha_0_50",
            "transferred_sensitivity",
            "standard_transferred_sensitivity",
        ],
    )
    df = pd.DataFrame(rows)
    if bool(args.make_plots):
        make_attack_plots(df, out_dir)
    write_json(
        out_dir / "metadata.json",
        {
            "experiment": "paper3_decentralized_map_attack",
            "dataset": display_name,
            "dataset_tag": dataset_tag,
            "feature_dim": int(feature_dim),
            "radius_grid": [float(v) for v in radii],
            "radius_report": radius_report,
            "args": vars(args),
            "graphs": graph_metadata,
            "rows_path": str(rows_path),
            "summary_path": str(summary_path),
        },
    )
    print(f"Wrote {rows_path} ({len(rows)} rows)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
