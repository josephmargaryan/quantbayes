#!/usr/bin/env python3
"""Official Paper 3 observer-specific decentralized Ball-DP experiment.

This script evaluates the topology-aware theorem from Paper 3.  For each graph,
attacked node, observer view, radius, and noise level, it computes

  * the whitened transferred sensitivity Δ_{A<-j}(r),
  * the observer-specific Ball-PN-RDP curve,
  * the direct Gaussian ReRo finite-prior reconstruction bound, and
  * the corresponding standard replacement-adjacent baseline.

The release model is the exact linear Gaussian observer model used in the paper:
``Y_A = (H_{A<-j} \\otimes I_p) s_j(z) + Gaussian noise``.  The feature map is the
clipped identity map on embedding vectors, so the Ball step sensitivity is
``min(L_z r, 2C)`` and the standard replacement sensitivity is ``2C``.
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
    DEFAULT_GRAPH_LIST,
    DEFAULT_MECHANISMS,
    DEFAULT_OBSERVER_MODES,
    DEFAULT_ORDERS,
    DEFAULT_RADIUS_TAGS,
    add_embedding_loader_args,
    build_transfer_for_observer,
    configure_matplotlib,
    covariance_time,
    embedding_radius_report,
    ensure_dir,
    graph_adjacency,
    graph_distances,
    load_embedding_dataset,
    metropolis_mixing_matrix,
    observer_accounting_row,
    observer_nodes_from_mode,
    parse_radius_grid,
    savefig_stem,
    slugify,
    write_json,
    write_rows_and_summary,
)
from quantbayes.ball_dp.decentralized import account_linear_gaussian_observer
from quantbayes.ball_dp.experiments.paper3_decentralized_common import (
    direct_gaussian_rero_bound,
    mechanism_step_sensitivities,
)


def as_float_list(values: Sequence[str | float]) -> list[float]:
    return [float(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=str, default="results_paper3")
    parser.add_argument("--dataset", type=str, default="MNIST-embeddings")
    parser.add_argument(
        "--synthetic-feature-dim",
        type=int,
        default=0,
        help="If positive, skip dataset loading and use synthetic features of this dimension.",
    )
    parser.add_argument("--synthetic-num-examples", type=int, default=4096)
    parser.add_argument("--graphs", nargs="+", default=list(DEFAULT_GRAPH_LIST))
    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument(
        "--lazy",
        type=float,
        default=0.0,
        help="Lazy-Metropolis self-loop interpolation; 0 means ordinary Metropolis.",
    )
    parser.add_argument(
        "--radius-grid",
        nargs="+",
        default=list(DEFAULT_RADIUS_TAGS),
        help="Radii as floats or tags q50/q80/q95 estimated from embeddings.",
    )
    parser.add_argument(
        "--noise-grid", nargs="+", type=float, default=[2.0, 4.0, 8.0, 16.0]
    )
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--feature-lipschitz", type=float, default=1.0)
    parser.add_argument("--block-decay", type=float, default=1.0)
    parser.add_argument("--prior-size", type=int, default=8)
    parser.add_argument(
        "--observer-modes", nargs="+", default=list(DEFAULT_OBSERVER_MODES)
    )
    parser.add_argument(
        "--attacked-nodes",
        nargs="+",
        type=int,
        default=None,
        help="Default: all nodes.",
    )
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=list(DEFAULT_MECHANISMS),
        choices=["ball", "standard"],
    )
    parser.add_argument("--orders", nargs="+", type=float, default=list(DEFAULT_ORDERS))
    parser.add_argument("--dp-delta", type=float, default=1e-6)
    parser.add_argument("--radius-estimation-seed", type=int, default=0)
    parser.add_argument(
        "--make-plots", action=argparse.BooleanOptionalAction, default=True
    )
    add_embedding_loader_args(parser)
    return parser.parse_args()


def load_reference_features(
    args: argparse.Namespace,
) -> tuple[str, str, np.ndarray, int, dict]:
    if int(args.synthetic_feature_dim) > 0:
        rng = np.random.default_rng(0)
        X = rng.normal(
            size=(int(args.synthetic_num_examples), int(args.synthetic_feature_dim))
        ).astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)
        tag = f"synthetic_d{int(args.synthetic_feature_dim)}"
        report = embedding_radius_report(X, seed=int(args.radius_estimation_seed))
        return tag, tag, X, int(args.synthetic_feature_dim), report
    data = load_embedding_dataset(args, args.dataset)
    tag = data.spec.tag
    display = data.spec.display_name
    X = np.asarray(data.X_train, dtype=np.float32)
    report = embedding_radius_report(X, seed=int(args.radius_estimation_seed))
    return tag, display, X, int(data.feature_dim), report


def make_node_heatmaps(
    *,
    out_dir: Path,
    dataset_tag: str,
    graph: str,
    W: np.ndarray,
    distances: np.ndarray,
    rounds: int,
    radius: float,
    noise_std: float,
    clip_norm: float,
    feature_dim: int,
    prior_size: int,
    orders: Sequence[float],
    dp_delta: float,
    feature_lipschitz: float,
    block_decay: float,
) -> None:
    import matplotlib.pyplot as plt

    m = int(W.shape[0])
    mat_delta = np.zeros((m, m), dtype=float)
    mat_bound = np.zeros((m, m), dtype=float)
    for observer in range(m):
        for attacked in range(m):
            H = build_transfer_for_observer(
                W=W,
                rounds=int(rounds),
                observer_nodes=(observer,),
                attacked_node=attacked,
            )
            cov = covariance_time(H.shape[0], float(noise_std))
            deltas = mechanism_step_sensitivities(
                mechanism="ball",
                radius=float(radius),
                clip_norm=float(clip_norm),
                rounds=int(rounds),
                feature_lipschitz=float(feature_lipschitz),
                decay=float(block_decay),
            )
            acct = account_linear_gaussian_observer(
                transfer_matrix=H,
                covariance=cov,
                block_sensitivities=deltas,
                parameter_dim=int(feature_dim),
                orders=tuple(float(a) for a in orders),
                radius=float(radius),
                dp_delta=float(dp_delta),
                attacked_node=attacked,
                observer=(observer,),
                covariance_mode="kron_eye",
                method="auto",
            )
            delta = float(np.sqrt(max(0.0, acct.sensitivity_sq)))
            mat_delta[observer, attacked] = delta
            mat_bound[observer, attacked] = direct_gaussian_rero_bound(
                kappa=1.0 / float(prior_size),
                transferred_sensitivity=delta,
            )

    for name, mat, label in [
        ("transferred_sensitivity", mat_delta, r"$\Delta_{A\leftarrow j}(r)$"),
        ("direct_rero_bound", mat_bound, "direct Gaussian ReRo bound"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(mat, aspect="auto")
        ax.set_xlabel("attacked node j")
        ax.set_ylabel("observer node a")
        ax.set_title(f"{dataset_tag}: {graph}, r={radius:.3g}, sigma={noise_std:g}")
        fig.colorbar(im, ax=ax, label=label)
        savefig_stem(fig, out_dir / "figures" / f"heatmap_{graph}_{name}")
        plt.close(fig)


def make_summary_plots(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return
    # Radius ablation at the smallest listed noise level: this is the policy-knob plot.
    for graph in sorted(df["graph"].unique()):
        sub = df[
            (df["graph"] == graph) & (df["observer_mode"].isin(["farthest", "all"]))
        ]
        if sub.empty:
            continue
        sigma = float(np.nanmin(sub["noise_std"].to_numpy(dtype=float)))
        sub = sub[np.isclose(sub["noise_std"], sigma)]
        grouped = (
            sub.groupby(["mechanism", "observer_mode", "radius"], dropna=False)[
                "direct_gaussian_rero_bound"
            ]
            .mean()
            .reset_index()
            .sort_values("radius")
        )
        fig, ax = plt.subplots()
        for (mechanism, observer_mode), g in grouped.groupby(
            ["mechanism", "observer_mode"]
        ):
            ax.plot(
                g["radius"],
                g["direct_gaussian_rero_bound"],
                marker="o",
                label=f"{mechanism}/{observer_mode}",
            )
        ax.set_xlabel("Ball radius r")
        ax.set_ylabel("mean direct ReRo bound")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"Radius ablation, {graph}, sigma={sigma:g}")
        ax.legend()
        savefig_stem(fig, out_dir / "figures" / f"radius_ablation_{graph}")
        plt.close(fig)

    # Noise ablation: check monotonicity of protection as sigma grows.
    for graph in sorted(df["graph"].unique()):
        sub = df[(df["graph"] == graph) & (df["mechanism"] == "ball")]
        if sub.empty:
            continue
        radius = float(np.nanmedian(sub["radius"].to_numpy(dtype=float)))
        radii = np.sort(sub["radius"].unique())
        if len(radii):
            radius = float(radii[len(radii) // 2])
        sub = sub[np.isclose(sub["radius"], radius)]
        grouped = (
            sub.groupby(["observer_mode", "noise_std"], dropna=False)[
                "direct_gaussian_rero_bound"
            ]
            .mean()
            .reset_index()
            .sort_values("noise_std")
        )
        fig, ax = plt.subplots()
        for observer_mode, g in grouped.groupby("observer_mode"):
            ax.plot(
                g["noise_std"],
                g["direct_gaussian_rero_bound"],
                marker="o",
                label=str(observer_mode),
            )
        ax.set_xscale("log")
        ax.set_xlabel("Gaussian noise std sigma")
        ax.set_ylabel("mean direct ReRo bound")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"Noise ablation, {graph}, r={radius:.3g}")
        ax.legend()
        savefig_stem(fig, out_dir / "figures" / f"noise_ablation_{graph}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    dataset_tag, display_name, X_ref, feature_dim, radius_report = (
        load_reference_features(args)
    )
    radii, radius_report = parse_radius_grid(
        args.radius_grid, X_ref, seed=int(args.radius_estimation_seed)
    )
    root = (
        Path(args.results_root)
        / "paper3"
        / "decentralized_observer"
        / slugify(dataset_tag)
    )
    out_dir = ensure_dir(root)
    attacked_nodes = (
        list(range(int(args.num_nodes)))
        if args.attacked_nodes is None
        else [int(v) for v in args.attacked_nodes]
    )

    rows: list[dict] = []
    graph_metadata: dict[str, dict] = {}
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
                for mechanism in args.mechanisms:
                    for attacked_node in attacked_nodes:
                        for observer_mode in args.observer_modes:
                            row = observer_accounting_row(
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
                                mechanism=str(mechanism),
                                orders=tuple(float(a) for a in args.orders),
                                dp_delta=float(args.dp_delta),
                                feature_lipschitz=float(args.feature_lipschitz),
                                block_decay=float(args.block_decay),
                            )
                            rows.append(row)
        if args.make_plots and radii and args.noise_grid:
            make_node_heatmaps(
                out_dir=out_dir,
                dataset_tag=dataset_tag,
                graph=str(graph),
                W=W,
                distances=D,
                rounds=int(args.rounds),
                radius=float(radii[min(1, len(radii) - 1)]),
                noise_std=float(args.noise_grid[min(1, len(args.noise_grid) - 1)]),
                clip_norm=float(args.clip_norm),
                feature_dim=int(feature_dim),
                prior_size=int(args.prior_size),
                orders=tuple(float(a) for a in args.orders),
                dp_delta=float(args.dp_delta),
                feature_lipschitz=float(args.feature_lipschitz),
                block_decay=float(args.block_decay),
            )

    rows_path, summary_path = write_rows_and_summary(
        rows=rows,
        out_dir=out_dir,
        rows_name="observer_rows.csv",
        summary_name="observer_summary.csv",
        group_cols=[
            "dataset_tag",
            "graph",
            "mechanism",
            "observer_mode",
            "radius",
            "noise_std",
        ],
        value_cols=[
            "transferred_sensitivity",
            "gaussian_dp_mu",
            "fdp_tradeoff_alpha_0_10",
            "fdp_tradeoff_alpha_0_25",
            "fdp_tradeoff_alpha_0_50",
            "direct_gaussian_rero_bound",
            "rdp_rero_bound",
            "dp_epsilon",
            "rdp_eps_alpha_16",
        ],
    )
    df = pd.DataFrame(rows)
    if args.make_plots:
        make_summary_plots(df, out_dir)
    write_json(
        out_dir / "metadata.json",
        {
            "experiment": "paper3_decentralized_observer",
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
