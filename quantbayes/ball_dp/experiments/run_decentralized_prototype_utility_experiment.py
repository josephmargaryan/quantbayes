#!/usr/bin/env python3
"""Official Paper 3 decentralized utility/privacy tradeoff experiment.

This is a lightweight decentralized learning benchmark.  Nodes compute clipped local
class-sum prototype states, add calibrated Gaussian noise, and run deterministic
Metropolis gossip.  The final node prototypes are evaluated by nearest-prototype
classification on the embedding test set.

The calibration compares Ball-local feature replacement, with sensitivity
``min(L_z r, 2C)``, to standard replacement adjacency, with sensitivity ``2C``.  This
makes the policy knob r unavoidable: smaller radii require less noise at the same
DP target and should therefore preserve more utility.

Counts are treated as public in this controlled benchmark, so the protected object is
the feature vector conditional on label.  The observer-specific MAP attack experiment
separately audits reconstruction under the exact Gaussian view model.
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
    DEFAULT_ORDERS,
    DEFAULT_RADIUS_TAGS,
    add_embedding_loader_args,
    configure_matplotlib,
    embedding_radius_report,
    ensure_dir,
    graph_adjacency,
    load_embedding_dataset,
    mechanism_noise_for_target_dp,
    metropolis_mixing_matrix,
    parse_radius_grid,
    run_noisy_prototype_gossip,
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
    parser.add_argument("--synthetic-num-classes", type=int, default=10)
    parser.add_argument("--synthetic-num-train", type=int, default=4096)
    parser.add_argument("--synthetic-num-test", type=int, default=1024)
    parser.add_argument("--graphs", nargs="+", default=list(DEFAULT_GRAPH_LIST))
    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--lazy", type=float, default=0.0)
    parser.add_argument("--radius-grid", nargs="+", default=list(DEFAULT_RADIUS_TAGS))
    parser.add_argument(
        "--epsilon-grid", nargs="+", type=float, default=[2.0, 4.0, 8.0]
    )
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=list(DEFAULT_MECHANISMS),
        choices=["ball", "standard"],
    )
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--feature-lipschitz", type=float, default=1.0)
    parser.add_argument("--dp-delta", type=float, default=1e-6)
    parser.add_argument("--orders", nargs="+", type=float, default=list(DEFAULT_ORDERS))
    parser.add_argument("--release-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-train-examples", type=int, default=6000)
    parser.add_argument("--max-test-examples", type=int, default=2000)
    parser.add_argument("--radius-estimation-seed", type=int, default=0)
    parser.add_argument(
        "--make-plots", action=argparse.BooleanOptionalAction, default=True
    )
    add_embedding_loader_args(parser)
    return parser.parse_args()


def stratified_or_random_subset(
    X: np.ndarray, y: np.ndarray, max_examples: int, seed: int
):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    n = len(y)
    if int(max_examples) <= 0 or int(max_examples) >= n:
        return X, y
    rng = np.random.default_rng(int(seed))
    labels, counts = np.unique(y, return_counts=True)
    if int(max_examples) < len(labels):
        idx = rng.choice(np.arange(n), size=int(max_examples), replace=False)
        return X[idx], y[idx]
    raw = counts / np.sum(counts) * int(max_examples)
    alloc = np.maximum(1, np.floor(raw).astype(int))
    alloc = np.minimum(alloc, counts)
    while int(np.sum(alloc)) > int(max_examples):
        j = int(np.argmax(alloc))
        if alloc[j] > 1:
            alloc[j] -= 1
        else:
            break
    while int(np.sum(alloc)) < int(max_examples):
        room = counts - alloc
        candidates = np.flatnonzero(room > 0)
        if candidates.size == 0:
            break
        j = int(candidates[np.argmax(room[candidates])])
        alloc[j] += 1
    chosen = []
    for label, take in zip(labels, alloc, strict=True):
        idx = np.flatnonzero(y == label)
        chosen.extend(rng.choice(idx, size=int(take), replace=False).tolist())
    rng.shuffle(chosen)
    idx = np.asarray(chosen, dtype=np.int64)
    return X[idx], y[idx]


def load_data(args: argparse.Namespace):
    if int(args.synthetic_feature_dim) > 0:
        rng = np.random.default_rng(42)
        K = int(args.synthetic_num_classes)
        p = int(args.synthetic_feature_dim)
        centers = rng.normal(size=(K, p)).astype(np.float32)
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        y_train = rng.integers(0, K, size=int(args.synthetic_num_train), dtype=np.int32)
        y_test = rng.integers(0, K, size=int(args.synthetic_num_test), dtype=np.int32)
        X_train = centers[y_train] + 0.25 * rng.normal(size=(len(y_train), p)).astype(
            np.float32
        )
        X_test = centers[y_test] + 0.25 * rng.normal(size=(len(y_test), p)).astype(
            np.float32
        )
        tag = f"synthetic_proto_d{p}"
        report = embedding_radius_report(X_train, seed=int(args.radius_estimation_seed))
        return tag, tag, X_train, y_train, X_test, y_test, K, p, report
    data = load_embedding_dataset(args, args.dataset)
    X_train = np.asarray(data.X_train, dtype=np.float32)
    y_train = np.asarray(data.y_train, dtype=np.int32)
    X_test = np.asarray(data.X_test, dtype=np.float32)
    y_test = np.asarray(data.y_test, dtype=np.int32)
    X_train, y_train = stratified_or_random_subset(
        X_train, y_train, int(args.max_train_examples), seed=0
    )
    X_test, y_test = stratified_or_random_subset(
        X_test, y_test, int(args.max_test_examples), seed=1
    )
    report = embedding_radius_report(X_train, seed=int(args.radius_estimation_seed))
    return (
        data.spec.tag,
        data.spec.display_name,
        X_train,
        y_train,
        X_test,
        y_test,
        int(data.num_classes),
        int(data.feature_dim),
        report,
    )


def make_utility_plots(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return
    for graph in sorted(df["graph"].unique()):
        sub = df[(df["graph"] == graph) & (df["mechanism"] != "none")]
        if sub.empty:
            continue
        # Accuracy vs epsilon at median radius.
        radii = np.sort(sub["radius"].unique())
        radius = float(radii[len(radii) // 2])
        sub_r = sub[np.isclose(sub["radius"], radius)]
        grouped = (
            sub_r.groupby(["mechanism", "target_epsilon"], dropna=False)[
                "accuracy_mean"
            ]
            .mean()
            .reset_index()
            .sort_values("target_epsilon")
        )
        fig, ax = plt.subplots()
        for mechanism, g in grouped.groupby("mechanism"):
            ax.plot(
                g["target_epsilon"],
                g["accuracy_mean"],
                marker="o",
                label=str(mechanism),
            )
        ax.set_xlabel("target epsilon")
        ax.set_ylabel("mean node accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"Prototype utility vs privacy, {graph}, r={radius:.3g}")
        ax.legend()
        savefig_stem(fig, out_dir / "figures" / f"accuracy_vs_epsilon_{graph}")
        plt.close(fig)

        # Accuracy and calibrated sigma vs radius at the largest epsilon.
        eps = float(np.max(sub["target_epsilon"].to_numpy(dtype=float)))
        sub_e = sub[np.isclose(sub["target_epsilon"], eps)]
        grouped = (
            sub_e.groupby(["mechanism", "radius"], dropna=False)
            .agg(accuracy=("accuracy_mean", "mean"), sigma=("noise_std", "mean"))
            .reset_index()
            .sort_values("radius")
        )
        fig, ax = plt.subplots()
        for mechanism, g in grouped.groupby("mechanism"):
            ax.plot(g["radius"], g["accuracy"], marker="o", label=str(mechanism))
        ax.set_xlabel("Ball radius r")
        ax.set_ylabel("mean node accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"Radius/utility ablation, {graph}, epsilon={eps:g}")
        ax.legend()
        savefig_stem(fig, out_dir / "figures" / f"accuracy_vs_radius_{graph}")
        plt.close(fig)

        fig, ax = plt.subplots()
        for mechanism, g in grouped.groupby("mechanism"):
            ax.plot(g["radius"], g["sigma"], marker="o", label=str(mechanism))
        ax.set_xlabel("Ball radius r")
        ax.set_ylabel("calibrated Gaussian noise std")
        ax.set_title(f"Noise calibration vs radius, {graph}, epsilon={eps:g}")
        ax.legend()
        savefig_stem(fig, out_dir / "figures" / f"sigma_vs_radius_{graph}")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    (
        dataset_tag,
        display_name,
        X_train,
        y_train,
        X_test,
        y_test,
        num_classes,
        feature_dim,
        radius_report,
    ) = load_data(args)
    radii, radius_report = parse_radius_grid(
        args.radius_grid, X_train, seed=int(args.radius_estimation_seed)
    )
    out_dir = ensure_dir(
        Path(args.results_root)
        / "paper3"
        / "decentralized_prototype_utility"
        / slugify(dataset_tag)
    )

    rows: list[dict] = []
    graph_metadata: dict[str, dict] = {}
    for graph in args.graphs:
        A = graph_adjacency(graph, int(args.num_nodes))
        W = metropolis_mixing_matrix(A, lazy=float(args.lazy))
        graph_metadata[str(graph)] = {
            "adjacency": A.astype(int).tolist(),
            "mixing_matrix": W.astype(float).tolist(),
        }
        for seed in args.release_seeds:
            baseline = run_noisy_prototype_gossip(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                W=W,
                rounds=int(args.rounds),
                num_classes=int(num_classes),
                clip_norm=float(args.clip_norm),
                noise_std=0.0,
                seed=int(seed),
            )
            rows.append(
                {
                    "dataset_tag": dataset_tag,
                    "graph": str(graph),
                    "mechanism": "none",
                    "radius": float("nan"),
                    "target_epsilon": float("inf"),
                    "calibrated_epsilon": 0.0,
                    "calibration_order_opt": float("nan"),
                    "noise_std": 0.0,
                    "sensitivity": 0.0,
                    "release_seed": int(seed),
                    "num_nodes": int(args.num_nodes),
                    "rounds": int(args.rounds),
                    **baseline,
                }
            )
        for radius in radii:
            for target_epsilon in args.epsilon_grid:
                for mechanism in args.mechanisms:
                    if mechanism == "ball":
                        sensitivity = min(
                            float(args.feature_lipschitz) * float(radius),
                            2.0 * float(args.clip_norm),
                        )
                    else:
                        sensitivity = 2.0 * float(args.clip_norm)
                    cal = mechanism_noise_for_target_dp(
                        target_epsilon=float(target_epsilon),
                        sensitivity=float(sensitivity),
                        orders=tuple(float(a) for a in args.orders),
                        delta=float(args.dp_delta),
                    )
                    for seed in args.release_seeds:
                        metrics = run_noisy_prototype_gossip(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            W=W,
                            rounds=int(args.rounds),
                            num_classes=int(num_classes),
                            clip_norm=float(args.clip_norm),
                            noise_std=float(cal["noise_std"]),
                            seed=int(seed),
                        )
                        rows.append(
                            {
                                "dataset_tag": dataset_tag,
                                "graph": str(graph),
                                "mechanism": str(mechanism),
                                "radius": float(radius),
                                "target_epsilon": float(target_epsilon),
                                "calibrated_epsilon": float(cal["epsilon"]),
                                "calibration_order_opt": float(cal["order_opt"]),
                                "noise_std": float(cal["noise_std"]),
                                "sensitivity": float(sensitivity),
                                "release_seed": int(seed),
                                "num_nodes": int(args.num_nodes),
                                "rounds": int(args.rounds),
                                **metrics,
                            }
                        )

    rows_path, summary_path = write_rows_and_summary(
        rows=rows,
        out_dir=out_dir,
        rows_name="prototype_utility_rows.csv",
        summary_name="prototype_utility_summary.csv",
        group_cols=["dataset_tag", "graph", "mechanism", "radius", "target_epsilon"],
        value_cols=[
            "accuracy_mean",
            "accuracy_min_node",
            "accuracy_std_node",
            "consensus_state_disagreement",
            "noise_std",
            "sensitivity",
        ],
    )
    df = pd.DataFrame(rows)
    if bool(args.make_plots):
        make_utility_plots(df, out_dir)
    write_json(
        out_dir / "metadata.json",
        {
            "experiment": "paper3_decentralized_prototype_utility",
            "dataset": display_name,
            "dataset_tag": dataset_tag,
            "feature_dim": int(feature_dim),
            "num_classes": int(num_classes),
            "train_examples": int(len(X_train)),
            "test_examples": int(len(X_test)),
            "radius_grid": [float(v) for v in radii],
            "radius_report": radius_report,
            "args": vars(args),
            "graphs": graph_metadata,
            "rows_path": str(rows_path),
            "summary_path": str(summary_path),
            "label_count_privacy_note": "Counts are treated as public; the protected contribution is the clipped feature vector conditional on label.",
        },
    )
    print(f"Wrote {rows_path} ({len(rows)} rows)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
