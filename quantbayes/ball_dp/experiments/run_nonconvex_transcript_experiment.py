#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from quantbayes.ball_dp.experiments.paper2_nonconvex_common import (
    DEFAULT_M_GRID,
    add_bound_args,
    add_embedding_loader_args,
    add_radius_args,
    add_theorem_model_args,
    add_training_args,
    as_float_list,
    as_int_list,
    calibrate_noise_multiplier_for_mechanism,
    compute_finite_exact_bounds,
    dataset_output_dir,
    ensure_dir,
    get_release_step_table,
    load_embeddings,
    make_dataset_metadata_row,
    make_theorem_spec_and_bounds,
    maybe_subsample_loaded_dataset,
    mechanism_list,
    rebuild_dataset_outputs,
    release_epsilon,
    release_utility_accuracy,
    resolve_dataset,
    run_id_for_payload,
    save_dataframe,
    savefig_stem,
    select_policy_radius,
    slugify,
    summarize_step_table,
    train_theorem_release,
    write_json_safe,
)
from quantbayes.ball_dp.theorem.registry import certified_lz


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.6),
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run official Paper 2 nonconvex transcript experiments: theorem-backed "
            "operator-norm tanh DP-SGD, Hayes/global transcript ReRo bounds, and utility."
        )
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--results-root", type=str, default="results_paper2_nonconvex")
    add_radius_args(parser)
    add_theorem_model_args(parser)
    add_training_args(parser)
    add_bound_args(parser)
    add_embedding_loader_args(parser)
    parser.add_argument(
        "--sweep",
        choices=("epsilon", "m", "both"),
        default="epsilon",
        help="Train across epsilon values and/or report multiple finite-prior sizes m.",
    )
    parser.add_argument("--fixed-epsilon", type=float, default=4.0)
    parser.add_argument("--fixed-m", type=int, default=8)
    parser.add_argument(
        "--save-step-tables",
        action="store_true",
        help="Save per-step sensitivity/accounting tables for each release.",
    )
    return parser.parse_args()


def selected_epsilons(args: argparse.Namespace) -> list[float]:
    if args.sweep in {"epsilon", "both"}:
        vals = sorted(set(as_float_list(args.epsilon_grid)))
    else:
        vals = [float(args.fixed_epsilon)]
    if not vals or any(v <= 0.0 for v in vals):
        raise ValueError("All epsilon values must be positive.")
    return vals


def selected_ms(args: argparse.Namespace) -> list[int]:
    if args.sweep in {"m", "both"}:
        vals = sorted(set(as_int_list(args.m_grid)))
    else:
        vals = [int(args.fixed_m)]
    if not vals or any(v < 2 for v in vals):
        raise ValueError("All finite-prior sizes m must be at least 2.")
    return vals


def _lineplot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    title: str,
    ylabel: str,
    out_stem: Path,
) -> None:
    if df.empty or x not in df or y not in df:
        return
    fig, ax = plt.subplots()
    for key, g in df.groupby(group, dropna=False):
        g2 = g.sort_values(x)
        xs = pd.to_numeric(g2[x], errors="coerce")
        ys = pd.to_numeric(g2[y], errors="coerce")
        mask = xs.notna() & ys.notna()
        if mask.any():
            ax.plot(xs[mask], ys[mask], marker="o", label=str(key))
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    savefig_stem(fig, out_stem)
    plt.close(fig)


def write_plots(dataset_dir: Path, raw: pd.DataFrame) -> None:
    if raw.empty:
        return
    plot_dir = ensure_dir(dataset_dir / "figures")
    grouped = (
        raw.groupby(["mechanism", "epsilon", "m"], dropna=False)
        .agg(
            accuracy=("accuracy", "mean"),
            primary_bound_hayes=("primary_bound_hayes", "mean"),
            bound_hayes_ball=("bound_hayes_ball", "mean"),
            bound_hayes_standard_same_noise=("bound_hayes_standard_same_noise", "mean"),
            mean_ball_to_standard_ratio=("mean_ball_to_standard_ratio", "mean"),
        )
        .reset_index()
    )
    _lineplot(
        grouped,
        x="epsilon",
        y="accuracy",
        group="mechanism",
        title="Paper 2 nonconvex utility vs epsilon",
        ylabel="public test accuracy",
        out_stem=plot_dir / "paper2_accuracy_vs_epsilon",
    )
    _lineplot(
        grouped,
        x="epsilon",
        y="primary_bound_hayes",
        group="mechanism",
        title="Finite-prior exact-identification Hayes bound vs epsilon",
        ylabel=r"$B_{1/m}^{test}$",
        out_stem=plot_dir / "paper2_hayes_bound_vs_epsilon",
    )
    _lineplot(
        grouped,
        x="epsilon",
        y="mean_ball_to_standard_ratio",
        group="mechanism",
        title="Mean Ball/standard per-step sensitivity ratio",
        ylabel=r"mean $\Delta_t(r)/(2C_t)$",
        out_stem=plot_dir / "paper2_sensitivity_ratio_vs_epsilon",
    )
    grouped_m = grouped.copy()
    grouped_m["mechanism_epsilon"] = (
        grouped_m["mechanism"].astype(str) + "/eps=" + grouped_m["epsilon"].astype(str)
    )
    _lineplot(
        grouped_m,
        x="m",
        y="primary_bound_hayes",
        group="mechanism_epsilon",
        title="Finite-prior Hayes bound vs support size",
        ylabel=r"$B_{1/m}^{test}$",
        out_stem=plot_dir / "paper2_hayes_bound_vs_m",
    )


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    ds_spec = resolve_dataset(args.dataset)
    eps_grid = selected_epsilons(args)
    m_grid = selected_ms(args)
    mechanisms = mechanism_list(str(args.mechanisms))

    data = maybe_subsample_loaded_dataset(load_embeddings(args, ds_spec), args)
    spec, bounds, spec_meta = make_theorem_spec_and_bounds(data, args)
    lz = certified_lz(spec, bounds)
    radius, radius_report = select_policy_radius(data, args)

    dataset_dir = dataset_output_dir(args.results_root, data.spec.tag)
    run_payload = {
        "script": "run_nonconvex_transcript_experiment.py",
        "dataset": data.spec.tag,
        "args": vars(args),
        "spec": spec_meta,
        "radius": radius,
        "lz": lz,
        "eps_grid": eps_grid,
        "m_grid": m_grid,
        "mechanisms": mechanisms,
    }
    run_id = run_id_for_payload(run_payload)
    run_dir = ensure_dir(dataset_dir / "runs" / run_id)
    write_json_safe(run_dir / "run_config.json", run_payload)
    write_json_safe(run_dir / "radius_report.json", radius_report)

    base_meta = make_dataset_metadata_row(
        data=data,
        radius=radius,
        radius_tag=str(args.radius),
        spec=spec,
        bounds=bounds,
        lz=float(lz),
        args=args,
    )

    rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []

    for epsilon in eps_grid:
        for mechanism in mechanisms:
            calibration = calibrate_noise_multiplier_for_mechanism(
                mechanism=mechanism,
                dataset_size=len(data.X_train),
                radius=radius,
                lz=float(lz),
                args=args,
                epsilon=float(epsilon),
            )
            noise_multiplier = float(calibration["noise_multiplier"])

            for seed in as_int_list(args.release_seeds):
                print(
                    f"=== {data.spec.display_name} | {mechanism} | eps={epsilon:g} | seed={seed} | noise_multiplier={noise_multiplier:.6g} ===",
                    flush=True,
                )
                release = train_theorem_release(
                    spec=spec,
                    bounds=bounds,
                    X_train=data.X_train,
                    y_train=data.y_train,
                    X_eval=data.X_test,
                    y_eval=data.y_test,
                    radius=radius,
                    lz=float(lz),
                    epsilon=float(epsilon),
                    noise_multiplier=noise_multiplier,
                    mechanism=mechanism,
                    seed=int(seed),
                    args=args,
                )

                step_rows = get_release_step_table(release)
                step_summary = summarize_step_table(step_rows)
                release_key_payload = {
                    "epsilon": epsilon,
                    "mechanism": mechanism,
                    "seed": int(seed),
                    "noise_multiplier": noise_multiplier,
                }
                release_key = run_id_for_payload(release_key_payload)
                if bool(args.save_step_tables):
                    for r in step_rows:
                        rr = dict(base_meta)
                        rr.update(r)
                        rr.update(
                            {
                                "run_id": run_id,
                                "release_key": release_key,
                                "mechanism": mechanism,
                                "epsilon": float(epsilon),
                                "release_seed": int(seed),
                                "noise_multiplier": noise_multiplier,
                            }
                        )
                        all_step_rows.append(rr)

                acc = release_utility_accuracy(release, data.X_test, data.y_test)
                eps_ball = release_epsilon(release, "ball")
                eps_std = release_epsilon(release, "standard")

                for m in m_grid:
                    bound_row = compute_finite_exact_bounds(
                        release,
                        m=int(m),
                        feature_dim=int(data.feature_dim),
                        include_rdp=bool(args.include_rdp),
                        include_composed_direct=bool(args.include_composed_direct),
                    )
                    primary_bound_hayes = (
                        bound_row.get("bound_hayes_ball")
                        if mechanism == "ball"
                        else bound_row.get("bound_hayes_standard_same_noise")
                    )
                    primary_bound_rdp = (
                        bound_row.get("bound_rdp_ball")
                        if mechanism == "ball"
                        else bound_row.get("bound_rdp_standard_same_noise")
                    )
                    row = dict(base_meta)
                    row.update(step_summary)
                    row.update(bound_row)
                    row.update(
                        {
                            "run_id": run_id,
                            "release_key": release_key,
                            "row_key": run_id_for_payload(
                                {
                                    **release_key_payload,
                                    "m": int(m),
                                    "dataset": data.spec.tag,
                                }
                            ),
                            "mechanism": mechanism,
                            "epsilon": float(epsilon),
                            "m": int(m),
                            "release_seed": int(seed),
                            "noise_multiplier": noise_multiplier,
                            "calibrated_accountant_epsilon": float(
                                calibration.get("epsilon", math.nan)
                            ),
                            "actual_epsilon_ball": eps_ball,
                            "actual_epsilon_standard": eps_std,
                            "primary_epsilon": (
                                eps_ball if mechanism == "ball" else eps_std
                            ),
                            "accuracy": acc,
                            "public_eval_loss": float(
                                release.utility_metrics.get(
                                    "public_eval_loss", math.nan
                                )
                            ),
                            "primary_bound_hayes": (
                                None
                                if primary_bound_hayes is None
                                else float(primary_bound_hayes)
                            ),
                            "primary_bound_rdp": (
                                None
                                if primary_bound_rdp is None
                                else float(primary_bound_rdp)
                            ),
                            "release_kind": str(release.release_kind),
                            "ball_regime": str(
                                release.extra.get("ball_regime", "unknown")
                            ),
                        }
                    )
                    rows.append(row)

    raw_df = pd.DataFrame(rows)
    save_dataframe(
        raw_df,
        run_dir / "nonconvex_transcript_seed_rows.csv",
        save_parquet_if_possible=False,
    )
    if all_step_rows:
        save_dataframe(
            pd.DataFrame(all_step_rows),
            run_dir / "nonconvex_transcript_step_rows.csv",
            save_parquet_if_possible=False,
        )

    rebuild_dataset_outputs(
        dataset_dir,
        seed_filename="nonconvex_transcript_seed_rows.csv",
        summary_filename="nonconvex_transcript_summary.csv",
    )
    dataset_raw_path = dataset_dir / "nonconvex_transcript_seed_rows.csv"
    if dataset_raw_path.exists():
        write_plots(dataset_dir, pd.read_csv(dataset_raw_path))

    print(f"Wrote {len(raw_df)} rows to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
