#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from quantbayes.ball_dp.api import (
    attack_nonconvex_ball_trace_finite_prior,
    make_trace_metadata_from_release,
)
from quantbayes.ball_dp.attacks.ball_policy import BallTraceMapAttackConfig
from quantbayes.ball_dp.attacks.gradient_based import (
    DPSGDTraceRecorder,
    subtract_known_batch_gradients,
)
from quantbayes.ball_dp.experiments.paper2_nonconvex_common import (
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
    find_feasible_support_banks,
    finite_prior_hash,
    get_release_step_table,
    load_embeddings,
    make_dataset_metadata_row,
    make_support_set,
    make_theorem_spec_and_bounds,
    maybe_subsample_loaded_dataset,
    mechanism_list,
    release_epsilon,
    release_utility_accuracy,
    resolve_dataset,
    run_id_for_payload,
    save_dataframe,
    savefig_stem,
    select_policy_radius,
    slugify,
    summarize_step_table,
    support_source_hash,
    train_theorem_release,
    write_json_safe,
)
from quantbayes.ball_dp.theorem.registry import certified_lz
from quantbayes.ball_dp.types import ArrayDataset, Record


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
            "Run official Paper 2 theorem-aligned finite-prior attacks against "
            "nonconvex DP-SGD transcripts."
        )
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument(
        "--results-root", type=str, default="results_paper2_nonconvex_attack"
    )
    add_radius_args(parser)
    add_theorem_model_args(parser)
    add_training_args(parser)
    add_bound_args(parser)
    add_embedding_loader_args(parser)

    # Attack runs default to the Paper-1-style high-utility privacy point,
    # but the shared --epsilon-grid argument from add_training_args remains user-overridable.
    parser.set_defaults(epsilon_grid=[8.0])
    parser.add_argument("--fixed-m", type=int, default=8)
    parser.add_argument(
        "--support-source",
        choices=("public_only", "train_and_public"),
        default="public_only",
    )
    parser.add_argument(
        "--support-selection",
        choices=("random", "farthest", "nearest"),
        default="farthest",
    )
    parser.add_argument(
        "--anchor-selection",
        choices=("random", "rare_class", "large_bank"),
        default="rare_class",
    )
    parser.add_argument("--num-supports", type=int, default=4)
    parser.add_argument("--support-draws", type=int, default=2)
    parser.add_argument("--targets-per-support", type=int, default=1)
    parser.add_argument("--anchor-seed", type=int, default=0)
    parser.add_argument("--max-feasible-search", type=int, default=5000)
    parser.add_argument("--anchor-indices", nargs="*", type=int, default=None)
    parser.add_argument("--strict-feasible-supports", action="store_true")
    parser.add_argument(
        "--attack-modes",
        nargs="+",
        choices=("known_inclusion", "unknown_inclusion"),
        default=["known_inclusion", "unknown_inclusion"],
    )
    parser.add_argument(
        "--known-step-mode",
        choices=("all", "present_steps"),
        default="all",
        help="Use 'all' for the KI oracle so absent target steps are correctly skipped as constants.",
    )
    parser.add_argument("--trace-capture-every", type=int, default=1)
    parser.add_argument("--attack-seed", type=int, default=0)
    return parser.parse_args()


def choose_target_positions(
    *, m: int, count: int, seed_parts: Sequence[int]
) -> list[int]:
    m = int(m)
    count = max(1, int(count))
    rng = np.random.default_rng(np.random.SeedSequence([int(v) for v in seed_parts]))
    if count >= m:
        return list(range(m))
    return [
        int(v) for v in rng.choice(np.arange(m), size=count, replace=False).tolist()
    ]


def replace_target_record(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_index: int,
    target_x: np.ndarray,
    target_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X_train, dtype=np.float32).copy()
    y = np.asarray(y_train, dtype=np.int32).copy()
    X[int(target_index)] = np.asarray(target_x, dtype=np.float32)
    y[int(target_index)] = int(target_y)
    return X, y


def rebuild_attack_outputs(dataset_dir: Path) -> None:
    frames: list[pd.DataFrame] = []
    for path in sorted(
        (dataset_dir / "runs").glob("*/nonconvex_transcript_attack_rows.csv")
    ):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            pass
    if not frames:
        return
    raw = pd.concat(frames, ignore_index=True)
    if "row_key" in raw.columns:
        raw = raw.drop_duplicates(subset=["row_key"], keep="last")
    save_dataframe(
        raw,
        dataset_dir / "nonconvex_transcript_attack_rows.csv",
        save_parquet_if_possible=False,
    )

    group_cols = [
        c
        for c in [
            "dataset_tag",
            "dataset_name",
            "mechanism",
            "epsilon",
            "m",
            "attack_mode",
            "radius_tag",
            "task",
            "num_steps",
            "batch_size",
            "clip_norm",
        ]
        if c in raw.columns
    ]
    rows = []
    for key, g in raw.groupby(group_cols, dropna=False):
        row = {}
        if not isinstance(key, tuple):
            key = (key,)
        for col, val in zip(group_cols, key, strict=True):
            row[col] = val
        row["n_trials"] = int(len(g))
        for col in [
            "exact_identification_success",
            "prior_exact_hit",
            "prior_rank",
            "prior_hit@5",
            "mse",
            "feature_l2",
            "primary_bound_hayes",
            "bound_hayes_ball",
            "bound_hayes_standard_same_noise",
            "accuracy",
            "selected_step_count",
        ]:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = (
                    float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
                )
        rows.append(row)
    summary = pd.DataFrame(rows)
    save_dataframe(
        summary,
        dataset_dir / "nonconvex_transcript_attack_summary.csv",
        save_parquet_if_possible=False,
    )


def write_attack_plots(dataset_dir: Path) -> None:
    raw_path = dataset_dir / "nonconvex_transcript_attack_rows.csv"
    if not raw_path.exists():
        return
    raw = pd.read_csv(raw_path)
    if raw.empty:
        return
    plot_dir = ensure_dir(dataset_dir / "figures")
    grouped = (
        raw.groupby(["attack_mode", "mechanism", "epsilon"], dropna=False)
        .agg(
            success=("exact_identification_success", "mean"),
            bound=("primary_bound_hayes", "mean"),
            prior_rank=("prior_rank", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots()
    for (attack_mode, mechanism), g in grouped.groupby(
        ["attack_mode", "mechanism"], dropna=False
    ):
        g = g.sort_values("epsilon")
        ax.plot(
            g["epsilon"], g["success"], marker="o", label=f"{mechanism}/{attack_mode}"
        )
        ax.plot(
            g["epsilon"],
            g["bound"],
            linestyle="--",
            marker="x",
            label=f"bound {mechanism}/{attack_mode}",
        )
    ax.set_xlabel("epsilon")
    ax.set_ylabel("exact identification probability")
    ax.set_title("Paper 2 transcript MAP attacks vs Hayes bound")
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig_stem(fig, plot_dir / "paper2_attack_success_vs_bound")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    if int(args.fixed_m) < 2:
        raise ValueError("fixed_m must be at least 2.")
    if int(args.trace_capture_every) <= 0:
        raise ValueError("trace_capture_every must be positive.")

    ds_spec = resolve_dataset(args.dataset)
    eps_grid = sorted(set(as_float_list(args.epsilon_grid)))
    mechanisms = mechanism_list(str(args.mechanisms))

    data = maybe_subsample_loaded_dataset(load_embeddings(args, ds_spec), args)
    spec, bounds, spec_meta = make_theorem_spec_and_bounds(data, args)
    lz = certified_lz(spec, bounds)
    radius, radius_report = select_policy_radius(data, args)

    banks = find_feasible_support_banks(
        data,
        radius_value=float(radius),
        max_required_m=int(args.fixed_m),
        num_supports=int(args.num_supports),
        anchor_seed=int(args.anchor_seed),
        source_mode=str(args.support_source),
        max_search=(
            None if args.max_feasible_search is None else int(args.max_feasible_search)
        ),
        explicit_anchor_indices=args.anchor_indices,
        strict=bool(args.strict_feasible_supports),
        anchor_selection=str(args.anchor_selection),
    )

    dataset_dir = dataset_output_dir(args.results_root, data.spec.tag)
    run_payload = {
        "script": "run_nonconvex_transcript_attack_experiment.py",
        "dataset": data.spec.tag,
        "args": vars(args),
        "spec": spec_meta,
        "radius": radius,
        "lz": lz,
        "eps_grid": eps_grid,
        "mechanisms": mechanisms,
        "anchor_indices": [int(b.anchor_index) for b in banks],
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

            for bank_i, bank in enumerate(banks):
                for support_draw in range(int(args.support_draws)):
                    X_support, y_support, source_ids, support_dists = make_support_set(
                        bank,
                        m=int(args.fixed_m),
                        draw_index=int(support_draw),
                        base_seed=int(args.anchor_seed),
                        selection=str(args.support_selection),
                    )
                    target_positions = choose_target_positions(
                        m=int(args.fixed_m),
                        count=int(args.targets_per_support),
                        seed_parts=(
                            args.anchor_seed,
                            bank.anchor_index,
                            support_draw,
                            99173,
                        ),
                    )

                    for target_pos in target_positions:
                        true_x = np.asarray(
                            X_support[int(target_pos)], dtype=np.float32
                        )
                        true_y = int(y_support[int(target_pos)])
                        X_trial, y_trial = replace_target_record(
                            data.X_train,
                            data.y_train,
                            target_index=int(bank.anchor_index),
                            target_x=true_x,
                            target_y=true_y,
                        )
                        true_record = Record(features=true_x, label=true_y)

                        support_key_payload = {
                            "anchor_index": int(bank.anchor_index),
                            "support_draw": int(support_draw),
                            "target_pos": int(target_pos),
                            "support_hash": finite_prior_hash(X_support, y_support),
                            "source_hash": support_source_hash(source_ids),
                        }

                        for seed in as_int_list(args.release_seeds):
                            print(
                                f"=== {data.spec.display_name} | {mechanism} | eps={epsilon:g} | seed={seed} | anchor={bank.anchor_index} | draw={support_draw} | target_pos={target_pos} ===",
                                flush=True,
                            )
                            recorder = DPSGDTraceRecorder(
                                capture_every=int(args.trace_capture_every),
                                keep_models=True,
                                keep_batch_indices=True,
                            )
                            release = train_theorem_release(
                                spec=spec,
                                bounds=bounds,
                                X_train=X_trial,
                                y_train=y_trial,
                                X_eval=data.X_test,
                                y_eval=data.y_test,
                                radius=radius,
                                lz=float(lz),
                                epsilon=float(epsilon),
                                noise_multiplier=noise_multiplier,
                                mechanism=mechanism,
                                seed=int(seed),
                                args=args,
                                trace_recorder=recorder,
                            )

                            trace_meta = make_trace_metadata_from_release(
                                release,
                                target_index=int(bank.anchor_index),
                                extra={
                                    "anchor_index": int(bank.anchor_index),
                                    "anchor_label": int(bank.anchor_label),
                                    "support_draw": int(support_draw),
                                    "target_position": int(target_pos),
                                    "support_source": str(args.support_source),
                                    "support_selection": str(args.support_selection),
                                },
                            )
                            trace = recorder.to_trace(
                                state=release.extra.get("model_state", None),
                                loss_name=spec.loss_name,
                                reduction=str(trace_meta.get("reduction", "mean")),
                                metadata=trace_meta,
                            )
                            residual_trace = subtract_known_batch_gradients(
                                trace,
                                ArrayDataset(X_trial, y_trial, name="trial_train"),
                                target_index=int(bank.anchor_index),
                                loss_name=spec.loss_name,
                                seed=int(args.attack_seed),
                            )

                            step_rows = get_release_step_table(release)
                            step_summary = summarize_step_table(step_rows)
                            bound_row = compute_finite_exact_bounds(
                                release,
                                m=int(args.fixed_m),
                                feature_dim=int(data.feature_dim),
                                include_rdp=bool(args.include_rdp),
                                include_composed_direct=bool(
                                    args.include_composed_direct
                                ),
                            )
                            primary_bound_hayes = (
                                bound_row.get("bound_hayes_ball")
                                if mechanism == "ball"
                                else bound_row.get("bound_hayes_standard_same_noise")
                            )
                            eps_ball = release_epsilon(release, "ball")
                            eps_std = release_epsilon(release, "standard")
                            acc = release_utility_accuracy(
                                release, data.X_test, data.y_test
                            )

                            for attack_mode in args.attack_modes:
                                cfg = BallTraceMapAttackConfig(
                                    mode=str(attack_mode),
                                    step_mode=(
                                        str(args.known_step_mode)
                                        if str(attack_mode) == "known_inclusion"
                                        else "all"
                                    ),
                                    seed=int(args.attack_seed),
                                )
                                attack_status = "ok"
                                attack_metrics: dict[str, float] = {}
                                attack_diag: dict[str, Any] = {}
                                try:
                                    attack = attack_nonconvex_ball_trace_finite_prior(
                                        residual_trace,
                                        X_support,
                                        y_support,
                                        cfg=cfg,
                                        target_index=int(bank.anchor_index),
                                        known_label=int(bank.anchor_label),
                                        true_record=true_record,
                                        eta_grid=(0.5,),
                                    )
                                    attack_status = str(attack.status)
                                    attack_metrics = dict(attack.metrics)
                                    attack_diag = dict(attack.diagnostics)
                                except Exception as exc:
                                    attack_status = f"error: {exc!r}"

                                row = dict(base_meta)
                                row.update(step_summary)
                                row.update(bound_row)
                                row.update(
                                    {
                                        "run_id": run_id,
                                        "row_key": run_id_for_payload(
                                            {
                                                "dataset": data.spec.tag,
                                                "epsilon": epsilon,
                                                "mechanism": mechanism,
                                                "release_seed": int(seed),
                                                "attack_mode": str(attack_mode),
                                                **support_key_payload,
                                            }
                                        ),
                                        "mechanism": mechanism,
                                        "epsilon": float(epsilon),
                                        "m": int(args.fixed_m),
                                        "release_seed": int(seed),
                                        "noise_multiplier": noise_multiplier,
                                        "actual_epsilon_ball": eps_ball,
                                        "actual_epsilon_standard": eps_std,
                                        "primary_epsilon": (
                                            eps_ball if mechanism == "ball" else eps_std
                                        ),
                                        "primary_bound_hayes": (
                                            None
                                            if primary_bound_hayes is None
                                            else float(primary_bound_hayes)
                                        ),
                                        "accuracy": acc,
                                        "attack_mode": str(attack_mode),
                                        "attack_status": attack_status,
                                        "anchor_index": int(bank.anchor_index),
                                        "anchor_label": int(bank.anchor_label),
                                        "support_draw": int(support_draw),
                                        "target_position": int(target_pos),
                                        "target_source_id": str(
                                            source_ids[int(target_pos)]
                                        ),
                                        "support_hash": str(
                                            support_key_payload["support_hash"]
                                        ),
                                        "support_source_hash": str(
                                            support_key_payload["source_hash"]
                                        ),
                                        "support_max_distance_to_anchor": float(
                                            np.max(support_dists)
                                        ),
                                        "support_mean_distance_to_anchor": float(
                                            np.mean(support_dists)
                                        ),
                                        "selected_step_count": float(
                                            attack_diag.get(
                                                "selected_step_count", math.nan
                                            )
                                        ),
                                        "predicted_prior_index": attack_diag.get(
                                            "predicted_prior_index", None
                                        ),
                                        "true_prior_index": attack_diag.get(
                                            "true_prior_index", None
                                        ),
                                        "exact_identification_success": float(
                                            attack_metrics.get(
                                                "exact_identification_success", math.nan
                                            )
                                        ),
                                        "prior_exact_hit": float(
                                            attack_metrics.get(
                                                "prior_exact_hit", math.nan
                                            )
                                        ),
                                        "prior_rank": float(
                                            attack_metrics.get("prior_rank", math.nan)
                                        ),
                                        "prior_hit@1": float(
                                            attack_metrics.get("prior_hit@1", math.nan)
                                        ),
                                        "prior_hit@5": float(
                                            attack_metrics.get("prior_hit@5", math.nan)
                                        ),
                                        "prior_hit@10": float(
                                            attack_metrics.get("prior_hit@10", math.nan)
                                        ),
                                        "mse": float(
                                            attack_metrics.get("mse", math.nan)
                                        ),
                                        "feature_l2": float(
                                            attack_metrics.get(
                                                "feature_l2",
                                                attack_metrics.get(
                                                    "distance", math.nan
                                                ),
                                            )
                                        ),
                                        "oblivious_kappa_attack": float(
                                            attack_metrics.get(
                                                "oblivious_kappa",
                                                1.0 / float(args.fixed_m),
                                            )
                                        ),
                                        "trace_steps_recorded": int(len(trace.steps)),
                                        "trace_capture_every": int(
                                            args.trace_capture_every
                                        ),
                                    }
                                )
                                rows.append(row)

    raw_df = pd.DataFrame(rows)
    save_dataframe(
        raw_df,
        run_dir / "nonconvex_transcript_attack_rows.csv",
        save_parquet_if_possible=False,
    )
    rebuild_attack_outputs(dataset_dir)
    write_attack_plots(dataset_dir)
    print(f"Wrote {len(raw_df)} attack rows to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
