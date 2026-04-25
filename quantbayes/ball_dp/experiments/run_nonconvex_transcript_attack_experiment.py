#!/usr/bin/env python3
"""Official Paper 2 finite-prior transcript attack experiment.

For each Ball center u, this script builds a same-label finite support S inside the
policy ball, replaces u by each candidate target z in S, trains a theorem-backed
operator-norm DP-SGD release while recording the sanitized transcript, residualizes
known non-target batch contributions, and evaluates the exact finite-prior
Bayesian transcript attacks from Paper 2:

  * KI: known-inclusion oracle MAP attack;
  * UI: unknown-inclusion MAP attack with Bernoulli inclusion marginalized out.

The reported theorem bound is the direct Hayes-style product-reference Ball-ReRo
bound evaluated at kappa=1/m.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import jax.random as jr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from quantbayes.ball_dp import (
    ArrayDataset,
    attack_nonconvex_ball_trace_finite_prior,
    make_trace_metadata_from_release,
    summarize_embedding_ball_radii,
)
from quantbayes.ball_dp.attacks import BallTraceMapAttackConfig, DPSGDTraceRecorder
from quantbayes.ball_dp.attacks.gradient_based import subtract_known_batch_gradients
from quantbayes.ball_dp.experiments.run_attack_experiment import (
    DEFAULT_DELTA,
    DEFAULT_EPS_GRID,
    DEFAULT_FIXED_EPSILON,
    DEFAULT_FIXED_M,
    DEFAULT_M_GRID,
    DEFAULT_ORDERS,
    RADIUS_TAG_TO_QUANTILE,
    LoadedDataset,
    as_float_list,
    as_int_list,
    configure_matplotlib,
    ensure_dir,
    find_feasible_support_banks,
    load_embeddings,
    make_support_set,
    radius_value_from_report,
    remove_train_index,
    resolve_dataset,
    savefig_stem,
    standard_radius_value_from_report,
    support_source_hash,
    write_json,
)
from quantbayes.ball_dp.serialization import save_dataframe
from quantbayes.ball_dp.theorem import (
    TheoremBounds,
    TheoremModelSpec,
    TrainConfig,
    certified_lz,
    fit_release,
    make_model,
)
from quantbayes.ball_dp.experiments.run_nonconvex_transcript_experiment import (
    DEFAULT_A,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CALIBRATION_STEPS,
    DEFAULT_CLIP_NORM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_INPUT_BOUND_MARGIN,
    DEFAULT_LAMBDA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_TEST_EXAMPLES,
    DEFAULT_MAX_TRAIN_EXAMPLES,
    compute_hayes_bounds,
    calibrate_noise,
    maybe_float,
    model_tag,
    resolve_task,
    row_key,
    stratified_subsample,
    summarize_mean_std,
)

DEFAULT_ATTACK_NUM_STEPS = 200
DEFAULT_TRACE_CAPTURE_EVERY = 1


def finite_or_nan(value: Any) -> float:
    out = maybe_float(value)
    return float("nan") if out is None else float(out)


def summarize_bernoulli(p: float, n: int) -> tuple[float, float]:
    p = float(p)
    n = int(n)
    if n <= 0 or not math.isfinite(p):
        return float("nan"), float("nan")
    se = math.sqrt(max(p * (1.0 - p), 0.0) / float(n))
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def actual_epsilon(release: Any, view: str) -> Optional[float]:
    ledger = release.privacy.ball if view == "ball" else release.privacy.standard
    if not ledger.dp_certificates:
        return None
    return maybe_float(ledger.dp_certificates[0].epsilon)


def make_subsampled_loaded_dataset(
    data: LoadedDataset,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> LoadedDataset:
    empirical_bound = float(np.max(np.linalg.norm(np.asarray(X_train), axis=1)))
    return LoadedDataset(
        spec=data.spec,
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int32),
        X_test=np.asarray(X_test, dtype=np.float32),
        y_test=np.asarray(y_test, dtype=np.int32),
        label_values=data.label_values,
        num_classes=int(len(np.unique(y_train))),
        feature_dim=int(X_train.shape[1]),
        empirical_embedding_bound=empirical_bound,
        backend=data.backend,
    )


def train_recorded_release(
    *,
    spec: TheoremModelSpec,
    bounds: TheoremBounds,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    mechanism: str,
    radius: float,
    epsilon: float,
    delta: float,
    noise_multiplier: float,
    num_steps: int,
    batch_size: int,
    clip_norm: float,
    learning_rate: float,
    eval_every: int,
    eval_batch_size: int,
    checkpoint_selection: str,
    release_seed: int,
    orders: Sequence[int | float],
    trace_capture_every: int,
    record_operator_norms: bool,
    operator_norms_every: int,
) -> tuple[Any, Any]:
    privacy = "ball_dp" if mechanism == "ball" else "standard_dp"
    model = make_model(
        spec,
        key=jr.PRNGKey(100_000 + int(release_seed)),
        init_project=True,
        bounds=bounds,
    )
    recorder = DPSGDTraceRecorder(
        capture_every=int(trace_capture_every),
        keep_models=True,
        keep_batch_indices=True,
    )
    train_cfg = TrainConfig(
        radius=float(radius),
        privacy=privacy,
        epsilon=float(epsilon),
        delta=float(delta),
        num_steps=int(num_steps),
        batch_size=int(batch_size),
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
        clip_norm=float(clip_norm),
        noise_multiplier=float(noise_multiplier),
        learning_rate=float(learning_rate),
        checkpoint_selection=str(checkpoint_selection),
        eval_every=int(eval_every),
        eval_batch_size=int(eval_batch_size),
        normalize_noisy_sum_by="batch_size",
        seed=int(release_seed),
    )
    release = fit_release(
        model,
        spec,
        bounds,
        X_train,
        y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        train_cfg=train_cfg,
        key=jr.PRNGKey(int(release_seed)),
        orders=tuple(int(a) for a in orders),
        trace_recorder=recorder,
        record_operator_norms=bool(record_operator_norms),
        operator_norms_every=int(operator_norms_every),
    )
    return release, recorder


def run_one_attack(
    *,
    residual_trace: Any,
    X_support: np.ndarray,
    y_support: np.ndarray,
    target_index: int,
    known_label: int,
    true_record: Any,
    mode: str,
    attack_seed: int,
    known_inclusion_step_mode: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    cfg = BallTraceMapAttackConfig(
        mode=mode,
        step_mode=(known_inclusion_step_mode if mode == "known_inclusion" else "all"),
        seed=int(attack_seed),
    )
    try:
        attack = attack_nonconvex_ball_trace_finite_prior(
            residual_trace,
            X_support,
            y_support,
            cfg=cfg,
            target_index=int(target_index),
            loss_name=residual_trace.loss_name,
            known_label=int(known_label),
            true_record=true_record,
            eta_grid=(0.5,),
        )
        return "ok", dict(attack.metrics), dict(attack.diagnostics)
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}", {}, {}


def target_positions_for_support(
    *,
    m: int,
    max_targets: Optional[int],
    seed: int,
    anchor_index: int,
    draw_index: int,
) -> list[int]:
    positions = np.arange(int(m), dtype=np.int64)
    if max_targets is None or int(max_targets) <= 0 or int(max_targets) >= int(m):
        return positions.tolist()
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), int(anchor_index), int(draw_index), int(m), 117]
        )
    )
    chosen = rng.choice(positions, size=int(max_targets), replace=False)
    return sorted(int(v) for v in chosen.tolist())


def rebuild_outputs(dataset_dir: Path) -> None:
    run_dirs = sorted((dataset_dir / "runs").glob("*"))
    trial_frames: list[pd.DataFrame] = []
    release_frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        trial_path = run_dir / "attack_trial_rows.csv"
        release_path = run_dir / "release_rows.csv"
        if trial_path.exists():
            trial_frames.append(pd.read_csv(trial_path))
        if release_path.exists():
            release_frames.append(pd.read_csv(release_path))
    if not trial_frames:
        return

    trials = pd.concat(trial_frames, ignore_index=True)
    if "trial_key" in trials.columns:
        trials = trials.drop_duplicates(subset=["trial_key"], keep="last")
    save_dataframe(
        trials, dataset_dir / "attack_trial_rows.csv", save_parquet_if_possible=False
    )

    releases = pd.DataFrame()
    if release_frames:
        releases = pd.concat(release_frames, ignore_index=True)
        if "release_key" in releases.columns:
            releases = releases.drop_duplicates(subset=["release_key"], keep="last")
        save_dataframe(
            releases, dataset_dir / "release_rows.csv", save_parquet_if_possible=False
        )

    config_cols = [
        "dataset_tag",
        "dataset_name",
        "model_tag",
        "task",
        "mechanism",
        "attack_mode",
        "radius_tag",
        "epsilon",
        "m",
        "delta",
        "num_steps",
        "batch_size",
        "clip_norm",
        "hidden_dim",
        "A",
        "Lambda",
    ]
    grouped = trials.groupby(config_cols, dropna=False)
    rows: list[dict[str, Any]] = []
    for key, frame in grouped:
        row = {col: val for col, val in zip(config_cols, key, strict=True)}
        ok = frame[frame["attack_status"].astype(str) == "ok"]
        row["n_trials"] = int(len(frame))
        row["n_ok_trials"] = int(len(ok))
        row["n_runs"] = int(frame["release_seed"].nunique())
        for col in [
            "exact_identification_success",
            "prior_rank",
            "prior_hit@5",
            "target_inclusion_count",
            "selected_step_count",
            "bound_hayes_ball",
            "bound_hayes_standard",
            "accuracy",
            "noise_multiplier",
        ]:
            if col in ok.columns and len(ok):
                mean, std, lo, hi = summarize_mean_std(ok[col].astype(float).tolist())
            else:
                mean = std = lo = hi = float("nan")
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
        if len(ok):
            lo, hi = summarize_bernoulli(
                float(ok["exact_identification_success"].mean()), len(ok)
            )
            row["exact_id_bernoulli_ci_low"] = lo
            row["exact_id_bernoulli_ci_high"] = hi
        else:
            row["exact_id_bernoulli_ci_low"] = float("nan")
            row["exact_id_bernoulli_ci_high"] = float("nan")
        rows.append(row)

    summary = (
        pd.DataFrame(rows)
        .sort_values(["mechanism", "attack_mode", "epsilon", "m"])
        .reset_index(drop=True)
    )
    save_dataframe(
        summary, dataset_dir / "attack_summary.csv", save_parquet_if_possible=False
    )

    figures_dir = ensure_dir(dataset_dir / "figures")
    if "exact_identification_success_mean" in summary.columns:
        for m_val in sorted(summary["m"].dropna().unique().tolist()):
            sub = summary[summary["m"] == m_val].copy()
            if sub.empty:
                continue
            fig, ax = plt.subplots()
            for mechanism, attack_mode, marker in [
                ("ball", "known_inclusion", "o"),
                ("ball", "unknown_inclusion", "s"),
                ("standard", "known_inclusion", "^"),
                ("standard", "unknown_inclusion", "D"),
            ]:
                mm = sub[
                    (sub["mechanism"] == mechanism)
                    & (sub["attack_mode"] == attack_mode)
                ].sort_values("epsilon")
                if mm.empty:
                    continue
                label = f"{mechanism} · {'KI' if attack_mode == 'known_inclusion' else 'UI'}"
                ax.plot(
                    mm["epsilon"],
                    mm["exact_identification_success_mean"],
                    marker=marker,
                    label=label,
                )
                ax.fill_between(
                    mm["epsilon"],
                    mm["exact_id_bernoulli_ci_low"],
                    mm["exact_id_bernoulli_ci_high"],
                    alpha=0.15,
                )
            ax.axhline(
                1.0 / float(m_val),
                linestyle="--",
                linewidth=1.8,
                label=f"baseline 1/{int(m_val)}",
            )
            ax.set_xscale("log", base=2)
            ax.set_xlabel("$\\varepsilon$ target")
            ax.set_ylabel("Exact-ID success")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"Transcript finite-prior attack · m={int(m_val)}")
            ax.legend()
            savefig_stem(
                fig, figures_dir / f"fig_attack_exactid_vs_epsilon_m{int(m_val)}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official Paper 2 theorem-aligned nonconvex transcript attack experiments."
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--results-root", type=str, default="results_paper2")
    parser.add_argument(
        "--task", choices=("auto", "binary", "multiclass"), default="auto"
    )
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--A", type=float, default=DEFAULT_A)
    parser.add_argument("--Lambda", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument(
        "--input-bound-margin", type=float, default=DEFAULT_INPUT_BOUND_MARGIN
    )
    parser.add_argument(
        "--radius", choices=tuple(RADIUS_TAG_TO_QUANTILE), default="q80"
    )
    parser.add_argument(
        "--standard-radius-source",
        choices=("empirical_diameter", "embedding_bound"),
        default="empirical_diameter",
    )
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument(
        "--epsilon-grid", nargs="+", type=float, default=list(DEFAULT_EPS_GRID)
    )
    parser.add_argument("--m-grid", nargs="+", type=int, default=list(DEFAULT_M_GRID))
    parser.add_argument("--fixed-epsilon", type=float, default=DEFAULT_FIXED_EPSILON)
    parser.add_argument("--fixed-m", type=int, default=DEFAULT_FIXED_M)
    parser.add_argument("--sweep", choices=("epsilon", "m", "both"), default="epsilon")
    parser.add_argument(
        "--mechanisms", choices=("ball", "standard", "both"), default="both"
    )
    parser.add_argument("--release-seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--num-steps", type=int, default=DEFAULT_ATTACK_NUM_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--clip-norm", type=float, default=DEFAULT_CLIP_NORM)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("last", "best_public_eval_loss", "best_public_eval_accuracy"),
        default="last",
    )
    parser.add_argument(
        "--trace-capture-every", type=int, default=DEFAULT_TRACE_CAPTURE_EVERY
    )
    parser.add_argument(
        "--known-inclusion-step-mode",
        choices=("all", "present_steps"),
        default="present_steps",
    )
    parser.add_argument(
        "--attack-modes",
        nargs="+",
        choices=("known_inclusion", "unknown_inclusion"),
        default=["known_inclusion", "unknown_inclusion"],
    )
    parser.add_argument("--attack-seed", type=int, default=0)
    parser.add_argument(
        "--max-train-examples", type=int, default=DEFAULT_MAX_TRAIN_EXAMPLES
    )
    parser.add_argument(
        "--max-test-examples", type=int, default=DEFAULT_MAX_TEST_EXAMPLES
    )
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument(
        "--calibration-bisection-steps", type=int, default=DEFAULT_CALIBRATION_STEPS
    )
    parser.add_argument("--include-rdp", action="store_true")
    parser.add_argument("--record-operator-norms", action="store_true")
    parser.add_argument("--operator-norms-every", type=int, default=100)

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
    parser.add_argument("--num-supports", type=int, default=2)
    parser.add_argument("--support-draws", type=int, default=1)
    parser.add_argument("--anchor-seed", type=int, default=0)
    parser.add_argument("--max-feasible-search", type=int, default=5000)
    parser.add_argument("--anchor-indices", nargs="*", type=int, default=None)
    parser.add_argument("--strict-feasible-supports", action="store_true")
    parser.add_argument("--max-targets-per-support", type=int, default=0)

    # Embedding-loader arguments shared with Paper 1.
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--embedding-cache-path", type=str, default=None)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument(
        "--encoder-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-exact-pairs", type=int, default=250_000)
    parser.add_argument("--max-sampled-pairs", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()
    ds_spec = resolve_dataset(args.dataset)
    loaded = load_embeddings(args, ds_spec)
    X_train, y_train = stratified_subsample(
        loaded.X_train,
        loaded.y_train,
        max_examples=args.max_train_examples,
        seed=int(args.subsample_seed),
    )
    X_test, y_test = stratified_subsample(
        loaded.X_test,
        loaded.y_test,
        max_examples=args.max_test_examples,
        seed=int(args.subsample_seed) + 17,
    )
    data = make_subsampled_loaded_dataset(
        loaded,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    num_classes = int(data.num_classes)
    feature_dim = int(data.feature_dim)
    task = resolve_task(args.task, num_classes)
    B_all = float(
        max(
            np.max(np.linalg.norm(data.X_train, axis=1)),
            np.max(np.linalg.norm(data.X_test, axis=1)),
        )
        * float(args.input_bound_margin)
    )
    bounds = TheoremBounds(B=B_all, A=float(args.A), Lambda=float(args.Lambda))
    spec = TheoremModelSpec(
        d_in=feature_dim,
        hidden_dim=int(args.hidden_dim),
        task=task,
        parameterization="dense",
        constraint="op",
        num_classes=(None if task == "binary" else int(num_classes)),
    )
    lz = certified_lz(spec, bounds)
    tag = model_tag(task=task, hidden_dim=args.hidden_dim, A=args.A, Lambda=args.Lambda)

    radius_report = summarize_embedding_ball_radii(
        data.X_train,
        data.y_train,
        quantiles=(0.50, 0.80, 0.95),
        max_exact_pairs=int(args.max_exact_pairs),
        max_sampled_pairs=int(args.max_sampled_pairs),
        seed=int(args.anchor_seed),
    )
    local_radius_value = radius_value_from_report(radius_report, args.radius)
    standard_radius_value = standard_radius_value_from_report(
        radius_report,
        source=str(args.standard_radius_source),
        embedding_bound=B_all,
    )

    eps_grid = sorted(set(as_float_list(args.epsilon_grid)))
    m_grid = sorted(set(as_int_list(args.m_grid)))
    config_pairs: list[tuple[float, int]] = []
    if args.sweep in {"epsilon", "both"}:
        config_pairs.extend((float(eps), int(args.fixed_m)) for eps in eps_grid)
    if args.sweep in {"m", "both"}:
        config_pairs.extend((float(args.fixed_epsilon), int(m)) for m in m_grid)
    config_pairs = sorted(set(config_pairs))
    required_ms = sorted({m for _, m in config_pairs})
    if not required_ms or min(required_ms) < 2:
        raise ValueError("All candidate support sizes m must be at least 2.")
    mechanisms = (
        ["ball", "standard"] if args.mechanisms == "both" else [str(args.mechanisms)]
    )

    feasible_banks = find_feasible_support_banks(
        data,
        radius_value=float(local_radius_value),
        max_required_m=max(required_ms),
        num_supports=int(args.num_supports),
        anchor_seed=int(args.anchor_seed),
        source_mode=str(args.support_source),
        max_search=args.max_feasible_search,
        explicit_anchor_indices=args.anchor_indices,
        strict=bool(args.strict_feasible_supports),
        anchor_selection=str(args.anchor_selection),
    )

    results_root = Path(args.results_root)
    dataset_dir = ensure_dir(
        results_root / "paper2" / "nonconvex_transcript_attack" / tag / ds_spec.tag
    )
    run_payload = {
        **vars(args),
        "dataset_spec": asdict(ds_spec),
        "model_tag": tag,
        "task": task,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "B_all": B_all,
        "certified_lz": float(lz),
        "n_train_used": int(len(data.X_train)),
        "n_test_used": int(len(data.X_test)),
    }
    run_id = hashlib.sha1(
        json.dumps(run_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    run_dir = ensure_dir(dataset_dir / "runs" / run_id)
    write_json(run_dir / "run_config.json", run_payload)
    write_json(run_dir / "radius_report.json", radius_report)
    write_json(
        run_dir / "dataset_metadata.json",
        {
            "dataset_name": ds_spec.display_name,
            "dataset_tag": ds_spec.tag,
            "n_train_used": int(len(data.X_train)),
            "n_test_used": int(len(data.X_test)),
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "task": task,
            "B_all": B_all,
            "A": float(args.A),
            "Lambda": float(args.Lambda),
            "certified_lz": float(lz),
            "radius_tag": args.radius,
            "radius_value": float(local_radius_value),
            "standard_radius_value_metadata_only": float(standard_radius_value),
            "critical_radius_for_clip_norm": float(
                2.0 * float(args.clip_norm) / max(float(lz), 1e-30)
            ),
            "backend": data.backend,
            "num_feasible_support_anchors": int(len(feasible_banks)),
        },
    )

    support_sets_meta: list[dict[str, Any]] = []
    support_sets: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray, list[str], np.ndarray, str]
    ] = {}
    for bank in feasible_banks:
        for draw_index in range(int(args.support_draws)):
            for m in required_ms:
                Xs, ys, source_ids, dists = make_support_set(
                    bank,
                    m=int(m),
                    draw_index=int(draw_index),
                    base_seed=int(args.anchor_seed),
                    selection=str(args.support_selection),
                )
                support_hash = support_source_hash(source_ids)
                support_sets[(bank.anchor_index, int(draw_index), int(m))] = (
                    Xs,
                    ys,
                    source_ids,
                    dists,
                    support_hash,
                )
                support_sets_meta.append(
                    {
                        "anchor_index": int(bank.anchor_index),
                        "anchor_label": int(bank.anchor_label),
                        "support_draw_index": int(draw_index),
                        "m": int(m),
                        "support_set_hash": support_hash,
                        "support_source_ids": source_ids,
                        "support_max_distance": float(np.max(dists)),
                        "support_mean_distance": float(np.mean(dists)),
                        "support_min_distance": float(np.min(dists)),
                        "support_bank_size": int(bank.bank_vectors.shape[0]),
                        "support_source_mode": str(args.support_source),
                        "support_selection": str(args.support_selection),
                    }
                )
    write_json(run_dir / "support_sets.json", support_sets_meta)

    release_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    calibration_cache: dict[tuple[str, float], dict[str, Any]] = {}

    for epsilon, m in config_pairs:
        for mechanism in mechanisms:
            cal_key = (mechanism, float(epsilon))
            if cal_key not in calibration_cache:
                calibration_cache[cal_key] = calibrate_noise(
                    target_epsilon=float(epsilon),
                    mechanism=mechanism,
                    n_total=int(len(data.X_train)),
                    num_steps=int(args.num_steps),
                    batch_size=int(args.batch_size),
                    clip_norm=float(args.clip_norm),
                    radius=float(local_radius_value),
                    lz=float(lz),
                    delta=float(args.delta),
                    orders=DEFAULT_ORDERS,
                    calibration_steps=int(args.calibration_bisection_steps),
                )
            cal = calibration_cache[cal_key]
            noise_multiplier = float(cal["noise_multiplier"])
            for bank in feasible_banks:
                X_minus, y_minus = remove_train_index(
                    data.X_train, data.y_train, bank.anchor_index
                )
                for draw_index in range(int(args.support_draws)):
                    X_support, y_support, source_ids, support_dists, support_hash = (
                        support_sets[(bank.anchor_index, int(draw_index), int(m))]
                    )
                    target_positions = target_positions_for_support(
                        m=int(m),
                        max_targets=(
                            None
                            if int(args.max_targets_per_support) <= 0
                            else int(args.max_targets_per_support)
                        ),
                        seed=int(args.attack_seed),
                        anchor_index=int(bank.anchor_index),
                        draw_index=int(draw_index),
                    )
                    for target_pos in target_positions:
                        target_sid = str(source_ids[int(target_pos)])
                        x_target = np.asarray(
                            X_support[int(target_pos)], dtype=np.float32
                        )
                        y_target = int(y_support[int(target_pos)])
                        X_full = np.concatenate(
                            [X_minus, x_target[None, :]], axis=0
                        ).astype(np.float32, copy=False)
                        y_full = np.concatenate(
                            [y_minus, np.asarray([y_target], dtype=np.int32)], axis=0
                        )
                        target_index = int(len(X_full) - 1)
                        full_ds = ArrayDataset(
                            X_full, y_full, name="paper2_attack_full"
                        )
                        true_record = full_ds.record(target_index)

                        for release_seed in as_int_list(args.release_seeds):
                            release_key_payload = {
                                "dataset": ds_spec.tag,
                                "model_tag": tag,
                                "task": task,
                                "mechanism": mechanism,
                                "epsilon": float(epsilon),
                                "m": int(m),
                                "release_seed": int(release_seed),
                                "radius_tag": args.radius,
                                "anchor_index": int(bank.anchor_index),
                                "support_set_hash": str(support_hash),
                                "target_source_id": target_sid,
                                "num_steps": int(args.num_steps),
                                "batch_size": int(args.batch_size),
                                "clip_norm": float(args.clip_norm),
                            }
                            release_key = row_key(release_key_payload)
                            release, recorder = train_recorded_release(
                                spec=spec,
                                bounds=bounds,
                                X_train=X_full,
                                y_train=y_full,
                                X_eval=data.X_test,
                                y_eval=data.y_test,
                                mechanism=mechanism,
                                radius=float(local_radius_value),
                                epsilon=float(epsilon),
                                delta=float(args.delta),
                                noise_multiplier=noise_multiplier,
                                num_steps=int(args.num_steps),
                                batch_size=int(args.batch_size),
                                clip_norm=float(args.clip_norm),
                                learning_rate=float(args.learning_rate),
                                eval_every=int(args.eval_every),
                                eval_batch_size=int(args.eval_batch_size),
                                checkpoint_selection=str(args.checkpoint_selection),
                                release_seed=int(release_seed),
                                orders=DEFAULT_ORDERS,
                                trace_capture_every=int(args.trace_capture_every),
                                record_operator_norms=bool(args.record_operator_norms),
                                operator_norms_every=int(args.operator_norms_every),
                            )
                            bound_metrics = compute_hayes_bounds(
                                release,
                                m=int(m),
                                feature_dim=feature_dim,
                                include_rdp=bool(args.include_rdp),
                            )
                            ratios = release.extra.get(
                                "ball_to_standard_sensitivity_ratio_by_step"
                            )
                            finite_ratios = [
                                float(v)
                                for v in (ratios or [])
                                if v is not None and math.isfinite(float(v))
                            ]
                            release_row = {
                                **release_key_payload,
                                "release_key": release_key,
                                "run_id": run_id,
                                "dataset_tag": ds_spec.tag,
                                "dataset_name": ds_spec.display_name,
                                "hidden_dim": int(args.hidden_dim),
                                "A": float(args.A),
                                "Lambda": float(args.Lambda),
                                "B_all": float(B_all),
                                "certified_lz": float(lz),
                                "num_classes": int(num_classes),
                                "feature_dim": int(feature_dim),
                                "n_train": int(len(X_full)),
                                "n_test": int(len(data.X_test)),
                                "radius_value": float(local_radius_value),
                                "standard_radius_value_metadata_only": float(
                                    standard_radius_value
                                ),
                                "delta": float(args.delta),
                                "learning_rate": float(args.learning_rate),
                                "noise_multiplier": noise_multiplier,
                                "calibrated_epsilon": float(
                                    cal.get("epsilon", float("nan"))
                                ),
                                "sample_rate": float(args.batch_size)
                                / float(len(X_full)),
                                "accuracy": maybe_float(
                                    release.utility_metrics.get("accuracy")
                                ),
                                "public_eval_loss": maybe_float(
                                    release.utility_metrics.get("public_eval_loss")
                                ),
                                "actual_epsilon_ball": actual_epsilon(release, "ball"),
                                "actual_epsilon_standard": actual_epsilon(
                                    release, "standard"
                                ),
                                "delta_ball_max": maybe_float(
                                    release.sensitivity.delta_ball
                                ),
                                "delta_std_max": maybe_float(
                                    release.sensitivity.delta_std
                                ),
                                "sensitivity_ratio_max": (
                                    float(max(finite_ratios))
                                    if finite_ratios
                                    else float("nan")
                                ),
                                "sensitivity_ratio_mean": (
                                    float(np.mean(finite_ratios))
                                    if finite_ratios
                                    else float("nan")
                                ),
                                "critical_radius_for_min_clip": maybe_float(
                                    release.extra.get("critical_radius_for_min_clip")
                                ),
                                "ball_regime": str(
                                    release.extra.get("ball_regime", "unknown")
                                ),
                                **bound_metrics,
                            }
                            release_rows.append(release_row)

                            trace_meta = make_trace_metadata_from_release(
                                release,
                                target_index=target_index,
                                extra={
                                    "residualized_against_known_batch": False,
                                    "support_set_hash": str(support_hash),
                                    "target_source_id": target_sid,
                                    "anchor_index": int(bank.anchor_index),
                                },
                            )
                            trace = recorder.to_trace(
                                state=release.extra.get("model_state", None),
                                loss_name=spec.loss_name,
                                reduction=str(trace_meta.get("reduction", "mean")),
                                metadata=trace_meta,
                            )
                            target_inclusion_count = int(
                                sum(
                                    bool(
                                        np.any(
                                            np.asarray(
                                                step.batch_indices, dtype=np.int64
                                            )
                                            == target_index
                                        )
                                    )
                                    for step in trace.steps
                                )
                            )
                            try:
                                residual_trace = subtract_known_batch_gradients(
                                    trace,
                                    full_ds,
                                    target_index=target_index,
                                    loss_name=spec.loss_name,
                                    seed=int(args.attack_seed),
                                )
                            except Exception as exc:
                                residual_trace = None
                                residual_error = f"error: {type(exc).__name__}: {exc}"
                            else:
                                residual_error = "ok"

                            for attack_mode in list(args.attack_modes):
                                if residual_trace is None:
                                    status, metrics, diagnostics = (
                                        residual_error,
                                        {},
                                        {},
                                    )
                                else:
                                    status, metrics, diagnostics = run_one_attack(
                                        residual_trace=residual_trace,
                                        X_support=X_support,
                                        y_support=y_support,
                                        target_index=target_index,
                                        known_label=int(bank.anchor_label),
                                        true_record=true_record,
                                        mode=str(attack_mode),
                                        attack_seed=int(args.attack_seed),
                                        known_inclusion_step_mode=str(
                                            args.known_inclusion_step_mode
                                        ),
                                    )
                                trial_payload = {
                                    "release_key": release_key,
                                    "attack_mode": str(attack_mode),
                                    "target_source_id": target_sid,
                                }
                                trial_key = row_key(trial_payload)
                                pred_idx = diagnostics.get("predicted_prior_index")
                                true_idx = diagnostics.get("true_prior_index")
                                trial_rows.append(
                                    {
                                        "trial_key": trial_key,
                                        "release_key": release_key,
                                        "run_id": run_id,
                                        "dataset_tag": ds_spec.tag,
                                        "dataset_name": ds_spec.display_name,
                                        "model_tag": tag,
                                        "task": task,
                                        "mechanism": mechanism,
                                        "attack_mode": str(attack_mode),
                                        "attack_status": status,
                                        "radius_tag": args.radius,
                                        "radius_value": float(local_radius_value),
                                        "epsilon": float(epsilon),
                                        "m": int(m),
                                        "delta": float(args.delta),
                                        "num_steps": int(args.num_steps),
                                        "batch_size": int(args.batch_size),
                                        "clip_norm": float(args.clip_norm),
                                        "hidden_dim": int(args.hidden_dim),
                                        "A": float(args.A),
                                        "Lambda": float(args.Lambda),
                                        "B_all": float(B_all),
                                        "certified_lz": float(lz),
                                        "release_seed": int(release_seed),
                                        "anchor_index": int(bank.anchor_index),
                                        "anchor_label": int(bank.anchor_label),
                                        "support_draw_index": int(draw_index),
                                        "support_set_hash": str(support_hash),
                                        "support_selection": str(
                                            args.support_selection
                                        ),
                                        "support_source_mode": str(args.support_source),
                                        "target_position": int(target_pos),
                                        "target_source_id": target_sid,
                                        "predicted_prior_index": (
                                            None if pred_idx is None else int(pred_idx)
                                        ),
                                        "true_prior_index": (
                                            None if true_idx is None else int(true_idx)
                                        ),
                                        "target_inclusion_count": int(
                                            target_inclusion_count
                                        ),
                                        "retained_trace_steps": int(len(trace.steps)),
                                        "selected_step_count": finite_or_nan(
                                            diagnostics.get("selected_step_count")
                                        ),
                                        "exact_identification_success": finite_or_nan(
                                            metrics.get("exact_identification_success")
                                        ),
                                        "prior_rank": finite_or_nan(
                                            metrics.get("prior_rank")
                                        ),
                                        "prior_hit@5": finite_or_nan(
                                            metrics.get("prior_hit@5")
                                        ),
                                        "mse": finite_or_nan(metrics.get("mse")),
                                        "oblivious_kappa": finite_or_nan(
                                            metrics.get(
                                                "oblivious_kappa",
                                                bound_metrics.get("oblivious_kappa"),
                                            )
                                        ),
                                        "accuracy": maybe_float(
                                            release.utility_metrics.get("accuracy")
                                        ),
                                        "noise_multiplier": noise_multiplier,
                                        **bound_metrics,
                                    }
                                )
                                print(
                                    f"[{ds_spec.tag}] mech={mechanism} eps={epsilon:g} m={m} "
                                    f"target={target_sid} seed={release_seed} mode={attack_mode} status={status}",
                                    flush=True,
                                )

    save_dataframe(
        pd.DataFrame(release_rows),
        run_dir / "release_rows.csv",
        save_parquet_if_possible=False,
    )
    save_dataframe(
        pd.DataFrame(trial_rows),
        run_dir / "attack_trial_rows.csv",
        save_parquet_if_possible=False,
    )
    rebuild_outputs(dataset_dir)


if __name__ == "__main__":
    main()
