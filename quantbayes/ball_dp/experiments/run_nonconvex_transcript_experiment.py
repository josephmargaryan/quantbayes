#!/usr/bin/env python3
"""Official Paper 2 nonconvex transcript utility/bound experiment.

This script trains theorem-backed one-hidden-layer tanh networks with operator-norm
constraints, releases the DP-SGD transcript implicitly through the trained release
artifact metadata, and evaluates the direct Hayes-style Ball-ReRo transcript bound.

The public experiment protocol deliberately differs from Paper 1 in one respect:
DP-SGD uses a noise multiplier.  Therefore this script first calibrates the scalar
noise multiplier to the requested (epsilon, delta) target under either the Ball or
standard clipping-only accountant, then runs training once with that calibrated
noise multiplier.
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
    ball_rero,
    make_finite_identification_prior,
    summarize_embedding_ball_radii,
)
from quantbayes.ball_dp.accountants.subsampled_gaussian import (
    calibrate_ball_sgd_noise_multiplier,
)
from quantbayes.ball_dp.experiments.run_attack_experiment import (
    DEFAULT_DELTA,
    DEFAULT_EPS_GRID,
    DEFAULT_FIXED_EPSILON,
    DEFAULT_FIXED_M,
    DEFAULT_M_GRID,
    DEFAULT_ORDERS,
    RADIUS_TAG_TO_QUANTILE,
    as_float_list,
    as_int_list,
    configure_matplotlib,
    ensure_dir,
    load_embeddings,
    radius_value_from_report,
    resolve_dataset,
    savefig_stem,
    standard_radius_value_from_report,
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

DEFAULT_HIDDEN_DIM = 128
DEFAULT_A = 4.0
DEFAULT_LAMBDA = 4.0
DEFAULT_CLIP_NORM = 1.0
DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_STEPS = 600
DEFAULT_LEARNING_RATE = 3e-3
DEFAULT_MAX_TRAIN_EXAMPLES = 6000
DEFAULT_MAX_TEST_EXAMPLES = 2000
DEFAULT_INPUT_BOUND_MARGIN = 1.001
DEFAULT_CALIBRATION_STEPS = 18


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def finite_prior_for_m(m: int, feature_dim: int) -> Any:
    # Geometry is irrelevant for FiniteExactIdentificationPrior at eta < 1.
    return make_finite_identification_prior(
        np.zeros((int(m), int(feature_dim)), dtype=np.float32), weights=None
    )


def summarize_mean_std(values: Sequence[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0, mean, mean
    std = float(np.std(arr, ddof=1))
    se = std / math.sqrt(float(arr.size))
    return mean, std, mean - 1.96 * se, mean + 1.96 * se


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_examples: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_examples is None or int(max_examples) <= 0 or int(max_examples) >= len(X):
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

    max_examples = int(max_examples)
    rng = np.random.default_rng(int(seed))
    y_arr = np.asarray(y, dtype=np.int32)
    labels, counts = np.unique(y_arr, return_counts=True)
    k = int(len(labels))
    if max_examples < k:
        raise ValueError(
            f"max_examples={max_examples} is smaller than the number of classes {k}."
        )

    # Start with proportional allocation, then distribute any leftover examples.
    raw = counts.astype(float) / float(np.sum(counts)) * float(max_examples)
    alloc = np.floor(raw).astype(int)
    alloc = np.maximum(alloc, 1)
    alloc = np.minimum(alloc, counts)

    while int(np.sum(alloc)) > max_examples:
        candidates = np.where(alloc > 1)[0]
        j = int(candidates[np.argmax(alloc[candidates])])
        alloc[j] -= 1
    while int(np.sum(alloc)) < max_examples:
        room = counts - alloc
        candidates = np.where(room > 0)[0]
        if candidates.size == 0:
            break
        frac = raw - np.floor(raw)
        j = int(candidates[np.argmax(frac[candidates])])
        alloc[j] += 1

    selected: list[int] = []
    for label, n_take in zip(labels, alloc, strict=True):
        idx = np.flatnonzero(y_arr == int(label))
        chosen = rng.choice(idx, size=int(n_take), replace=False)
        selected.extend(int(i) for i in chosen.tolist())
    rng.shuffle(selected)
    idx = np.asarray(selected, dtype=np.int64)
    return np.asarray(X[idx], dtype=np.float32), np.asarray(y_arr[idx], dtype=np.int32)


def resolve_task(task_arg: str, num_classes: int) -> str:
    task = str(task_arg).strip().lower()
    if task == "auto":
        return "binary" if int(num_classes) == 2 else "multiclass"
    if task == "binary" and int(num_classes) != 2:
        raise ValueError("--task binary is only valid for two-class datasets.")
    if task not in {"binary", "multiclass"}:
        raise ValueError("--task must be one of {'auto', 'binary', 'multiclass'}.")
    return task


def model_tag(*, task: str, hidden_dim: int, A: float, Lambda: float) -> str:
    return (
        f"tanh_op_{task}_h{int(hidden_dim)}_" f"A{float(A):g}_L{float(Lambda):g}"
    ).replace(".", "p")


def actual_epsilon(release: Any, view: str) -> Optional[float]:
    ledger = release.privacy.ball if view == "ball" else release.privacy.standard
    if not ledger.dp_certificates:
        return None
    return maybe_float(ledger.dp_certificates[0].epsilon)


def first_order(release: Any, view: str) -> Optional[float]:
    ledger = release.privacy.ball if view == "ball" else release.privacy.standard
    certs = ledger.dp_certificates
    if not certs:
        return None
    return maybe_float(certs[0].order_opt)


def compute_hayes_bounds(
    release: Any,
    *,
    m: int,
    feature_dim: int,
    include_rdp: bool,
) -> dict[str, float]:
    prior = finite_prior_for_m(int(m), int(feature_dim))
    eta_grid = (0.5,)
    out: dict[str, float] = {
        "oblivious_kappa": float("nan"),
        "bound_hayes_ball": float("nan"),
        "bound_hayes_standard": float("nan"),
        "bound_rdp_ball": float("nan"),
        "bound_rdp_standard": float("nan"),
    }
    report_hayes = ball_rero(release, prior=prior, eta_grid=eta_grid, mode="hayes")
    p = report_hayes.points[0]
    out["oblivious_kappa"] = float(p.kappa)
    out["bound_hayes_ball"] = float(p.gamma_ball)
    out["bound_hayes_standard"] = (
        float("nan") if p.gamma_standard is None else float(p.gamma_standard)
    )
    if include_rdp:
        try:
            report_rdp = ball_rero(release, prior=prior, eta_grid=eta_grid, mode="rdp")
            rp = report_rdp.points[0]
            out["bound_rdp_ball"] = float(rp.gamma_ball)
            out["bound_rdp_standard"] = (
                float("nan") if rp.gamma_standard is None else float(rp.gamma_standard)
            )
        except Exception:
            pass
    return out


def calibrate_noise(
    *,
    target_epsilon: float,
    mechanism: str,
    n_total: int,
    num_steps: int,
    batch_size: int,
    clip_norm: float,
    radius: float,
    lz: float,
    delta: float,
    orders: Sequence[int | float],
    calibration_steps: int,
) -> dict[str, Any]:
    view = "ball" if mechanism == "ball" else "standard"
    return calibrate_ball_sgd_noise_multiplier(
        target_epsilon=float(target_epsilon),
        accounting_view=view,
        orders=tuple(float(a) for a in orders),
        dataset_size=int(n_total),
        num_steps=int(num_steps),
        batch_size=int(batch_size),
        clip_norm=float(clip_norm),
        radius=float(radius),
        lz=float(lz),
        dp_delta=float(delta),
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
        num_bisection_steps=int(calibration_steps),
    )


def train_one_release(
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
    record_operator_norms: bool,
    operator_norms_every: int,
) -> Any:
    privacy = "ball_dp" if mechanism == "ball" else "standard_dp"
    model = make_model(
        spec,
        key=jr.PRNGKey(100_000 + int(release_seed)),
        init_project=True,
        bounds=bounds,
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
    return fit_release(
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
        record_operator_norms=bool(record_operator_norms),
        operator_norms_every=int(operator_norms_every),
    )


def row_key(payload: dict[str, Any]) -> str:
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def rebuild_outputs(dataset_dir: Path) -> None:
    run_dirs = sorted((dataset_dir / "runs").glob("*"))
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "transcript_seed_rows.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return

    raw = pd.concat(frames, ignore_index=True)
    if "row_key" in raw.columns:
        raw = raw.drop_duplicates(subset=["row_key"], keep="last")
    save_dataframe(
        raw, dataset_dir / "transcript_seed_rows.csv", save_parquet_if_possible=False
    )

    config_cols = [
        "dataset_tag",
        "dataset_name",
        "model_tag",
        "task",
        "mechanism",
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
    metric_cols = [
        "accuracy",
        "public_eval_loss",
        "noise_multiplier",
        "actual_epsilon_ball",
        "actual_epsilon_standard",
        "bound_hayes_ball",
        "bound_hayes_standard",
        "bound_rdp_ball",
        "bound_rdp_standard",
        "delta_ball_max",
        "delta_std_max",
        "sensitivity_ratio_max",
        "sensitivity_ratio_mean",
        "critical_radius_for_min_clip",
    ]
    rows: list[dict[str, Any]] = []
    for key, frame in raw.groupby(config_cols, dropna=False):
        row = {col: val for col, val in zip(config_cols, key, strict=True)}
        row["n_runs"] = int(frame["release_seed"].nunique())
        for col in metric_cols:
            if col not in frame.columns:
                continue
            mean, std, lo, hi = summarize_mean_std(frame[col].astype(float).tolist())
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
        rows.append(row)
    summary = (
        pd.DataFrame(rows)
        .sort_values(["mechanism", "epsilon", "m"])
        .reset_index(drop=True)
    )
    save_dataframe(
        summary, dataset_dir / "transcript_summary.csv", save_parquet_if_possible=False
    )

    figures_dir = ensure_dir(dataset_dir / "figures")
    for y_col, ylabel, stem in [
        ("accuracy_mean", "Accuracy", "fig_accuracy_vs_epsilon"),
        (
            "noise_multiplier_mean",
            "Noise multiplier",
            "fig_noise_multiplier_vs_epsilon",
        ),
        (
            "bound_hayes_ball_mean",
            "Hayes Ball-ReRo bound",
            "fig_hayes_ball_bound_vs_epsilon",
        ),
        (
            "bound_hayes_standard_mean",
            "Hayes standard bound",
            "fig_hayes_standard_bound_vs_epsilon",
        ),
    ]:
        if y_col not in summary.columns:
            continue
        sub_m_values = sorted(summary["m"].dropna().unique().tolist())
        for m_val in sub_m_values:
            sub = summary[summary["m"] == m_val].copy()
            if sub.empty:
                continue
            fig, ax = plt.subplots()
            for mechanism, marker in (("ball", "o"), ("standard", "s")):
                mm = sub[sub["mechanism"] == mechanism].sort_values("epsilon")
                if mm.empty:
                    continue
                ax.plot(mm["epsilon"], mm[y_col], marker=marker, label=mechanism)
                lo_col = y_col.replace("_mean", "_ci_low")
                hi_col = y_col.replace("_mean", "_ci_high")
                if lo_col in mm.columns and hi_col in mm.columns:
                    ax.fill_between(mm["epsilon"], mm[lo_col], mm[hi_col], alpha=0.15)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("$\\varepsilon$ target")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs $\\varepsilon$ · m={int(m_val)}")
            ax.legend()
            savefig_stem(fig, figures_dir / f"{stem}_m{int(m_val)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official Paper 2 nonconvex transcript utility/bound experiments."
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
    parser.add_argument("--release-seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
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
    data = load_embeddings(args, ds_spec)

    X_train, y_train = stratified_subsample(
        data.X_train,
        data.y_train,
        max_examples=args.max_train_examples,
        seed=int(args.subsample_seed),
    )
    X_test, y_test = stratified_subsample(
        data.X_test,
        data.y_test,
        max_examples=args.max_test_examples,
        seed=int(args.subsample_seed) + 17,
    )
    num_classes = int(len(np.unique(y_train)))
    feature_dim = int(X_train.shape[1])
    task = resolve_task(args.task, num_classes)

    B_all = float(
        max(
            np.max(np.linalg.norm(X_train, axis=1)),
            np.max(np.linalg.norm(X_test, axis=1)),
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
        X_train,
        y_train,
        quantiles=(0.50, 0.80, 0.95),
        max_exact_pairs=int(args.max_exact_pairs),
        max_sampled_pairs=int(args.max_sampled_pairs),
        seed=int(args.subsample_seed),
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
    mechanisms = (
        ["ball", "standard"] if args.mechanisms == "both" else [str(args.mechanisms)]
    )

    results_root = Path(args.results_root)
    dataset_dir = ensure_dir(
        results_root / "paper2" / "nonconvex_transcript" / tag / ds_spec.tag
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
        "n_train_used": int(len(X_train)),
        "n_test_used": int(len(X_test)),
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
            "n_train_used": int(len(X_train)),
            "n_test_used": int(len(X_test)),
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
        },
    )

    rows: list[dict[str, Any]] = []
    calibration_cache: dict[tuple[str, float], dict[str, Any]] = {}
    for epsilon, m in config_pairs:
        for mechanism in mechanisms:
            cal_key = (mechanism, float(epsilon))
            if cal_key not in calibration_cache:
                calibration_cache[cal_key] = calibrate_noise(
                    target_epsilon=float(epsilon),
                    mechanism=mechanism,
                    n_total=int(len(X_train)),
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
            for release_seed in as_int_list(args.release_seeds):
                release = train_one_release(
                    spec=spec,
                    bounds=bounds,
                    X_train=X_train,
                    y_train=y_train,
                    X_eval=X_test,
                    y_eval=y_test,
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
                    record_operator_norms=bool(args.record_operator_norms),
                    operator_norms_every=int(args.operator_norms_every),
                )
                bound_metrics = compute_hayes_bounds(
                    release,
                    m=int(m),
                    feature_dim=feature_dim,
                    include_rdp=bool(args.include_rdp),
                )
                ratios = release.extra.get("ball_to_standard_sensitivity_ratio_by_step")
                finite_ratios = [
                    float(v)
                    for v in (ratios or [])
                    if v is not None and math.isfinite(float(v))
                ]
                key_payload = {
                    "dataset": ds_spec.tag,
                    "model_tag": tag,
                    "task": task,
                    "mechanism": mechanism,
                    "epsilon": float(epsilon),
                    "m": int(m),
                    "release_seed": int(release_seed),
                    "radius_tag": args.radius,
                    "num_steps": int(args.num_steps),
                    "batch_size": int(args.batch_size),
                    "clip_norm": float(args.clip_norm),
                }
                rows.append(
                    {
                        **key_payload,
                        "row_key": row_key(key_payload),
                        "run_id": run_id,
                        "dataset_tag": ds_spec.tag,
                        "dataset_name": ds_spec.display_name,
                        "model_tag": tag,
                        "hidden_dim": int(args.hidden_dim),
                        "A": float(args.A),
                        "Lambda": float(args.Lambda),
                        "B_all": float(B_all),
                        "certified_lz": float(lz),
                        "task": task,
                        "num_classes": int(num_classes),
                        "feature_dim": int(feature_dim),
                        "n_train": int(len(X_train)),
                        "n_test": int(len(X_test)),
                        "radius_tag": args.radius,
                        "radius_value": float(local_radius_value),
                        "standard_radius_value_metadata_only": float(
                            standard_radius_value
                        ),
                        "delta": float(args.delta),
                        "learning_rate": float(args.learning_rate),
                        "noise_multiplier": noise_multiplier,
                        "calibrated_accounting_view": mechanism,
                        "calibrated_epsilon": float(cal.get("epsilon", float("nan"))),
                        "sample_rate": float(args.batch_size) / float(len(X_train)),
                        "accuracy": maybe_float(
                            release.utility_metrics.get("accuracy")
                        ),
                        "public_eval_loss": maybe_float(
                            release.utility_metrics.get("public_eval_loss")
                        ),
                        "actual_epsilon_ball": actual_epsilon(release, "ball"),
                        "actual_epsilon_standard": actual_epsilon(release, "standard"),
                        "order_opt_ball": first_order(release, "ball"),
                        "order_opt_standard": first_order(release, "standard"),
                        "delta_ball_max": maybe_float(release.sensitivity.delta_ball),
                        "delta_std_max": maybe_float(release.sensitivity.delta_std),
                        "sensitivity_ratio_max": (
                            float(max(finite_ratios)) if finite_ratios else float("nan")
                        ),
                        "sensitivity_ratio_mean": (
                            float(np.mean(finite_ratios))
                            if finite_ratios
                            else float("nan")
                        ),
                        "critical_radius_for_min_clip": maybe_float(
                            release.extra.get("critical_radius_for_min_clip")
                        ),
                        "ball_regime": str(release.extra.get("ball_regime", "unknown")),
                        "selected_checkpoint_step": maybe_float(
                            release.extra.get("selected_checkpoint_step")
                        ),
                        **bound_metrics,
                    }
                )
                print(
                    f"[{ds_spec.tag}] mechanism={mechanism} eps={epsilon:g} m={m} seed={release_seed} "
                    f"acc={rows[-1]['accuracy']} hayes_ball={rows[-1]['bound_hayes_ball']:.4g}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    save_dataframe(
        df, run_dir / "transcript_seed_rows.csv", save_parquet_if_possible=False
    )
    rebuild_outputs(dataset_dir)


if __name__ == "__main__":
    main()
