#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses as dc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import jax
import jax.random as jr

from quantbayes.ball_dp.api import (
    ball_rero,
    calibrate_ball_sgd_noise_multiplier,
    evaluate_release_classifier,
    get_release_step_table,
    make_finite_identification_prior,
)
from quantbayes.ball_dp.experiments.run_attack_experiment import (
    DEFAULT_RADIUS_TAG,
    RADIUS_TAG_TO_QUANTILE,
    LoadedDataset,
    SupportBank,
    as_float_list,
    as_int_list,
    build_support_bank,
    ensure_dir,
    find_feasible_support_banks,
    finite_prior_hash,
    load_embeddings,
    make_support_set,
    radius_value_from_report,
    resolve_dataset,
    savefig_stem,
    slugify,
    summarize_embedding_ball_radii,
    support_source_hash,
    write_json,
)
from quantbayes.ball_dp.serialization import save_dataframe
from quantbayes.ball_dp.theorem.registry import certified_lz, make_model
from quantbayes.ball_dp.theorem.specs import (
    TheoremBounds,
    TheoremModelSpec,
    TrainConfig,
)
from quantbayes.ball_dp.theorem.workflows import fit_release as theorem_fit_release

"""
%%bash
set -euo pipefail

RESULTS=/content/quantbayes/results_paper2_nonconvex_official

DATASETS=(
  "AG News-embeddings"
  "BANKING77-embeddings"
  "CIFAR-10-embeddings"
  "DBpedia-14-embeddings"
  "Emotion-embeddings"
  "IMDb-embeddings"
  "MNIST-embeddings"
  "TREC-6-embeddings"
  "Yelp Review Full-embeddings"
)

for DS in "${DATASETS[@]}"; do
  echo "=== Paper 2 utility/bounds: $DS ==="

  python quantbayes/ball_dp/experiments/run_nonconvex_transcript_experiment.py \
    --results-root "$RESULTS" \
    --dataset "$DS" \
    --radius q80 \
    --hidden-dim 128 \
    --A 4.0 \
    --Lambda 4.0 \
    --sweep epsilon \
    --fixed-m 8 \
    --epsilon-grid 4 8 \
    --release-seeds 0 1 \
    --num-steps 400 \
    --batch-size 128 \
    --clip-norm 1.0 \
    --learning-rate 1e-3 \
    --mechanisms both \
    --record-operator-norms \
    --operator-norms-every 100
done

python quantbayes/ball_dp/experiments/aggregate_nonconvex_transcript.py \
  --results-root "$RESULTS"

"""

DEFAULT_ORDERS = tuple(list(range(2, 65)) + [80, 96, 128, 160, 192, 256])
DEFAULT_EPS_GRID = (4.0, 8.0)
DEFAULT_RELEASE_SEEDS = (0, 1)
DEFAULT_M_GRID = (8,)
DEFAULT_DELTA = 1e-6
DEFAULT_HIDDEN_DIM = 128
DEFAULT_A = 4.0
DEFAULT_LAMBDA = 4.0
DEFAULT_NUM_STEPS = 400
DEFAULT_BATCH_SIZE = 128
DEFAULT_CLIP_NORM = 1.0
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EVAL_EVERY = 100
DEFAULT_EVAL_BATCH_SIZE = 1024


def json_default(obj: Any) -> Any:
    if dc.is_dataclass(obj):
        return dc.asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (tuple, list)):
        return [json_default(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): json_default(v) for k, v in obj.items()}
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return repr(obj)


def write_json_safe(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_default(payload), indent=2, sort_keys=True))


def add_embedding_loader_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--embedding-cache-path", type=str, default=None)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument(
        "--encoder-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--max-length", type=int, default=256)


def add_radius_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--radius", choices=tuple(RADIUS_TAG_TO_QUANTILE), default=DEFAULT_RADIUS_TAG
    )
    parser.add_argument(
        "--radius-value",
        type=float,
        default=None,
        help="Override quantile-based radius selection with a numeric policy radius.",
    )
    parser.add_argument("--max-exact-pairs", type=int, default=250_000)
    parser.add_argument("--max-sampled-pairs", type=int, default=100_000)


def add_theorem_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task", choices=("auto", "binary", "multiclass"), default="auto"
    )
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--A", type=float, default=DEFAULT_A)
    parser.add_argument("--Lambda", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument(
        "--input-bound",
        type=float,
        default=None,
        help="Public theorem input-norm bound B. Defaults to empirical train/test max norm times --input-bound-scale.",
    )
    parser.add_argument("--input-bound-scale", type=float, default=1.01)


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument(
        "--epsilon-grid", nargs="+", type=float, default=list(DEFAULT_EPS_GRID)
    )
    parser.add_argument(
        "--release-seeds", nargs="*", type=int, default=list(DEFAULT_RELEASE_SEEDS)
    )
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--clip-norm", type=float, default=DEFAULT_CLIP_NORM)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("last", "best_public_eval_loss", "best_public_eval_accuracy"),
        default="last",
    )
    parser.add_argument(
        "--normalize-noisy-sum-by",
        choices=("batch_size", "none"),
        default="batch_size",
        help="Use only theorem-backed fixed normalization modes for Paper 2.",
    )
    parser.add_argument(
        "--mechanisms",
        choices=("ball", "standard", "both"),
        default="ball",
        help="Ball is the main Paper 2 mechanism; standard is an optional clipping-only comparator.",
    )
    parser.add_argument("--calibration-lower", type=float, default=1e-3)
    parser.add_argument("--calibration-upper", type=float, default=0.25)
    parser.add_argument("--calibration-max-upper", type=float, default=256.0)
    parser.add_argument("--calibration-bisection-steps", type=int, default=14)
    parser.add_argument("--record-operator-norms", action="store_true")
    parser.add_argument("--operator-norms-every", type=int, default=100)
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Optional class-balanced subsample for smoke tests. Omit for official full-data runs.",
    )
    parser.add_argument(
        "--max-test-examples",
        type=int,
        default=None,
        help="Optional class-balanced public-eval subsample for smoke tests. Omit for official full-data runs.",
    )
    parser.add_argument("--subsample-seed", type=int, default=0)


def add_bound_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--m-grid", nargs="+", type=int, default=list(DEFAULT_M_GRID))
    parser.add_argument("--include-rdp", action="store_true", default=True)
    parser.add_argument("--no-include-rdp", dest="include_rdp", action="store_false")
    parser.add_argument(
        "--include-composed-direct",
        action="store_true",
        help="Also evaluate the looser per-step direct composition mode.",
    )


def mechanism_list(mechanisms: str) -> list[str]:
    return ["ball", "standard"] if str(mechanisms) == "both" else [str(mechanisms)]


def class_balanced_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_examples: Optional[int],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y)
    n = int(len(X))
    if max_examples is None or int(max_examples) <= 0 or int(max_examples) >= n:
        idx = np.arange(n, dtype=np.int64)
        return X, y, idx

    max_examples = int(max_examples)
    rng = np.random.default_rng(int(seed))
    labels = np.unique(y)
    per_class = max(1, max_examples // max(1, len(labels)))
    chosen: list[np.ndarray] = []
    for lab in labels:
        pool = np.flatnonzero(y == lab)
        if pool.size == 0:
            continue
        take = min(int(pool.size), per_class)
        chosen.append(rng.choice(pool, size=take, replace=False))

    idx = np.concatenate(chosen) if chosen else np.arange(0, dtype=np.int64)
    if idx.size < max_examples:
        remaining = np.setdiff1d(np.arange(n, dtype=np.int64), idx, assume_unique=False)
        if remaining.size:
            extra = rng.choice(
                remaining,
                size=min(max_examples - int(idx.size), int(remaining.size)),
                replace=False,
            )
            idx = np.concatenate([idx, extra])
    idx = np.asarray(idx[:max_examples], dtype=np.int64)
    rng.shuffle(idx)
    return X[idx], y[idx], idx


def maybe_subsample_loaded_dataset(
    data: LoadedDataset, args: argparse.Namespace
) -> LoadedDataset:
    Xtr, ytr, train_idx = class_balanced_subsample(
        data.X_train,
        data.y_train,
        getattr(args, "max_train_examples", None),
        seed=int(getattr(args, "subsample_seed", 0)),
    )
    Xte, yte, test_idx = class_balanced_subsample(
        data.X_test,
        data.y_test,
        getattr(args, "max_test_examples", None),
        seed=int(getattr(args, "subsample_seed", 0)) + 17,
    )
    if len(train_idx) == len(data.X_train) and len(test_idx) == len(data.X_test):
        return data

    label_values = np.unique(np.concatenate([ytr, yte], axis=0))
    mapping = {int(v): i for i, v in enumerate(label_values.tolist())}
    ytr = np.asarray([mapping[int(v)] for v in ytr], dtype=np.int32)
    yte = np.asarray([mapping[int(v)] for v in yte], dtype=np.int32)
    return LoadedDataset(
        spec=data.spec,
        X_train=np.asarray(Xtr, dtype=np.float32),
        y_train=np.asarray(ytr, dtype=np.int32),
        X_test=np.asarray(Xte, dtype=np.float32),
        y_test=np.asarray(yte, dtype=np.int32),
        label_values=label_values.astype(np.int32, copy=False),
        num_classes=int(len(label_values)),
        feature_dim=int(np.asarray(Xtr).reshape(len(Xtr), -1).shape[1]),
        empirical_embedding_bound=float(
            np.max(np.linalg.norm(Xtr.reshape(len(Xtr), -1), axis=1))
        ),
        backend=data.backend,
    )


def select_policy_radius(
    data: LoadedDataset,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    radius_report = summarize_embedding_ball_radii(
        data.X_train,
        data.y_train,
        quantiles=(0.50, 0.80, 0.95),
        max_exact_pairs=int(args.max_exact_pairs),
        max_sampled_pairs=int(args.max_sampled_pairs),
        seed=int(getattr(args, "anchor_seed", getattr(args, "subsample_seed", 0))),
    )
    if getattr(args, "radius_value", None) is not None:
        radius = float(args.radius_value)
    else:
        radius = radius_value_from_report(radius_report, str(args.radius))
    return float(radius), radius_report


def infer_task(task: str, num_classes: int) -> str:
    task = str(task).lower()
    if task == "auto":
        return "binary" if int(num_classes) == 2 else "multiclass"
    if task == "binary" and int(num_classes) != 2:
        raise ValueError("task='binary' requires exactly two classes.")
    return task


def make_theorem_spec_and_bounds(
    data: LoadedDataset,
    args: argparse.Namespace,
) -> tuple[TheoremModelSpec, TheoremBounds, dict[str, float | str | int]]:
    X_all = np.concatenate(
        [
            np.asarray(data.X_train, dtype=np.float32).reshape(len(data.X_train), -1),
            np.asarray(data.X_test, dtype=np.float32).reshape(len(data.X_test), -1),
        ],
        axis=0,
    )
    empirical_all = float(np.max(np.linalg.norm(X_all, axis=1)))
    if args.input_bound is None:
        B_all = float(empirical_all * float(args.input_bound_scale))
    else:
        B_all = float(args.input_bound)
        if empirical_all > B_all + 1e-5:
            raise ValueError(
                f"Empirical train/test input norm {empirical_all:.6g} exceeds public theorem bound B={B_all:.6g}."
            )

    task = infer_task(str(args.task), int(data.num_classes))
    spec = TheoremModelSpec(
        d_in=int(data.feature_dim),
        hidden_dim=int(args.hidden_dim),
        task=task,
        num_classes=None if task == "binary" else int(data.num_classes),
        parameterization="dense",
        constraint="op",
    )
    bounds = TheoremBounds(B=float(B_all), A=float(args.A), Lambda=float(args.Lambda))
    meta = {
        "task": task,
        "d_in": int(data.feature_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_classes": int(data.num_classes),
        "B": float(B_all),
        "empirical_input_norm_max_train_test": float(empirical_all),
        "A": float(args.A),
        "Lambda": float(args.Lambda),
        "parameterization": "dense",
        "constraint": "op",
    }
    return spec, bounds, meta


def privacy_kind_for_mechanism(mechanism: str) -> str:
    mechanism = str(mechanism).lower()
    if mechanism == "ball":
        return "ball_dp"
    if mechanism == "standard":
        return "standard_dp"
    raise ValueError("mechanism must be one of {'ball', 'standard'}.")


def calibrate_noise_multiplier_for_mechanism(
    *,
    mechanism: str,
    dataset_size: int,
    radius: float,
    lz: float,
    args: argparse.Namespace,
    epsilon: float,
) -> dict[str, Any]:
    return calibrate_ball_sgd_noise_multiplier(
        dataset_size=int(dataset_size),
        radius=float(radius),
        lz=float(lz),
        num_steps=int(args.num_steps),
        batch_size=int(args.batch_size),
        clip_norm=float(args.clip_norm),
        target_epsilon=float(epsilon),
        delta=float(args.delta),
        privacy=privacy_kind_for_mechanism(mechanism),
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
        orders=tuple(int(v) for v in DEFAULT_ORDERS),
        lower=float(args.calibration_lower),
        upper=float(args.calibration_upper),
        max_upper=float(args.calibration_max_upper),
        num_bisection_steps=int(args.calibration_bisection_steps),
    )


def train_theorem_release(
    *,
    spec: TheoremModelSpec,
    bounds: TheoremBounds,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    radius: float,
    lz: float,
    epsilon: float,
    noise_multiplier: float,
    mechanism: str,
    seed: int,
    args: argparse.Namespace,
    trace_recorder: Any = None,
):
    init_key, train_key = jr.split(jr.PRNGKey(int(seed)), 2)
    model = make_model(spec, key=init_key, init_project=True, bounds=bounds)
    train_cfg = TrainConfig(
        radius=float(radius),
        privacy=privacy_kind_for_mechanism(mechanism),
        epsilon=float(epsilon),
        delta=float(args.delta),
        num_steps=int(args.num_steps),
        batch_size=int(args.batch_size),
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
        clip_norm=float(args.clip_norm),
        noise_multiplier=float(noise_multiplier),
        learning_rate=float(args.learning_rate),
        checkpoint_selection=str(args.checkpoint_selection),
        eval_every=int(args.eval_every),
        eval_batch_size=int(args.eval_batch_size),
        normalize_noisy_sum_by=str(args.normalize_noisy_sum_by),
        seed=int(seed),
    )
    return theorem_fit_release(
        model,
        spec,
        bounds,
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.int32),
        train_cfg=train_cfg,
        X_eval=np.asarray(X_eval, dtype=np.float32),
        y_eval=np.asarray(y_eval, dtype=np.int32),
        key=train_key,
        trace_recorder=trace_recorder,
        orders=tuple(int(v) for v in DEFAULT_ORDERS),
        record_operator_norms=bool(getattr(args, "record_operator_norms", False)),
        operator_norms_every=int(getattr(args, "operator_norms_every", 100)),
        warn_if_ball_equals_standard=False,
    )


def finite_exact_prior(m: int, feature_dim: int):
    samples = np.zeros((int(m), int(feature_dim)), dtype=np.float32)
    return make_finite_identification_prior(samples, weights=None)


def _maybe_report_point(report: Any) -> dict[str, float | None | str]:
    p = report.points[0]
    return {
        "mode": str(report.mode),
        "kappa": float(p.kappa),
        "gamma_ball": float(p.gamma_ball),
        "gamma_standard": None if p.gamma_standard is None else float(p.gamma_standard),
        "alpha_opt_ball": None if p.alpha_opt_ball is None else float(p.alpha_opt_ball),
        "alpha_opt_standard": (
            None if p.alpha_opt_standard is None else float(p.alpha_opt_standard)
        ),
    }


def compute_finite_exact_bounds(
    release: Any,
    *,
    m: int,
    feature_dim: int,
    include_rdp: bool = True,
    include_composed_direct: bool = False,
) -> dict[str, float | None | str]:
    prior = finite_exact_prior(int(m), int(feature_dim))
    eta_grid = (0.5,)
    out: dict[str, float | None | str] = {
        "m": int(m),
        "oblivious_kappa": 1.0 / float(m),
        "bound_hayes_ball": float("nan"),
        "bound_hayes_standard_same_noise": float("nan"),
        "bound_rdp_ball": float("nan"),
        "bound_rdp_standard_same_noise": float("nan"),
        "bound_direct_composed_ball": float("nan"),
        "bound_direct_composed_standard_same_noise": float("nan"),
    }

    try:
        hayes = ball_rero(release, prior=prior, eta_grid=eta_grid, mode="hayes")
        p = _maybe_report_point(hayes)
        out["bound_hayes_ball"] = p["gamma_ball"]
        out["bound_hayes_standard_same_noise"] = p["gamma_standard"]
        out["hayes_mode"] = p["mode"]
    except Exception as exc:
        out["hayes_error"] = repr(exc)

    if include_rdp:
        try:
            rdp = ball_rero(release, prior=prior, eta_grid=eta_grid, mode="rdp")
            p = _maybe_report_point(rdp)
            out["bound_rdp_ball"] = p["gamma_ball"]
            out["bound_rdp_standard_same_noise"] = p["gamma_standard"]
            out["rdp_alpha_opt_ball"] = p["alpha_opt_ball"]
            out["rdp_alpha_opt_standard"] = p["alpha_opt_standard"]
        except Exception as exc:
            out["rdp_error"] = repr(exc)

    if include_composed_direct:
        try:
            direct = ball_rero(
                release, prior=prior, eta_grid=eta_grid, mode="ball_sgd_direct"
            )
            p = _maybe_report_point(direct)
            out["bound_direct_composed_ball"] = p["gamma_ball"]
            out["bound_direct_composed_standard_same_noise"] = p["gamma_standard"]
        except Exception as exc:
            out["direct_composed_error"] = repr(exc)
    return out


def release_epsilon(release: Any, view: str) -> float:
    ledger = release.privacy.ball if str(view) == "ball" else release.privacy.standard
    if not ledger.dp_certificates:
        return float("nan")
    return float(ledger.dp_certificates[0].epsilon)


def release_utility_accuracy(release: Any, X: np.ndarray, y: np.ndarray) -> float:
    if "accuracy" in release.utility_metrics:
        return float(release.utility_metrics["accuracy"])
    try:
        out = evaluate_release_classifier(release, X, y, batch_size=1024)
        return float(out.get("accuracy", float("nan")))
    except Exception:
        return float("nan")


def summarize_step_table(step_rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not step_rows:
        return {
            "mean_sample_rate": float("nan"),
            "mean_delta_ball": float("nan"),
            "mean_delta_standard": float("nan"),
            "mean_ball_to_standard_ratio": float("nan"),
            "max_direct_c_ball": float("nan"),
            "max_direct_c_standard": float("nan"),
        }
    df = pd.DataFrame(step_rows)

    def mean_col(name: str) -> float:
        return float(pd.to_numeric(df.get(name), errors="coerce").mean())

    def max_col(name: str) -> float:
        return float(pd.to_numeric(df.get(name), errors="coerce").max())

    return {
        "mean_sample_rate": mean_col("sample_rate"),
        "mean_delta_ball": mean_col("delta_ball"),
        "mean_delta_standard": mean_col("delta_standard"),
        "mean_ball_to_standard_ratio": mean_col("ball_to_standard_ratio"),
        "max_direct_c_ball": max_col("direct_c_ball"),
        "max_direct_c_standard": max_col("direct_c_standard"),
    }


def run_id_for_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha1(
        json.dumps(json_default(payload), sort_keys=True).encode()
    ).hexdigest()[:12]


def dataset_output_dir(results_root: str | Path, dataset_tag: str) -> Path:
    return Path(results_root) / str(dataset_tag)


def rebuild_dataset_outputs(
    dataset_dir: Path, *, seed_filename: str, summary_filename: str
) -> None:
    frames: list[pd.DataFrame] = []
    for path in sorted((Path(dataset_dir) / "runs").glob(f"*/{seed_filename}")):
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
        raw, Path(dataset_dir) / seed_filename, save_parquet_if_possible=False
    )

    group_cols = [
        c
        for c in [
            "dataset_tag",
            "dataset_name",
            "mechanism",
            "epsilon",
            "m",
            "radius_tag",
            "task",
            "hidden_dim",
            "num_steps",
            "batch_size",
            "clip_norm",
        ]
        if c in raw.columns
    ]
    numeric_cols = [
        c
        for c in raw.columns
        if c not in group_cols and pd.api.types.is_numeric_dtype(raw[c])
    ]
    if not group_cols or not numeric_cols:
        return
    grouped = raw.groupby(group_cols, dropna=False)
    rows = []
    for key, g in grouped:
        row = {}
        if not isinstance(key, tuple):
            key = (key,)
        for col, val in zip(group_cols, key, strict=True):
            row[col] = val
        row["n_rows"] = int(len(g))
        for col in numeric_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = (
                float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    save_dataframe(
        summary, Path(dataset_dir) / summary_filename, save_parquet_if_possible=False
    )


def make_dataset_metadata_row(
    *,
    data: LoadedDataset,
    radius: float,
    radius_tag: str,
    spec: TheoremModelSpec,
    bounds: TheoremBounds,
    lz: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "dataset_tag": data.spec.tag,
        "dataset_name": data.spec.display_name,
        "n_train": int(len(data.X_train)),
        "n_test": int(len(data.X_test)),
        "feature_dim": int(data.feature_dim),
        "num_classes": int(data.num_classes),
        "radius_tag": str(radius_tag),
        "radius_value": float(radius),
        "task": str(spec.task),
        "hidden_dim": int(spec.hidden_dim),
        "constraint": str(spec.constraint),
        "parameterization": str(spec.parameterization),
        "B": float(bounds.B),
        "A": float(bounds.A),
        "Lambda": float(bounds.Lambda),
        "certified_lz": float(lz),
        "critical_radius_for_clip": (
            float(2.0 * float(args.clip_norm) / float(lz)) if lz > 0 else float("inf")
        ),
        "num_steps": int(args.num_steps),
        "batch_size": int(args.batch_size),
        "sample_rate": float(args.batch_size) / float(len(data.X_train)),
        "clip_norm": float(args.clip_norm),
        "learning_rate": float(args.learning_rate),
        "delta": float(args.delta),
    }
