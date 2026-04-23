#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import jax

from quantbayes.ball_dp import (
    attack_convex_ball_output_finite_prior,
    ball_rero,
    fit_convex,
    make_finite_identification_prior,
    select_ball_radius,
    summarize_embedding_ball_radii,
)
from quantbayes.ball_dp.serialization import save_dataframe

DEFAULT_EPS_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_M_GRID = (8, 16, 32, 64)
DEFAULT_RELEASE_SEEDS = (0, 1, 2)
DEFAULT_RADIUS_TAG = "q80"
DEFAULT_FIXED_EPSILON = 1.0
DEFAULT_FIXED_M = 16
DEFAULT_DELTA = 1e-6
DEFAULT_EMBEDDING_BOUND = 1.0
DEFAULT_LAM = 1e-2
DEFAULT_MAX_ITER = 100
DEFAULT_ORDERS = tuple(list(range(2, 65)) + [80, 96, 128, 160, 192, 256])

MODEL_ALIASES = {
    "ridge_prototype": "ridge_prototype",
    "ridge-prototype": "ridge_prototype",
    "softmax_logistic": "softmax_logistic",
    "softmax-logistic": "softmax_logistic",
    "binary_logistic": "binary_logistic",
    "binary-logistic": "binary_logistic",
    "sqr_hinge_svm": "squared_hinge",
    "squared_hinge": "squared_hinge",
    "squared-hinge": "squared_hinge",
    "svm": "squared_hinge",
}

RADIUS_TAG_TO_QUANTILE = {
    "q50": 0.50,
    "q80": 0.80,
    "q95": 0.95,
}


@dataclass(frozen=True)
class DatasetSpec:
    canonical_name: str
    tag: str
    module: str
    loader_fn: str
    kind: str
    aliases: tuple[str, ...]
    display_name: str
    extra_kwargs: dict[str, Any] | None = None


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        canonical_name="MNIST-embeddings",
        tag="mnist",
        module="quantbayes.ball_dp.experiments.load_mnist_embeddings",
        loader_fn="load_or_create_mnist_resnet18_embeddings",
        kind="image",
        aliases=("mnist", "mnist_embeddings", "mnist-embeddings"),
        display_name="MNIST-embeddings",
    ),
    DatasetSpec(
        canonical_name="AG News-embeddings",
        tag="ag_news",
        module="quantbayes.ball_dp.experiments.load_ag_news_embeddings",
        loader_fn="load_or_create_ag_news_text_embeddings",
        kind="text",
        aliases=(
            "ag_news",
            "ag-news",
            "ag_news_embeddings",
            "ag news-embeddings",
            "ag news",
        ),
        display_name="AG News-embeddings",
    ),
    DatasetSpec(
        canonical_name="BANKING77-embeddings",
        tag="banking77",
        module="quantbayes.ball_dp.experiments.load_banking77_embeddings",
        loader_fn="load_or_create_banking77_text_embeddings",
        kind="text",
        aliases=("banking77", "banking77_embeddings", "banking77-embeddings"),
        display_name="BANKING77-embeddings",
    ),
    DatasetSpec(
        canonical_name="CIFAR-10-embeddings",
        tag="cifar10",
        module="quantbayes.ball_dp.experiments.load_cifar10_embeddings",
        loader_fn="load_or_create_cifar10_resnet18_embeddings",
        kind="image",
        aliases=("cifar10", "cifar-10", "cifar10_embeddings", "cifar-10-embeddings"),
        display_name="CIFAR-10-embeddings",
    ),
    DatasetSpec(
        canonical_name="DBpedia-14-embeddings",
        tag="dbpedia14",
        module="quantbayes.ball_dp.experiments.load_dbpedia14_embeddings",
        loader_fn="load_or_create_dbpedia14_text_embeddings",
        kind="text",
        aliases=(
            "dbpedia14",
            "dbpedia-14",
            "dbpedia14_embeddings",
            "dbpedia-14-embeddings",
        ),
        display_name="DBpedia-14-embeddings",
    ),
    DatasetSpec(
        canonical_name="Emotion-embeddings",
        tag="emotion",
        module="quantbayes.ball_dp.experiments.load_emotion_embeddings",
        loader_fn="load_or_create_emotion_text_embeddings",
        kind="text",
        aliases=("emotion", "emotion_embeddings", "emotion-embeddings"),
        display_name="Emotion-embeddings",
    ),
    DatasetSpec(
        canonical_name="Yelp Review Full-embeddings",
        tag="yelp_review_full",
        module="quantbayes.ball_dp.experiments.load_yelp_review_full_embeddings",
        loader_fn="load_or_create_yelp_review_full_text_embeddings",
        kind="text",
        aliases=(
            "yelp_review_full",
            "yelp-review-full",
            "yelp review full",
            "yelp review full-embeddings",
            "yelp review full_embeddings",
        ),
        display_name="Yelp Review Full-embeddings",
    ),
    DatasetSpec(
        canonical_name="IMDb-embeddings",
        tag="imdb",
        module="quantbayes.ball_dp.experiments.load_imdb_embeddings",
        loader_fn="load_or_create_imdb_text_embeddings",
        kind="text",
        aliases=("imdb", "imdb_embeddings", "imdb-embeddings"),
        display_name="IMDb-embeddings",
    ),
    DatasetSpec(
        canonical_name="TREC-6-embeddings",
        tag="trec6",
        module="quantbayes.ball_dp.experiments.load_trec_embeddings",
        loader_fn="load_or_create_trec_text_embeddings",
        kind="text",
        aliases=("trec6", "trec-6", "trec6_embeddings", "trec-6-embeddings"),
        display_name="TREC-6-embeddings",
        extra_kwargs={"label_space": "coarse"},
    ),
)


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
            "lines.linewidth": 2.3,
            "lines.markersize": 7,
        }
    )


BALL_COLOR = "#1f77b4"
STANDARD_COLOR = "#d62728"
BASELINE_COLOR = "#6b7280"


def canonicalize_model(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_")
    if key not in MODEL_ALIASES:
        raise ValueError(
            "Unsupported model. Choose one of "
            "{'ridge_prototype', 'softmax_logistic', 'binary_logistic', 'sqr_hinge_svm'}."
        )
    return MODEL_ALIASES[key]


def resolve_dataset(name: str) -> DatasetSpec:
    key = str(name).strip().lower().replace(" ", "_")
    for spec in DATASETS:
        all_keys = {spec.canonical_name.lower().replace(" ", "_")} | {
            alias.lower().replace(" ", "_") for alias in spec.aliases
        }
        if key in all_keys:
            return spec
    supported = ", ".join(spec.canonical_name for spec in DATASETS)
    raise ValueError(f"Unknown dataset {name!r}. Supported datasets: {supported}")


def as_float_list(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values]


def as_int_list(values: Sequence[int]) -> list[int]:
    return [int(v) for v in values]


def qkey(value: float) -> str:
    return f"q{float(value):.2f}".replace(".", "")


def slugify(text: str) -> str:
    out = []
    for ch in str(text).lower():
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def savefig_stem(fig: plt.Figure, stem: str | Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def actual_ball_epsilon(release: Any) -> Optional[float]:
    certs = getattr(release.privacy.ball, "dp_certificates", [])
    if not certs:
        return None
    return maybe_float(certs[0].epsilon)


def actual_standard_epsilon_same_noise(release: Any) -> Optional[float]:
    certs = getattr(release.privacy.standard, "dp_certificates", [])
    if not certs:
        return None
    return maybe_float(certs[0].epsilon)


def finite_prior_hash(X: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha1()
    x_rounded = np.round(np.asarray(X, dtype=np.float32), 7)
    y_arr = np.asarray(y, dtype=np.int32)
    h.update(x_rounded.tobytes())
    h.update(y_arr.tobytes())
    return h.hexdigest()[:16]


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_values: np.ndarray
    num_classes: int
    feature_dim: int
    empirical_embedding_bound: float
    backend: str


@dataclass
class TargetPool:
    target_index: int
    target_label: int
    target_vector: np.ndarray
    distractor_vectors: np.ndarray
    distractor_source_ids: list[str]
    distractor_distances: np.ndarray


def load_embeddings(args: argparse.Namespace, spec: DatasetSpec) -> LoadedDataset:
    module = importlib.import_module(spec.module)
    loader = getattr(module, spec.loader_fn)

    common_kwargs: dict[str, Any] = {
        "batch_size": int(args.embedding_batch_size),
        "require_jax_gpu": not bool(args.allow_cpu_fallback),
        "cache_path": (
            None
            if args.embedding_cache_path is None
            else str(args.embedding_cache_path)
        ),
        "force_recompute": bool(args.force_recompute_embeddings),
    }
    if spec.extra_kwargs:
        common_kwargs.update(spec.extra_kwargs)

    if spec.kind == "image":
        common_kwargs.update(
            {
                "data_root": str(args.data_root),
                "num_workers": int(args.num_workers),
            }
        )
    elif spec.kind == "text":
        common_kwargs.update(
            {
                "output_root": str(args.data_root),
                "model_name": str(args.encoder_model_name),
                "max_length": int(args.max_length),
                "hf_cache_dir": (
                    None if args.hf_cache_dir is None else str(args.hf_cache_dir)
                ),
            }
        )
    else:
        raise ValueError(f"Unsupported dataset kind: {spec.kind}")

    X_train, y_train, X_test, y_test = loader(**common_kwargs)
    X_train_np = np.asarray(jax.device_get(X_train), dtype=np.float32)
    y_train_np = np.asarray(jax.device_get(y_train), dtype=np.int32).reshape(-1)
    X_test_np = np.asarray(jax.device_get(X_test), dtype=np.float32)
    y_test_np = np.asarray(jax.device_get(y_test), dtype=np.int32).reshape(-1)

    # Ensure contiguous class labels shared across train/test.
    label_values = np.unique(np.concatenate([y_train_np, y_test_np], axis=0))
    mapping = {int(v): i for i, v in enumerate(label_values.tolist())}
    y_train_np = np.asarray([mapping[int(v)] for v in y_train_np], dtype=np.int32)
    y_test_np = np.asarray([mapping[int(v)] for v in y_test_np], dtype=np.int32)

    empirical_bound = float(np.max(np.linalg.norm(X_train_np, axis=1)))

    return LoadedDataset(
        spec=spec,
        X_train=X_train_np,
        y_train=y_train_np,
        X_test=X_test_np,
        y_test=y_test_np,
        label_values=label_values.astype(np.int32, copy=False),
        num_classes=int(len(np.unique(y_train_np))),
        feature_dim=int(X_train_np.shape[1]),
        empirical_embedding_bound=empirical_bound,
        backend=jax.default_backend(),
    )


def validate_model_dataset_compatibility(
    model_family: str, data: LoadedDataset
) -> None:
    if model_family in {"binary_logistic", "squared_hinge"} and data.num_classes != 2:
        raise ValueError(
            f"Model {model_family!r} is binary-only, but dataset {data.spec.display_name!r} "
            f"has {data.num_classes} classes."
        )


def radius_value_from_report(report: dict[str, Any], radius_tag: str) -> float:
    if radius_tag not in RADIUS_TAG_TO_QUANTILE:
        raise ValueError(
            f"Unsupported radius tag {radius_tag!r}. Use one of {tuple(RADIUS_TAG_TO_QUANTILE)}"
        )
    quantile = RADIUS_TAG_TO_QUANTILE[radius_tag]
    return float(
        select_ball_radius(
            report,
            strategy="max_labelwise_quantile",
            quantile=quantile,
            allow_observed_max=False,
        )
    )


def make_run_id(args: argparse.Namespace, spec: DatasetSpec, model_family: str) -> str:
    payload = {
        "dataset": spec.tag,
        "model": model_family,
        "radius": args.radius,
        "lam": float(args.lam),
        "delta": float(args.delta),
        "eps_grid": as_float_list(args.epsilon_grid),
        "m_grid": as_int_list(args.m_grid),
        "fixed_epsilon": float(args.fixed_epsilon),
        "fixed_m": int(args.fixed_m),
        "release_seeds": as_int_list(args.release_seeds),
        "num_targets": int(args.num_targets),
        "candidate_draws": int(args.candidate_draws),
        "target_seed": int(args.target_seed),
        "solver": str(args.solver),
        "max_iter": int(args.max_iter),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return digest


def seed_seq(*parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(p) for p in parts]))


def dedup_vectors(
    vectors: np.ndarray,
    source_ids: Sequence[str],
    distances: np.ndarray,
    *,
    exclude_value: Optional[np.ndarray] = None,
    rounding_decimals: int = 7,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    seen: set[bytes] = set()
    kept_vectors: list[np.ndarray] = []
    kept_ids: list[str] = []
    kept_dists: list[float] = []
    exclude_key = None
    if exclude_value is not None:
        exclude_key = np.round(
            np.asarray(exclude_value, dtype=np.float32), rounding_decimals
        ).tobytes()
    for vec, src, dist in zip(vectors, source_ids, distances, strict=True):
        key = np.round(np.asarray(vec, dtype=np.float32), rounding_decimals).tobytes()
        if exclude_key is not None and key == exclude_key:
            continue
        if key in seen:
            continue
        seen.add(key)
        kept_vectors.append(np.asarray(vec, dtype=np.float32))
        kept_ids.append(str(src))
        kept_dists.append(float(dist))
    if not kept_vectors:
        return (
            np.zeros((0, int(vectors.shape[1])), dtype=np.float32),
            [],
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.stack(kept_vectors, axis=0).astype(np.float32, copy=False),
        kept_ids,
        np.asarray(kept_dists, dtype=np.float32),
    )


def build_target_pool(
    data: LoadedDataset,
    *,
    target_index: int,
    radius_value: float,
    tol: float = 1e-7,
) -> TargetPool:
    x_target = np.asarray(data.X_train[target_index], dtype=np.float32)
    target_label = int(data.y_train[target_index])

    train_same = np.where(data.y_train == target_label)[0]
    train_same = train_same[train_same != int(target_index)]
    test_same = np.where(data.y_test == target_label)[0]

    train_vectors = data.X_train[train_same]
    test_vectors = data.X_test[test_same]
    train_dists = (
        np.linalg.norm(train_vectors - x_target[None, :], axis=1)
        if train_vectors.size
        else np.zeros((0,), dtype=np.float32)
    )
    test_dists = (
        np.linalg.norm(test_vectors - x_target[None, :], axis=1)
        if test_vectors.size
        else np.zeros((0,), dtype=np.float32)
    )

    keep_train = train_dists <= float(radius_value) + float(tol)
    keep_test = test_dists <= float(radius_value) + float(tol)

    kept_vectors = np.concatenate(
        [train_vectors[keep_train], test_vectors[keep_test]], axis=0
    )
    kept_ids = [f"train:{int(i)}" for i in train_same[keep_train].tolist()] + [
        f"test:{int(i)}" for i in test_same[keep_test].tolist()
    ]
    kept_dists = np.concatenate(
        [train_dists[keep_train], test_dists[keep_test]], axis=0
    )

    kept_vectors, kept_ids, kept_dists = dedup_vectors(
        kept_vectors,
        kept_ids,
        kept_dists,
        exclude_value=x_target,
    )

    return TargetPool(
        target_index=int(target_index),
        target_label=target_label,
        target_vector=x_target,
        distractor_vectors=kept_vectors,
        distractor_source_ids=kept_ids,
        distractor_distances=kept_dists,
    )


def find_feasible_targets(
    data: LoadedDataset,
    *,
    radius_value: float,
    max_required_m: int,
    num_targets: int,
    target_seed: int,
    max_search: Optional[int],
    explicit_target_indices: Optional[Sequence[int]] = None,
    strict: bool = False,
) -> list[TargetPool]:
    if explicit_target_indices:
        candidate_indices = [int(i) for i in explicit_target_indices]
    else:
        rng = np.random.default_rng(int(target_seed))
        candidate_indices = rng.permutation(len(data.X_train)).tolist()
        if max_search is not None:
            candidate_indices = candidate_indices[: int(max_search)]

    pools: list[TargetPool] = []
    for idx in candidate_indices:
        pool = build_target_pool(
            data, target_index=int(idx), radius_value=float(radius_value)
        )
        if int(pool.distractor_vectors.shape[0]) >= int(max_required_m) - 1:
            pools.append(pool)
        if len(pools) >= int(num_targets):
            break

    if len(pools) < int(num_targets) and strict:
        raise RuntimeError(
            f"Requested {num_targets} feasible targets, but found only {len(pools)} "
            f"for radius {radius_value:.6f} and m_max={max_required_m}."
        )
    if not pools:
        raise RuntimeError(
            f"No feasible targets found for radius {radius_value:.6f} and m_max={max_required_m}."
        )
    return pools


def make_candidate_set(
    pool: TargetPool,
    *,
    m: int,
    draw_index: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if int(m) < 2:
        raise ValueError(
            "m must be at least 2 so the support contains the truth and at least one distractor."
        )
    available = int(pool.distractor_vectors.shape[0])
    need = int(m) - 1
    if available < need:
        raise ValueError(
            f"Target {pool.target_index} only has {available} distractors in-radius, but m={m} needs {need}."
        )

    rng = seed_seq(base_seed, pool.target_index, draw_index)
    perm = rng.permutation(available)
    chosen = perm[:need]
    distractors = np.asarray(pool.distractor_vectors[chosen], dtype=np.float32)
    distractor_ids = [pool.distractor_source_ids[int(i)] for i in chosen.tolist()]
    distractor_dists = np.asarray(pool.distractor_distances[chosen], dtype=np.float32)

    X_candidates = np.concatenate(
        [distractors, pool.target_vector[None, :]],
        axis=0,
    ).astype(np.float32, copy=False)
    y_candidates = np.full((int(m),), int(pool.target_label), dtype=np.int32)
    source_ids = list(distractor_ids) + [f"target:{pool.target_index}"]
    dists = np.concatenate(
        [distractor_dists, np.array([0.0], dtype=np.float32)], axis=0
    )

    # Randomize final candidate order to avoid a fixed truth position.
    shuffle_rng = seed_seq(base_seed, pool.target_index, draw_index, int(m), 17)
    order = shuffle_rng.permutation(int(m))
    X_candidates = X_candidates[order]
    y_candidates = y_candidates[order]
    source_ids = [source_ids[int(i)] for i in order.tolist()]
    dists = dists[order]
    return X_candidates, y_candidates, source_ids, dists


def fit_release_for_mechanism(
    *,
    data: LoadedDataset,
    model_family: str,
    mechanism: str,
    epsilon: float,
    local_radius_value: float,
    embedding_bound: float,
    lam: float,
    delta: float,
    max_iter: int,
    seed: int,
    orders: Sequence[float],
    solver: str,
) -> Any:
    if mechanism == "ball":
        release_radius = float(local_radius_value)
    elif mechanism == "standard":
        release_radius = float(2.0 * embedding_bound)
    else:
        raise ValueError(f"Unknown mechanism {mechanism!r}")

    return fit_convex(
        data.X_train,
        data.y_train,
        X_eval=data.X_test,
        y_eval=data.y_test,
        model_family=model_family,
        privacy="ball_dp",
        radius=float(release_radius),
        lam=float(lam),
        epsilon=float(epsilon),
        delta=float(delta),
        embedding_bound=float(embedding_bound),
        num_classes=int(data.num_classes),
        orders=tuple(float(v) for v in orders),
        max_iter=int(max_iter),
        solver=str(solver),
        seed=int(seed),
    )


def make_dummy_finite_prior(m: int, feature_dim: int) -> Any:
    samples = np.zeros((int(m), int(feature_dim)), dtype=np.float32)
    return make_finite_identification_prior(samples, weights=None)


def compute_release_bound_metrics(
    release: Any,
    *,
    m: int,
    feature_dim: int,
) -> dict[str, float]:
    prior = make_dummy_finite_prior(int(m), int(feature_dim))
    eta_grid = (0.5,)

    out: dict[str, float] = {
        "bound_generic_dp": float("nan"),
        "bound_direct": float("nan"),
        "bound_rdp": float("nan"),
        "bound_direct_standard_same_noise": float("nan"),
        "oblivious_kappa": float("nan"),
    }
    report_dp = ball_rero(release, prior=prior, eta_grid=eta_grid, mode="dp")
    report_direct = ball_rero(
        release, prior=prior, eta_grid=eta_grid, mode="gaussian_direct"
    )
    out["bound_generic_dp"] = float(report_dp.points[0].gamma_ball)
    out["bound_direct"] = float(report_direct.points[0].gamma_ball)
    if report_direct.points[0].gamma_standard is not None:
        out["bound_direct_standard_same_noise"] = float(
            report_direct.points[0].gamma_standard
        )
    try:
        report_rdp = ball_rero(release, prior=prior, eta_grid=eta_grid, mode="rdp")
        out["bound_rdp"] = float(report_rdp.points[0].gamma_ball)
    except Exception:
        out["bound_rdp"] = float("nan")
    out["oblivious_kappa"] = float(report_direct.points[0].kappa)
    return out


def summarize_bernoulli(p: float, n: int) -> tuple[float, float]:
    p = float(p)
    n = int(n)
    if n <= 0:
        return float("nan"), float("nan")
    se = math.sqrt(max(p * (1.0 - p), 0.0) / float(n))
    lo = max(0.0, p - 1.96 * se)
    hi = min(1.0, p + 1.96 * se)
    return lo, hi


def numeric_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(np.mean(np.asarray(series, dtype=float)))


def numeric_std(series: pd.Series) -> float:
    arr = np.asarray(series, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def rebuild_attack_dataset_outputs(
    dataset_dir: Path,
    *,
    spec: DatasetSpec,
    model_family: str,
    plot_radius_tag: str,
    fixed_epsilon: float,
    fixed_m: int,
) -> None:
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

    trial_df = pd.concat(trial_frames, ignore_index=True)
    if "trial_key" in trial_df.columns:
        trial_df = trial_df.drop_duplicates(subset=["trial_key"], keep="last")
    save_dataframe(
        trial_df, dataset_dir / "attack_trial_rows.csv", save_parquet_if_possible=False
    )

    if release_frames:
        release_df = pd.concat(release_frames, ignore_index=True)
        if "release_key" in release_df.columns:
            release_df = release_df.drop_duplicates(subset=["release_key"], keep="last")
        save_dataframe(
            release_df, dataset_dir / "release_rows.csv", save_parquet_if_possible=False
        )
    else:
        release_df = pd.DataFrame()

    config_cols = [
        "dataset_tag",
        "dataset_name",
        "model_family",
        "mechanism",
        "candidate_radius_tag",
        "candidate_radius_value",
        "release_radius_value",
        "epsilon",
        "m",
        "delta",
        "lam",
        "embedding_bound",
        "solver",
        "max_iter",
    ]

    trial_group = trial_df.groupby(config_cols, dropna=False)
    summary = trial_group.agg(
        n_runs=("run_id", "nunique"),
        n_trials=("exact_identification_success", "count"),
        num_targets=("target_index", "nunique"),
        candidate_draws=("candidate_draw_index", "nunique"),
        release_seeds=("release_seed", "nunique"),
        exact_id_mean=("exact_identification_success", "mean"),
        oblivious_kappa=("oblivious_kappa", "mean"),
        candidate_pool_size_mean=("candidate_pool_size", "mean"),
        candidate_max_distance_mean=("candidate_max_distance", "mean"),
    ).reset_index()

    ci_bounds = [
        summarize_bernoulli(p, n)
        for p, n in zip(summary["exact_id_mean"], summary["n_trials"], strict=True)
    ]
    summary["exact_id_ci_low"] = [lo for lo, _ in ci_bounds]
    summary["exact_id_ci_high"] = [hi for _, hi in ci_bounds]
    summary["attack_advantage"] = summary["exact_id_mean"] - summary["oblivious_kappa"]

    if not release_df.empty:
        release_metric_cols = [
            "dataset_tag",
            "dataset_name",
            "model_family",
            "mechanism",
            "candidate_radius_tag",
            "candidate_radius_value",
            "release_radius_value",
            "epsilon",
            "m",
            "delta",
            "lam",
            "embedding_bound",
            "solver",
            "max_iter",
            "release_seed",
            "release_sigma",
            "release_accuracy",
            "actual_ball_epsilon",
            "same_noise_standard_epsilon",
            "bound_generic_dp",
            "bound_direct",
            "bound_rdp",
            "bound_direct_standard_same_noise",
        ]
        existing_cols = [c for c in release_metric_cols if c in release_df.columns]
        rel = release_df[existing_cols].drop_duplicates()
        metric_group = (
            rel.groupby(config_cols, dropna=False)
            .agg(
                release_sigma_mean=("release_sigma", "mean"),
                release_sigma_std=("release_sigma", numeric_std),
                release_accuracy_mean=("release_accuracy", "mean"),
                release_accuracy_std=("release_accuracy", numeric_std),
                actual_ball_epsilon_mean=("actual_ball_epsilon", "mean"),
                same_noise_standard_epsilon_mean=(
                    "same_noise_standard_epsilon",
                    "mean",
                ),
                bound_generic_dp_mean=("bound_generic_dp", "mean"),
                bound_direct_mean=("bound_direct", "mean"),
                bound_rdp_mean=("bound_rdp", "mean"),
                bound_direct_standard_same_noise_mean=(
                    "bound_direct_standard_same_noise",
                    "mean",
                ),
            )
            .reset_index()
        )
        summary = summary.merge(metric_group, on=config_cols, how="left")

    summary = summary.sort_values(
        ["mechanism", "candidate_radius_tag", "epsilon", "m"]
    ).reset_index(drop=True)
    save_dataframe(
        summary, dataset_dir / "attack_summary.csv", save_parquet_if_possible=False
    )

    figures_dir = ensure_dir(dataset_dir / "figures")
    dataset_title = f"{spec.display_name} · {model_family}"

    # Exact-ID vs epsilon (fixed m).
    eps_mask = (summary["candidate_radius_tag"] == plot_radius_tag) & (
        summary["m"] == int(fixed_m)
    )
    eps_df = summary[eps_mask].copy()
    if not eps_df.empty:
        fig, ax = plt.subplots()
        for mechanism, color, label in (
            ("ball", BALL_COLOR, "Ball-DP"),
            ("standard", STANDARD_COLOR, "Standard DP"),
        ):
            sub = eps_df[eps_df["mechanism"] == mechanism].sort_values("epsilon")
            if sub.empty:
                continue
            ax.plot(
                sub["epsilon"],
                sub["exact_id_mean"],
                marker="o",
                color=color,
                label=label,
            )
            ax.fill_between(
                sub["epsilon"],
                sub["exact_id_ci_low"],
                sub["exact_id_ci_high"],
                color=color,
                alpha=0.18,
            )
        baseline = 1.0 / float(fixed_m)
        ax.axhline(
            baseline,
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=2.0,
            label=f"Oblivious baseline = 1/{fixed_m}",
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("$\\varepsilon$")
        ax.set_ylabel("Empirical exact-ID")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            f"Exact-ID vs $\\varepsilon$\n{dataset_title}, radius={plot_radius_tag}, m={fixed_m}"
        )
        ax.legend(ncol=1)
        savefig_stem(
            fig,
            figures_dir / f"fig_attack_exactid_vs_epsilon_{spec.tag}_{model_family}",
        )

    # Exact-ID vs m (fixed epsilon).
    m_mask = (summary["candidate_radius_tag"] == plot_radius_tag) & np.isclose(
        summary["epsilon"], float(fixed_epsilon)
    )
    m_df = summary[m_mask].copy()
    if not m_df.empty:
        fig, ax = plt.subplots()
        for mechanism, color, label in (
            ("ball", BALL_COLOR, "Ball-DP"),
            ("standard", STANDARD_COLOR, "Standard DP"),
        ):
            sub = m_df[m_df["mechanism"] == mechanism].sort_values("m")
            if sub.empty:
                continue
            ax.plot(
                sub["m"], sub["exact_id_mean"], marker="o", color=color, label=label
            )
            ax.fill_between(
                sub["m"],
                sub["exact_id_ci_low"],
                sub["exact_id_ci_high"],
                color=color,
                alpha=0.18,
            )
        m_sorted = np.sort(m_df["m"].unique())
        if m_sorted.size:
            baseline = 1.0 / m_sorted.astype(float)
            ax.plot(
                m_sorted,
                baseline,
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=2.0,
                marker="s",
                label="Oblivious baseline = 1/m",
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("m")
        ax.set_ylabel("Empirical exact-ID")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            f"Exact-ID vs m\n{dataset_title}, radius={plot_radius_tag}, $\\varepsilon$={fixed_epsilon:g}"
        )
        ax.legend(ncol=1)
        savefig_stem(
            fig, figures_dir / f"fig_attack_exactid_vs_m_{spec.tag}_{model_family}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run convex finite-prior exact-identification attack experiments for Ball-DP "
            "and the matched standard-DP comparator. Candidate supports are target-centered, "
            "same-label, and constrained to the selected local Ball radius."
        )
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--results-root", type=str, default="results")

    parser.add_argument(
        "--radius", choices=tuple(RADIUS_TAG_TO_QUANTILE), default=DEFAULT_RADIUS_TAG
    )
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument(
        "--embedding-bound", type=float, default=DEFAULT_EMBEDDING_BOUND
    )
    parser.add_argument("--lam", type=float, default=DEFAULT_LAM)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument(
        "--solver",
        choices=("lbfgs_fullbatch", "gd_fullbatch"),
        default="lbfgs_fullbatch",
    )

    parser.add_argument(
        "--epsilon-grid", nargs="+", type=float, default=list(DEFAULT_EPS_GRID)
    )
    parser.add_argument("--m-grid", nargs="+", type=int, default=list(DEFAULT_M_GRID))
    parser.add_argument("--fixed-epsilon", type=float, default=DEFAULT_FIXED_EPSILON)
    parser.add_argument("--fixed-m", type=int, default=DEFAULT_FIXED_M)
    parser.add_argument("--sweep", choices=("epsilon", "m", "both"), default="both")
    parser.add_argument(
        "--mechanisms", choices=("ball", "standard", "both"), default="both"
    )

    parser.add_argument(
        "--release-seeds", nargs="*", type=int, default=list(DEFAULT_RELEASE_SEEDS)
    )
    parser.add_argument("--num-targets", type=int, default=8)
    parser.add_argument("--candidate-draws", type=int, default=3)
    parser.add_argument("--target-seed", type=int, default=0)
    parser.add_argument("--max-feasible-search", type=int, default=None)
    parser.add_argument("--target-indices", nargs="*", type=int, default=None)
    parser.add_argument("--strict-feasible-targets", action="store_true")

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

    parser.add_argument("--max-exact-pairs", type=int, default=250_000)
    parser.add_argument("--max-sampled-pairs", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    spec = resolve_dataset(args.dataset)
    model_family = canonicalize_model(args.model)
    mechanisms = (
        ["ball", "standard"] if args.mechanisms == "both" else [str(args.mechanisms)]
    )

    eps_grid = sorted(set(as_float_list(args.epsilon_grid)))
    m_grid = sorted(set(as_int_list(args.m_grid)))
    if int(args.fixed_m) < 2:
        raise ValueError("fixed_m must be at least 2.")
    if any(int(m) < 2 for m in m_grid):
        raise ValueError("All m-grid values must be at least 2.")

    data = load_embeddings(args, spec)
    validate_model_dataset_compatibility(model_family, data)

    if float(data.empirical_embedding_bound) > float(args.embedding_bound) + 1e-3:
        raise ValueError(
            f"Empirical training embedding bound {data.empirical_embedding_bound:.6f} exceeds "
            f"the supplied public bound {float(args.embedding_bound):.6f}."
        )

    radius_report = summarize_embedding_ball_radii(
        data.X_train,
        data.y_train,
        quantiles=(0.50, 0.80, 0.95),
        max_exact_pairs=int(args.max_exact_pairs),
        max_sampled_pairs=int(args.max_sampled_pairs),
        seed=int(args.target_seed),
    )
    local_radius_value = radius_value_from_report(radius_report, args.radius)

    unique_epsilons = set()
    if args.sweep in {"epsilon", "both"}:
        unique_epsilons.update(eps_grid)
    if args.sweep in {"m", "both"}:
        unique_epsilons.add(float(args.fixed_epsilon))
    unique_epsilons = sorted(unique_epsilons)

    required_ms = set()
    if args.sweep in {"m", "both"}:
        required_ms.update(m_grid)
    if args.sweep in {"epsilon", "both"}:
        required_ms.add(int(args.fixed_m))
    required_ms = sorted(required_ms)
    max_required_m = max(required_ms)

    feasible_pools = find_feasible_targets(
        data,
        radius_value=float(local_radius_value),
        max_required_m=int(max_required_m),
        num_targets=int(args.num_targets),
        target_seed=int(args.target_seed),
        max_search=args.max_feasible_search,
        explicit_target_indices=args.target_indices,
        strict=bool(args.strict_feasible_targets),
    )

    results_root = Path(args.results_root)
    dataset_dir = ensure_dir(results_root / "attack" / model_family / spec.tag)
    run_id = make_run_id(args, spec, model_family)
    run_dir = ensure_dir(dataset_dir / "runs" / run_id)

    write_json(
        run_dir / "run_config.json",
        {
            **vars(args),
            "dataset_spec": asdict(spec),
            "model_family": model_family,
        },
    )
    write_json(run_dir / "radius_report.json", radius_report)
    write_json(
        run_dir / "dataset_metadata.json",
        {
            "dataset_name": spec.display_name,
            "dataset_tag": spec.tag,
            "num_train": int(data.X_train.shape[0]),
            "num_test": int(data.X_test.shape[0]),
            "num_classes": int(data.num_classes),
            "feature_dim": int(data.feature_dim),
            "empirical_embedding_bound": float(data.empirical_embedding_bound),
            "public_embedding_bound": float(args.embedding_bound),
            "candidate_radius_tag": args.radius,
            "candidate_radius_value": float(local_radius_value),
            "backend": data.backend,
            "release_seeds": as_int_list(args.release_seeds),
            "num_feasible_targets": int(len(feasible_pools)),
        },
    )

    # Precompute candidate sets and save inspectable metadata.
    candidate_sets_meta: list[dict[str, Any]] = []
    candidate_sets: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray, list[str], np.ndarray]
    ] = {}
    for pool in feasible_pools:
        for draw_index in range(int(args.candidate_draws)):
            for m in required_ms:
                Xc, yc, source_ids, dists = make_candidate_set(
                    pool,
                    m=int(m),
                    draw_index=int(draw_index),
                    base_seed=int(args.target_seed),
                )
                c_hash = finite_prior_hash(Xc, yc)
                candidate_sets[(pool.target_index, int(draw_index), int(m))] = (
                    Xc,
                    yc,
                    source_ids,
                    dists,
                )
                candidate_sets_meta.append(
                    {
                        "target_index": int(pool.target_index),
                        "target_label": int(pool.target_label),
                        "candidate_draw_index": int(draw_index),
                        "m": int(m),
                        "candidate_set_hash": c_hash,
                        "candidate_source_ids": source_ids,
                        "candidate_max_distance": float(np.max(dists)),
                        "candidate_mean_distance": float(np.mean(dists)),
                        "candidate_min_distance": float(np.min(dists)),
                        "candidate_pool_size": int(pool.distractor_vectors.shape[0]),
                    }
                )
    write_json(run_dir / "candidate_sets.json", candidate_sets_meta)

    # Precompute releases for every mechanism/epsilon/seed pair.
    releases: dict[tuple[str, float, int], Any] = {}
    release_rows: list[dict[str, Any]] = []
    for mechanism in mechanisms:
        for epsilon in unique_epsilons:
            for release_seed in as_int_list(args.release_seeds):
                release = fit_release_for_mechanism(
                    data=data,
                    model_family=model_family,
                    mechanism=mechanism,
                    epsilon=float(epsilon),
                    local_radius_value=float(local_radius_value),
                    embedding_bound=float(args.embedding_bound),
                    lam=float(args.lam),
                    delta=float(args.delta),
                    max_iter=int(args.max_iter),
                    seed=int(release_seed),
                    orders=DEFAULT_ORDERS,
                    solver=str(args.solver),
                )
                releases[(mechanism, float(epsilon), int(release_seed))] = release

                for m in required_ms:
                    bound_metrics = compute_release_bound_metrics(
                        release,
                        m=int(m),
                        feature_dim=int(data.feature_dim),
                    )
                    release_radius_value = (
                        float(local_radius_value)
                        if mechanism == "ball"
                        else float(2.0 * float(args.embedding_bound))
                    )
                    release_key_payload = {
                        "dataset": spec.tag,
                        "model": model_family,
                        "mechanism": mechanism,
                        "epsilon": float(epsilon),
                        "m": int(m),
                        "release_seed": int(release_seed),
                        "candidate_radius_tag": args.radius,
                        "candidate_radius_value": float(local_radius_value),
                        "release_radius_value": float(release_radius_value),
                        "lam": float(args.lam),
                        "delta": float(args.delta),
                        "solver": str(args.solver),
                        "max_iter": int(args.max_iter),
                    }
                    release_key = hashlib.sha1(
                        json.dumps(release_key_payload, sort_keys=True).encode("utf-8")
                    ).hexdigest()[:20]
                    release_rows.append(
                        {
                            **release_key_payload,
                            "run_id": run_id,
                            "dataset_tag": spec.tag,
                            "dataset_name": spec.display_name,
                            "model_family": model_family,
                            "embedding_bound": float(args.embedding_bound),
                            "release_sigma": maybe_float(release.privacy.ball.sigma),
                            "release_accuracy": maybe_float(
                                release.utility_metrics.get("accuracy")
                            ),
                            "actual_ball_epsilon": actual_ball_epsilon(release),
                            "same_noise_standard_epsilon": actual_standard_epsilon_same_noise(
                                release
                            ),
                            **bound_metrics,
                            "release_key": release_key,
                        }
                    )

    release_df = pd.DataFrame(release_rows)
    save_dataframe(
        release_df, run_dir / "release_rows.csv", save_parquet_if_possible=False
    )

    trial_rows: list[dict[str, Any]] = []
    config_pairs: list[tuple[float, int]] = []
    if args.sweep in {"epsilon", "both"}:
        config_pairs.extend((float(eps), int(args.fixed_m)) for eps in eps_grid)
    if args.sweep in {"m", "both"}:
        config_pairs.extend((float(args.fixed_epsilon), int(m)) for m in m_grid)
    config_pairs = sorted(set(config_pairs))

    for epsilon, m in config_pairs:
        for mechanism in mechanisms:
            release_radius_value = (
                float(local_radius_value)
                if mechanism == "ball"
                else float(2.0 * float(args.embedding_bound))
            )
            for pool in feasible_pools:
                for draw_index in range(int(args.candidate_draws)):
                    X_candidates, y_candidates, source_ids, candidate_dists = (
                        candidate_sets[(pool.target_index, int(draw_index), int(m))]
                    )
                    candidate_hash = finite_prior_hash(X_candidates, y_candidates)
                    for release_seed in as_int_list(args.release_seeds):
                        release = releases[
                            (mechanism, float(epsilon), int(release_seed))
                        ]
                        attack, _, _ = attack_convex_ball_output_finite_prior(
                            release,
                            data.X_train,
                            data.y_train,
                            target_index=int(pool.target_index),
                            X_candidates=X_candidates,
                            y_candidates=y_candidates,
                            prior_weights=None,
                            known_label=int(pool.target_label),
                            eta_grid=(0.5,),
                        )
                        exact_success = float(
                            attack.metrics.get(
                                "exact_identification_success", float("nan")
                            )
                        )
                        release_key_payload = {
                            "dataset": spec.tag,
                            "model": model_family,
                            "mechanism": mechanism,
                            "epsilon": float(epsilon),
                            "m": int(m),
                            "release_seed": int(release_seed),
                            "candidate_radius_tag": args.radius,
                            "candidate_radius_value": float(local_radius_value),
                            "release_radius_value": float(release_radius_value),
                            "lam": float(args.lam),
                            "delta": float(args.delta),
                            "solver": str(args.solver),
                            "max_iter": int(args.max_iter),
                        }
                        release_key = hashlib.sha1(
                            json.dumps(release_key_payload, sort_keys=True).encode(
                                "utf-8"
                            )
                        ).hexdigest()[:20]
                        trial_key_payload = {
                            **release_key_payload,
                            "target_index": int(pool.target_index),
                            "candidate_draw_index": int(draw_index),
                            "candidate_set_hash": candidate_hash,
                        }
                        trial_key = hashlib.sha1(
                            json.dumps(trial_key_payload, sort_keys=True).encode(
                                "utf-8"
                            )
                        ).hexdigest()[:20]
                        trial_rows.append(
                            {
                                "trial_key": trial_key,
                                "release_key": release_key,
                                "run_id": run_id,
                                "dataset_tag": spec.tag,
                                "dataset_name": spec.display_name,
                                "model_family": model_family,
                                "mechanism": mechanism,
                                "candidate_radius_tag": args.radius,
                                "candidate_radius_value": float(local_radius_value),
                                "release_radius_value": float(release_radius_value),
                                "epsilon": float(epsilon),
                                "m": int(m),
                                "delta": float(args.delta),
                                "lam": float(args.lam),
                                "embedding_bound": float(args.embedding_bound),
                                "solver": str(args.solver),
                                "max_iter": int(args.max_iter),
                                "target_index": int(pool.target_index),
                                "target_label": int(pool.target_label),
                                "candidate_draw_index": int(draw_index),
                                "release_seed": int(release_seed),
                                "candidate_pool_size": int(
                                    pool.distractor_vectors.shape[0]
                                ),
                                "candidate_set_hash": candidate_hash,
                                "candidate_source_ids": json.dumps(source_ids),
                                "candidate_max_distance": float(
                                    np.max(candidate_dists)
                                ),
                                "candidate_mean_distance": float(
                                    np.mean(candidate_dists)
                                ),
                                "candidate_min_distance": float(
                                    np.min(candidate_dists)
                                ),
                                "exact_identification_success": exact_success,
                                "prior_rank": maybe_float(
                                    attack.metrics.get("prior_rank")
                                ),
                                "prior_hit_at_1": maybe_float(
                                    attack.metrics.get("prior_hit@1")
                                ),
                                "prior_hit_at_5": maybe_float(
                                    attack.metrics.get("prior_hit@5")
                                ),
                                "oblivious_kappa": 1.0 / float(m),
                                "release_sigma": maybe_float(
                                    release.privacy.ball.sigma
                                ),
                                "release_accuracy": maybe_float(
                                    release.utility_metrics.get("accuracy")
                                ),
                                "actual_ball_epsilon": actual_ball_epsilon(release),
                                "same_noise_standard_epsilon": actual_standard_epsilon_same_noise(
                                    release
                                ),
                            }
                        )

    trial_df = pd.DataFrame(trial_rows)
    save_dataframe(
        trial_df, run_dir / "attack_trial_rows.csv", save_parquet_if_possible=False
    )

    rebuild_attack_dataset_outputs(
        dataset_dir,
        spec=spec,
        model_family=model_family,
        plot_radius_tag=str(args.radius),
        fixed_epsilon=float(args.fixed_epsilon),
        fixed_m=int(args.fixed_m),
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "dataset": spec.display_name,
                "model_family": model_family,
                "run_id": run_id,
                "results_dir": str(dataset_dir),
                "run_dir": str(run_dir),
                "candidate_radius_tag": args.radius,
                "candidate_radius_value": float(local_radius_value),
                "num_feasible_targets": int(len(feasible_pools)),
                "num_trial_rows": int(len(trial_df)),
                "num_release_rows": int(len(release_df)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
