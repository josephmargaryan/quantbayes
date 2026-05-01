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
from typing import Any, Optional, Sequence

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
    diagnose_convex_ball_output_finite_prior,
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
DEFAULT_FIXED_EPSILON = 4.0
DEFAULT_FIXED_M = 8
DEFAULT_DELTA = 1e-6
DEFAULT_EMBEDDING_BOUND = 1.0
DEFAULT_LAM = 1e-2
DEFAULT_MAX_ITER = 100
DEFAULT_RIDGE_SENSITIVITY_MODE = "global"
DEFAULT_SUPPORT_SELECTION = "random"
DEFAULT_ANCHOR_SELECTION = "random"
DEFAULT_STANDARD_RADIUS_SOURCE = "empirical_diameter"
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


def support_source_hash(source_ids: Sequence[str]) -> str:
    h = hashlib.sha1()
    for sid in source_ids:
        h.update(str(sid).encode("utf-8"))
        h.update(b"\n")
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
class SupportBank:
    anchor_index: int
    anchor_label: int
    anchor_vector: np.ndarray
    bank_vectors: np.ndarray
    bank_source_ids: list[str]
    bank_distances: np.ndarray


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


def standard_radius_value_from_report(
    report: dict[str, Any],
    *,
    source: str,
    embedding_bound: float,
) -> float:
    key = str(source).strip().lower()
    if key == "embedding_bound":
        return float(2.0 * float(embedding_bound))
    if key == "empirical_diameter":
        return float(
            select_ball_radius(
                report,
                strategy="global_max",
                quantile=1.0,
                allow_observed_max=True,
            )
        )
    raise ValueError(
        "standard_radius_source must be one of {'empirical_diameter', 'embedding_bound'}."
    )


def make_run_id(args: argparse.Namespace, spec: DatasetSpec, model_family: str) -> str:
    payload = {
        "dataset": spec.tag,
        "dataset_spec": asdict(spec),
        "model": model_family,
        "args": vars(args),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return digest


def seed_seq(*parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(p) for p in parts]))


def dedup_vectors_and_ids(
    vectors: np.ndarray,
    source_ids: Sequence[str],
    distances: np.ndarray,
    *,
    rounding_decimals: int = 7,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    seen: set[bytes] = set()
    kept_vectors: list[np.ndarray] = []
    kept_ids: list[str] = []
    kept_dists: list[float] = []
    for vec, src, dist in zip(vectors, source_ids, distances, strict=True):
        key = np.round(np.asarray(vec, dtype=np.float32), rounding_decimals).tobytes()
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


def remove_train_index(
    X: np.ndarray, y: np.ndarray, index: int
) -> tuple[np.ndarray, np.ndarray]:
    idx = int(index)
    return (
        np.concatenate([X[:idx], X[idx + 1 :]], axis=0),
        np.concatenate([y[:idx], y[idx + 1 :]], axis=0),
    )


def build_support_bank(
    data: LoadedDataset,
    *,
    anchor_index: int,
    radius_value: float,
    source_mode: str,
    tol: float = 1e-7,
) -> SupportBank:
    anchor_index = int(anchor_index)
    anchor_vec = np.asarray(data.X_train[anchor_index], dtype=np.float32)
    anchor_label = int(data.y_train[anchor_index])

    # The anchor defines the center u and the reduced dataset D^- = D \ {u},
    # but it is not forced into the finite support S. This keeps the empirical
    # protocol aligned with the fixed-support uniform-prior model used in the theory.
    vectors: list[np.ndarray] = []
    source_ids: list[str] = []
    distances: list[float] = []

    test_same = np.where(data.y_test == anchor_label)[0]
    if test_same.size:
        test_vectors = np.asarray(data.X_test[test_same], dtype=np.float32)
        test_dists = np.linalg.norm(test_vectors - anchor_vec[None, :], axis=1)
        keep_test = test_dists <= float(radius_value) + float(tol)
        for idx, vec, dist in zip(
            test_same[keep_test].tolist(),
            test_vectors[keep_test],
            test_dists[keep_test],
            strict=True,
        ):
            vectors.append(np.asarray(vec, dtype=np.float32))
            source_ids.append(f"test:{int(idx)}")
            distances.append(float(dist))

    if source_mode == "train_and_public":
        train_same = np.where(data.y_train == anchor_label)[0]
        train_same = train_same[train_same != anchor_index]
        if train_same.size:
            train_vectors = np.asarray(data.X_train[train_same], dtype=np.float32)
            train_dists = np.linalg.norm(train_vectors - anchor_vec[None, :], axis=1)
            keep_train = train_dists <= float(radius_value) + float(tol)
            for idx, vec, dist in zip(
                train_same[keep_train].tolist(),
                train_vectors[keep_train],
                train_dists[keep_train],
                strict=True,
            ):
                vectors.append(np.asarray(vec, dtype=np.float32))
                source_ids.append(f"train:{int(idx)}")
                distances.append(float(dist))

    if vectors:
        vec_arr = np.stack(vectors, axis=0).astype(np.float32, copy=False)
        dist_arr = np.asarray(distances, dtype=np.float32)
        vec_arr, source_ids, dist_arr = dedup_vectors_and_ids(
            vec_arr,
            source_ids,
            dist_arr,
        )
    else:
        vec_arr = np.zeros((0, int(data.feature_dim)), dtype=np.float32)
        source_ids = []
        dist_arr = np.zeros((0,), dtype=np.float32)

    return SupportBank(
        anchor_index=anchor_index,
        anchor_label=anchor_label,
        anchor_vector=anchor_vec,
        bank_vectors=vec_arr,
        bank_source_ids=source_ids,
        bank_distances=dist_arr,
    )


def find_feasible_support_banks(
    data: LoadedDataset,
    *,
    radius_value: float,
    max_required_m: int,
    num_supports: int,
    anchor_seed: int,
    source_mode: str,
    max_search: Optional[int],
    explicit_anchor_indices: Optional[Sequence[int]] = None,
    strict: bool = False,
    anchor_selection: str = "random",
) -> list[SupportBank]:
    strategy = str(anchor_selection).strip().lower()
    if strategy not in {"random", "rare_class", "large_bank"}:
        raise ValueError(
            "anchor_selection must be one of {'random', 'rare_class', 'large_bank'}."
        )

    rng = np.random.default_rng(int(anchor_seed))
    if explicit_anchor_indices:
        candidate_indices = [int(i) for i in explicit_anchor_indices]
    else:
        if strategy == "rare_class":
            counts = np.bincount(
                np.asarray(data.y_train, dtype=np.int64),
                minlength=int(data.num_classes),
            )
            jitter = rng.random(len(data.X_train))
            candidate_indices = sorted(
                range(len(data.X_train)),
                key=lambda i: (int(counts[int(data.y_train[i])]), float(jitter[i])),
            )
        else:
            candidate_indices = rng.permutation(len(data.X_train)).tolist()
        if max_search is not None and strategy != "large_bank":
            candidate_indices = candidate_indices[: int(max_search)]

    banks: list[SupportBank] = []
    if strategy == "large_bank" and not explicit_anchor_indices:
        scored: list[tuple[int, SupportBank]] = []
        search_indices = candidate_indices
        if max_search is not None:
            search_indices = search_indices[: int(max_search)]
        for idx in search_indices:
            bank = build_support_bank(
                data,
                anchor_index=int(idx),
                radius_value=float(radius_value),
                source_mode=str(source_mode),
            )
            if int(bank.bank_vectors.shape[0]) >= int(max_required_m):
                scored.append((int(bank.bank_vectors.shape[0]), bank))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        banks = [bank for _, bank in scored[: int(num_supports)]]
    else:
        for idx in candidate_indices:
            bank = build_support_bank(
                data,
                anchor_index=int(idx),
                radius_value=float(radius_value),
                source_mode=str(source_mode),
            )
            if int(bank.bank_vectors.shape[0]) >= int(max_required_m):
                banks.append(bank)
            if len(banks) >= int(num_supports):
                break

    if len(banks) < int(num_supports) and strict:
        raise RuntimeError(
            f"Requested {num_supports} feasible support anchors, but found only {len(banks)} "
            f"for radius {radius_value:.6f}, m_max={max_required_m}, source_mode={source_mode!r}, "
            f"and anchor_selection={strategy!r}."
        )
    if not banks:
        raise RuntimeError(
            f"No feasible support anchors found for radius {radius_value:.6f}, "
            f"m_max={max_required_m}, source_mode={source_mode!r}, and anchor_selection={strategy!r}."
        )
    return banks


def _greedy_farthest_positions(
    vectors: np.ndarray,
    *,
    m: int,
    rng: np.random.Generator,
    anchor_distances: Optional[np.ndarray] = None,
) -> np.ndarray:
    X = np.asarray(vectors, dtype=np.float32)
    n = int(X.shape[0])
    if n < int(m):
        raise ValueError("not enough vectors for farthest-point support selection")

    if anchor_distances is None:
        first = int(rng.integers(0, n))
    else:
        # Stress choice with replicate diversity: start from a random member of
        # the farthest tail from the Ball center, then run max-min packing.
        d_anchor = np.asarray(anchor_distances, dtype=np.float64)
        top_k = min(n, max(int(m), 4 * int(m)))
        farthest_tail = np.argsort(-d_anchor)[:top_k]
        first = int(rng.choice(farthest_tail))

    selected = [first]
    min_d = np.linalg.norm(X - X[first][None, :], axis=1)
    min_d[first] = -np.inf
    while len(selected) < int(m):
        # Random tie-breaking avoids systematic index artifacts while preserving
        # the max-min packing objective.
        best_val = float(np.max(min_d))
        candidates = np.flatnonzero(np.isclose(min_d, best_val, rtol=1e-7, atol=1e-7))
        nxt = int(rng.choice(candidates)) if candidates.size else int(np.argmax(min_d))
        selected.append(nxt)
        d_new = np.linalg.norm(X - X[nxt][None, :], axis=1)
        min_d = np.minimum(min_d, d_new)
        min_d[selected] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def make_support_set(
    bank: SupportBank,
    *,
    m: int,
    draw_index: int,
    base_seed: int,
    selection: str = "random",
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if int(m) < 2:
        raise ValueError("m must be at least 2.")
    available = int(bank.bank_vectors.shape[0])
    if available < int(m):
        raise ValueError(
            f"Anchor {bank.anchor_index} only has bank size {available}, but m={m}."
        )

    rng = seed_seq(base_seed, bank.anchor_index, draw_index, int(m), 991)
    mode = str(selection).strip().lower()
    if mode == "random":
        positions = rng.permutation(available)[: int(m)]
    elif mode == "farthest":
        positions = _greedy_farthest_positions(
            bank.bank_vectors,
            m=int(m),
            rng=rng,
            anchor_distances=bank.bank_distances,
        )
    elif mode == "nearest":
        positions = np.argsort(np.asarray(bank.bank_distances, dtype=np.float64))[
            : int(m)
        ]
    else:
        raise ValueError(
            "support selection must be one of {'random', 'farthest', 'nearest'}."
        )

    X_support = np.asarray(bank.bank_vectors[positions], dtype=np.float32)
    y_support = np.full((int(m),), int(bank.anchor_label), dtype=np.int32)
    source_ids = [bank.bank_source_ids[int(i)] for i in positions.tolist()]
    dists = np.asarray(bank.bank_distances[positions], dtype=np.float32)
    return X_support, y_support, source_ids, dists


def fit_release_for_mechanism(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    model_family: str,
    mechanism: str,
    epsilon: float,
    local_radius_value: float,
    standard_radius_value: float,
    embedding_bound: float,
    lam: float,
    delta: float,
    max_iter: int,
    seed: int,
    orders: Sequence[float],
    solver: str,
    ridge_sensitivity_mode: str,
) -> Any:
    if mechanism == "ball":
        release_radius = float(local_radius_value)
    elif mechanism == "standard":
        release_radius = float(standard_radius_value)
    else:
        raise ValueError(f"Unknown mechanism {mechanism!r}")

    return fit_convex(
        X_train,
        y_train,
        model_family=model_family,
        privacy="ball_dp",
        radius=float(release_radius),
        lam=float(lam),
        epsilon=float(epsilon),
        delta=float(delta),
        embedding_bound=float(embedding_bound),
        standard_radius=float(standard_radius_value),
        ridge_sensitivity_mode=str(ridge_sensitivity_mode),
        num_classes=int(num_classes),
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
        "standard_radius_value",
        "support_source_mode",
        "support_selection",
        "ridge_sensitivity_mode",
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
        num_support_anchors=("anchor_index", "nunique"),
        support_draws=("support_draw_index", "nunique"),
        release_seeds=("release_seed", "nunique"),
        exact_id_mean=("exact_identification_success", "mean"),
        oblivious_kappa=("oblivious_kappa", "mean"),
        support_bank_size_mean=("support_bank_size", "mean"),
        support_max_distance_mean=("support_max_distance", "mean"),
    ).reset_index()

    ci_bounds = [
        summarize_bernoulli(p, n)
        for p, n in zip(summary["exact_id_mean"], summary["n_trials"], strict=True)
    ]
    summary["exact_id_ci_low"] = [lo for lo, _ in ci_bounds]
    summary["exact_id_ci_high"] = [hi for _, hi in ci_bounds]
    summary["attack_advantage"] = summary["exact_id_mean"] - summary["oblivious_kappa"]

    diag_rows: list[dict[str, Any]] = []
    for key, frame in trial_group:
        diag = {col: val for col, val in zip(config_cols, key, strict=True)}
        if "predicted_source_id" in frame.columns:
            pred_ids = frame["predicted_source_id"].fillna("<missing>").astype(str)
            counts = pred_ids.value_counts(dropna=False)
            mode_count = int(counts.iloc[0]) if not counts.empty else 0
            diag["predicted_source_unique_count"] = int(len(counts))
            diag["predicted_source_mode_share"] = (
                float(mode_count / float(len(frame))) if len(frame) else float("nan")
            )
            diag["predicted_source_mode_id"] = (
                str(counts.index[0]) if not counts.empty else None
            )
        else:
            diag["predicted_source_unique_count"] = float("nan")
            diag["predicted_source_mode_share"] = float("nan")
            diag["predicted_source_mode_id"] = None
        if {"predicted_source_id", "target_source_id"}.issubset(frame.columns):
            diag["predicted_matches_target_rate"] = float(
                np.mean(
                    frame["predicted_source_id"]
                    .fillna("<missing>")
                    .astype(str)
                    .to_numpy()
                    == frame["target_source_id"]
                    .fillna("<missing>")
                    .astype(str)
                    .to_numpy()
                )
            )
        else:
            diag["predicted_matches_target_rate"] = float("nan")
        if "predicted_source_is_anchor" in frame.columns:
            diag["predicted_anchor_rate"] = float(
                np.mean(frame["predicted_source_is_anchor"].astype(float))
            )
        else:
            diag["predicted_anchor_rate"] = float("nan")
        if "support_contains_anchor" in frame.columns:
            diag["support_contains_anchor_rate"] = float(
                np.mean(frame["support_contains_anchor"].astype(float))
            )
        else:
            diag["support_contains_anchor_rate"] = float("nan")

        mean_metric_cols = [
            "posterior_top1_probability",
            "posterior_true_probability",
            "posterior_entropy",
            "posterior_effective_candidates",
            "log_score_gap_top2",
            "log_score_gap_truth_to_top",
            "bound_direct_instance_center_max",
            "bound_direct_instance_finite_opt",
            "model_center_radius_over_sigma_max",
            "model_center_radius_over_sigma_mean",
            "model_pairwise_snr_median",
            "model_pairwise_snr_mean",
            "model_nn_snr_median",
            "model_nn_snr_mean",
            "feature_pairwise_distance_median",
            "feature_nn_distance_median",
            "ridge_inverse_noise_tau",
            "ridge_inverse_noise_tau_over_radius",
            "ridge_count_dilution",
            "ridge_feature_pairwise_snr_median",
            "ridge_feature_nn_snr_median",
        ]
        for col in mean_metric_cols:
            if col in frame.columns:
                diag[f"{col}_mean"] = float(np.nanmean(frame[col].astype(float)))
            else:
                diag[f"{col}_mean"] = float("nan")
        diag_rows.append(diag)

    diag_df = pd.DataFrame(diag_rows)
    if not diag_df.empty:
        summary = summary.merge(diag_df, on=config_cols, how="left")

    if not release_df.empty:
        release_metric_cols = [
            "dataset_tag",
            "dataset_name",
            "model_family",
            "mechanism",
            "candidate_radius_tag",
            "candidate_radius_value",
            "release_radius_value",
            "standard_radius_value",
            "support_source_mode",
            "support_selection",
            "ridge_sensitivity_mode",
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
            label=f"Uniform-prior baseline = 1/{fixed_m}",
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
                label="Uniform-prior baseline = 1/m",
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
            "Run theorem-aligned convex finite-prior exact-identification attack experiments. "
            "For each anchor record u, the script fixes a support S inside the local Ball around u, "
            "then evaluates every target choice in S under a uniform prior while holding D^- fixed."
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
        "--support-source",
        choices=("public_only", "train_and_public"),
        default="public_only",
        help=(
            "Candidate bank used to form fixed supports. 'public_only' uses same-label "
            "test/public embeddings inside the selected Ball. 'train_and_public' additionally allows same-label "
            "training embeddings inside the Ball."
        ),
    )
    parser.add_argument(
        "--support-selection",
        choices=("random", "farthest", "nearest"),
        default=DEFAULT_SUPPORT_SELECTION,
        help=(
            "How to choose the finite support from the feasible candidate bank. "
            "'farthest' is the strongest simple stress test: a greedy max-min packing inside the Ball."
        ),
    )
    parser.add_argument(
        "--anchor-selection",
        choices=("random", "rare_class", "large_bank"),
        default=DEFAULT_ANCHOR_SELECTION,
        help=(
            "How to choose Ball centers/anchors. 'rare_class' stresses the ridge prototype by preferring "
            "classes with smaller training counts; 'large_bank' maximizes candidate availability."
        ),
    )
    parser.add_argument(
        "--standard-radius-source",
        choices=("empirical_diameter", "embedding_bound"),
        default=DEFAULT_STANDARD_RADIUS_SOURCE,
        help=(
            "Radius used for the standard global comparator. 'empirical_diameter' matches the paper's "
            "same-label observed diameter; 'embedding_bound' uses the conservative 2B comparator."
        ),
    )
    parser.add_argument(
        "--ridge-sensitivity-mode",
        choices=("global", "count_aware"),
        default=DEFAULT_RIDGE_SENSITIVITY_MODE,
        help=(
            "Ridge-prototype sensitivity calibration. 'global' uses 2r/(2+lambda n); "
            "'count_aware' uses the exact public-count sensitivity max_c 2r/(2n_c+lambda n)."
        ),
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
    parser.add_argument(
        "--num-supports", "--num-targets", dest="num_supports", type=int, default=8
    )
    parser.add_argument(
        "--support-draws",
        "--candidate-draws",
        dest="support_draws",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--anchor-seed", "--target-seed", dest="anchor_seed", type=int, default=0
    )
    parser.add_argument("--max-feasible-search", type=int, default=None)
    parser.add_argument(
        "--anchor-indices",
        "--target-indices",
        dest="anchor_indices",
        nargs="*",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--strict-feasible-supports",
        "--strict-feasible-targets",
        dest="strict_feasible_supports",
        action="store_true",
    )

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
        seed=int(args.anchor_seed),
    )
    local_radius_value = radius_value_from_report(radius_report, args.radius)
    standard_radius_value = standard_radius_value_from_report(
        radius_report,
        source=str(args.standard_radius_source),
        embedding_bound=float(args.embedding_bound),
    )

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

    feasible_banks = find_feasible_support_banks(
        data,
        radius_value=float(local_radius_value),
        max_required_m=int(max_required_m),
        num_supports=int(args.num_supports),
        anchor_seed=int(args.anchor_seed),
        source_mode=str(args.support_source),
        max_search=args.max_feasible_search,
        explicit_anchor_indices=args.anchor_indices,
        strict=bool(args.strict_feasible_supports),
        anchor_selection=str(args.anchor_selection),
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
            "standard_radius_source": str(args.standard_radius_source),
            "standard_radius_value": float(standard_radius_value),
            "support_source_mode": str(args.support_source),
            "support_selection": str(args.support_selection),
            "anchor_selection": str(args.anchor_selection),
            "ridge_sensitivity_mode": str(args.ridge_sensitivity_mode),
            "backend": data.backend,
            "release_seeds": as_int_list(args.release_seeds),
            "num_feasible_support_anchors": int(len(feasible_banks)),
            "support_excludes_anchor_from_candidates": True,
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
                        "support_contains_anchor": any(
                            str(s).startswith("anchor:") for s in source_ids
                        ),
                    }
                )
    write_json(run_dir / "support_sets.json", support_sets_meta)

    config_pairs: list[tuple[float, int]] = []
    if args.sweep in {"epsilon", "both"}:
        config_pairs.extend((float(eps), int(args.fixed_m)) for eps in eps_grid)
    if args.sweep in {"m", "both"}:
        config_pairs.extend((float(args.fixed_epsilon), int(m)) for m in m_grid)
    config_pairs = sorted(set(config_pairs))

    release_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    release_cache: dict[tuple[int, str, float, str, int], Any] = {}
    release_key_cache: dict[tuple[int, str, float, str, int, int], str] = {}
    bound_cache: dict[tuple[int, str, float, str, int, int], dict[str, float]] = {}
    diagnostics_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    for epsilon, m in config_pairs:
        for bank in feasible_banks:
            X_minus, y_minus = remove_train_index(
                data.X_train, data.y_train, bank.anchor_index
            )
            for draw_index in range(int(args.support_draws)):
                X_support, y_support, source_ids, support_dists, support_hash = (
                    support_sets[(bank.anchor_index, int(draw_index), int(m))]
                )
                for target_pos, target_sid in enumerate(source_ids):
                    x_target = np.asarray(X_support[target_pos], dtype=np.float32)
                    y_target = int(y_support[target_pos])
                    X_full = np.concatenate(
                        [X_minus, x_target[None, :]], axis=0
                    ).astype(np.float32, copy=False)
                    y_full = np.concatenate(
                        [y_minus, np.asarray([y_target], dtype=np.int32)], axis=0
                    )
                    target_index = int(len(X_full) - 1)

                    for mechanism in mechanisms:
                        release_radius_value = (
                            float(local_radius_value)
                            if mechanism == "ball"
                            else float(standard_radius_value)
                        )
                        for release_seed in as_int_list(args.release_seeds):
                            cache_key = (
                                int(bank.anchor_index),
                                str(target_sid),
                                float(epsilon),
                                str(mechanism),
                                int(release_seed),
                            )
                            release = release_cache.get(cache_key)
                            if release is None:
                                release = fit_release_for_mechanism(
                                    X_train=X_full,
                                    y_train=y_full,
                                    num_classes=int(data.num_classes),
                                    model_family=model_family,
                                    mechanism=mechanism,
                                    epsilon=float(epsilon),
                                    local_radius_value=float(local_radius_value),
                                    standard_radius_value=float(standard_radius_value),
                                    embedding_bound=float(args.embedding_bound),
                                    lam=float(args.lam),
                                    delta=float(args.delta),
                                    max_iter=int(args.max_iter),
                                    seed=int(release_seed),
                                    orders=DEFAULT_ORDERS,
                                    solver=str(args.solver),
                                    ridge_sensitivity_mode=str(
                                        args.ridge_sensitivity_mode
                                    ),
                                )
                                release_cache[cache_key] = release

                            release_key_lookup = (
                                int(bank.anchor_index),
                                str(target_sid),
                                float(epsilon),
                                str(mechanism),
                                int(release_seed),
                                int(m),
                            )
                            release_key = release_key_cache.get(release_key_lookup)
                            if release_key is None:
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
                                    "standard_radius_value": float(
                                        standard_radius_value
                                    ),
                                    "support_source_mode": str(args.support_source),
                                    "support_selection": str(args.support_selection),
                                    "ridge_sensitivity_mode": str(
                                        args.ridge_sensitivity_mode
                                    ),
                                    "anchor_index": int(bank.anchor_index),
                                    "target_source_id": str(target_sid),
                                    "lam": float(args.lam),
                                    "delta": float(args.delta),
                                    "solver": str(args.solver),
                                    "max_iter": int(args.max_iter),
                                }
                                release_key = hashlib.sha1(
                                    json.dumps(
                                        release_key_payload, sort_keys=True
                                    ).encode("utf-8")
                                ).hexdigest()[:20]
                                release_key_cache[release_key_lookup] = release_key

                                bound_key = release_key_lookup
                                bound_metrics = bound_cache.get(bound_key)
                                if bound_metrics is None:
                                    bound_metrics = compute_release_bound_metrics(
                                        release,
                                        m=int(m),
                                        feature_dim=int(data.feature_dim),
                                    )
                                    bound_cache[bound_key] = bound_metrics
                                release_rows.append(
                                    {
                                        **release_key_payload,
                                        "run_id": run_id,
                                        "dataset_tag": spec.tag,
                                        "dataset_name": spec.display_name,
                                        "model_family": model_family,
                                        "embedding_bound": float(args.embedding_bound),
                                        "standard_radius_value": float(
                                            standard_radius_value
                                        ),
                                        "support_selection": str(
                                            args.support_selection
                                        ),
                                        "ridge_sensitivity_mode": str(
                                            args.ridge_sensitivity_mode
                                        ),
                                        "release_sigma": maybe_float(
                                            release.privacy.ball.sigma
                                        ),
                                        "release_accuracy": maybe_float(
                                            release.utility_metrics.get("accuracy")
                                        ),
                                        "actual_ball_epsilon": actual_ball_epsilon(
                                            release
                                        ),
                                        "same_noise_standard_epsilon": actual_standard_epsilon_same_noise(
                                            release
                                        ),
                                        **bound_metrics,
                                        "release_key": release_key,
                                    }
                                )

                            attack, _, _ = attack_convex_ball_output_finite_prior(
                                release,
                                X_full,
                                y_full,
                                target_index=target_index,
                                X_candidates=X_support,
                                y_candidates=y_support,
                                prior_weights=None,
                                known_label=int(bank.anchor_label),
                                eta_grid=(0.5,),
                            )
                            exact_success = float(
                                attack.metrics.get(
                                    "exact_identification_success", float("nan")
                                )
                            )
                            diagnostics_key = (
                                int(bank.anchor_index),
                                str(support_hash),
                                str(target_sid),
                                float(epsilon),
                                str(mechanism),
                                int(release_seed),
                                int(m),
                            )
                            support_diagnostics = diagnostics_cache.get(diagnostics_key)
                            if support_diagnostics is None:
                                support_diagnostics = (
                                    diagnose_convex_ball_output_finite_prior(
                                        release,
                                        X_full,
                                        y_full,
                                        target_index=target_index,
                                        X_candidates=X_support,
                                        y_candidates=y_support,
                                        prior_weights=None,
                                        known_label=int(bank.anchor_label),
                                        center_features=np.asarray(
                                            bank.anchor_vector, dtype=np.float32
                                        ),
                                        center_label=int(bank.anchor_label),
                                    )
                                )
                                diagnostics_cache[diagnostics_key] = dict(
                                    support_diagnostics
                                )
                            pred_idx_raw = attack.diagnostics.get(
                                "predicted_prior_index"
                            )
                            pred_idx = (
                                int(pred_idx_raw) if pred_idx_raw is not None else None
                            )
                            true_idx_raw = attack.diagnostics.get("true_prior_index")
                            true_idx = (
                                int(true_idx_raw) if true_idx_raw is not None else None
                            )
                            if pred_idx is not None and 0 <= pred_idx < len(source_ids):
                                predicted_source_id = str(source_ids[pred_idx])
                            else:
                                predicted_source_id = None
                            support_contains_anchor = any(
                                str(s).startswith("anchor:") for s in source_ids
                            )

                            trial_key_payload = {
                                "release_key": release_key,
                                "support_set_hash": support_hash,
                                "target_source_id": str(target_sid),
                                "target_position": int(target_pos),
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
                                    "standard_radius_value": float(
                                        standard_radius_value
                                    ),
                                    "support_source_mode": str(args.support_source),
                                    "support_selection": str(args.support_selection),
                                    "ridge_sensitivity_mode": str(
                                        args.ridge_sensitivity_mode
                                    ),
                                    "epsilon": float(epsilon),
                                    "m": int(m),
                                    "delta": float(args.delta),
                                    "lam": float(args.lam),
                                    "embedding_bound": float(args.embedding_bound),
                                    "solver": str(args.solver),
                                    "max_iter": int(args.max_iter),
                                    "anchor_index": int(bank.anchor_index),
                                    "anchor_label": int(bank.anchor_label),
                                    "support_draw_index": int(draw_index),
                                    "release_seed": int(release_seed),
                                    "target_position": int(target_pos),
                                    "target_source_id": str(target_sid),
                                    "target_source_is_anchor": float(
                                        str(target_sid).startswith("anchor:")
                                    ),
                                    "predicted_prior_index": (
                                        pred_idx if pred_idx is not None else np.nan
                                    ),
                                    "true_prior_index": (
                                        true_idx if true_idx is not None else np.nan
                                    ),
                                    "predicted_source_id": predicted_source_id,
                                    "predicted_source_is_anchor": (
                                        float(
                                            str(predicted_source_id).startswith(
                                                "anchor:"
                                            )
                                        )
                                        if predicted_source_id is not None
                                        else np.nan
                                    ),
                                    "support_contains_anchor": float(
                                        support_contains_anchor
                                    ),
                                    "support_bank_size": int(
                                        bank.bank_vectors.shape[0]
                                    ),
                                    "support_set_hash": support_hash,
                                    "support_source_ids": json.dumps(source_ids),
                                    "support_max_distance": float(
                                        np.max(support_dists)
                                    ),
                                    "support_mean_distance": float(
                                        np.mean(support_dists)
                                    ),
                                    "support_min_distance": float(
                                        np.min(support_dists)
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
                                    "posterior_top1_probability": maybe_float(
                                        attack.metrics.get("posterior_top1_probability")
                                    ),
                                    "posterior_true_probability": maybe_float(
                                        attack.metrics.get("posterior_true_probability")
                                    ),
                                    "posterior_entropy": maybe_float(
                                        attack.metrics.get("posterior_entropy")
                                    ),
                                    "posterior_effective_candidates": maybe_float(
                                        attack.metrics.get(
                                            "posterior_effective_candidates"
                                        )
                                    ),
                                    "log_score_gap_top2": maybe_float(
                                        attack.metrics.get("log_score_gap_top2")
                                    ),
                                    "log_score_gap_truth_to_top": maybe_float(
                                        attack.metrics.get("log_score_gap_truth_to_top")
                                    ),
                                    "oblivious_kappa": 1.0 / float(m),
                                    "bound_direct_instance_center_max": maybe_float(
                                        support_diagnostics.get(
                                            "bound_direct_instance_center_max"
                                        )
                                    ),
                                    "bound_direct_instance_finite_opt": maybe_float(
                                        support_diagnostics.get(
                                            "bound_direct_instance_finite_opt"
                                        )
                                    ),
                                    "model_center_radius_over_sigma_max": maybe_float(
                                        support_diagnostics.get(
                                            "model_center_radius_over_sigma_max"
                                        )
                                    ),
                                    "model_center_radius_over_sigma_mean": maybe_float(
                                        support_diagnostics.get(
                                            "model_center_radius_over_sigma_mean"
                                        )
                                    ),
                                    "model_pairwise_snr_median": maybe_float(
                                        support_diagnostics.get(
                                            "model_pairwise_snr_median"
                                        )
                                    ),
                                    "model_pairwise_snr_mean": maybe_float(
                                        support_diagnostics.get(
                                            "model_pairwise_snr_mean"
                                        )
                                    ),
                                    "model_nn_snr_median": maybe_float(
                                        support_diagnostics.get("model_nn_snr_median")
                                    ),
                                    "model_nn_snr_mean": maybe_float(
                                        support_diagnostics.get("model_nn_snr_mean")
                                    ),
                                    "feature_pairwise_distance_median": maybe_float(
                                        support_diagnostics.get(
                                            "feature_pairwise_distance_median"
                                        )
                                    ),
                                    "feature_nn_distance_median": maybe_float(
                                        support_diagnostics.get(
                                            "feature_nn_distance_median"
                                        )
                                    ),
                                    "ridge_inverse_noise_tau": maybe_float(
                                        support_diagnostics.get(
                                            "ridge_inverse_noise_tau"
                                        )
                                    ),
                                    "ridge_inverse_noise_tau_over_radius": maybe_float(
                                        support_diagnostics.get(
                                            "ridge_inverse_noise_tau_over_radius"
                                        )
                                    ),
                                    "ridge_count_dilution": maybe_float(
                                        support_diagnostics.get("ridge_count_dilution")
                                    ),
                                    "ridge_feature_pairwise_snr_median": maybe_float(
                                        support_diagnostics.get(
                                            "ridge_feature_pairwise_snr_median"
                                        )
                                    ),
                                    "ridge_feature_nn_snr_median": maybe_float(
                                        support_diagnostics.get(
                                            "ridge_feature_nn_snr_median"
                                        )
                                    ),
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

    release_df = pd.DataFrame(release_rows)
    save_dataframe(
        release_df, run_dir / "release_rows.csv", save_parquet_if_possible=False
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
                "standard_radius_source": str(args.standard_radius_source),
                "standard_radius_value": float(standard_radius_value),
                "support_source_mode": str(args.support_source),
                "support_selection": str(args.support_selection),
                "anchor_selection": str(args.anchor_selection),
                "ridge_sensitivity_mode": str(args.ridge_sensitivity_mode),
                "num_feasible_support_anchors": int(len(feasible_banks)),
                "num_trial_rows": int(len(trial_df)),
                "num_release_rows": int(len(release_df)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
