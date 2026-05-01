# quantbayes/ball_dp/informed.py

from __future__ import annotations

import dataclasses as dc
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
import optax

from .attacks.model_based import (
    FlatRecordCodec,
    ParametersOnlyFeatureMap,
    build_shadow_corpus,
    train_shadow_reconstructor,
)
from .config import ReconstructorTrainingConfig, ShadowCorpusConfig
from .types import (
    ArrayDataset,
    Record,
    ReconstructorArtifact,
    ReleaseArtifact,
    ShadowCorpus,
)

BuildModelFn = Callable[[int], Any]


@dc.dataclass(frozen=True)
class InformedAttackData:
    """Paper-style split for the informed-adversary model-based attack."""

    d_minus: ArrayDataset
    target: Record
    d_bar: ArrayDataset
    nearest_shadow_target: Optional[Record]
    base_indices: np.ndarray
    shadow_indices: np.ndarray
    target_index: int
    target_source: str


def _append_record(ds: ArrayDataset, record: Record, *, name: str) -> ArrayDataset:
    x = np.concatenate(
        [np.asarray(ds.X), np.asarray(record.features)[None, ...]], axis=0
    )
    y = np.concatenate(
        [
            np.asarray(ds.y),
            np.asarray([int(record.label)], dtype=np.asarray(ds.y).dtype),
        ],
        axis=0,
    )
    return ArrayDataset(x, y, name=name)


def _build_model_and_state(build_model: BuildModelFn, seed: int):
    built = build_model(int(seed))
    if isinstance(built, tuple) and len(built) == 2:
        model, state = built
        return model, state
    return built, None


def _shadow_records(shadow_targets: ArrayDataset | Sequence[Record]) -> list[Record]:
    if isinstance(shadow_targets, ArrayDataset):
        return [shadow_targets.record(i) for i in range(len(shadow_targets))]
    return [
        Record(features=np.asarray(r.features), label=int(r.label))
        for r in shadow_targets
    ]


def _nearest_shadow_target(target: Record, d_bar: ArrayDataset) -> Optional[Record]:
    if len(d_bar) == 0:
        return None
    target_x = np.asarray(target.features, dtype=np.float32).reshape(-1)
    shadow_x = np.asarray(d_bar.X, dtype=np.float32).reshape(len(d_bar), -1)
    idx = int(np.argmin(np.linalg.norm(shadow_x - target_x[None, :], axis=1)))
    return d_bar.record(idx)


def prepare_informed_attack_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_index: int,
    base_train_size: int,
    num_shadow: int,
    X_target: Optional[np.ndarray] = None,
    y_target: Optional[np.ndarray] = None,
    seed: int = 0,
) -> InformedAttackData:
    """Construct D^-, z, and D_bar for the informed-adversary workflow.

    Default behavior uses the training set both for D^- and the target pool, excluding the
    target from D^- when X_target / y_target are omitted.

    For the paper-style split, pass X_target=X_test and y_target=y_test.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if (X_target is None) != (y_target is None):
        raise ValueError("Provide both X_target and y_target, or neither.")

    target_from_train = X_target is None
    if target_from_train:
        X_target = X_train
        y_target = y_train
        target_source = "train"
    else:
        X_target = np.asarray(X_target)
        y_target = np.asarray(y_target)
        target_source = "target_pool"

    target = Record(
        features=np.asarray(X_target[int(target_index)]),
        label=int(np.asarray(y_target)[int(target_index)]),
    )

    rng = np.random.default_rng(int(seed))
    available = np.arange(len(X_train), dtype=np.int64)
    if target_from_train:
        available = available[available != int(target_index)]

    if int(base_train_size) > len(available):
        raise ValueError("base_train_size is larger than the available training pool.")

    base_indices = np.sort(
        rng.choice(available, size=int(base_train_size), replace=False)
    ).astype(np.int64)

    remaining = available[~np.isin(available, base_indices)]
    if int(num_shadow) > len(remaining):
        raise ValueError("num_shadow is larger than the remaining training pool.")

    shadow_indices = np.sort(
        rng.choice(remaining, size=int(num_shadow), replace=False)
    ).astype(np.int64)

    d_minus = ArrayDataset(
        X=np.asarray(X_train[base_indices]),
        y=np.asarray(y_train[base_indices]),
        name="d_minus",
    )
    d_bar = ArrayDataset(
        X=np.asarray(X_train[shadow_indices]),
        y=np.asarray(y_train[shadow_indices]),
        name="d_bar",
    )

    return InformedAttackData(
        d_minus=d_minus,
        target=target,
        d_bar=d_bar,
        nearest_shadow_target=_nearest_shadow_target(target, d_bar),
        base_indices=base_indices,
        shadow_indices=shadow_indices,
        target_index=int(target_index),
        target_source=target_source,
    )


def train_released_model(
    d_minus: ArrayDataset,
    target: Record,
    *,
    build_model: BuildModelFn,
    optimizer: optax.GradientTransformation,
    seed: int = 0,
    **fit_kwargs: Any,
) -> ReleaseArtifact:
    """Train the released model on D^- ∪ {z} using fit_ball_sgd."""
    model, state = _build_model_and_state(build_model, seed)
    released_dataset = _append_record(d_minus, target, name="released_train")

    from .api import fit_ball_sgd

    return fit_ball_sgd(
        model,
        optimizer,
        np.asarray(released_dataset.X),
        np.asarray(released_dataset.y),
        state=state,
        seed=int(seed),
        **fit_kwargs,
    )


def build_attack_corpus(
    d_minus: ArrayDataset,
    shadow_targets: ArrayDataset | Sequence[Record],
    *,
    build_model: BuildModelFn,
    optimizer: optax.GradientTransformation,
    corpus_cfg: Optional[ShadowCorpusConfig] = None,
    shadow_seed_policy: Literal["fixed", "vary"] = "fixed",
    fixed_shadow_seed: Optional[int] = None,
    seed: int = 0,
    **fit_kwargs: Any,
) -> ShadowCorpus:
    """Train shadow models on D^- ∪ {z̄_i} and return the reconstructor corpus."""
    targets = _shadow_records(shadow_targets)
    cfg = corpus_cfg or ShadowCorpusConfig(
        num_trials=len(targets),
        train_frac=0.7,
        val_frac=0.15,
        seed=int(seed),
        store_releases=False,
    )

    def victim_train_fn(shadow_ds: ArrayDataset, init_seed: int) -> ReleaseArtifact:
        model, state = _build_model_and_state(build_model, init_seed)

        from .api import fit_ball_sgd

        return fit_ball_sgd(
            model,
            optimizer,
            np.asarray(shadow_ds.X),
            np.asarray(shadow_ds.y),
            state=state,
            seed=int(init_seed),
            **fit_kwargs,
        )

    return build_shadow_corpus(
        d_minus=d_minus,
        shadow_targets=targets,
        victim_train_fn=victim_train_fn,
        feature_map=ParametersOnlyFeatureMap(),
        cfg=cfg,
        record_codec=FlatRecordCodec(feature_shape=d_minus.feature_shape),
        seed_policy=str(shadow_seed_policy),
        fixed_seed=None if fixed_shadow_seed is None else int(fixed_shadow_seed),
    )


def train_reconstructor(
    corpus: ShadowCorpus,
    *,
    cfg: Optional[ReconstructorTrainingConfig] = None,
) -> ReconstructorArtifact:
    """Train the RecoNN-style reconstructor."""
    cfg = cfg or ReconstructorTrainingConfig(
        hidden_dims=(1000, 1000),
        batch_size=128,
        num_epochs=30,
        patience=30,
        learning_rate=1e-3,
        seed=0,
    )
    return train_shadow_reconstructor(corpus, cfg)
