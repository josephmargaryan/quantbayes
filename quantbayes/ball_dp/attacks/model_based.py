# quantbayes/ball_dp/attacks/model_based.py

from __future__ import annotations

import dataclasses as dc
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from ..config import ReconstructorTrainingConfig, ShadowCorpusConfig
from ..metrics import reconstruction_metrics
from ..types import (
    ArrayDataset,
    AttackFeatureMap,
    AttackResult,
    ReconstructorArtifact,
    Record,
    RecordCodec,
    ReleaseArtifact,
    ShadowCorpus,
    ShadowExample,
)

VictimTrainFn = Callable[[ArrayDataset, int], ReleaseArtifact]
PredictFn = Callable[[Any, Any, jnp.ndarray, jax.Array], jnp.ndarray]


def _float_parameter_vector(
    obj: Any,
    *,
    selector: Optional[Callable[[Any], Any]] = None,
) -> np.ndarray:
    selected = obj if selector is None else selector(obj)
    filtered = eqx.filter(selected, eqx.is_inexact_array)
    leaves = []
    for leaf in jax.tree_util.tree_leaves(filtered):
        if leaf is None:
            continue
        arr = np.asarray(leaf)
        if np.issubdtype(arr.dtype, np.floating):
            leaves.append(arr.astype(np.float32, copy=False).reshape(-1))
    if not leaves:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(leaves, axis=0).astype(np.float32, copy=False)


def _class_frequencies(y: np.ndarray, label_values: Sequence[int]) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    out = np.zeros((len(label_values),), dtype=np.float32)
    if y.size == 0:
        return out
    for i, v in enumerate(label_values):
        out[i] = float(np.mean(y == int(v)))
    return out


def _dataset_stats(
    ds: ArrayDataset, *, label_values: Optional[Sequence[int]] = None
) -> np.ndarray:
    x = np.asarray(ds.X, dtype=np.float32).reshape(len(ds), -1)
    if label_values is None:
        label_values = tuple(
            int(v) for v in sorted(np.unique(np.asarray(ds.y, dtype=np.int64)).tolist())
        )
    label_values = tuple(int(v) for v in label_values)
    if x.size == 0:
        base = np.zeros((8,), dtype=np.float32)
        return np.concatenate(
            [base, np.zeros((len(label_values),), dtype=np.float32)], axis=0
        )

    norms = np.linalg.norm(x, axis=1)
    stats = np.asarray(
        [
            float(x.mean()),
            float(x.std()),
            float(norms.mean()),
            float(norms.std()),
            float(norms.min()),
            float(norms.max()),
            float(len(ds)),
            float(x.shape[1]),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [stats, _class_frequencies(np.asarray(ds.y), label_values)], axis=0
    )


def _batched_predict(
    model: Any,
    state: Any,
    predict_fn: PredictFn,
    x: np.ndarray,
    *,
    batch_size: int = 256,
    seed: int = 0,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    outputs: list[np.ndarray] = []
    rng = np.random.default_rng(int(seed))
    for lo in range(0, len(x), int(batch_size)):
        xb = jnp.asarray(x[lo : lo + int(batch_size)], dtype=jnp.float32)
        keys = jr.split(jr.PRNGKey(int(rng.integers(0, 2**31 - 1))), xb.shape[0])
        out = jax.vmap(lambda a, k: predict_fn(model, state, a, k), in_axes=(0, 0))(
            xb, keys
        )
        outputs.append(np.asarray(out, dtype=np.float32).reshape(xb.shape[0], -1))
    return np.concatenate(outputs, axis=0)


def _gaussian_projection_matrix(in_dim: int, out_dim: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    mat = rng.normal(size=(int(in_dim), int(out_dim))).astype(np.float32)
    mat /= np.sqrt(max(int(out_dim), 1))
    return mat


@dc.dataclass
class FlatRecordCodec(RecordCodec):
    feature_shape: Tuple[int, ...]
    dtype: str = "float32"

    def encode_record(self, record: Record) -> Tuple[np.ndarray, int]:
        x = np.asarray(record.features, dtype=np.dtype(self.dtype)).reshape(-1)
        return x.astype(np.float32, copy=False), int(record.label)

    def decode_record(self, features: np.ndarray, label: int) -> Record:
        x = np.asarray(features, dtype=np.dtype(self.dtype)).reshape(self.feature_shape)
        return Record(features=x, label=int(label))

    def metadata(self) -> Mapping[str, Any]:
        return {
            "feature_shape": tuple(int(v) for v in self.feature_shape),
            "dtype": self.dtype,
        }


class ParametersOnlyFeatureMap(AttackFeatureMap):
    """Attack features consisting only of model parameters.

    Optional random projection is useful when parameter vectors are very large.
    """

    def __init__(
        self,
        *,
        parameter_selector: Optional[Callable[[Any], Any]] = None,
        random_projection_dim: Optional[int] = None,
        projection_seed: int = 0,
    ):
        self.parameter_selector = parameter_selector
        self.random_projection_dim = (
            None if random_projection_dim is None else int(random_projection_dim)
        )
        self.projection_seed = int(projection_seed)
        self._projection: Optional[np.ndarray] = None
        self._input_dim: Optional[int] = None

    def _project(self, x: np.ndarray) -> np.ndarray:
        if self.random_projection_dim is None:
            return x.astype(np.float32, copy=False)
        if self._projection is None or self._input_dim != int(x.size):
            self._input_dim = int(x.size)
            self._projection = _gaussian_projection_matrix(
                int(x.size),
                int(self.random_projection_dim),
                seed=self.projection_seed,
            )
        return np.asarray(x @ self._projection, dtype=np.float32)

    def __call__(
        self, release: ReleaseArtifact, d_minus: ArrayDataset, aux: Optional[Any]
    ) -> np.ndarray:
        del d_minus, aux
        vec = _float_parameter_vector(release.payload, selector=self.parameter_selector)
        return self._project(vec)

    def metadata(self) -> Mapping[str, Any]:
        return {
            "name": "parameters_only",
            "random_projection_dim": self.random_projection_dim,
            "projection_seed": self.projection_seed,
            "parameter_selector": (
                None
                if self.parameter_selector is None
                else repr(self.parameter_selector)
            ),
        }


class ParametersPlusDatasetStatsFeatureMap(AttackFeatureMap):
    def __init__(
        self,
        *,
        parameter_selector: Optional[Callable[[Any], Any]] = None,
        random_projection_dim: Optional[int] = None,
        projection_seed: int = 0,
        label_values: Optional[Sequence[int]] = None,
    ):
        self.param_map = ParametersOnlyFeatureMap(
            parameter_selector=parameter_selector,
            random_projection_dim=random_projection_dim,
            projection_seed=projection_seed,
        )
        self.label_values = (
            None if label_values is None else tuple(int(v) for v in label_values)
        )

    def __call__(
        self, release: ReleaseArtifact, d_minus: ArrayDataset, aux: Optional[Any]
    ) -> np.ndarray:
        del aux
        return np.concatenate(
            [
                self.param_map(release, d_minus, None),
                _dataset_stats(d_minus, label_values=self.label_values),
            ],
            axis=0,
        )

    def metadata(self) -> Mapping[str, Any]:
        meta = dict(self.param_map.metadata())
        meta["name"] = "parameters_plus_dataset_stats"
        meta["label_values"] = self.label_values
        return meta


class LogitsAndDatasetStatsFeatureMap(AttackFeatureMap):
    def __init__(
        self,
        *,
        anchor_x: np.ndarray,
        predict_fn: PredictFn,
        state: Any = None,
        logits_batch_size: int = 256,
        logits_seed: int = 0,
        label_values: Optional[Sequence[int]] = None,
    ):
        self.anchor_x = np.asarray(anchor_x, dtype=np.float32)
        self.predict_fn = predict_fn
        self.state = state
        self.logits_batch_size = int(logits_batch_size)
        self.logits_seed = int(logits_seed)
        self.label_values = (
            None if label_values is None else tuple(int(v) for v in label_values)
        )

    def __call__(
        self, release: ReleaseArtifact, d_minus: ArrayDataset, aux: Optional[Any]
    ) -> np.ndarray:
        del aux
        state = (
            self.state
            if self.state is not None
            else release.extra.get("model_state", None)
        )
        logits = _batched_predict(
            release.payload,
            state,
            self.predict_fn,
            self.anchor_x,
            batch_size=self.logits_batch_size,
            seed=self.logits_seed,
        ).reshape(-1)
        return np.concatenate(
            [
                logits.astype(np.float32, copy=False),
                _dataset_stats(d_minus, label_values=self.label_values),
            ],
            axis=0,
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "name": "logits_and_dataset_stats",
            "anchor_n": int(len(self.anchor_x)),
            "logits_batch_size": self.logits_batch_size,
            "logits_seed": self.logits_seed,
            "label_values": self.label_values,
        }


class ParametersLogitsAndDatasetStatsFeatureMap(AttackFeatureMap):
    def __init__(
        self,
        *,
        anchor_x: np.ndarray,
        predict_fn: PredictFn,
        state: Any = None,
        parameter_selector: Optional[Callable[[Any], Any]] = None,
        random_projection_dim: Optional[int] = None,
        projection_seed: int = 0,
        logits_batch_size: int = 256,
        logits_seed: int = 0,
        label_values: Optional[Sequence[int]] = None,
    ):
        self.param_map = ParametersOnlyFeatureMap(
            parameter_selector=parameter_selector,
            random_projection_dim=random_projection_dim,
            projection_seed=projection_seed,
        )
        self.logit_map = LogitsAndDatasetStatsFeatureMap(
            anchor_x=anchor_x,
            predict_fn=predict_fn,
            state=state,
            logits_batch_size=logits_batch_size,
            logits_seed=logits_seed,
            label_values=label_values,
        )

    def __call__(
        self, release: ReleaseArtifact, d_minus: ArrayDataset, aux: Optional[Any]
    ) -> np.ndarray:
        del aux
        logits_and_stats = self.logit_map(release, d_minus, None)
        return np.concatenate(
            [self.param_map(release, d_minus, None), logits_and_stats], axis=0
        )

    def metadata(self) -> Mapping[str, Any]:
        meta = dict(self.param_map.metadata())
        meta.update(self.logit_map.metadata())
        meta["name"] = "parameters_logits_and_dataset_stats"
        return meta


def make_attack_feature_map(
    name: str,
    *,
    parameter_selector: Optional[Callable[[Any], Any]] = None,
    random_projection_dim: Optional[int] = None,
    projection_seed: int = 0,
    anchor_x: Optional[np.ndarray] = None,
    predict_fn: Optional[PredictFn] = None,
    state: Any = None,
    logits_batch_size: int = 256,
    logits_seed: int = 0,
    label_values: Optional[Sequence[int]] = None,
) -> AttackFeatureMap:
    """Factory for supported shadow-attack feature maps."""
    key = str(name).strip().lower()
    if key == "parameters_only":
        return ParametersOnlyFeatureMap(
            parameter_selector=parameter_selector,
            random_projection_dim=random_projection_dim,
            projection_seed=projection_seed,
        )
    if key == "parameters_plus_dataset_stats":
        return ParametersPlusDatasetStatsFeatureMap(
            parameter_selector=parameter_selector,
            random_projection_dim=random_projection_dim,
            projection_seed=projection_seed,
            label_values=label_values,
        )
    if key == "logits_and_dataset_stats":
        if anchor_x is None or predict_fn is None:
            raise ValueError(
                "logits_and_dataset_stats requires both anchor_x=... and predict_fn=... ."
            )
        return LogitsAndDatasetStatsFeatureMap(
            anchor_x=anchor_x,
            predict_fn=predict_fn,
            state=state,
            logits_batch_size=logits_batch_size,
            logits_seed=logits_seed,
            label_values=label_values,
        )
    if key == "parameters_logits_and_dataset_stats":
        if anchor_x is None or predict_fn is None:
            raise ValueError(
                "parameters_logits_and_dataset_stats requires both anchor_x=... and predict_fn=... ."
            )
        return ParametersLogitsAndDatasetStatsFeatureMap(
            anchor_x=anchor_x,
            predict_fn=predict_fn,
            state=state,
            parameter_selector=parameter_selector,
            random_projection_dim=random_projection_dim,
            projection_seed=projection_seed,
            logits_batch_size=logits_batch_size,
            logits_seed=logits_seed,
            label_values=label_values,
        )
    raise ValueError(f"Unsupported attack feature map: {name!r}")


@dc.dataclass
class _Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> "_Standardizer":
        mean = np.asarray(x.mean(axis=0), dtype=np.float32)
        std = np.asarray(x.std(axis=0), dtype=np.float32)
        std = np.where(std > float(eps), std, np.ones_like(std, dtype=np.float32))
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(
            np.float32
        )


class _ReconstructorMLP(eqx.Module):
    hidden: Tuple[eqx.nn.Linear, ...]
    feat_head: eqx.nn.Linear
    label_head: eqx.nn.Linear

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        target_dim: int,
        num_classes: int,
        *,
        key: jax.Array,
    ):
        hidden_dims = tuple(int(v) for v in hidden_dims)
        keys = jr.split(key, len(hidden_dims) + 2)
        hidden: list[eqx.nn.Linear] = []
        prev = int(input_dim)
        for i, width in enumerate(hidden_dims):
            hidden.append(eqx.nn.Linear(prev, int(width), key=keys[i]))
            prev = int(width)
        self.hidden = tuple(hidden)
        self.feat_head = eqx.nn.Linear(prev, int(target_dim), key=keys[-2])
        self.label_head = eqx.nn.Linear(prev, int(num_classes), key=keys[-1])

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = jnp.asarray(x)
        for layer in self.hidden:
            h = jax.nn.relu(layer(h))
        return self.feat_head(h), self.label_head(h)


def _huber_loss(x: jnp.ndarray, y: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    err = x - y
    abs_err = jnp.abs(err)
    quad = jnp.minimum(abs_err, delta)
    lin = abs_err - quad
    return jnp.mean(0.5 * quad * quad + delta * lin)


def _feature_loss(kind: str, pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    key = str(kind).lower()
    if key == "mse_ce":
        return jnp.mean(jnp.square(pred - target))
    if key == "l1_ce":
        return jnp.mean(jnp.abs(pred - target))
    if key == "huber_ce":
        return _huber_loss(pred, target)
    raise ValueError(f"Unsupported reconstructor loss: {kind!r}")


def _split_counts(n: int, train_frac: float, val_frac: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    n_train = int(np.floor(float(train_frac) * n))
    n_val = int(np.floor(float(val_frac) * n))
    n_train = min(max(n_train, 1), n)
    n_val = min(max(n_val, 0), n - n_train)
    n_test = n - n_train - n_val
    if n_test <= 0 and n >= 3:
        n_test = 1
        if n_val > 0:
            n_val -= 1
        else:
            n_train -= 1
    return int(n_train), int(n_val), int(n_test)


def _augment_with_side_info(
    attack_features: np.ndarray,
    *,
    side_info_regime: str,
    label: Optional[int],
    label_values: Sequence[int],
) -> np.ndarray:
    x = np.asarray(attack_features, dtype=np.float32).reshape(-1)
    if str(side_info_regime) != "known_label":
        return x
    if label is None:
        raise ValueError(
            "side_info_regime='known_label' requires label to be provided."
        )
    mapping = {int(v): i for i, v in enumerate(label_values)}
    if int(label) not in mapping:
        raise ValueError(
            f"Label {label!r} not present in label mapping {tuple(label_values)!r}."
        )
    onehot = np.zeros((len(label_values),), dtype=np.float32)
    onehot[mapping[int(label)]] = 1.0
    return np.concatenate([x, onehot], axis=0)


def _append_record(ds: ArrayDataset, record: Record) -> ArrayDataset:
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
    return ArrayDataset(x, y, name=f"{ds.name}_plus_target")


def build_shadow_corpus(
    *,
    d_minus: ArrayDataset,
    shadow_targets: Sequence[Record],
    victim_train_fn: VictimTrainFn,
    feature_map: AttackFeatureMap,
    cfg: ShadowCorpusConfig,
    record_codec: Optional[RecordCodec] = None,
    seed_policy: Literal["vary", "fixed"] = "vary",
    fixed_seed: Optional[int] = None,
) -> ShadowCorpus:
    """Build a shadow corpus for the informed-adversary model-based attack.

    The fixed dataset is the attacker's known ``D_-``. Each shadow example is produced by
    training a victim release on ``D_- ∪ {z_i}`` for an auxiliary candidate ``z_i``.
    """
    if not shadow_targets:
        raise ValueError("shadow_targets must be non-empty.")

    codec = record_codec or FlatRecordCodec(feature_shape=d_minus.feature_shape)
    rng = np.random.default_rng(int(cfg.seed))
    label_values = tuple(sorted({int(r.label) for r in shadow_targets}))

    n_total = int(cfg.num_trials)
    replace = n_total > len(shadow_targets)
    choice = rng.choice(len(shadow_targets), size=n_total, replace=replace)

    examples: list[ShadowExample] = []
    for trial_idx, target_idx in enumerate(choice.tolist()):
        target = shadow_targets[int(target_idx)]
        seed = int(
            fixed_seed
            if seed_policy == "fixed" and fixed_seed is not None
            else (cfg.seed if seed_policy == "fixed" else (cfg.seed + trial_idx))
        )
        shadow_ds = _append_record(d_minus, target)
        release = victim_train_fn(shadow_ds, seed)

        attack_features = np.asarray(
            feature_map(
                release,
                d_minus,
                (
                    {"known_label": int(target.label)}
                    if str(cfg.side_info_regime) == "known_label"
                    else None
                ),
            ),
            dtype=np.float32,
        ).reshape(-1)
        attack_features = _augment_with_side_info(
            attack_features,
            side_info_regime=str(cfg.side_info_regime),
            label=(
                int(target.label)
                if str(cfg.side_info_regime) == "known_label"
                else None
            ),
            label_values=label_values,
        )
        target_features, target_label = codec.encode_record(target)

        metadata: Dict[str, Any] = {
            "trial_index": int(trial_idx),
            "seed": int(seed),
            "release_kind": str(release.release_kind),
            "model_family": str(release.model_family),
        }
        if bool(cfg.store_releases):
            metadata["release"] = release

        examples.append(
            ShadowExample(
                attack_features=np.asarray(attack_features, dtype=np.float32),
                target_features=np.asarray(target_features, dtype=np.float32),
                target_label=int(target_label),
                metadata=metadata,
            )
        )

    perm = rng.permutation(len(examples))
    n_train, n_val, n_test = _split_counts(len(examples), cfg.train_frac, cfg.val_frac)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val : n_train + n_val + n_test]

    def _gather(idxs: np.ndarray) -> list[ShadowExample]:
        return [examples[int(i)] for i in idxs.tolist()]

    return ShadowCorpus(
        train_examples=_gather(train_idx),
        val_examples=_gather(val_idx),
        test_examples=_gather(test_idx),
        config_snapshot=dc.asdict(cfg),
        codec_metadata=dict(codec.metadata()),
        feature_metadata={
            "attack_feature_map": dict(feature_map.metadata()),
            "side_info_regime": str(cfg.side_info_regime),
            "label_values": tuple(int(v) for v in label_values),
            "input_feature_dim": int(examples[0].attack_features.size),
        },
    )


def _stack_examples(
    examples: Sequence[ShadowExample], label_mapping: Mapping[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not examples:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    x = np.stack(
        [
            np.asarray(ex.attack_features, dtype=np.float32).reshape(-1)
            for ex in examples
        ],
        axis=0,
    )
    y_feat = np.stack(
        [
            np.asarray(ex.target_features, dtype=np.float32).reshape(-1)
            for ex in examples
        ],
        axis=0,
    )
    y_lbl = np.asarray(
        [label_mapping[int(ex.target_label)] for ex in examples], dtype=np.int32
    )
    return x, y_feat, y_lbl


def train_shadow_reconstructor(
    corpus: ShadowCorpus, cfg: ReconstructorTrainingConfig
) -> ReconstructorArtifact:
    """Train a shadow-model reconstructor for final-model nonconvex attacks."""
    all_examples = (
        list(corpus.train_examples)
        + list(corpus.val_examples)
        + list(corpus.test_examples)
    )
    if not all_examples:
        raise ValueError("Shadow corpus is empty.")

    label_values = tuple(
        int(v) for v in corpus.feature_metadata.get("label_values", (0,))
    )
    label_mapping = {int(v): i for i, v in enumerate(label_values)} or {0: 0}
    if not label_values:
        label_values = (0,)

    x_train, y_feat_train, y_lbl_train = _stack_examples(
        corpus.train_examples, label_mapping
    )
    x_val, y_feat_val, y_lbl_val = _stack_examples(corpus.val_examples, label_mapping)

    if x_train.shape[0] == 0:
        raise ValueError("Shadow corpus needs at least one training example.")

    x_std = _Standardizer.fit(x_train)
    y_std = _Standardizer.fit(y_feat_train)

    x_train_n = x_std.transform(x_train)
    y_feat_train_n = y_std.transform(y_feat_train)
    x_val_n = x_std.transform(x_val) if x_val.shape[0] else x_val
    y_feat_val_n = y_std.transform(y_feat_val) if y_feat_val.shape[0] else y_feat_val

    input_dim = int(x_train_n.shape[1])
    target_dim = int(y_feat_train_n.shape[1])
    num_classes = int(len(label_values))

    key = jr.PRNGKey(int(cfg.seed))
    model = _ReconstructorMLP(
        input_dim=input_dim,
        hidden_dims=tuple(int(v) for v in cfg.hidden_dims),
        target_dim=target_dim,
        num_classes=num_classes,
        key=key,
    )
    opt = optax.adamw(
        learning_rate=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay)
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def _batch_loss(model: _ReconstructorMLP, xb, yb_feat, yb_lbl):
        feat_pred, lbl_logits = jax.vmap(model)(xb)
        feat_loss = _feature_loss(cfg.loss, feat_pred, yb_feat)
        lbl_loss = optax.softmax_cross_entropy_with_integer_labels(
            lbl_logits, yb_lbl
        ).mean()
        total = (
            float(cfg.feature_loss_weight) * feat_loss
            + float(cfg.label_loss_weight) * lbl_loss
        )
        acc = jnp.mean((jnp.argmax(lbl_logits, axis=-1) == yb_lbl).astype(jnp.float32))
        return total, (feat_loss, lbl_loss, acc)

    @eqx.filter_jit
    def _step(model, opt_state, xb, yb_feat, yb_lbl):
        (loss, aux), grads = eqx.filter_value_and_grad(_batch_loss, has_aux=True)(
            model, xb, yb_feat, yb_lbl
        )
        updates, opt_state = opt.update(
            eqx.filter(grads, eqx.is_inexact_array),
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    @eqx.filter_jit
    def _evaluate(model, xb, yb_feat, yb_lbl):
        return _batch_loss(model, xb, yb_feat, yb_lbl)

    rng = np.random.default_rng(int(cfg.seed))
    best_model = model
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    batch_size = max(1, int(cfg.batch_size))
    num_epochs = max(1, int(cfg.num_epochs))
    patience = max(1, int(cfg.patience))

    for epoch in range(1, num_epochs + 1):
        perm = rng.permutation(len(x_train_n))
        for lo in range(0, len(perm), batch_size):
            idx = perm[lo : lo + batch_size]
            xb = jnp.asarray(x_train_n[idx], dtype=jnp.float32)
            yb_feat = jnp.asarray(y_feat_train_n[idx], dtype=jnp.float32)
            yb_lbl = jnp.asarray(y_lbl_train[idx], dtype=jnp.int32)
            model, opt_state, _, _ = _step(model, opt_state, xb, yb_feat, yb_lbl)

        eval_x = x_val_n if x_val_n.shape[0] > 0 else x_train_n
        eval_feat = y_feat_val_n if y_feat_val_n.shape[0] > 0 else y_feat_train_n
        eval_lbl = y_lbl_val if x_val_n.shape[0] > 0 else y_lbl_train
        val_loss, _ = _evaluate(
            model,
            jnp.asarray(eval_x, dtype=jnp.float32),
            jnp.asarray(eval_feat, dtype=jnp.float32),
            jnp.asarray(eval_lbl, dtype=jnp.int32),
        )
        current_val = float(val_loss)

        if current_val + 1e-12 < best_val:
            best_val = current_val
            best_epoch = int(epoch)
            best_model = model
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    x_test, y_feat_test, y_lbl_test = _stack_examples(
        corpus.test_examples, label_mapping
    )
    test_metrics: Dict[str, float] = {}
    if x_test.shape[0] > 0:
        x_test_n = x_std.transform(x_test)
        y_feat_test_n = y_std.transform(y_feat_test)
        test_loss, (test_feat_loss, test_lbl_loss, test_acc) = _evaluate(
            best_model,
            jnp.asarray(x_test_n, dtype=jnp.float32),
            jnp.asarray(y_feat_test_n, dtype=jnp.float32),
            jnp.asarray(y_lbl_test, dtype=jnp.int32),
        )
        test_metrics = {
            "test_total_loss": float(test_loss),
            "test_feature_loss": float(test_feat_loss),
            "test_label_loss": float(test_lbl_loss),
            "test_label_accuracy": float(test_acc),
        }

    return ReconstructorArtifact(
        model=best_model,
        state=None,
        config_snapshot={
            "reconstructor": dc.asdict(cfg),
            "feature_metadata": dict(corpus.feature_metadata),
            "codec_metadata": dict(corpus.codec_metadata),
        },
        validation_metrics={
            "best_val_total_loss": float(best_val),
            "best_epoch": int(best_epoch),
            **test_metrics,
        },
        feature_dim=int(input_dim),
        target_feature_dim=int(target_dim),
        num_classes=int(num_classes),
        extra={
            "x_mean": np.asarray(x_std.mean, dtype=np.float32),
            "x_std": np.asarray(x_std.std, dtype=np.float32),
            "target_mean": np.asarray(y_std.mean, dtype=np.float32),
            "target_std": np.asarray(y_std.std, dtype=np.float32),
            "feature_shape": tuple(
                int(v)
                for v in corpus.codec_metadata.get("feature_shape", (target_dim,))
            ),
            "dtype": str(corpus.codec_metadata.get("dtype", "float32")),
            "label_values": tuple(int(v) for v in label_values),
            "side_info_regime": str(
                corpus.feature_metadata.get("side_info_regime", "none")
            ),
            "attack_feature_map": dict(
                corpus.feature_metadata.get("attack_feature_map", {})
            ),
        },
    )


def run_model_based_attack(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    reconstructor: ReconstructorArtifact,
    feature_map: AttackFeatureMap,
    true_record: Optional[Record] = None,
    known_label: Optional[int] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    ball_center: Optional[np.ndarray] = None,
    ball_radius: Optional[float] = None,
    box_bounds: Optional[Tuple[float, float]] = None,
) -> AttackResult:
    """Run the final-model shadow-reconstructor attack.

    This attack matches the informed-adversary setting of Balle et al.: the attacker knows
    ``D_-``, the release, the training algorithm, and optional side information such as the label.
    """
    label_values = tuple(int(v) for v in reconstructor.extra.get("label_values", (0,)))
    side_info_regime = str(reconstructor.extra.get("side_info_regime", "none"))
    if side_info_regime == "known_label" and known_label is None:
        raise ValueError(
            "This reconstructor was trained with side_info_regime='known_label'; please pass known_label=... at attack time."
        )

    attack_features = np.asarray(
        feature_map(
            release,
            d_minus,
            {"known_label": int(known_label)} if known_label is not None else None,
        ),
        dtype=np.float32,
    ).reshape(-1)
    attack_features = _augment_with_side_info(
        attack_features,
        side_info_regime=side_info_regime,
        label=None if known_label is None else int(known_label),
        label_values=label_values,
    )

    x_mean = np.asarray(reconstructor.extra["x_mean"], dtype=np.float32)
    x_std = np.asarray(reconstructor.extra["x_std"], dtype=np.float32)
    y_mean = np.asarray(reconstructor.extra["target_mean"], dtype=np.float32)
    y_std = np.asarray(reconstructor.extra["target_std"], dtype=np.float32)

    x_norm = ((attack_features - x_mean) / x_std).astype(np.float32)
    recon_model = eqx.nn.inference_mode(reconstructor.model, value=True)
    feat_pred_n, label_logits = recon_model(jnp.asarray(x_norm, dtype=jnp.float32))
    feat_pred = np.asarray(feat_pred_n, dtype=np.float32) * y_std + y_mean
    label_logits = np.asarray(label_logits, dtype=np.float32)

    if box_bounds is not None:
        lo, hi = float(box_bounds[0]), float(box_bounds[1])
        feat_pred = np.clip(feat_pred, lo, hi)

    if ball_center is not None and ball_radius is not None:
        center = np.asarray(ball_center, dtype=np.float32).reshape(-1)
        diff = feat_pred.reshape(-1) - center
        norm = float(np.linalg.norm(diff))
        radius = float(ball_radius)
        if norm > radius and norm > 0.0:
            feat_pred = center + (radius / norm) * diff

    if known_label is not None:
        y_hat = int(known_label)
        status = "ok_known_label"
    else:
        idx = int(np.argmax(label_logits))
        y_hat = int(label_values[idx]) if idx < len(label_values) else idx
        status = "ok"

    feature_shape = tuple(
        int(v) for v in reconstructor.extra.get("feature_shape", (feat_pred.size,))
    )
    pred_features = np.asarray(feat_pred, dtype=np.float32).reshape(feature_shape)
    pred_record = Record(features=pred_features, label=int(y_hat))
    metrics = (
        {}
        if true_record is None
        else reconstruction_metrics(true_record, pred_record, eta_grid=eta_grid)
    )

    return AttackResult(
        attack_family="model_based_shadow_reconstructor",
        z_hat=np.asarray(pred_features),
        y_hat=int(y_hat),
        status=status,
        diagnostics={
            "label_logits": label_logits,
            "attack_feature_dim": int(attack_features.size),
            "reconstructor_validation_metrics": dict(reconstructor.validation_metrics),
            "feature_map_metadata": dict(
                reconstructor.extra.get("attack_feature_map", {})
            ),
        },
        metrics=metrics,
    )
