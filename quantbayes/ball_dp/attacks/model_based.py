# quantbayes/ball_dp/attacks/model_based.py

from __future__ import annotations

import dataclasses as dc
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Literal

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
    """Paper-faithful feature map for the informed-adversary attack.

    The attack model receives the flattened parameter vector of the released model
    as input. An optional `parameter_selector` is exposed to support the paper's
    practical observation that using only later layers can help for very wide
    models.
    """

    def __init__(
        self,
        *,
        parameter_selector: Optional[Callable[[Any], Any]] = None,
    ):
        self.parameter_selector = parameter_selector

    def __call__(
        self, release: ReleaseArtifact, d_minus: ArrayDataset, aux: Optional[Any]
    ) -> np.ndarray:
        del d_minus, aux
        return _float_parameter_vector(
            release.payload,
            selector=self.parameter_selector,
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "name": "parameters_only",
            "parameter_selector": (
                None
                if self.parameter_selector is None
                else repr(self.parameter_selector)
            ),
        }


def make_attack_feature_map(
    name: str,
    *,
    parameter_selector: Optional[Callable[[Any], Any]] = None,
) -> AttackFeatureMap:
    key = str(name).strip().lower().replace("-", "_")
    if key in {"parameters_only", "reconn", "model_parameters"}:
        return ParametersOnlyFeatureMap(parameter_selector=parameter_selector)
    raise ValueError(
        "The stable informed-adversary model-based attack supports only "
        "'parameters_only' / 'reconn'."
    )


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


def _normalize_rows_np(x: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if mode == "none":
        return arr
    if mode != "l2":
        raise ValueError("mode must be one of {'none', 'l2'}.")
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (arr / norms).astype(np.float32, copy=False)


def _normalize_last_axis_jnp(x: jnp.ndarray, mode: str) -> jnp.ndarray:
    if mode == "none":
        return x
    if mode != "l2":
        raise ValueError("mode must be one of {'none', 'l2'}.")
    norms = jnp.linalg.norm(x, axis=-1, keepdims=True)
    norms = jnp.maximum(norms, 1e-12)
    return x / norms


class _RecoNN(eqx.Module):
    hidden: Tuple[eqx.nn.Linear, ...]
    out: eqx.nn.Linear
    use_sigmoid_output: bool = dc.field(default=False)

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        target_dim: int,
        *,
        key: jax.Array,
        use_sigmoid_output: bool,
    ):
        hidden_dims = tuple(int(v) for v in hidden_dims)
        keys = jr.split(key, len(hidden_dims) + 1)
        hidden: list[eqx.nn.Linear] = []
        prev = int(input_dim)
        for i, width in enumerate(hidden_dims):
            hidden.append(eqx.nn.Linear(prev, int(width), key=keys[i]))
            prev = int(width)
        self.hidden = tuple(hidden)
        self.out = eqx.nn.Linear(prev, int(target_dim), key=keys[-1])
        self.use_sigmoid_output = bool(use_sigmoid_output)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = jnp.asarray(x)
        for layer in self.hidden:
            h = jax.nn.relu(layer(h))
        out = self.out(h)
        if self.use_sigmoid_output:
            out = jax.nn.sigmoid(out)
        return out


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
    """Construct the shadow dataset for the informed-adversary attack.

    This follows the paper's three-step workflow: keep D^- fixed, train one shadow
    model on D^- ∪ {z_i} per auxiliary target z_i, then train a reconstructor on
    the resulting (flattened parameters, target) pairs.
    """
    if not shadow_targets:
        raise ValueError("shadow_targets must be non-empty.")

    codec = record_codec or FlatRecordCodec(feature_shape=d_minus.feature_shape)
    rng = np.random.default_rng(int(cfg.seed))
    label_values = tuple(int(v) for v in sorted({int(r.label) for r in shadow_targets}))

    n_total = int(cfg.num_trials)
    replace = n_total > len(shadow_targets)
    choice = rng.choice(len(shadow_targets), size=n_total, replace=replace)

    examples: list[ShadowExample] = []
    for trial_idx, target_idx in enumerate(choice.tolist()):
        target = shadow_targets[int(target_idx)]
        if seed_policy == "fixed":
            shadow_seed = int(cfg.seed if fixed_seed is None else fixed_seed)
        elif seed_policy == "vary":
            shadow_seed = int(cfg.seed + trial_idx)
        else:
            raise ValueError("seed_policy must be one of {'vary', 'fixed'}.")

        shadow_ds = _append_record(d_minus, target)
        release = victim_train_fn(shadow_ds, shadow_seed)
        attack_features = np.asarray(
            feature_map(release, d_minus, None),
            dtype=np.float32,
        ).reshape(-1)
        target_features, target_label = codec.encode_record(target)

        metadata: dict[str, Any] = {
            "trial_index": int(trial_idx),
            "seed": int(shadow_seed),
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
    n_train = int(np.floor(float(cfg.train_frac) * len(examples)))
    n_val = int(np.floor(float(cfg.val_frac) * len(examples)))
    n_train = min(max(n_train, 1), len(examples))
    n_val = min(max(n_val, 0), len(examples) - n_train)
    n_test = len(examples) - n_train - n_val
    if n_test <= 0 and len(examples) >= 3:
        n_test = 1
        if n_val > 0:
            n_val -= 1
        else:
            n_train -= 1

    def _gather(idxs: np.ndarray) -> list[ShadowExample]:
        return [examples[int(i)] for i in idxs.tolist()]

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val : n_train + n_val + n_test]

    return ShadowCorpus(
        train_examples=_gather(train_idx),
        val_examples=_gather(val_idx),
        test_examples=_gather(test_idx),
        config_snapshot=dc.asdict(cfg),
        codec_metadata=dict(codec.metadata()),
        feature_metadata={
            "attack_feature_map": dict(feature_map.metadata()),
            "label_values": tuple(int(v) for v in label_values),
            "input_feature_dim": int(examples[0].attack_features.size),
            "seed_policy": str(seed_policy),
            "fixed_seed": None if fixed_seed is None else int(fixed_seed),
        },
    )


def _stack_examples(
    examples: Sequence[ShadowExample],
) -> Tuple[np.ndarray, np.ndarray]:
    if not examples:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
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
    return x, y_feat


def train_shadow_reconstructor(
    corpus: ShadowCorpus,
    cfg: ReconstructorTrainingConfig,
) -> ReconstructorArtifact:
    """Train the RecoNN-style informed-adversary reconstructor.

    Defaults follow DeepMind's public minimal implementation closely:
      - flattened parameter vectors as input,
      - train-only standardization on parameter input,
      - two hidden layers of width 1000,
      - RMSProp optimizer,
      - feature reconstruction loss = MSE + MAE,
      - fixed initialization seed.
    """
    all_examples = (
        list(corpus.train_examples)
        + list(corpus.val_examples)
        + list(corpus.test_examples)
    )
    if not all_examples:
        raise ValueError("Shadow corpus is empty.")

    x_train, y_feat_train = _stack_examples(corpus.train_examples)
    x_val, y_feat_val = _stack_examples(corpus.val_examples)
    x_test, y_feat_test = _stack_examples(corpus.test_examples)

    if x_train.shape[0] == 0:
        raise ValueError("Shadow corpus needs at least one training example.")

    x_std = _Standardizer.fit(x_train)
    x_train_n = x_std.transform(x_train)
    x_val_n = x_std.transform(x_val) if x_val.shape[0] else x_val
    x_test_n = x_std.transform(x_test) if x_test.shape[0] else x_test

    input_dim = int(x_train_n.shape[1])
    target_dim = int(y_feat_train.shape[1])
    label_values = tuple(
        int(v) for v in corpus.feature_metadata.get("label_values", (0,))
    )
    if not label_values:
        label_values = (0,)

    target_normalization = str(cfg.target_normalization)
    output_normalization = str(cfg.output_normalization)
    loss_name = str(cfg.loss_name)

    if loss_name == "cosine" and (
        target_normalization != "l2" or output_normalization != "l2"
    ):
        raise ValueError(
            "loss_name='cosine' requires target_normalization='l2' "
            "and output_normalization='l2'."
        )

    y_feat_train = _normalize_rows_np(y_feat_train, target_normalization)
    y_feat_val = _normalize_rows_np(y_feat_val, target_normalization)
    y_feat_test = _normalize_rows_np(y_feat_test, target_normalization)

    use_sigmoid_output = False

    hidden_dims = (
        tuple(int(v) for v in cfg.hidden_dims) if cfg.hidden_dims else (1000, 1000)
    )
    key = jr.PRNGKey(int(cfg.seed))
    model = _RecoNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        target_dim=target_dim,
        key=key,
        use_sigmoid_output=use_sigmoid_output,
    )

    optimizer = optax.rmsprop(float(cfg.learning_rate))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def _loss_fn(model: _RecoNN, xb, yb_feat):
        pred_raw = jax.vmap(model)(xb)
        pred = _normalize_last_axis_jnp(pred_raw, output_normalization)
        mse = jnp.mean(jnp.square(pred - yb_feat))
        mae = jnp.mean(jnp.abs(pred - yb_feat))
        cosine_similarity = jnp.mean(jnp.sum(pred * yb_feat, axis=-1))
        cosine_error = 1.0 - cosine_similarity

        if loss_name == "mse_mae":
            total = mse + mae
        elif loss_name == "cosine":
            total = cosine_error
        else:
            raise ValueError("loss_name must be one of {'mse_mae', 'cosine'}.")

        return total, (mae, mse, cosine_similarity, cosine_error)

    @eqx.filter_jit
    def _step(model, opt_state, xb, yb_feat):
        (loss, aux), grads = eqx.filter_value_and_grad(_loss_fn, has_aux=True)(
            model, xb, yb_feat
        )
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_inexact_array),
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    @eqx.filter_jit
    def _evaluate_batch(model, xb, yb_feat):
        return _loss_fn(model, xb, yb_feat)

    def _evaluate_split(model, x_arr, y_arr, *, batch_size_eval=1024):
        n = int(x_arr.shape[0])
        if n == 0:
            return {
                "total_loss": float("nan"),
                "mae": float("nan"),
                "mse": float("nan"),
                "cosine_similarity": float("nan"),
                "cosine_error": float("nan"),
            }

        total_sum = 0.0
        mae_sum = 0.0
        mse_sum = 0.0
        cosine_similarity_sum = 0.0
        cosine_error_sum = 0.0

        for lo in range(0, n, int(batch_size_eval)):
            hi = min(lo + int(batch_size_eval), n)
            xb = jnp.asarray(x_arr[lo:hi], dtype=jnp.float32)
            yb = jnp.asarray(y_arr[lo:hi], dtype=jnp.float32)
            loss_b, (mae_b, mse_b, cosine_similarity_b, cosine_error_b) = (
                _evaluate_batch(model, xb, yb)
            )
            m = hi - lo
            total_sum += float(loss_b) * m
            mae_sum += float(mae_b) * m
            mse_sum += float(mse_b) * m
            cosine_similarity_sum += float(cosine_similarity_b) * m
            cosine_error_sum += float(cosine_error_b) * m

        return {
            "total_loss": total_sum / float(n),
            "mae": mae_sum / float(n),
            "mse": mse_sum / float(n),
            "cosine_similarity": cosine_similarity_sum / float(n),
            "cosine_error": cosine_error_sum / float(n),
        }

    rng = np.random.default_rng(int(cfg.seed))
    best_model = model
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    batch_size = max(1, int(cfg.batch_size))
    num_epochs = max(1, int(cfg.num_epochs))
    patience = max(1, int(cfg.patience))

    train_curve_history: list[dict[str, float]] = []
    val_curve_history: list[dict[str, float]] = []
    update_loss_history: list[dict[str, float]] = []

    used_val_split = x_val_n.shape[0] > 0

    for epoch in range(1, num_epochs + 1):
        perm = rng.permutation(len(x_train_n))

        # Per-minibatch train losses.
        for step_in_epoch, lo in enumerate(range(0, len(perm), batch_size), start=1):
            idx = perm[lo : lo + batch_size]
            xb = jnp.asarray(x_train_n[idx], dtype=jnp.float32)
            yb_feat = jnp.asarray(y_feat_train[idx], dtype=jnp.float32)
            (
                model,
                opt_state,
                step_loss,
                (
                    step_mae,
                    step_mse,
                    step_cosine_similarity,
                    step_cosine_error,
                ),
            ) = _step(model, opt_state, xb, yb_feat)

            update_loss_history.append(
                {
                    "epoch": int(epoch),
                    "step_in_epoch": int(step_in_epoch),
                    "global_step": int(len(update_loss_history) + 1),
                    "train_total_loss": float(step_loss),
                    "train_mae": float(step_mae),
                    "train_mse": float(step_mse),
                    "train_cosine_similarity": float(step_cosine_similarity),
                    "train_cosine_error": float(step_cosine_error),
                    "batch_size": int(len(idx)),
                }
            )

        # Full-train epoch metrics.
        train_metrics = _evaluate_split(
            model,
            x_train_n,
            y_feat_train,
            batch_size_eval=max(256, batch_size),
        )
        train_curve_history.append(
            {
                "epoch": int(epoch),
                "train_total_loss": float(train_metrics["total_loss"]),
                "train_mae": float(train_metrics["mae"]),
                "train_mse": float(train_metrics["mse"]),
                "train_cosine_similarity": float(train_metrics["cosine_similarity"]),
                "train_cosine_error": float(train_metrics["cosine_error"]),
            }
        )

        # Full-val epoch metrics (or train metrics if no val split exists).
        eval_x = x_val_n if used_val_split else x_train_n
        eval_feat = y_feat_val if used_val_split else y_feat_train
        val_metrics = _evaluate_split(
            model,
            eval_x,
            eval_feat,
            batch_size_eval=max(256, batch_size),
        )
        val_curve_history.append(
            {
                "epoch": int(epoch),
                "val_total_loss": float(val_metrics["total_loss"]),
                "val_mae": float(val_metrics["mae"]),
                "val_mse": float(val_metrics["mse"]),
                "val_cosine_similarity": float(val_metrics["cosine_similarity"]),
                "val_cosine_error": float(val_metrics["cosine_error"]),
            }
        )

        current_val = float(val_metrics["total_loss"])

        if current_val + 1e-12 < best_val:
            best_val = current_val
            best_epoch = int(epoch)
            best_model = model
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # Final metrics on the selected best model.
    final_train_metrics = _evaluate_split(
        best_model,
        x_train_n,
        y_feat_train,
        batch_size_eval=max(256, batch_size),
    )

    if used_val_split:
        final_val_metrics = _evaluate_split(
            best_model,
            x_val_n,
            y_feat_val,
            batch_size_eval=max(256, batch_size),
        )
    else:
        final_val_metrics = dict(final_train_metrics)

    test_metrics: dict[str, float] = {}
    if x_test_n.shape[0] > 0:
        test_eval = _evaluate_split(
            best_model,
            x_test_n,
            y_feat_test,
            batch_size_eval=max(256, batch_size),
        )
        test_metrics = {
            "test_total_loss": float(test_eval["total_loss"]),
            "test_mae": float(test_eval["mae"]),
            "test_mse": float(test_eval["mse"]),
            "test_cosine_similarity": float(test_eval["cosine_similarity"]),
            "test_cosine_error": float(test_eval["cosine_error"]),
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
            "final_train_total_loss": float(final_train_metrics["total_loss"]),
            "final_train_mae": float(final_train_metrics["mae"]),
            "final_train_mse": float(final_train_metrics["mse"]),
            "final_train_cosine_similarity": float(
                final_train_metrics["cosine_similarity"]
            ),
            "final_train_cosine_error": float(final_train_metrics["cosine_error"]),
            "final_val_total_loss": float(final_val_metrics["total_loss"]),
            "final_val_mae": float(final_val_metrics["mae"]),
            "final_val_mse": float(final_val_metrics["mse"]),
            "final_val_cosine_similarity": float(
                final_val_metrics["cosine_similarity"]
            ),
            "final_val_cosine_error": float(final_val_metrics["cosine_error"]),
            **test_metrics,
        },
        feature_dim=int(input_dim),
        target_feature_dim=int(target_dim),
        num_classes=int(len(label_values)),
        extra={
            "x_mean": np.asarray(x_std.mean, dtype=np.float32),
            "x_std": np.asarray(x_std.std, dtype=np.float32),
            "feature_shape": tuple(
                int(v)
                for v in corpus.codec_metadata.get("feature_shape", (target_dim,))
            ),
            "dtype": str(corpus.codec_metadata.get("dtype", "float32")),
            "label_values": tuple(int(v) for v in label_values),
            "attack_feature_map": dict(
                corpus.feature_metadata.get("attack_feature_map", {})
            ),
            "use_sigmoid_output": bool(use_sigmoid_output),
            "target_normalization": str(target_normalization),
            "output_normalization": str(output_normalization),
            "loss_name": str(loss_name),
            "used_validation_split": bool(used_val_split),
            "train_curve_history": list(train_curve_history),
            "val_curve_history": list(val_curve_history),
            "update_loss_history": list(update_loss_history),
        },
    )


def _project_to_ball(
    x: np.ndarray,
    *,
    ball_center: Optional[np.ndarray],
    ball_radius: Optional[float],
) -> np.ndarray:
    if ball_center is None or ball_radius is None:
        return x
    center = np.asarray(ball_center, dtype=np.float32).reshape(-1)
    flat = np.asarray(x, dtype=np.float32).reshape(-1)
    diff = flat - center
    norm = float(np.linalg.norm(diff))
    radius = float(ball_radius)
    if norm > radius and norm > 0.0:
        flat = center + (radius / norm) * diff
    return flat


def run_model_based_attack(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    reconstructor: ReconstructorArtifact,
    feature_map: Optional[AttackFeatureMap] = None,
    true_record: Optional[Record] = None,
    known_label: Optional[int] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
    ball_center: Optional[np.ndarray] = None,
    ball_radius: Optional[float] = None,
) -> AttackResult:
    """Run the informed-adversary final-model attack."""
    feature_map = ParametersOnlyFeatureMap() if feature_map is None else feature_map

    attack_features = np.asarray(
        feature_map(release, d_minus, None),
        dtype=np.float32,
    ).reshape(-1)

    x_mean = np.asarray(reconstructor.extra["x_mean"], dtype=np.float32)
    x_std = np.asarray(reconstructor.extra["x_std"], dtype=np.float32)
    x_norm = ((attack_features - x_mean) / x_std).astype(np.float32)

    recon_model = eqx.nn.inference_mode(reconstructor.model, value=True)
    feat_pred = np.asarray(
        recon_model(jnp.asarray(x_norm, dtype=jnp.float32)),
        dtype=np.float32,
    ).reshape(-1)

    output_normalization = str(reconstructor.extra.get("output_normalization", "none"))
    feat_pred = _normalize_rows_np(feat_pred, output_normalization).reshape(-1)

    feat_pred = _project_to_ball(
        feat_pred,
        ball_center=ball_center,
        ball_radius=ball_radius,
    )

    feature_shape = tuple(
        int(v) for v in reconstructor.extra.get("feature_shape", (feat_pred.size,))
    )
    pred_features = np.asarray(feat_pred, dtype=np.float32).reshape(feature_shape)

    metrics: dict[str, float] = {}
    if true_record is not None:
        metrics["feature_l2"] = float(
            np.linalg.norm(
                np.asarray(true_record.features, dtype=np.float32).reshape(-1)
                - np.asarray(pred_features, dtype=np.float32).reshape(-1),
                ord=2,
            )
        )
        if known_label is not None:
            pred_record = Record(
                features=np.asarray(pred_features), label=int(known_label)
            )
            metrics.update(
                reconstruction_metrics(true_record, pred_record, eta_grid=eta_grid)
            )

    return AttackResult(
        attack_family="model_based_informed_adversary",
        z_hat=np.asarray(pred_features),
        y_hat=None if known_label is None else int(known_label),
        status="ok_known_label" if known_label is not None else "ok",
        diagnostics={
            "attack_feature_dim": int(attack_features.size),
            "reconstructor_validation_metrics": dict(reconstructor.validation_metrics),
            "feature_map_metadata": dict(
                reconstructor.extra.get("attack_feature_map", {})
            ),
        },
        metrics=metrics,
    )
