from __future__ import annotations

# Demo / example workflows for the public API.
#
# This file is intentionally written as "how to use the library in practice":
# - load a real dataset (MNIST / CIFAR-10) or pass your own arrays,
# - train convex or nonconvex releases,
# - run empirical reconstruction attacks,
# - compute Ball-ReRo certificates,
# - save paper-style visualizations.
#
# Notes
# -----
# The nonconvex demos are grouped by observation model:
#   1) final-model access   -> demo_nonconvex_model_based_attack(...)
#   2) retained trace access -> demo_nonconvex_trace_attacks(...)
#
# That is why there are two main nonconvex attack demos even though the attack
# package has multiple files.

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax

from quantbayes.ball_dp.api import (
    attack_convex,
    attack_nonconvex_gradient_baseline,
    attack_nonconvex_model_based,
    attack_nonconvex_prior_aware_trace,
    attack_nonconvex_trace_optimization,
    ball_rero,
    build_nonconvex_shadow_corpus,
    fit_ball_sgd,
    fit_convex,
    fit_shadow_reconstructor,
    get_operator_norm_history,
    get_public_curve_history,
    get_release_step_table,
    make_empirical_ball_prior,
    make_gradient_observation,
)
from quantbayes.ball_dp.attacks import (
    DPSGDTraceRecorder,
    FlatRecordCodec,
    TraceOptimizationAttackConfig,
    make_attack_feature_map,
    subtract_known_batch_gradients,
)
from quantbayes.ball_dp.config import ReconstructorTrainingConfig, ShadowCorpusConfig
from quantbayes.ball_dp.plots import (
    plot_attack_candidates,
    plot_attack_label_logits,
    plot_attack_result,
    plot_attack_result_with_reference,
    plot_attack_score_histogram,
    plot_convex_attack_result,
    plot_convex_model_parameters,
    plot_operator_norm_history,
    plot_release_curves,
    plot_rero_report,
)
from quantbayes.ball_dp.types import ArrayDataset, Record

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _flatten_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(len(X), -1)


def _feature_shape(X: np.ndarray) -> tuple[int, ...]:
    X = np.asarray(X)
    return tuple(int(v) for v in X.shape[1:])


def _box_bounds_from_data(X: np.ndarray) -> tuple[float, float]:
    X = np.asarray(X, dtype=np.float32)
    return float(np.min(X)), float(np.max(X))


def _same_label_records(
    X: np.ndarray,
    y: np.ndarray,
    *,
    label: int,
    max_count: int,
) -> list[Record]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    idx = np.where(y == int(label))[0][: int(max_count)]
    return [
        Record(features=np.asarray(X[i]).reshape(-1), label=int(y[i]))
        for i in idx.tolist()
    ]


def _make_aux_pool(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    label: int,
    max_count: int,
    exclude_train_index: int | None = None,
) -> list[Record]:
    out = _same_label_records(X_test, y_test, label=label, max_count=max_count)
    if len(out) >= max_count:
        return out[:max_count]

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)
    idx = np.where(y_train == int(label))[0]
    if exclude_train_index is not None:
        idx = idx[idx != int(exclude_train_index)]

    for i in idx.tolist():
        out.append(
            Record(features=np.asarray(X_train[i]).reshape(-1), label=int(y_train[i]))
        )
        if len(out) >= max_count:
            break
    return out[:max_count]


def _nearest_neighbor_record(
    true_record: Record,
    pool: list[Record],
) -> Record:
    if not pool:
        raise ValueError("Reference pool is empty.")
    x = np.asarray(true_record.features, dtype=np.float32).reshape(-1)
    dists = [
        float(np.linalg.norm(np.asarray(r.features, dtype=np.float32).reshape(-1) - x))
        for r in pool
    ]
    return pool[int(np.argmin(dists))]


def _trace_seen_indices(trace) -> list[int]:
    seen: set[int] = set()
    for step in trace.steps:
        batch_idx = np.asarray(step.batch_indices, dtype=np.int64)
        seen.update(int(i) for i in batch_idx.tolist())
    out = sorted(seen)
    if not out:
        raise ValueError("Trace contains no retained batch indices.")
    return out


def _choose_trace_target_index(
    trace,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    min_aux_pool: int = 16,
) -> int:
    """Choose an index that definitely appears in the retained trace."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    seen = _trace_seen_indices(trace)

    def pool_size(idx: int) -> int:
        label = int(y_train[int(idx)])
        return int(np.sum(y_test == label) + np.sum(y_train == label) - 1)

    viable = [idx for idx in seen if pool_size(idx) >= int(min_aux_pool)]
    if viable:
        return int(viable[0])
    return int(seen[0])


# -----------------------------------------------------------------------------
# Convex demos
# -----------------------------------------------------------------------------


def demo_convex_ridge_noiseless(X_train, y_train, X_test, y_test):
    """Noiseless ridge-prototype release, direct attack, and Ball-ReRo."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)

    B = float(np.max(np.linalg.norm(_flatten_X(X_train), axis=1)))
    r = float(2.0 * B)

    release = fit_convex(
        X_train,
        y_train,
        X_eval=X_test,
        y_eval=y_test,
        model_family="ridge_prototype",
        radius=r,
        lam=1e-2,
        embedding_bound=B,
        privacy="noiseless",
        solver="lbfgs_fullbatch",
        seed=0,
    )

    plot_convex_model_parameters(
        release,
        out_path="artifacts/convex_ridge_prototypes_noiseless.png",
    )

    target_index = 1
    attack, d_minus, true_record = attack_convex(
        release,
        X_train,
        y_train,
        target_index=target_index,
        eta_grid=(0.1, 0.2, 0.5, 1.0),
    )

    print("Convex release kind:", release.release_kind)
    print("Convex attack status:", attack.status)
    print("Convex attack metrics:", attack.metrics)

    plot_convex_attack_result(
        attack,
        true_record,
        out_path="artifacts/convex_attack_pair.png",
    )

    prior = make_empirical_ball_prior(
        _flatten_X(X_train),
        y_train,
        label=int(true_record.label),
        max_samples=2048,
    )
    report = ball_rero(
        release,
        prior=prior,
        eta_grid=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
        mode="auto",
    )
    plot_rero_report(report, out_path="artifacts/convex_rero.png")

    return release, attack, report


def demo_convex_ridge_ball_dp(X_train, y_train, X_test, y_test):
    """Private ridge-prototype release with the same API as the noiseless case."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test)

    B = float(np.max(np.linalg.norm(_flatten_X(X_train), axis=1)))
    r = float(2.0 * B)

    release = fit_convex(
        X_train,
        y_train,
        X_eval=X_test,
        y_eval=y_test,
        model_family="ridge_prototype",
        radius=r,
        lam=1e-2,
        embedding_bound=B,
        privacy="ball_dp",
        epsilon=8.0,
        delta=1e-6,
        solver="lbfgs_fullbatch",
        seed=0,
    )

    plot_convex_model_parameters(
        release,
        out_path="artifacts/convex_ridge_prototypes_ball_dp.png",
    )
    return release


# -----------------------------------------------------------------------------
# Nonconvex model
# -----------------------------------------------------------------------------


class ExampleMLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, *, key):
        keys = jr.split(key, 3)
        self.layers = (
            eqx.nn.Linear(input_dim, hidden_dim, key=keys[0]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            eqx.nn.Linear(hidden_dim, num_classes, key=keys[2]),
        )

    def __call__(self, x, *, key=None, state=None):
        h = jnp.asarray(x).reshape(-1)
        h = jax.nn.relu(self.layers[0](h))
        h = jax.nn.relu(self.layers[1](h))
        return self.layers[2](h), state


def build_example_model(input_dim: int, num_classes: int, *, seed: int):
    return ExampleMLP(input_dim, 128, num_classes, key=jr.PRNGKey(seed))


# -----------------------------------------------------------------------------
# Nonconvex training demo
# -----------------------------------------------------------------------------


def demo_nonconvex_training(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    privacy: str = "ball_rdp",
    num_steps: int = 500,
    batch_size: int = 128,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.1,
    seed: int = 0,
):
    """Train a Ball-SGD release and show public curves, operator norms, and ReRo."""
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    X_train_flat = _flatten_X(X_train)
    X_test_flat = _flatten_X(X_test)

    input_dim = int(X_train_flat.shape[1])
    num_classes = int(np.max(y_train)) + 1

    B = float(np.max(np.linalg.norm(X_train_flat, axis=1)))
    r = float(0.5 * B)
    L_z = 2.0  # Replace by your theorem-backed constant.

    model = build_example_model(input_dim, num_classes, seed=seed)
    optimizer = optax.adam(1e-3)

    release = fit_ball_sgd(
        model,
        optimizer,
        X_train_flat,
        y_train,
        X_eval=X_test_flat,
        y_eval=y_test,
        radius=r,
        lz=L_z,
        privacy=privacy,
        num_steps=num_steps,
        batch_size=batch_size,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        loss_name="softmax_cross_entropy",
        frobenius_reg_strength=1e-5,
        spectral_reg_strength=0.0,
        spectral_reg_kwargs={
            "apply_to": ("conv", "linear"),
            "conv_mode": "tn",
            "conv_tn_iters": 8,
        },
        record_operator_norms=True,
        operator_norms_every=max(25, num_steps // 10),
        operator_norm_kwargs={
            "conv_mode": "tn",
            "conv_tn_iters": 8,
        },
        checkpoint_selection="best_public_eval_accuracy",
        eval_every=max(25, num_steps // 10),
        seed=seed,
    )

    print("Nonconvex release kind:", release.release_kind)
    print("Utility metrics:", release.utility_metrics)
    print("Selected checkpoint step:", release.extra.get("selected_checkpoint_step"))
    print("Ball regime:", release.extra.get("ball_regime"))
    print(
        "Critical radius for min clip:",
        release.extra.get("critical_radius_for_min_clip"),
    )

    plot_release_curves(release, out_path="artifacts/nonconvex_public_curves.png")
    plot_operator_norm_history(
        release,
        out_path="artifacts/nonconvex_operator_norms.png",
    )

    public_history = get_public_curve_history(release)
    operator_history = get_operator_norm_history(release)
    step_table = get_release_step_table(release)

    steps = np.asarray([row["step"] for row in public_history], dtype=np.int64)
    public_eval_loss = np.asarray(
        [row.get("public_eval_loss", np.nan) for row in public_history],
        dtype=np.float64,
    )
    public_eval_acc = np.asarray(
        [row.get("public_eval_accuracy", np.nan) for row in public_history],
        dtype=np.float64,
    )

    plt.figure(figsize=(7, 4))
    plt.plot(steps, public_eval_loss, label="public_eval_loss")
    plt.plot(steps, public_eval_acc, label="public_eval_accuracy")
    plt.xlabel("step")
    plt.title("Custom public-eval plot")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/nonconvex_public_curves_custom.png", dpi=160)
    plt.close()

    if operator_history:
        op_steps = np.asarray([row["step"] for row in operator_history], dtype=np.int64)
        op_max = np.asarray(
            [row["summary"].get("max_sigma", np.nan) for row in operator_history],
            dtype=np.float64,
        )
        plt.figure(figsize=(7, 4))
        plt.plot(op_steps, op_max, label="max_sigma")
        plt.xlabel("step")
        plt.title("Custom operator-norm plot")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("artifacts/nonconvex_operator_norms_custom.png", dpi=160)
        plt.close()

    prior = make_empirical_ball_prior(
        X_train_flat,
        y_train,
        label=int(y_train[0]),
        max_samples=2048,
    )
    report = ball_rero(
        release,
        prior=prior,
        eta_grid=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
        mode="auto",
    )
    plot_rero_report(report, out_path="artifacts/nonconvex_training_rero.png")

    print("First three step rows:")
    for row in step_table[:3]:
        print(row)

    return release, public_history, operator_history, step_table, report


# -----------------------------------------------------------------------------
# Nonconvex model-based informed-adversary demo
# -----------------------------------------------------------------------------


def demo_nonconvex_model_based_attack(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    privacy: str = "ball_rdp",
    train_steps: int = 300,
    shadow_steps: int = 150,
    batch_size: int = 40,
    shadow_trials: int = 64,
    noise_multiplier: float = 1.1,
    target_index: int = 1,
    seed: int = 0,
):
    """End-to-end informed-adversary final-model attack demo."""
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    orig_feature_shape = _feature_shape(X_train)

    X_train_flat = _flatten_X(X_train)
    X_test_flat = _flatten_X(X_test)

    input_dim = int(X_train_flat.shape[1])
    num_classes = int(np.max(y_train)) + 1
    B = float(np.max(np.linalg.norm(X_train_flat, axis=1)))
    r = float(0.5 * B)
    L_z = 2.0

    model = build_example_model(input_dim, num_classes, seed=seed)
    optimizer = optax.adam(1e-3)

    release = fit_ball_sgd(
        model,
        optimizer,
        X_train_flat,
        y_train,
        X_eval=X_test_flat,
        y_eval=y_test,
        radius=r,
        lz=L_z,
        privacy=privacy,
        num_steps=train_steps,
        batch_size=batch_size,
        clip_norm=1.0,
        noise_multiplier=noise_multiplier,
        checkpoint_selection="best_public_eval_accuracy",
        eval_every=max(25, train_steps // 6),
        seed=seed,
    )

    ds = ArrayDataset(X_train_flat, y_train, name="train")
    d_minus, true_record = ds.remove_index(int(target_index))

    aux_points = _make_aux_pool(
        X_train_flat,
        y_train,
        X_test_flat,
        y_test,
        label=int(true_record.label),
        max_count=max(32, shadow_trials * 2),
        exclude_train_index=int(target_index),
    )
    if len(aux_points) < 8:
        raise ValueError("Need a reasonable auxiliary pool for the model-based demo.")

    def victim_train_fn(shadow_ds: ArrayDataset, seed_one: int):
        shadow_model = build_example_model(input_dim, num_classes, seed=seed_one)
        shadow_opt = optax.adam(1e-3)
        return fit_ball_sgd(
            shadow_model,
            shadow_opt,
            np.asarray(shadow_ds.X, dtype=np.float32),
            np.asarray(shadow_ds.y),
            X_eval=X_test_flat,
            y_eval=y_test,
            radius=r,
            lz=L_z,
            privacy=privacy,
            num_steps=shadow_steps,
            batch_size=batch_size,
            clip_norm=1.0,
            noise_multiplier=noise_multiplier,
            checkpoint_selection="best_public_eval_accuracy",
            eval_every=max(25, shadow_steps // 4),
            seed=seed_one,
        )

    feature_map = make_attack_feature_map(
        "parameters_plus_dataset_stats",
        random_projection_dim=1024,
        projection_seed=seed,
        label_values=tuple(int(v) for v in np.unique(y_train).tolist()),
    )

    corpus = build_nonconvex_shadow_corpus(
        d_minus=d_minus,
        shadow_targets=aux_points,
        victim_train_fn=victim_train_fn,
        feature_map=feature_map,
        cfg=ShadowCorpusConfig(
            num_trials=min(int(shadow_trials), len(aux_points)),
            train_frac=0.7,
            val_frac=0.15,
            side_info_regime="known_label",
            seed=seed,
        ),
        record_codec=FlatRecordCodec(feature_shape=orig_feature_shape),
    )

    reconstructor = fit_shadow_reconstructor(
        corpus,
        ReconstructorTrainingConfig(
            hidden_dims=(512, 512),
            batch_size=64,
            num_epochs=200,
            patience=20,
            learning_rate=1e-3,
            weight_decay=1e-5,
            seed=seed,
        ),
    )

    attack = attack_nonconvex_model_based(
        release,
        d_minus,
        reconstructor=reconstructor,
        feature_map=feature_map,
        true_record=true_record,
        known_label=int(true_record.label),
        eta_grid=(0.1, 0.2, 0.5, 1.0),
        box_bounds=_box_bounds_from_data(X_train_flat),
    )

    nn_ref = _nearest_neighbor_record(true_record, aux_points)

    print("Model-based nonconvex attack:", attack.status, attack.metrics)

    plot_attack_result_with_reference(
        attack,
        true_record,
        nn_ref.features,
        feature_shape=orig_feature_shape,
        reference_title="NN baseline",
        out_path="artifacts/nonconvex_model_based_triplet.png",
    )

    logits = np.asarray(attack.diagnostics.get("label_logits", []))
    if logits.size > 0 and logits.size <= 20 and float(np.std(logits)) > 1e-8:
        plot_attack_label_logits(
            attack,
            out_path="artifacts/nonconvex_model_based_label_logits.png",
        )

    prior = make_empirical_ball_prior(
        X_train_flat,
        y_train,
        label=int(true_record.label),
        max_samples=2048,
    )
    report = ball_rero(
        release,
        prior=prior,
        eta_grid=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
        mode="auto",
    )
    plot_rero_report(
        report,
        out_path="artifacts/nonconvex_model_based_rero.png",
    )

    return release, attack, true_record, report


# -----------------------------------------------------------------------------
# Nonconvex trace / gradient-access demo
# -----------------------------------------------------------------------------


def demo_nonconvex_trace_attacks(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    privacy: str = "ball_rdp",
    num_steps: int = 50,
    batch_size: int = 32,
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.1,
    target_index: int | None = None,
    seed: int = 0,
):
    """Gradient baseline, finite-prior ranking, and continuous trace optimization."""
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    orig_feature_shape = _feature_shape(X_train)

    X_train_flat = _flatten_X(X_train)
    X_test_flat = _flatten_X(X_test)

    input_dim = int(X_train_flat.shape[1])
    num_classes = int(np.max(y_train)) + 1
    B = float(np.max(np.linalg.norm(X_train_flat, axis=1)))
    r = float(0.5 * B)
    L_z = 2.0

    model = build_example_model(input_dim, num_classes, seed=seed)
    optimizer = optax.adam(1e-3)
    recorder = DPSGDTraceRecorder(
        capture_every=1,
        keep_models=True,
        keep_batch_indices=True,
    )

    release = fit_ball_sgd(
        model,
        optimizer,
        X_train_flat,
        y_train,
        X_eval=X_test_flat,
        y_eval=y_test,
        radius=r,
        lz=L_z,
        privacy=privacy,
        num_steps=num_steps,
        batch_size=batch_size,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        checkpoint_selection="last",
        eval_every=max(10, num_steps // 2),
        seed=seed,
        trace_recorder=recorder,
    )

    trace = recorder.to_trace(
        state=None,
        loss_name="softmax_cross_entropy",
        reduction="mean",
        metadata={
            "dataset_size": int(len(X_train_flat)),
            "sample_rate": float(batch_size) / float(len(X_train_flat)),
            "feature_shape": orig_feature_shape,
        },
    )

    seen = _trace_seen_indices(trace)
    if target_index is None:
        target_index = _choose_trace_target_index(
            trace,
            y_train,
            y_test,
            min_aux_pool=16,
        )
    else:
        target_index = int(target_index)
        if target_index not in seen:
            raise ValueError(
                f"Requested target_index={target_index} does not appear in any retained trace step. "
                "Increase num_steps / batch_size, reduce capture_every, or leave "
                "target_index=None to auto-select a valid target."
            )

    ds = ArrayDataset(X_train_flat, y_train, name="train")
    d_minus, true_record = ds.remove_index(target_index)

    print(
        f"Trace demo selected target_index={target_index}, "
        f"label={int(true_record.label)}"
    )

    residual_trace = subtract_known_batch_gradients(
        trace,
        ds,
        target_index=target_index,
    )

    present_steps = [
        s
        for s in residual_trace.steps
        if int(target_index)
        in set(np.asarray(s.batch_indices, dtype=np.int64).tolist())
    ]
    if not present_steps:
        raise RuntimeError(
            "Residualized trace contains no retained step with the target index."
        )

    step0 = present_steps[0]
    observation = make_gradient_observation(
        step0.model_before,
        step0.observed_private_gradient,
        state=None,
        batch_size=int(len(step0.batch_indices)),
        reduction="mean",
        metadata={"feature_shape": orig_feature_shape},
    )

    baseline = attack_nonconvex_gradient_baseline(
        observation,
        method="geiping",
        known_label=int(true_record.label),
        clip_norm=float(step0.clip_norm),
        box_bounds=_box_bounds_from_data(X_train_flat),
        true_record=true_record,
    )
    print("Gradient baseline:", baseline.status, baseline.metrics)
    plot_attack_result(
        baseline,
        true_record,
        feature_shape=orig_feature_shape,
        out_path="artifacts/nonconvex_gradient_baseline_pair.png",
    )

    prior_same_label = _make_aux_pool(
        X_train_flat,
        y_train,
        X_test_flat,
        y_test,
        label=int(true_record.label),
        max_count=256,
        exclude_train_index=target_index,
    )
    if len(prior_same_label) < 8:
        raise ValueError(
            "Need a reasonable same-label prior pool for the trace attacks."
        )

    nn_ref = _nearest_neighbor_record(true_record, prior_same_label)

    trace_rank = attack_nonconvex_prior_aware_trace(
        trace,
        prior_same_label,
        dataset=ds,
        target_index=target_index,
        known_label=int(true_record.label),
        true_record=true_record,
        score_mode="present_steps",
    )
    print("Trace ranking attack:", trace_rank.status, trace_rank.metrics)

    plot_attack_result_with_reference(
        trace_rank,
        true_record,
        nn_ref.features,
        feature_shape=orig_feature_shape,
        reference_title="NN baseline",
        out_path="artifacts/nonconvex_trace_rank_triplet.png",
    )
    plot_attack_candidates(
        trace_rank,
        feature_shape=orig_feature_shape,
        true_record=true_record,
        top_k=6,
        out_path="artifacts/nonconvex_trace_rank_candidates.png",
    )
    plot_attack_score_histogram(
        trace_rank,
        out_path="artifacts/nonconvex_trace_rank_scores.png",
    )

    trace_opt = attack_nonconvex_trace_optimization(
        trace,
        cfg=TraceOptimizationAttackConfig(
            objective="guo2023",
            step_mode="present_steps",
            num_steps=2000,
            learning_rate=1e-2,
            num_restarts=3,
            box_bounds=_box_bounds_from_data(X_train_flat),
            seed=seed,
        ),
        dataset=ds,
        target_index=target_index,
        known_label=int(true_record.label),
        true_record=true_record,
        feature_shape=orig_feature_shape,
    )
    print("Trace optimization attack:", trace_opt.status, trace_opt.metrics)

    plot_attack_result_with_reference(
        trace_opt,
        true_record,
        nn_ref.features,
        feature_shape=orig_feature_shape,
        reference_title="NN baseline",
        out_path="artifacts/nonconvex_trace_opt_triplet.png",
    )
    plot_attack_score_histogram(
        trace_opt,
        out_path="artifacts/nonconvex_trace_opt_objectives.png",
    )

    prior = make_empirical_ball_prior(
        X_train_flat,
        y_train,
        label=int(true_record.label),
        max_samples=2048,
    )
    report = ball_rero(
        release,
        prior=prior,
        eta_grid=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
        mode="auto",
    )
    plot_rero_report(
        report,
        out_path="artifacts/nonconvex_trace_rero.png",
    )

    return release, baseline, trace_rank, trace_opt, true_record, report
