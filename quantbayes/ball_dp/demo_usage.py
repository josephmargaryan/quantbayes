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
# The nonconvex attack demos below are the stable paper-faithful attacks:
#   - Balle/Cherubin/Hayes informed-adversary model-based attack (RecoNN style),
#   - Hayes/Mahloujifar/Balle prior-aware DP-SGD trace attack (Algorithm 2 / 3),
#   - Hayes/Mahloujifar/Balle DP-SGD trace optimization attack (Equation (1)).

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax

from quantbayes.ball_dp.api import (
    attack_convex,
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
    plot_attack_result,
    plot_attack_result_with_reference,
    plot_attack_score_histogram,
    plot_convex_attack_result,
    plot_convex_model_parameters,
    plot_operator_norm_history,
    plot_release_curves,
    plot_rero_report,
)
from quantbayes.ball_dp.attacks.spear import (
    SpearAttackConfig,
    run_spear_model_batch_attack,
    run_spear_trace_step_attack,
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


def _random_record_pool(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_count: int,
    exclude_train_index: int | None = None,
    exclude_train_indices: np.ndarray | list[int] | None = None,
    seed: int = 0,
) -> list[Record]:
    rng = np.random.default_rng(int(seed))
    train_ids = np.arange(len(X_train), dtype=np.int64)

    excluded: set[int] = set()
    if exclude_train_index is not None:
        excluded.add(int(exclude_train_index))
    if exclude_train_indices is not None:
        excluded.update(
            int(i) for i in np.asarray(exclude_train_indices, dtype=np.int64).tolist()
        )

    if excluded:
        keep_mask = ~np.isin(train_ids, np.asarray(sorted(excluded), dtype=np.int64))
        train_ids = train_ids[keep_mask]

    candidates = [("test", int(i)) for i in range(len(X_test))] + [
        ("train", int(i)) for i in train_ids.tolist()
    ]
    rng.shuffle(candidates)

    out: list[Record] = []
    for src, idx in candidates:
        if src == "test":
            out.append(
                Record(
                    features=np.asarray(X_test[idx], dtype=np.float32).reshape(-1),
                    label=int(y_test[idx]),
                )
            )
        else:
            out.append(
                Record(
                    features=np.asarray(X_train[idx], dtype=np.float32).reshape(-1),
                    label=int(y_train[idx]),
                )
            )
        if len(out) >= int(max_count):
            break
    return out


def _make_fixed_base_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_index: int,
    base_train_size: int,
    seed: int = 0,
) -> tuple[ArrayDataset, np.ndarray]:
    """Sample a fixed smaller D^- from the training set, excluding the target."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)

    if not (1 <= int(base_train_size) < len(X_train)):
        raise ValueError(
            f"base_train_size must be in [1, {len(X_train)-1}], got {base_train_size}."
        )

    rng = np.random.default_rng(int(seed))
    pool = np.arange(len(X_train), dtype=np.int64)
    pool = pool[pool != int(target_index)]
    chosen = rng.choice(pool, size=int(base_train_size), replace=False)

    base_ds = ArrayDataset(
        X_train[chosen],
        y_train[chosen],
        name=f"train_base_{int(base_train_size)}",
    )
    return base_ds, np.asarray(chosen, dtype=np.int64)


def _append_record_to_dataset(
    ds: ArrayDataset,
    record: Record,
    *,
    name: str | None = None,
) -> ArrayDataset:
    x = np.concatenate(
        [np.asarray(ds.X), np.asarray(record.features, dtype=np.float32)[None, ...]],
        axis=0,
    )
    y = np.concatenate(
        [
            np.asarray(ds.y),
            np.asarray([int(record.label)], dtype=np.asarray(ds.y).dtype),
        ],
        axis=0,
    )
    return ArrayDataset(x, y, name=(ds.name if name is None else name))


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
    *,
    seed: int = 0,
) -> int:
    seen = _trace_seen_indices(trace)
    rng = np.random.default_rng(int(seed))
    return int(rng.choice(np.asarray(seen, dtype=np.int64)))


def _paper_prior_with_truth(
    true_record: Record,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    size: int,
    exclude_train_index: int,
    seed: int = 0,
) -> list[Record]:
    if int(size) < 2:
        raise ValueError("Paper prior size must be at least 2.")
    others = _random_record_pool(
        X_train,
        y_train,
        X_test,
        y_test,
        max_count=int(size) - 1,
        exclude_train_index=exclude_train_index,
        seed=seed,
    )
    return [true_record] + others


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
    clip_norm: float = 1.0,
    base_train_size: int | None = None,  # NEW
    target_index: int = 1,
    seed: int = 0,
):
    """End-to-end informed-adversary final-model attack demo.

    If base_train_size is not None, the victim and every shadow are trained on
    the same fixed smaller base set D^- plus one target/example record.
    """
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

    full_ds = ArrayDataset(X_train_flat, y_train, name="train_full")
    true_record = full_ds.record(int(target_index))

    # Choose the attacker-known fixed base set D^-.
    if base_train_size is None:
        d_minus, _ = full_ds.remove_index(int(target_index))
        excluded_train_indices = np.asarray([int(target_index)], dtype=np.int64)
    else:
        d_minus, base_indices = _make_fixed_base_dataset(
            X_train_flat,
            y_train,
            target_index=int(target_index),
            base_train_size=int(base_train_size),
            seed=seed,
        )
        excluded_train_indices = np.concatenate(
            [base_indices, np.asarray([int(target_index)], dtype=np.int64)],
            axis=0,
        )

    victim_ds = _append_record_to_dataset(
        d_minus,
        true_record,
        name="victim_train",
    )

    # Victim release.
    victim_model = build_example_model(input_dim, num_classes, seed=seed)
    victim_opt = optax.adam(1e-3)

    victim_batch_size = min(int(batch_size), len(victim_ds))
    release = fit_ball_sgd(
        victim_model,
        victim_opt,
        np.asarray(victim_ds.X, dtype=np.float32),
        np.asarray(victim_ds.y),
        X_eval=X_test_flat,
        y_eval=y_test,
        radius=r,
        lz=L_z,
        privacy=privacy,
        num_steps=train_steps,
        batch_size=victim_batch_size,
        clip_norm=float(clip_norm),
        noise_multiplier=noise_multiplier,
        checkpoint_selection="best_public_eval_accuracy",
        eval_every=max(25, train_steps // 6),
        seed=seed,
    )

    # Auxiliary points available to the informed adversary.
    aux_points = _random_record_pool(
        X_train_flat,
        y_train,
        X_test_flat,
        y_test,
        max_count=max(32, shadow_trials * 2),
        exclude_train_indices=excluded_train_indices,
        seed=seed,
    )
    if len(aux_points) < 8:
        raise ValueError("Need a reasonable auxiliary pool for the model-based demo.")

    def victim_train_fn(shadow_ds: ArrayDataset, fixed_init_seed: int):
        # All shadows share the same initialization seed as the victim.
        shadow_model = build_example_model(
            input_dim,
            num_classes,
            seed=fixed_init_seed,
        )
        shadow_opt = optax.adam(1e-3)

        shadow_batch_size = min(int(batch_size), len(shadow_ds))
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
            batch_size=shadow_batch_size,
            clip_norm=float(clip_norm),
            noise_multiplier=noise_multiplier,
            checkpoint_selection="best_public_eval_accuracy",
            eval_every=max(25, shadow_steps // 4),
            seed=fixed_init_seed,
        )

    feature_map = make_attack_feature_map("parameters_only")

    corpus = build_nonconvex_shadow_corpus(
        d_minus=d_minus,
        shadow_targets=aux_points,
        victim_train_fn=victim_train_fn,
        feature_map=feature_map,
        cfg=ShadowCorpusConfig(
            num_trials=min(int(shadow_trials), len(aux_points)),
            train_frac=0.7,
            val_frac=0.15,
            seed=seed,
        ),
        record_codec=FlatRecordCodec(
            feature_shape=orig_feature_shape,
            box_bounds=_box_bounds_from_data(X_train),
        ),
        seed_policy="fixed",
        fixed_seed=seed,
    )

    reconstructor = fit_shadow_reconstructor(
        corpus,
        ReconstructorTrainingConfig(
            hidden_dims=(1000, 1000),
            batch_size=32,
            num_epochs=200,
            patience=25,
            learning_rate=3e-4,
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
        box_bounds=_box_bounds_from_data(X_train),
    )

    nn_ref = _nearest_neighbor_record(true_record, aux_points)

    print("Model-based nonconvex attack:", attack.status, attack.metrics)
    print(
        "Victim/shadow base size:",
        len(d_minus),
        "| Victim train size:",
        len(victim_ds),
        "| Victim batch size used:",
        victim_batch_size,
    )

    plot_attack_result_with_reference(
        attack,
        true_record,
        nn_ref.features,
        feature_shape=orig_feature_shape,
        reference_title="NN baseline",
        out_path="artifacts/nonconvex_model_based_triplet.png",
    )

    return release, attack, true_record


# -----------------------------------------------------------------------------
# Nonconvex DP-SGD trace-access demo
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
    paper_prior_size: int = 10,
    seed: int = 0,
):
    """Prior-aware ranking and Equation (1) trace optimization."""
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
        target_index = _choose_trace_target_index(trace, seed=seed)
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
        seed=seed,
    )

    paper_prior = _paper_prior_with_truth(
        true_record,
        X_train_flat,
        y_train,
        X_test_flat,
        y_test,
        size=paper_prior_size,
        exclude_train_index=target_index,
        seed=seed,
    )
    nn_ref = _nearest_neighbor_record(true_record, paper_prior[1:])

    trace_rank = attack_nonconvex_prior_aware_trace(
        trace,
        paper_prior,
        dataset=ds,
        target_index=target_index,
        true_record=true_record,
        algorithm="auto",
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
        top_k=min(6, len(paper_prior)),
        out_path="artifacts/nonconvex_trace_rank_candidates.png",
    )
    plot_attack_score_histogram(
        trace_rank,
        out_path="artifacts/nonconvex_trace_rank_scores.png",
    )

    trace_opt = attack_nonconvex_trace_optimization(
        residual_trace,
        cfg=TraceOptimizationAttackConfig(
            step_mode="all",
            num_steps=2000,  # smaller than the paper for a fast demo
            learning_rate=1e-2,  # paper default
            num_restarts=2,  # smaller than the paper for a fast demo
            box_bounds=_box_bounds_from_data(X_train),
            seed=seed,
        ),
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

    return release, trace_rank, trace_opt, true_record, report


############### Extra demos for the SPEAR attacks ###############


def _plot_spear_batch_grid(
    true_batch: np.ndarray,
    recon_batch: np.ndarray,
    *,
    feature_shape: tuple[int, ...],
    title: str,
    out_path: str,
    max_items: int = 8,
) -> None:
    true_batch = np.asarray(true_batch, dtype=np.float32)
    recon_batch = np.asarray(recon_batch, dtype=np.float32)
    if true_batch.shape != recon_batch.shape:
        raise ValueError(
            f"true_batch.shape={true_batch.shape} must match recon_batch.shape={recon_batch.shape}."
        )

    n_show = min(int(max_items), int(true_batch.shape[0]))
    ncols = n_show
    fig, axes = plt.subplots(2, ncols, figsize=(1.9 * ncols, 4.0))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[:, None]

    for j in range(n_show):
        true_img = np.asarray(true_batch[j]).reshape(feature_shape)
        recon_img = np.asarray(recon_batch[j]).reshape(feature_shape)

        ax_true = axes[0, j]
        ax_rec = axes[1, j]

        if true_img.ndim == 2:
            ax_true.imshow(true_img, cmap="gray")
            ax_rec.imshow(recon_img, cmap="gray")
        elif true_img.ndim == 3 and true_img.shape[0] in {1, 3}:
            ax_true.imshow(np.transpose(true_img, (1, 2, 0)))
            ax_rec.imshow(np.transpose(recon_img, (1, 2, 0)))
        else:
            ax_true.imshow(true_img)
            ax_rec.imshow(recon_img)

        ax_true.set_title(f"true[{j}]")
        ax_rec.set_title(f"recon[{j}]")
        ax_true.axis("off")
        ax_rec.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def demo_nonconvex_spear_exact_batch_attack(
    X_train,
    y_train,
    *,
    batch_size: int = 8,
    target_index=None,
    layer_path: tuple[object, ...] = ("layers", 0),
    max_samples: int = 20_000,
    false_rejection_rate: float = 1e-5,
    zero_tol: float = 1e-7,
    seed: int = 0,
):
    """Paper-faithful SPEAR demo on exact raw gradients of the first linear+ReLU block.

    This does *not* use DP-SGD. It computes the exact batch gradient on the chosen batch,
    exactly as required by the SPEAR paper.

    If target_index is provided, that record is forced into the attacked batch and placed
    first in the returned batch order.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    orig_feature_shape = _feature_shape(X_train)
    X_train_flat = _flatten_X(X_train).astype(np.float64)

    if int(batch_size) < 2:
        raise ValueError("SPEAR demo requires batch_size >= 2.")
    if int(batch_size) > len(X_train_flat):
        raise ValueError(
            f"Requested batch_size={batch_size} exceeds dataset size={len(X_train_flat)}."
        )

    input_dim = int(X_train_flat.shape[1])
    num_classes = int(np.max(y_train)) + 1
    model = build_example_model(input_dim, num_classes, seed=seed)

    rng = np.random.default_rng(int(seed))
    if target_index is None:
        batch_indices = rng.choice(
            len(X_train_flat), size=int(batch_size), replace=False
        )
    else:
        target_index = int(target_index)
        if target_index < 0 or target_index >= len(X_train_flat):
            raise IndexError(
                f"target_index={target_index} is out of range for dataset of size {len(X_train_flat)}."
            )
        pool = np.arange(len(X_train_flat), dtype=np.int64)
        pool = pool[pool != target_index]
        others = rng.choice(pool, size=int(batch_size) - 1, replace=False)
        batch_indices = np.concatenate(
            [
                np.asarray([target_index], dtype=np.int64),
                np.asarray(others, dtype=np.int64),
            ]
        )

    xb = np.asarray(X_train_flat[batch_indices], dtype=np.float64)
    yb = np.asarray(y_train[batch_indices])

    attack = run_spear_model_batch_attack(
        model,
        xb,
        yb,
        layer_path=layer_path,
        loss_name="softmax_cross_entropy",
        reduction="mean",
        cfg=SpearAttackConfig(
            max_samples=int(max_samples),
            batch_size=None,  # exact paper mode: infer b as rank(dL/dW)
            false_rejection_rate=float(false_rejection_rate),
            zero_tol=float(zero_tol),
            random_seed=int(seed),
            greedy_swap_rule="best_improvement",
            noisy_mode=False,
        ),
        true_batch=xb,
        eta_grid=(0.1, 0.2, 0.5, 1.0),
        seed=seed,
    )

    print("SPEAR exact batch status:", attack.status)
    print("SPEAR exact batch metrics:", attack.metrics)
    print(
        "SPEAR exact diagnostics:",
        {
            k: attack.diagnostics.get(k)
            for k in [
                "batch_size",
                "inferred_rank",
                "requested_batch_size",
                "tau_zero_count_threshold",
                "candidate_pool_size",
                "candidate_pool_rank",
                "max_candidate_pool_rank",
                "n_rank_b_minus_1_submatrices",
                "n_sparse_enough",
                "n_duplicates",
                "best_gamma",
                "failure_reason",
                "true_batch_shape_mismatch",
                "metrics_skipped_reason",
            ]
        },
    )

    recon = attack.x_hat_aligned if attack.x_hat_aligned is not None else attack.x_hat
    if recon is None:
        print("SPEAR exact attack did not return a reconstruction; skipping plot.")
        return attack, np.asarray(batch_indices, dtype=np.int64)

    recon = np.asarray(recon, dtype=np.float64)
    if recon.shape != xb.shape:
        print(
            "SPEAR returned a reconstruction with shape different from the attacked batch; "
            "skipping plot.",
            {"recovered_shape": tuple(recon.shape), "true_shape": tuple(xb.shape)},
        )
        return attack, np.asarray(batch_indices, dtype=np.int64)

    _plot_spear_batch_grid(
        xb,
        recon,
        feature_shape=orig_feature_shape,
        title=f"SPEAR exact batch attack ({attack.status})",
        out_path="artifacts/spear_exact_batch.png",
        max_items=min(8, int(batch_size)),
    )

    return attack, np.asarray(batch_indices, dtype=np.int64)


def demo_nonconvex_spear_private_step_comparison(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    privacy: str = "ball_rdp",
    epsilon: float | None = None,
    delta: float | None = None,
    batch_size: int = 8,
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.5,
    layer_path: tuple[object, ...] = ("layers", 0),
    max_samples_exact: int = 20_000,
    max_samples_private: int = 20_000,
    zero_tol_exact: float = 1e-10,
    zero_tol_private: float = 1e-4,
    seed: int = 0,
):
    """Compare exact SPEAR to SPEAR run on one privatized Ball-SGD trace step.

    Workflow:
      1. Train for exactly one Ball-SGD step while recording the sanitized gradient transcript.
      2. Reconstruct the *same* minibatch exactly from the raw gradient of `model_before`.
      3. Reconstruct the batch again from the privatized observed gradient stored in the trace.
      4. Compute the Ball-ReRo certificate on the resulting private release.

    Important:
      - `exact_attack` is the paper-faithful SPEAR attack.
      - `private_attack` is the paper-inspired noisy-gradient adaptation.
    """
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
    L_z = 2.0  # replace by your theorem-backed constant for the chosen architecture

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
        epsilon=epsilon,
        delta=delta,
        num_steps=1,
        batch_size=batch_size,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        loss_name="softmax_cross_entropy",
        checkpoint_selection="last",
        eval_every=1,
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
    if not trace.steps:
        raise RuntimeError("Trace recorder did not capture any step.")
    step = trace.steps[0]

    batch_indices = np.asarray(step.batch_indices, dtype=np.int64)
    xb = np.asarray(X_train_flat[batch_indices], dtype=np.float32)
    yb = np.asarray(y_train[batch_indices])

    exact_attack = run_spear_model_batch_attack(
        step.model_before,
        xb,
        yb,
        layer_path=layer_path,
        loss_name="softmax_cross_entropy",
        reduction="mean",
        cfg=SpearAttackConfig(
            max_samples=int(max_samples_exact),
            batch_size=None,
            false_rejection_rate=1e-5,
            zero_tol=float(zero_tol_exact),
            random_seed=int(seed),
            greedy_swap_rule="best_improvement",
            noisy_mode=False,
        ),
        true_batch=xb,
        eta_grid=(0.1, 0.2, 0.5, 1.0),
        seed=seed,
    )

    private_attack = run_spear_trace_step_attack(
        step,
        layer_path=layer_path,
        cfg=SpearAttackConfig(
            max_samples=int(max_samples_private),
            batch_size=int(batch_size),
            false_rejection_rate=1e-5,
            zero_tol=float(zero_tol_private),
            random_seed=int(seed),
            greedy_swap_rule="best_improvement",
            noisy_mode=True,
            noisy_gamma_target=0.98,
            noisy_submatrix_rows=int(batch_size + 1),
        ),
        true_batch=xb,
        eta_grid=(0.1, 0.2, 0.5, 1.0),
    )

    print("SPEAR exact status:", exact_attack.status)
    print("SPEAR exact metrics:", exact_attack.metrics)
    print("SPEAR private status:", private_attack.status)
    print("SPEAR private metrics:", private_attack.metrics)
    print("Private release kind:", release.release_kind)
    print("Primary privacy view:", release.attack_metadata.get("primary_privacy_view"))

    exact_recon = (
        exact_attack.x_hat_aligned
        if exact_attack.x_hat_aligned is not None
        else exact_attack.x_hat
    )
    private_recon = (
        private_attack.x_hat_aligned
        if private_attack.x_hat_aligned is not None
        else private_attack.x_hat
    )
    if exact_recon is None or private_recon is None:
        raise RuntimeError("SPEAR demo attack failed to return a reconstruction.")

    _plot_spear_batch_grid(
        xb,
        exact_recon,
        feature_shape=orig_feature_shape,
        title=f"SPEAR exact baseline ({exact_attack.status})",
        out_path="artifacts/spear_exact_baseline.png",
        max_items=min(8, int(batch_size)),
    )
    _plot_spear_batch_grid(
        xb,
        private_recon,
        feature_shape=orig_feature_shape,
        title=f"SPEAR on privatized step ({private_attack.status})",
        out_path="artifacts/spear_private_step.png",
        max_items=min(8, int(batch_size)),
    )

    prior = make_empirical_ball_prior(
        X_train_flat,
        max_samples=2048,
    )
    report = ball_rero(
        release,
        prior=prior,
        eta_grid=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
        mode="auto",
    )
    plot_rero_report(report, out_path="artifacts/spear_private_rero.png")

    return release, exact_attack, private_attack, report
