# quantbayes/ball_dp/convex/ball_output_attacks.py

from __future__ import annotations

import dataclasses as dc
from typing import Any, Callable, Dict, Optional, Sequence, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from ..config import (
    ConvexOptimizationConfig,
    ConvexReleaseConfig,
    GaussianCalibrationConfig,
    from_dict,
)
from ..metrics import reconstruction_metrics
from ..types import ArrayDataset, AttackResult, Record, ReleaseArtifact
from ..attacks.ball_priors import BallAttackPrior
from .releases import _solve_nonprivate_erm


@dc.dataclass
class BallOutputMapAttackConfig:
    """Ball-constrained MAP attack on Gaussian output perturbation.

    ridge_prototype:
        The continuous MAP objective is optimized by projected gradient descent.
        The finite-prior Bayes attack is analytic and vectorized over candidates.

    binary_logistic / softmax_logistic / squared_hinge:
        The objective is exact given the stored release solver snapshot, but each
        candidate/proposal requires solving a deterministic convex ERM. That path
        remains serial unless _solve_nonprivate_erm itself is rewritten as a
        batched JAX-pure solver.
    """

    optimizer: Literal["adam", "sgd"] = "adam"
    num_steps: int = 300
    learning_rate: float = 1e-2
    num_restarts: int = 6

    random_search_steps: int = 80
    proposals_per_step: int = 12
    initial_proposal_scale: float = 0.20
    proposal_decay: float = 0.97

    seed: int = 0


def _make_optimizer(
    name: str,
    learning_rate: float,
) -> optax.GradientTransformation:
    key = str(name).lower()
    if key == "adam":
        return optax.adam(float(learning_rate))
    if key == "sgd":
        return optax.sgd(float(learning_rate))
    raise ValueError("optimizer must be one of {'adam', 'sgd'}.")


def _release_sigma(release: ReleaseArtifact) -> float:
    sigma = release.privacy.ball.sigma
    if sigma is None or float(sigma) <= 0.0:
        raise ValueError(
            "Ball output MAP is defined here only for Gaussian output perturbation "
            "(positive sigma). For noiseless convex releases, use the existing "
            "equation-solving convex attack."
        )
    return float(sigma)


def _payload_vector(obj: Any) -> np.ndarray:
    """Flatten floating-point leaves from a release payload.

    This is intentionally generic for non-ridge convex heads. For ridge_prototype
    finite-prior scoring below, we avoid this helper and use payload.prototypes
    directly, because the analytic sufficient-statistic formula is safer.
    """
    leaves: list[np.ndarray] = []

    def visit(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            arr = np.asarray(x)
            if np.issubdtype(arr.dtype, np.floating):
                leaves.append(arr.astype(np.float32, copy=False).reshape(-1))
            return
        if dc.is_dataclass(x):
            for field in dc.fields(x):
                visit(getattr(x, field.name))
            return
        if isinstance(x, dict):
            for value in x.values():
                visit(value)
            return
        if isinstance(x, (list, tuple)):
            for value in x:
                visit(value)
            return
        if hasattr(x, "__dict__") and not isinstance(x, (str, bytes)):
            for value in vars(x).values():
                visit(value)
            return

    visit(obj)
    if not leaves:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(leaves, axis=0).astype(np.float32, copy=False)


def _resolve_label_space(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    known_label: Optional[int],
    label_space: Optional[Sequence[int]],
    true_record: Optional[Record],
) -> list[int]:
    if known_label is not None:
        return [int(known_label)]
    if label_space is not None:
        labels = [int(v) for v in label_space]
        if not labels:
            raise ValueError("label_space must be non-empty.")
        return labels

    meta_labels = release.dataset_metadata.get("label_values", None)
    if meta_labels is not None:
        labels = [int(v) for v in meta_labels]
        if labels:
            return labels

    if len(d_minus) > 0:
        return [int(v) for v in sorted(np.unique(np.asarray(d_minus.y)).tolist())]

    if true_record is not None:
        return [int(true_record.label)]

    raise ValueError(
        "Could not infer the candidate label set. Provide known_label=... or label_space=...."
    )


def _candidate_dataset(
    d_minus: ArrayDataset,
    x: np.ndarray,
    label: int,
) -> ArrayDataset:
    x_arr = np.asarray(x, dtype=np.asarray(d_minus.X).dtype).reshape(
        d_minus.feature_shape
    )
    y_dtype = np.asarray(d_minus.y).dtype
    X = np.concatenate([np.asarray(d_minus.X), x_arr[None, ...]], axis=0)
    y = np.concatenate(
        [np.asarray(d_minus.y), np.asarray([int(label)], dtype=y_dtype)],
        axis=0,
    )
    return ArrayDataset(X, y, name=f"{d_minus.name}_plus_candidate")


def _check_d_minus_size_matches_release(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
) -> int:
    n_total = int(release.dataset_metadata["n_total"])
    if int(len(d_minus) + 1) != n_total:
        raise ValueError(
            f"d_minus size mismatch: expected n_total-1 = {n_total - 1}, "
            f"got len(d_minus)={len(d_minus)}."
        )
    return n_total


def _ridge_minus_sufficient_statistics(
    d_minus: ArrayDataset,
    *,
    num_classes: int,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class counts and class sums for the known d_minus records."""
    X_minus = np.asarray(d_minus.X, dtype=np.float32)
    y_minus = np.asarray(d_minus.y, dtype=np.int64)

    if X_minus.ndim != 2:
        raise ValueError(
            "ridge_prototype expects d_minus.X to have shape (n_minus, dim)."
        )
    if X_minus.shape[1] != int(dim):
        raise ValueError(
            "ridge_prototype feature dimension mismatch: "
            f"d_minus.X.shape[1]={X_minus.shape[1]}, prototype dim={dim}."
        )
    if y_minus.shape[0] != X_minus.shape[0]:
        raise ValueError("d_minus.X and d_minus.y have inconsistent leading sizes.")

    if y_minus.size > 0:
        if np.any(y_minus < 0) or np.any(y_minus >= int(num_classes)):
            raise ValueError(
                f"d_minus contains labels outside [0, {int(num_classes) - 1}]."
            )

    counts = np.bincount(y_minus, minlength=int(num_classes))[: int(num_classes)]
    counts = counts.astype(np.int64, copy=False)

    sums = np.zeros((int(num_classes), int(dim)), dtype=np.float32)
    if y_minus.size > 0:
        np.add.at(sums, y_minus, X_minus)

    return counts, sums


def _ridge_mu_from_sufficient_statistics(
    counts: np.ndarray,
    sums: np.ndarray,
    *,
    lam: float,
    n_total: int,
) -> np.ndarray:
    counts_f = counts.astype(np.float32, copy=False)
    denom = 2.0 * counts_f[:, None] + float(lam) * float(n_total)
    return np.where(denom > 0.0, 2.0 * sums / denom, 0.0).astype(np.float32, copy=False)


def _initial_points(
    prior: BallAttackPrior,
    *,
    num_restarts: int,
    rng: np.random.Generator,
    warm_start: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    wanted = max(1, int(num_restarts))
    out: list[np.ndarray] = []

    def maybe_add(x: np.ndarray) -> None:
        x = np.asarray(
            prior.project_np(np.asarray(x, dtype=np.float32)), dtype=np.float32
        )
        if any(np.allclose(x, y, atol=1e-7, rtol=0.0) for y in out):
            return
        out.append(x)

    if warm_start is not None:
        maybe_add(np.asarray(warm_start, dtype=np.float32))
    maybe_add(np.asarray(prior.center, dtype=np.float32))

    need = max(0, wanted - len(out))
    if need > 0:
        samples = prior.sample(need, rng)
        for i in range(len(samples)):
            maybe_add(np.asarray(samples[i], dtype=np.float32))

    return out[:wanted]


def _convex_cfg_from_release(release: ReleaseArtifact) -> ConvexReleaseConfig:
    train_cfg = dict(release.training_config)
    if "gaussian_config" not in train_cfg or "optimization_config" not in train_cfg:
        raise ValueError(
            "release.training_config is missing gaussian_config / optimization_config. "
            "Patch convex/releases.py as instructed, regenerate the release artifact, and retry."
        )

    gauss_cfg = from_dict(GaussianCalibrationConfig, train_cfg["gaussian_config"])
    opt_cfg = from_dict(ConvexOptimizationConfig, train_cfg["optimization_config"])

    return ConvexReleaseConfig(
        model_family=str(release.model_family),
        radius=float(train_cfg["radius"]),
        lam=float(train_cfg["lam"]),
        gaussian=gauss_cfg,
        optimization=opt_cfg,
        epsilon=None,
        delta=None,
        sigma=float(release.privacy.ball.sigma),
        orders=tuple(
            float(v) for v in train_cfg.get("orders", (2, 3, 4, 5, 8, 16, 32, 64, 128))
        ),
        dp_deltas_for_rdp=tuple(),
        embedding_bound=(
            None
            if train_cfg.get("embedding_bound", None) is None
            else float(train_cfg["embedding_bound"])
        ),
        standard_radius=(
            None
            if train_cfg.get("standard_radius", None) is None
            else float(train_cfg["standard_radius"])
        ),
        num_classes=(
            None
            if train_cfg.get("num_classes", None) is None
            else int(train_cfg["num_classes"])
        ),
        lz_mode=str(train_cfg.get("lz_mode", "paper_default")),
        provided_lz=(
            None
            if train_cfg.get("provided_lz", None) is None
            else float(train_cfg["provided_lz"])
        ),
        use_exact_sensitivity_if_available=bool(
            train_cfg.get("use_exact_sensitivity_if_available", True)
        ),
        ridge_sensitivity_mode=str(train_cfg.get("ridge_sensitivity_mode", "global")),
        seed=int(train_cfg.get("seed", 0)),
        store_nonprivate_reference=False,
    )


def _assemble_attack_result(
    *,
    attack_family: str,
    status: str,
    best_label: int,
    best_x: np.ndarray,
    per_label_best: Dict[int, float],
    candidate_points: Dict[int, np.ndarray],
    objective_by_label: Dict[int, Callable[[Any], Any]],
    prior: BallAttackPrior,
    true_record: Optional[Record],
    eta_grid: Sequence[float],
    diagnostics_extra: Dict[str, Any],
) -> AttackResult:
    truth_objective = None
    objective_gap = None
    if true_record is not None and int(true_record.label) in objective_by_label:
        truth_x = np.asarray(true_record.features, dtype=np.float32).reshape(
            np.asarray(prior.center).shape
        )
        truth_obj = float(
            objective_by_label[int(true_record.label)](
                jnp.asarray(truth_x, dtype=jnp.float32)
            )
        )
        truth_objective = float(truth_obj)
        objective_gap = float(per_label_best[int(best_label)] - truth_obj)

    pred_record = Record(features=np.asarray(best_x), label=int(best_label))
    metrics = (
        {}
        if true_record is None
        else reconstruction_metrics(
            true_record,
            pred_record,
            eta_grid=tuple(float(v) for v in eta_grid),
        )
    )
    if objective_gap is not None:
        metrics["objective_gap_to_truth"] = float(objective_gap)

    sorted_candidates = sorted(per_label_best.items(), key=lambda kv: kv[1])
    candidates = [
        (int(lbl), np.asarray(candidate_points[int(lbl)], dtype=np.float32))
        for lbl, _ in sorted_candidates[: min(10, len(sorted_candidates))]
    ]

    diagnostics = {
        "objective": float(per_label_best[int(best_label)]),
        "per_label_best_objective": dict(per_label_best),
        "prior_metadata": dict(prior.metadata()),
        "objective_at_truth": truth_objective,
        "objective_gap_to_truth": objective_gap,
        **dict(diagnostics_extra),
    }

    return AttackResult(
        attack_family=str(attack_family),
        z_hat=np.asarray(best_x, dtype=np.float32),
        y_hat=int(best_label),
        status=str(status),
        diagnostics=diagnostics,
        metrics=metrics,
        candidates=candidates,
    )


def _run_ridge_output_map_attack(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    prior: BallAttackPrior,
    cfg: BallOutputMapAttackConfig,
    known_label: Optional[int],
    label_space: Optional[Sequence[int]],
    true_record: Optional[Record],
    eta_grid: Sequence[float],
) -> AttackResult:
    sigma = _release_sigma(release)
    lam = float(release.training_config["lam"])
    n_total = _check_d_minus_size_matches_release(release, d_minus)

    if not hasattr(release.payload, "prototypes"):
        raise TypeError(
            "ridge_prototype Ball output MAP expects release.payload.prototypes."
        )

    mus_rel = np.asarray(release.payload.prototypes, dtype=np.float32)
    if mus_rel.ndim != 2:
        raise ValueError("Expected prototype matrix with shape (num_classes, dim).")

    k, dim = mus_rel.shape
    labels = _resolve_label_space(
        release,
        d_minus,
        known_label=known_label,
        label_space=label_space,
        true_record=true_record,
    )

    counts_minus, S_minus = _ridge_minus_sufficient_statistics(
        d_minus,
        num_classes=k,
        dim=dim,
    )
    mu_minus = _ridge_mu_from_sufficient_statistics(
        counts_minus,
        S_minus,
        lam=lam,
        n_total=n_total,
    )

    baseline_sq = np.sum((mus_rel - mu_minus) ** 2, axis=1).astype(np.float32)

    mus_rel_j = jnp.asarray(mus_rel, dtype=jnp.float32)
    S_minus_j = jnp.asarray(S_minus, dtype=jnp.float32)
    baseline_sq_j = jnp.asarray(baseline_sq, dtype=jnp.float32)

    optimizer = _make_optimizer(str(cfg.optimizer), float(cfg.learning_rate))
    rng = np.random.default_rng(int(cfg.seed))

    per_label_best: Dict[int, float] = {}
    candidate_points: Dict[int, np.ndarray] = {}
    objective_by_label: Dict[int, Callable[[Any], Any]] = {}

    best_obj = float("inf")
    best_x = None
    best_label = None

    for label in labels:
        label = int(label)
        if label < 0 or label >= k:
            raise ValueError(
                f"label={label} is outside the ridge prototype index range [0, {k - 1}]."
            )

        const_other = float(
            (jnp.sum(baseline_sq_j) - baseline_sq_j[label]) / (2.0 * sigma * sigma)
        )
        alpha = float(
            2.0 * (float(counts_minus[label]) + 1.0) + float(lam) * float(n_total)
        )
        warm_start = 0.5 * alpha * mus_rel[label] - S_minus[label]

        def objective_fn(x_var: jnp.ndarray) -> jnp.ndarray:
            mu_y = 2.0 * (S_minus_j[label] + x_var) / alpha
            sq = jnp.sum((mus_rel_j[label] - mu_y) ** 2)
            return (
                jnp.asarray(const_other, dtype=jnp.float32)
                + sq / (2.0 * sigma * sigma)
                + prior.negative_log_density(x_var)
            )

        objective_by_label[label] = objective_fn

        label_best = float("inf")
        label_best_x = None

        value_and_grad = jax.value_and_grad(objective_fn)

        for x0_np in _initial_points(
            prior,
            num_restarts=int(cfg.num_restarts),
            rng=rng,
            warm_start=warm_start,
        ):
            x = jnp.asarray(x0_np, dtype=jnp.float32)
            opt_state = optimizer.init(x)

            best_restart_obj = float(objective_fn(x))
            best_restart_x = x

            for _ in range(max(1, int(cfg.num_steps))):
                _, grad_x = value_and_grad(x)
                updates, opt_state = optimizer.update(grad_x, opt_state, x)
                x = optax.apply_updates(x, updates)
                x = prior.project(x)

                curr = float(objective_fn(x))
                if curr < best_restart_obj:
                    best_restart_obj = curr
                    best_restart_x = x

            if best_restart_obj < label_best:
                label_best = best_restart_obj
                label_best_x = np.asarray(best_restart_x, dtype=np.float32)

        if label_best_x is None:
            raise RuntimeError("Ridge Ball output MAP failed to produce a candidate.")

        per_label_best[label] = float(label_best)
        candidate_points[label] = np.asarray(label_best_x, dtype=np.float32)

        if float(label_best) < best_obj:
            best_obj = float(label_best)
            best_x = np.asarray(label_best_x, dtype=np.float32)
            best_label = int(label)

    if best_x is None or best_label is None:
        raise RuntimeError("Ridge Ball output MAP failed to produce a final candidate.")

    return _assemble_attack_result(
        attack_family="ball_output_map_ridge_prototype",
        status="ok_known_label" if known_label is not None else "ok",
        best_label=int(best_label),
        best_x=np.asarray(best_x, dtype=np.float32),
        per_label_best=per_label_best,
        candidate_points=candidate_points,
        objective_by_label=objective_by_label,
        prior=prior,
        true_record=true_record,
        eta_grid=eta_grid,
        diagnostics_extra={
            "algorithm": "projected_gradient_descent",
            "exact_objective_given_gaussian_output_model": True,
            "optimizer": str(cfg.optimizer),
            "num_steps": int(cfg.num_steps),
            "num_restarts": int(cfg.num_restarts),
            "sigma": float(sigma),
        },
    )


def _projected_random_search(
    objective_np: Callable[[np.ndarray], float],
    *,
    prior: BallAttackPrior,
    cfg: BallOutputMapAttackConfig,
    rng: np.random.Generator,
    warm_start: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, int]:
    points = _initial_points(
        prior,
        num_restarts=int(cfg.num_restarts),
        rng=rng,
        warm_start=warm_start,
    )

    eval_count = 0
    best_global_obj = float("inf")
    best_global_x = None

    base_scale = float(cfg.initial_proposal_scale) * max(float(prior.radius), 1e-6)

    for x0 in points:
        x_curr = np.asarray(
            prior.project_np(np.asarray(x0, dtype=np.float32)), dtype=np.float32
        )
        obj_curr = float(objective_np(x_curr))
        eval_count += 1

        if obj_curr < best_global_obj:
            best_global_obj = obj_curr
            best_global_x = np.asarray(x_curr, dtype=np.float32)

        step_scale = float(base_scale)
        for _ in range(max(1, int(cfg.random_search_steps))):
            improved = False
            for _ in range(max(1, int(cfg.proposals_per_step))):
                delta = rng.normal(size=x_curr.shape).astype(np.float32)
                prop = np.asarray(
                    prior.project_np(x_curr + step_scale * delta),
                    dtype=np.float32,
                )
                prop_obj = float(objective_np(prop))
                eval_count += 1

                if prop_obj < obj_curr:
                    x_curr = prop
                    obj_curr = prop_obj
                    improved = True

                    if obj_curr < best_global_obj:
                        best_global_obj = obj_curr
                        best_global_x = np.asarray(x_curr, dtype=np.float32)

            if not improved:
                step_scale *= float(cfg.proposal_decay)

    if best_global_x is None:
        raise RuntimeError("Projected random search failed to produce a candidate.")

    return (
        np.asarray(best_global_x, dtype=np.float32),
        float(best_global_obj),
        int(eval_count),
    )


def _run_generic_output_map_attack(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    prior: BallAttackPrior,
    cfg: BallOutputMapAttackConfig,
    known_label: Optional[int],
    label_space: Optional[Sequence[int]],
    true_record: Optional[Record],
    eta_grid: Sequence[float],
) -> AttackResult:
    sigma = _release_sigma(release)
    _check_d_minus_size_matches_release(release, d_minus)

    release_cfg = _convex_cfg_from_release(release)
    noisy_vec = _payload_vector(release.payload)
    if noisy_vec.size == 0:
        raise ValueError(
            "Could not extract a floating-point parameter vector from release.payload."
        )

    labels = _resolve_label_space(
        release,
        d_minus,
        known_label=known_label,
        label_space=label_space,
        true_record=true_record,
    )

    rng = np.random.default_rng(int(cfg.seed))
    per_label_best: Dict[int, float] = {}
    candidate_points: Dict[int, np.ndarray] = {}
    objective_by_label: Dict[int, Callable[[Any], Any]] = {}
    evals_by_label: Dict[int, int] = {}

    best_obj = float("inf")
    best_x = None
    best_label = None

    for label in labels:
        label = int(label)

        def objective_np(x_np: np.ndarray) -> float:
            candidate_ds = _candidate_dataset(d_minus, x_np, label)
            candidate_payload, _, _ = _solve_nonprivate_erm(
                candidate_ds,
                release_cfg,
                jr.PRNGKey(int(release_cfg.seed)),
            )
            cand_vec = _payload_vector(candidate_payload)
            if cand_vec.shape != noisy_vec.shape:
                raise ValueError(
                    "Candidate parameter vector shape mismatch: "
                    f"release shape={noisy_vec.shape}, candidate shape={cand_vec.shape}."
                )
            diff = noisy_vec - cand_vec
            return float(
                0.5 * float(np.dot(diff, diff)) / (sigma * sigma)
                + float(
                    prior.negative_log_density_np(np.asarray(x_np, dtype=np.float32))
                )
            )

        def objective_jax(x_var: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(
                objective_np(np.asarray(x_var, dtype=np.float32)),
                dtype=jnp.float32,
            )

        objective_by_label[label] = objective_jax

        best_x_label, best_obj_label, evals = _projected_random_search(
            objective_np,
            prior=prior,
            cfg=cfg,
            rng=rng,
            warm_start=np.asarray(prior.center, dtype=np.float32),
        )
        per_label_best[label] = float(best_obj_label)
        candidate_points[label] = np.asarray(best_x_label, dtype=np.float32)
        evals_by_label[label] = int(evals)

        if float(best_obj_label) < best_obj:
            best_obj = float(best_obj_label)
            best_x = np.asarray(best_x_label, dtype=np.float32)
            best_label = int(label)

    if best_x is None or best_label is None:
        raise RuntimeError(
            "Generic Ball output MAP failed to produce a final candidate."
        )

    return _assemble_attack_result(
        attack_family="ball_output_map_convex_generic",
        status="ok_known_label" if known_label is not None else "ok",
        best_label=int(best_label),
        best_x=np.asarray(best_x, dtype=np.float32),
        per_label_best=per_label_best,
        candidate_points=candidate_points,
        objective_by_label=objective_by_label,
        prior=prior,
        true_record=true_record,
        eta_grid=eta_grid,
        diagnostics_extra={
            "algorithm": "projected_random_search",
            "exact_objective_given_release_solver_snapshot": True,
            "outer_search_is_approximate": True,
            "objective_evaluations_by_label": dict(evals_by_label),
            "random_search_steps": int(cfg.random_search_steps),
            "proposals_per_step": int(cfg.proposals_per_step),
            "initial_proposal_scale": float(cfg.initial_proposal_scale),
            "proposal_decay": float(cfg.proposal_decay),
            "sigma": float(sigma),
            "release_solver_snapshot": dc.asdict(release_cfg.optimization),
            "release_gaussian_snapshot": dc.asdict(release_cfg.gaussian),
        },
    )


def run_convex_ball_output_map_attack(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    prior: BallAttackPrior,
    cfg: Optional[BallOutputMapAttackConfig] = None,
    known_label: Optional[int] = None,
    label_space: Optional[Sequence[int]] = None,
    true_record: Optional[Record] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
) -> AttackResult:
    """Ball-constrained MAP attack for Gaussian output perturbation.

    ridge_prototype:
        exact Gaussian posterior objective + projected GD

    binary_logistic / softmax_logistic / squared_hinge:
        exact Gaussian likelihood given the stored deterministic solver snapshot,
        optimized by projected random search because each candidate requires solving
        a convex ERM.
    """
    fam = str(release.model_family)
    if fam not in {
        "ridge_prototype",
        "binary_logistic",
        "softmax_logistic",
        "squared_hinge",
    }:
        raise ValueError(
            "run_convex_ball_output_map_attack supports the convex families only."
        )

    if fam == "ridge_prototype":
        return _run_ridge_output_map_attack(
            release,
            d_minus,
            prior=prior,
            cfg=BallOutputMapAttackConfig() if cfg is None else cfg,
            known_label=known_label,
            label_space=label_space,
            true_record=true_record,
            eta_grid=eta_grid,
        )

    return _run_generic_output_map_attack(
        release,
        d_minus,
        prior=prior,
        cfg=BallOutputMapAttackConfig() if cfg is None else cfg,
        known_label=known_label,
        label_space=label_space,
        true_record=true_record,
        eta_grid=eta_grid,
    )


def _records_close_for_finite_prior(
    a: Record, b: Record, *, atol: float = 1e-7
) -> bool:
    xa = np.asarray(a.features, dtype=np.float32).reshape(-1)
    xb = np.asarray(b.features, dtype=np.float32).reshape(-1)
    return (
        int(a.label) == int(b.label)
        and xa.shape == xb.shape
        and np.allclose(xa, xb, atol=atol, rtol=0.0)
    )


def _normalize_finite_prior_records(
    prior_records: Sequence[Record],
    prior_weights: Optional[Sequence[float]],
    *,
    known_label: Optional[int],
) -> tuple[list[Record], np.ndarray]:
    if not prior_records:
        raise ValueError("prior_records must be non-empty.")

    if prior_weights is None:
        weights = [1.0 for _ in prior_records]
    else:
        if len(prior_weights) != len(prior_records):
            raise ValueError(
                "prior_weights must have the same length as prior_records."
            )
        weights = [float(v) for v in prior_weights]

    kept_records: list[Record] = []
    kept_weights: list[float] = []
    for rec, w in zip(prior_records, weights):
        if known_label is not None and int(rec.label) != int(known_label):
            continue
        kept_records.append(
            Record(
                features=np.asarray(rec.features, dtype=np.float32),
                label=int(rec.label),
            )
        )
        kept_weights.append(float(w))

    if not kept_records:
        raise ValueError(
            "Finite prior became empty after applying known_label filtering."
        )

    probs = np.asarray(kept_weights, dtype=np.float64)
    if np.any(~np.isfinite(probs)) or np.any(probs <= 0.0):
        raise ValueError("prior_weights must be finite and strictly positive.")
    probs = probs / float(np.sum(probs))
    return kept_records, probs.astype(np.float64, copy=False)


def _finite_prior_result_from_log_scores(
    *,
    filtered_records: Sequence[Record],
    probs: np.ndarray,
    log_scores_arr: np.ndarray,
    known_label: Optional[int],
    true_record: Optional[Record],
    eta_grid: Sequence[float],
    diagnostics_extra: Dict[str, Any],
) -> AttackResult:
    if len(filtered_records) == 0:
        raise ValueError("filtered_records must be non-empty.")

    log_scores_arr = np.asarray(log_scores_arr, dtype=np.float64)
    if log_scores_arr.shape != (len(filtered_records),):
        raise ValueError(
            "log_scores_arr must have shape (len(filtered_records),), got "
            f"{log_scores_arr.shape} for {len(filtered_records)} records."
        )

    best_idx = int(np.argmax(log_scores_arr))
    best_record = filtered_records[best_idx]

    shifted = log_scores_arr - float(np.max(log_scores_arr))
    posterior = np.exp(shifted)
    posterior = posterior / float(np.sum(posterior))

    posterior_entropy = float(
        -np.sum(posterior * np.log(np.maximum(posterior, 1e-300)))
    )
    posterior_effective_candidates = float(np.exp(posterior_entropy))

    sorted_scores = np.sort(log_scores_arr)[::-1]
    top2_gap = (
        float(sorted_scores[0] - sorted_scores[1])
        if sorted_scores.size >= 2
        else float("inf")
    )

    metrics = (
        {}
        if true_record is None
        else reconstruction_metrics(
            true_record,
            best_record,
            eta_grid=tuple(float(v) for v in eta_grid),
        )
    )
    metrics["oblivious_kappa"] = float(np.max(probs))
    metrics["posterior_top1_probability"] = float(posterior[best_idx])
    metrics["posterior_entropy"] = posterior_entropy
    metrics["posterior_effective_candidates"] = posterior_effective_candidates
    metrics["log_score_gap_top2"] = top2_gap

    true_prior_index = None
    if true_record is not None:
        true_prior_index = next(
            (
                i
                for i, rec in enumerate(filtered_records)
                if _records_close_for_finite_prior(rec, true_record)
            ),
            None,
        )
        feat_pred = np.asarray(best_record.features, dtype=np.float32).reshape(-1)
        feat_true = np.asarray(true_record.features, dtype=np.float32).reshape(-1)
        metrics["mse"] = float(np.mean((feat_pred - feat_true) ** 2))
        metrics["exact_identification_success"] = float(
            _records_close_for_finite_prior(best_record, true_record)
        )
        if true_prior_index is not None:
            order = np.argsort(-log_scores_arr)
            rank = int(np.where(order == true_prior_index)[0][0]) + 1
            metrics["prior_exact_hit"] = float(best_idx == true_prior_index)
            metrics["prior_rank"] = float(rank)
            metrics["posterior_true_probability"] = float(posterior[true_prior_index])
            metrics["log_score_gap_truth_to_top"] = float(
                log_scores_arr[true_prior_index] - log_scores_arr[best_idx]
            )
            for kk in (1, 5, 10):
                metrics[f"prior_hit@{kk}"] = float(rank <= kk)

    top_order = np.argsort(-log_scores_arr)[: min(10, len(filtered_records))]
    candidates = [
        (
            int(filtered_records[int(i)].label),
            np.asarray(filtered_records[int(i)].features, dtype=np.float32),
        )
        for i in top_order.tolist()
    ]

    diagnostics = {
        "prior_size": int(len(filtered_records)),
        "prior_weights": probs.astype(float).tolist(),
        "oblivious_kappa": float(np.max(probs)),
        "candidate_log_scores": log_scores_arr.astype(float).tolist(),
        "candidate_log_posteriors": np.log(np.maximum(posterior, 1e-300))
        .astype(float)
        .tolist(),
        "candidate_posterior_probs": posterior.astype(float).tolist(),
        "candidate_objectives": (-log_scores_arr).astype(float).tolist(),
        "posterior_entropy": posterior_entropy,
        "posterior_effective_candidates": posterior_effective_candidates,
        "log_score_gap_top2": top2_gap,
        "predicted_prior_index": int(best_idx),
        "true_prior_index": true_prior_index,
        "true_record_in_prior": bool(true_prior_index is not None),
        **dict(diagnostics_extra),
    }

    return AttackResult(
        attack_family="ball_output_finite_prior_exact_bayes",
        z_hat=np.asarray(best_record.features, dtype=np.float32),
        y_hat=int(best_record.label),
        status="ok_known_label" if known_label is not None else "ok",
        diagnostics=diagnostics,
        metrics=metrics,
        candidates=candidates,
    )


def _ridge_finite_prior_log_scores_vectorized(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    filtered_records: Sequence[Record],
    probs: np.ndarray,
    *,
    sigma: float,
    lam: float,
    n_total: int,
) -> np.ndarray:
    """Vectorized finite-prior log posterior scores for ridge_prototype.

    For each finite-prior candidate (x_i, y_i), this evaluates

        log p_i - ||theta_noisy - theta_nonprivate(d_minus U {(x_i,y_i)})||^2
                  / (2 sigma^2)

    using the closed-form ridge prototype sufficient statistics. No candidate ERM
    solve is performed.
    """
    if not hasattr(release.payload, "prototypes"):
        raise TypeError(
            "ridge_prototype finite-prior attack expects release.payload.prototypes."
        )

    mus_rel = np.asarray(release.payload.prototypes, dtype=np.float32)
    if mus_rel.ndim != 2:
        raise ValueError("Expected prototype matrix with shape (num_classes, dim).")

    k, dim = mus_rel.shape
    counts_minus, S_minus = _ridge_minus_sufficient_statistics(
        d_minus,
        num_classes=k,
        dim=dim,
    )
    mu_minus = _ridge_mu_from_sufficient_statistics(
        counts_minus,
        S_minus,
        lam=lam,
        n_total=n_total,
    )

    X = np.stack(
        [
            np.asarray(rec.features, dtype=np.float32).reshape(-1)
            for rec in filtered_records
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    y = np.asarray([int(rec.label) for rec in filtered_records], dtype=np.int64)

    if X.ndim != 2 or X.shape[1] != dim:
        raise ValueError(
            "Finite-prior candidate feature dimension mismatch for ridge_prototype: "
            f"candidate feature dim={X.shape[1] if X.ndim == 2 else None}, "
            f"prototype dim={dim}."
        )
    if np.any(y < 0) or np.any(y >= k):
        raise ValueError(
            f"Finite-prior candidate labels must lie in [0, {k - 1}] for ridge_prototype."
        )

    mus_rel_j = jnp.asarray(mus_rel, dtype=jnp.float32)
    mu_minus_j = jnp.asarray(mu_minus, dtype=jnp.float32)
    S_minus_j = jnp.asarray(S_minus, dtype=jnp.float32)
    counts_minus_j = jnp.asarray(counts_minus, dtype=jnp.float32)

    X_j = jnp.asarray(X, dtype=jnp.float32)
    y_j = jnp.asarray(y, dtype=jnp.int32)
    probs_j = jnp.asarray(probs, dtype=jnp.float32)

    baseline_sq_by_class = jnp.sum((mus_rel_j - mu_minus_j) ** 2, axis=1)
    baseline_total_sq = jnp.sum(baseline_sq_by_class)

    S_y = S_minus_j[y_j]
    counts_y = counts_minus_j[y_j]

    alpha_y = 2.0 * (counts_y + 1.0) + float(lam) * float(n_total)
    mu_y = 2.0 * (S_y + X_j) / alpha_y[:, None]

    row_sq = jnp.sum((mus_rel_j[y_j] - mu_y) ** 2, axis=1)
    total_sq = baseline_total_sq - baseline_sq_by_class[y_j] + row_sq

    log_scores = jnp.log(probs_j) - total_sq / (2.0 * float(sigma) * float(sigma))
    return np.asarray(log_scores, dtype=np.float64)


def _run_ridge_output_finite_prior_attack(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    prior_records: Sequence[Record],
    prior_weights: Optional[Sequence[float]],
    known_label: Optional[int],
    true_record: Optional[Record],
    eta_grid: Sequence[float],
) -> AttackResult:
    sigma = _release_sigma(release)
    n_total = _check_d_minus_size_matches_release(release, d_minus)
    lam = float(release.training_config["lam"])

    filtered_records, probs = _normalize_finite_prior_records(
        prior_records,
        prior_weights,
        known_label=known_label,
    )

    log_scores_arr = _ridge_finite_prior_log_scores_vectorized(
        release,
        d_minus,
        filtered_records,
        probs,
        sigma=sigma,
        lam=lam,
        n_total=n_total,
    )

    return _finite_prior_result_from_log_scores(
        filtered_records=filtered_records,
        probs=probs,
        log_scores_arr=log_scores_arr,
        known_label=known_label,
        true_record=true_record,
        eta_grid=eta_grid,
        diagnostics_extra={
            "algorithm": "finite_prior_exact_bayes_ridge_analytic_vectorized",
            "model_family": "ridge_prototype",
            "exact_posterior_over_finite_support": True,
            "exact_posterior_up_to_additive_constant": True,
            "finite_prior_scoring_is_vectorized": True,
            "candidate_axis_size": int(len(filtered_records)),
            "sigma": float(sigma),
            "lam": float(lam),
            "n_total": int(n_total),
        },
    )


def _run_generic_output_finite_prior_attack(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    prior_records: Sequence[Record],
    prior_weights: Optional[Sequence[float]],
    known_label: Optional[int],
    true_record: Optional[Record],
    eta_grid: Sequence[float],
) -> AttackResult:
    sigma = _release_sigma(release)
    _check_d_minus_size_matches_release(release, d_minus)

    release_cfg = _convex_cfg_from_release(release)
    noisy_vec = _payload_vector(release.payload)
    if noisy_vec.size == 0:
        raise ValueError(
            "Could not extract a floating-point parameter vector from release.payload."
        )

    filtered_records, probs = _normalize_finite_prior_records(
        prior_records,
        prior_weights,
        known_label=known_label,
    )

    log_scores: list[float] = []
    for rec, prob in zip(filtered_records, probs):
        candidate_ds = _candidate_dataset(
            d_minus,
            np.asarray(rec.features, dtype=np.float32),
            int(rec.label),
        )
        candidate_payload, _, _ = _solve_nonprivate_erm(
            candidate_ds,
            release_cfg,
            jr.PRNGKey(int(release_cfg.seed)),
        )
        cand_vec = _payload_vector(candidate_payload)
        if cand_vec.shape != noisy_vec.shape:
            raise ValueError(
                "Candidate parameter vector shape mismatch: "
                f"release shape={noisy_vec.shape}, candidate shape={cand_vec.shape}."
            )
        diff = noisy_vec - cand_vec
        log_score = float(
            np.log(prob) - 0.5 * float(np.dot(diff, diff)) / (sigma * sigma)
        )
        log_scores.append(log_score)

    log_scores_arr = np.asarray(log_scores, dtype=np.float64)

    return _finite_prior_result_from_log_scores(
        filtered_records=filtered_records,
        probs=probs,
        log_scores_arr=log_scores_arr,
        known_label=known_label,
        true_record=true_record,
        eta_grid=eta_grid,
        diagnostics_extra={
            "algorithm": "finite_prior_exact_bayes_serial_erm_solves",
            "model_family": str(release.model_family),
            "exact_posterior_over_finite_support": True,
            "exact_posterior_up_to_additive_constant": True,
            "exact_posterior_given_release_solver_snapshot": True,
            "finite_prior_scoring_is_vectorized": False,
            "candidate_axis_size": int(len(filtered_records)),
            "sigma": float(sigma),
            "release_solver_snapshot": dc.asdict(release_cfg.optimization),
            "release_gaussian_snapshot": dc.asdict(release_cfg.gaussian),
        },
    )


def run_convex_ball_output_finite_prior_attack(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    prior_records: Sequence[Record],
    prior_weights: Optional[Sequence[float]] = None,
    known_label: Optional[int] = None,
    true_record: Optional[Record] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
) -> AttackResult:
    """Finite-prior Bayes attack for Gaussian output perturbation.

    ridge_prototype:
        exact finite-prior posterior score, evaluated analytically and vectorized
        over all candidate records.

    binary_logistic / softmax_logistic / squared_hinge:
        exact finite-prior posterior score given the stored deterministic solver
        snapshot, evaluated by one ERM solve per candidate. This remains serial
        unless the convex ERM solver itself is rewritten as a batched JAX solver.
    """
    fam = str(release.model_family)
    if fam not in {
        "ridge_prototype",
        "binary_logistic",
        "softmax_logistic",
        "squared_hinge",
    }:
        raise ValueError(
            "run_convex_ball_output_finite_prior_attack supports the convex families only."
        )

    if fam == "ridge_prototype":
        return _run_ridge_output_finite_prior_attack(
            release,
            d_minus,
            prior_records=prior_records,
            prior_weights=prior_weights,
            known_label=known_label,
            true_record=true_record,
            eta_grid=eta_grid,
        )

    return _run_generic_output_finite_prior_attack(
        release,
        d_minus,
        prior_records=prior_records,
        prior_weights=prior_weights,
        known_label=known_label,
        true_record=true_record,
        eta_grid=eta_grid,
    )
