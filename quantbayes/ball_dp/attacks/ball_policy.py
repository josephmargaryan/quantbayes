# quantbayes/ball_dp/attacks/ball_policy.py

from __future__ import annotations

import dataclasses as dc
from typing import Any, Dict, Optional, Sequence, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import numpy as np
from jax.scipy.special import logsumexp

from ..metrics import reconstruction_metrics
from ..nonconvex.per_example import ExampleLossFn, resolve_loss_fn
from ..types import ArrayDataset, AttackResult, Record
from .ball_priors import BallAttackPrior
from .gradient_based import (
    DPSGDTrace,
    _clip_single_grad,
    _single_example_grad,
    _tree_l2_sq,
    _tree_scalar_mul,
    _tree_sub,
    subtract_known_batch_gradients,
)


@dc.dataclass
class BallTraceMapAttackConfig:
    """Projected MAP attack against a residualized DP-SGD trace.

    mode='known_inclusion'
        exact posterior objective under the logged inclusion threat model

    mode='unknown_inclusion'
        exact posterior objective under the Bernoulli inclusion mixture model
    """

    mode: Literal["known_inclusion", "unknown_inclusion"] = "known_inclusion"
    optimizer: Literal["adam", "sgd"] = "adam"
    num_steps: int = 500
    learning_rate: float = 1e-2
    num_restarts: int = 5
    step_mode: Literal["all", "present_steps"] = "present_steps"
    sampling_probability: Optional[float] = None
    per_step_sampling_probabilities: Optional[tuple[float, ...]] = None
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


def _resolve_target_index(
    trace: DPSGDTrace,
    target_index: Optional[int],
) -> Optional[int]:
    if target_index is not None:
        return int(target_index)
    maybe = trace.metadata.get("target_index", None)
    return None if maybe is None else int(maybe)


def _step_contains_target(step: Any, target_index: int) -> bool:
    batch_idx = np.asarray(step.batch_indices, dtype=np.int64)
    if batch_idx.size == 0:
        raise ValueError(
            "Trace step is missing batch_indices; cannot determine target inclusion."
        )
    return bool(np.any(batch_idx == int(target_index)))


def _known_inclusion_selected_steps(
    trace: DPSGDTrace,
    *,
    target_index: int,
    step_mode: str,
) -> tuple[list[Any], str]:
    mode = str(step_mode).lower()
    if mode not in {"all", "present_steps"}:
        raise ValueError("step_mode must be one of {'all', 'present_steps'}.")

    if mode == "all":
        return list(trace.steps), "all"

    selected = []
    for step in trace.steps:
        if _step_contains_target(step, int(target_index)):
            selected.append(step)
    if not selected:
        raise ValueError("No retained trace step contains the target index.")
    return selected, "present_steps"


def _resolve_sampling_probabilities(
    trace: DPSGDTrace,
    *,
    selected_steps: Sequence[Any],
    cfg: BallTraceMapAttackConfig,
) -> list[float]:
    if cfg.per_step_sampling_probabilities is not None:
        qs = [float(v) for v in cfg.per_step_sampling_probabilities]
        if len(qs) != len(selected_steps):
            raise ValueError(
                "per_step_sampling_probabilities must match the selected step count."
            )
    elif cfg.sampling_probability is not None:
        qs = [float(cfg.sampling_probability) for _ in selected_steps]
    else:
        dataset_size = trace.metadata.get("dataset_size", None)
        sample_rate = trace.metadata.get("sample_rate", None)

        if dataset_size is not None:
            n_total = int(dataset_size)
            qs = []
            for step in selected_steps:
                batch_idx = np.asarray(step.batch_indices, dtype=np.int64)
                if batch_idx.size > 0:
                    qs.append(float(batch_idx.size) / float(n_total))
                elif sample_rate is not None:
                    qs.append(float(sample_rate))
                else:
                    raise ValueError(
                        "Could not infer per-step sampling probabilities from the trace."
                    )
        elif sample_rate is not None:
            qs = [float(sample_rate) for _ in selected_steps]
        else:
            raise ValueError(
                "Unknown-inclusion MAP requires sampling_probability=..., "
                "per_step_sampling_probabilities=..., or trace metadata containing "
                "dataset_size / sample_rate."
            )

    for q in qs:
        if not (0.0 <= float(q) <= 1.0):
            raise ValueError("Sampling probabilities must lie in [0, 1].")
    return qs


def _candidate_mean_tree(
    step: Any,
    trace: DPSGDTrace,
    *,
    loss_fn: ExampleLossFn,
    x_var: jnp.ndarray,
    label: int,
    seed: int,
) -> Any:
    if step.model_before is None:
        raise ValueError(
            "Ball-trace MAP requires model snapshots at all retained steps."
        )

    grad_key = jr.PRNGKey(int(seed + 10007 * int(step.step) + 131 * int(label)))
    cand = _single_example_grad(
        step.model_before,
        trace.state,
        loss_fn,
        x_var,
        jnp.asarray(label),
        grad_key,
    )
    cand = _clip_single_grad(cand, float(step.clip_norm))

    reduction = str(trace.reduction).lower()
    if reduction == "mean":
        batch_size = int(len(step.batch_indices))
        if batch_size <= 0:
            raise ValueError(
                "Trace step is missing batch_indices; cannot rescale mean-reduced gradients."
            )
        cand = _tree_scalar_mul(cand, 1.0 / float(batch_size))
    elif reduction != "sum":
        raise ValueError("trace.reduction must be one of {'mean', 'sum'}.")

    return cand


def _restart_points(
    prior: BallAttackPrior,
    *,
    num_restarts: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    wanted = max(1, int(num_restarts))
    out: list[np.ndarray] = [
        prior.project_np(np.asarray(prior.center, dtype=np.float32))
    ]
    if wanted > 1:
        samples = prior.sample(wanted - 1, rng)
        out.extend(
            np.asarray(samples[i], dtype=np.float32) for i in range(len(samples))
        )
    return out[:wanted]


def _resolve_label_space(
    *,
    known_label: Optional[int],
    label_space: Optional[Sequence[int]],
    dataset: Optional[ArrayDataset],
    true_record: Optional[Record],
) -> list[int]:
    if known_label is not None:
        return [int(known_label)]
    if label_space is not None:
        labels = [int(v) for v in label_space]
        if not labels:
            raise ValueError("label_space must be non-empty.")
        return labels
    if dataset is not None:
        labels = sorted(np.unique(np.asarray(dataset.y)).tolist())
        return [int(v) for v in labels]
    if true_record is not None:
        return [int(true_record.label)]
    raise ValueError(
        "Provide known_label, label_space, dataset, or true_record to determine the candidate label set."
    )


def run_ball_trace_map_attack(
    trace: DPSGDTrace,
    *,
    prior: BallAttackPrior,
    cfg: Optional[BallTraceMapAttackConfig] = None,
    dataset: Optional[ArrayDataset] = None,
    target_index: Optional[int] = None,
    loss_name: Optional[str] = None,
    loss_fn: Optional[ExampleLossFn] = None,
    known_label: Optional[int] = None,
    label_space: Optional[Sequence[int]] = None,
    true_record: Optional[Record] = None,
    eta_grid: Sequence[float] = (0.1, 0.2, 0.5, 1.0),
) -> AttackResult:
    """Ball-constrained MAP attack on a DP-SGD trace.

    Important:
      - If every retained step has strictly positive effective_noise_std, then the
        optimized objective is the exact negative log-posterior up to
        candidate-independent constants under the configured Gaussian trace model.
      - The exact objective is defined on a *residualized* trace, i.e. after subtracting
        all known non-target batch contributions.
      - If you pass dataset=... and target_index=..., this function performs that
        residualization automatically.
      - If you do not pass dataset=... and target_index=..., then trace.metadata must
        indicate that the trace is already residualized.
      - If a retained step has zero noise, the implementation reduces to
        squared-residual matching for that step rather than a hard-constraint
        noiseless posterior.
    """
    cfg = BallTraceMapAttackConfig() if cfg is None else cfg
    if not trace.steps:
        raise ValueError("Trace is empty.")

    resolved_loss_fn = (
        resolve_loss_fn(trace.loss_name if loss_name is None else loss_name)
        if loss_fn is None
        else loss_fn
    )

    target_index = _resolve_target_index(trace, target_index)
    trace_dataset_size = trace.metadata.get("dataset_size", None)
    if dataset is not None and trace_dataset_size is not None:
        if int(trace_dataset_size) != int(len(dataset)):
            raise ValueError(
                f"dataset size mismatch: trace metadata says {trace_dataset_size}, "
                f"but len(dataset)={len(dataset)}."
            )

    residualized = bool(trace.metadata.get("residualized_against_known_batch", False))
    if dataset is not None and target_index is not None and not residualized:
        work_trace = subtract_known_batch_gradients(
            trace,
            dataset,
            target_index=int(target_index),
            loss_name=trace.loss_name if loss_name is None else str(loss_name),
            loss_fn=resolved_loss_fn,
            seed=int(cfg.seed),
        )
    else:
        work_trace = trace

    if not bool(work_trace.metadata.get("residualized_against_known_batch", False)):
        raise ValueError(
            "Ball-trace MAP needs a residualized trace. "
            "Either pass dataset=... and target_index=... so the wrapper can subtract "
            "known non-target batch contributions, or call "
            "subtract_known_batch_gradients(...) first."
        )

    labels = _resolve_label_space(
        known_label=known_label,
        label_space=label_space,
        dataset=dataset,
        true_record=true_record,
    )

    mode = str(cfg.mode).lower()
    if mode not in {"known_inclusion", "unknown_inclusion"}:
        raise ValueError(
            "cfg.mode must be one of {'known_inclusion', 'unknown_inclusion'}."
        )

    if mode == "known_inclusion":
        if target_index is None:
            raise ValueError(
                "Known-inclusion MAP requires target_index to be known, either explicitly "
                "or via trace.metadata['target_index']."
            )
        selected_steps, effective_step_mode = _known_inclusion_selected_steps(
            work_trace,
            target_index=int(target_index),
            step_mode=str(cfg.step_mode),
        )
        q_by_step = None
    else:
        selected_steps = list(work_trace.steps)
        effective_step_mode = "all"
        q_by_step = _resolve_sampling_probabilities(
            work_trace,
            selected_steps=selected_steps,
            cfg=cfg,
        )
        for step in selected_steps:
            if float(step.effective_noise_std) <= 0.0:
                raise ValueError(
                    "Unknown-inclusion mixture MAP currently requires strictly positive "
                    "effective_noise_std at every retained step."
                )

    rng = np.random.default_rng(int(cfg.seed))
    optimizer = _make_optimizer(str(cfg.optimizer), float(cfg.learning_rate))

    per_label_best: Dict[int, float] = {}
    candidate_points: Dict[int, np.ndarray] = {}
    objective_by_label: Dict[int, Any] = {}

    best_obj = float("inf")
    best_x = None
    best_label = None

    for label in labels:
        label_local = int(label)

        def objective_fn(
            x_var: jnp.ndarray,
            *,
            _label: int = label_local,
        ) -> jnp.ndarray:
            total = prior.negative_log_density(x_var)

            for idx, step in enumerate(selected_steps):
                obs = step.observed_private_gradient
                cand = _candidate_mean_tree(
                    step,
                    work_trace,
                    loss_fn=resolved_loss_fn,
                    x_var=x_var,
                    label=_label,
                    seed=int(cfg.seed),
                )

                sigma = float(step.effective_noise_std)
                if mode == "known_inclusion":
                    if effective_step_mode == "present_steps":
                        include = True
                    else:
                        include = _step_contains_target(step, int(target_index))
                    if not include:
                        continue

                    sq = _tree_l2_sq(_tree_sub(obs, cand))
                    if sigma > 0.0:
                        total = total + sq / (2.0 * sigma * sigma)
                    else:
                        total = total + sq
                else:
                    q = float(q_by_step[idx])
                    sq0 = _tree_l2_sq(obs)
                    sq1 = _tree_l2_sq(_tree_sub(obs, cand))

                    if q <= 0.0:
                        total = total + sq0 / (2.0 * sigma * sigma)
                    elif q >= 1.0:
                        total = total + sq1 / (2.0 * sigma * sigma)
                    else:
                        q_j = jnp.asarray(q, dtype=jnp.float32)
                        a0 = jnp.log1p(-q_j) - sq0 / (2.0 * sigma * sigma)
                        a1 = jnp.log(q_j) - sq1 / (2.0 * sigma * sigma)
                        total = total - logsumexp(jnp.stack([a0, a1]))

            return total

        objective_by_label[label_local] = objective_fn

        label_best = float("inf")
        label_best_x = None

        value_and_grad = jax.value_and_grad(objective_fn)
        for x0_np in _restart_points(
            prior,
            num_restarts=int(cfg.num_restarts),
            rng=rng,
        ):
            x = jnp.asarray(x0_np, dtype=jnp.float32)
            opt_state = optimizer.init(x)

            current_obj = float(objective_fn(x))
            best_restart_obj = current_obj
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
            raise RuntimeError("Ball-trace MAP failed to produce a candidate.")

        per_label_best[label_local] = float(label_best)
        candidate_points[label_local] = np.asarray(label_best_x, dtype=np.float32)

        if float(label_best) < best_obj:
            best_obj = float(label_best)
            best_x = np.asarray(label_best_x, dtype=np.float32)
            best_label = label_local

    if best_x is None or best_label is None:
        raise RuntimeError("Ball-trace MAP failed to produce a final candidate.")

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
        objective_gap = float(best_obj - truth_obj)

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

    all_pos_noise = bool(
        all(float(step.effective_noise_std) > 0.0 for step in selected_steps)
    )

    diagnostics: Dict[str, Any] = {
        "mode": str(mode),
        "algorithm": "projected_map",
        "objective": float(best_obj),
        "per_label_best_objective": dict(per_label_best),
        "selected_step_count": int(len(selected_steps)),
        "selected_steps": [int(step.step) for step in selected_steps],
        "step_mode": str(effective_step_mode),
        "trace_residualized": bool(
            work_trace.metadata.get("residualized_against_known_batch", False)
        ),
        "logged_transcript_reduction": str(work_trace.reduction),
        "all_retained_steps_have_positive_noise": all_pos_noise,
        "exact_posterior_up_to_constants": all_pos_noise,
        "prior_metadata": dict(prior.metadata()),
        "objective_at_truth": truth_objective,
        "objective_gap_to_truth": objective_gap,
        "optimizer": str(cfg.optimizer),
        "num_steps": int(cfg.num_steps),
        "num_restarts": int(cfg.num_restarts),
        "target_index": None if target_index is None else int(target_index),
        "known_inclusion_skips_absent_step_constants": bool(
            mode == "known_inclusion" and effective_step_mode == "all"
        ),
    }
    if q_by_step is not None:
        diagnostics["sampling_probabilities"] = list(map(float, q_by_step))

    return AttackResult(
        attack_family=f"ball_trace_map_{mode}",
        z_hat=np.asarray(best_x, dtype=np.float32),
        y_hat=int(best_label),
        status="ok_known_label" if known_label is not None else "ok",
        diagnostics=diagnostics,
        metrics=metrics,
        candidates=candidates,
    )
