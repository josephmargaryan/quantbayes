# quantbayes/ball_dp/convex/releases.py

from __future__ import annotations

from typing import Any, Dict, Optional
import dataclasses as dc

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..accountants.convex_output import build_convex_gaussian_ledgers
from ..accountants.gaussian import gaussian_sigma
from ..config import ConvexReleaseConfig
from ..metrics import accuracy_from_logits
from ..types import ArrayDataset, ReleaseArtifact, SensitivityMetadata
from .models.binary_logistic import (
    BinaryLogisticModel,
    binary_logistic_loss,
    encode_binary_pm1,
    binary_logits,
)
from .models.ridge_prototype import (
    PrototypeRelease,
    fit_ridge_prototypes,
    prototype_count_aware_ball_sensitivity,
    prototype_exact_ball_sensitivity,
    prototype_predict,
)
from .models.softmax_logistic import (
    SoftmaxLinearModel,
    softmax_loss,
    softmax_logits,
)
from .models.squared_hinge import (
    SquaredHingeModel,
    squared_hinge_loss,
    squared_hinge_scores,
)
from .sensitivity import (
    approximate_solution_sensitivity,
    lz_binary_logistic_bound,
    lz_prototypes_exact,
    lz_softmax_linear_bound,
    lz_squared_hinge_bound,
    output_sensitivity_upper,
    standard_radius_from_embedding_bound,
)
from .solvers import solve_convex_model


def _validate_positive_lam(cfg: ConvexReleaseConfig) -> None:
    if not np.isfinite(float(cfg.lam)) or float(cfg.lam) <= 0.0:
        raise ValueError(
            "Convex Ball-ERM releases require lam > 0. "
            "The sensitivity and residual certificates rely on strong convexity."
        )


def _validate_convex_design_matrix(dataset: ArrayDataset) -> None:
    X = np.asarray(dataset.X)
    y = np.asarray(dataset.y)

    if X.ndim != 2:
        raise ValueError(
            f"Convex linear models expect dataset.X with shape (n, d), got {X.shape}."
        )
    if y.ndim != 1:
        raise ValueError(
            f"Convex linear models expect dataset.y with shape (n,), got {y.shape}."
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"dataset.X and dataset.y disagree on n: {X.shape[0]} vs {y.shape[0]}."
        )
    if X.shape[0] == 0:
        raise ValueError("Convex ERM requires a non-empty dataset.")


def _add_gaussian_noise_to_payload(payload: Any, sigma: float, key: jr.PRNGKey) -> Any:
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("Gaussian noise sigma must be finite and nonnegative.")

    if sigma == 0.0:
        return payload

    if isinstance(payload, PrototypeRelease):
        noisy = payload.prototypes + sigma * np.asarray(
            jr.normal(key, shape=payload.prototypes.shape, dtype=jnp.float32)
        )
        return PrototypeRelease(
            prototypes=np.asarray(noisy, dtype=np.float32),
            counts=np.asarray(payload.counts),
        )

    if dc.is_dataclass(payload) and not isinstance(payload, eqx.Module):
        fields = {}
        keys = jr.split(key, len(dc.fields(payload)))
        for k, field in zip(keys, dc.fields(payload)):
            fields[field.name] = _add_gaussian_noise_to_payload(
                getattr(payload, field.name),
                sigma,
                k,
            )
        return type(payload)(**fields)

    leaves, treedef = jax.tree_util.tree_flatten(payload)
    keys = jr.split(key, max(1, len(leaves)))

    new_leaves = []
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, (jnp.ndarray, np.ndarray)) and np.issubdtype(
            np.asarray(leaf).dtype,
            np.floating,
        ):
            new_leaves.append(
                leaf
                + sigma
                * jr.normal(
                    keys[i],
                    shape=np.asarray(leaf).shape,
                    dtype=np.asarray(leaf).dtype,
                )
            )
        else:
            new_leaves.append(leaf)

    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def _compute_lz(cfg: ConvexReleaseConfig) -> tuple[float, str]:
    if cfg.lz_mode == "provided":
        if cfg.provided_lz is None:
            raise ValueError("provided_lz must be set when lz_mode='provided'.")
        return float(cfg.provided_lz), "provided"

    if cfg.model_family == "ridge_prototype":
        return lz_prototypes_exact(), "exact"

    if cfg.embedding_bound is None:
        raise ValueError(
            "embedding_bound is required for theorem-backed convex L_z bounds."
        )

    if cfg.model_family == "softmax_logistic":
        return (
            lz_softmax_linear_bound(
                B=cfg.embedding_bound,
                lam=cfg.lam,
                include_bias=True,
            ),
            "upper_bound",
        )

    if cfg.model_family == "squared_hinge":
        return (
            lz_squared_hinge_bound(
                B=cfg.embedding_bound,
                lam=cfg.lam,
                include_bias=True,
            ),
            "upper_bound",
        )

    if cfg.model_family == "binary_logistic":
        return (
            lz_binary_logistic_bound(
                B=cfg.embedding_bound,
                lam=cfg.lam,
                include_bias=True,
            ),
            "glm_upper_bound",
        )

    raise ValueError(cfg.model_family)


def _standard_radius(cfg: ConvexReleaseConfig) -> Optional[float]:
    if cfg.standard_radius is not None:
        return float(cfg.standard_radius)
    return standard_radius_from_embedding_bound(cfg.embedding_bound)


def _utility_metrics(
    payload: Any,
    fam: str,
    dataset: Optional[ArrayDataset],
) -> Dict[str, float]:
    if dataset is None:
        return {}

    if fam == "ridge_prototype":
        preds = prototype_predict(payload, np.asarray(dataset.X))
        return {"accuracy": float(np.mean(preds == dataset.y))}

    if fam == "softmax_logistic":
        logits = softmax_logits(payload, np.asarray(dataset.X))
        return {"accuracy": accuracy_from_logits(logits, dataset.y)}

    if fam == "binary_logistic":
        logits = binary_logits(payload, np.asarray(dataset.X))
        y01 = (
            (dataset.y.astype(np.int64) > 0).astype(np.int64)
            if set(np.unique(dataset.y).tolist()) == {-1, 1}
            else dataset.y.astype(np.int64)
        )
        return {"accuracy": accuracy_from_logits(logits, y01)}

    if fam == "squared_hinge":
        scores = squared_hinge_scores(payload, np.asarray(dataset.X))
        y01 = (dataset.y.astype(np.int64) > 0).astype(np.int64)
        return {"accuracy": accuracy_from_logits(scores, y01)}

    return {}


def _solve_nonprivate_erm(
    dataset: ArrayDataset,
    cfg: ConvexReleaseConfig,
    key: jr.PRNGKey,
):
    _validate_positive_lam(cfg)
    _validate_convex_design_matrix(dataset)

    x = jnp.asarray(dataset.X, dtype=jnp.float32)
    fam = cfg.model_family

    if fam == "ridge_prototype":
        num_classes = int(cfg.num_classes or (int(np.max(dataset.y)) + 1))
        payload = fit_ridge_prototypes(
            np.asarray(dataset.X),
            np.asarray(dataset.y),
            num_classes=num_classes,
            lam=cfg.lam,
        )
        return payload, None, True

    if fam == "softmax_logistic":
        num_classes = int(cfg.num_classes or (int(np.max(dataset.y)) + 1))
        model = SoftmaxLinearModel(x.shape[1], num_classes, key=key)
        res = solve_convex_model(
            model,
            loss_fn=softmax_loss,
            x=x,
            y=jnp.asarray(dataset.y, dtype=jnp.int32),
            lam=cfg.lam,
            key=key,
            cfg=cfg.optimization,
        )
        return res.model, res.cert, False

    if fam == "binary_logistic":
        model = BinaryLogisticModel(x.shape[1], key=key)
        y_pm1 = encode_binary_pm1(dataset.y)
        res = solve_convex_model(
            model,
            loss_fn=binary_logistic_loss,
            x=x,
            y=jnp.asarray(y_pm1, dtype=jnp.float32),
            lam=cfg.lam,
            key=key,
            cfg=cfg.optimization,
        )
        return res.model, res.cert, False

    if fam == "squared_hinge":
        model = SquaredHingeModel(x.shape[1], key=key)
        y_pm1 = encode_binary_pm1(dataset.y)
        res = solve_convex_model(
            model,
            loss_fn=squared_hinge_loss,
            x=x,
            y=jnp.asarray(y_pm1, dtype=jnp.float32),
            lam=cfg.lam,
            key=key,
            cfg=cfg.optimization,
        )
        return res.model, res.cert, False

    raise ValueError(fam)


def _sensitivity(
    cfg: ConvexReleaseConfig,
    *,
    dataset: ArrayDataset,
    opt_cert,
    lz: float,
) -> tuple[float, Optional[float], str]:
    n_total = int(len(dataset))
    std_radius = _standard_radius(cfg)

    if cfg.model_family == "ridge_prototype" and cfg.use_exact_sensitivity_if_available:
        counts = dataset.class_counts(num_classes=cfg.num_classes)
        mode = str(getattr(cfg, "ridge_sensitivity_mode", "global"))

        if mode == "count_aware":
            delta_ball = prototype_count_aware_ball_sensitivity(
                radius=cfg.radius,
                lam=cfg.lam,
                n_total=n_total,
                counts=counts,
            )
            delta_std = (
                None
                if std_radius is None
                else prototype_count_aware_ball_sensitivity(
                    radius=std_radius,
                    lam=cfg.lam,
                    n_total=n_total,
                    counts=counts,
                )
            )
            tag = "exact_count_aware_public_counts"

        elif mode == "global":
            delta_ball = prototype_exact_ball_sensitivity(
                radius=cfg.radius,
                lam=cfg.lam,
                n_total=n_total,
            )
            delta_std = (
                None
                if std_radius is None
                else prototype_exact_ball_sensitivity(
                    radius=std_radius,
                    lam=cfg.lam,
                    n_total=n_total,
                )
            )
            tag = "exact_global_count_worst_case"

        else:
            raise ValueError(
                "ridge_sensitivity_mode must be one of {'global', 'count_aware'}."
            )

    else:
        delta_ball = output_sensitivity_upper(
            lz=lz,
            lam=cfg.lam,
            n_total=n_total,
            radius=cfg.radius,
        )
        delta_std = (
            None
            if std_radius is None
            else output_sensitivity_upper(
                lz=lz,
                lam=cfg.lam,
                n_total=n_total,
                radius=std_radius,
            )
        )
        tag = "upper_bound"

    if opt_cert is not None and not bool(opt_cert.exact_solution):
        tag = f"{tag}+iterative_solver_not_theorem_backed"

    if opt_cert is not None and float(opt_cert.sensitivity_addon) > 0.0:
        delta_ball = approximate_solution_sensitivity(
            exact_sensitivity=delta_ball,
            parameter_error_bound=opt_cert.parameter_error_bound,
        )
        if delta_std is not None:
            delta_std = approximate_solution_sensitivity(
                exact_sensitivity=delta_std,
                parameter_error_bound=opt_cert.parameter_error_bound,
            )
        tag = f"{tag}+local_residual_heuristic"

    return float(delta_ball), None if delta_std is None else float(delta_std), tag


def _theorem_backed_exact_reconstruction(*, opt_cert, is_noiseless: bool) -> bool:
    if not bool(is_noiseless):
        return False
    if opt_cert is None:
        return True
    if not bool(opt_cert.exact_solution):
        return False
    if float(opt_cert.sensitivity_addon) > 0.0:
        return False
    return True


def _artifact(
    *,
    cfg: ConvexReleaseConfig,
    payload: Any,
    dual_ledger,
    dataset: ArrayDataset,
    utility_dataset: Optional[ArrayDataset],
    lz: float,
    lz_source: str,
    delta_ball: float,
    delta_std: Optional[float],
    exact_tag: str,
    opt_cert,
    nonprivate_reference: Any | None,
    release_kind: str,
    theorem_backed_exact_reconstruction: bool,
) -> ReleaseArtifact:
    extra = {}
    if cfg.store_nonprivate_reference and nonprivate_reference is not None:
        extra["nonprivate_reference"] = nonprivate_reference

    return ReleaseArtifact(
        release_kind=release_kind,
        payload=payload,
        model_family=cfg.model_family,
        architecture=cfg.model_family,
        training_config={
            "radius": float(cfg.radius),
            "lam": float(cfg.lam),
            "orders": tuple(float(a) for a in cfg.orders),
            "gaussian_method": str(cfg.gaussian.method),
            "gaussian_tol": float(cfg.gaussian.tol),
            "gaussian_config": dc.asdict(cfg.gaussian),
            "optimization_config": dc.asdict(cfg.optimization),
            "embedding_bound": (
                None if cfg.embedding_bound is None else float(cfg.embedding_bound)
            ),
            "standard_radius": (
                None if cfg.standard_radius is None else float(cfg.standard_radius)
            ),
            "num_classes": (None if cfg.num_classes is None else int(cfg.num_classes)),
            "lz_mode": str(cfg.lz_mode),
            "provided_lz": (
                None if cfg.provided_lz is None else float(cfg.provided_lz)
            ),
            "use_exact_sensitivity_if_available": bool(
                cfg.use_exact_sensitivity_if_available
            ),
            "ridge_sensitivity_mode": str(
                getattr(cfg, "ridge_sensitivity_mode", "global")
            ),
            "epsilon_requested": (None if cfg.epsilon is None else float(cfg.epsilon)),
            "delta_requested": (None if cfg.delta is None else float(cfg.delta)),
            "sigma_requested": (None if cfg.sigma is None else float(cfg.sigma)),
            "seed": int(cfg.seed),
        },
        privacy=dual_ledger,
        sensitivity=SensitivityMetadata(
            lz=float(lz),
            lz_source=lz_source,
            radius=float(cfg.radius),
            delta_ball=float(delta_ball),
            delta_std=None if delta_std is None else float(delta_std),
            exact_vs_upper=exact_tag,
        ),
        optimization=opt_cert,
        attack_metadata={
            "theorem_backed_exact_reconstruction": bool(
                theorem_backed_exact_reconstruction
            ),
        },
        dataset_metadata={
            "n_total": len(dataset),
            "feature_shape": dataset.feature_shape,
            "num_classes": int(cfg.num_classes or dataset.num_classes),
            "label_values": tuple(
                int(v)
                for v in np.unique(np.asarray(dataset.y, dtype=np.int64)).tolist()
            ),
            "class_counts": tuple(
                int(v)
                for v in dataset.class_counts(num_classes=cfg.num_classes).tolist()
            ),
        },
        utility_metrics=_utility_metrics(payload, cfg.model_family, utility_dataset),
        extra=extra,
    )


def run_convex_ball_erm_dp(
    dataset: ArrayDataset,
    cfg: ConvexReleaseConfig,
    *,
    eval_dataset: Optional[ArrayDataset] = None,
) -> ReleaseArtifact:
    if cfg.epsilon is None or cfg.delta is None:
        raise ValueError("Convex Ball-ERM-DP requires epsilon and delta.")

    key = jr.PRNGKey(cfg.seed)
    solve_key, noise_key = jr.split(key)

    nonprivate, opt_cert, _ = _solve_nonprivate_erm(dataset, cfg, solve_key)

    lz, lz_source = _compute_lz(cfg)
    delta_ball, delta_std, tag = _sensitivity(
        cfg,
        dataset=dataset,
        opt_cert=opt_cert,
        lz=lz,
    )

    sigma = gaussian_sigma(
        delta_ball,
        cfg.epsilon,
        cfg.delta,
        method=cfg.gaussian.method,
        tol=cfg.gaussian.tol,
    )

    noisy_payload = _add_gaussian_noise_to_payload(nonprivate, sigma, noise_key)
    standard_radius = _standard_radius(cfg)

    dual_ledger = build_convex_gaussian_ledgers(
        sigma=sigma,
        delta_ball=delta_ball,
        delta_std=delta_std,
        radius=cfg.radius,
        standard_radius=standard_radius,
        orders=cfg.orders,
        dp_delta=cfg.delta,
        gaussian_method=cfg.gaussian.method,
        gaussian_tol=cfg.gaussian.tol,
    )

    return _artifact(
        cfg=cfg,
        payload=noisy_payload,
        dual_ledger=dual_ledger,
        dataset=dataset,
        utility_dataset=eval_dataset,
        lz=lz,
        lz_source=lz_source,
        delta_ball=delta_ball,
        delta_std=delta_std,
        exact_tag=tag,
        opt_cert=opt_cert,
        nonprivate_reference=nonprivate,
        release_kind="convex_ball_erm_dp",
        theorem_backed_exact_reconstruction=_theorem_backed_exact_reconstruction(
            opt_cert=opt_cert,
            is_noiseless=(float(sigma) == 0.0),
        ),
    )


def run_convex_ball_erm_rdp(
    dataset: ArrayDataset,
    cfg: ConvexReleaseConfig,
    *,
    eval_dataset: Optional[ArrayDataset] = None,
) -> ReleaseArtifact:
    if cfg.sigma is None:
        raise ValueError("Convex Ball-ERM-RDP requires sigma.")

    key = jr.PRNGKey(cfg.seed)
    solve_key, noise_key = jr.split(key)

    nonprivate, opt_cert, _ = _solve_nonprivate_erm(dataset, cfg, solve_key)

    lz, lz_source = _compute_lz(cfg)
    delta_ball, delta_std, tag = _sensitivity(
        cfg,
        dataset=dataset,
        opt_cert=opt_cert,
        lz=lz,
    )

    noisy_payload = _add_gaussian_noise_to_payload(nonprivate, cfg.sigma, noise_key)

    dual_ledger = build_convex_gaussian_ledgers(
        sigma=cfg.sigma,
        delta_ball=delta_ball,
        delta_std=delta_std,
        radius=cfg.radius,
        standard_radius=_standard_radius(cfg),
        orders=cfg.orders,
        dp_delta=None,
        gaussian_method=cfg.gaussian.method,
        gaussian_tol=cfg.gaussian.tol,
        extra_dp_deltas=cfg.dp_deltas_for_rdp,
    )

    return _artifact(
        cfg=cfg,
        payload=noisy_payload,
        dual_ledger=dual_ledger,
        dataset=dataset,
        utility_dataset=eval_dataset,
        lz=lz,
        lz_source=lz_source,
        delta_ball=delta_ball,
        delta_std=delta_std,
        exact_tag=tag,
        opt_cert=opt_cert,
        nonprivate_reference=nonprivate,
        release_kind="convex_ball_erm_rdp",
        theorem_backed_exact_reconstruction=_theorem_backed_exact_reconstruction(
            opt_cert=opt_cert,
            is_noiseless=(float(cfg.sigma) == 0.0),
        ),
    )


def run_convex_noiseless_erm_release(
    dataset: ArrayDataset,
    cfg: ConvexReleaseConfig,
    *,
    eval_dataset: Optional[ArrayDataset] = None,
) -> ReleaseArtifact:
    """Release the exact/certified convex ERM solution without privacy noise."""
    key = jr.PRNGKey(cfg.seed)

    nonprivate, opt_cert, _ = _solve_nonprivate_erm(dataset, cfg, key)

    lz, lz_source = _compute_lz(cfg)
    delta_ball, delta_std, tag = _sensitivity(
        cfg,
        dataset=dataset,
        opt_cert=opt_cert,
        lz=lz,
    )

    standard_radius = _standard_radius(cfg)

    dual_ledger = build_convex_gaussian_ledgers(
        sigma=1.0,
        delta_ball=delta_ball,
        delta_std=delta_std,
        radius=cfg.radius,
        standard_radius=standard_radius,
        orders=cfg.orders,
        dp_delta=None,
        gaussian_method=cfg.gaussian.method,
        gaussian_tol=cfg.gaussian.tol,
    )

    dual_ledger.ball.mechanism = "noiseless_convex_release"
    dual_ledger.ball.sigma = None
    dual_ledger.ball.rdp_curve = None
    dual_ledger.ball.dp_certificates = []
    dual_ledger.ball.notes.append(
        "Noiseless release for convex reconstruction and nonprivate baseline workflows. "
        "Theorem-backed exactness is recorded in "
        "attack_metadata['theorem_backed_exact_reconstruction']."
    )

    dual_ledger.standard.mechanism = "noiseless_convex_release"
    dual_ledger.standard.sigma = None
    dual_ledger.standard.rdp_curve = None
    dual_ledger.standard.dp_certificates = []
    dual_ledger.standard.notes.append(
        "Noiseless release for convex reconstruction and nonprivate baseline workflows. "
        "Theorem-backed exactness is recorded in "
        "attack_metadata['theorem_backed_exact_reconstruction']."
    )

    return _artifact(
        cfg=cfg,
        payload=nonprivate,
        dual_ledger=dual_ledger,
        dataset=dataset,
        utility_dataset=eval_dataset,
        lz=lz,
        lz_source=lz_source,
        delta_ball=delta_ball,
        delta_std=delta_std,
        exact_tag=tag,
        opt_cert=opt_cert,
        nonprivate_reference=None,
        release_kind="convex_noiseless_erm",
        theorem_backed_exact_reconstruction=_theorem_backed_exact_reconstruction(
            opt_cert=opt_cert,
            is_noiseless=True,
        ),
    )
