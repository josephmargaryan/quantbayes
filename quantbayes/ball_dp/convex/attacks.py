# quantbayes/ball_dp/convex/attacks.py

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..metrics import reconstruction_metrics
from ..types import ArrayDataset, AttackResult, Record, ReleaseArtifact
from .models.binary_logistic import (
    BinaryLogisticModel,
    binary_logistic_missing_gradient,
    encode_binary_pm1,
)
from .models.ridge_prototype import PrototypeRelease
from .models.softmax_logistic import SoftmaxLinearModel, softmax_missing_gradient
from .models.squared_hinge import SquaredHingeModel, squared_hinge_missing_gradient


def _record_metrics(
    true_record: Optional[Record],
    pred_features: Optional[np.ndarray],
    pred_label: Optional[int],
    eta_grid: tuple[float, ...],
) -> Dict[str, float]:
    if true_record is None or pred_features is None or pred_label is None:
        return {}
    pred = Record(
        features=np.asarray(pred_features, dtype=np.float32), label=int(pred_label)
    )
    return reconstruction_metrics(true_record, pred, eta_grid=eta_grid)


def _validate_common_attack_inputs(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    tol: float,
) -> tuple[float, int]:
    tol_f = float(tol)
    if not np.isfinite(tol_f) or tol_f <= 0.0:
        raise ValueError("tol must be finite and strictly positive.")

    if "lam" not in release.training_config:
        raise ValueError("release.training_config is missing 'lam'.")
    if "n_total" not in release.dataset_metadata:
        raise ValueError("release.dataset_metadata is missing 'n_total'.")

    lam = float(release.training_config["lam"])
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError(
            "Convex noiseless reconstruction requires lam > 0. "
            "The algebraic attacks rely on the strongly convex ERM equations."
        )

    n_total = int(release.dataset_metadata["n_total"])
    if n_total <= 0:
        raise ValueError("release.dataset_metadata['n_total'] must be positive.")

    if int(len(d_minus) + 1) != n_total:
        raise ValueError(
            f"d_minus size mismatch: expected n_total-1 = {n_total - 1}, "
            f"got len(d_minus)={len(d_minus)}."
        )

    X = np.asarray(d_minus.X)
    y = np.asarray(d_minus.y)
    if X.ndim != 2:
        raise ValueError(
            f"Convex attacks expect d_minus.X with shape (n_minus, d), got {X.shape}."
        )
    if y.ndim != 1:
        raise ValueError(
            f"Convex attacks expect d_minus.y with shape (n_minus,), got {y.shape}."
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"d_minus.X and d_minus.y disagree on n_minus: {X.shape[0]} vs {y.shape[0]}."
        )

    return lam, n_total


def _restore_binary_label(y_hat_pm1: int, reference_labels: np.ndarray | None) -> int:
    if reference_labels is not None:
        uniq = sorted(np.unique(np.asarray(reference_labels)).tolist())
        if uniq == [-1, 1]:
            return int(y_hat_pm1)
    return 1 if int(y_hat_pm1) == 1 else 0


def _ridge_sufficient_statistics(
    d_minus: ArrayDataset,
    *,
    num_classes: int,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(d_minus.X, dtype=np.float64)
    y = np.asarray(d_minus.y, dtype=np.int64)

    if X.ndim != 2 or X.shape[1] != int(dim):
        raise ValueError(
            f"ridge_prototype expects d_minus.X with shape (n_minus, {dim}), got {X.shape}."
        )

    if y.size > 0:
        if np.any(y < 0) or np.any(y >= int(num_classes)):
            raise ValueError(
                f"d_minus contains ridge labels outside [0, {int(num_classes) - 1}]."
            )

    counts = np.bincount(y, minlength=int(num_classes))[: int(num_classes)]
    counts = counts.astype(np.int64, copy=False)

    class_sums = np.zeros((int(num_classes), int(dim)), dtype=np.float64)
    if y.size > 0:
        np.add.at(class_sums, y, X)

    return counts, class_sums


def _ridge_mu_minus(
    counts_minus: np.ndarray,
    class_sums: np.ndarray,
    *,
    lam: float,
    n_total: int,
) -> np.ndarray:
    counts_f = counts_minus.astype(np.float64, copy=False)
    denom = 2.0 * counts_f[:, None] + float(lam) * float(n_total)
    return np.where(denom > 0.0, 2.0 * class_sums / denom, 0.0)


def _ridge_candidate_by_class(
    mus_rel: np.ndarray,
    counts_minus: np.ndarray,
    class_sums: np.ndarray,
    *,
    lam: float,
    n_total: int,
) -> np.ndarray:
    alpha = 2.0 * (counts_minus.astype(np.float64) + 1.0) + float(lam) * float(n_total)
    return (0.5 * alpha[:, None] * mus_rel) - class_sums


def _ridge_attack(
    release: PrototypeRelease,
    d_minus: ArrayDataset,
    *,
    lam: float,
    n_total: int,
    known_label: Optional[int],
    tol: float,
    true_record: Optional[Record],
    eta_grid: tuple[float, ...],
) -> AttackResult:
    mus_rel = np.asarray(release.prototypes, dtype=np.float64)
    if mus_rel.ndim != 2:
        raise ValueError(
            "ridge_prototype release.prototypes must have shape (num_classes, dim)."
        )

    k, dim = mus_rel.shape
    attack_name = "convex_ridge_prototype"

    counts_minus, class_sums = _ridge_sufficient_statistics(
        d_minus,
        num_classes=k,
        dim=dim,
    )
    mu_minus = _ridge_mu_minus(
        counts_minus,
        class_sums,
        lam=lam,
        n_total=n_total,
    )
    z_by_class = _ridge_candidate_by_class(
        mus_rel,
        counts_minus,
        class_sums,
        lam=lam,
        n_total=n_total,
    )

    diffs = np.linalg.norm(mus_rel - mu_minus, axis=1)
    candidate_order = np.argsort(-diffs)
    candidates = [
        (int(c), np.asarray(z_by_class[int(c)], dtype=np.float32))
        for c in candidate_order[: min(10, k)].tolist()
    ]

    diagnostics = {
        "mu_minus": np.asarray(mu_minus, dtype=np.float32),
        "counts_minus": counts_minus.astype(np.int64),
        "class_sums_minus": np.asarray(class_sums, dtype=np.float32),
        "diff_norms": diffs.astype(float),
        "candidate_norms_by_class": np.linalg.norm(z_by_class, axis=1).astype(float),
    }

    if known_label is not None:
        y_hat = int(known_label)
        if y_hat < 0 or y_hat >= k:
            raise ValueError(
                f"known_label={y_hat} is outside valid class range [0, {k - 1}]."
            )
        z_hat = np.asarray(z_by_class[y_hat], dtype=np.float32)
        return AttackResult(
            attack_family=attack_name,
            z_hat=z_hat,
            y_hat=y_hat,
            status="exact_given_label",
            diagnostics=diagnostics,
            metrics=_record_metrics(true_record, z_hat, y_hat, eta_grid),
            candidates=candidates,
        )

    changed = np.where(diffs > float(tol))[0]

    if changed.size == 1:
        y_hat = int(changed[0])
        z_hat = np.asarray(z_by_class[y_hat], dtype=np.float32)
        return AttackResult(
            attack_family=attack_name,
            z_hat=z_hat,
            y_hat=y_hat,
            status="exact_unique_changed_class",
            diagnostics=diagnostics,
            metrics=_record_metrics(true_record, z_hat, y_hat, eta_grid),
            candidates=candidates,
        )

    y_hat = int(candidate_order[0])
    z_hat = np.asarray(z_by_class[y_hat], dtype=np.float32)

    status = (
        "fallback_no_detectable_class_shift"
        if changed.size == 0
        else "fallback_largest_shift_class"
    )

    diagnostics = {
        **diagnostics,
        "changed_class_indices": changed.astype(np.int64).tolist(),
    }

    return AttackResult(
        attack_family=attack_name,
        z_hat=z_hat,
        y_hat=y_hat,
        status=status,
        diagnostics=diagnostics,
        metrics=_record_metrics(true_record, z_hat, y_hat, eta_grid),
        candidates=candidates,
    )


def _softmax_attack(
    model: SoftmaxLinearModel,
    d_minus: ArrayDataset,
    *,
    lam: float,
    n_total: int,
    known_label: Optional[int],
    tol: float,
    true_record: Optional[Record],
    eta_grid: tuple[float, ...],
) -> AttackResult:
    G = np.asarray(
        softmax_missing_gradient(
            model,
            d_minus.X,
            d_minus.y,
            lam=lam,
            n_total=n_total,
        ),
        dtype=np.float64,
    )

    if G.ndim != 2 or G.shape[1] < 2:
        raise ValueError(
            "softmax_missing_gradient must return shape (num_classes, d + 1)."
        )

    a = np.asarray(G[:, -1], dtype=np.float64)
    attack_name = "convex_softmax_logistic"

    neg = np.where(a < -float(tol))[0]
    if known_label is not None:
        y_hat = int(known_label)
        if y_hat < 0 or y_hat >= G.shape[0]:
            raise ValueError(
                f"known_label={y_hat} is outside valid class range [0, {G.shape[0] - 1}]."
            )
    else:
        y_hat = int(neg[0]) if neg.size == 1 else int(np.argmin(a))

    denom = float(np.dot(a, a))
    row_sum_residual = float(np.linalg.norm(G.sum(axis=0)))

    if denom <= float(tol):
        return AttackResult(
            attack_family=attack_name,
            z_hat=None,
            y_hat=y_hat,
            status="degenerate_last_column",
            diagnostics={
                "missing_gradient": G.astype(np.float32),
                "a": a.astype(np.float32),
                "known_label_used": known_label is not None,
                "negative_row_indices": neg.astype(np.int64).tolist(),
                "row_sum_residual": row_sum_residual,
            },
            metrics={},
        )

    z_aug_raw = (a[:, None] * G).sum(axis=0) / denom
    residual_raw = float(np.linalg.norm(G - a[:, None] * z_aug_raw[None, :]))
    G_norm = float(np.linalg.norm(G))
    relative_residual = residual_raw / max(G_norm, float(tol))

    normalizable = abs(float(z_aug_raw[-1])) > float(tol)

    base_diagnostics = {
        "missing_gradient": G.astype(np.float32),
        "a": a.astype(np.float32),
        "known_label_used": known_label is not None,
        "negative_row_indices": neg.astype(np.int64).tolist(),
        "rank1_residual": residual_raw,
        "relative_rank1_residual": relative_residual,
        "row_sum_residual": row_sum_residual,
    }

    if normalizable:
        z_aug = z_aug_raw / z_aug_raw[-1]
        z_hat = np.asarray(z_aug[:-1], dtype=np.float32)

        if known_label is not None:
            status = "exact_given_label"
        elif neg.size == 1:
            status = "exact"
        else:
            status = "fallback_rank1_regression"

        return AttackResult(
            attack_family=attack_name,
            z_hat=z_hat,
            y_hat=y_hat,
            status=status,
            diagnostics={
                **base_diagnostics,
                "last_coordinate_after_normalization": float(z_aug[-1]),
            },
            metrics=_record_metrics(true_record, z_hat, y_hat, eta_grid),
        )

    z_hat = np.asarray(z_aug_raw[:-1], dtype=np.float32)
    return AttackResult(
        attack_family=attack_name,
        z_hat=z_hat,
        y_hat=y_hat,
        status="fallback_rank1_regression_without_normalization",
        diagnostics={
            **base_diagnostics,
            "last_coordinate_raw": float(z_aug_raw[-1]),
        },
        metrics=_record_metrics(true_record, z_hat, y_hat, eta_grid),
    )


def _binary_logistic_attack(
    model: BinaryLogisticModel,
    d_minus: ArrayDataset,
    *,
    lam: float,
    n_total: int,
    tol: float,
    true_record: Optional[Record],
    eta_grid: tuple[float, ...],
) -> AttackResult:
    y_minus_pm1 = encode_binary_pm1(d_minus.y)
    g = np.asarray(
        binary_logistic_missing_gradient(
            model,
            d_minus.X,
            y_minus_pm1,
            lam=lam,
            n_total=n_total,
        ),
        dtype=np.float64,
    )

    attack_name = "convex_binary_logistic"

    if g.ndim != 1 or g.shape[0] < 2:
        raise ValueError("binary_logistic_missing_gradient must return shape (d + 1,).")

    if abs(float(g[-1])) <= float(tol):
        return AttackResult(
            attack_family=attack_name,
            z_hat=None,
            y_hat=None,
            status="degenerate_bias_gradient",
            diagnostics={"missing_gradient": g.astype(np.float32)},
            metrics={},
        )

    y_hat_pm1 = -1 if g[-1] > 0 else 1
    z_aug = g / g[-1]
    z_hat = np.asarray(z_aug[:-1], dtype=np.float32)
    y_hat = _restore_binary_label(y_hat_pm1, d_minus.y)

    return AttackResult(
        attack_family=attack_name,
        z_hat=z_hat,
        y_hat=y_hat,
        status="exact",
        diagnostics={
            "missing_gradient": g.astype(np.float32),
            "y_hat_pm1": int(y_hat_pm1),
            "bias_component": float(g[-1]),
        },
        metrics=_record_metrics(true_record, z_hat, y_hat, eta_grid),
    )


def _squared_hinge_attack(
    model: SquaredHingeModel,
    d_minus: ArrayDataset,
    *,
    lam: float,
    n_total: int,
    tol: float,
    true_record: Optional[Record],
    eta_grid: tuple[float, ...],
) -> AttackResult:
    y_minus_pm1 = encode_binary_pm1(d_minus.y)
    g = np.asarray(
        squared_hinge_missing_gradient(
            model,
            d_minus.X,
            y_minus_pm1,
            lam=lam,
            n_total=n_total,
        ),
        dtype=np.float64,
    )

    attack_name = "convex_squared_hinge"

    if g.ndim != 1 or g.shape[0] < 2:
        raise ValueError("squared_hinge_missing_gradient must return shape (d + 1,).")

    if np.linalg.norm(g) <= float(tol):
        return AttackResult(
            attack_family=attack_name,
            z_hat=None,
            y_hat=None,
            status="inactive_margin_non_identifiable",
            diagnostics={"missing_gradient": g.astype(np.float32)},
            metrics={},
        )

    if abs(float(g[-1])) <= float(tol):
        return AttackResult(
            attack_family=attack_name,
            z_hat=None,
            y_hat=None,
            status="degenerate_bias_gradient",
            diagnostics={"missing_gradient": g.astype(np.float32)},
            metrics={},
        )

    y_hat_pm1 = -1 if g[-1] > 0 else 1
    z_aug = g / g[-1]
    z_hat = np.asarray(z_aug[:-1], dtype=np.float32)
    margin = float(y_hat_pm1 * (np.dot(np.asarray(model.w), z_hat) + float(model.b)))
    status = (
        "exact_support_vector_branch"
        if margin < 1.0 + 1e-8
        else "recovered_but_margin_check_failed"
    )
    y_hat = _restore_binary_label(y_hat_pm1, d_minus.y)

    return AttackResult(
        attack_family=attack_name,
        z_hat=z_hat,
        y_hat=y_hat,
        status=status,
        diagnostics={
            "missing_gradient": g.astype(np.float32),
            "y_hat_pm1": int(y_hat_pm1),
            "margin": margin,
            "bias_component": float(g[-1]),
        },
        metrics=_record_metrics(true_record, z_hat, y_hat, eta_grid),
    )


def run_convex_attack(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    side_info: Optional[Dict[str, Any]] = None,
    true_record: Optional[Record] = None,
    tol: float = 1e-8,
    eta_grid: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0),
) -> AttackResult:
    """Public noiseless convex reconstruction API.

    This function implements algebraic / missing-gradient attacks derived for
    noiseless convex ERM releases.

    It is not the correct attack for Gaussian output-perturbation releases. For
    those, use run_convex_ball_output_map_attack(...), which optimizes the exact
    Gaussian posterior objective.
    """
    sigma_ball = release.privacy.ball.sigma
    sigma_std = release.privacy.standard.sigma
    if sigma_ball is not None or sigma_std is not None:
        raise ValueError(
            "run_convex_attack is the noiseless convex reconstruction API. "
            "This release carries Gaussian output noise (sigma is not None). "
            "Use run_convex_ball_output_map_attack(...) instead."
        )

    lam, n_total = _validate_common_attack_inputs(
        release,
        d_minus,
        tol=tol,
    )

    fam = str(release.model_family)
    known_label = None if side_info is None else side_info.get("known_label")
    theorem_backed_exact = bool(
        release.attack_metadata.get("theorem_backed_exact_reconstruction", False)
    )

    if fam == "ridge_prototype":
        attack = _ridge_attack(
            release.payload,
            d_minus,
            lam=lam,
            n_total=n_total,
            known_label=known_label,
            tol=tol,
            true_record=true_record,
            eta_grid=eta_grid,
        )
    elif fam == "softmax_logistic":
        attack = _softmax_attack(
            release.payload,
            d_minus,
            lam=lam,
            n_total=n_total,
            known_label=known_label,
            tol=tol,
            true_record=true_record,
            eta_grid=eta_grid,
        )
    elif fam == "binary_logistic":
        attack = _binary_logistic_attack(
            release.payload,
            d_minus,
            lam=lam,
            n_total=n_total,
            tol=tol,
            true_record=true_record,
            eta_grid=eta_grid,
        )
    elif fam == "squared_hinge":
        attack = _squared_hinge_attack(
            release.payload,
            d_minus,
            lam=lam,
            n_total=n_total,
            tol=tol,
            true_record=true_record,
            eta_grid=eta_grid,
        )
    else:
        raise ValueError(fam)

    if not theorem_backed_exact:
        attack.status = f"noncertified:{attack.status}"

    return attack
