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
    pred = Record(features=np.asarray(pred_features), label=int(pred_label))
    return reconstruction_metrics(true_record, pred, eta_grid=eta_grid)


def _restore_binary_label(y_hat_pm1: int, reference_labels: np.ndarray | None) -> int:
    if reference_labels is not None:
        uniq = sorted(np.unique(np.asarray(reference_labels)).tolist())
        if uniq == [-1, 1]:
            return int(y_hat_pm1)
    return 1 if int(y_hat_pm1) == 1 else 0


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
    mus_rel = np.asarray(release.prototypes)
    k = mus_rel.shape[0]
    attack_name = "convex_ridge_prototype"

    class_sums = np.zeros_like(mus_rel)
    for c in range(k):
        idx = np.where(d_minus.y == c)[0]
        if idx.size > 0:
            class_sums[c] = d_minus.X[idx].sum(axis=0)

    mu_minus = np.zeros_like(mus_rel)
    counts_minus = np.bincount(d_minus.y.astype(np.int64), minlength=k)
    for c in range(k):
        denom = 2.0 * counts_minus[c] + float(lam) * float(n_total)
        if denom > 0:
            mu_minus[c] = (2.0 * class_sums[c]) / denom

    if known_label is not None:
        y_hat = int(known_label)
        if y_hat < 0 or y_hat >= k:
            raise ValueError(
                f"known_label={y_hat} is outside valid class range [0, {k - 1}]."
            )
        z_hat = (
            (2.0 * (counts_minus[y_hat] + 1) + float(lam) * float(n_total)) / 2.0
        ) * mus_rel[y_hat] - class_sums[y_hat]
        return AttackResult(
            attack_name,
            z_hat=np.asarray(z_hat),
            y_hat=y_hat,
            status="exact_given_label",
            diagnostics={"mu_minus": mu_minus, "counts_minus": counts_minus},
            metrics=_record_metrics(true_record, z_hat, y_hat, eta_grid),
        )

    diffs = np.linalg.norm(mus_rel - mu_minus, axis=1)
    changed = np.where(diffs > tol)[0]

    if changed.size == 1:
        y_hat = int(changed[0])
        z_hat = (
            (2.0 * (counts_minus[y_hat] + 1) + float(lam) * float(n_total)) / 2.0
        ) * mus_rel[y_hat] - class_sums[y_hat]
        return AttackResult(
            attack_name,
            z_hat=np.asarray(z_hat),
            y_hat=y_hat,
            status="exact_unique_changed_class",
            diagnostics={
                "mu_minus": mu_minus,
                "counts_minus": counts_minus,
                "diff_norms": diffs,
            },
            metrics=_record_metrics(true_record, z_hat, y_hat, eta_grid),
        )

    y_hat = int(np.argmax(diffs))
    z_hat = (
        (2.0 * (counts_minus[y_hat] + 1) + float(lam) * float(n_total)) / 2.0
    ) * mus_rel[y_hat] - class_sums[y_hat]

    if changed.size == 0:
        candidates = [(int(c), mu_minus[c].copy()) for c in range(k)]
        return AttackResult(
            attack_name,
            z_hat=np.asarray(z_hat),
            y_hat=y_hat,
            status="fallback_no_detectable_class_shift",
            diagnostics={
                "mu_minus": mu_minus,
                "counts_minus": counts_minus,
                "diff_norms": diffs,
            },
            candidates=candidates,
            metrics=_record_metrics(true_record, z_hat, y_hat, eta_grid),
        )

    return AttackResult(
        attack_name,
        z_hat=np.asarray(z_hat),
        y_hat=y_hat,
        status="fallback_largest_shift_class",
        diagnostics={
            "mu_minus": mu_minus,
            "counts_minus": counts_minus,
            "diff_norms": diffs,
        },
        metrics=_record_metrics(true_record, z_hat, y_hat, eta_grid),
    )


def _softmax_attack(
    model: SoftmaxLinearModel,
    d_minus: ArrayDataset,
    *,
    lam: float,
    n_total: int,
    tol: float,
    true_record: Optional[Record],
    eta_grid: tuple[float, ...],
) -> AttackResult:
    G = softmax_missing_gradient(model, d_minus.X, d_minus.y, lam=lam, n_total=n_total)
    a = G[:, -1]
    attack_name = "convex_softmax_logistic"

    neg = np.where(a < -tol)[0]
    y_hat = int(neg[0]) if neg.size == 1 else int(np.argmin(a))

    denom = float(np.dot(a, a))
    if abs(denom) <= tol:
        return AttackResult(
            attack_name,
            z_hat=None,
            y_hat=y_hat,
            status="degenerate_last_column",
            diagnostics={"missing_gradient": G, "a": a},
        )

    z_aug_raw = (a[:, None] * G).sum(axis=0) / denom
    residual_raw = float(np.linalg.norm(G - a[:, None] * z_aug_raw[None, :]))

    if neg.size == 1 and abs(z_aug_raw[-1]) > tol:
        z_aug = z_aug_raw / z_aug_raw[-1]
        z_hat = z_aug[:-1]
        return AttackResult(
            attack_name,
            z_hat=np.asarray(z_hat),
            y_hat=y_hat,
            status="exact",
            diagnostics={
                "missing_gradient": G,
                "a": a,
                "last_coordinate_after_normalization": float(z_aug[-1]),
                "row_residual": residual_raw,
            },
            metrics=_record_metrics(true_record, z_hat, y_hat, eta_grid),
        )

    if abs(z_aug_raw[-1]) > tol:
        z_aug = z_aug_raw / z_aug_raw[-1]
        z_hat = z_aug[:-1]
        status = "fallback_rank1_regression"
        diagnostics = {
            "missing_gradient": G,
            "a": a,
            "rank1_residual": residual_raw,
            "last_coordinate_after_normalization": float(z_aug[-1]),
        }
    else:
        z_hat = z_aug_raw[:-1]
        status = "fallback_rank1_regression_without_normalization"
        diagnostics = {
            "missing_gradient": G,
            "a": a,
            "rank1_residual": residual_raw,
            "last_coordinate_raw": float(z_aug_raw[-1]),
        }

    return AttackResult(
        attack_name,
        z_hat=np.asarray(z_hat),
        y_hat=y_hat,
        status=status,
        diagnostics=diagnostics,
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
    g = binary_logistic_missing_gradient(
        model, d_minus.X, y_minus_pm1, lam=lam, n_total=n_total
    )
    attack_name = "convex_binary_logistic"

    if abs(g[-1]) <= tol:
        return AttackResult(
            attack_name,
            z_hat=None,
            y_hat=None,
            status="degenerate_bias_gradient",
            diagnostics={"missing_gradient": g},
        )

    y_hat_pm1 = -1 if g[-1] > 0 else 1
    z_aug = g / g[-1]
    z_hat = z_aug[:-1]
    y_hat = _restore_binary_label(y_hat_pm1, d_minus.y)

    return AttackResult(
        attack_name,
        z_hat=np.asarray(z_hat),
        y_hat=y_hat,
        status="exact",
        diagnostics={
            "missing_gradient": g,
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
    g = squared_hinge_missing_gradient(
        model, d_minus.X, y_minus_pm1, lam=lam, n_total=n_total
    )
    attack_name = "convex_squared_hinge"

    if np.linalg.norm(g) <= tol:
        return AttackResult(
            attack_name,
            z_hat=None,
            y_hat=None,
            status="inactive_margin_non_identifiable",
            diagnostics={"missing_gradient": g},
        )

    if abs(g[-1]) <= tol:
        return AttackResult(
            attack_name,
            z_hat=None,
            y_hat=None,
            status="degenerate_bias_gradient",
            diagnostics={"missing_gradient": g},
        )

    y_hat_pm1 = -1 if g[-1] > 0 else 1
    z_aug = g / g[-1]
    z_hat = z_aug[:-1]
    margin = float(y_hat_pm1 * (np.dot(np.asarray(model.w), z_hat) + float(model.b)))
    status = (
        "exact_support_vector_branch"
        if margin < 1.0 + 1e-8
        else "recovered_but_margin_check_failed"
    )
    y_hat = _restore_binary_label(y_hat_pm1, d_minus.y)

    return AttackResult(
        attack_name,
        z_hat=np.asarray(z_hat),
        y_hat=y_hat,
        status=status,
        diagnostics={
            "missing_gradient": g,
            "y_hat_pm1": int(y_hat_pm1),
            "margin": margin,
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
    """
    Public convex reconstruction API.

    The computation is chosen solely by `release.model_family`.
    Exactness guarantees are recorded in release metadata; they are not exposed
    as a user-facing attack mode.
    """
    fam = release.model_family
    lam = float(release.training_config["lam"])
    n_total = int(release.dataset_metadata["n_total"])
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
