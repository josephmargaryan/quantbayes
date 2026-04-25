# quantbayes/ball_dp/convex/finite_prior_diagnostics.py

from __future__ import annotations

from typing import Any, Optional, Sequence

import jax.random as jr
import numpy as np

from ..evaluation.rero import (
    gaussian_direct_ball_rero_bound,
    gaussian_direct_finite_prior_bound,
)
from ..types import ArrayDataset, Record, ReleaseArtifact
from .ball_output_attacks import (
    _candidate_dataset,
    _convex_cfg_from_release,
    _normalize_finite_prior_records,
    _payload_vector,
    _release_sigma,
)
from .models.ridge_prototype import (
    prototype_exact_ball_sensitivity,
    prototype_instance_ball_sensitivity,
    prototype_known_label_inverse_noise_scale,
)
from .releases import _solve_nonprivate_erm


def _summary_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = int(X.shape[0])
    if n < 2:
        return np.zeros((0,), dtype=np.float64)
    diffs = X[:, None, :] - X[None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    tri = np.triu_indices(n, k=1)
    return d[tri]


def _nearest_neighbor_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = int(X.shape[0])
    if n < 2:
        return np.zeros((0,), dtype=np.float64)
    diffs = X[:, None, :] - X[None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


def _solve_payload_vector_for_record(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    rec: Record,
) -> np.ndarray:
    release_cfg = _convex_cfg_from_release(release)
    candidate_ds = _candidate_dataset(
        d_minus,
        np.asarray(rec.features, dtype=np.float32),
        int(rec.label),
    )
    payload, _, _ = _solve_nonprivate_erm(
        candidate_ds,
        release_cfg,
        jr.PRNGKey(int(release_cfg.seed)),
    )
    return _payload_vector(payload).astype(np.float64, copy=False)


def compute_convex_finite_prior_diagnostics(
    release: ReleaseArtifact,
    d_minus: ArrayDataset,
    *,
    prior_records: Sequence[Record],
    prior_weights: Optional[Sequence[float]] = None,
    known_label: Optional[int] = None,
    center_record: Optional[Record] = None,
) -> dict[str, float | int | str | bool | None]:
    """Support-level diagnostics for finite-prior Gaussian exact identification.

    The finite-prior MAP rule is Bayes-optimal for the supplied candidate support.
    This helper explains the difficulty of that support by computing actual
    model-space candidate separations and direct finite-Gaussian upper bounds.

    The returned direct finite bound is theorem-backed for the concrete Gaussian
    hypothesis testing problem: it uses the exact candidate means
    f(D^- union {z_i}) and a reference Gaussian with the same covariance.
    """
    sigma = _release_sigma(release)
    filtered_records, probs = _normalize_finite_prior_records(
        prior_records,
        prior_weights,
        known_label=known_label,
    )
    m = int(len(filtered_records))
    if m <= 0:
        raise ValueError("Finite prior support is empty.")

    candidate_features = np.stack(
        [
            np.asarray(rec.features, dtype=np.float64).reshape(-1)
            for rec in filtered_records
        ],
        axis=0,
    )
    candidate_labels = np.asarray(
        [int(rec.label) for rec in filtered_records], dtype=np.int64
    )

    candidate_vecs = np.stack(
        [
            _solve_payload_vector_for_record(release, d_minus, rec)
            for rec in filtered_records
        ],
        axis=0,
    )

    if center_record is not None:
        ref_vec = _solve_payload_vector_for_record(release, d_minus, center_record)
        reference_kind = "center_record"
        center_feature = np.asarray(center_record.features, dtype=np.float64).reshape(
            -1
        )
    else:
        # This arbitrary Gaussian reference is still valid for the finite Gaussian
        # testing bound, but it is not tied to the Ball-ReRo center u.
        ref_vec = np.average(candidate_vecs, axis=0, weights=probs)
        reference_kind = "weighted_mean_parameter"
        center_feature = np.average(candidate_features, axis=0, weights=probs)

    model_center_dists = np.linalg.norm(candidate_vecs - ref_vec[None, :], axis=1)
    model_pairwise = _pairwise_distances(candidate_vecs)
    model_nn = _nearest_neighbor_distances(candidate_vecs)
    feature_center_dists = np.linalg.norm(
        candidate_features - center_feature[None, :], axis=1
    )
    feature_pairwise = _pairwise_distances(candidate_features)
    feature_nn = _nearest_neighbor_distances(candidate_features)

    kappa = float(np.max(probs))
    max_model_radius = float(np.max(model_center_dists)) if m else 0.0
    direct_center_max = gaussian_direct_ball_rero_bound(
        kappa=kappa,
        sensitivity=max_model_radius,
        sigma=float(sigma),
    )
    direct_finite_opt = gaussian_direct_finite_prior_bound(
        weights=probs,
        mean_distances_over_sigma=model_center_dists / float(sigma),
    )

    out: dict[str, float | int | str | bool | None] = {
        "diagnostic_kind": "convex_finite_prior_gaussian",
        "diagnostic_reference": reference_kind,
        "diagnostic_model_family": str(release.model_family),
        "prior_size": int(m),
        "prior_kappa": float(kappa),
        "diagnostic_sigma": float(sigma),
        "bound_direct_instance_center_max": float(direct_center_max),
        "bound_direct_instance_finite_opt": float(direct_finite_opt),
        "model_center_radius_max": max_model_radius,
        "model_center_radius_mean": float(np.mean(model_center_dists)),
        "model_center_radius_over_sigma_max": float(max_model_radius / float(sigma)),
        "model_center_radius_over_sigma_mean": float(
            np.mean(model_center_dists / float(sigma))
        ),
        "candidate_label_unique_count": int(len(np.unique(candidate_labels))),
    }
    out.update(_summary_stats("model_pairwise_distance", model_pairwise))
    out.update(_summary_stats("model_pairwise_snr", model_pairwise / float(sigma)))
    out.update(_summary_stats("model_nn_distance", model_nn))
    out.update(_summary_stats("model_nn_snr", model_nn / float(sigma)))
    out.update(_summary_stats("feature_center_distance", feature_center_dists))
    out.update(_summary_stats("feature_pairwise_distance", feature_pairwise))
    out.update(_summary_stats("feature_nn_distance", feature_nn))

    if str(release.model_family) == "ridge_prototype" and known_label is not None:
        label = int(known_label)
        y_minus = np.asarray(d_minus.y, dtype=np.int64)
        n_label_minus = int(np.sum(y_minus == label))
        n_total = int(release.dataset_metadata.get("n_total", len(d_minus) + 1))
        lam = float(release.training_config["lam"])
        radius = float(
            release.training_config.get("radius", release.privacy.ball.radius)
        )
        alpha_y = 2.0 * (float(n_label_minus) + 1.0) + lam * float(n_total)
        tau_y = prototype_known_label_inverse_noise_scale(
            sigma=float(sigma),
            lam=lam,
            n_total=n_total,
            n_label_minus=n_label_minus,
        )
        global_delta = prototype_exact_ball_sensitivity(
            radius=radius,
            lam=lam,
            n_total=n_total,
        )
        instance_delta = prototype_instance_ball_sensitivity(
            radius=radius,
            lam=lam,
            n_total=n_total,
            class_count=n_label_minus + 1,
        )
        out.update(
            {
                "ridge_known_label": True,
                "ridge_label": int(label),
                "ridge_n_label_minus": int(n_label_minus),
                "ridge_n_label_full": int(n_label_minus + 1),
                "ridge_alpha_y": float(alpha_y),
                "ridge_feature_to_model_scale": float(2.0 / alpha_y),
                "ridge_inverse_noise_tau": float(tau_y),
                "ridge_inverse_noise_tau_over_radius": (
                    float(tau_y / radius) if radius > 0.0 else float("nan")
                ),
                "ridge_global_delta_at_radius": float(global_delta),
                "ridge_instance_delta_at_radius": float(instance_delta),
                "ridge_count_dilution": (
                    float(instance_delta / global_delta)
                    if global_delta > 0.0
                    else float("nan")
                ),
            }
        )
        out.update(
            _summary_stats(
                "ridge_feature_pairwise_snr", feature_pairwise / float(tau_y)
            )
        )
        out.update(_summary_stats("ridge_feature_nn_snr", feature_nn / float(tau_y)))
    else:
        out.update(
            {
                "ridge_known_label": False,
                "ridge_label": None,
                "ridge_n_label_minus": None,
                "ridge_n_label_full": None,
                "ridge_alpha_y": float("nan"),
                "ridge_feature_to_model_scale": float("nan"),
                "ridge_inverse_noise_tau": float("nan"),
                "ridge_inverse_noise_tau_over_radius": float("nan"),
                "ridge_global_delta_at_radius": float("nan"),
                "ridge_instance_delta_at_radius": float("nan"),
                "ridge_count_dilution": float("nan"),
                "ridge_feature_pairwise_snr_min": float("nan"),
                "ridge_feature_pairwise_snr_median": float("nan"),
                "ridge_feature_pairwise_snr_mean": float("nan"),
                "ridge_feature_pairwise_snr_max": float("nan"),
                "ridge_feature_nn_snr_min": float("nan"),
                "ridge_feature_nn_snr_median": float("nan"),
                "ridge_feature_nn_snr_mean": float("nan"),
                "ridge_feature_nn_snr_max": float("nan"),
            }
        )

    return out
