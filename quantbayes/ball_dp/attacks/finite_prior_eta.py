# quantbayes/ball_dp/attacks/finite_prior_eta.py

from __future__ import annotations

import dataclasses as dc
from typing import Any, Literal, Sequence

import numpy as np


EtaDecoderMode = Literal[
    "exact_map",
    "support_eta_bayes",
    "arbitrary_eta_bayes",
]


@dc.dataclass(frozen=True)
class SubsetBall:
    """Minimum-enclosing-ball information for one nonempty support subset."""

    mask: int
    size: int
    radius: float
    center: np.ndarray


@dc.dataclass(frozen=True)
class FiniteSupportEtaGeometry:
    """Precomputed geometry for finite-prior eta-loss decisions.

    Parameters
    ----------
    X:
        Support features, shape (m, ...). Internally flattened to (m, d).
    y:
        Support labels, shape (m,). The eta-loss in this helper is feature-space
        Euclidean distance. In the canonical finite-prior setup all support
        labels are equal, so this matches the label-preserving metric.
    weights:
        Prior weights on the support.
    pairwise_distances:
        Matrix D_{ij} = ||x_i - x_j||_2.
    subset_masks:
        Integer bitmask for every nonempty subset.
    subset_membership:
        Boolean matrix of shape (num_subsets, m).
    subset_radii:
        Minimum-enclosing-ball radius of each subset.
    subset_centers:
        One corresponding enclosing-ball center for each subset.
    """

    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    pairwise_distances: np.ndarray
    subset_masks: np.ndarray
    subset_membership: np.ndarray
    subset_radii: np.ndarray
    subset_centers: np.ndarray

    @property
    def m(self) -> int:
        return int(self.X.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.X.reshape(self.m, -1).shape[1])

    @property
    def diameter(self) -> float:
        return float(np.max(self.pairwise_distances)) if self.m else 0.0


def _as_flat_features(X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim < 2:
        raise ValueError("X must have shape (m, ...).")
    return X_arr.reshape(X_arr.shape[0], -1)


def _normalize_weights(weights: Sequence[float] | None, m: int) -> np.ndarray:
    if weights is None:
        w = np.full((int(m),), 1.0 / float(m), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != (int(m),):
            raise ValueError(f"weights must have shape ({int(m)},).")
        if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
            raise ValueError("weights must be finite and strictly positive.")
        w = w / float(np.sum(w))
    return w.astype(np.float64, copy=False)


def pairwise_l2(X: np.ndarray) -> np.ndarray:
    Xf = _as_flat_features(X)
    return np.linalg.norm(Xf[:, None, :] - Xf[None, :, :], axis=-1)


def subset_indices_from_mask(mask: int, m: int) -> tuple[int, ...]:
    mask_i = int(mask)
    return tuple(i for i in range(int(m)) if (mask_i >> i) & 1)


def _circumsphere_from_boundary(P: np.ndarray) -> tuple[np.ndarray, float]:
    """Affine circumsphere candidate determined by boundary points P.

    For k>=2, this computes the affine-hull center equidistant from boundary
    points using least squares. It is used inside exhaustive active-set
    enumeration for small supports.
    """
    P = np.asarray(P, dtype=np.float64)
    k, d = P.shape

    if k == 0:
        return np.zeros((d,), dtype=np.float64), -np.inf

    if k == 1:
        return P[0].copy(), 0.0

    p0 = P[0]
    A = P[1:] - p0
    G = A @ A.T
    b = 0.5 * np.sum(A * A, axis=1)

    beta, *_ = np.linalg.lstsq(G, b, rcond=1e-10)
    center = p0 + beta @ A
    radius = float(np.max(np.linalg.norm(P - center[None, :], axis=1)))
    return center, radius


def min_enclosing_ball_small(
    P: np.ndarray,
    *,
    tol: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Exact minimum enclosing ball for small point sets by active-set search.

    This enumerates all possible active boundary subsets. It is intended for
    small finite supports, e.g. m <= 10 or m <= 12.
    """
    P = np.asarray(P, dtype=np.float64)
    n, d = P.shape

    if n == 0:
        raise ValueError("P must be nonempty.")

    if n == 1:
        return P[0].copy(), 0.0

    best_radius = float("inf")
    best_center: np.ndarray | None = None

    for mask in range(1, 1 << n):
        idx = subset_indices_from_mask(mask, n)
        center, radius = _circumsphere_from_boundary(P[list(idx)])

        if radius >= best_radius:
            continue

        d_all = np.linalg.norm(P - center[None, :], axis=1)
        if np.all(d_all <= radius + float(tol)):
            best_radius = float(radius)
            best_center = np.asarray(center, dtype=np.float64)

    if best_center is None:
        # This should not happen, because the full set is always feasible.
        return np.mean(P, axis=0), float(
            np.max(np.linalg.norm(P - np.mean(P, axis=0), axis=1))
        )

    return best_center, float(best_radius)


def precompute_subset_balls(
    X: np.ndarray,
    *,
    max_m: int = 12,
    tol: float = 1e-10,
) -> list[SubsetBall]:
    """Precompute minimum-enclosing balls for every nonempty subset."""
    Xf = _as_flat_features(X)
    m = int(Xf.shape[0])

    if m <= 0:
        raise ValueError("support must be nonempty.")

    if m > int(max_m):
        raise ValueError(
            f"Exact subset enumeration requested for m={m}, but max_m={int(max_m)}. "
            "Increase max_m only if you understand the exponential cost."
        )

    out: list[SubsetBall] = []

    for mask in range(1, 1 << m):
        idx = subset_indices_from_mask(mask, m)
        center, radius = min_enclosing_ball_small(Xf[list(idx)], tol=float(tol))
        out.append(
            SubsetBall(
                mask=int(mask),
                size=int(len(idx)),
                radius=float(radius),
                center=np.asarray(center, dtype=np.float64),
            )
        )

    return out


def build_finite_support_eta_geometry(
    X: np.ndarray,
    y: Sequence[int] | np.ndarray | None = None,
    weights: Sequence[float] | None = None,
    *,
    max_m_for_exact_subsets: int = 12,
    tol: float = 1e-10,
) -> FiniteSupportEtaGeometry:
    """Build all geometry needed for exact finite-prior eta-Bayes decisions."""
    Xf = _as_flat_features(X)
    m, d = Xf.shape

    if m < 1:
        raise ValueError("support must contain at least one point.")

    if y is None:
        y_arr = np.zeros((m,), dtype=np.int32)
    else:
        y_arr = np.asarray(y, dtype=np.int32).reshape(-1)
        if y_arr.shape != (m,):
            raise ValueError(f"y must have shape ({m},).")

    w = _normalize_weights(weights, m)
    D = pairwise_l2(Xf)

    subset_balls = precompute_subset_balls(
        Xf,
        max_m=int(max_m_for_exact_subsets),
        tol=float(tol),
    )

    subset_masks = np.asarray([sb.mask for sb in subset_balls], dtype=np.int64)
    subset_radii = np.asarray([sb.radius for sb in subset_balls], dtype=np.float64)
    subset_centers = np.stack([sb.center for sb in subset_balls], axis=0).reshape(-1, d)

    membership = np.zeros((len(subset_balls), m), dtype=bool)
    for row, mask in enumerate(subset_masks.tolist()):
        membership[row, list(subset_indices_from_mask(int(mask), m))] = True

    return FiniteSupportEtaGeometry(
        X=Xf.astype(np.float64, copy=False),
        y=y_arr.astype(np.int32, copy=False),
        weights=w,
        pairwise_distances=D.astype(np.float64, copy=False),
        subset_masks=subset_masks,
        subset_membership=membership,
        subset_radii=subset_radii,
        subset_centers=subset_centers.astype(np.float64, copy=False),
    )


def eta_grid_from_geometry(
    geometry: FiniteSupportEtaGeometry,
    *,
    dense_points: int = 201,
    include_midpoints: bool = False,
) -> np.ndarray:
    """Eta grid containing all support-center and arbitrary-center breakpoints."""
    D = np.asarray(geometry.pairwise_distances, dtype=np.float64)
    pairwise_vals = D[np.triu_indices(D.shape[0], k=1)]

    vals = [
        np.array([0.0], dtype=np.float64),
        pairwise_vals,
        np.asarray(geometry.subset_radii, dtype=np.float64),
    ]

    if int(dense_points) > 0:
        vals.append(
            np.linspace(
                0.0, float(geometry.diameter), int(dense_points), dtype=np.float64
            )
        )

    eta = np.concatenate(vals)
    eta = eta[np.isfinite(eta)]
    eta = eta[eta >= 0.0]
    eta = np.unique(np.round(eta, decimals=12))
    eta.sort()

    if include_midpoints and eta.size >= 2:
        mids = 0.5 * (eta[:-1] + eta[1:])
        eta = np.unique(np.concatenate([eta, mids]))
        eta.sort()

    return eta.astype(np.float64, copy=False)


def finite_support_kappa_rows(
    geometry: FiniteSupportEtaGeometry,
    eta_grid: Sequence[float],
) -> list[dict[str, float]]:
    """Compute oblivious eta-loss baselines for the finite support.

    kappa_support_center:
        Best prior mass captured by an eta-ball centered at one support point.

    kappa_exact_arbitrary_center:
        Best prior mass captured by an eta-ball centered anywhere in feature space.

    kappa_upper_2eta:
        Support-centered 2 eta upper envelope, useful as a conservative comparison.
    """
    w = np.asarray(geometry.weights, dtype=np.float64)
    D = np.asarray(geometry.pairwise_distances, dtype=np.float64)

    subset_masses = geometry.subset_membership.astype(np.float64) @ w

    rows: list[dict[str, float]] = []

    for eta in np.asarray(eta_grid, dtype=np.float64):
        eta_f = float(eta)

        support_scores = (D <= eta_f + 1e-12).astype(np.float64).T @ w
        support_scores_2eta = (D <= 2.0 * eta_f + 1e-12).astype(np.float64).T @ w

        feasible = geometry.subset_radii <= eta_f + 1e-10
        arbitrary = float(np.max(subset_masses[feasible])) if np.any(feasible) else 0.0

        rows.append(
            {
                "eta": eta_f,
                "kappa_support_center": float(np.max(support_scores)),
                "kappa_exact_arbitrary_center": arbitrary,
                "kappa_upper_2eta": float(np.max(support_scores_2eta)),
            }
        )

    return rows


def normalize_posterior(posterior: Sequence[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(posterior, dtype=np.float64).reshape(-1)

    if p.size == 0:
        raise ValueError("posterior must be nonempty.")

    if np.any(~np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("posterior entries must be finite and nonnegative.")

    total = float(np.sum(p))
    if total <= 0.0:
        raise ValueError("posterior must have positive total mass.")

    return (p / total).astype(np.float64, copy=False)


def posterior_probabilities_from_attack_result(result: Any) -> np.ndarray:
    """Extract candidate posterior probabilities from an AttackResult-like object."""
    diagnostics = getattr(result, "diagnostics", None)
    if diagnostics is None:
        raise ValueError("attack result has no diagnostics dictionary.")

    for key in (
        "candidate_posterior_probs",
        "candidate_posterior_probabilities",
        "posterior_probabilities",
    ):
        if key in diagnostics and diagnostics[key] is not None:
            return normalize_posterior(diagnostics[key])

    if "candidate_log_posteriors" in diagnostics:
        logp = np.asarray(diagnostics["candidate_log_posteriors"], dtype=np.float64)
        shifted = logp - float(np.max(logp))
        return normalize_posterior(np.exp(shifted))

    raise KeyError(
        "Could not find posterior probabilities in attack diagnostics. "
        "Expected one of candidate_posterior_probs, "
        "candidate_posterior_probabilities, posterior_probabilities, "
        "or candidate_log_posteriors."
    )


def _map_index_from_posterior(posterior: np.ndarray) -> int:
    return int(np.argmax(np.asarray(posterior, dtype=np.float64)))


def exact_map_eta_decision(
    geometry: FiniteSupportEtaGeometry,
    posterior: Sequence[float] | np.ndarray,
    *,
    eta: float,
    true_index: int | None = None,
) -> dict[str, Any]:
    """Exact-ID MAP prediction, evaluated under eta-loss."""
    p = normalize_posterior(posterior)

    if p.shape != (geometry.m,):
        raise ValueError(f"posterior must have shape ({geometry.m},).")

    pred_idx = _map_index_from_posterior(p)
    covered = geometry.pairwise_distances[:, pred_idx] <= float(eta) + 1e-12
    posterior_success = float(np.sum(p[covered]))

    empirical_success = None
    if true_index is not None:
        ti = int(true_index)
        empirical_success = float(bool(covered[ti]))

    return {
        "decoder_mode": "exact_map",
        "eta": float(eta),
        "posterior_success_probability": posterior_success,
        "empirical_success": empirical_success,
        "predicted_prior_index": int(pred_idx),
        "predicted_subset_mask": int(1 << pred_idx),
        "predicted_subset_size": int(np.sum(covered)),
        "predicted_subset_indices": tuple(np.flatnonzero(covered).astype(int).tolist()),
        "true_index": None if true_index is None else int(true_index),
    }


def support_eta_bayes_decision(
    geometry: FiniteSupportEtaGeometry,
    posterior: Sequence[float] | np.ndarray,
    *,
    eta: float,
    true_index: int | None = None,
) -> dict[str, Any]:
    """Bayes-optimal eta decoder restricted to output a support point."""
    p = normalize_posterior(posterior)

    if p.shape != (geometry.m,):
        raise ValueError(f"posterior must have shape ({geometry.m},).")

    within = geometry.pairwise_distances <= float(eta) + 1e-12
    scores = within.astype(np.float64).T @ p

    pred_idx = int(np.argmax(scores))
    covered = within[:, pred_idx]

    empirical_success = None
    if true_index is not None:
        ti = int(true_index)
        empirical_success = float(bool(covered[ti]))

    return {
        "decoder_mode": "support_eta_bayes",
        "eta": float(eta),
        "posterior_success_probability": float(scores[pred_idx]),
        "empirical_success": empirical_success,
        "predicted_prior_index": int(pred_idx),
        "predicted_subset_mask": int(
            sum((1 << int(i)) for i in np.flatnonzero(covered).tolist())
        ),
        "predicted_subset_size": int(np.sum(covered)),
        "predicted_subset_indices": tuple(np.flatnonzero(covered).astype(int).tolist()),
        "true_index": None if true_index is None else int(true_index),
    }


def arbitrary_eta_bayes_decision(
    geometry: FiniteSupportEtaGeometry,
    posterior: Sequence[float] | np.ndarray,
    *,
    eta: float,
    true_index: int | None = None,
) -> dict[str, Any]:
    """Bayes-optimal eta decoder allowed to output any embedding center.

    The optimal posterior success probability is

        max_{A subset S : rad(A) <= eta} sum_{i in A} posterior_i.

    The returned predicted_subset identifies the optimal captured subset. The
    corresponding predicted center is the minimum-enclosing-ball center of that
    subset.
    """
    p = normalize_posterior(posterior)

    if p.shape != (geometry.m,):
        raise ValueError(f"posterior must have shape ({geometry.m},).")

    subset_masses = geometry.subset_membership.astype(np.float64) @ p
    feasible = geometry.subset_radii <= float(eta) + 1e-10

    if not np.any(feasible):
        # This should not happen for eta >= 0 because singletons are feasible.
        best_row = int(np.argmax(subset_masses))
    else:
        masked = np.where(feasible, subset_masses, -np.inf)
        best_row = int(np.argmax(masked))

    best_mask = int(geometry.subset_masks[best_row])
    covered = geometry.subset_membership[best_row]
    pred_indices = tuple(np.flatnonzero(covered).astype(int).tolist())

    empirical_success = None
    if true_index is not None:
        ti = int(true_index)
        empirical_success = float(bool(covered[ti]))

    return {
        "decoder_mode": "arbitrary_eta_bayes",
        "eta": float(eta),
        "posterior_success_probability": float(subset_masses[best_row]),
        "empirical_success": empirical_success,
        "predicted_prior_index": None,
        "predicted_subset_mask": best_mask,
        "predicted_subset_size": int(np.sum(covered)),
        "predicted_subset_indices": pred_indices,
        "predicted_center": geometry.subset_centers[best_row].copy(),
        "predicted_radius": float(geometry.subset_radii[best_row]),
        "true_index": None if true_index is None else int(true_index),
    }


def evaluate_eta_decoders(
    geometry: FiniteSupportEtaGeometry,
    posterior: Sequence[float] | np.ndarray,
    eta_grid: Sequence[float],
    *,
    true_index: int | None = None,
    modes: Sequence[EtaDecoderMode] = (
        "exact_map",
        "support_eta_bayes",
        "arbitrary_eta_bayes",
    ),
) -> list[dict[str, Any]]:
    """Evaluate finite-prior eta-loss decision rules on one posterior."""
    p = normalize_posterior(posterior)

    if p.shape != (geometry.m,):
        raise ValueError(f"posterior must have shape ({geometry.m},).")

    allowed = {"exact_map", "support_eta_bayes", "arbitrary_eta_bayes"}
    modes_tuple = tuple(str(m) for m in modes)
    bad = sorted(set(modes_tuple).difference(allowed))
    if bad:
        raise ValueError(f"unknown eta decoder mode(s): {bad}")

    rows: list[dict[str, Any]] = []

    for eta in np.asarray(eta_grid, dtype=np.float64):
        eta_f = float(eta)

        if "exact_map" in modes_tuple:
            rows.append(
                exact_map_eta_decision(
                    geometry,
                    p,
                    eta=eta_f,
                    true_index=true_index,
                )
            )

        if "support_eta_bayes" in modes_tuple:
            rows.append(
                support_eta_bayes_decision(
                    geometry,
                    p,
                    eta=eta_f,
                    true_index=true_index,
                )
            )

        if "arbitrary_eta_bayes" in modes_tuple:
            rows.append(
                arbitrary_eta_bayes_decision(
                    geometry,
                    p,
                    eta=eta_f,
                    true_index=true_index,
                )
            )

    return rows
