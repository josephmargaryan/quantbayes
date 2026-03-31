from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


__all__ = [
    "summarize_embedding_ball_radii",
    "select_ball_radius",
]


_DEFAULT_QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99, 1.0)


def _quantile_key(q: float) -> str:
    return f"q{float(q):.3f}".rstrip("0").rstrip(".")


def _validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    qs = tuple(float(q) for q in quantiles)
    if not qs:
        raise ValueError("quantiles must be non-empty.")
    for q in qs:
        if not (0.0 <= q <= 1.0):
            raise ValueError("all quantiles must lie in [0, 1].")
    return qs


def _python_scalar(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    return x


def _coerce_labels(y: np.ndarray) -> tuple[np.ndarray, str]:
    arr = np.asarray(y)
    if arr.ndim == 0:
        raise ValueError("y must have shape [N] or [N, K].")
    if arr.ndim == 2 and arr.shape[1] > 1:
        return np.asarray(np.argmax(arr, axis=1)), "one_hot_argmax"

    flat = np.asarray(arr).reshape(-1)
    if flat.ndim != 1:
        raise ValueError("Could not coerce y to a 1D label array.")

    if np.issubdtype(flat.dtype, np.floating):
        rounded = np.rint(flat)
        if np.allclose(flat, rounded):
            flat = rounded.astype(np.int64)
    return flat, "index"


def _exact_pairwise_l2_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = int(X.shape[0])
    if n < 2:
        return np.zeros((0,), dtype=np.float32)

    sq_norms = np.sum(X * X, axis=1, keepdims=True)
    sq_dists = np.maximum(sq_norms + sq_norms.T - 2.0 * (X @ X.T), 0.0)
    tri = np.triu_indices(n, k=1)
    return np.sqrt(sq_dists[tri], dtype=np.float32)


def _sample_pairwise_l2_distances(
    X: np.ndarray,
    num_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = int(X.shape[0])
    m = int(num_pairs)
    if n < 2 or m <= 0:
        return np.zeros((0,), dtype=np.float32)

    # Draw ordered distinct pairs i.i.d. from the off-diagonal index set.
    i = rng.integers(0, n, size=m, endpoint=False)
    j = rng.integers(0, n - 1, size=m, endpoint=False)
    j = j + (j >= i)
    diffs = X[i] - X[j]
    return np.linalg.norm(diffs, axis=1).astype(np.float32, copy=False)


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    qs = np.asarray(tuple(float(q) for q in quantiles), dtype=np.float64)

    if values.size == 0:
        return np.full(qs.shape, np.nan, dtype=np.float64)
    if values.size != weights.size:
        raise ValueError("values and weights must have the same length.")
    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative.")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("sum of weights must be positive.")

    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w)
    cdf = (cdf - 0.5 * w) / total
    cdf[0] = max(cdf[0], 0.0)
    cdf[-1] = min(cdf[-1], 1.0)
    return np.interp(qs, cdf, v)


def summarize_embedding_ball_radii(
    X: np.ndarray,
    y: np.ndarray,
    *,
    quantiles: Sequence[float] = _DEFAULT_QUANTILES,
    max_exact_pairs: int = 250_000,
    max_sampled_pairs: int = 100_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize within-label pairwise distances for principled Ball-radius choice.

    This helper is intended for the embedding-space setting in which Ball adjacency
    uses a label-preserving Euclidean feature metric. Cross-label distances are
    therefore irrelevant to the protected neighbor relation and are intentionally
    omitted here.

    Parameters
    ----------
    X:
        Array with shape [N, d] or [N, ...]. Non-leading dimensions are flattened.
    y:
        Label array with shape [N] or [N, K]. If shape [N, K], labels are inferred
        via argmax along the last axis.
    quantiles:
        Pairwise-distance quantiles to report.
    max_exact_pairs:
        If a class has at most this many within-class pairs, all pairwise distances
        are computed exactly for that class.
    max_sampled_pairs:
        Otherwise, this many within-class pairs are sampled i.i.d. to estimate the
        pairwise-distance distribution for that class.
    seed:
        RNG seed used only when pair sampling is needed.

    Returns
    -------
    dict
        A structured report containing per-label summaries, pooled within-label
        quantiles, conservative max-labelwise quantiles, and candidate radii.
    """
    qs = _validate_quantiles(quantiles)
    if int(max_exact_pairs) < 1:
        raise ValueError("max_exact_pairs must be >= 1.")
    if int(max_sampled_pairs) < 1:
        raise ValueError("max_sampled_pairs must be >= 1.")

    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim < 2:
        raise ValueError("X must have shape [N, d] or [N, ...].")
    n = int(X_arr.shape[0])
    X_flat = X_arr.reshape(n, -1)

    labels, label_mode = _coerce_labels(np.asarray(y))
    if int(labels.shape[0]) != n:
        raise ValueError("X and y must have the same leading dimension.")

    rng = np.random.default_rng(int(seed))
    unique_labels = np.unique(labels)

    per_label: list[dict[str, Any]] = []
    pooled_values: list[np.ndarray] = []
    pooled_weights: list[np.ndarray] = []

    for label in unique_labels:
        mask = labels == label
        Xc = X_flat[mask]
        n_c = int(Xc.shape[0])
        num_pairs_total = n_c * (n_c - 1) // 2

        row: dict[str, Any] = {
            "label": _python_scalar(label),
            "n_examples": n_c,
            "num_pairs_total": int(num_pairs_total),
        }

        if n_c < 2:
            row.update(
                {
                    "num_pairs_used": 0,
                    "pair_sampling_mode": "insufficient_examples",
                    "mean_distance": None,
                    "max_distance_observed": None,
                    "max_distance_exact": None,
                    "quantiles": {_quantile_key(q): None for q in qs},
                }
            )
            per_label.append(row)
            continue

        if num_pairs_total <= int(max_exact_pairs):
            distances = _exact_pairwise_l2_distances(Xc)
            mode = "exact"
            max_distance_exact: Optional[float] = float(np.max(distances))
        else:
            num_pairs_used = min(int(max_sampled_pairs), int(num_pairs_total))
            distances = _sample_pairwise_l2_distances(Xc, num_pairs_used, rng)
            mode = "sampled_iid_pairs"
            max_distance_exact = None

        num_pairs_used = int(distances.shape[0])
        qvals = np.quantile(distances, qs)
        weight_per_sample = float(num_pairs_total) / float(max(num_pairs_used, 1))

        pooled_values.append(distances.astype(np.float64, copy=False))
        pooled_weights.append(
            np.full((num_pairs_used,), weight_per_sample, dtype=np.float64)
        )

        row.update(
            {
                "num_pairs_used": num_pairs_used,
                "pair_sampling_mode": mode,
                "mean_distance": float(np.mean(distances)),
                "max_distance_observed": float(np.max(distances)),
                "max_distance_exact": max_distance_exact,
                "quantiles": {
                    _quantile_key(q): float(v) for q, v in zip(qs, qvals, strict=True)
                },
            }
        )
        per_label.append(row)

    valid_rows = [row for row in per_label if row["num_pairs_used"] > 0]
    if not valid_rows:
        raise ValueError(
            "No label had at least two examples, so no within-label pairwise distances "
            "could be computed."
        )

    pooled_values_arr = np.concatenate(pooled_values, axis=0)
    pooled_weights_arr = np.concatenate(pooled_weights, axis=0)
    pooled_qvals = _weighted_quantiles(pooled_values_arr, pooled_weights_arr, qs)

    max_labelwise_quantiles: dict[str, float] = {}
    mean_labelwise_quantiles: dict[str, float] = {}
    candidate_radii: dict[str, float] = {}
    for q in qs:
        key = _quantile_key(q)
        labelwise = [float(row["quantiles"][key]) for row in valid_rows]
        pooled_val = float(pooled_qvals[list(qs).index(q)])
        max_val = float(np.max(labelwise))
        mean_val = float(np.mean(labelwise))

        max_labelwise_quantiles[key] = max_val
        mean_labelwise_quantiles[key] = mean_val
        candidate_radii[f"pooled_{key}"] = pooled_val
        candidate_radii[f"max_labelwise_{key}"] = max_val
        candidate_radii[f"mean_labelwise_{key}"] = mean_val

    all_exact = all(row["pair_sampling_mode"] == "exact" for row in valid_rows)
    if all_exact:
        global_max_exact: Optional[float] = float(
            np.max([float(row["max_distance_exact"]) for row in valid_rows])
        )
    else:
        global_max_exact = None
    global_max_observed = float(
        np.max([float(row["max_distance_observed"]) for row in valid_rows])
    )
    candidate_radii["global_max_observed"] = global_max_observed
    if global_max_exact is not None:
        candidate_radii["global_max_exact"] = global_max_exact

    return {
        "metric": "within_label_l2_embedding_distance",
        "label_mode": label_mode,
        "n_examples": n,
        "embedding_dimension": int(X_flat.shape[1]),
        "num_labels": int(unique_labels.shape[0]),
        "quantiles": tuple(float(q) for q in qs),
        "pair_estimation": {
            "max_exact_pairs": int(max_exact_pairs),
            "max_sampled_pairs": int(max_sampled_pairs),
            "seed": int(seed),
        },
        "per_label": per_label,
        "pooled_within_label_quantiles": {
            _quantile_key(q): float(v) for q, v in zip(qs, pooled_qvals, strict=True)
        },
        "max_labelwise_quantiles": max_labelwise_quantiles,
        "mean_labelwise_quantiles": mean_labelwise_quantiles,
        "global_max_within_label_distance_observed": global_max_observed,
        "global_max_within_label_distance_exact": global_max_exact,
        "candidate_radii": candidate_radii,
        "notes": [
            "Cross-label distances are intentionally excluded because the Ball metric is label-preserving.",
            "'pooled_*' radii summarize the distribution of a random within-label pair.",
            "'max_labelwise_*' radii are more conservative: they take the worst class-specific quantile.",
            "'global_max_*' is the within-label diameter and is usually very outlier-sensitive.",
        ],
    }


def select_ball_radius(
    report: Mapping[str, Any],
    *,
    strategy: str = "max_labelwise_quantile",
    quantile: float = 0.95,
    allow_observed_max: bool = False,
) -> float:
    """Extract a scalar Ball radius from a radius-selection report.

    Recommended default:
        strategy="max_labelwise_quantile", quantile=0.95

    Supported strategies:
        - "max_labelwise_quantile"
        - "pooled_quantile"
        - "mean_labelwise_quantile"
        - "global_max"
    """
    strat = str(strategy).strip().lower()
    qkey = _quantile_key(float(quantile))

    if strat == "max_labelwise_quantile":
        table = report.get("max_labelwise_quantiles", {})
        if qkey not in table:
            raise KeyError(f"Quantile {qkey!r} not present in report.")
        return float(table[qkey])

    if strat == "pooled_quantile":
        table = report.get("pooled_within_label_quantiles", {})
        if qkey not in table:
            raise KeyError(f"Quantile {qkey!r} not present in report.")
        return float(table[qkey])

    if strat == "mean_labelwise_quantile":
        table = report.get("mean_labelwise_quantiles", {})
        if qkey not in table:
            raise KeyError(f"Quantile {qkey!r} not present in report.")
        return float(table[qkey])

    if strat == "global_max":
        exact = report.get("global_max_within_label_distance_exact", None)
        if exact is not None:
            return float(exact)
        if allow_observed_max:
            observed = report.get("global_max_within_label_distance_observed", None)
            if observed is None:
                raise KeyError("Report does not contain an observed within-label max.")
            return float(observed)
        raise ValueError(
            "global_max requested, but the report does not contain an exact global max. "
            "Either recompute with a larger max_exact_pairs or set allow_observed_max=True."
        )

    raise ValueError(
        "Unsupported strategy. Use one of "
        "{'max_labelwise_quantile', 'pooled_quantile', 'mean_labelwise_quantile', 'global_max'}."
    )
