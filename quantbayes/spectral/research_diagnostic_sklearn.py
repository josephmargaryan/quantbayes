# quantbayes/stochax/diagnostics/research_diagnostic_sklearn.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "compute_diagnostics_sklearn",
    "pretty_print_diagnostics",
    "plot_margin_distribution",
]

import math
import numpy as np

"""
Example Usage:
from quantbayes.stochax.diagnostics.research_diagnostic_sklearn import (
    compute_diagnostics_sklearn, pretty_print_diagnostics, plot_margin_distribution
)

# Fit any of your spectral sklearn models
clf = SpectralCirculantLogisticRegression(
    padded_dim=None, K=None, feature_scaling="sobolev", sobolev_s=1.0,
    C=1.0, max_iter=500, verbose=False
).fit(X_train, y_train)

d = compute_diagnostics_sklearn(clf, margin_subset=(X_val, y_val))
pretty_print_diagnostics(d)
plot_margin_distribution(d)

"""

# =============================== helpers =====================================


def _as_logits(est, X: np.ndarray) -> np.ndarray:
    """
    Get logits/scores:
      - prefer decision_function
      - else for binary, convert predict_proba to logit
    """
    if hasattr(est, "decision_function"):
        z = est.decision_function(X)
    elif hasattr(est, "predict_proba"):
        P = est.predict_proba(X)
        if P.ndim == 2 and P.shape[1] == 2:
            p1 = np.clip(P[:, 1], 1e-12, 1 - 1e-12)
            z = np.log(p1 / (1 - p1))  # logit
        else:
            # multiclass without decision_function → fall back to centered scores
            # (not ideal; most linear estimators expose decision_function)
            z = np.log(np.clip(P, 1e-12, 1)) - np.log(
                np.clip(1 - P, 1e-12, 1)
            )  # rough log-odds
    else:
        raise ValueError("Estimator must have decision_function or predict_proba.")
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    return z


def _multiclass_margins(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Binary logits: (N,1) → margin = (2y-1)*logit  with y in {0,1}
    Multiclass logits: (N,K) → s_y - max_{j≠y} s_j
    """
    y = np.asarray(y)
    z = np.asarray(logits)
    if z.ndim == 1 or z.shape[1] == 1:  # binary
        z1 = z.reshape(-1)
        # accept {0,1} or {±1}
        if set(np.unique(y)).issubset({0, 1}):
            s = 2.0 * y.astype(np.float64) - 1.0
        else:
            s = y.astype(np.float64)  # assume ±1
        return s * z1
    N, K = z.shape
    y_int = y.astype(int)
    true = z[np.arange(N), y_int]
    z_mask = z.copy()
    z_mask[np.arange(N), y_int] = -np.inf
    other = np.max(z_mask, axis=1)
    return true - other


def _summary(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    q05, q50, q95 = np.quantile(a, [0.05, 0.5, 0.95])
    return dict(
        mean=float(a.mean()),
        std=float(a.std()),
        q05=float(q05),
        q50=float(q50),
        q95=float(q95),
    )


def _gamma_at(arr: np.ndarray, eps: float) -> float:
    return float((np.asarray(arr) <= float(eps)).mean())


def _weight_matrix(est) -> np.ndarray:
    """
    Return W in R^{K×D} for mapping x ↦ logits = W x + b.
    Supports:
      - your Spectral* estimators: expose W_ (C×D) or weight_ (D,) for binary
      - sklearn Linear models: coef_ (C×D)
    """
    if hasattr(est, "W_"):
        W = np.asarray(est.W_)
    elif hasattr(est, "weight_"):
        W = np.asarray(est.weight_)
        if W.ndim == 1:
            W = W[None, :]
    elif hasattr(est, "coef_"):
        W = np.asarray(est.coef_)
        if W.ndim == 1:
            W = W[None, :]
    else:
        raise ValueError("Cannot extract weight matrix: need .W_, .weight_, or .coef_.")
    return W


def _operator_norm(W: np.ndarray) -> float:
    """Spectral norm of W: largest singular value."""
    # for speed we can use np.linalg.norm(W, 2) but SVD is clearer
    s = np.linalg.svd(W, compute_uv=False)
    return float(s[0]) if s.size else 0.0


# ============================== main API =====================================


def compute_diagnostics_sklearn(
    est: Any,
    *,
    margin_subset: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    L_override: Optional[float] = None,  # if you want to force a specific L
) -> Dict[str, Any]:
    """
    Diagnostics for linear (logit) models in sklearn style.
      - L_used = ||W||_2 exact (unless you pass L_override)
      - margins / normalized margins / certified radius LB
    """
    W = _weight_matrix(est)  # (K, D)
    L_exact = _operator_norm(W)  # exact Lipschitz of x↦logits
    L_used = float(L_override) if (L_override is not None) else max(1e-12, L_exact)

    out: Dict[str, Any] = {
        "layer_coverage_pct": 100.0,
        "n_layers_covered": 1,
        "serial_prod_sigma_bound": float(L_exact),  # single linear layer
        "skipaware_lipschitz_bound": float(L_used),
        "sigmas": [(float(L_exact), "LinearMap", "W", "hint")],
        "cert_only_sigmas": [(float(L_exact), "LinearMap", "W", "hint")],
        "empirical_sigmas": [],
    }

    if margin_subset is not None:
        Xs, Ys = margin_subset
        logits = _as_logits(est, Xs)

        margins = _multiclass_margins(logits, Ys)
        mstats = _summary(margins)

        norm_margins = margins / L_used
        nstats = _summary(norm_margins)

        radii = np.maximum(margins, 0.0) / (2.0 * L_used)
        rstats = _summary(radii)

        out.update(
            {
                # raw margins
                "margins_mean": mstats["mean"],
                "margins_std": mstats["std"],
                "margins_q05": mstats["q05"],
                "margins_q50": mstats["q50"],
                "margins_q95": mstats["q95"],
                "margins_frac_nonpos": float((margins <= 0).mean()),
                "gamma@0.0": _gamma_at(margins, 0.0),
                "gamma@0.1": _gamma_at(margins, 0.1),
                "gamma@0.5": _gamma_at(margins, 0.5),
                # normalized margins
                "norm_margins_mean": nstats["mean"],
                "norm_margins_std": nstats["std"],
                "norm_margins_q05": nstats["q05"],
                "norm_margins_q50": nstats["q50"],
                "norm_margins_q95": nstats["q95"],
                # certified radius LB
                "cert_radius_lb_mean": rstats["mean"],
                "cert_radius_lb_q05": rstats["q05"],
                "cert_radius_lb_q50": rstats["q50"],
                "cert_radius_lb_q95": rstats["q95"],
                # arrays (for plotting)
                "_margins_array": margins.astype(np.float64),
                "_norm_margins_array": norm_margins.astype(np.float64),
                "_radii_array": radii.astype(np.float64),
                "_logits_array": logits.astype(np.float64),
            }
        )

    return out


def pretty_print_diagnostics(d: Dict[str, Any]) -> None:
    print("— diagnostics (linear, exact) —")
    print(
        f"layer coverage: {d.get('layer_coverage_pct',0.0):.1f}%  ({d.get('n_layers_covered',0)} covered)"
    )
    print(f"exact_operator_norm: {d.get('serial_prod_sigma_bound', float('nan')):g}")
    print(f"L_used: {d.get('skipaware_lipschitz_bound', float('nan')):g}")

    if "margins_mean" in d:
        print(
            "margins: mean={:.3f}, std={:.3f}, q05={:.4g}, q50={:.3f}, q95={:.3f}, frac<=0={:.4g}".format(
                d["margins_mean"],
                d["margins_std"],
                d["margins_q05"],
                d["margins_q50"],
                d["margins_q95"],
                d["margins_frac_nonpos"],
            )
        )
        for eps in (0.0, 0.1, 0.5):
            k = f"gamma@{eps}"
            if k in d:
                print(f"{k}: {d[k]:.4g}")

    if "norm_margins_mean" in d:
        print(
            "normalized margins: mean={:.3f}, std={:.3f}, q05={:.4g}, q50={:.3f}, q95={:.3f}".format(
                d["norm_margins_mean"],
                d["norm_margins_std"],
                d["norm_margins_q05"],
                d["norm_margins_q50"],
                d["norm_margins_q95"],
            )
        )

    if "cert_radius_lb_mean" in d:
        print(
            "cert_radius_lb: mean={:.3f}, q05={:.4g}, q50={:.3f}, q95={:.3f}".format(
                d["cert_radius_lb_mean"],
                d["cert_radius_lb_q05"],
                d["cert_radius_lb_q50"],
                d["cert_radius_lb_q95"],
            )
        )


def plot_margin_distribution(
    d: Dict[str, Any],
    *,
    bins: int = 60,
    which: Tuple[str, ...] = ("raw", "normalized", "radius"),
    show: bool = True,
):
    import matplotlib.pyplot as plt

    avail = []
    if "raw" in which and "_margins_array" in d:
        avail.append(("Margins", np.asarray(d["_margins_array"], float)))
    if "normalized" in which and "_norm_margins_array" in d:
        avail.append(
            ("Normalized margins", np.asarray(d["_norm_margins_array"], float))
        )
    if "radius" in which and "_radii_array" in d:
        avail.append(("Certified radius (LB)", np.asarray(d["_radii_array"], float)))

    if not avail:
        raise ValueError(
            "Nothing to plot. Provide margin_subset to compute_diagnostics_sklearn."
        )

    rows = len(avail)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3.5 * rows))
    if rows == 1:
        axes = np.array([axes], dtype=object)

    for i, (title, arr) in enumerate(avail):
        axh = axes[i, 0]
        axc = axes[i, 1]

        axh.hist(arr, bins=bins, density=True)
        axh.axvline(0.0, linestyle="--")
        q05, q50, q95 = np.quantile(arr, [0.05, 0.5, 0.95])
        axh.set_title(f"{title}\nq05={q05:.3f}, q50={q50:.3f}, q95={q95:.3f}")
        axh.set_xlabel(title.lower())
        axh.set_ylabel("density")

        xs = np.linspace(float(arr.min()), float(arr.max()), 512)
        cdf = np.array([(arr <= x).mean() for x in xs])
        axc.plot(xs, cdf)
        axc.axvline(0.0, linestyle="--")
        axc.set_ylim(0, 1)
        if "radius" in title:
            axc.set_title("CDF")
        else:
            axc.set_title(f"CDF (frac≤0 = {(arr <= 0).mean():.4f})")
        axc.set_xlabel(title.lower())
        axc.set_ylabel("CDF")

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes
