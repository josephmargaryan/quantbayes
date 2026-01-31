# quantbayes/stochax/utils/research_diagnostics.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "plot_margin_distribution",
    "plot_margin_overlays",  # NEW
    "pretty_print_diagnostics",
    "compute_diagnostics",
    "compute_and_save_diagnostics",  # NEW
    "save_diagnostics_npz",  # NEW
    "load_diagnostics_npz",  # NEW
]

import math
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


"""
Example Usage:

from functools import partial
from quantbayes.stochax.diagnostics.research_diagnostic import (
    compute_and_save_diagnostics,
    load_diagnostics_npz,
    plot_margin_overlays,
)
from quantbayes.stochax import predict
from quantbayes.stochax.utils.lip_upper import make_lipschitz_upper_fn

# Pick one certified global bound (keep it the same across runs)
L_fn = make_lipschitz_upper_fn(
    conv_mode="circ_plus_lr",    # or "tn", but be consistent
    conv_tn_iters=8,
    conv_input_shape=(32, 32),
)

# Train/evaluate normally; then:
d_clean = compute_and_save_diagnostics(
    model_clean, state_clean,
    margin_subset=(X_val_clean, y_val_clean),
    predict_fn=predict,
    lipschitz_upper_bound_fn=L_fn,
    save_path="diag_cifar10_clean.npz",
    save_meta={"dataset": "CIFAR10", "labels": "clean"},
)

d_rand = compute_and_save_diagnostics(
    model_rand, state_rand,
    margin_subset=(X_val_rand, y_val_rand),
    predict_fn=predict,
    lipschitz_upper_bound_fn=L_fn,
    save_path="diag_cifar10_rand.npz",
    save_meta={"dataset": "CIFAR10", "labels": "random"},
)

# Later (or in another script):
D1 = load_diagnostics_npz("diag_cifar10_clean.npz")
D2 = load_diagnostics_npz("diag_cifar10_rand.npz")
plot_margin_overlays([D1, D2],
                     labels=["CIFAR-10", "CIFAR-10 (random labels)"],
                     which=("normalized",))   # or ("raw","normalized")

"""


# =============================================================================
# Internal utilities
# =============================================================================


def _is_param_svd_conv_name(name: str) -> bool:
    # Never treat these modules' hints as convolution operator norms.
    # They typically expose a diagnostic kernel-matrix proxy, not the conv operator.
    PARAM_SVD_CONV = {"SpectralConv2d", "AdaptiveSpectralConv2d"}
    return name in PARAM_SVD_CONV


def _try_operator_norm_hint(x: Any) -> Optional[float]:
    """Return a certified operator-norm hint if available; else None."""
    if hasattr(x, "__operator_norm_hint__"):
        try:
            val = x.__operator_norm_hint__()
            if val is not None:
                val = float(val)
                if not (math.isfinite(val) and val > 0):
                    return None
                # Block known param-SVD conv classes from being treated as conv op norms.
                name = type(x).__name__
                if _is_param_svd_conv_name(name):
                    return None
                return val
        except Exception:
            return None
    return None


class _SigmaEntry(eqx.Module):
    sigma: float
    qualname: str
    path: str
    source: str  # "hint" | "empirical"

    def as_tuple(self) -> Tuple[float, str, str, str]:
        return (self.sigma, self.qualname, self.path, self.source)


def _gather_sigmas_no_recurse(obj: Any, path: str, out: List[_SigmaEntry]) -> None:
    """
    Append certified σ entries into `out`.
    Policy:
      1) If module supplies a certified __operator_norm_hint__, use it and STOP.
      2) Else recurse into children.
    """
    hint = _try_operator_norm_hint(obj)
    if hint is not None:
        out.append(_SigmaEntry(hint, type(obj).__name__, path, "hint"))
        return

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _gather_sigmas_no_recurse(v, f"{path}[{i}]", out)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _gather_sigmas_no_recurse(v, f"{path}.{k}", out)
        return
    if isinstance(obj, eqx.Module):
        for k, v in vars(obj).items():
            _gather_sigmas_no_recurse(v, f"{path}.{k}", out)
        return


def _default_predict_logits(
    model: eqx.Module, state: Any, X: jnp.ndarray, key: Optional[jr.KeyArray] = None
) -> jnp.ndarray:
    """
    Generic, library-free prediction: vmapped inference, returns raw outputs/logits.
    If your project uses a special head-selection, pass your `predict_fn` to
    compute_diagnostics to mirror training-time behavior.
    """
    if key is None:
        key = jr.PRNGKey(0)
    inference_model = eqx.nn.inference_mode(model)

    def single(x, k):
        out, _ = inference_model(x, k, state)
        return out

    keys = jr.split(key, X.shape[0])
    return jax.vmap(single, in_axes=(0, 0))(X, keys)


def _call_predict(
    predict_fn: Callable[..., jnp.ndarray],
    model: eqx.Module,
    state: Any,
    X: jnp.ndarray,
    key: Optional[jr.KeyArray],
) -> jnp.ndarray:
    """Call predict_fn with (model,state,X,key) if possible; else without key."""
    try:
        return predict_fn(model, state, X, key)
    except TypeError:
        return predict_fn(model, state, X)


# =============================================================================
# Margins & stats
# =============================================================================


def multiclass_margins(logits: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Margins for classification:
      - Binary: logits (N,), or (N,1) → margin = s * logit, with s in {-1,+1}
                logits (N,2) → margin = z_true - max z_other
      - Multiclass (N,K) → margin = z_true - max_{j≠true} z_j
    """
    logits = jnp.asarray(logits)
    y = jnp.asarray(y)

    if logits.ndim == 1:
        s = 2.0 * y.astype(jnp.float32) - 1.0
        return s * logits

    if logits.ndim == 2 and logits.shape[-1] == 1:
        z = jnp.squeeze(logits, -1)
        s = 2.0 * y.astype(jnp.float32) - 1.0
        return s * z

    if logits.ndim == 2:
        K = logits.shape[-1]
        if K == 2:
            true = logits[jnp.arange(logits.shape[0]), y.astype(jnp.int32)]
            other = jnp.where(y.astype(bool), logits[:, 0], logits[:, 1])
            return true - other
        else:
            true = logits[jnp.arange(logits.shape[0]), y.astype(jnp.int32)]
            mask = jnp.eye(K, dtype=bool)[y.astype(jnp.int32)]
            neg_inf = jnp.full_like(logits, -jnp.inf)
            masked = jnp.where(mask, neg_inf, logits)
            best_other = jnp.max(masked, axis=-1)
            return true - best_other

    raise ValueError(
        f"Expected (N,), (N,1), or (N,K) logits; got {tuple(logits.shape)}"
    )


def _summary_stats(arr: jnp.ndarray) -> Dict[str, float]:
    a = jnp.asarray(arr, dtype=jnp.float32)
    q = jnp.quantile(a, jnp.array([0.05, 0.5, 0.95], dtype=jnp.float32))
    return {
        "mean": float(jnp.mean(a)),
        "std": float(jnp.std(a)),
        "q05": float(q[0]),
        "q50": float(q[1]),
        "q95": float(q[2]),
    }


def gamma_at_eps(margins: jnp.ndarray, eps: float) -> float:
    return float(jnp.mean(margins <= jnp.float32(eps)))


# =============================================================================
# Main API
# =============================================================================


def compute_diagnostics(
    model: eqx.Module,
    state: Any,
    *,
    margin_subset: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    predict_fn: Optional[
        Callable[[eqx.Module, Any, jnp.ndarray, Optional[jr.KeyArray]], jnp.ndarray]
    ] = None,
    key: Optional[jr.KeyArray] = None,
    lipschitz_upper_bound_fn: Callable[..., Union[float, jnp.ndarray]],
    include_coverage: bool = False,
    include_sigma_lists: bool = False,
    robust_eps: Sequence[float] = (
        0.01,
        0.02,
        0.03,
    ),  # input-space ε (ℓ2) for robust acc/err
    gamma_eps: Sequence[float] = (0.0, 0.1, 0.5),  # margin thresholds ε for γ(ε)=P[m≤ε]
) -> Dict[str, Any]:
    """
    Certified-only diagnostics.

    robust_eps: sequence of input-space ℓ2 radii ε at which to report
        robust_acc(ε) = mean(r ≥ ε) and robust_err(ε) = 1 - robust_acc(ε),
        where r = max(margin, 0) / (2L). Set to () to disable.
    gamma_eps: sequence of margin thresholds (in logit units) at which to report
        γ(ε) = P[m ≤ ε]. Set to () to disable.
    """

    # 1) Gather certified per-layer hints (for optional reporting only).
    sig_entries: List[_SigmaEntry] = []
    _gather_sigmas_no_recurse(model, path="model", out=sig_entries)
    covered_cert = [e for e in sig_entries if e.source == "hint"]

    # 2) Global certified L (required)
    try:
        L_val = float(lipschitz_upper_bound_fn(model, state))
    except TypeError:
        L_val = float(lipschitz_upper_bound_fn(model))  # allow (model,) signature
    if not (math.isfinite(L_val) and L_val > 0):
        raise ValueError(
            "Certified global bound function returned a non-finite or non-positive value."
        )
    out: Dict[str, Any] = {
        "skipaware_lipschitz_bound": float(L_val),
        "L_source": "override",  # explicitly from the certified override
        "L_clamped_to_one": False,
        "L_raw_override": float(L_val),
    }

    # Optional coverage & lists
    if include_coverage:
        n_candidates = len(sig_entries)
        n_cert = len(covered_cert)
        coverage_pct = 100.0 * (n_cert / max(1, n_candidates))
        out.update(
            {
                "layer_coverage_pct": float(coverage_pct),
                "n_layers_covered": int(n_cert),
                "n_layers_considered": int(n_candidates),
            }
        )
    if include_sigma_lists:
        out.update(
            {
                "cert_only_sigmas": [
                    e.as_tuple()
                    for e in sorted(covered_cert, key=lambda t: t.sigma, reverse=True)
                ],
            }
        )

    # 3) Margins & certified radii (if data provided)
    if margin_subset is not None:
        Xs, Ys = margin_subset
        logits = (
            _default_predict_logits(model, state, Xs, key)
            if predict_fn is None
            else _call_predict(predict_fn, model, state, Xs, key)
        )

        def _margins_from_logits(
            logits_arr: jnp.ndarray, y_arr: jnp.ndarray
        ) -> jnp.ndarray:
            if logits_arr.ndim >= y_arr.ndim + 1 and logits_arr.shape[-1] >= 2:
                K = logits_arr.shape[-1]
                y_oh = jnp.take_along_axis(
                    logits_arr, y_arr[..., None], axis=-1
                ).squeeze(-1)
                z_max_others = jnp.max(
                    jnp.where(
                        jax.nn.one_hot(y_arr, K, dtype=logits_arr.dtype) > 0,
                        -jnp.inf,
                        logits_arr,
                    ),
                    axis=-1,
                )
                return (y_oh - z_max_others).astype(jnp.float32)
            z = logits_arr
            if z.ndim == y_arr.ndim + 1 and z.shape[-1] == 1:
                z = z.squeeze(-1)
            sgn = 2.0 * y_arr.astype(z.dtype) - 1.0
            return (sgn * z).astype(jnp.float32)

        margins = _margins_from_logits(logits, Ys)
        mstats = _summary_stats(margins)

        L_safe = jnp.float32(max(L_val, 1e-12))
        norm_margins = margins / L_safe
        nstats = _summary_stats(norm_margins)

        radii = jnp.maximum(margins, 0.0) / (2.0 * L_safe)
        rstats = _summary_stats(radii)

        out.update(
            {
                "margins_mean": mstats["mean"],
                "margins_std": mstats["std"],
                "margins_q05": mstats["q05"],
                "margins_q50": mstats["q50"],
                "margins_q95": mstats["q95"],
                "margins_frac_nonpos": float(jnp.mean(margins <= 0)),
                "norm_margins_mean": nstats["mean"],
                "norm_margins_std": nstats["std"],
                "norm_margins_q05": nstats["q05"],
                "norm_margins_q50": nstats["q50"],
                "norm_margins_q95": nstats["q95"],
                "cert_radius_lb_mean": rstats["mean"],
                "cert_radius_lb_q05": rstats["q05"],
                "cert_radius_lb_q50": rstats["q50"],
                "cert_radius_lb_q95": rstats["q95"],
                "_margins_array": margins,
                "_norm_margins_array": norm_margins,
                "_radii_array": radii,
                "_logits_array": jnp.asarray(logits),
            }
        )

        # Margin CDF γ(ε) = P[m ≤ ε] at the requested margin thresholds (ε in margin units)
        if gamma_eps:
            eps_grid = tuple(float(e) for e in gamma_eps)
            out["margin_eps_grid"] = np.asarray(eps_grid, dtype=float)
            for e in eps_grid:
                out[f"gamma@{e}"] = gamma_at_eps(margins, e)

        # Accuracy estimate (for convenience): 1 - γ(0)
        gamma0 = out.get("gamma@0.0", None)
        if gamma0 is None:
            gamma0 = gamma_at_eps(margins, 0.0)
        out["acc_estimate"] = float(1.0 - gamma0)

        # Robust accuracy/error at input-space radii ε (ℓ2), via radii
        if robust_eps:
            r_grid = tuple(float(e) for e in robust_eps)
            out["robust_eps_grid"] = np.asarray(r_grid, dtype=float)
            for e in r_grid:
                rob_acc = float(jnp.mean(radii >= jnp.float32(e)))
                out[f"robust_acc@eps={e}"] = rob_acc
                out[f"robust_err@eps={e}"] = float(1.0 - rob_acc)

    return out


# =============================================================================
# Presentation helpers
# =============================================================================


def pretty_print_diagnostics(
    diag: Dict[str, Any],
    *,
    topk_layer_hints: int = 10,
) -> str:
    """
    Render a certified-only diagnostics report as a string.

    Prints ONLY certified quantities:
      • Global Lipschitz upper bound L
      • (Optional) coverage of certified per-layer hints
      • (Optional) summary stats for margins, normalized margins, and certified radii
      • (Optional) γ(ε)=P[m ≤ ε] at margin-threshold grid (ε in margin units)
      • (Optional) robust_acc(ε), robust_err(ε) at input-space ε radii (ℓ2)
    """

    def _fmt(x: Any) -> str:
        try:
            return f"{float(x):.6g}"
        except Exception:
            return str(x)

    def _have(*keys: str) -> bool:
        return all(k in diag for k in keys)

    lines: List[str] = []
    lines.append("— Certified diagnostics (ℓ2) —")

    # Global L (required)
    if "skipaware_lipschitz_bound" not in diag:
        raise KeyError("pretty_print_diagnostics: missing 'skipaware_lipschitz_bound'.")
    L = float(diag["skipaware_lipschitz_bound"])
    L_src = diag.get("L_source", "certified")
    lines.append(f"L (global certified upper bound): {_fmt(L)} [{L_src}]")
    if bool(diag.get("L_clamped_to_one", False)):
        lines.append("Note: L was clamped to ≥ 1.0 for normalization.")
    if _have("layer_coverage_pct", "n_layers_covered", "n_layers_considered"):
        lines.append(
            "Certified layer-hint coverage: "
            f"{int(diag['n_layers_covered'])}/{int(diag['n_layers_considered'])} "
            f"({_fmt(diag['layer_coverage_pct'])}%)"
        )

    # Margins & normalized margins & certified radii (optional block)
    have_margins = _have(
        "margins_mean",
        "margins_std",
        "margins_q05",
        "margins_q50",
        "margins_q95",
        "margins_frac_nonpos",
    )
    have_norm_margins = _have(
        "norm_margins_mean",
        "norm_margins_std",
        "norm_margins_q05",
        "norm_margins_q50",
        "norm_margins_q95",
    )
    have_radii = _have(
        "cert_radius_lb_mean",
        "cert_radius_lb_q05",
        "cert_radius_lb_q50",
        "cert_radius_lb_q95",
    )

    if have_margins or have_norm_margins or have_radii:
        lines.append("")
    if have_margins:
        lines.append("Margins  m(x,y) = z_y − max_{j≠y} z_j :")
        lines.append(
            "  "
            f"mean={_fmt(diag['margins_mean'])}  "
            f"std={_fmt(diag['margins_std'])}  "
            f"q05={_fmt(diag['margins_q05'])}  "
            f"q50={_fmt(diag['margins_q50'])}  "
            f"q95={_fmt(diag['margins_q95'])}  "
            f"frac≤0={_fmt(diag['margins_frac_nonpos'])}"
        )
        gamma0 = float(diag.get("gamma@0.0", diag["margins_frac_nonpos"]))
        lines.append(
            f"  ⇒ error ≈ γ(0) = {_fmt(gamma0)};   accuracy ≈ 1−γ(0) = {_fmt(1 - gamma0)}"
        )

    if have_norm_margins:
        lines.append("Normalized margins  m/L  (with L above):")
        lines.append(
            "  "
            f"mean={_fmt(diag['norm_margins_mean'])}  "
            f"std={_fmt(diag['norm_margins_std'])}  "
            f"q05={_fmt(diag['norm_margins_q05'])}  "
            f"q50={_fmt(diag['norm_margins_q50'])}  "
            f"q95={_fmt(diag['norm_margins_q95'])}"
        )

    if have_radii:
        lines.append("Certified radii  r(x) = max(m,0)/(2L) :")
        lines.append(
            "  "
            f"mean={_fmt(diag['cert_radius_lb_mean'])}  "
            f"q05={_fmt(diag['cert_radius_lb_q05'])}  "
            f"q50={_fmt(diag['cert_radius_lb_q50'])}  "
            f"q95={_fmt(diag['cert_radius_lb_q95'])}"
        )

    # Margin CDF: γ(ε) = P[m ≤ ε] at the explicit ε grid (margin units)
    # Try to read grid; if missing, infer from keys.
    gamma_map: Dict[float, float] = {}
    for k, v in diag.items():
        if isinstance(k, str) and k.startswith("gamma@"):
            try:
                gamma_map[float(k.split("@", 1)[1])] = float(v)
            except Exception:
                pass
    eps_grid = None
    if "margin_eps_grid" in diag:
        try:
            eps_grid = [float(e) for e in np.asarray(diag["margin_eps_grid"]).ravel()]
        except Exception:
            eps_grid = None
    if eps_grid is None and gamma_map:
        eps_grid = sorted(gamma_map.keys())

    if eps_grid:
        lines.append("Margin CDF:  γ(ε) = P[m ≤ ε]   (ε in margin units)")
        lines.append("  ε-grid: " + ", ".join(_fmt(e) for e in eps_grid))
        lines.append(
            "  "
            + "  ".join(
                f"γ({ _fmt(e) })={ _fmt(gamma_map.get(e, float('nan'))) }"
                for e in eps_grid
            )
        )

    # Robust acc/err at input ε (ℓ2), with explicit ε grid
    rob_acc_map: Dict[float, float] = {}
    rob_err_map: Dict[float, float] = {}
    for k, v in diag.items():
        if isinstance(k, str) and k.startswith("robust_acc@eps="):
            try:
                rob_acc_map[float(k.split("=", 1)[1])] = float(v)
            except Exception:
                pass
        if isinstance(k, str) and k.startswith("robust_err@eps="):
            try:
                rob_err_map[float(k.split("=", 1)[1])] = float(v)
            except Exception:
                pass
    rgrid = None
    if "robust_eps_grid" in diag:
        try:
            rgrid = [float(e) for e in np.asarray(diag["robust_eps_grid"]).ravel()]
        except Exception:
            rgrid = None
    if rgrid is None and (rob_acc_map or rob_err_map):
        rgrid = sorted(set(list(rob_acc_map.keys()) + list(rob_err_map.keys())))

    if rgrid:
        lines.append(
            "Robust metrics (input space, ℓ2):  robust_acc(ε)=P[r≥ε],  robust_err(ε)=1−robust_acc(ε)"
        )
        lines.append("  ε-grid: " + ", ".join(_fmt(e) for e in rgrid))
        if rob_acc_map:
            lines.append(
                "  "
                + "  ".join(
                    f"robust_acc({ _fmt(e) })={ _fmt(rob_acc_map.get(e, float('nan'))) }"
                    for e in rgrid
                )
            )
        if rob_err_map:
            lines.append(
                "  "
                + "  ".join(
                    f"robust_err({ _fmt(e) })={ _fmt(rob_err_map.get(e, float('nan'))) }"
                    for e in rgrid
                )
            )

    # Top-k certified hint entries (not used to form L; for inspection only)
    if topk_layer_hints > 0 and "cert_only_sigmas" in diag and diag["cert_only_sigmas"]:
        entries = list(diag["cert_only_sigmas"])
        try:
            entries.sort(key=lambda t: float(t[0]), reverse=True)
        except Exception:
            entries.sort(key=lambda t: str(t[0]), reverse=True)
        k = min(topk_layer_hints, len(entries))
        if k > 0:
            lines.append("")
            lines.append(
                f"Top-{k} certified per-layer operator norms (hints; not used to form L):"
            )
            for i in range(k):
                e = entries[i]
                sigma = e[0] if len(e) > 0 else None
                type_name = e[1] if len(e) > 1 else "<?>"
                path = e[2] if len(e) > 2 else "<?>"
                lines.append(f"  {i+1:>2}. σ={_fmt(sigma)}  {type_name}  @ {path}")

    lines.append("")
    lines.append(
        "Cheat sheet: m = z_y − max_{j≠y} z_j.  r = max(m,0)/(2L).  robust_acc(ε)=P[r≥ε]=1−P[m≤2Lε].  γ(ε)=P[m≤ε]."
    )
    return "\n".join(lines)


def plot_margin_distribution(
    d: Dict[str, Any],
    *,
    bins: int = 60,
    which: Tuple[str, ...] = ("raw", "normalized", "radius"),
    show: bool = True,
):
    """
    Plot distributions. `which` can be any subset of:
      - "raw":        raw margins
      - "normalized": margins divided by L_used (or L_used*||x|| if you added that option)
      - "radius":     certified radius lower bound m_+/(2L)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    avail = []
    if "raw" in which and "_margins_array" in d:
        avail.append(("raw", jnp.asarray(d["_margins_array"]).astype(float), "Margins"))
    if "normalized" in which and "_norm_margins_array" in d:
        avail.append(
            (
                "normalized",
                jnp.asarray(d["_norm_margins_array"]).astype(float),
                "Normalized margins",
            )
        )
    if "radius" in which and "_radii_array" in d:
        avail.append(
            (
                "radius",
                jnp.asarray(d["_radii_array"]).astype(float),
                "Certified radius (lower bound)",
            )
        )

    if not avail:
        raise ValueError(
            "Nothing to plot. Call compute_diagnostics(..., margin_subset=(X,y)) first."
        )

    rows = len(avail)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3.5 * rows))
    if rows == 1:
        # Make axes indexable as axes[0, 0], axes[0, 1]
        axes = np.array([axes], dtype=object)

    for i, (_, arr, title) in enumerate(avail):
        # Always use NumPy for plotting
        arr_np = np.asarray(arr)

        # histogram
        axh = axes[i, 0]
        axh.hist(arr_np, bins=bins, density=True)
        axh.axvline(0.0, linestyle="--")
        q05, q50, q95 = np.quantile(arr_np, [0.05, 0.5, 0.95])
        axh.set_title(f"{title}\nq05={q05:.3f}, q50={q50:.3f}, q95={q95:.3f}")
        axh.set_xlabel(title.lower())
        axh.set_ylabel("density")

        # CDF
        axc = axes[i, 1]
        xs = np.linspace(float(arr_np.min()), float(arr_np.max()), 512)
        cdf = np.array([(arr_np <= x).mean() for x in xs])
        axc.plot(xs, cdf)
        axc.axvline(0.0, linestyle="--")
        axc.set_ylim(0, 1)
        if title.startswith("Certified"):
            axc.set_title("CDF")
        else:
            frac_nonpos = float((arr_np <= 0).mean())
            axc.set_title(f"CDF  (frac≤0 = {frac_nonpos:.4f})")
        axc.set_xlabel(title.lower())
        axc.set_ylabel("CDF")

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


# ================================
# NEW: lightweight persistence API
# ================================


def save_diagnostics_npz(
    d: Dict[str, Any],
    path: Union[str, os.PathLike],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist arrays + a few key scalars to a .npz (non-breaking helper)."""
    import numpy as _np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pack = {}
    # Arrays (use empty arrays if missing so files are uniform)
    for k in ("_margins_array", "_norm_margins_array", "_radii_array", "_logits_array"):
        v = d.get(k, None)
        pack[k] = _np.asarray(v) if v is not None else _np.asarray([])

    # Scalars used for context / reproducibility
    for k in (
        "layer_coverage_pct",
        "n_layers_covered",
        "serial_prod_sigma_bound",
        "skipaware_lipschitz_bound",
        "margins_mean",
        "margins_std",
        "margins_q05",
        "margins_q50",
        "margins_q95",
        "norm_margins_mean",
        "norm_margins_std",
        "norm_margins_q05",
        "norm_margins_q50",
        "norm_margins_q95",
        "cert_radius_lb_mean",
        "cert_radius_lb_q05",
        "cert_radius_lb_q50",
        "cert_radius_lb_q95",
    ):
        if k in d and d[k] is not None:
            pack[k] = _np.array(d[k])

    # Optional metadata (turn dict into a tiny JSON string)
    if meta is None:
        meta = {}
    try:
        import json as _json

        pack["_meta_json"] = _np.array(_json.dumps(meta))
    except Exception:
        pass

    _np.savez_compressed(path, **pack)


def load_diagnostics_npz(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Load a previously saved diagnostics .npz into a dict using the same keys."""
    import numpy as _np

    out: Dict[str, Any] = {}
    with _np.load(path, allow_pickle=False) as z:
        for k in z.files:
            out[k] = z[k]
    # Be forgiving: promote numpy scalars to Python floats
    for k, v in list(out.items()):
        if isinstance(v, _np.ndarray) and v.ndim == 0:
            try:
                out[k] = float(v)
            except Exception:
                pass
    # For convenience, also mirror arrays under the original keys (if present)
    for k in ("_margins_array", "_norm_margins_array", "_radii_array", "_logits_array"):
        out.setdefault(k, out.get(k, None))
    return out


def compute_and_save_diagnostics(
    *args,
    save_path: Optional[Union[str, os.PathLike]] = None,
    save_meta: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Thin wrapper: compute_diagnostics(...) then optionally save to .npz.
    This avoids changing compute_diagnostics' signature.
    """
    d = compute_diagnostics(*args, **kwargs)
    if save_path is not None:
        save_diagnostics_npz(d, save_path, meta=save_meta)
    return d


# =================================
# NEW: overlay plotter (multi-runs)
# =================================


def plot_margin_overlays(
    ds: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    *,
    which: Tuple[str, ...] = ("normalized",),
    bins: int = 60,
    show: bool = True,
    # --- new styling knobs ---
    colors: Optional[Sequence[Optional[str]]] = None,  # e.g. ["#000000","#D55E00"]
    linestyles: Optional[Sequence[Optional[str]]] = None,  # e.g. ["-","--"]
    linewidth: float = 1.75,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.5, 3.5),  # width, height PER ROW
    grid: bool = False,
    savepath: Optional[Union[str, os.PathLike]] = None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    assert len(ds) == len(labels), "labels must match number of diagnostics"

    sel = {
        "raw": ("_margins_array", "Margins", "margin"),
        "normalized": (
            "_norm_margins_array",
            "Normalized margins",
            "normalized margin",
        ),
        "radius": ("_radii_array", "Certified radius (lower bound)", "radius"),
    }
    which = tuple(w for w in which if w in sel)
    if not which:
        raise ValueError(
            "`which` must include at least one of: 'raw','normalized','radius'."
        )

    rows = len(which)
    fig, axes = plt.subplots(rows, 2, figsize=(figsize[0], figsize[1] * rows))
    if rows == 1:
        axes = np.array([axes], dtype=object)

    # defaults for colors/linestyles
    if colors is None:
        colors = [None] * len(ds)
    if linestyles is None:
        linestyles = [None] * len(ds)
    assert len(colors) == len(ds) and len(linestyles) == len(ds)

    for i, w in enumerate(which):
        key, row_title, xlabel = sel[w]
        arrs = []
        for d in ds:
            a = d.get(key, None)
            if a is None or (hasattr(a, "size") and a.size == 0):
                arrs.append(np.asarray([]))
            else:
                arrs.append(np.asarray(a).ravel())

        axh, axc = axes[i, 0], axes[i, 1]
        nonempty = [a for a in arrs if a.size]
        if not nonempty:
            raise RuntimeError(
                f"No data found for {w}; did you pass margin_subset and L_fn?"
            )
        lo = min(float(a.min()) for a in nonempty)
        hi = max(float(a.max()) for a in nonempty)
        edges = np.linspace(lo, hi, bins + 1)

        # histogram (step outlines)
        for a, lab, col, ls in zip(arrs, labels, colors, linestyles):
            if a.size:
                axh.hist(
                    a,
                    bins=edges,
                    density=True,
                    histtype="step",
                    label=lab,
                    color=col,
                    linestyle=ls,
                    linewidth=linewidth,
                )
        axh.axvline(0.0, linestyle="--", color="#777777", linewidth=1.0)
        axh.set_title((row_title + " — overlay") if title is None else title)
        axh.set_xlabel(xlabel)
        axh.set_ylabel("density")
        axh.legend()

        # CDF overlay
        xs = np.linspace(lo, hi, 512)
        for a, lab, col, ls in zip(arrs, labels, colors, linestyles):
            if a.size:
                cdf = np.array([(a <= x).mean() for x in xs])
                axc.plot(
                    xs,
                    cdf,
                    label=lab,
                    linestyle=ls,
                    linewidth=linewidth + 0.25,
                    color=col,
                )
        axc.axvline(0.0, linestyle="--", color="#777777", linewidth=1.0)
        axc.set_ylim(0, 1)
        axc.set_title("CDF — overlay")
        axc.set_xlabel(xlabel)
        axc.set_ylabel("CDF")
        axc.legend()

        if grid:
            axh.grid(True, linestyle=":", alpha=0.3)
            axc.grid(True, linestyle=":", alpha=0.3)

    if title is not None and rows > 1:
        fig.suptitle(title, y=1.02, fontsize=12)

    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig, axes
