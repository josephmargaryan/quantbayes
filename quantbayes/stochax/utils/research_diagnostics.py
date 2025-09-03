# quantbayes/stochax/diagnostics/research_diagnostic.py
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


def _sv_max_power_flat(W: jnp.ndarray, iters: int = 2) -> jnp.ndarray:
    """Approximate top singular value by flattening to (out, -1)."""
    W2 = jnp.reshape(W, (W.shape[0], -1))
    v = jnp.ones((W2.shape[1],), dtype=W2.dtype)
    v = v / (jnp.linalg.norm(v) + 1e-12)

    def body(_, v):
        u = W2 @ v
        u = u / (jnp.linalg.norm(u) + 1e-12)
        v = W2.T @ u
        v = v / (jnp.linalg.norm(v) + 1e-12)
        return v

    v = jax.lax.fori_loop(0, max(iters, 1), body, v)
    return jnp.linalg.norm(W2 @ v)


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


def _prod_sigmas_stable(sigmas: Sequence[float]) -> float:
    if not sigmas:
        return 1.0
    logs = jnp.array(
        [jnp.log(jnp.clip(s, 1e-12, 1e12)) for s in sigmas], dtype=jnp.float32
    )
    return float(jnp.exp(jnp.sum(logs)))


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
) -> Dict[str, Any]:
    """
    Certified-only diagnostics.
      - Requires a certified global Lipschitz bound function (BN-aware).
      - No empirical layer estimates; no heuristic fallbacks.
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
    }
    out["L_raw_override"] = float(L_val)

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
                "gamma@0.0": gamma_at_eps(margins, 0.0),
                "gamma@0.1": gamma_at_eps(margins, 0.1),
                "gamma@0.5": gamma_at_eps(margins, 0.5),
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

    Expects the dictionary returned by `compute_diagnostics(...)` (the certified-only
    version). This function prints ONLY certified quantities:
      - Global Lipschitz upper bound L (skip/parallel-aware, BN running stats)
      - (Optional) coverage of certified per-layer hints
      - (Optional) summary stats for margins, normalized margins, and certified radii
      - (Optional) gamma@ε entries if present
      - (Optional) top-k certified per-layer operator norms from hints (for inspection only)

    No heuristic/empirical layer estimates are displayed. No σ-product lines are shown.

    Args:
        diag: Output dict from `compute_diagnostics(...)`.
        topk_layer_hints: How many certified per-layer hint entries to show (0 disables).

    Returns:
        A single formatted string (no side effects).
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
        raise KeyError(
            "pretty_print_diagnostics: missing 'skipaware_lipschitz_bound' in diagnostics dict."
        )
    L = diag["skipaware_lipschitz_bound"]
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
        lines.append("Margins (y − max other):")
        lines.append(
            "  "
            f"mean={_fmt(diag['margins_mean'])}  "
            f"std={_fmt(diag['margins_std'])}  "
            f"q05={_fmt(diag['margins_q05'])}  "
            f"q50={_fmt(diag['margins_q50'])}  "
            f"q95={_fmt(diag['margins_q95'])}  "
            f"frac≤0={_fmt(diag['margins_frac_nonpos'])}"
        )

    if have_norm_margins:
        lines.append("Normalized margins (by L):")
        lines.append(
            "  "
            f"mean={_fmt(diag['norm_margins_mean'])}  "
            f"std={_fmt(diag['norm_margins_std'])}  "
            f"q05={_fmt(diag['norm_margins_q05'])}  "
            f"q50={_fmt(diag['norm_margins_q50'])}  "
            f"q95={_fmt(diag['norm_margins_q95'])}"
        )

    if have_radii:
        lines.append("Certified radii (lower bounds):")
        lines.append(
            "  "
            f"mean={_fmt(diag['cert_radius_lb_mean'])}  "
            f"q05={_fmt(diag['cert_radius_lb_q05'])}  "
            f"q50={_fmt(diag['cert_radius_lb_q50'])}  "
            f"q95={_fmt(diag['cert_radius_lb_q95'])}"
        )

    # gamma@ε lines if present (e.g., 'gamma@0.0', 'gamma@0.1', ...)
    gamma_items = [
        (k, diag[k])
        for k in diag.keys()
        if isinstance(k, str) and k.startswith("gamma@")
    ]
    if gamma_items:

        def _parse_eps(k: str) -> float:
            try:
                return float(k.split("@", 1)[1])
            except Exception:
                return float("inf")

        gamma_items.sort(key=lambda kv: _parse_eps(kv[0]))
        lines.append("Certified γ@ε (fraction with margin ≤ ε):")  # <-- fixed label
        lines.append("  " + "  ".join(f"{k}={_fmt(v)}" for k, v in gamma_items))

    # Top-k certified hint entries (not used to form L; for inspection only)
    if topk_layer_hints > 0 and "cert_only_sigmas" in diag and diag["cert_only_sigmas"]:
        entries = list(diag["cert_only_sigmas"])
        # Expect tuples of (sigma, type_name, path, source)
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
        "All values are computed with certified bounds only (no heuristics). "
        "BatchNorm uses running statistics."
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
    which: Tuple[str, ...] = ("normalized",),  # default to the most comparable view
    bins: int = 60,
    show: bool = True,
):
    """
    Overlay multiple margin distributions (and their CDFs) in a single figure.

    Parameters
    ----------
    ds : list of diagnostics dicts            (e.g., from compute_diagnostics or load_diagnostics_npz)
    labels : list of legend labels            (same length as ds)
    which : subset of {"raw","normalized","radius"}; multiple -> stacked rows
    bins : histogram bins
    show : call plt.show()

    Notes
    -----
    • For fair overlays on 'normalized' margins, ensure each diagnostics used the
      *same* certified global bound construction (same conv_mode / input_shape).
    • This function is additive and does not affect existing APIs.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    assert len(ds) == len(labels), "labels must match number of diagnostics"

    # Map selector -> (key, pretty title, x-label)
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
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3.5 * rows))
    if rows == 1:
        axes = np.array([axes], dtype=object)

    for i, w in enumerate(which):
        key, title, xlabel = sel[w]
        # Gather arrays for this view
        arrs = []
        for d in ds:
            a = d.get(key, None)
            if a is None or (hasattr(a, "size") and a.size == 0):
                arrs.append(np.asarray([]))
            else:
                arrs.append(np.asarray(a).ravel())

        # Histogram overlay
        axh = axes[i, 0]
        common_min = min((a.min() for a in arrs if a.size), default=0.0)
        common_max = max((a.max() for a in arrs if a.size), default=1.0)
        edges = np.linspace(common_min, common_max, bins + 1)

        for a, lab in zip(arrs, labels):
            if a.size:
                axh.hist(a, bins=edges, density=True, histtype="step", label=lab)
        axh.axvline(0.0, linestyle="--")
        axh.set_title(title + " — overlay")
        axh.set_xlabel(xlabel)
        axh.set_ylabel("density")
        axh.legend()

        # CDF overlay
        axc = axes[i, 1]
        xs = np.linspace(common_min, common_max, 512)
        for a, lab in zip(arrs, labels):
            if a.size:
                cdf = np.array([(a <= x).mean() for x in xs])
                axc.plot(xs, cdf, label=lab)
        axc.axvline(0.0, linestyle="--")
        axc.set_ylim(0, 1)
        axc.set_title("CDF — overlay")
        axc.set_xlabel(xlabel)
        axc.set_ylabel("CDF")
        axc.legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes
