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


def _gather_sigmas_no_recurse(
    obj: Any, path: str, out: List[_SigmaEntry], allow_empirical: bool
) -> None:
    """
    Append σ entries into `out`.
    Policy:
      1) If module supplies a certified __operator_norm_hint__, use it and STOP.
      2) Otherwise, if allow_empirical=True, we MAY add empirical estimates for
         simple linear/conv layers (flatten + quick power-iter), then STOP.
      3) Else, recurse into children.
    """
    # 1) Certified hint wins and is atomic (do not dive deeper).
    hint = _try_operator_norm_hint(obj)
    if hint is not None:
        out.append(_SigmaEntry(hint, type(obj).__name__, path, "hint"))
        return

    # 2) Optional empirical fallback (NOT certified; off by default).
    if allow_empirical:
        try:
            import equinox.nn as nn  # local import to avoid hard coupling
        except Exception:
            nn = None

        if nn is not None:
            # Linear
            if isinstance(obj, getattr(nn, "Linear", ())) and hasattr(obj, "weight"):
                s = _sv_max_power_flat(obj.weight, iters=2)
                out.append(_SigmaEntry(float(s), type(obj).__name__, path, "empirical"))
                return

            # Conv / ConvTranspose families -> treat as flattened kernel proxy (empirical)
            conv_like = tuple(
                getattr(nn, k)
                for k in ("Conv", "Conv1d", "Conv2d", "Conv3d")
                if hasattr(nn, k)
            ) + tuple(
                getattr(nn, k)
                for k in (
                    "ConvTranspose",
                    "ConvTranspose1d",
                    "ConvTranspose2d",
                    "ConvTranspose3d",
                )
                if hasattr(nn, k)
            )
            if isinstance(obj, conv_like) and hasattr(obj, "weight"):
                s = _sv_max_power_flat(obj.weight, iters=2)
                out.append(_SigmaEntry(float(s), type(obj).__name__, path, "empirical"))
                return

    # 3) Recurse
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _gather_sigmas_no_recurse(v, f"{path}[{i}]", out, allow_empirical)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _gather_sigmas_no_recurse(v, f"{path}.{k}", out, allow_empirical)
        return
    if isinstance(obj, eqx.Module):
        for k, v in vars(obj).items():
            _gather_sigmas_no_recurse(v, f"{path}.{k}", out, allow_empirical)
        return
    # Leaf and no σ → nothing to add


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
    compute_empirical: bool = False,  # if True, allow non-certified σ for uncovered layers
    forbid_unknown_as_one: bool = True,  # if True and uncovered layers exist, clamp L>=1
    lipschitz_upper_bound_fn: Optional[
        Callable[[eqx.Module], Union[float, jnp.ndarray]]
    ] = None,
) -> Dict[str, Any]:
    """
    Compute model diagnostics:
      - layer coverage (fraction of linear-like ops with certified hints)
      - σ product bound (certified-only, no double-count via wrapper)
      - optional margins, Lipschitz-normalized margins, and certified radius summaries

    Notes
    -----
    • The σ product is a *serial* multiply of certified hints only.
      For true skip/parallel structure (e.g., residual blocks), prefer passing a
      global bound via `lipschitz_upper_bound_fn`, which becomes the L used for
      normalized margins and certified radii.
    """
    # 1) Gather σ entries — certified hints are atomic (no recursion under them).
    sig_entries: List[_SigmaEntry] = []
    _gather_sigmas_no_recurse(
        model, path="model", out=sig_entries, allow_empirical=compute_empirical
    )

    covered_cert = [e for e in sig_entries if e.source == "hint"]
    covered_emp = [e for e in sig_entries if e.source == "empirical"]

    n_candidates = len(sig_entries)
    n_cert = len(covered_cert)
    coverage = 100.0 * (n_cert / max(1, n_candidates))

    # Certified-only σ product (stable)
    cert_sigmas = [e.sigma for e in covered_cert]
    serial_prod_sigma_bound = _prod_sigmas_stable(cert_sigmas) if cert_sigmas else 1.0

    # Optional override (e.g., conv-aware global L)
    L_from_override: Optional[float] = None
    if lipschitz_upper_bound_fn is not None:
        try:
            L_from_override = float(lipschitz_upper_bound_fn(model))
            if not (math.isfinite(L_from_override) and L_from_override > 0):
                L_from_override = None
        except Exception:
            L_from_override = None

    # Safe Lipschitz used for normalized margins + radius:
    # prefer override; else clamp σ-product to >= 1 if forbid_unknown_as_one
    if L_from_override is not None:
        L_used = float(L_from_override)
    else:
        L_used = float(serial_prod_sigma_bound)
        if forbid_unknown_as_one:
            L_used = float(max(1.0, L_used))

    out: Dict[str, Any] = {
        "layer_coverage_pct": float(coverage),
        "n_layers_covered": int(n_cert),
        "serial_prod_sigma_bound": float(serial_prod_sigma_bound),
        "skipaware_lipschitz_bound": float(L_used),
        "sigmas": [
            e.as_tuple()
            for e in sorted(sig_entries, key=lambda t: t.sigma, reverse=True)
        ],
        "cert_only_sigmas": [
            e.as_tuple()
            for e in sorted(covered_cert, key=lambda t: t.sigma, reverse=True)
        ],
        "empirical_sigmas": [
            e.as_tuple()
            for e in sorted(covered_emp, key=lambda t: t.sigma, reverse=True)
        ],
    }

    # 2) Margins & certified radii (if data provided)
    if margin_subset is not None:
        Xs, Ys = margin_subset

        # Choose prediction path
        if predict_fn is None:
            logits = _default_predict_logits(model, state, Xs, key=key)
        else:
            logits = _call_predict(predict_fn, model, state, Xs, key)

        # Raw margins
        margins = multiclass_margins(logits, Ys).astype(jnp.float32)
        mstats = _summary_stats(margins)

        # Normalized margins (by L_used)
        norm_margins = margins / jnp.float32(max(L_used, 1e-12))
        nstats = _summary_stats(norm_margins)

        # Certified L2 radius lower bound: r >= margin_+ / (2L)
        radii = jnp.maximum(margins, 0.0) / (2.0 * jnp.float32(max(L_used, 1e-12)))
        rstats = _summary_stats(radii)

        out.update(
            {
                # Raw margins
                "margins_mean": mstats["mean"],
                "margins_std": mstats["std"],
                "margins_q05": mstats["q05"],
                "margins_q50": mstats["q50"],
                "margins_q95": mstats["q95"],
                "margins_frac_nonpos": float(jnp.mean(margins <= 0)),
                "gamma@0.0": gamma_at_eps(margins, 0.0),
                "gamma@0.1": gamma_at_eps(margins, 0.1),
                "gamma@0.5": gamma_at_eps(margins, 0.5),
                # Normalized margins
                "norm_margins_mean": nstats["mean"],
                "norm_margins_std": nstats["std"],
                "norm_margins_q05": nstats["q05"],
                "norm_margins_q50": nstats["q50"],
                "norm_margins_q95": nstats["q95"],
                # Certified radius (lower bound)
                "cert_radius_lb_mean": rstats["mean"],
                "cert_radius_lb_q05": rstats["q05"],
                "cert_radius_lb_q50": rstats["q50"],
                "cert_radius_lb_q95": rstats["q95"],
                # Arrays for plotting
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


def pretty_print_diagnostics(d: Dict[str, Any]) -> None:
    print("— diagnostics (CERTIFIED where marked) —")
    cov = d.get("layer_coverage_pct", 0.0)
    n_cov = d.get("n_layers_covered", 0)
    print(f"layer coverage: {cov:.1f}%  ({n_cov} covered)")

    sp = d.get("serial_prod_sigma_bound", None)
    if sp is not None:
        print(f"serial_prod_sigma_bound: {sp:g}")

    L = d.get("skipaware_lipschitz_bound", None)
    if L is not None:
        print(f"skipaware_lipschitz_bound: {L:g}")

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
            key = f"gamma@{eps}"
            if key in d:
                print(f"{key}: {d[key]:.4g}")

    if "norm_margins_mean" in d:
        # scientific notation so tiny values don't print as 0.000
        print(
            "normalized margins: mean={:.3e}, std={:.3e}, q05={:.3e}, q50={:.3e}, q95={:.3e}".format(
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

    cert = d.get("cert_only_sigmas", [])
    emp = d.get("empirical_sigmas", [])

    def _fmt(entries: List[Tuple[float, str, str, str]]) -> List[str]:
        rows = []
        for s, qn, pth, src in entries:
            tag = "[hint]" if src == "hint" else "[emp]"
            rows.append(f"  {s:.4g}      {qn:<24} @ {pth:<30} {tag}")
        return rows

    if cert:
        print("top-σ (certified) layers:")
        for line in _fmt(cert[:8]):
            print(line)
    if emp:
        print("top-σ (empirical) layers:")
        for line in _fmt(emp[:8]):
            print(line)


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
