import numpy as np
import jax
import numpy as np
import jax.numpy as jnp
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from numpyro.infer import Predictive

__all__ = [
    "one_hot",
    "predictive_entropy_from_proba",
    "avg_nll_from_proba",
    "brier_from_proba",
    "ece_mce_from_proba",
    "auroc_from_proba",
    "fpr95_from_proba",
    "save_pred_entropy_from_arrays_trained_minimal",
    "plot_id_vs_ood_entropy_kde",
    "plot_id_vs_ood_entropy_kde_full_and_zoom",
    "save_entropy_kde_pair",
    "_get_spectral_latents",
    "save_spectral_posterior_1d",
]


# ----- helpers -----
def one_hot(y, num_classes):
    y = np.asarray(y)
    oh = np.zeros((y.size, num_classes), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh


def predictive_entropy_from_proba(p, eps=1e-12):
    p = np.clip(np.asarray(p), eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def avg_nll_from_proba(p, y, eps=1e-12):
    p = np.clip(np.asarray(p), eps, 1.0)
    return -np.mean(np.log(p[np.arange(len(y)), y]))


def brier_from_proba(p, y, num_classes):
    y1h = one_hot(y, num_classes)
    diff = np.asarray(p) - y1h
    return np.mean(np.sum(diff * diff, axis=-1))


def ece_mce_from_proba(p, y, n_bins=15):
    p = np.asarray(p)
    y = np.asarray(y)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        gap = abs(bin_conf - bin_acc)
        ece += (mask.mean()) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def auroc_from_proba(p_id: np.ndarray, p_ood: np.ndarray) -> float:
    """
    AUROC for ID vs OOD using -entropy as the score (higher = more ID-like).
    """
    H_id = predictive_entropy_from_proba(p_id)
    H_ood = predictive_entropy_from_proba(p_ood)
    s_id, s_ood = -H_id, -H_ood
    labels = np.concatenate(
        [np.ones_like(s_id, dtype=int), np.zeros_like(s_ood, dtype=int)]
    )
    scores = np.concatenate([s_id, s_ood])
    return float(roc_auc_score(labels, scores))


def fpr95_from_proba(p_id: np.ndarray, p_ood: np.ndarray) -> float:
    """
    FPR at 95% TPR using -entropy as the score (higher = more ID-like).
    Threshold is the 5th percentile of ID scores.
    """
    H_id = predictive_entropy_from_proba(p_id)
    H_ood = predictive_entropy_from_proba(p_ood)
    s_id, s_ood = -H_id, -H_ood
    thr = np.quantile(s_id, 0.05)  # TPR ≈ 95% on ID
    return float((s_ood >= thr).mean())


def _predictive_entropy_from_proba(p, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def save_pred_entropy_from_arrays_trained_minimal(
    clf,
    out_path: str,
    *,
    X_id_test: np.ndarray,
    y_id_test: np.ndarray,
    X_ood_test: np.ndarray,
    model_tag: str = "model",
):
    """
    No training here. Uses an already-fitted Bayesian classifier `clf`
    with predict_proba(X) -> (N, C).

    Saves NPZ with keys: H_id, H_ood, y_id, model_tag
    """
    X_id_test = np.asarray(X_id_test)
    X_ood_test = np.asarray(X_ood_test)
    y_id_test = np.asarray(y_id_test).astype(np.int64)

    # Predictive probabilities
    p_id = np.asarray(clf.predict_proba(X_id_test))
    p_ood = np.asarray(clf.predict_proba(X_ood_test))

    # Predictive entropy
    H_id = _predictive_entropy_from_proba(p_id).astype(np.float32)
    H_ood = _predictive_entropy_from_proba(p_ood).astype(np.float32)

    # Save minimal payload (what your plot needs)
    np.savez_compressed(
        out_path,
        model_tag=np.array(model_tag),
        H_id=H_id,
        H_ood=H_ood,
        y_id=y_id_test,
    )
    print(f"[saved] {out_path} | tag={model_tag} | n_id={H_id.size} n_ood={H_ood.size}")


def _reflect_kde(samples, xs, lower=0.0, upper=None, bw="scott"):
    x = np.asarray(samples, dtype=np.float64).ravel()
    if upper is None:
        upper = float(xs.max())
    xr = [x]
    if np.any(x < lower):
        xr.append(2 * lower - x[x < lower])
    if np.any(x > upper):
        xr.append(2 * upper - x[x > upper])
    xr = np.concatenate(xr)
    kde = gaussian_kde(xr)
    if bw is not None:
        kde.set_bandwidth(bw_method=bw)
    y = kde(xs)
    y /= np.trapezoid(y, xs) + 1e-12
    return y


def _load_entropy_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    tag = str(d.get("model_tag", "model"))
    H_id = np.asarray(d["H_id"], dtype=np.float64)
    H_ood = np.asarray(d["H_ood"], dtype=np.float64)
    return tag, H_id, H_ood


def plot_id_vs_ood_entropy_kde(
    npz_path: str,
    *,
    num_classes: int = 10,
    zoom: tuple[float, float] | None = None,
    bw="scott",
    color_id="rgba(31,119,180,0.6)",
    color_ood="rgba(255,127,14,0.35)",
    show_medians: bool = False,
):
    """Single-panel figure. If zoom=None -> full range [0, ln K], else only the zoom window."""
    tag, H_id, H_ood = _load_entropy_npz(npz_path)

    xmax = float(np.log(num_classes))
    H_id = np.clip(H_id, 0.0, xmax)
    H_ood = np.clip(H_ood, 0.0, xmax)

    if zoom is None:
        z0, z1 = 0.0, xmax
        title = f"{tag} — Predictive Entropy"
    else:
        z0, z1 = max(0.0, zoom[0]), min(xmax, zoom[1])
        title = f"{tag} — Predictive Entropy (Zoom {z0:.2f}–{z1:.2f})"

    xs = np.linspace(z0, z1, 700)
    y_id = _reflect_kde(H_id, xs, 0.0, xmax, bw=bw)
    y_ood = _reflect_kde(H_ood, xs, 0.0, xmax, bw=bw)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=y_ood,
            mode="lines",
            name="OOD (KDE)",
            line=dict(color=color_ood.replace("0.35", "1.0"), width=3),
            fill="tozeroy",
            fillcolor=color_ood,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=y_id,
            mode="lines",
            name="ID (KDE)",
            line=dict(color=color_id.replace("0.6", "1.0"), width=3),
            fill="tozeroy",
            fillcolor=color_id,
        )
    )

    if show_medians:
        m_id, m_ood = float(np.median(H_id)), float(np.median(H_ood))
        if z0 <= m_ood <= z1:
            fig.add_vline(
                x=m_ood, line_dash="dash", line_color=color_ood.replace("0.35", "1.0")
            )
        if z0 <= m_id <= z1:
            fig.add_vline(
                x=m_id, line_dash="dash", line_color=color_id.replace("0.6", "1.0")
            )

    fig.update_xaxes(range=[z0, z1], title="Predictive entropy (nats)")
    fig.update_yaxes(title="Density")
    fig.update_layout(
        template="plotly_white", title=title, bargap=0.02, legend=dict(orientation="h")
    )
    return fig


def plot_id_vs_ood_entropy_kde_full_and_zoom(
    npz_path: str,
    *,
    num_classes: int = 10,
    zoom: tuple[float, float] = (0.0, 0.12),
    bw="scott",
    color_id="rgba(31,119,180,0.6)",
    color_ood="rgba(255,127,14,0.35)",
    show_medians: bool = False,
):
    """Backwards-compatible two-panel figure (full + zoom) in one canvas."""
    tag, H_id, H_ood = _load_entropy_npz(npz_path)
    xmax = float(np.log(num_classes))
    H_id = np.clip(H_id, 0.0, xmax)
    H_ood = np.clip(H_ood, 0.0, xmax)
    xs_full = np.linspace(0.0, xmax, 700)
    y_id_full = _reflect_kde(H_id, xs_full, 0.0, xmax, bw=bw)
    y_ood_full = _reflect_kde(H_ood, xs_full, 0.0, xmax, bw=bw)

    z0, z1 = max(0.0, zoom[0]), min(xmax, zoom[1])
    xs_zoom = np.linspace(z0, z1, 700)
    y_id_zoom = _reflect_kde(H_id, xs_zoom, 0.0, xmax, bw=bw)
    y_ood_zoom = _reflect_kde(H_ood, xs_zoom, 0.0, xmax, bw=bw)

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Full", f"Zoom {z0:.2f}–{z1:.2f}")
    )
    # full
    fig.add_trace(
        go.Scatter(
            x=xs_full,
            y=y_ood_full,
            mode="lines",
            name="OOD (KDE)",
            line=dict(color=color_ood.replace("0.35", "1.0"), width=3),
            fill="tozeroy",
            fillcolor=color_ood,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xs_full,
            y=y_id_full,
            mode="lines",
            name="ID (KDE)",
            line=dict(color=color_id.replace("0.6", "1.0"), width=3),
            fill="tozeroy",
            fillcolor=color_id,
        ),
        row=1,
        col=1,
    )
    # zoom
    fig.add_trace(
        go.Scatter(
            x=xs_zoom,
            y=y_ood_zoom,
            mode="lines",
            showlegend=False,
            line=dict(color=color_ood.replace("0.35", "1.0"), width=3),
            fill="tozeroy",
            fillcolor=color_ood,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=xs_zoom,
            y=y_id_zoom,
            mode="lines",
            showlegend=False,
            line=dict(color=color_id.replace("0.6", "1.0"), width=3),
            fill="tozeroy",
            fillcolor=color_id,
        ),
        row=1,
        col=2,
    )

    if show_medians:
        m_id, m_ood = float(np.median(H_id)), float(np.median(H_ood))
        fig.add_vline(
            x=m_ood,
            line_dash="dash",
            line_color=color_ood.replace("0.35", "1.0"),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=m_id,
            line_dash="dash",
            line_color=color_id.replace("0.6", "1.0"),
            row=1,
            col=1,
        )
        if z0 <= m_ood <= z1:
            fig.add_vline(
                x=m_ood,
                line_dash="dash",
                line_color=color_ood.replace("0.35", "1.0"),
                row=1,
                col=2,
            )
        if z0 <= m_id <= z1:
            fig.add_vline(
                x=m_id,
                line_dash="dash",
                line_color=color_id.replace("0.6", "1.0"),
                row=1,
                col=2,
            )

    fig.update_xaxes(range=[0.0, xmax], title="Entropy (nats)", row=1, col=1)
    fig.update_xaxes(range=[z0, z1], title="Entropy (zoom)", row=1, col=2)
    fig.update_yaxes(title="Density", row=1, col=1)
    fig.update_yaxes(title="Density", row=1, col=2)
    fig.update_layout(
        template="plotly_white", title=f"{tag} — Predictive Entropy", bargap=0.02
    )
    return fig


def save_entropy_kde_pair(
    npz_path: str,
    out_full: str,
    out_zoom: str,
    *,
    num_classes: int = 10,
    zoom: tuple[float, float] = (0.0, 0.12),
    bw="scott",
    dpi: int = 300,
):
    # colors similar to Plotly default
    col_id = "#1f77b4"
    col_ood = "#ff7f0e"

    tag, H_id, H_ood = _load_entropy_npz(npz_path)
    xmax = float(np.log(num_classes))
    H_id = np.clip(H_id, 0.0, xmax)
    H_ood = np.clip(H_ood, 0.0, xmax)

    # --- Full range ---
    xs = np.linspace(0.0, xmax, 700)
    y_id = _reflect_kde(H_id, xs, 0.0, xmax, bw=bw)
    y_ood = _reflect_kde(H_ood, xs, 0.0, xmax, bw=bw)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, y_ood, color=col_ood, lw=2, label="OOD (KDE)")
    ax.fill_between(xs, 0, y_ood, color=col_ood, alpha=0.35)
    ax.plot(xs, y_id, color=col_id, lw=2, label="ID (KDE)")
    ax.fill_between(xs, 0, y_id, color=col_id, alpha=0.35)
    ax.set_xlabel("Predictive entropy (nats) [0, ln K]")
    ax.set_ylabel("Density")
    ax.set_title(f"{tag} — Predictive Entropy (full)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_full, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # --- Zoom ---
    z0, z1 = max(0.0, zoom[0]), min(xmax, zoom[1])
    xs = np.linspace(z0, z1, 700)
    y_id = _reflect_kde(H_id, xs, 0.0, xmax, bw=bw)
    y_ood = _reflect_kde(H_ood, xs, 0.0, xmax, bw=bw)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, y_ood, color=col_ood, lw=2, label="OOD (KDE)")
    ax.fill_between(xs, 0, y_ood, color=col_ood, alpha=0.35)
    ax.plot(xs, y_id, color=col_id, lw=2, label="ID (KDE)")
    ax.fill_between(xs, 0, y_id, color=col_id, alpha=0.35)
    ax.set_xlim(z0, z1)
    ax.set_xlabel("Predictive entropy (zoom)")
    ax.set_ylabel("Density")
    ax.set_title(f"{tag} — Predictive Entropy (zoom {z0:.2f}–{z1:.2f})")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_zoom, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out_full}\n[saved] {out_zoom}")


def _get_spectral_latents(
    clf, D: int, *, prefix: str = "rfft1d", n_samples: int = 256, rng_seed: int = 0
):
    """
    Pull posterior samples for spectral latents from a *fitted* clf.
    Works for SVI (guide+params) or MCMC (posterior_samples).
    """
    X_dummy = jnp.zeros((1, D), dtype=jnp.float32)
    key = jax.random.PRNGKey(int(rng_seed))
    sites = [f"{prefix}_real", f"{prefix}_imag"]

    backend = getattr(clf, "_backend_", None)
    if backend == "svi":
        pred = Predictive(
            clf.model,
            guide=clf._inference_.guide,
            params=clf.params_,
            num_samples=int(n_samples),
            return_sites=sites,  # set here (constructor), not in __call__
        )
    elif backend == "mcmc":
        pred = Predictive(
            clf.model,
            posterior_samples=clf.posterior_samples_,
            num_samples=int(n_samples),
            return_sites=sites,
        )
    else:
        raise ValueError("Classifier not fitted or unsupported backend.")

    samples = pred(key, X_dummy, y=None)
    missing = [s for s in sites if s not in samples]
    if missing:
        raise KeyError(
            f"Requested sites {missing} not returned. "
            f"Check the layer name (prefix='{prefix}') and that the guide samples those sites."
        )

    re = np.asarray(samples[f"{prefix}_real"])
    im = np.asarray(samples[f"{prefix}_imag"])
    if re.ndim == 3:
        re = re[:, 0, :]
    if im.ndim == 3:
        im = im[:, 0, :]
    return re, im  # (S, K)


def save_spectral_posterior_1d(
    clf,
    *,
    D: int,
    padded_dim: int | None,
    out_path: str,
    prefix: str = "rfft1d",
    n_samples: int = 256,
    ci=(0.05, 0.95),
    normalize_freq: bool = True,
    logy: bool = False,
    title: str | None = None,
    rng_seed: int = 0,
):
    """
    Same API as before, but:
      - pads to full half-band (k_half) so x spans [0,1] when normalize_freq=True
      - draws a dashed vertical line at the active K cutoff
    """
    re, im = _get_spectral_latents(
        clf, D, prefix=prefix, n_samples=n_samples, rng_seed=rng_seed
    )
    amp = np.sqrt(re**2 + im**2)  # (S, K)
    m = amp.mean(axis=0)  # (K,)
    lo, hi = np.quantile(amp, q=ci, axis=0)  # (K,)

    Hpad = int(padded_dim or D)
    k_half = Hpad // 2 + 1
    K = m.shape[0]

    # pad to full half-band so normalization is consistent
    if K < k_half:
        pad = k_half - K
        m = np.pad(m, (0, pad))
        lo = np.pad(lo, (0, pad))
        hi = np.pad(hi, (0, pad))
    elif K > k_half:
        m, lo, hi = m[:k_half], lo[:k_half], hi[:k_half]  # safety

    if normalize_freq:
        x = np.linspace(0.0, 1.0, k_half)
        cutoff = (min(K, k_half) - 1) / max(1, (k_half - 1))
    else:
        x = np.arange(k_half)
        cutoff = min(K, k_half) - 1

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, m, lw=2, label="posterior mean amplitude")
    ax.fill_between(x, lo, hi, alpha=0.25, label=f"{int((ci[1]-ci[0])*100)}% CI")
    # show active-band cutoff
    ax.axvline(cutoff, ls="--", c="gray", lw=1, label="K cutoff")

    ax.set_xlabel("normalized frequency" if normalize_freq else "frequency bin")
    ax.set_ylabel("amplitude")
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
