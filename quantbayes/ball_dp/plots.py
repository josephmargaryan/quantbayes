# quantbayes/ball_dp/plots.py
from __future__ import annotations

import dataclasses as dc
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .types import AttackResult, ReRoReport, ReleaseArtifact, Record


def _save_or_show(fig, out_path: Optional[str]) -> None:
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=160)
        plt.close(fig)
    else:
        plt.show()


def _reshape_like_feature(
    x: np.ndarray,
    feature_shape: Sequence[int] | None,
) -> np.ndarray:
    arr = np.asarray(x)
    if feature_shape is not None:
        feature_shape = tuple(int(v) for v in feature_shape)
        if int(np.prod(feature_shape)) == int(arr.size):
            arr = arr.reshape(feature_shape)
    if arr.ndim == 1:
        n = arr.size
        side = int(round(n**0.5))
        if side * side == n:
            arr = arr.reshape(side, side)
        else:
            return arr[None, :]
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _imshow_array(ax, arr: np.ndarray, *, cmap: str = "gray") -> None:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        ax.imshow(arr, cmap=cmap)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        ax.imshow(arr[..., 0], cmap=cmap)
    else:
        ax.imshow(arr)
    ax.axis("off")


def _as_image(x: np.ndarray, feature_shape: Sequence[int] | None = None) -> np.ndarray:
    return _reshape_like_feature(np.asarray(x), feature_shape)


def _iter_float_arrays(obj: Any) -> list[np.ndarray]:
    out: list[np.ndarray] = []

    def visit(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, np.floating):
                out.append(np.asarray(x))
            return
        if dc.is_dataclass(x):
            for field in dc.fields(x):
                visit(getattr(x, field.name))
            return
        if isinstance(x, dict):
            for value in x.values():
                visit(value)
            return
        if isinstance(x, (list, tuple)):
            for value in x:
                visit(value)
            return
        if hasattr(x, "__dict__") and not isinstance(x, (str, bytes)):
            for value in vars(x).values():
                visit(value)
            return

    visit(obj)
    return out


def _extract_linear_weight_array(payload: Any) -> np.ndarray:
    for name in ("W", "w", "weight", "weights", "coef_"):
        if hasattr(payload, name):
            arr = np.asarray(getattr(payload, name))
            if np.issubdtype(arr.dtype, np.floating) and arr.size > 0:
                return arr

    arrays = [arr for arr in _iter_float_arrays(payload) if arr.size > 1]
    if not arrays:
        raise ValueError(
            "Could not find a floating-point parameter array to visualize."
        )

    def key(arr: np.ndarray) -> tuple[int, int]:
        return (int(arr.ndim >= 2), int(arr.size))

    arrays.sort(key=key, reverse=True)
    return arrays[0]


def _metric_text_from_attack(attack: AttackResult) -> Optional[str]:
    parts = []
    for k, v in sorted(attack.metrics.items()):
        if np.isscalar(v):
            parts.append(f"{k}={float(v):.4g}")
    return None if not parts else " | ".join(parts)


def plot_reconstruction_pair(
    true_features: np.ndarray,
    reconstructed_features: np.ndarray,
    *,
    feature_shape: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    metric_text: Optional[str] = None,
    out_path: Optional[str] = None,
) -> None:
    """Plot a target / reconstruction pair."""
    true_img = _as_image(true_features, feature_shape=feature_shape)
    recon_img = _as_image(reconstructed_features, feature_shape=feature_shape)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax, img, ax_title in zip(
        axes, [true_img, recon_img], ["Target", "Reconstruction"]
    ):
        _imshow_array(ax, img)
        ax.set_title(ax_title)

    text_parts = []
    if title is not None:
        text_parts.append(str(title))
    if metric_text is not None:
        text_parts.append(str(metric_text))
    if text_parts:
        fig.suptitle(" | ".join(text_parts))
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_reconstruction_triplet(
    true_features: np.ndarray,
    reconstructed_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    feature_shape: Optional[Sequence[int]] = None,
    titles: Sequence[str] = ("Target", "Reconstruction", "Reference"),
    title: Optional[str] = None,
    metric_text: Optional[str] = None,
    out_path: Optional[str] = None,
) -> None:
    """Plot target / reconstruction / reference side-by-side."""
    imgs = [
        _as_image(true_features, feature_shape=feature_shape),
        _as_image(reconstructed_features, feature_shape=feature_shape),
        _as_image(reference_features, feature_shape=feature_shape),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, ax_title in zip(axes, imgs, titles):
        _imshow_array(ax, img)
        ax.set_title(str(ax_title))

    text_parts = []
    if title is not None:
        text_parts.append(str(title))
    if metric_text is not None:
        text_parts.append(str(metric_text))
    if text_parts:
        fig.suptitle(" | ".join(text_parts))
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_attack_result(
    attack: AttackResult,
    true_record: Record,
    *,
    feature_shape: Optional[Sequence[int]] = None,
    out_path: Optional[str] = None,
) -> None:
    """Plot the true target and an arbitrary attack reconstruction."""
    if attack.z_hat is None:
        raise ValueError("attack.z_hat is None; no reconstruction to plot.")
    plot_reconstruction_pair(
        np.asarray(true_record.features),
        np.asarray(attack.z_hat),
        feature_shape=feature_shape,
        title=f"{attack.attack_family} ({attack.status})",
        metric_text=_metric_text_from_attack(attack),
        out_path=out_path,
    )


def plot_attack_result_with_reference(
    attack: AttackResult,
    true_record: Record,
    reference_features: np.ndarray,
    *,
    feature_shape: Optional[Sequence[int]] = None,
    reference_title: str = "Reference",
    out_path: Optional[str] = None,
) -> None:
    """Plot target, attack reconstruction, and a reference baseline."""
    if attack.z_hat is None:
        raise ValueError("attack.z_hat is None; no reconstruction to plot.")
    plot_reconstruction_triplet(
        np.asarray(true_record.features),
        np.asarray(attack.z_hat),
        np.asarray(reference_features),
        feature_shape=feature_shape,
        titles=("Target", "Reconstruction", reference_title),
        title=f"{attack.attack_family} ({attack.status})",
        metric_text=_metric_text_from_attack(attack),
        out_path=out_path,
    )


def plot_attack_candidates(
    attack: AttackResult,
    *,
    feature_shape: Optional[Sequence[int]] = None,
    true_record: Optional[Record] = None,
    top_k: int = 6,
    out_path: Optional[str] = None,
) -> None:
    """Plot the top candidate reconstructions stored in attack.candidates."""
    if not attack.candidates:
        raise ValueError("attack.candidates is missing or empty.")

    candidates = attack.candidates[: int(top_k)]
    n_panels = len(candidates) + (1 if true_record is not None else 0)
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.7 * nrows))
    axes = np.asarray(axes).reshape(-1)

    idx_ax = 0
    if true_record is not None:
        _imshow_array(
            axes[idx_ax],
            _as_image(np.asarray(true_record.features), feature_shape=feature_shape),
        )
        axes[idx_ax].set_title(f"Target\nlabel={int(true_record.label)}")
        idx_ax += 1

    for rank, (label, features) in enumerate(candidates, start=1):
        _imshow_array(
            axes[idx_ax],
            _as_image(np.asarray(features), feature_shape=feature_shape),
        )
        axes[idx_ax].set_title(f"rank={rank}\nlabel={label}")
        idx_ax += 1

    for ax in axes[idx_ax:]:
        ax.axis("off")

    fig.suptitle(f"Top candidates: {attack.attack_family}")
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_attack_score_histogram(
    attack: AttackResult,
    *,
    out_path: Optional[str] = None,
) -> None:
    """Plot score/objective diagnostics when available."""
    diagnostics = dict(attack.diagnostics)

    if "final_scores" in diagnostics:
        scores = np.asarray(diagnostics["final_scores"], dtype=np.float32).reshape(-1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(scores, bins=min(40, max(10, len(scores) // 4)), alpha=0.8)
        ax.axvline(np.max(scores), color="red", linestyle="--", label="best")
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.set_title(f"Score histogram: {attack.attack_family}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save_or_show(fig, out_path)
        return

    if "per_label_best_objective" in diagnostics:
        data = diagnostics["per_label_best_objective"]
        labels = list(map(str, data.keys()))
        vals = np.asarray(list(data.values()), dtype=np.float32)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, vals)
        ax.set_xlabel("label")
        ax.set_ylabel("best objective")
        ax.set_title(f"Per-label objective: {attack.attack_family}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save_or_show(fig, out_path)
        return

    raise ValueError(
        "Attack diagnostics do not contain 'final_scores' or 'per_label_best_objective'."
    )


def plot_attack_label_logits(
    attack: AttackResult,
    *,
    out_path: Optional[str] = None,
) -> None:
    """Plot predicted label logits when available."""
    diagnostics = dict(attack.diagnostics)
    if "label_logits" not in diagnostics:
        raise ValueError("Attack diagnostics do not contain 'label_logits'.")

    logits = np.asarray(diagnostics["label_logits"], dtype=np.float32).reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(len(logits)), logits)
    ax.set_xlabel("class")
    ax.set_ylabel("logit")
    ax.set_title(f"Label logits: {attack.attack_family}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_convex_attack_result(
    attack: AttackResult,
    true_record: Record,
    *,
    out_path: Optional[str] = None,
) -> None:
    """Plot the true target and the convex reconstruction candidate."""
    if attack.z_hat is None:
        raise ValueError("attack.z_hat is None; no reconstruction to plot.")

    metric_text = _metric_text_from_attack(attack)
    plot_reconstruction_pair(
        np.asarray(true_record.features),
        np.asarray(attack.z_hat),
        title=f"{attack.attack_family} ({attack.status})",
        metric_text=metric_text or None,
        out_path=out_path,
    )


def plot_release_curves(
    release: ReleaseArtifact,
    *,
    out_path: Optional[str] = None,
) -> None:
    """Plot the stored public evaluation curves from a release artifact."""
    history = release.extra.get("public_curve_history", None)
    if not history:
        raise ValueError("release.extra['public_curve_history'] is missing or empty.")

    steps = np.asarray([row["step"] for row in history], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7, 4))

    if any("public_eval_loss" in row for row in history):
        losses = [row.get("public_eval_loss", np.nan) for row in history]
        ax.plot(steps, losses, label="public_eval_loss")
    if any("public_eval_accuracy" in row for row in history):
        accs = [row.get("public_eval_accuracy", np.nan) for row in history]
        ax.plot(steps, accs, label="public_eval_accuracy")

    ax.set_xlabel("step")
    ax.set_title("Public evaluation curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_operator_norm_history(
    release: ReleaseArtifact,
    *,
    out_path: Optional[str] = None,
) -> None:
    """Plot the max / mean operator-norm summaries stored on a release."""
    history = release.extra.get("operator_norm_history", None)
    if not history:
        raise ValueError("release.extra['operator_norm_history'] is missing or empty.")

    steps = np.asarray([row["step"] for row in history], dtype=np.int64)
    max_sigma = [row["summary"].get("max_sigma", np.nan) for row in history]
    mean_sigma = [row["summary"].get("mean_sigma", np.nan) for row in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, max_sigma, label="max_sigma")
    ax.plot(steps, mean_sigma, label="mean_sigma")
    ax.set_xlabel("step")
    ax.set_title("Operator norm history")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_ridge_prototypes(
    release: ReleaseArtifact,
    *,
    class_names: Optional[Sequence[str]] = None,
    out_path: Optional[str] = None,
) -> None:
    """Visualize learned ridge prototypes stored in a convex release."""
    payload = release.payload
    if not hasattr(payload, "prototypes"):
        raise ValueError("release.payload does not expose a 'prototypes' attribute.")

    prototypes = np.asarray(payload.prototypes, dtype=np.float32)
    if prototypes.ndim != 2:
        raise ValueError(
            f"Expected prototypes with shape (num_classes, flat_dim), got {prototypes.shape}."
        )

    feature_shape = release.dataset_metadata.get("feature_shape", None)
    counts = getattr(payload, "counts", None)
    num_classes = prototypes.shape[0]
    ncols = min(5, num_classes)
    nrows = int(np.ceil(num_classes / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        if idx >= num_classes:
            ax.axis("off")
            continue
        proto = _reshape_like_feature(prototypes[idx], feature_shape)
        _imshow_array(ax, proto)
        title = str(idx) if class_names is None else str(class_names[idx])
        if counts is not None and idx < len(counts):
            title = f"{title}\ncount={int(counts[idx])}"
        ax.set_title(title)

    fig.suptitle("Learned ridge prototypes")
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_linear_model_weights(
    release: ReleaseArtifact,
    *,
    class_names: Optional[Sequence[str]] = None,
    out_path: Optional[str] = None,
) -> None:
    """Visualize weight vectors from a linear convex model release."""
    if release.model_family == "ridge_prototype":
        raise ValueError(
            "Ridge-prototype releases should be visualized with plot_ridge_prototypes(...)."
        )

    feature_shape = release.dataset_metadata.get("feature_shape", None)
    W = np.asarray(_extract_linear_weight_array(release.payload), dtype=np.float32)

    if W.ndim == 1:
        panels = W[None, :]
    elif W.ndim == 2:
        panels = W
    else:
        panels = W.reshape(W.shape[0], -1)

    num_panels = panels.shape[0]
    ncols = min(5, num_panels)
    nrows = int(np.ceil(num_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        if idx >= num_panels:
            ax.axis("off")
            continue
        arr = _reshape_like_feature(panels[idx], feature_shape)
        _imshow_array(ax, arr)
        if class_names is not None and idx < len(class_names):
            title = str(class_names[idx])
        elif num_panels == 1:
            title = "weight"
        else:
            title = f"class {idx}"
        ax.set_title(title)

    fig.suptitle("Linear model weights")
    fig.tight_layout()
    _save_or_show(fig, out_path)


def plot_convex_model_parameters(
    release: ReleaseArtifact,
    *,
    class_names: Optional[Sequence[str]] = None,
    out_path: Optional[str] = None,
) -> None:
    """Visualize convex model parameters in a family-specific way."""
    if release.model_family == "ridge_prototype":
        return plot_ridge_prototypes(
            release, class_names=class_names, out_path=out_path
        )
    return plot_linear_model_weights(
        release, class_names=class_names, out_path=out_path
    )


def plot_rero_report(report: ReRoReport, *, out_path: Optional[str] = None) -> None:
    """Plot the Ball-ReRo upper bound curve."""
    etas = [p.eta for p in report.points]
    gamma_ball = [p.gamma_ball for p in report.points]
    gamma_standard = [p.gamma_standard for p in report.points]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(etas, gamma_ball, marker="o", label="gamma_ball")
    if any(v is not None for v in gamma_standard):
        ax.plot(
            etas,
            [np.nan if v is None else v for v in gamma_standard],
            marker="o",
            label="gamma_standard",
        )
    ax.set_xlabel("eta")
    ax.set_ylabel("success upper bound")
    ax.set_title(f"Ball-ReRo ({report.mode})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, out_path)
