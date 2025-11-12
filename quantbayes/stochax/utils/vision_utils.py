"""
metrics_and_utils.py

JAX/Equinox-friendly metrics (classification + segmentation) and utilities:
- top-k accuracy, confusion matrix, per-class precision/recall/F1
- segmentation IoU/mIoU, Dice (per-class), Boundary F1 (tolerant boundary match)
- EMA helper (init/update/swap)
- dtype casting helpers (simple mixed precision hygiene)
- optax helper to add global-norm clipping

All functions are pure and jit-friendly where it matters.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.ndimage as ndi
import optax
import equinox as eqx

__all__ = [
    "topk_accuracy",
    "classification_confusion_matrix",
    "classification_report_from_cm",
    "segmentation_confusion_matrix",
    "iou_from_cm",
    "dice_from_cm",
    "boundary_f1",
    "EMA",
    "init_ema",
    "update_ema",
    "swap_ema_params",
    "expected_calibration_error",
    "brier_score",
]


# ----------------------------- Classification ----------------------------- #


@jax.jit
def topk_accuracy(
    logits: jnp.ndarray, targets: jnp.ndarray, ks: Sequence[int] = (1, 5)
) -> Dict[int, jnp.ndarray]:
    """
    logits: [B, C]
    targets: [B] int32
    returns {k: acc@k} with shape []
    """
    ks = tuple(sorted(ks))
    topk = jnp.argsort(logits, axis=-1)[:, ::-1][:, : ks[-1]]  # [B, Kmax]
    correct = topk == targets[:, None]
    out = {}
    for k in ks:
        out[k] = jnp.mean(jnp.any(correct[:, :k], axis=1))
    return out


@jax.jit
def classification_confusion_matrix(
    logits_or_preds: jnp.ndarray,
    targets: jnp.ndarray,
    num_classes: int,
    from_logits: bool = True,
) -> jnp.ndarray:
    """
    Returns [C, C] matrix M where M[i, j] == count(pred=j, true=i).
    """
    preds = jnp.argmax(logits_or_preds, axis=-1) if from_logits else logits_or_preds
    t = targets.reshape(-1)
    p = preds.reshape(-1)

    mask = (t >= 0) & (t < num_classes)
    t = t[mask]
    p = p[mask]

    cm = jnp.zeros((num_classes, num_classes), dtype=jnp.int32)
    cm = cm.at[t, p].add(1)
    return cm


def classification_report_from_cm(cm: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    cm: [C, C] with cm[i, j] = count(true=i, pred=j)
    returns per-class: precision, recall, f1
    """
    tp = jnp.diag(cm).astype(jnp.float32)
    fp = cm.sum(axis=0).astype(jnp.float32) - tp
    fn = cm.sum(axis=1).astype(jnp.float32) - tp
    precision = tp / jnp.clip(tp + fp, 1e-9)
    recall = tp / jnp.clip(tp + fn, 1e-9)
    f1 = 2 * precision * recall / jnp.clip(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


# ------------------------------ Segmentation ------------------------------ #


def _seg_to_labels(
    logits_or_labels: jnp.ndarray, from_logits: bool, channel_axis: int
) -> jnp.ndarray:
    if from_logits:
        x = jnp.moveaxis(logits_or_labels, channel_axis, -1)
        return jnp.argmax(x, axis=-1)
    return logits_or_labels


@jax.jit
def segmentation_confusion_matrix(
    logits_or_labels: jnp.ndarray,  # [B, C, H, W] or [B, H, W]
    targets_hw: jnp.ndarray,  # [B, H, W]
    num_classes: int,
    *,
    from_logits: bool = True,
    channel_axis: int = 1,
    ignore_index: int | None = None,
) -> jnp.ndarray:
    """
    Returns [C, C] matrix M where M[i, j] == count(pred=j, true=i), aggregated over pixels.
    """
    pred_hw = _seg_to_labels(logits_or_labels, from_logits, channel_axis)  # [B, H, W]
    t = targets_hw.reshape(-1)
    p = pred_hw.reshape(-1)

    mask = (t >= 0) & (t < num_classes)
    if ignore_index is not None:
        mask = mask & (t != ignore_index)

    t = t[mask]
    p = p[mask]

    cm = jnp.zeros((num_classes, num_classes), dtype=jnp.int32)
    cm = cm.at[t, p].add(1)
    return cm


def iou_from_cm(cm: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (per_class_iou [C], miou []).
    IoU_c = TP / (TP+FP+FN).
    """
    tp = jnp.diag(cm).astype(jnp.float32)
    fp = cm.sum(axis=0).astype(jnp.float32) - tp
    fn = cm.sum(axis=1).astype(jnp.float32) - tp
    denom = jnp.clip(tp + fp + fn, 1e-9)
    iou = tp / denom
    return iou, jnp.mean(iou)


def dice_from_cm(cm: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (per_class_dice [C], mean_dice []).
    Dice_c = 2*TP / (2*TP + FP + FN)
    """
    tp = jnp.diag(cm).astype(jnp.float32)
    fp = cm.sum(axis=0).astype(jnp.float32) - tp
    fn = cm.sum(axis=1).astype(jnp.float32) - tp
    denom = jnp.clip(2 * tp + fp + fn, 1e-9)
    dice = 2 * tp / denom
    return dice, jnp.mean(dice)


# -------- Boundary F1 (tolerant). Good for thin structures / edges. -------- #


def _binary_boundary_map(mask_hw: jnp.ndarray) -> jnp.ndarray:
    """
    Simple boundary extraction via morphological gradient.
    mask_hw: [H, W], bool
    returns: [H, W], bool boundary
    """
    # max/min filter in 3x3 neighbourhood
    maxf = ndi.maximum_filter(mask_hw.astype(jnp.float32), size=3, mode="nearest")
    minf = ndi.minimum_filter(mask_hw.astype(jnp.float32), size=3, mode="nearest")
    return (maxf - minf) > 0.0


def _dilate_bool(mask_hw: jnp.ndarray, radius: int) -> jnp.ndarray:
    if radius <= 0:
        return mask_hw
    size = 2 * radius + 1
    return (
        ndi.maximum_filter(mask_hw.astype(jnp.float32), size=size, mode="nearest") > 0.0
    )


def boundary_f1(
    logits_or_labels: jnp.ndarray,  # [B,C,H,W] or [B,H,W]
    targets_hw: jnp.ndarray,  # [B,H,W]
    num_classes: int,
    *,
    from_logits: bool = True,
    channel_axis: int = 1,
    ignore_index: int | None = None,
    tolerance: int = 2,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Per-class boundary F1 with tolerance (px). Returns (per_class_BF1 [C], mean_BF1 []).
    For each class c:
      precision = |B_pred ∧ dilate(B_true, tol)| / |B_pred|
      recall    = |B_true ∧ dilate(B_pred, tol)| / |B_true|
      BF1 = 2PR / (P+R)
    """
    preds = _seg_to_labels(logits_or_labels, from_logits, channel_axis)  # [B,H,W]
    B, H, W = preds.shape

    # Mask ignored
    if ignore_index is not None:
        valid = targets_hw != ignore_index
    else:
        valid = jnp.ones_like(targets_hw, dtype=bool)

    def per_class(c):
        # binary masks
        p = (preds == c) & valid
        t = (targets_hw == c) & valid

        pb = _binary_boundary_map(p)
        tb = _binary_boundary_map(t)

        if tolerance > 0:
            tb_d = _dilate_bool(tb, tolerance)
            pb_d = _dilate_bool(pb, tolerance)
        else:
            tb_d, pb_d = tb, pb

        # counts summed over batch
        tp_prec = jnp.sum((pb & tb_d).astype(jnp.float32))
        tp_rec = jnp.sum((tb & pb_d).astype(jnp.float32))
        denom_p = jnp.clip(jnp.sum(pb.astype(jnp.float32)), 1e-9)
        denom_t = jnp.clip(jnp.sum(tb.astype(jnp.float32)), 1e-9)

        precision = tp_prec / denom_p
        recall = tp_rec / denom_t
        f1 = 2 * precision * recall / jnp.clip(precision + recall, 1e-9)
        return f1

    bf1 = jax.vmap(per_class)(jnp.arange(num_classes))
    return bf1, jnp.mean(bf1)


# --------------------------------- EMA --------------------------------- #


@dataclass
class EMA:
    # arrays-only pytree whose structure matches `eqx.partition(model, eqx.is_inexact_array)[0]`
    params: Any
    decay: float


def init_ema(model: Any, decay: float = 0.999) -> EMA:
    # arrays-only params tree; no Nones mixed in (partition is preferable to filter here)
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    return EMA(params=params, decay=float(decay))


# JIT **arrays-only** function; do not pass the EMA object through JIT.
@eqx.filter_jit
def _ema_update_arrays(ema_params, model_params, decay: float):
    # Correct EMA update: new = decay*old + (1-decay)*current
    return jax.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p, ema_params, model_params
    )


def update_ema(ema: EMA, model: Any) -> EMA:
    # Partition model to arrays-only, call the jitted arrays-only updater.
    model_params, _ = eqx.partition(model, eqx.is_inexact_array)
    new_params = _ema_update_arrays(ema.params, model_params, float(ema.decay))
    return EMA(params=new_params, decay=ema.decay)


def swap_ema_params(model: Any, ema: EMA) -> Any:
    """Return a copy of `model` with EMA params swapped in."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    # Ensure trees align: ema.params must match `params` structure.
    return eqx.combine(ema.params, static)


### Metrics


def expected_calibration_error(logits, targets, n_bins: int = 15):
    # logits: [B,C], targets: [B] int32
    probs = jax.nn.softmax(logits, axis=-1)
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == targets).astype(jnp.float32)

    bins = jnp.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi)
        denom = jnp.maximum(1, mask.sum())
        acc = (correct * mask).sum() / denom
        avg_conf = (conf * mask).sum() / denom
        ece += (mask.mean()) * jnp.abs(acc - avg_conf)
    return ece


def brier_score(logits, targets, num_classes):
    probs = jax.nn.softmax(logits, axis=-1)
    onehot = jax.nn.one_hot(targets, num_classes)
    return jnp.mean(jnp.sum((probs - onehot) ** 2, axis=-1))
