from __future__ import annotations

"""Compact high-level workflows for dense <-> spectral retrofit experiments.

The public design goal is simple:

1. instantiate the model family/variant you want,
2. optionally load a pretrained checkpoint,
3. optionally retrofit dense Linear leaves into ``SVDDense`` or ``RFFTCirculant1D``,
4. optionally freeze only the SVD bases (``U,V``) or everything except ``s``.

This keeps the theory-facing workflow explicit while avoiding model-family-specific
ad hoc scripts.
"""

from typing import Any, Callable, Literal

import equinox as eqx
import jax.tree_util as jtu

from quantbayes.stochax.layers import SVDDense
from quantbayes.stochax.utils.linear_surgery import (
    replace_square_linears_with_rfft as _replace_square_linears_with_rfft,
)
from quantbayes.stochax.utils.optim_util import make_freeze_mask

AttentionFamily = Literal["vit", "dino", "swin"]
AttentionSVDMode = Literal["attn_only", "attn_mlp", "all_linear"]
LinearPredicate = Callable[[str, eqx.nn.Linear], bool]

__all__ = [
    "AttentionFamily",
    "AttentionSVDMode",
    "replace_linears_with_svd",
    "replace_attention_linears_with_svd",
    "replace_square_linears_with_rfft",
    "replace_attention_linears_with_rfft",
    "make_svd_basis_freeze_mask",
    "make_s_only_freeze_mask",
]


def _path_to_str(path_entries: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for pe in path_entries:
        if hasattr(pe, "name"):
            parts.append(str(pe.name))
        elif hasattr(pe, "key"):
            parts.append(str(pe.key))
        elif hasattr(pe, "idx"):
            parts.append(f"[{pe.idx}]")
        else:
            parts.append(str(pe))
    return ".".join(parts).replace(".[", "[")


def _full_rank(linear: eqx.nn.Linear) -> int:
    return int(min(linear.weight.shape[0], linear.weight.shape[1]))


def _resolve_rank(
    linear: eqx.nn.Linear,
    *,
    rank: int | None,
    rank_ratio: float | None,
) -> int:
    if rank is not None and rank_ratio is not None:
        raise ValueError("Pass at most one of rank= or rank_ratio=.")

    full_rank = _full_rank(linear)
    if rank is None and rank_ratio is None:
        return full_rank
    if rank is not None:
        return max(1, min(int(rank), full_rank))

    ratio = float(rank_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"rank_ratio must lie in (0, 1], got {rank_ratio!r}.")
    return max(1, min(int(round(ratio * full_rank)), full_rank))


def _vit_or_dino_predicate(mode: AttentionSVDMode) -> LinearPredicate:
    def predicate(path: str, _: eqx.nn.Linear) -> bool:
        tokens = set(path.split("."))
        is_attn = bool({"q_proj", "k_proj", "v_proj", "out_proj"} & tokens)
        is_mlp = bool({"linear1", "linear2", "mlp1", "mlp2"} & tokens)
        is_patch = (
            path.endswith("patch_embedding.linear") or ".patch_embedding.linear" in path
        )
        is_head = "head" in tokens

        if mode == "attn_only":
            return is_attn
        if mode == "attn_mlp":
            return is_attn or is_mlp
        if mode == "all_linear":
            return is_attn or is_mlp or is_patch or is_head
        raise ValueError(f"Unknown mode: {mode}")

    return predicate


def _swin_predicate(mode: AttentionSVDMode) -> LinearPredicate:
    def predicate(path: str, _: eqx.nn.Linear) -> bool:
        tokens = set(path.split("."))
        is_attn = bool({"qkv", "proj"} & tokens)
        is_mlp = bool({"fc1", "fc2"} & tokens)
        is_downsample = "reduction" in tokens
        is_head = "head" in tokens

        if mode == "attn_only":
            return is_attn
        if mode == "attn_mlp":
            return is_attn or is_mlp
        if mode == "all_linear":
            return is_attn or is_mlp or is_downsample or is_head
        raise ValueError(f"Unknown mode: {mode}")

    return predicate


def _attention_predicate(
    family: AttentionFamily, mode: AttentionSVDMode
) -> LinearPredicate:
    if family in {"vit", "dino"}:
        return _vit_or_dino_predicate(mode)
    if family == "swin":
        return _swin_predicate(mode)
    raise ValueError(f"Unsupported attention family: {family!r}.")


def replace_linears_with_svd(
    model,
    *,
    rank: int | None = None,
    rank_ratio: float | None = None,
    alpha_init: float = 1.0,
    predicate: LinearPredicate | None = None,
):
    """Replace selected ``eqx.nn.Linear`` leaves with ``SVDDense``.

    By default this uses full rank, so the replacement is an exact transplant of
    the dense weights (up to numerical precision). Use ``rank_ratio < 1`` or
    ``rank < full_rank`` only when you intentionally want a truncated SVD.
    """

    predicate = (lambda _path, _leaf: True) if predicate is None else predicate
    is_linear_leaf = lambda node: isinstance(node, eqx.nn.Linear)

    leaves_with_path, treedef = jtu.tree_flatten_with_path(
        model, is_leaf=is_linear_leaf
    )
    new_leaves = []
    report = {
        "n_linear_total": 0,
        "n_replaced": 0,
        "replaced": [],
    }

    for path_entries, leaf in leaves_with_path:
        if not isinstance(leaf, eqx.nn.Linear):
            new_leaves.append(leaf)
            continue

        path = _path_to_str(path_entries)
        report["n_linear_total"] += 1

        if not predicate(path, leaf):
            new_leaves.append(leaf)
            continue

        chosen_rank = _resolve_rank(leaf, rank=rank, rank_ratio=rank_ratio)
        new_leaf = SVDDense.from_linear(leaf, rank=chosen_rank, alpha_init=alpha_init)
        new_leaves.append(new_leaf)

        report["n_replaced"] += 1
        report["replaced"].append(
            {
                "path": path,
                "rank": chosen_rank,
                "full_rank": _full_rank(leaf),
            }
        )

    new_model = jtu.tree_unflatten(treedef, new_leaves)
    return new_model, report


def replace_attention_linears_with_svd(
    model,
    *,
    family: AttentionFamily,
    mode: AttentionSVDMode = "attn_mlp",
    rank: int | None = None,
    rank_ratio: float | None = None,
    alpha_init: float = 1.0,
):
    """Retrofitting helper for ViT, Swin, and DINO attention-family models.

    Examples
    --------
    - ``family="vit",  mode="attn_only"`` replaces only q/k/v/out projections.
    - ``family="dino", mode="attn_mlp"`` replaces attention and MLP projections.
    - ``family="swin", mode="all_linear"`` also replaces patch-merging and head.
    """

    new_model, report = replace_linears_with_svd(
        model,
        rank=rank,
        rank_ratio=rank_ratio,
        alpha_init=alpha_init,
        predicate=_attention_predicate(family, mode),
    )
    report["family"] = family
    report["mode"] = mode
    return new_model, report


def replace_square_linears_with_rfft(
    model,
    *,
    alpha_init: float = 1.0,
    predicate: LinearPredicate | None = None,
    warmstart: bool = True,
    key: Any | None = None,
):
    """Replace selected square dense-like linear leaves with ``RFFTCirculant1D``.

    This is a thin public wrapper over the shared low-level retrofit helper in
    ``quantbayes.stochax.utils.linear_surgery``. Only square dense matrices are
    replaced. When ``warmstart=True`` they are projected onto the nearest
    circulant matrix in Frobenius norm and the resulting RFFT half-spectrum is
    copied into the new layer.
    """

    return _replace_square_linears_with_rfft(
        model,
        alpha_init=alpha_init,
        predicate=predicate,
        warmstart=warmstart,
        key=key,
    )


def replace_attention_linears_with_rfft(
    model,
    *,
    family: AttentionFamily,
    mode: AttentionSVDMode = "attn_only",
    alpha_init: float = 1.0,
    warmstart: bool = True,
    key: Any | None = None,
):
    """Retrofitting helper for ViT, Swin, and DINO attention-family models.

    Notes
    -----
    Only *square* dense projections are eligible for RFFT retrofits. Any matched
    non-square leaves are left unchanged and reported in ``report["skipped"]``.
    This matters in particular for Swin, where ``qkv`` has shape ``(3d, d)`` and
    is therefore skipped, while ``proj`` remains eligible.
    """

    new_model, report = replace_square_linears_with_rfft(
        model,
        alpha_init=alpha_init,
        predicate=_attention_predicate(family, mode),
        warmstart=warmstart,
        key=key,
    )
    report["family"] = family
    report["mode"] = mode
    return new_model, report


def make_svd_basis_freeze_mask(model, *, freeze_alpha: bool = True):
    """Return an Optax-style freeze mask that zeros updates to ``U`` and ``V``.

    ``True`` means "freeze this leaf". Use it with
    ``optax.masked(optax.set_to_zero(), mask)``.

    Set ``freeze_alpha=True`` to also freeze ``alpha_raw`` so the strict trainable
    set inside each ``SVDDense`` block is ``{s, bias}``.
    """

    names = ["U", "V"]
    if freeze_alpha:
        names.append("alpha_raw")
    return make_freeze_mask(model, names=tuple(names))


def make_s_only_freeze_mask(
    model,
    *,
    train_bias: bool = False,
    train_alpha: bool = False,
):
    """Return a freeze mask for the strict "train only s" regime.

    By default this freezes every inexact array except leaves named ``s``.
    Optionally allow ``bias`` and/or ``alpha_raw`` to remain trainable.
    """

    params_tree, _ = eqx.partition(model, eqx.is_inexact_array)
    leaves_with_path, treedef = jtu.tree_flatten_with_path(params_tree)

    masks = []
    for path_entries, leaf in leaves_with_path:
        if leaf is None:
            masks.append(False)
            continue

        tokens = set(_path_to_str(path_entries).split("."))
        train_leaf = "s" in tokens
        if train_bias and "bias" in tokens:
            train_leaf = True
        if train_alpha and "alpha_raw" in tokens:
            train_leaf = True
        masks.append(not train_leaf)

    return jtu.tree_unflatten(treedef, masks)
