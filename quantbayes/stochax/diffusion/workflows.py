from __future__ import annotations

"""Dense -> SVDDense/RFFT workflows for diffusion models.

These helpers mirror the successful vision workflow but stay intentionally thin:
model-family-specific path predicates live here, while the generic tree surgery
stays in :mod:`quantbayes.stochax.utils.linear_surgery`.
"""

from typing import Literal

from quantbayes.stochax.utils.linear_surgery import (
    LinearPredicate,
    make_s_only_freeze_mask,
    make_svd_basis_freeze_mask,
    replace_linears_with_svd,
    replace_square_linears_with_rfft,
    retrofit_linears,
)

DiffusionRetrofitMode = Literal["attn_only", "attn_mlp", "all_linear"]
DiffusionRetrofitVariant = Literal["dense", "svd", "rfft"]

__all__ = [
    "DiffusionRetrofitMode",
    "DiffusionRetrofitVariant",
    "replace_diffusion_linears_with_svd",
    "replace_diffusion_square_linears_with_rfft",
    "retrofit_diffusion_model",
    "make_svd_basis_freeze_mask",
    "make_s_only_freeze_mask",
]


def _diffusion_predicate(mode: DiffusionRetrofitMode) -> LinearPredicate:
    def predicate(path: str, _leaf) -> bool:
        tokens = set(path.replace("[", ".[").split("."))
        is_attn = ("attn" in tokens) or ("attention" in tokens)
        is_mlp = bool(
            {
                "mlp",
                "ada_mod",
                "adaLN_modulation",
                "ada_modulation",
            }
            & tokens
        )

        if mode == "attn_only":
            return is_attn
        if mode == "attn_mlp":
            return is_attn or is_mlp
        if mode == "all_linear":
            return True
        raise ValueError(f"Unknown diffusion retrofit mode: {mode!r}.")

    return predicate


def replace_diffusion_linears_with_svd(
    model,
    *,
    mode: DiffusionRetrofitMode = "all_linear",
    rank: int | None = None,
    rank_ratio: float | None = None,
    alpha_init: float = 1.0,
):
    new_model, report = replace_linears_with_svd(
        model,
        rank=rank,
        rank_ratio=rank_ratio,
        alpha_init=alpha_init,
        predicate=_diffusion_predicate(mode),
    )
    report["mode"] = mode
    return new_model, report


def replace_diffusion_square_linears_with_rfft(
    model,
    *,
    mode: DiffusionRetrofitMode = "all_linear",
    alpha_init: float = 1.0,
    warmstart: bool = True,
    key=None,
):
    new_model, report = replace_square_linears_with_rfft(
        model,
        alpha_init=alpha_init,
        predicate=_diffusion_predicate(mode),
        warmstart=warmstart,
        key=key,
    )
    report["mode"] = mode
    return new_model, report


def retrofit_diffusion_model(
    model,
    *,
    variant: DiffusionRetrofitVariant,
    mode: DiffusionRetrofitMode = "all_linear",
    rank: int | None = None,
    rank_ratio: float | None = None,
    alpha_init: float = 1.0,
    warmstart: bool = True,
    key=None,
):
    """Dispatch helper for dense/SVD/RFFT diffusion experiments.

    Typical use-cases:
      - ``variant="svd"`` for exact dense -> SVDDense transplant.
      - ``variant="rfft"`` for square dense -> RFFT1D replacement.
      - ``mode="attn_mlp"`` to focus on transformer-style attention + MLP blocks.
      - ``mode="all_linear"`` to include patch/time/final projections too.
    """

    new_model, report = retrofit_linears(
        model,
        variant=variant,
        rank=rank,
        rank_ratio=rank_ratio,
        alpha_init=alpha_init,
        predicate=_diffusion_predicate(mode),
        warmstart=warmstart,
        key=key,
    )
    report["mode"] = mode
    return new_model, report
