from __future__ import annotations

"""Dense -> SVDDense/RFFT workflows for VAE models."""

from typing import Literal

from quantbayes.stochax.utils.linear_surgery import (
    LinearPredicate,
    make_s_only_freeze_mask,
    make_svd_basis_freeze_mask,
    replace_linears_with_svd,
    replace_square_linears_with_rfft,
    retrofit_linears,
)

VAERetrofitMode = Literal[
    "encoder_only",
    "decoder_only",
    "attention_only",
    "all_linear",
]
VAERetrofitVariant = Literal["dense", "svd", "rfft"]

__all__ = [
    "VAERetrofitMode",
    "VAERetrofitVariant",
    "replace_vae_linears_with_svd",
    "replace_vae_square_linears_with_rfft",
    "retrofit_vae_model",
    "make_svd_basis_freeze_mask",
    "make_s_only_freeze_mask",
]


def _vae_predicate(mode: VAERetrofitMode) -> LinearPredicate:
    def predicate(path: str, _leaf) -> bool:
        tokens = set(path.replace("[", ".[").split("."))
        in_encoder = "encoder" in tokens
        in_decoder = "decoder" in tokens
        is_attn = ("attn" in tokens) or ("attention" in tokens)

        if mode == "encoder_only":
            return in_encoder
        if mode == "decoder_only":
            return in_decoder
        if mode == "attention_only":
            return is_attn
        if mode == "all_linear":
            return True
        raise ValueError(f"Unknown VAE retrofit mode: {mode!r}.")

    return predicate


def replace_vae_linears_with_svd(
    model,
    *,
    mode: VAERetrofitMode = "all_linear",
    rank: int | None = None,
    rank_ratio: float | None = None,
    alpha_init: float = 1.0,
):
    new_model, report = replace_linears_with_svd(
        model,
        rank=rank,
        rank_ratio=rank_ratio,
        alpha_init=alpha_init,
        predicate=_vae_predicate(mode),
    )
    report["mode"] = mode
    return new_model, report


def replace_vae_square_linears_with_rfft(
    model,
    *,
    mode: VAERetrofitMode = "all_linear",
    alpha_init: float = 1.0,
    warmstart: bool = True,
    key=None,
):
    new_model, report = replace_square_linears_with_rfft(
        model,
        alpha_init=alpha_init,
        predicate=_vae_predicate(mode),
        warmstart=warmstart,
        key=key,
    )
    report["mode"] = mode
    return new_model, report


def retrofit_vae_model(
    model,
    *,
    variant: VAERetrofitVariant,
    mode: VAERetrofitMode = "all_linear",
    rank: int | None = None,
    rank_ratio: float | None = None,
    alpha_init: float = 1.0,
    warmstart: bool = True,
    key=None,
):
    """Dispatch helper for dense/SVD/RFFT VAE experiments."""

    new_model, report = retrofit_linears(
        model,
        variant=variant,
        rank=rank,
        rank_ratio=rank_ratio,
        alpha_init=alpha_init,
        predicate=_vae_predicate(mode),
        warmstart=warmstart,
        key=key,
    )
    report["mode"] = mode
    return new_model, report
