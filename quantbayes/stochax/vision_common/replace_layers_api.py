# quantbayes/stochax/vision_common/replace_layers_api.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Literal, Optional

import equinox as eqx
import jax.random as jr

from quantbayes.stochax.vision_common.pretrained_resnet import (
    load_imagenet_resnet,
)
from quantbayes.stochax.vision_common.pretrained_vgg import load_imagenet_vgg
from quantbayes.stochax.vision_common.spectral_surgery import (
    spectralize_resnet_3x3_to_svdconv,
    spectralize_resnet_3x3_stride1_to_rfft,
    spectralize_resnet_linear_to_svddense,
    spectralize_cnn_3x3_stride1_to_rfft,  # NEW
    warmstart_rfft_from_vanilla,  # already present, reuse
    warmstart_svd_from_vanilla,  # already present, reuse
)
from quantbayes.stochax.vision_common.pretrained_vit import load_imagenet_vit
from quantbayes.stochax.vision_common.pretrained_convnext import (
    load_imagenet_convnext as _load_convnext,
)
from quantbayes.stochax.vision_common.pretrained_efficientnet import (
    load_imagenet_efficientnet as _load_eff,
)
from quantbayes.stochax.vision_common.pretrained_swin import (
    load_imagenet_swin as _load_swin,
)

SurgeryKind = Literal[
    "none",
    "svdconv",
    "rfftconv",
    "svddense_linear",
    "svdconv+svddense",
    "rfftconv+svddense",
]


# ------------------------- Core building blocks ------------------------- #
def apply_resnet_replacements(
    model: eqx.Module,
    *,
    kind: SurgeryKind,
    input_hw: Tuple[int, int] = (224, 224),
    alpha_init: float = 1.0,
    key: jr.PRNGKey = jr.PRNGKey(0),
    svddense_rank: int | None = None,  # NEW
    svddense_rank_cap: int = 512,  # NEW
) -> Tuple[eqx.Module, Dict[str, Any]]:
    """Apply layer replacements to a ResNet-style tree. Returns (new_model, aux)."""
    aux: Dict[str, Any] = {}
    if kind == "none":
        return model, aux

    k1, k2 = jr.split(key)

    if kind in ("svdconv", "svdconv+svddense"):
        model = spectralize_resnet_3x3_to_svdconv(model, alpha_init=alpha_init, key=k1)

    if kind in ("rfftconv", "rfftconv+svddense"):
        model, spatial = spectralize_resnet_3x3_stride1_to_rfft(
            model, input_hw=input_hw, key=k1
        )
        aux["rfft_spatial"] = spatial

    if kind in ("svddense_linear", "svdconv+svddense", "rfftconv+svddense"):
        model = spectralize_resnet_linear_to_svddense(
            model,
            alpha_init=alpha_init,
            key=k2,
            rank=svddense_rank,  # pass through
            rank_cap=svddense_rank_cap,  # pass through
        )

    return model, aux


def warmstart_resnet_from_imagenet(
    model: eqx.Module,
    npz_path: str,
    *,
    kind: SurgeryKind,
    strict_fc: bool = False,
    rfft_spatial: Optional[Dict[str, Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any]]:
    """Warm-start the (possibly spectralized) model from torchvision .npz."""
    if kind in ("svdconv", "svddense_linear", "svdconv+svddense"):
        sw = "svd"
    elif kind in ("rfftconv", "rfftconv+svddense"):
        sw = "fft"
    else:
        sw = "skip"

    new_model, report = load_imagenet_resnet(
        model,
        npz_path,
        strict_fc=strict_fc,
        spectral_warmstart=sw,
        rfft_spatial=rfft_spatial,
        verbose=verbose,
    )
    return new_model, report


# ------------------------- UNet encoder versions ------------------------- #
def apply_unet_encoder_replacements(
    unet: eqx.Module,
    *,
    kind: SurgeryKind,
    input_hw: Tuple[int, int] = (224, 224),
    alpha_init: float = 1.0,
    key: jr.PRNGKey = jr.PRNGKey(0),
    svddense_rank: int | None = None,  # NEW
    svddense_rank_cap: int = 512,  # NEW
) -> Tuple[eqx.Module, Dict[str, Any]]:
    """Apply replacements to UNet.encoder only."""
    aux: Dict[str, Any] = {}
    if kind == "none":
        return unet, aux

    enc = unet.encoder
    k1, k2 = jr.split(key)

    if kind in ("svdconv", "svdconv+svddense"):
        enc = spectralize_resnet_3x3_to_svdconv(enc, alpha_init=alpha_init, key=k1)

    if kind in ("rfftconv", "rfftconv+svddense"):
        enc, spatial = spectralize_resnet_3x3_stride1_to_rfft(
            enc, input_hw=input_hw, key=k1
        )
        aux["rfft_spatial"] = spatial

    if kind in ("svddense_linear", "svdconv+svddense", "rfftconv+svddense"):
        enc = spectralize_resnet_linear_to_svddense(
            enc,
            alpha_init=alpha_init,
            key=k2,
            rank=svddense_rank,  # NEW
            rank_cap=svddense_rank_cap,  # NEW
        )

    unet = eqx.tree_at(lambda m: m.encoder, unet, enc)
    return unet, aux


def _build_svddense_from_linear(
    lin: eqx.nn.Linear,
    *,
    alpha_init: float,
    key: jr.KeyArray,
    rank: int | None = None,  # NEW
    rank_cap: int = 512,  # NEW
):
    import jax
    import jax.numpy as jnp
    from quantbayes.stochax.layers.spectral_layers import SVDDense  # type: ignore

    W = lin.weight
    out_features, in_features = int(W.shape[0]), int(W.shape[1])

    r_target = min(out_features, in_features)
    if rank is not None:
        r_target = min(r_target, int(rank))
    r = min(r_target, int(rank_cap))

    kU, kV, ks = jr.split(key, 3)
    # THIN QR for memory safety
    U0 = jr.normal(kU, (out_features, r), W.dtype)
    V0 = jr.normal(kV, (in_features, r), W.dtype)
    U = jnp.linalg.qr(U0, mode="reduced")[0]
    V = jnp.linalg.qr(V0, mode="reduced")[0]
    U = jax.lax.stop_gradient(U)
    V = jax.lax.stop_gradient(V)
    s = jr.normal(ks, (r,), W.dtype) * 0.01
    b = (
        lin.bias
        if getattr(lin, "bias", None) is not None
        else jnp.zeros((out_features,), W.dtype)
    )
    return SVDDense(U=U, V=V, s=s, bias=b, alpha_init=alpha_init)


VitSurgeryKind = Literal[
    "none",
    "svddense_linear",  # replace all eqx.nn.Linear (patch, qkv/out, mlp, head)
    "svddense_mlp_only",  # replace only MLP (fc1/fc2) + head
    "svddense_qkv_only",  # replace only q/k/v/out projections
    "svddense_qkv_mlp",  # replace q/k/v/out + MLP (+ head optional)
]


def apply_vit_replacements(
    model: eqx.Module,
    *,
    kind: Literal[
        "none",
        "svddense_linear",
        "svddense_mlp_only",
        "svddense_qkv_only",
        "svddense_qkv_mlp",
    ],
    alpha_init: float = 1.0,
    key: jr.KeyArray = jr.PRNGKey(0),
    replace_head: bool = True,
    svddense_rank: int | None = None,  # NEW
    svddense_rank_cap: int = 512,  # NEW
) -> Tuple[eqx.Module, Dict[str, Any]]:
    aux: Dict[str, Any] = {}
    if kind == "none":
        return model, aux

    keys = iter(jr.split(key, 100_000))

    def want(path: str) -> bool:
        if kind == "svddense_linear":
            return True
        if kind == "svddense_mlp_only":
            return (
                (".linear1" in path)
                or (".linear2" in path)
                or (replace_head and path.endswith("head"))
            )
        if kind == "svddense_qkv_only":
            return (
                (".attention.q_proj" in path)
                or (".attention.k_proj" in path)
                or (".attention.v_proj" in path)
                or (".attention.out_proj" in path)
            )
        if kind == "svddense_qkv_mlp":
            return (
                (".attention.q_proj" in path)
                or (".attention.k_proj" in path)
                or (".attention.v_proj" in path)
                or (".attention.out_proj" in path)
                or (".linear1" in path)
                or (".linear2" in path)
                or (replace_head and path.endswith("head"))
            )
        return False

    def _replace(obj, path: str):
        if isinstance(obj, eqx.nn.Linear) and want(path):
            return _build_svddense_from_linear(
                obj,
                alpha_init=alpha_init,
                key=next(keys),
                rank=svddense_rank,  # NEW
                rank_cap=svddense_rank_cap,  # NEW
            )
        return obj

    # generic pytree walk
    def walk(obj, prefix=""):
        if isinstance(obj, eqx.Module):
            new = obj
            for name, child in vars(obj).items():
                p = name if prefix == "" else f"{prefix}.{name}"
                rep = _replace(child, p)
                if rep is child:
                    rep = walk(child, p)
                if rep is not child:
                    try:
                        new = eqx.tree_at(lambda m: getattr(m, name), new, rep)
                    except TypeError:
                        object.__setattr__(new, name, rep)
            return new
        if isinstance(obj, tuple):
            return tuple(
                walk(c, f"{prefix}.{i}" if prefix else str(i))
                for i, c in enumerate(obj)
            )
        if isinstance(obj, list):
            return [
                walk(c, f"{prefix}.{i}" if prefix else str(i))
                for i, c in enumerate(obj)
            ]
        if isinstance(obj, dict):
            return {
                k: walk(v, f"{prefix}.{k}" if prefix else k) for k, v in obj.items()
            }
        return obj

    return walk(model), aux


def warmstart_vit_from_imagenet(
    model: eqx.Module,
    npz_path: str,
    *,
    kind: Literal[
        "none",
        "svddense_linear",
        "svddense_mlp_only",
        "svddense_qkv_only",
        "svddense_qkv_mlp",
    ],
    strict_fc: bool = True,
    verbose: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any]]:
    """Warm-start a (possibly spectralized) ViT from torchvision .npz."""
    sw = "svd" if kind != "none" else "skip"
    new_model, report = load_imagenet_vit(
        model,
        npz_path,
        strict_fc=strict_fc,
        spectral_warmstart=sw,  # SVD init for SVDDense everywhere we replaced
        verbose=verbose,
    )
    return new_model, report


def spectralize_and_warmstart_vit(
    model: eqx.Module,
    *,
    kind: Literal[
        "none",
        "svddense_linear",
        "svddense_mlp_only",
        "svddense_qkv_only",
        "svddense_qkv_mlp",
    ],
    npz_path: Optional[str] = None,
    alpha_init: float = 1.0,
    strict_fc: bool = True,
    key: jr.KeyArray = jr.PRNGKey(0),
    verbose: bool = True,
    replace_head: bool = True,
    svddense_rank: int | None = None,  # NEW
    svddense_rank_cap: int = 512,  # NEW
) -> Tuple[eqx.Module, Dict[str, Any], Dict[str, Any]]:
    model, aux = apply_vit_replacements(
        model,
        kind=kind,
        alpha_init=alpha_init,
        key=key,
        replace_head=replace_head,
        svddense_rank=svddense_rank,  # NEW
        svddense_rank_cap=svddense_rank_cap,  # NEW
    )
    report: Dict[str, Any] = {}
    if npz_path is not None:
        model, report = warmstart_vit_from_imagenet(
            model, npz_path, kind=kind, strict_fc=strict_fc, verbose=verbose
        )
    return model, report, aux


def warmstart_unet_encoder_from_imagenet_with_kind(
    unet: eqx.Module,
    npz_path: str,
    *,
    kind: SurgeryKind,
    strict_fc: bool = False,
    rfft_spatial: Optional[Dict[str, Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any]]:
    """Warm-start UNet.encoder only from torchvision .npz."""
    if kind in ("svdconv", "svddense_linear", "svdconv+svddense"):
        sw = "svd"
    elif kind in ("rfftconv", "rfftconv+svddense"):
        sw = "fft"
    else:
        sw = "skip"

    new_enc, report = load_imagenet_resnet(
        unet.encoder,
        npz_path,
        strict_fc=strict_fc,
        spectral_warmstart=sw,
        rfft_spatial=rfft_spatial,
        verbose=verbose,
    )
    unet = eqx.tree_at(lambda m: m.encoder, unet, new_enc)
    return unet, report


def warmstart_vgg_from_imagenet(
    model: eqx.Module,
    npz_path: str,
    *,
    kind: SurgeryKind,
    strict_fc: bool = False,
    rfft_spatial: Optional[Dict[str, Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any]]:
    """Warm-start a (possibly spectralized) VGG-style model from torchvision .npz."""
    if kind in ("svdconv", "svddense_linear", "svdconv+svddense"):
        sw = "svd"
    elif kind in ("rfftconv", "rfftconv+svddense"):
        sw = "fft"
    else:
        sw = "skip"

    new_model, report = load_imagenet_vgg(
        model,
        npz_path,
        strict_fc=strict_fc,
        spectral_warmstart=sw,
        rfft_spatial=rfft_spatial,
        verbose=verbose,
    )
    return new_model, report


def spectralize_and_warmstart_vgg(
    model: eqx.Module,
    *,
    kind: SurgeryKind,
    npz_path: Optional[str] = None,
    input_hw: Tuple[int, int] = (224, 224),
    svddense_rank: int | None = None,  # NEW
    svddense_rank_cap: int = 512,  # NEW
    alpha_init: float = 1.0,
    strict_fc: bool = False,
    key: jr.PRNGKey = jr.PRNGKey(0),
    verbose: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any], Dict[str, Any]]:
    """
    One call that:
      1) applies CNN-agnostic spectral surgery to `model` (works for VGG)
      2) optionally warm-starts from a torchvision VGG `.npz`
    Returns (new_model, report, aux). `aux` may contain rfft_spatial.
    """
    # Use the CNN-agnostic helpers to handle VGG reliably
    model, aux = apply_cnn_replacements(
        model,
        kind=kind,
        input_hw=input_hw,
        alpha_init=alpha_init,
        key=key,
        svddense_rank=svddense_rank,
        svddense_rank_cap=svddense_rank_cap,
    )
    report: Dict[str, Any] = {}
    if npz_path is not None:
        model, report = warmstart_vgg_from_imagenet(
            model,
            npz_path,
            kind=kind,
            strict_fc=strict_fc,
            rfft_spatial=aux.get("rfft_spatial"),
            verbose=verbose,
        )
    return model, report, aux


# ========================= NEW ONE-SHOT WRAPPERS ========================= #
def spectralize_and_warmstart_resnet(
    model: eqx.Module,
    *,
    kind: SurgeryKind,
    npz_path: Optional[str] = None,
    input_hw: Tuple[int, int] = (224, 224),
    svddense_rank: int | None = None,  # NEW
    svddense_rank_cap: int = 512,  # NEW
    alpha_init: float = 1.0,
    strict_fc: bool = False,
    key: jr.PRNGKey = jr.PRNGKey(0),
    verbose: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any], Dict[str, Any]]:
    """
    One call that:
      1) applies spectral surgery to `model`
      2) optionally warm-starts from `npz_path`
    Returns (new_model, report, aux). `aux` may contain rfft_spatial.
    """
    model, aux = apply_resnet_replacements(
        model,
        kind=kind,
        input_hw=input_hw,
        alpha_init=alpha_init,
        key=key,
        svddense_rank=svddense_rank,
        svddense_rank_cap=svddense_rank_cap,
    )
    report: Dict[str, Any] = {}
    if npz_path is not None:
        model, report = warmstart_resnet_from_imagenet(
            model,
            npz_path,
            kind=kind,
            strict_fc=strict_fc,
            rfft_spatial=aux.get("rfft_spatial"),
            verbose=verbose,
        )
    return model, report, aux


def spectralize_and_warmstart_unet_encoder(
    unet: eqx.Module,
    *,
    kind: SurgeryKind,
    npz_path: Optional[str] = None,
    input_hw: Tuple[int, int] = (224, 224),
    svddense_rank: int | None = None,  # NEW
    svddense_rank_cap: int = 512,  # NEW
    alpha_init: float = 1.0,
    strict_fc: bool = False,
    key: jr.PRNGKey = jr.PRNGKey(0),
    verbose: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any], Dict[str, Any]]:
    """
    One call that:
      1) applies spectral surgery to UNet.encoder
      2) optionally warm-starts encoder from `npz_path`
    Returns (new_unet, report, aux). `aux` may contain rfft_spatial.
    """
    unet, aux = apply_unet_encoder_replacements(
        unet,
        kind=kind,
        input_hw=input_hw,
        alpha_init=alpha_init,
        key=key,
        svddense_rank=svddense_rank,
        svddense_rank_cap=svddense_rank_cap,
    )
    report: Dict[str, Any] = {}
    if npz_path is not None:
        unet, report = warmstart_unet_encoder_from_imagenet_with_kind(
            unet,
            npz_path,
            kind=kind,
            strict_fc=strict_fc,
            rfft_spatial=aux.get("rfft_spatial"),
            verbose=verbose,
        )
    return unet, report, aux


def apply_cnn_replacements(
    model: eqx.Module,
    *,
    kind: SurgeryKind,
    input_hw: Tuple[int, int] = (224, 224),
    alpha_init: float = 1.0,
    key: jr.PRNGKey = jr.PRNGKey(0),
    svddense_rank: int | None = None,  # NEW
    svddense_rank_cap: int = 512,  # NEW
) -> Tuple[eqx.Module, Dict[str, Any]]:
    """
    Generic CNN version (works for VGG, ResNet, UNet encoders, etc.).
    Uses CNN-agnostic RFFT spatial inference.
    """
    aux: Dict[str, Any] = {}
    if kind == "none":
        return model, aux

    k1, k2 = jr.split(key)

    if kind in ("svdconv", "svdconv+svddense"):
        model = spectralize_resnet_3x3_to_svdconv(model, alpha_init=alpha_init, key=k1)

    if kind in ("rfftconv", "rfftconv+svddense"):
        model, spatial = spectralize_cnn_3x3_stride1_to_rfft(
            model, input_hw=input_hw, key=k1
        )
        aux["rfft_spatial"] = spatial

    if kind in ("svddense_linear", "svdconv+svddense", "rfftconv+svddense"):
        model = spectralize_resnet_linear_to_svddense(
            model,
            alpha_init=alpha_init,
            key=k2,
            rank=svddense_rank,  # NEW
            rank_cap=svddense_rank_cap,  # NEW
        )

    return model, aux


def spectralize_and_warmstart_cnn(
    vanilla_model: eqx.Module,
    *,
    kind: SurgeryKind,
    input_hw: Tuple[int, int] = (224, 224),
    alpha_init: float = 1.0,
    key: jr.PRNGKey = jr.PRNGKey(0),
) -> Tuple[eqx.Module, Dict[str, Any], Dict[str, Any]]:
    """
    One-shot for arbitrary CNNs (e.g., VGG):
      1) apply replacements
      2) path-based warm-start from the *vanilla* model in memory
    Returns (spec_model, report, aux) where report may contain 'svd'/'rfft' subreports.
    """
    spec_model, aux = apply_cnn_replacements(
        vanilla_model, kind=kind, input_hw=input_hw, alpha_init=alpha_init, key=key
    )
    report: Dict[str, Any] = {}

    # Warm-start by kind (from the vanilla model you passed in)
    if "rfftconv" in kind:
        spec_model, rep_rfft = warmstart_rfft_from_vanilla(
            vanilla_model, spec_model, rfft_spatial=aux.get("rfft_spatial")
        )
        report["rfft"] = rep_rfft
    if "svdconv" in kind:
        spec_model, rep_svd = warmstart_svd_from_vanilla(vanilla_model, spec_model)
        report["svd"] = rep_svd

    return spec_model, report, aux


# === DINO / DINOv2 wrappers ===============================================
from quantbayes.stochax.vision_backbones.dino.dinov2_loader import (
    load_dinov2 as _load_dino,
)

VitSurgeryKind = Literal[
    "none",
    "svddense_linear",  # replace all eqx.nn.Linear with SVDDense
    "svddense_mlp_only",  # only MLP (fc1/fc2) + head
    "svddense_qkv_only",  # only Q/K/V/Out projections
    "svddense_qkv_mlp",  # qkv+out + mlp (+head optional via replace_head)
]


def apply_dino_replacements(
    model: eqx.Module,
    *,
    kind: Literal[
        "none",
        "svddense_linear",
        "svddense_mlp_only",
        "svddense_qkv_only",
        "svddense_qkv_mlp",
    ],
    alpha_init: float = 1.0,
    key: jr.KeyArray = jr.PRNGKey(0),
    replace_head: bool = True,
    svddense_rank: int | None = None,
    svddense_rank_cap: int = 512,
) -> tuple[eqx.Module, dict[str, Any]]:
    # We can share the ViT path rules because our DINO module uses identical
    # attribute names (patch_embedding.linear, attention_blocks[i].*, head, norm).
    return apply_vit_replacements(
        model,
        kind=kind,
        alpha_init=alpha_init,
        key=key,
        replace_head=replace_head,
        svddense_rank=svddense_rank,
        svddense_rank_cap=svddense_rank_cap,
    )


def warmstart_dino_from_pretrained(
    model: eqx.Module,
    npz_path: str,
    *,
    kind: Literal[
        "none",
        "svddense_linear",
        "svddense_mlp_only",
        "svddense_qkv_only",
        "svddense_qkv_mlp",
    ],
    strict_fc: bool = False,
    verbose: bool = True,
) -> tuple[eqx.Module, dict[str, Any]]:
    sw = "svd" if kind != "none" else "skip"
    new_model, report = _load_dino(
        model,
        npz_path,
        strict_fc=strict_fc,
        spectral_warmstart=sw,
        verbose=verbose,
    )
    return new_model, report


def spectralize_and_warmstart_dino(
    model: eqx.Module,
    *,
    kind: Literal[
        "none",
        "svddense_linear",
        "svddense_mlp_only",
        "svddense_qkv_only",
        "svddense_qkv_mlp",
    ],
    npz_path: str | None = None,
    alpha_init: float = 1.0,
    strict_fc: bool = False,
    key: jr.KeyArray = jr.PRNGKey(0),
    verbose: bool = True,
    replace_head: bool = True,
    svddense_rank: int | None = None,
    svddense_rank_cap: int = 512,
) -> tuple[eqx.Module, dict[str, Any], dict[str, Any]]:
    model, aux = apply_dino_replacements(
        model,
        kind=kind,
        alpha_init=alpha_init,
        key=key,
        replace_head=replace_head,
        svddense_rank=svddense_rank,
        svddense_rank_cap=svddense_rank_cap,
    )
    report: dict[str, Any] = {}
    if npz_path is not None:
        model, report = warmstart_dino_from_pretrained(
            model, npz_path, kind=kind, strict_fc=strict_fc, verbose=verbose
        )
    return model, report, aux


def spectralize_and_warmstart_convnext(
    model: eqx.Module,
    *,
    kind: SurgeryKind,  # recommend "svddense_linear" for ConvNeXt
    npz_path: str | None = None,
    input_hw: Tuple[int, int] = (224, 224),
    alpha_init: float = 1.0,
    strict_fc: bool = True,
    key: jr.KeyArray = jr.PRNGKey(0),
    svddense_rank: int | None = None,
    svddense_rank_cap: int = 512,
    verbose: bool = True,
) -> tuple[eqx.Module, dict[str, Any], dict[str, Any]]:
    # Surgery: use CNN-agnostic helpers; rfft only touches 3x3 s=1 so for ConvNeXt prefer SVDDense
    model, aux = apply_cnn_replacements(
        model,
        kind=kind,
        input_hw=input_hw,
        alpha_init=alpha_init,
        key=key,
        svddense_rank=svddense_rank,
        svddense_rank_cap=svddense_rank_cap,
    )
    report: dict[str, Any] = {}
    if npz_path is not None:
        sw = "svd" if ("svddense" in kind) else "skip"
        model, report = _load_convnext(
            model, npz_path, strict_fc=strict_fc, spectral_warmstart=sw, verbose=verbose
        )
    return model, report, aux


# ---------------------- EfficientNet ----------------------
def spectralize_and_warmstart_efficientnet(
    model: eqx.Module,
    *,
    kind: SurgeryKind,  # "svddense_linear" is safest; rfft covers only some 3x3
    npz_path: str | None = None,
    input_hw: Tuple[int, int] = (224, 224),
    alpha_init: float = 1.0,
    strict_fc: bool = True,
    key: jr.KeyArray = jr.PRNGKey(0),
    svddense_rank: int | None = None,
    svddense_rank_cap: int = 512,
    verbose: bool = True,
) -> tuple[eqx.Module, dict[str, Any], dict[str, Any]]:
    model, aux = apply_cnn_replacements(
        model,
        kind=kind,
        input_hw=input_hw,
        alpha_init=alpha_init,
        key=key,
        svddense_rank=svddense_rank,
        svddense_rank_cap=svddense_rank_cap,
    )
    report: dict[str, Any] = {}
    if npz_path is not None:
        sw = "svd" if ("svddense" in kind) else "skip"
        model, report = _load_eff(
            model, npz_path, strict_fc=strict_fc, spectral_warmstart=sw, verbose=verbose
        )
    return model, report, aux


# ---------------------- Swin: replacements + warm-start ----------------------
SwinSurgeryKind = Literal[
    "none",
    "svddense_linear",  # replace all Linear
    "svddense_attn_mlp",  # qkv/proj + MLP fc1/fc2 (+ head optional)
]


def apply_swin_replacements(
    model: eqx.Module,
    *,
    kind: SwinSurgeryKind,
    alpha_init: float = 1.0,
    key: jr.KeyArray = jr.PRNGKey(0),
    replace_head: bool = True,
    svddense_rank: int | None = None,
    svddense_rank_cap: int = 512,
) -> tuple[eqx.Module, dict[str, Any]]:
    if kind == "none":
        return model, {}
    # Walk and replace eqx.nn.Linear leaves based on path rules
    keys = iter(jr.split(key, 100_000))

    def want(path: str) -> bool:
        if kind == "svddense_linear":
            return isinstance(path, str)
        if kind == "svddense_attn_mlp":
            return (
                (".attn.qkv" in path)
                or (".attn.proj" in path)
                or (".mlp.fc1" in path)
                or (".mlp.fc2" in path)
                or (replace_head and path.endswith("head"))
            )
        return False

    def _replace(obj, path: str):
        if isinstance(obj, eqx.nn.Linear) and want(path):
            from quantbayes.stochax.vision_common.spectral_surgery import (
                _linear_to_svddense,
            )

            return _linear_to_svddense(
                obj,
                rank=svddense_rank,
                key=next(keys),
                alpha_init=alpha_init,
                rank_cap=svddense_rank_cap,
            )
        return obj

    def walk(obj, prefix=""):
        if isinstance(obj, eqx.Module):
            new = obj
            for name, child in vars(obj).items():
                p = name if prefix == "" else f"{prefix}.{name}"
                rep = _replace(child, p)
                if rep is child:
                    rep = walk(child, p)
                if rep is not child:
                    try:
                        new = eqx.tree_at(lambda m: getattr(m, name), new, rep)
                    except TypeError:
                        object.__setattr__(new, name, rep)
            return new
        if isinstance(obj, tuple):
            return tuple(
                walk(c, f"{prefix}.{i}" if prefix else str(i))
                for i, c in enumerate(obj)
            )
        if isinstance(obj, list):
            return [
                walk(c, f"{prefix}.{i}" if prefix else str(i))
                for i, c in enumerate(obj)
            ]
        if isinstance(obj, dict):
            return {
                k: walk(v, f"{prefix}.{k}" if prefix else k) for k, v in obj.items()
            }
        return obj

    return walk(model), {}


def spectralize_and_warmstart_swin(
    model: eqx.Module,
    *,
    kind: SwinSurgeryKind,
    npz_path: str | None = None,
    alpha_init: float = 1.0,
    strict_fc: bool = True,
    key: jr.KeyArray = jr.PRNGKey(0),
    replace_head: bool = True,
    svddense_rank: int | None = None,
    svddense_rank_cap: int = 512,
    verbose: bool = True,
) -> tuple[eqx.Module, dict[str, Any], dict[str, Any]]:
    model, aux = apply_swin_replacements(
        model,
        kind=kind,
        alpha_init=alpha_init,
        key=key,
        replace_head=replace_head,
        svddense_rank=svddense_rank,
        svddense_rank_cap=svddense_rank_cap,
    )
    report: dict[str, Any] = {}
    if npz_path is not None:
        sw = "svd" if kind != "none" else "skip"
        model, report = _load_swin(
            model, npz_path, strict_fc=strict_fc, spectral_warmstart=sw, verbose=verbose
        )
    return model, report, aux
