from __future__ import annotations

"""Public pretrained-weight loading API.

This module is the plain, non-spectral entry point for loading pretrained
checkpoints into vision models. The goal is to keep the common path simple:
instantiate the model you actually want, then load weights into it.

The lower-level `replace_layers_api` module still exists for experiments that
*change* the architecture (for example spectral surgery), but it is no longer
required for ordinary transfer learning.
"""

from typing import Literal, Optional

import equinox as eqx


Family = Literal[
    "resnet",
    "vit",
    "vgg",
    "convnext",
    "efficientnet",
    "swin",
    "inception",
    "vit_resnet_backbone",
    "dino",
    "unet_encoder",
]

AutoFamily = Family | Literal["auto"]


def _iter_tree(obj):
    yield obj
    if isinstance(obj, eqx.Module):
        for child in vars(obj).values():
            yield from _iter_tree(child)
    elif isinstance(obj, (list, tuple)):
        for child in obj:
            yield from _iter_tree(child)
    elif isinstance(obj, dict):
        for child in obj.values():
            yield from _iter_tree(child)


def _contains_type(obj, typ) -> bool:
    return any(isinstance(node, typ) for node in _iter_tree(obj))


def _infer_linear_spectral_warmstart(model) -> Literal["skip", "svd", "rfft1d"]:
    try:
        from quantbayes.stochax.layers.spectral_layers import SVDDense, RFFTCirculant1D
    except Exception:
        return "skip"

    has_svd = _contains_type(model, SVDDense)
    has_rfft1d = _contains_type(model, RFFTCirculant1D)
    if has_svd and has_rfft1d:
        raise ValueError(
            "Mixed SVDDense and RFFTCirculant1D leaves are not supported by the simple loader. "
            "Use the lower-level vision_common.pretrained_* loaders directly."
        )
    if has_rfft1d:
        return "rfft1d"
    if has_svd:
        return "svd"
    return "skip"


def _require_npz(
    npz_path: Optional[str], default_name: Optional[str], family: str
) -> str:
    if npz_path is not None:
        return npz_path
    if default_name is not None:
        return default_name
    raise ValueError(
        f"npz_path is required for family={family!r}. Either pass it explicitly or use a model "
        "whose checkpoint name can be inferred unambiguously."
    )


def infer_pretrained_family(model) -> Family:
    """Infer the pretrained-loader family from a model instance.

    This is intentionally heuristic. It is meant to make the common path easy,
    not to guess every possible custom model. When inference is ambiguous, pass
    `family=` explicitly.
    """
    cls_name = type(model).__name__.lower()
    module_name = type(model).__module__.lower()

    if "vit_resnet" in module_name or "vitresnet" in cls_name:
        return "vit_resnet_backbone"
    if ".vision_backbones.dino" in module_name or "dino" in cls_name:
        return "dino"
    if hasattr(model, "encoder") and (
        "unet" in cls_name
        or hasattr(getattr(model, "encoder", None), "spec_name")
        or hasattr(getattr(model, "encoder", None), "layers1")
    ):
        return "unet_encoder"
    if (
        module_name.endswith(".rfft_swin")
        or module_name.endswith(".swin")
        or "swin" in cls_name
    ):
        return "swin"
    if (
        module_name.endswith(".rfft_vit")
        or module_name.endswith(".vit")
        or "visiontransformer" in cls_name
    ):
        return "vit"
    if module_name.endswith(".resnet") or "resnetclassifier" in cls_name:
        return "resnet"
    if module_name.endswith(".vgg") or cls_name == "vgg":
        return "vgg"
    if module_name.endswith(".convnext") or "convnext" in cls_name:
        return "convnext"
    if module_name.endswith(".efficient_net") or "efficientnet" in cls_name:
        return "efficientnet"
    if module_name.endswith(".inception") or "inception" in cls_name:
        return "inception"

    if all(
        hasattr(model, name)
        for name in ("layers1", "layers2", "layers3", "layers4", "fc")
    ):
        return "resnet"
    if all(
        hasattr(model, name) for name in ("patch_embedding", "attention_blocks", "head")
    ):
        return "vit"
    if (
        all(hasattr(model, name) for name in ("features", "norm", "head"))
        and isinstance(getattr(model, "arch", None), str)
        and getattr(model, "arch").startswith("swin")
    ):
        return "swin"

    raise ValueError(
        "Could not infer pretrained family from model type "
        f"{type(model).__module__}.{type(model).__name__}. Pass family= explicitly."
    )


def default_pretrained_checkpoint(
    model, *, family: Family | None = None
) -> Optional[str]:
    """Return the default torchvision-style `.npz` name when it is inferable."""
    fam = infer_pretrained_family(model) if family is None else family

    if fam == "resnet":
        backbone = getattr(model, "backbone", None)
        return f"{backbone}_imagenet.npz" if isinstance(backbone, str) else None

    if fam in {"vgg", "convnext", "efficientnet", "swin"}:
        arch = getattr(model, "arch", None)
        return f"{arch}_imagenet.npz" if isinstance(arch, str) else None

    if fam == "inception":
        return "inception_v3_imagenet.npz"

    if fam == "vit_resnet_backbone":
        backbone_mod = getattr(model, "backbone", None)
        backbone_name = getattr(backbone_mod, "backbone", None)
        return (
            f"{backbone_name}_imagenet.npz" if isinstance(backbone_name, str) else None
        )

    if fam == "unet_encoder":
        encoder = getattr(model, "encoder", None)
        spec_name = getattr(encoder, "spec_name", None)
        return f"{spec_name}_imagenet.npz" if isinstance(spec_name, str) else None

    # ViT and DINO typically require an explicit checkpoint choice because the
    # architecture variant is not encoded in a stable public attribute.
    return None


def load_pretrained_resnet(
    model,
    npz_path: Optional[str] = None,
    *,
    strict_fc: bool = True,
):
    """Load torchvision-style ResNet weights into a dense ResNet classifier."""
    from quantbayes.stochax.vision_classification.models.resnet import (
        load_torchvision_resnet,
    )

    return load_torchvision_resnet(
        model,
        _require_npz(
            npz_path, default_pretrained_checkpoint(model, family="resnet"), "resnet"
        ),
        strict_fc=strict_fc,
    )


def load_pretrained_vit(
    model,
    npz_path: Optional[str] = None,
    *,
    strict_fc: bool = True,
    verbose: bool = True,
):
    """Load ViT weights into a dense, SVDDense, or RFFT-1D ViT-style model."""
    from quantbayes.stochax.vision_common.pretrained_vit import load_imagenet_vit

    spectral = _infer_linear_spectral_warmstart(model)
    loaded, _ = load_imagenet_vit(
        model,
        _require_npz(
            npz_path, default_pretrained_checkpoint(model, family="vit"), "vit"
        ),
        strict_fc=strict_fc,
        spectral_warmstart=spectral,
        verbose=verbose,
    )
    return loaded


def load_pretrained_vgg(
    model,
    npz_path: Optional[str] = None,
    *,
    strict_fc: bool = True,
):
    """Load torchvision-style VGG weights."""
    from quantbayes.stochax.vision_classification.models.vgg import load_torchvision_vgg

    return load_torchvision_vgg(
        model,
        _require_npz(
            npz_path, default_pretrained_checkpoint(model, family="vgg"), "vgg"
        ),
        strict_fc=strict_fc,
    )


def load_pretrained_convnext(
    model,
    npz_path: Optional[str] = None,
    *,
    strict_fc: bool = True,
):
    """Load torchvision-style ConvNeXt weights."""
    from quantbayes.stochax.vision_classification.models.convnext import (
        load_torchvision_convnext,
    )

    return load_torchvision_convnext(
        model,
        _require_npz(
            npz_path,
            default_pretrained_checkpoint(model, family="convnext"),
            "convnext",
        ),
        strict_fc=strict_fc,
    )


def load_pretrained_efficientnet(
    model,
    npz_path: Optional[str] = None,
    *,
    strict_fc: bool = True,
):
    """Load torchvision-style EfficientNet weights."""
    from quantbayes.stochax.vision_classification.models.efficient_net import (
        load_torchvision_efficientnet,
    )

    return load_torchvision_efficientnet(
        model,
        _require_npz(
            npz_path,
            default_pretrained_checkpoint(model, family="efficientnet"),
            "efficientnet",
        ),
        strict_fc=strict_fc,
    )


def load_pretrained_swin(
    model,
    npz_path: Optional[str] = None,
    *,
    strict_fc: bool = True,
    verbose: bool = True,
):
    """Load Swin weights into a dense, SVDDense, or RFFT-1D Swin-style model."""
    from quantbayes.stochax.vision_common.pretrained_swin import load_imagenet_swin

    spectral = _infer_linear_spectral_warmstart(model)
    loaded, _ = load_imagenet_swin(
        model,
        _require_npz(
            npz_path, default_pretrained_checkpoint(model, family="swin"), "swin"
        ),
        strict_fc=strict_fc,
        spectral_warmstart=spectral,
        verbose=verbose,
    )
    return loaded


def load_pretrained_inception(
    model,
    npz_path: Optional[str] = None,
    *,
    strict_fc: bool = True,
):
    """Load torchvision-style Inception-v3 weights."""
    from quantbayes.stochax.vision_classification.models.inception import (
        load_torchvision_inception_v3,
    )

    return load_torchvision_inception_v3(
        model,
        _require_npz(
            npz_path,
            default_pretrained_checkpoint(model, family="inception"),
            "inception",
        ),
        strict_fc=strict_fc,
    )


def load_pretrained_vit_resnet_backbone(model, npz_path: Optional[str] = None):
    """Load a torchvision ResNet checkpoint into the ResNet backbone of a ViT-ResNet hybrid."""
    from quantbayes.stochax.vision_classification.models.vit_resnet import (
        load_torchvision_resnet_backbone,
    )

    return load_torchvision_resnet_backbone(
        model,
        _require_npz(
            npz_path,
            default_pretrained_checkpoint(model, family="vit_resnet_backbone"),
            "vit_resnet_backbone",
        ),
    )


def load_pretrained_dino(
    model,
    npz_path: Optional[str] = None,
    *,
    strict_fc: bool = False,
    verbose: bool = True,
):
    """Load DINO/DINOv2-style weights into a dense or SVDDense backbone."""
    from quantbayes.stochax.vision_backbones.dino.dinov2_loader import load_dinov2

    spectral = _infer_linear_spectral_warmstart(model)
    if spectral == "rfft1d":
        raise ValueError(
            "DINO loader currently supports dense or SVDDense leaves, not RFFTCirculant1D."
        )
    loaded, _ = load_dinov2(
        model,
        _require_npz(
            npz_path, default_pretrained_checkpoint(model, family="dino"), "dino"
        ),
        strict_fc=strict_fc,
        spectral_warmstart=spectral,
        verbose=verbose,
    )
    return loaded


def load_pretrained_unet_encoder(
    unet,
    npz_path: Optional[str] = None,
    *,
    strict_fc: bool = False,
):
    """Load a torchvision ResNet checkpoint into `unet.encoder` only."""
    from quantbayes.stochax.vision_segmentation.load_pretrained_weights.load_resnet import (
        load_imagenet_resnet,
    )

    encoder = getattr(unet, "encoder", None)
    new_encoder = load_imagenet_resnet(
        encoder,
        _require_npz(
            npz_path,
            default_pretrained_checkpoint(unet, family="unet_encoder"),
            "unet_encoder",
        ),
        strict_fc=strict_fc,
    )
    return eqx.tree_at(lambda m: m.encoder, unet, new_encoder)


def load_pretrained(
    model,
    npz_path: Optional[str] = None,
    *,
    family: AutoFamily = "auto",
    strict_fc: bool | None = None,
    verbose: bool | None = None,
):
    """Generic plain pretrained loader.

    Parameters
    ----------
    model:
        Instantiated model in its final architecture.
    npz_path:
        Path to a `.npz` checkpoint. If omitted, the loader will try to infer the
        standard torchvision-style filename from model metadata.
    family:
        Explicit loader family or `"auto"` to infer from the model instance.
    strict_fc:
        If `None`, use the family-specific default. Set `False` for common
        transfer-learning cases where the classifier head shape differs.
    verbose:
        Optional verbosity override for loaders that expose detailed reports.
    """
    fam = infer_pretrained_family(model) if family == "auto" else family

    if strict_fc is None:
        strict_fc = False if fam in {"dino", "unet_encoder"} else True
    if verbose is None:
        verbose = True

    if fam == "resnet":
        return load_pretrained_resnet(model, npz_path, strict_fc=strict_fc)
    if fam == "vit":
        return load_pretrained_vit(
            model, npz_path, strict_fc=strict_fc, verbose=verbose
        )
    if fam == "vgg":
        return load_pretrained_vgg(model, npz_path, strict_fc=strict_fc)
    if fam == "convnext":
        return load_pretrained_convnext(model, npz_path, strict_fc=strict_fc)
    if fam == "efficientnet":
        return load_pretrained_efficientnet(model, npz_path, strict_fc=strict_fc)
    if fam == "swin":
        return load_pretrained_swin(
            model, npz_path, strict_fc=strict_fc, verbose=verbose
        )
    if fam == "inception":
        return load_pretrained_inception(model, npz_path, strict_fc=strict_fc)
    if fam == "vit_resnet_backbone":
        return load_pretrained_vit_resnet_backbone(model, npz_path)
    if fam == "dino":
        return load_pretrained_dino(
            model, npz_path, strict_fc=strict_fc, verbose=verbose
        )
    if fam == "unet_encoder":
        return load_pretrained_unet_encoder(model, npz_path, strict_fc=strict_fc)
    raise ValueError(f"Unsupported family={fam!r}.")


def load_pretrained_model(
    model,
    *,
    family: AutoFamily = "auto",
    npz_path: Optional[str] = None,
    strict_fc: bool | None = None,
    verbose: bool | None = None,
):
    """Backward-compatible alias for :func:`load_pretrained`."""
    return load_pretrained(
        model,
        npz_path=npz_path,
        family=family,
        strict_fc=strict_fc,
        verbose=verbose,
    )


__all__ = [
    "AutoFamily",
    "Family",
    "default_pretrained_checkpoint",
    "infer_pretrained_family",
    "load_pretrained",
    "load_pretrained_model",
    "load_pretrained_resnet",
    "load_pretrained_vit",
    "load_pretrained_vgg",
    "load_pretrained_convnext",
    "load_pretrained_efficientnet",
    "load_pretrained_swin",
    "load_pretrained_inception",
    "load_pretrained_vit_resnet_backbone",
    "load_pretrained_dino",
    "load_pretrained_unet_encoder",
]
