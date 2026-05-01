from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from typing import Any, Literal, Optional

import equinox as eqx
import jax.random as jr


Family = Literal["vit", "dino", "swin"]
Variant = Literal["dense", "rfft", "svd"]
Pool = Literal["cls", "mean_patch"]


@dataclass(frozen=True)
class ViTSpec:
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    patch_size: int
    checkpoint: str


@dataclass(frozen=True)
class DinoSpec:
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    patch_size: int


VIT_SPECS: dict[str, ViTSpec] = {
    "vit_b_16": ViTSpec(
        embedding_dim=768,
        hidden_dim=3072,
        num_heads=12,
        num_layers=12,
        patch_size=16,
        checkpoint="vit_b_16_imagenet.npz",
    ),
    "vit_b_32": ViTSpec(
        embedding_dim=768,
        hidden_dim=3072,
        num_heads=12,
        num_layers=12,
        patch_size=32,
        checkpoint="vit_b_32_imagenet.npz",
    ),
    "vit_l_16": ViTSpec(
        embedding_dim=1024,
        hidden_dim=4096,
        num_heads=16,
        num_layers=24,
        patch_size=16,
        checkpoint="vit_l_16_imagenet.npz",
    ),
    "vit_l_32": ViTSpec(
        embedding_dim=1024,
        hidden_dim=4096,
        num_heads=16,
        num_layers=24,
        patch_size=32,
        checkpoint="vit_l_32_imagenet.npz",
    ),
    # Kept aligned with your save script filename, even though the source weights
    # in torchvision are IMAGENET1K_SWAG_E2E_V1.
    "vit_h_14": ViTSpec(
        embedding_dim=1280,
        hidden_dim=5120,
        num_heads=16,
        num_layers=32,
        patch_size=14,
        checkpoint="vit_h_14_imagenet.npz",
    ),
}

DINO_SPECS: dict[str, DinoSpec] = {
    "vits14": DinoSpec(
        embedding_dim=384,
        hidden_dim=1536,
        num_heads=6,
        num_layers=12,
        patch_size=14,
    ),
    "vitb14": DinoSpec(
        embedding_dim=768,
        hidden_dim=3072,
        num_heads=12,
        num_layers=12,
        patch_size=14,
    ),
    "vitl14": DinoSpec(
        embedding_dim=1024,
        hidden_dim=4096,
        num_heads=16,
        num_layers=24,
        patch_size=14,
    ),
    "vitg14": DinoSpec(
        embedding_dim=1536,
        hidden_dim=6144,
        num_heads=24,
        num_layers=40,
        patch_size=14,
    ),
}

SWIN_ARCHES: tuple[str, ...] = ("swin_t", "swin_s", "swin_b")
SWIN_CHECKPOINTS: dict[str, str] = {
    "swin_t": "swin_t_imagenet.npz",
    "swin_s": "swin_s_imagenet.npz",
    "swin_b": "swin_b_imagenet.npz",
}


def available_arches(family: Family) -> tuple[str, ...]:
    if family == "vit":
        return tuple(VIT_SPECS.keys())
    if family == "dino":
        return tuple(DINO_SPECS.keys())
    if family == "swin":
        return SWIN_ARCHES
    raise ValueError(f"Unknown family: {family}")


def default_checkpoint_name(
    family: Family,
    arch: str,
    *,
    registers: bool = False,
) -> str:
    if family == "vit":
        return VIT_SPECS[arch].checkpoint
    if family == "dino":
        suffix = "reg" if registers else "noreg"
        return f"dinov2_{arch}_{suffix}.npz"
    if family == "swin":
        return SWIN_CHECKPOINTS[arch]
    raise ValueError(f"Unknown family: {family}")


def _resolve_symbol(candidates: list[tuple[str, str]]):
    errors: list[str] = []
    for module_name, symbol_name in candidates:
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, symbol_name):
                return getattr(mod, symbol_name)
            errors.append(f"{module_name}:{symbol_name} not found")
        except Exception as e:
            errors.append(f"{module_name}:{symbol_name} -> {e!r}")
    joined = "\n  - ".join(errors)
    raise ImportError(f"Could not resolve symbol from candidates:\n  - {joined}")


def _class_for(family: Family, variant: Variant):
    if family == "vit":
        if variant == "dense":
            return _resolve_symbol(
                [
                    (
                        "quantbayes.stochax.vision_classification.models",
                        "VisionTransformer",
                    ),
                    (
                        "quantbayes.stochax.vision_classification.models.vit",
                        "VisionTransformer",
                    ),
                ]
            )
        if variant == "rfft":
            return _resolve_symbol(
                [
                    (
                        "quantbayes.stochax.vision_classification.models",
                        "RFFTVisionTransformer",
                    ),
                    (
                        "quantbayes.stochax.vision_classification.models.rfft_vit",
                        "RFFTVisionTransformer",
                    ),
                    (
                        "quantbayes.stochax.vision_classification.models.rfft_vit",
                        "VisionTransformer",
                    ),
                ]
            )
        if variant == "svd":
            return _resolve_symbol(
                [
                    (
                        "quantbayes.stochax.vision_classification.models",
                        "SVDVisionTransformer",
                    ),
                    (
                        "quantbayes.stochax.vision_classification.models.svd_vit",
                        "SVDVisionTransformer",
                    ),
                ]
            )

    if family == "dino":
        if variant == "dense":
            return _resolve_symbol(
                [
                    (
                        "quantbayes.stochax.vision_backbones.dino",
                        "DinoVisionTransformer",
                    ),
                    (
                        "quantbayes.stochax.vision_backbones.dino.vit_dino_eqx",
                        "DinoVisionTransformer",
                    ),
                ]
            )
        if variant == "rfft":
            return _resolve_symbol(
                [
                    (
                        "quantbayes.stochax.vision_backbones.dino",
                        "RFFTDinoVisionTransformer",
                    ),
                    (
                        "quantbayes.stochax.vision_backbones.dino.rfft_vit_dino_eqx",
                        "RFFTDinoVisionTransformer",
                    ),
                ]
            )
        if variant == "svd":
            return _resolve_symbol(
                [
                    (
                        "quantbayes.stochax.vision_backbones.dino",
                        "SVDDinoVisionTransformer",
                    ),
                    (
                        "quantbayes.stochax.vision_backbones.dino.svd_vit_dino_eqx",
                        "SVDDinoVisionTransformer",
                    ),
                ]
            )

    if family == "swin":
        if variant == "dense":
            return _resolve_symbol(
                [
                    (
                        "quantbayes.stochax.vision_classification.models",
                        "SwinTransformer",
                    ),
                    (
                        "quantbayes.stochax.vision_classification.models.swin",
                        "SwinTransformer",
                    ),
                ]
            )
        if variant == "rfft":
            return _resolve_symbol(
                [
                    (
                        "quantbayes.stochax.vision_classification.models",
                        "RFFTSwinTransformer",
                    ),
                    (
                        "quantbayes.stochax.vision_classification.models.rfft_swin",
                        "RFFTSwinTransformer",
                    ),
                ]
            )
        if variant == "svd":
            return _resolve_symbol(
                [
                    (
                        "quantbayes.stochax.vision_classification.models",
                        "SVDSwinTransformer",
                    ),
                    (
                        "quantbayes.stochax.vision_classification.models.svd_swin",
                        "SVDSwinTransformer",
                    ),
                ]
            )

    raise ValueError(f"Unknown family/variant combination: {family}/{variant}")


def _filter_ctor_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs
    accepted = {name for name in sig.parameters if name != "self"}
    return {k: v for k, v in kwargs.items() if k in accepted}


def _variant_defaults(family: Family, variant: Variant) -> dict[str, Any]:
    if variant == "dense":
        return {}

    if variant == "rfft":
        # Some RFFT classes expose `use_spectral_proj`, some do not.
        # We pass it and then filter by constructor signature.
        return {
            "use_spectral_proj": True,
        }

    if variant == "svd":
        # These match the SVD scripts we discussed.
        defaults = {
            "svd_mode": "attn_mlp",
            "svd_rank_ratio": 0.25,
        }
        # If you later decide Swin/DINO want different defaults, override via kwargs.
        return defaults

    raise ValueError(f"Unknown variant: {variant}")


def _require_divisible(image_size: int, patch_size: int, *, family: str, arch: str):
    if image_size % patch_size != 0:
        raise ValueError(
            f"{family} arch={arch} expects image_size divisible by patch_size={patch_size}; "
            f"got image_size={image_size}."
        )


def make_vit_model(
    arch: str,
    *,
    variant: Variant = "dense",
    image_size: int = 224,
    num_classes: int = 1000,
    channels: int = 3,
    dropout_rate: float = 0.0,
    key=None,
    **variant_kwargs,
):
    spec = VIT_SPECS[arch]
    _require_divisible(image_size, spec.patch_size, family="vit", arch=arch)
    key = jr.PRNGKey(0) if key is None else key

    cls = _class_for("vit", variant)
    num_patches = (image_size // spec.patch_size) ** 2

    ctor = dict(
        embedding_dim=spec.embedding_dim,
        hidden_dim=spec.hidden_dim,
        num_heads=spec.num_heads,
        num_layers=spec.num_layers,
        patch_size=spec.patch_size,
        num_patches=num_patches,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        channels=channels,
        key=key,
    )
    ctor.update(_variant_defaults("vit", variant))
    ctor.update(variant_kwargs)
    ctor = _filter_ctor_kwargs(cls, ctor)

    model, state = eqx.nn.make_with_state(cls)(**ctor)
    return model, state


def make_dino_model(
    arch: str,
    *,
    variant: Variant = "dense",
    image_size: int = 224,
    num_classes: int = 1000,
    registers: bool = False,
    pool: Pool = "cls",
    dropout_rate: float = 0.0,
    key=None,
    **variant_kwargs,
):
    spec = DINO_SPECS[arch]
    _require_divisible(image_size, spec.patch_size, family="dino", arch=arch)
    key = jr.PRNGKey(0) if key is None else key

    cls = _class_for("dino", variant)
    num_patches = (image_size // spec.patch_size) ** 2

    ctor = dict(
        embedding_dim=spec.embedding_dim,
        hidden_dim=spec.hidden_dim,
        num_heads=spec.num_heads,
        num_layers=spec.num_layers,
        patch_size=spec.patch_size,
        num_patches=num_patches,
        num_classes=num_classes,
        n_register_tokens=4 if registers else 0,
        dropout_rate=dropout_rate,
        pool=pool,
        key=key,
    )
    ctor.update(_variant_defaults("dino", variant))
    ctor.update(variant_kwargs)
    ctor = _filter_ctor_kwargs(cls, ctor)

    model, state = eqx.nn.make_with_state(cls)(**ctor)
    return model, state


def make_swin_model(
    arch: str,
    *,
    variant: Variant = "dense",
    num_classes: int = 1000,
    key=None,
    **variant_kwargs,
):
    if arch not in SWIN_ARCHES:
        raise ValueError(f"Unknown swin arch: {arch}. Choices: {SWIN_ARCHES}")
    key = jr.PRNGKey(0) if key is None else key

    cls = _class_for("swin", variant)
    ctor = dict(
        arch=arch,
        num_classes=num_classes,
        key=key,
    )
    ctor.update(_variant_defaults("swin", variant))
    ctor.update(variant_kwargs)
    ctor = _filter_ctor_kwargs(cls, ctor)

    model, state = eqx.nn.make_with_state(cls)(**ctor)
    return model, state


def make_model(
    family: Family,
    arch: str,
    *,
    variant: Variant = "dense",
    image_size: int = 224,
    num_classes: int = 1000,
    registers: bool = False,
    pool: Pool = "cls",
    key=None,
    **variant_kwargs,
):
    if family == "vit":
        return make_vit_model(
            arch,
            variant=variant,
            image_size=image_size,
            num_classes=num_classes,
            key=key,
            **variant_kwargs,
        )
    if family == "dino":
        return make_dino_model(
            arch,
            variant=variant,
            image_size=image_size,
            num_classes=num_classes,
            registers=registers,
            pool=pool,
            key=key,
            **variant_kwargs,
        )
    if family == "swin":
        return make_swin_model(
            arch,
            variant=variant,
            num_classes=num_classes,
            key=key,
            **variant_kwargs,
        )
    raise ValueError(f"Unknown family: {family}")


def load_pretrained_for_spec(
    model,
    *,
    family: Family,
    arch: str,
    npz_path: Optional[str] = None,
    registers: bool = False,
    strict_fc: bool = False,
    verbose: bool = True,
):
    from quantbayes.stochax.vision_common import load_pretrained

    ckpt = npz_path or default_checkpoint_name(
        family,
        arch,
        registers=registers,
    )
    return load_pretrained(
        model,
        family=family,
        npz_path=ckpt,
        strict_fc=strict_fc,
        verbose=verbose,
    )


def make_and_load_model(
    family: Family,
    arch: str,
    *,
    variant: Variant = "dense",
    image_size: int = 224,
    num_classes: int = 1000,
    registers: bool = False,
    pool: Pool = "cls",
    npz_path: Optional[str] = None,
    strict_fc: bool = False,
    verbose: bool = True,
    key=None,
    **variant_kwargs,
):
    model, state = make_model(
        family,
        arch,
        variant=variant,
        image_size=image_size,
        num_classes=num_classes,
        registers=registers,
        pool=pool,
        key=key,
        **variant_kwargs,
    )
    model = load_pretrained_for_spec(
        model,
        family=family,
        arch=arch,
        npz_path=npz_path,
        registers=registers,
        strict_fc=strict_fc,
        verbose=verbose,
    )
    return model, state


__all__ = [
    "available_arches",
    "default_checkpoint_name",
    "VIT_SPECS",
    "DINO_SPECS",
    "SWIN_ARCHES",
    "SWIN_CHECKPOINTS",
    "make_vit_model",
    "make_dino_model",
    "make_swin_model",
    "make_model",
    "load_pretrained_for_spec",
    "make_and_load_model",
]


if __name__ == "__main__":
    print("ViT arches :", available_arches("vit"))
    print("DINO arches:", available_arches("dino"))
    print("Swin arches:", available_arches("swin"))

    print("Default ViT checkpoint :", default_checkpoint_name("vit", "vit_b_16"))
    print(
        "Default DINO checkpoint:",
        default_checkpoint_name("dino", "vitb14", registers=True),
    )
    print("Default Swin checkpoint:", default_checkpoint_name("swin", "swin_t"))
