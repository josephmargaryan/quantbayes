from __future__ import annotations
import re
from typing import Dict

import numpy as np
import jax.numpy as jnp
import equinox as eqx


"""
How to use example:

import equinox as eqx, jax.numpy as jnp, numpy as np
import jax.random as jr
from quantbayes.stochax.vision_segmentation.load_pretrained_weights import load_imagenet_resnet34
from quantbayes.stochax.vision_segmentation.models.unet_backbone import UNet, ResNetEncoder

import torch, numpy as np
from torchvision.models import resnet34

torch_model = resnet34(weights="IMAGENET1K_V1")
np.savez("resnet34_imagenet.npz",
         **{k: v.cpu().numpy() for k, v in torch_model.state_dict().items()})
print("✓ saved resnet34_imagenet.npz")

key = jr.key(0)
model, state = eqx.nn.make_with_state(UNetBackbone)( #
    backbone="resnet34", in_ch=3, out_ch=1, key=key
)

new_encoder = load_imagenet_resnet34(model.encoder, "resnet34_imagenet.npz")

model = eqx.tree_at(lambda m: m.encoder,
                    model,
                    new_encoder,
                    is_leaf=lambda x: isinstance(x, ResNetEncoder))

print("✓ encoder initialised from ImageNet")

"""


def _rename_pt_key(k: str) -> str | None:
    if k.startswith(("conv1.", "bn1.")):
        return k
    if k.startswith("fc."):
        return None
    k = re.sub(r"^layer(\d+)\.", r"layers\1.", k)
    k = k.replace(".downsample.0.", ".down_conv.")
    k = k.replace(".downsample.1.", ".down_bn.")
    return k


def _build_keymap(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {new: v for k, v in raw.items() if (new := _rename_pt_key(k)) is not None}


def _copy_into(
    mod: eqx.Module, pt: Dict[str, np.ndarray], prefix: str = ""
) -> eqx.Module:

    for name, attr in vars(mod).items():
        full = f"{prefix}{name}"

        if isinstance(attr, eqx.nn.Conv2d) and f"{full}.weight" in pt:
            w = jnp.asarray(pt[f"{full}.weight"])
            if f"{full}.bias" in pt:
                b = jnp.asarray(pt[f"{full}.bias"])
                new_attr = eqx.tree_at(lambda m: (m.weight, m.bias), attr, (w, b))
            else:
                new_attr = eqx.tree_at(lambda m: m.weight, attr, w)

        elif isinstance(attr, eqx.nn.BatchNorm) and f"{full}.weight" in pt:
            new_attr = eqx.tree_at(
                lambda m: (m.weight, m.bias),
                attr,
                (
                    jnp.asarray(pt[f"{full}.weight"]),
                    jnp.asarray(pt[f"{full}.bias"]),
                ),
            )

        # 2.3 Nested module
        elif isinstance(attr, eqx.Module):
            new_attr = _copy_into(attr, pt, prefix=full + ".")

        # 2.4 Everything else (static fields, None, etc.)
        else:
            new_attr = attr

        # **Only modify when something changed (skip static ints/bools)**
        if new_attr is attr:
            continue

        mod = eqx.tree_at(lambda m: getattr(m, name), mod, new_attr)

    return mod


def load_imagenet_resnet(eqx_encoder: eqx.Module, npz_path: str) -> eqx.Module:
    pt = _build_keymap(np.load(npz_path))
    return _copy_into(eqx_encoder, pt)


def load_imagenet_resnet18(enc, npz="resnet18_imagenet.npz"):
    return load_imagenet_resnet(enc, npz)


def load_imagenet_resnet34(enc, npz="resnet34_imagenet.npz"):
    return load_imagenet_resnet(enc, npz)


def load_imagenet_resnet50(enc, npz="resnet50_imagenet.npz"):
    return load_imagenet_resnet(enc, npz)
