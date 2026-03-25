# quantbayes/stochax/vision_segmentation/load_pretrained_weights/load_resnet.py
from __future__ import annotations
import equinox as eqx

from quantbayes.stochax.vision_common.pretrained_resnet import (
    load_imagenet_resnet as _load,
    load_imagenet_resnet18 as _load18,
    load_imagenet_resnet34 as _load34,
    load_imagenet_resnet50 as _load50,
)


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


def load_imagenet_resnet(eqx_encoder: eqx.Module, npz_path: str, **kw) -> eqx.Module:
    new_enc, _ = _load(eqx_encoder, npz_path, **kw)
    return new_enc


def load_imagenet_resnet18(enc, npz="resnet18_imagenet.npz", **kw):
    return _load18(enc, npz, **kw)[0]


def load_imagenet_resnet34(enc, npz="resnet34_imagenet.npz", **kw):
    return _load34(enc, npz, **kw)[0]


def load_imagenet_resnet50(enc, npz="resnet50_imagenet.npz", **kw):
    return _load50(enc, npz, **kw)[0]
