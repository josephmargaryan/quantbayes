"""
### Load in torchvision weights
!python quantbayes/stochax/save_imagenet_resnets.py
!python quantbayes/stochax/save_imagenet_vits.py
!python quantbayes/stochax/save_imagenet_vggs.py

## ResNet Classifier

import jax.random as jr
import jax.numpy as jnp
import equinox as eqx

from quantbayes.stochax.vision_classification.models.resnet import ResNetClassifier

from quantbayes.stochax.vision_common.replace_layers_api import (
    spectralize_and_warmstart_resnet,
)
BACKBONE = "resnet18"

# 2) Optimizer (AdamW) — U,V are frozen by construction; we also avoid WD on spectral params.
opt, opt_state, aux_opt = build_optimizer(
    model,
    OptimizerConfig(
        algorithm="adamw",
        lr=3e-4,
        weight_decay=0.05,
        decay_mask=DecayMaskConfig(
            forbid_svd_factors=True,        # never decay U/V
            decay_spectral_weights=False,   # don't decay s by default
            decay_spectral_alpha=False,     # don't decay alpha_raw/delta_z
        ),
        labels=LabelConfig(
            head_lr_mult=10.0,              # optional: larger LR for classifier head
            spectral_lr_mult=0.5,           # optional: smaller LR for spectral leaves
        ),
        clip_global_norm=1.0,
        agc_clip=None,
    ),
)
print("[opt groups]", aux_opt["summary"]["groups"])



key = jr.PRNGKey(0)
model, state = eqx.nn.make_with_state(ResNetClassifier)(
    backbone=BACKBONE, num_classes=10, key=key
)

# choose a variant: "none", "svdconv", "svdconv+svddense", "rfftconv", "rfftconv+svddense", "svddense_linear
KIND = "rfftconv"

model, report, aux = spectralize_and_warmstart_resnet(
    model,
    kind=KIND,
    npz_path="resnet18_imagenet.npz",   # or None to skip warm-start
    strict_fc=False,                     # skip fc.* if num_classes != 1000
    input_hw=(32, 32),                   # set to your train resolution (important for rfft)
    alpha_init=1.0,
    svddense_rank=None,         # None = full rank (min(out,in))
    svddense_rank_cap=512,      # safety cap for big FCs
    key=jr.PRNGKey(777),
    verbose=True,
)
print("[loader report]", report.get("coverage"), report.get("spectral_warmstarted"))



## ResNet UNet backbone

from quantbayes.stochax.vision_common.replace_layers_api import (
    spectralize_and_warmstart_unet_encoder,
)

from quantbayes.stochax.vision_segmentation.models.unet_backbone import UNetBackbone

# init UNet with ResNet encoder
unet, state = eqx.nn.make_with_state(UNetBackbone)(
    out_ch=1, backbone="resnet34", key=jr.PRNGKey(0)
)

KIND = "svdconv"   # choose a variant: "none", "svdconv", "svdconv+svddense", "rfftconv", "rfftconv+svddense"

unet, report, aux = spectralize_and_warmstart_unet_encoder(
    unet,
    kind=KIND,
    npz_path="resnet34_imagenet.npz",
    strict_fc=False,
    input_hw=(256, 256),                    # your training resolution
    alpha_init=1.0,
    svddense_rank=None,         # None = full rank (min(out,in))
    svddense_rank_cap=512,      # safety cap for big FCs
    key=jr.PRNGKey(123),
    verbose=True,
)
print("[encoder loader report]", report.get("coverage"), report.get("spectral_warmstarted"))


## VGG Classifier

import jax.random as jr
import equinox as eqx

from quantbayes.stochax.vision_classification.models.vgg import VGG  # your eqx VGG
from quantbayes.stochax.vision_common.replace_layers_api import spectralize_and_warmstart_vgg

key = jr.PRNGKey(0)
model, state = eqx.nn.make_with_state(VGG)(num_classes=10, arch="vgg16_bn", key=key)

KIND = "svddense_linear"  # or "none", "rfftconv+svddense", "svdconv+svddense", "svdconv", "rfftconv", "svddense_linear"
model, report, aux = spectralize_and_warmstart_vgg(
    model,
    kind=KIND,
    npz_path="vgg16_bn_imagenet.npz",
    strict_fc=False,        # skip classifier head if your num_classes != 1000
    input_hw=(224, 224),    # set to your train resolution (important for rfft)
    alpha_init=1.0,
    svddense_rank=None,         # None = full rank (min(out,in))
    svddense_rank_cap=512,      # safety cap for big FCs
    key=jr.PRNGKey(777),
    verbose=True,
)
print("[VGG loader report]", report.get("coverage"), report.get("spectral_warmstarted"))


## ViT Classifier

import equinox as eqx, jax.random as jr
from quantbayes.stochax.vision_classification.models.vit import VisionTransformer
from quantbayes.stochax.vision_common.replace_layers_api import spectralize_and_warmstart_vit

H = W = 224; patch = 16; num_patches = (H//patch)*(W//patch)
key = jr.PRNGKey(0)

model, state = eqx.nn.make_with_state(VisionTransformer)(
    embedding_dim=768, hidden_dim=768*4, num_heads=12, num_layers=12,
    dropout_rate=0.1, patch_size=patch, num_patches=num_patches,
    num_classes=1000, channels=3, key=key,
)

# Replace all linear layers → SVDDense + warm-start (SVD) from TV weights.
model, report, _ = spectralize_and_warmstart_vit(
    model,
    kind="svddense_linear",            # or "svddense_mlp_only", "svddense_qkv_only", "svddense_qkv_mlp", "none", "svddense_linear"
    npz_path="vit_b_16_imagenet.npz",
    strict_fc=True,                    # set False if your num_classes != 1000
    alpha_init=1.0,
    svddense_rank=None,         # None = full rank (min(out,in))
    svddense_rank_cap=512,      # safety cap for big FCs
    key=jr.PRNGKey(777),
    verbose=True,
)


# assume: model, state already created (e.g., a ViT-shaped module for your DINO weights)

from quantbayes.stochax.vision_common.replace_layers_api import spectralize_and_warmstart_dino

model, report, _ = spectralize_and_warmstart_dino(
    model,
    kind="svddense_qkv_mlp",          # or "svddense_linear"
    npz_path=None,                    # fill if you have a DINO .npz; else None
    strict_fc=False,
    key=jr.PRNGKey(7),
    verbose=True,
    svddense_rank=None,
    svddense_rank_cap=512,
)
print("[DINO loader]", report.get("coverage"), report.get("spectral_warmstarted"))

opt, opt_state, aux_opt = build_optimizer(model, OPT_CFG)
print("[DINO opt groups]", aux_opt["summary"]["groups"])


# assume: model, state already created (e.g., ConvNeXt(...))

from quantbayes.stochax.vision_common.replace_layers_api import spectralize_and_warmstart_convnext

model, report, _ = spectralize_and_warmstart_convnext(
    model,
    kind="svddense_linear",           # recommended for ConvNeXt
    npz_path=None,                    # fill if you have a .npz; else None
    strict_fc=True,
    key=jr.PRNGKey(9),
    verbose=True,
    svddense_rank=None,
    svddense_rank_cap=512,
)
print("[ConvNeXt loader]", report.get("coverage"), report.get("spectral_warmstarted"))

opt, opt_state, aux_opt = build_optimizer(model, OPT_CFG)
print("[ConvNeXt opt groups]", aux_opt["summary"]["groups"])

# assume: model, state already created (e.g., EfficientNet(...))

from quantbayes.stochax.vision_common.replace_layers_api import spectralize_and_warmstart_efficientnet

model, report, _ = spectralize_and_warmstart_efficientnet(
    model,
    kind="svddense_linear",           # safest broad coverage
    npz_path=None,                    # fill if you have a .npz; else None
    strict_fc=True,
    key=jr.PRNGKey(11),
    verbose=True,
    svddense_rank=None,
    svddense_rank_cap=512,
)
print("[EfficientNet loader]", report.get("coverage"), report.get("spectral_warmstarted"))

opt, opt_state, aux_opt = build_optimizer(model, OPT_CFG)
print("[EfficientNet opt groups]", aux_opt["summary"]["groups"])

# assume: model, state already created (e.g., SwinTransformer(...))

from quantbayes.stochax.vision_common.replace_layers_api import spectralize_and_warmstart_swin

model, report, _ = spectralize_and_warmstart_swin(
    model,
    kind="svddense_attn_mlp",         # qkv/proj + MLP (and head if present)
    npz_path=None,                    # fill if you have a .npz; else None
    strict_fc=True,
    key=jr.PRNGKey(13),
    verbose=True,
    svddense_rank=None,
    svddense_rank_cap=512,
)
print("[Swin loader]", report.get("coverage"), report.get("spectral_warmstarted"))

opt, opt_state, aux_opt = build_optimizer(model, OPT_CFG)
print("[Swin opt groups]", aux_opt["summary"]["groups"])


"""
