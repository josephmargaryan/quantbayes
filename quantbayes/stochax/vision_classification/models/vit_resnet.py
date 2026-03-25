"""
ViT–ResNet Hybrid (Equinox)

A Vision Transformer encoder stacked on top of a ResNet backbone (feature extractor).
- Single-sample forward with channel-first inputs [C, H, W]
- BatchNorm uses mode="batch" (no EMA state carried)
- All __call__ signatures are (self, x, key, state) -> (logits, state)
- Backbone weights can be loaded from torchvision (ImageNet) .npz files.

Backbones supported out-of-the-box: "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
Output stage selectable via `out_stage` ∈ {1,2,3,4} (default=4):
  stage 1: after layer1  (stride 4)
  stage 2: after layer2  (stride 8)
  stage 3: after layer3  (stride 16)
  stage 4: after layer4  (stride 32, default)

Positional embeddings:
  - Learnable 2D grid for a canonical image size (img_size)
  - Interpolated (bilinear) to match feature-map H×W at runtime
  - Separate learnable CLS token

Weight Loading (torchvision → Equinox backbone):
------------------------------------------------
1) Save torchvision weights once (same flow as your UNet-ResNet):
   ----------------------------------------------------------------
   # save_torchvision_resnets.py
   from pathlib import Path
   import numpy as np
   from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

   CHECKPOINTS = {
       "resnet18": resnet18,
       "resnet34": resnet34,
       "resnet50": resnet50,
       "resnet101": resnet101,
       "resnet152": resnet152,
   }

   def main():
       for name, builder in CHECKPOINTS.items():
           print(f"⇢ downloading {name} …")
           model = builder(weights="IMAGENET1K_V1")
           ckpt_path = Path(f"{name}_imagenet.npz")
           print(f"↳ saving → {ckpt_path}")
           np.savez(ckpt_path, **{k: v.cpu().numpy() for k, v in model.state_dict().items()})
           print(f"✓ done {ckpt_path}\\n")

   if __name__ == "__main__":
       main()
   ----------------------------------------------------------------

2) Initialize hybrid model and load backbone weights:
   ----------------------------------------------------------------
   import equinox as eqx, jax.random as jr, jax.numpy as jnp
   from quantbayes.stochax.vision_classification.models.vit_resnet_hybrid import (
       ViTResNetHybrid,
       load_imagenet_resnet34_backbone,
   )

   key = jr.PRNGKey(0)
   model, state = eqx.nn.make_with_state(ViTResNetHybrid)(
       backbone="resnet34",
       num_classes=1000,
       img_size=224,            # canonical image size for pos-embed init
       embed_dim=768,
       hidden_dim=3072,
       num_heads=12,
       num_layers=12,
       dropout_rate=0.1,
       out_stage=4,             # use final stride-32 features
       key=key,
   )

   # Load only the backbone from torchvision weights (.npz)
   model = load_imagenet_resnet34_backbone(model, "resnet34_imagenet.npz")
   ----------------------------------------------------------------

3) Train/eval with your usual utilities (single-sample model; your trainer handles batching).

Author: Joseph Margaryan (library conventions), hybrid composition and loader adapted for classification/Vision Transformer use.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.image as jimg

# Reuse blocks/specs from your ResNet module
from quantbayes.stochax.vision_classification.models.resnet import (
    BasicBlock,
    Bottleneck,
    _RESNET_SPECS,
)


# ----------------------------- Utilities ----------------------------- #
def _stage_stride(out_stage: int) -> int:
    # conv1: /2, maxpool: /2, layer2: /2, layer3: /2, layer4: /2
    # After layer1 -> /4; layer2 -> /8; layer3 -> /16; layer4 -> /32
    if out_stage == 1:
        return 4
    if out_stage == 2:
        return 8
    if out_stage == 3:
        return 16
    if out_stage == 4:
        return 32
    raise ValueError(f"Invalid out_stage={out_stage}; must be 1..4")


def _flatten_tokens(y: jnp.ndarray) -> jnp.ndarray:
    """[C, H, W] -> [H*W, C]"""
    return jnp.transpose(y, (1, 2, 0)).reshape((-1, y.shape[0]))


def _interpolate_pos_grid(
    pos_grid: jnp.ndarray, new_hw: Tuple[int, int]
) -> jnp.ndarray:
    """pos_grid: [H0, W0, D] -> resized to [H, W, D] using bilinear."""
    H0, W0, D = pos_grid.shape
    H, W = new_hw
    if (H, W) == (H0, W0):
        return pos_grid
    # jax.image.resize expects [*, H, W, C] or [H, W, C]
    return jimg.resize(pos_grid, (H, W, D), method="bilinear")


# ---------------------- ResNet Feature Backbone ---------------------- #
class ResNetBackboneFeatures(eqx.Module):
    """ResNet feature extractor (no FC), returns feature-map from a chosen stage."""

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    pool: eqx.nn.MaxPool2d
    layers1: Tuple[eqx.Module, ...]
    layers2: Tuple[eqx.Module, ...]
    layers3: Tuple[eqx.Module, ...]
    layers4: Tuple[eqx.Module, ...]
    backbone: str = eqx.field(static=True)
    out_stage: int = eqx.field(static=True)

    def __init__(self, *, backbone: str = "resnet34", out_stage: int = 4, key):
        if backbone not in _RESNET_SPECS:
            raise ValueError(f"Unknown backbone '{backbone}'.")
        if out_stage not in (1, 2, 3, 4):
            raise ValueError("out_stage must be in {1,2,3,4}.")
        spec = _RESNET_SPECS[backbone]
        Block = spec["block"]
        layer_sizes = spec["layers"]

        # keys
        num_blocks = sum(layer_sizes)
        ks = list(jr.split(key, 1 + num_blocks))  # conv1 + blocks (pool is static)

        self.conv1 = eqx.nn.Conv2d(3, 64, 7, stride=2, padding=3, key=ks[0])
        self.bn1 = eqx.nn.BatchNorm(64, axis_name="batch", mode="batch")
        self.pool = eqx.nn.MaxPool2d(3, 2, padding=1)

        def _make_layer(cin: int, cout: int, blocks: int, stride: int, kiter):
            mods: List[eqx.Module] = []
            c_in = cin
            for i in range(blocks):
                s = stride if i == 0 else 1
                mods.append(Block(c_in, cout, s, key=next(kiter)))
                c_in = cout * (4 if Block is Bottleneck else 1)
            return tuple(mods), c_in

        kiter = iter(ks[1:])
        self.layers1, ch1 = _make_layer(64, 64, layer_sizes[0], 1, kiter)
        self.layers2, ch2 = _make_layer(ch1, 128, layer_sizes[1], 2, kiter)
        self.layers3, ch3 = _make_layer(ch2, 256, layer_sizes[2], 2, kiter)
        self.layers4, _ = _make_layer(ch3, 512, layer_sizes[3], 2, kiter)

        self.backbone = backbone
        self.out_stage = out_stage

    def __call__(self, x, key, state):
        # x: [3,H,W]
        k0, key = jr.split(key)
        x = self.conv1(x, key=k0)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        x = self.pool(x)

        out1 = x
        for block in self.layers1:
            kb, key = jr.split(key)
            out1, state = block(out1, key=kb, state=state)

        out2 = out1
        for block in self.layers2:
            kb, key = jr.split(key)
            out2, state = block(out2, key=kb, state=state)

        out3 = out2
        for block in self.layers3:
            kb, key = jr.split(key)
            out3, state = block(out3, key=kb, state=state)

        out4 = out3
        for block in self.layers4:
            kb, key = jr.split(key)
            out4, state = block(out4, key=kb, state=state)

        if self.out_stage == 1:
            return out1, state
        if self.out_stage == 2:
            return out2, state
        if self.out_stage == 3:
            return out3, state
        return out4, state


# -------------------------- Transformer Blocks -------------------------- #
class AttentionBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        *,
        key,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.layer_norm1 = eqx.nn.LayerNorm(embed_dim)
        self.layer_norm2 = eqx.nn.LayerNorm(embed_dim)
        self.attention = eqx.nn.MultiheadAttention(num_heads, embed_dim, key=k1)
        self.linear1 = eqx.nn.Linear(embed_dim, hidden_dim, key=k2)
        self.linear2 = eqx.nn.Linear(hidden_dim, embed_dim, key=k3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        # x: [N_tokens, D]
        x_norm = jax.vmap(self.layer_norm1)(x)
        attn_out = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        x_norm = jax.vmap(self.layer_norm2)(x)
        mlp_hidden = jax.vmap(self.linear1)(x_norm)
        mlp_hidden = jax.nn.gelu(mlp_hidden)

        k1, k2 = jr.split(key, 2)
        mlp_hidden = self.dropout1(mlp_hidden, key=k1)
        mlp_out = jax.vmap(self.linear2)(mlp_hidden)
        mlp_out = self.dropout2(mlp_out, key=k2)
        x = x + mlp_out
        return x


# -------------------------- ViT–ResNet Hybrid -------------------------- #
class ViTResNetHybrid(eqx.Module):
    # Backbone
    backbone: ResNetBackboneFeatures
    proj: eqx.nn.Conv2d  # 1×1 projection to embed_dim

    # Transformer
    cls_token: jnp.ndarray  # [1, D]
    pos_grid: jnp.ndarray  # [H0, W0, D]; learnable 2D positional grid
    blocks: Tuple[AttentionBlock, ...]
    dropout: eqx.nn.Dropout
    head: eqx.nn.Sequential

    # Static config
    embed_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    img_size: int = eqx.field(static=True)
    out_stage: int = eqx.field(static=True)
    base_grid: Tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        *,
        backbone: str = "resnet34",
        out_stage: int = 4,
        img_size: int = 224,
        embed_dim: int = 768,
        hidden_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout_rate: float = 0.1,
        num_classes: int = 1000,
        key,
    ):
        # Keys
        k_backbone, k_proj, k_cls, k_pos, k_blocks, k_head = jr.split(key, 6)

        # Backbone feature extractor
        self.backbone = ResNetBackboneFeatures(
            backbone=backbone, out_stage=out_stage, key=k_backbone
        )

        # Compute base feature grid from canonical img_size
        stride = _stage_stride(out_stage)
        H0 = max(1, img_size // stride)
        W0 = max(1, img_size // stride)
        self.base_grid = (H0, W0)

        # 1×1 projection to tokens (C -> D)
        # We don't know C until runtime; but we can safely set in_ch to the
        # expected spec from backbone: last stage channels for chosen block.
        # For BasicBlock: stage4 channels = 512; Bottleneck: also 2048 at stage4.
        # Safer path: infer from spec:
        spec = _RESNET_SPECS[backbone]
        final_channels = (
            spec["channels"][out_stage]
            if len(spec["channels"]) > out_stage
            else spec["channels"][-1]
        )
        # Note: For basic (18/34): channels = [64,64,128,256,512]
        #       For bottleneck (50/101/152): [64,256,512,1024,2048]
        in_ch = final_channels

        self.proj = eqx.nn.Conv2d(in_ch, embed_dim, kernel_size=1, key=k_proj)

        # Learnable CLS token and 2D positional grid
        self.cls_token = jr.normal(k_cls, (1, embed_dim))
        self.pos_grid = jr.normal(k_pos, (H0, W0, embed_dim))

        # Transformer encoder
        blocks = []
        block_keys = jr.split(k_blocks, num_layers)
        for kb in block_keys:
            blocks.append(
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout_rate, key=kb)
            )
        self.blocks = tuple(blocks)

        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.head = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(embed_dim),
                eqx.nn.Linear(embed_dim, num_classes, key=k_head),
            ]
        )

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.img_size = img_size
        self.out_stage = out_stage

    def __call__(self, x, key, state):
        """
        x: [C,H,W] single sample, C should be 3
        Returns: (logits [num_classes], state)
        """
        # Backbone features
        kb, key = jr.split(key)
        feat, state = self.backbone(x, key=kb, state=state)  # [C', H', W']

        # 1×1 projection to embedding dim
        kproj, key = jr.split(key)
        feat = self.proj(feat, key=kproj)  # [D, H', W']

        # Tokens
        tokens = _flatten_tokens(feat)  # [H'*W', D]

        # Positional embeddings (interpolate grid to H'×W')
        Hf, Wf = feat.shape[-2], feat.shape[-1]
        pos_hw = _interpolate_pos_grid(self.pos_grid, (Hf, Wf))  # [H', W', D]
        pos_tokens = pos_hw.reshape((-1, self.embed_dim))  # [H'*W', D]

        # CLS + add pos
        x_tok = jnp.concatenate([self.cls_token, tokens], axis=0)  # [1+N, D]
        pos_all = jnp.concatenate([jnp.zeros_like(self.cls_token), pos_tokens], axis=0)
        x_tok = x_tok + pos_all

        # Dropout + Transformer
        keys = jr.split(key, self.num_layers + 1)
        x_tok = self.dropout(x_tok, key=keys[0])
        for blk, kblk in zip(self.blocks, keys[1:]):
            x_tok = blk(x_tok, key=kblk)

        # Classification head on CLS
        cls = x_tok[0]
        logits = self.head(cls)
        return logits, state


# ---------------------- Torchvision Backbone Loader ---------------------- #
def _rename_pt_key(k: str) -> str:
    # Map torchvision -> our module names (backbone only)
    k = k.replace("downsample.0.", "down_conv.")
    k = k.replace("downsample.1.", "down_bn.")
    k = k.replace("layer1.", "layers1.")
    k = k.replace("layer2.", "layers2.")
    k = k.replace("layer3.", "layers3.")
    k = k.replace("layer4.", "layers4.")
    return k  # conv1., bn1. stay; fc.* exists but will be ignored


def _copy_into_tree(obj, pt: Dict[str, jnp.ndarray], prefix: str = ""):
    """Recursively copy weights (conv/linear/batchnorm + nested tuples)."""
    if isinstance(obj, eqx.Module):
        for name, attr in vars(obj).items():
            full = f"{prefix}{name}"

            if isinstance(attr, eqx.nn.Conv2d):
                new_attr = attr
                if f"{full}.weight" in pt:
                    new_attr = eqx.tree_at(
                        lambda m: m.weight, new_attr, pt[f"{full}.weight"]
                    )
                if f"{full}.bias" in pt:
                    new_attr = eqx.tree_at(
                        lambda m: m.bias, new_attr, pt[f"{full}.bias"]
                    )
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, new_attr)
                continue

            if isinstance(attr, eqx.nn.Linear):
                new_attr = attr
                if f"{full}.weight" in pt:
                    new_attr = eqx.tree_at(
                        lambda m: m.weight, new_attr, pt[f"{full}.weight"]
                    )
                if f"{full}.bias" in pt:
                    new_attr = eqx.tree_at(
                        lambda m: m.bias, new_attr, pt[f"{full}.bias"]
                    )
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, new_attr)
                continue

            if isinstance(attr, eqx.nn.BatchNorm):
                w_key, b_key = f"{full}.weight", f"{full}.bias"
                if (w_key in pt) or (b_key in pt):
                    obj = eqx.tree_at(
                        lambda m: (getattr(m, name).weight, getattr(m, name).bias),
                        obj,
                        (
                            pt.get(w_key, getattr(attr, "weight")),
                            pt.get(b_key, getattr(attr, "bias")),
                        ),
                    )
                continue

            if isinstance(attr, tuple):
                new_tuple = []
                for i, child in enumerate(attr):
                    child_full = f"{full}.{i}"
                    new_tuple.append(_copy_into_tree(child, pt, prefix=child_full))
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, tuple(new_tuple))
                continue

            # skip static/None
        return obj
    if isinstance(obj, tuple):
        return tuple(_copy_into_tree(x, pt, prefix=prefix) for x in obj)
    return obj


def load_torchvision_resnet_backbone(
    model: ViTResNetHybrid, npz_path: str
) -> ViTResNetHybrid:
    """Load torchvision ResNet weights into `model.backbone` only."""
    import numpy as np

    raw = dict(np.load(npz_path))
    pt = {}
    for k, v in raw.items():
        nk = _rename_pt_key(k)
        # Skip FC weights (not present in backbone)
        if nk.startswith("fc."):
            continue
        pt[nk] = jnp.asarray(v)
    new_backbone = _copy_into_tree(model.backbone, pt, prefix="")
    return eqx.tree_at(lambda m: m.backbone, model, new_backbone)


def load_imagenet_resnet18_backbone(
    model: ViTResNetHybrid, npz="resnet18_imagenet.npz"
):
    return load_torchvision_resnet_backbone(model, npz)


def load_imagenet_resnet34_backbone(
    model: ViTResNetHybrid, npz="resnet34_imagenet.npz"
):
    return load_torchvision_resnet_backbone(model, npz)


def load_imagenet_resnet50_backbone(
    model: ViTResNetHybrid, npz="resnet50_imagenet.npz"
):
    return load_torchvision_resnet_backbone(model, npz)


def load_imagenet_resnet101_backbone(
    model: ViTResNetHybrid, npz="resnet101_imagenet.npz"
):
    return load_torchvision_resnet_backbone(model, npz)


def load_imagenet_resnet152_backbone(
    model: ViTResNetHybrid, npz="resnet152_imagenet.npz"
):
    return load_torchvision_resnet_backbone(model, npz)


# ----------------------------- Smoke test ----------------------------- #
if __name__ == "__main__":
    """
    Synthetic classification smoke test for ViT–ResNet Hybrid.
    Mirrors your SimpleCNN/ResNet tests; replace with a real dataset in practice.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import augmax
    from augmax import InputType
    import optax
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jr

    # Your training utilities (assumed available)
    from quantbayes.stochax import (
        train,
        predict,
        make_augmax_augment,
        multiclass_loss,
    )

    rng = np.random.RandomState(0)
    N, C, H, W, NUM_CLASSES = 1024, 3, 96, 96, 10
    X_np = rng.rand(N, C, H, W).astype("float32")
    y_np = rng.randint(0, NUM_CLASSES, size=(N,)).astype("int32")

    # train/val split
    split = int(0.8 * N)
    X_train, X_val = X_np[:split], X_np[split:]
    y_train, y_val = y_np[:split], y_np[split:]

    # light augmentation
    transform = augmax.Chain(
        augmax.HorizontalFlip(),
        augmax.Rotate(angle_range=10),
        input_types=[InputType.IMAGE, InputType.METADATA],
    )
    augment_fn = make_augmax_augment(transform)

    master_key = jr.PRNGKey(42)
    model_key, train_key = jr.split(master_key)

    # Instantiate hybrid
    model, state = eqx.nn.make_with_state(ViTResNetHybrid)(
        backbone="resnet34",
        out_stage=4,  # final feature map (stride 32)
        img_size=H,  # canonical size for pos-embed init; we set to current H
        embed_dim=256,
        hidden_dim=1024,
        num_heads=8,
        num_layers=6,
        dropout_rate=0.1,
        num_classes=NUM_CLASSES,
        key=model_key,
    )

    # (Optional) Load ImageNet weights for the backbone only
    # model = load_imagenet_resnet34_backbone(model, "resnet34_imagenet.npz")

    # Optimizer
    lr_sched = optax.cosine_decay_schedule(1e-3, decay_steps=400)
    optimizer = optax.adamw(
        learning_rate=lr_sched,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=1e-4,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Train
    best_model, best_state, tr_loss, va_loss = train(
        model=model,
        state=state,
        opt_state=opt_state,
        optimizer=optimizer,
        loss_fn=multiclass_loss,
        X_train=jnp.array(X_train),
        y_train=jnp.array(y_train),
        X_val=jnp.array(X_val),
        y_val=jnp.array(y_val),
        batch_size=64,
        num_epochs=10,
        patience=3,
        key=train_key,
        augment_fn=augment_fn,
        lambda_spec=0.0,
    )

    # Eval
    logits = predict(best_model, best_state, jnp.array(X_val), train_key)
    print("Predictions shape:", logits.shape)

    # Curves
    plt.figure()
    plt.plot(tr_loss, label="train")
    plt.plot(va_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("ViT–ResNet Hybrid (resnet34, out_stage=4)")
    plt.show()
