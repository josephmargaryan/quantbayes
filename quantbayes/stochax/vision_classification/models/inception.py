"""
Equinox Inception v3 (torchvision-compatible) with AuxLogits and weight loader.

- Single-sample forward with channel-first inputs [C, H, W].
- BatchNorm uses mode="batch" (no EMA / running stats).
- __call__(self, x, key, state) -> (logits, state).
- Includes AuxLogits head (same structure as torchvision). We *do not*
  return aux logits in __call__ to keep your training API simple; you can
  invoke the aux head manually if you need it for auxiliary loss.

Torchvision weights
-------------------
1) Save torchvision weights once:
   ----------------------------------------------------------------
   # save_torchvision_inception_v3.py
   from pathlib import Path
   import numpy as np
   from torchvision.models import inception_v3

   def main():
       print("⇢ downloading inception_v3 …")
       model = inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
       ckpt_path = Path("inception_v3_imagenet.npz")
       print(f"↳ saving → {ckpt_path}")
       np.savez(ckpt_path, **{k: v.cpu().numpy() for k, v in model.state_dict().items()})
       print(f"✓ done {ckpt_path}")

   if __name__ == "__main__":
       main()
   ----------------------------------------------------------------

2) Initialize and load into Equinox:
   ----------------------------------------------------------------
   import equinox as eqx, jax.random as jr
   from quantbayes.stochax.vision_classification.models.inception_v3 import (
       InceptionV3, load_imagenet_inception_v3
   )

   key = jr.PRNGKey(0)
   model, state = eqx.nn.make_with_state(InceptionV3)(
       num_classes=1000,
       aux_logits=True,   # instantiate Aux head so its weights load
       dropout=0.5,
       key=key,
   )

   # If your num_classes != 1000 and you want to reuse features:
   # load with strict_fc=False to skip final FC (and Aux FC).
   model = load_imagenet_inception_v3(model, "inception_v3_imagenet.npz", strict_fc=True)
   ----------------------------------------------------------------
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# --------------------------- small utilities --------------------------- #
class AdaptiveAvgPool2d(eqx.Module):
    """Exact AdaptiveAvgPool2d to (out_h, out_w) for single-sample CHW tensors."""

    out_h: int = eqx.field(static=True)
    out_w: int = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray):
        # x: [C, H, W] -> [C, out_h, out_w]
        C, H, W = x.shape
        # Reduce H
        h_slices = []
        for i in range(self.out_h):
            h0 = int(jnp.floor(i * H / self.out_h))
            h1 = int(jnp.ceil((i + 1) * H / self.out_h))
            h1 = max(h1, h0 + 1)
            h_slices.append(jnp.mean(x[:, h0:h1, :], axis=1))  # [C, W]
        tmp = jnp.stack(h_slices, axis=1)  # [C, out_h, W]
        # Reduce W
        w_slices = []
        for j in range(self.out_w):
            w0 = int(jnp.floor(j * W / self.out_w))
            w1 = int(jnp.ceil((j + 1) * W / self.out_w))
            w1 = max(w1, w0 + 1)
            w_slices.append(jnp.mean(tmp[:, :, w0:w1], axis=2))  # [C, out_h]
        out = jnp.stack(w_slices, axis=2)  # [C, out_h, out_w]
        return out


# --------------------------- primitive layers --------------------------- #
class BasicConv2d(eqx.Module):
    """Conv -> BN -> ReLU; mirrors torchvision BasicConv2d naming (`conv`, `bn`)."""

    conv: eqx.nn.Conv2d
    bn: eqx.nn.BatchNorm
    stride: Tuple[int, int] = eqx.field(static=True)
    padding: Tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        cin: int,
        cout: int,
        kernel_size: Tuple[int, int] | int,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        *,
        key,
    ):
        (k1,) = jr.split(key, 1)
        self.conv = eqx.nn.Conv2d(
            cin,
            cout,
            kernel_size,
            stride=stride,
            padding=padding,
            use_bias=False,
            key=k1,
        )
        self.bn = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

    def __call__(self, x, key, state):
        x = self.conv(x, key=key)
        x, state = self.bn(x, state)
        x = jax.nn.relu(x)
        return x, state


# ----------------------------- Inception A ----------------------------- #
class InceptionA(eqx.Module):
    branch1x1: BasicConv2d
    branch5x5_1: BasicConv2d
    branch5x5_2: BasicConv2d
    branch3x3dbl_1: BasicConv2d
    branch3x3dbl_2: BasicConv2d
    branch3x3dbl_3: BasicConv2d
    branch_pool: BasicConv2d
    avgpool: eqx.nn.AvgPool2d

    def __init__(self, in_channels: int, pool_features: int, *, key):
        k = iter(jr.split(key, 7))
        self.branch1x1 = BasicConv2d(in_channels, 64, 1, key=next(k))
        self.branch5x5_1 = BasicConv2d(in_channels, 48, 1, key=next(k))
        self.branch5x5_2 = BasicConv2d(48, 64, 5, padding=2, key=next(k))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1, key=next(k))
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, padding=1, key=next(k))
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3, padding=1, key=next(k))
        self.branch_pool = BasicConv2d(in_channels, pool_features, 1, key=next(k))
        self.avgpool = eqx.nn.AvgPool2d(3, stride=1, padding=1)

    def __call__(self, x, key, state):
        k = iter(jr.split(key, 7))
        b1, state = self.branch1x1(x, key=next(k), state=state)
        b5, state = self.branch5x5_1(x, key=next(k), state=state)
        b5, state = self.branch5x5_2(b5, key=next(k), state=state)
        b3, state = self.branch3x3dbl_1(x, key=next(k), state=state)
        b3, state = self.branch3x3dbl_2(b3, key=next(k), state=state)
        b3, state = self.branch3x3dbl_3(b3, key=next(k), state=state)
        bp = self.avgpool(x)
        bp, state = self.branch_pool(bp, key=next(k), state=state)
        out = jnp.concatenate([b1, b5, b3, bp], axis=0)
        return out, state


# ----------------------------- Inception B (Reduction A) ----------------------------- #
class InceptionB(eqx.Module):
    branch3x3: BasicConv2d
    branch3x3dbl_1: BasicConv2d
    branch3x3dbl_2: BasicConv2d
    branch3x3dbl_3: BasicConv2d
    maxpool: eqx.nn.MaxPool2d

    def __init__(self, in_channels: int, *, key):
        k = iter(jr.split(key, 4))
        self.branch3x3 = BasicConv2d(in_channels, 384, 3, stride=2, key=next(k))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1, key=next(k))
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, padding=1, key=next(k))
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3, stride=2, key=next(k))
        self.maxpool = eqx.nn.MaxPool2d(3, stride=2)

    def __call__(self, x, key, state):
        k = iter(jr.split(key, 3))
        b3, state = self.branch3x3(x, key=next(k), state=state)
        b3d, state = self.branch3x3dbl_1(x, key=next(k), state=state)
        b3d, state = self.branch3x3dbl_2(b3d, key=next(k), state=state)
        b3d, state = self.branch3x3dbl_3(b3d, key=next(k), state=state)
        bp = self.maxpool(x)
        out = jnp.concatenate(
            [b3, b3d, bp], axis=0
        )  # 384 + 96 + in_ch (pooled) = expected 768 when in_ch=288
        return out, state


# ----------------------------- Inception C (17x17 factorized) ----------------------------- #
class InceptionC(eqx.Module):
    branch1x1: BasicConv2d
    branch7x7_1: BasicConv2d
    branch7x7_2: BasicConv2d
    branch7x7_3: BasicConv2d
    branch7x7dbl_1: BasicConv2d
    branch7x7dbl_2: BasicConv2d
    branch7x7dbl_3: BasicConv2d
    branch7x7dbl_4: BasicConv2d
    branch7x7dbl_5: BasicConv2d
    branch_pool: BasicConv2d
    avgpool: eqx.nn.AvgPool2d

    def __init__(self, in_channels: int, channels_7x7: int, *, key):
        k = iter(jr.split(key, 10))
        c = channels_7x7
        self.branch1x1 = BasicConv2d(in_channels, 192, 1, key=next(k))

        self.branch7x7_1 = BasicConv2d(in_channels, c, 1, key=next(k))
        self.branch7x7_2 = BasicConv2d(c, c, (1, 7), padding=(0, 3), key=next(k))
        self.branch7x7_3 = BasicConv2d(c, 192, (7, 1), padding=(3, 0), key=next(k))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c, 1, key=next(k))
        self.branch7x7dbl_2 = BasicConv2d(c, c, (7, 1), padding=(3, 0), key=next(k))
        self.branch7x7dbl_3 = BasicConv2d(c, c, (1, 7), padding=(0, 3), key=next(k))
        self.branch7x7dbl_4 = BasicConv2d(c, c, (7, 1), padding=(3, 0), key=next(k))
        self.branch7x7dbl_5 = BasicConv2d(c, 192, (1, 7), padding=(0, 3), key=next(k))

        self.branch_pool = BasicConv2d(in_channels, 192, 1, key=next(k))
        self.avgpool = eqx.nn.AvgPool2d(3, stride=1, padding=1)

    def __call__(self, x, key, state):
        k = iter(jr.split(key, 10))
        b1, state = self.branch1x1(x, key=next(k), state=state)

        b7, state = self.branch7x7_1(x, key=next(k), state=state)
        b7, state = self.branch7x7_2(b7, key=next(k), state=state)
        b7, state = self.branch7x7_3(b7, key=next(k), state=state)

        b7d, state = self.branch7x7dbl_1(x, key=next(k), state=state)
        b7d, state = self.branch7x7dbl_2(b7d, key=next(k), state=state)
        b7d, state = self.branch7x7dbl_3(b7d, key=next(k), state=state)
        b7d, state = self.branch7x7dbl_4(b7d, key=next(k), state=state)
        b7d, state = self.branch7x7dbl_5(b7d, key=next(k), state=state)

        bp = self.avgpool(x)
        bp, state = self.branch_pool(bp, key=next(k), state=state)

        out = jnp.concatenate([b1, b7, b7d, bp], axis=0)  # 192+192+192+192 = 768
        return out, state


# ----------------------------- Inception D (Reduction B) ----------------------------- #
class InceptionD(eqx.Module):
    branch3x3_1: BasicConv2d
    branch3x3_2: BasicConv2d
    branch7x7x3_1: BasicConv2d
    branch7x7x3_2: BasicConv2d
    branch7x7x3_3: BasicConv2d
    branch7x7x3_4: BasicConv2d
    maxpool: eqx.nn.MaxPool2d

    def __init__(self, in_channels: int, *, key):
        k = iter(jr.split(key, 6))
        self.branch3x3_1 = BasicConv2d(in_channels, 192, 1, key=next(k))
        self.branch3x3_2 = BasicConv2d(192, 320, 3, stride=2, key=next(k))

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, 1, key=next(k))
        self.branch7x7x3_2 = BasicConv2d(192, 192, (1, 7), padding=(0, 3), key=next(k))
        self.branch7x7x3_3 = BasicConv2d(192, 192, (7, 1), padding=(3, 0), key=next(k))
        self.branch7x7x3_4 = BasicConv2d(192, 192, 3, stride=2, key=next(k))

        self.maxpool = eqx.nn.MaxPool2d(3, stride=2)

    def __call__(self, x, key, state):
        k = iter(jr.split(key, 6))
        b3, state = self.branch3x3_1(x, key=next(k), state=state)
        b3, state = self.branch3x3_2(b3, key=next(k), state=state)

        b7, state = self.branch7x7x3_1(x, key=next(k), state=state)
        b7, state = self.branch7x7x3_2(b7, key=next(k), state=state)
        b7, state = self.branch7x7x3_3(b7, key=next(k), state=state)
        b7, state = self.branch7x7x3_4(b7, key=next(k), state=state)

        bp = self.maxpool(x)
        out = jnp.concatenate(
            [b3, b7, bp], axis=0
        )  # 320 + 192 + 768 = 1280 when in=768
        return out, state


# ----------------------------- Inception E (8x8 split branches) ----------------------------- #
class InceptionE(eqx.Module):
    branch1x1: BasicConv2d

    branch3x3_1: BasicConv2d
    branch3x3_2a: BasicConv2d
    branch3x3_2b: BasicConv2d

    branch3x3dbl_1: BasicConv2d
    branch3x3dbl_2: BasicConv2d
    branch3x3dbl_3a: BasicConv2d
    branch3x3dbl_3b: BasicConv2d

    branch_pool: BasicConv2d
    avgpool: eqx.nn.AvgPool2d

    def __init__(self, in_channels: int, *, key):
        k = iter(jr.split(key, 10))
        self.branch1x1 = BasicConv2d(in_channels, 320, 1, key=next(k))

        self.branch3x3_1 = BasicConv2d(in_channels, 384, 1, key=next(k))
        self.branch3x3_2a = BasicConv2d(384, 384, (1, 3), padding=(0, 1), key=next(k))
        self.branch3x3_2b = BasicConv2d(384, 384, (3, 1), padding=(1, 0), key=next(k))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, 1, key=next(k))
        self.branch3x3dbl_2 = BasicConv2d(448, 384, 3, padding=1, key=next(k))
        self.branch3x3dbl_3a = BasicConv2d(
            384, 384, (1, 3), padding=(0, 1), key=next(k)
        )
        self.branch3x3dbl_3b = BasicConv2d(
            384, 384, (3, 1), padding=(1, 0), key=next(k)
        )

        self.branch_pool = BasicConv2d(in_channels, 192, 1, key=next(k))
        self.avgpool = eqx.nn.AvgPool2d(3, stride=1, padding=1)

    def __call__(self, x, key, state):
        k = iter(jr.split(key, 10))
        b1, state = self.branch1x1(x, key=next(k), state=state)

        b3, state = self.branch3x3_1(x, key=next(k), state=state)
        b3a, state = self.branch3x3_2a(b3, key=next(k), state=state)
        b3b, state = self.branch3x3_2b(b3, key=next(k), state=state)
        b3 = jnp.concatenate([b3a, b3b], axis=0)

        b3d, state = self.branch3x3dbl_1(x, key=next(k), state=state)
        b3d, state = self.branch3x3dbl_2(b3d, key=next(k), state=state)
        b3da, state = self.branch3x3dbl_3a(b3d, key=next(k), state=state)
        b3db, state = self.branch3x3dbl_3b(b3d, key=next(k), state=state)
        b3d = jnp.concatenate([b3da, b3db], axis=0)

        bp = self.avgpool(x)
        bp, state = self.branch_pool(bp, key=next(k), state=state)

        out = jnp.concatenate([b1, b3, b3d, bp], axis=0)  # 320 + 768 + 768 + 192 = 2048
        return out, state


# ----------------------------- Aux logits ----------------------------- #
class InceptionAux(eqx.Module):
    avgpool: AdaptiveAvgPool2d
    conv0: BasicConv2d
    conv1: BasicConv2d
    fc: eqx.nn.Linear
    dropout_p: float = eqx.field(static=True)

    def __init__(
        self, in_channels: int, num_classes: int, dropout: float = 0.7, *, key
    ):
        k0, k1, k2 = jr.split(key, 3)
        self.avgpool = AdaptiveAvgPool2d(5, 5)
        self.conv0 = BasicConv2d(in_channels, 128, 1, key=k0)
        self.conv1 = BasicConv2d(128, 768, 5, key=k1)  # valid -> [768, 1, 1]
        self.fc = eqx.nn.Linear(768, num_classes, key=k2)
        self.dropout_p = dropout

    def __call__(self, x, key, state):
        # Returns aux logits; caller can combine with main loss.
        (k0,) = jr.split(key, 1)
        x = self.avgpool(x)
        x, state = self.conv0(x, key=k0, state=state)
        (k1,) = jr.split(key, 1)
        x, state = self.conv1(x, key=k1, state=state)
        x = x.reshape(-1)  # [768]
        x = eqx.nn.Dropout(self.dropout_p)(x, key=k0)  # stateless dropout instance
        x = self.fc(x)
        return x, state


# ----------------------------- Inception v3 ----------------------------- #
class InceptionV3(eqx.Module):
    # Stem
    Conv2d_1a_3x3: BasicConv2d
    Conv2d_2a_3x3: BasicConv2d
    Conv2d_2b_3x3: BasicConv2d
    maxpool1: eqx.nn.MaxPool2d
    Conv2d_3b_1x1: BasicConv2d
    Conv2d_4a_3x3: BasicConv2d
    maxpool2: eqx.nn.MaxPool2d

    # Mixed blocks
    Mixed_5b: InceptionA
    Mixed_5c: InceptionA
    Mixed_5d: InceptionA
    Mixed_6a: InceptionB
    Mixed_6b: InceptionC
    Mixed_6c: InceptionC
    Mixed_6d: InceptionC
    Mixed_6e: InceptionC
    Mixed_7a: InceptionD
    Mixed_7b: InceptionE
    Mixed_7c: InceptionE

    # Heads
    avgpool: AdaptiveAvgPool2d
    dropout: eqx.nn.Dropout
    fc: eqx.nn.Linear

    # Aux head (optional)
    AuxLogits: InceptionAux | None

    num_classes: int = eqx.field(static=True)
    aux_logits: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int = 1000,
        aux_logits: bool = True,
        dropout: float = 0.5,
        key,
    ):
        # Keys
        k_stem, *k_rest = jr.split(key, 20)
        k_it = iter(k_rest)

        # Stem
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, key=k_stem)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, key=next(k_it))  # valid
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, padding=1, key=next(k_it))  # same
        self.maxpool1 = eqx.nn.MaxPool2d(3, stride=2)

        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1, key=next(k_it))
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3, key=next(k_it))  # valid
        self.maxpool2 = eqx.nn.MaxPool2d(3, stride=2)

        # Mixed 5x (35x35)
        self.Mixed_5b = InceptionA(192, pool_features=32, key=next(k_it))  # -> 256
        self.Mixed_5c = InceptionA(256, pool_features=64, key=next(k_it))  # -> 288
        self.Mixed_5d = InceptionA(288, pool_features=64, key=next(k_it))  # -> 288

        # Mixed 6a (Reduction A) -> 17x17, then 6b-e
        self.Mixed_6a = InceptionB(288, key=next(k_it))  # -> 768
        self.Mixed_6b = InceptionC(768, channels_7x7=128, key=next(k_it))  # -> 768
        self.Mixed_6c = InceptionC(768, channels_7x7=128, key=next(k_it))
        self.Mixed_6d = InceptionC(768, channels_7x7=128, key=next(k_it))
        self.Mixed_6e = InceptionC(768, channels_7x7=128, key=next(k_it))

        # Aux logits head (off by default in __call__)
        self.AuxLogits = (
            InceptionAux(768, num_classes, dropout=0.7, key=next(k_it))
            if aux_logits
            else None
        )

        # Mixed 7a (Reduction B) -> 8x8, then 7b-c
        self.Mixed_7a = InceptionD(768, key=next(k_it))  # -> 1280
        self.Mixed_7b = InceptionE(1280, key=next(k_it))  # -> 2048
        self.Mixed_7c = InceptionE(2048, key=next(k_it))  # -> 2048

        # Head
        self.avgpool = AdaptiveAvgPool2d(1, 1)
        self.dropout = eqx.nn.Dropout(dropout)
        self.fc = eqx.nn.Linear(2048, num_classes, key=next(k_it))

        self.num_classes = num_classes
        self.aux_logits = aux_logits

    def _check_input(self, x: jnp.ndarray):
        if x.ndim != 3:
            raise ValueError(
                f"InceptionV3 expects single sample [C,H,W]; got {tuple(x.shape)}."
            )
        if x.shape[0] != 3:
            raise ValueError(f"Expected 3 input channels; got {x.shape[0]}.")

    def __call__(self, x, key, state):
        """Forward (main logits only). For AuxLogits, call `forward_with_aux`."""
        self._check_input(x)

        # split a bunch of keys; we’ll slice progressively
        keys = list(jr.split(key, 64))
        k_it = iter(keys)

        # Stem
        x, state = self.Conv2d_1a_3x3(x, key=next(k_it), state=state)
        x, state = self.Conv2d_2a_3x3(x, key=next(k_it), state=state)
        x, state = self.Conv2d_2b_3x3(x, key=next(k_it), state=state)
        x = self.maxpool1(x)

        x, state = self.Conv2d_3b_1x1(x, key=next(k_it), state=state)
        x, state = self.Conv2d_4a_3x3(x, key=next(k_it), state=state)
        x = self.maxpool2(x)

        # Mixed 5x
        x, state = self.Mixed_5b(x, key=next(k_it), state=state)
        x, state = self.Mixed_5c(x, key=next(k_it), state=state)
        x, state = self.Mixed_5d(x, key=next(k_it), state=state)

        # Mixed 6a and 6[b-e]
        x, state = self.Mixed_6a(x, key=next(k_it), state=state)
        x, state = self.Mixed_6b(x, key=next(k_it), state=state)
        x, state = self.Mixed_6c(x, key=next(k_it), state=state)
        x, state = self.Mixed_6d(x, key=next(k_it), state=state)
        x, state = self.Mixed_6e(x, key=next(k_it), state=state)

        # (Aux head available here if you want it, via forward_with_aux)

        # Mixed 7[a-c]
        x, state = self.Mixed_7a(x, key=next(k_it), state=state)
        x, state = self.Mixed_7b(x, key=next(k_it), state=state)
        x, state = self.Mixed_7c(x, key=next(k_it), state=state)

        # Head
        x = self.avgpool(x).reshape(-1)  # [2048]
        x = self.dropout(x, key=next(k_it))
        logits = self.fc(x)
        return logits, state

    def forward_with_aux(self, x, key, state):
        """Optional: returns (logits, aux_logits, state). Aux only if `self.AuxLogits` is not None."""
        self._check_input(x)
        keys = list(jr.split(key, 80))
        k_it = iter(keys)

        # Stem
        x, state = self.Conv2d_1a_3x3(x, key=next(k_it), state=state)
        x, state = self.Conv2d_2a_3x3(x, key=next(k_it), state=state)
        x, state = self.Conv2d_2b_3x3(x, key=next(k_it), state=state)
        x = self.maxpool1(x)
        x, state = self.Conv2d_3b_1x1(x, key=next(k_it), state=state)
        x, state = self.Conv2d_4a_3x3(x, key=next(k_it), state=state)
        x = self.maxpool2(x)

        # Mixed 5x
        x, state = self.Mixed_5b(x, key=next(k_it), state=state)
        x, state = self.Mixed_5c(x, key=next(k_it), state=state)
        x, state = self.Mixed_5d(x, key=next(k_it), state=state)

        # Mixed 6[a-e]
        x, state = self.Mixed_6a(x, key=next(k_it), state=state)
        x, state = self.Mixed_6b(x, key=next(k_it), state=state)
        x, state = self.Mixed_6c(x, key=next(k_it), state=state)
        x, state = self.Mixed_6d(x, key=next(k_it), state=state)
        x_mid = x
        x, state = self.Mixed_6e(x, key=next(k_it), state=state)

        aux = None
        if self.AuxLogits is not None:
            (aux_key,) = jr.split(next(k_it), 1)
            aux, _ = self.AuxLogits(
                x_mid, key=aux_key, state=state
            )  # aux uses same state pipe

        x, state = self.Mixed_7a(x, key=next(k_it), state=state)
        x, state = self.Mixed_7b(x, key=next(k_it), state=state)
        x, state = self.Mixed_7c(x, key=next(k_it), state=state)

        x = self.avgpool(x).reshape(-1)
        x = self.dropout(x, key=next(k_it))
        logits = self.fc(x)
        return logits, aux, state


# ----------------------------- Weight loading ----------------------------- #
def _rename_pt_key(k: str) -> str | None:
    """
    Keep torchvision naming; drop BN running stats/counters and Dropout.
    This keeps keys like:
      - Conv2d_1a_3x3.conv.weight
      - Conv2d_1a_3x3.bn.{weight,bias}
      - Mixed_5b.branch3x3dbl_2.conv.weight
      - AuxLogits.fc.{weight,bias}
      - fc.{weight,bias}
    """
    if any(s in k for s in ("running_mean", "running_var", "num_batches_tracked")):
        return None
    # no renames needed since we mirrored names; keep as-is
    return k


def _copy_into_tree(obj, pt: Dict[str, jnp.ndarray], prefix: str = ""):
    """Recursively copy Conv/Linear and BN affine (and tuples) into an Equinox pytree."""
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

            # Stateless layers (pools, dropout) / None / static fields -> skip
        return obj

    if isinstance(obj, tuple):
        return tuple(_copy_into_tree(x, pt, prefix=prefix) for x in obj)
    return obj


def load_torchvision_inception_v3(
    model: InceptionV3, npz_path: str, *, strict_fc: bool = True
) -> InceptionV3:
    """Load a torchvision inception_v3 .npz (from state_dict()) into this model."""
    import numpy as np

    raw = dict(np.load(npz_path))
    pt: Dict[str, jnp.ndarray] = {}
    for k, v in raw.items():
        nk = _rename_pt_key(k)
        if nk is None:
            continue
        pt[nk] = jnp.asarray(v)

    # Handle main FC
    if "fc.weight" in pt and "fc.bias" in pt:
        want_out, want_in = model.fc.weight.shape
        have_out, have_in = pt["fc.weight"].shape
        if (want_out != have_out) or (want_in != have_in):
            if strict_fc:
                raise ValueError(
                    f"FC shape mismatch: want {(want_out, want_in)} vs have {(have_out, have_in)}. "
                    f"Set strict_fc=False to skip loading final FC."
                )
            pt.pop("fc.weight", None)
            pt.pop("fc.bias", None)

    # Handle AuxLogits FC if present in our model
    if model.AuxLogits is None:
        pt.pop("AuxLogits.fc.weight", None)
        pt.pop("AuxLogits.fc.bias", None)
    else:
        if ("AuxLogits.fc.weight" in pt) and ("AuxLogits.fc.bias" in pt):
            want_out, want_in = model.AuxLogits.fc.weight.shape
            have_out, have_in = pt["AuxLogits.fc.weight"].shape
            if (want_out != have_out) or (want_in != have_in):
                if strict_fc:
                    raise ValueError(
                        f"Aux FC mismatch: want {(want_out, want_in)} vs have {(have_out, have_in)}. "
                        f"Set strict_fc=False or instantiate with matching num_classes."
                    )
                pt.pop("AuxLogits.fc.weight", None)
                pt.pop("AuxLogits.fc.bias", None)

    return _copy_into_tree(model, pt, prefix="")


# convenience alias
def load_imagenet_inception_v3(
    model: InceptionV3, npz="inception_v3_imagenet.npz", strict_fc: bool = True
) -> InceptionV3:
    return load_torchvision_inception_v3(model, npz, strict_fc=strict_fc)


# ----------------------------- Smoke test ----------------------------- #
if __name__ == "__main__":
    """
    Synthetic classification smoke test for Inception v3.
    Uses 299×299 RGB (canonical). Replace with real data for experiments.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import augmax
    from augmax import InputType
    import optax
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jr

    # Your training utilities
    from quantbayes.stochax import (
        train,
        predict,
        make_augmax_augment,
        multiclass_loss,
    )

    rng = np.random.RandomState(0)
    N, C, H, W, NUM_CLASSES = 256, 3, 299, 299, 10
    X_np = rng.rand(N, C, H, W).astype("float32")
    y_np = rng.randint(0, NUM_CLASSES, size=(N,)).astype("int32")

    split = int(0.8 * N)
    X_train, X_val = X_np[:split], X_np[split:]
    y_train, y_val = y_np[:split], y_np[split:]

    transform = augmax.Chain(
        augmax.HorizontalFlip(),
        augmax.Rotate(angle_range=10),
        input_types=[InputType.IMAGE, InputType.METADATA],
    )
    augment_fn = make_augmax_augment(transform)

    master_key = jr.PRNGKey(42)
    model_key, train_key = jr.split(master_key)

    model, state = eqx.nn.make_with_state(InceptionV3)(
        num_classes=NUM_CLASSES,
        aux_logits=True,  # instantiate aux head so weights can load if desired
        dropout=0.5,
        key=model_key,
    )

    # Optional pretrained load (skips fc when shapes mismatch)
    # model = load_imagenet_inception_v3(model, "inception_v3_imagenet.npz", strict_fc=False)

    lr_sched = optax.cosine_decay_schedule(1e-3, decay_steps=300)
    optimizer = optax.adamw(
        learning_rate=lr_sched,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=1e-4,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

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
        batch_size=16,  # big model; keep small for smoke
        num_epochs=6,
        patience=2,
        key=train_key,
        augment_fn=augment_fn,
        lambda_spec=0.0,
    )

    logits = predict(best_model, best_state, jnp.array(X_val), train_key)
    print("Predictions shape:", logits.shape)

    plt.figure()
    plt.plot(tr_loss, label="train")
    plt.plot(va_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Inception v3 smoke test")
    plt.show()
