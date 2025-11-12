"""
Equinox UNet with selectable ResNet encoder backbone.

Backbones currently supported: "resnet18", "resnet34", "resnet50".
BatchNorm uses mode="batch" so no EMA state is carried around.
All __call__ signatures are (self, x, key, state).

Author: Joseph Margaryan
"""

from __future__ import annotations
from typing import Any, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.image  # for upsampling


def _match(x: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    """Pad or crop so spatial dims of `x` = spatial dims of `ref`."""
    h, w = x.shape[-2:]
    H, W = ref.shape[-2:]
    dh, dw = H - h, W - w
    if dh > 0 or dw > 0:  # pad
        pads = [(0, 0)] * (x.ndim - 2) + [
            (dh // 2, dh - dh // 2),
            (dw // 2, dw - dw // 2),
        ]
        x = jnp.pad(x, pads)
    if dh < 0 or dw < 0:  # crop
        sh, sw = (-dh) // 2, (-dw) // 2
        x = x[(..., slice(sh, sh + H), slice(sw, sw + W))]
    return x


def _center_crop_or_pad_nchw(x: jnp.ndarray, H2: int, W2: int) -> jnp.ndarray:
    C, h, w = x.shape
    dh, dw = H2 - h, W2 - w
    if dh > 0 or dw > 0:
        pad_h1 = max(0, dh) // 2
        pad_h2 = max(0, dh) - pad_h1
        pad_w1 = max(0, dw) // 2
        pad_w2 = max(0, dw) - pad_w1
        x = jnp.pad(x, ((0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2)))
        _, h, w = x.shape
        dh, dw = H2 - h, W2 - w
    if dh < 0 or dw < 0:
        sh = (-dh) // 2
        sw = (-dw) // 2
        x = x[:, sh : sh + H2, sw : sw + W2]
    return x


def _downsample_skip_to(identity: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    C, H, W = identity.shape
    C2, H2, W2 = ref.shape
    if (H, W) == (H2, W2):
        return identity
    if C != C2:
        raise ValueError(
            f"Skip/branch channel mismatch (C={C} vs C2={C2}) without a downsample conv."
        )
    sh = H // H2 if H2 > 0 else 1
    sw = W // W2 if W2 > 0 else 1
    if (H2 * sh == H) and (W2 * sw == W) and (sh == sw) and (sh >= 1):
        s = sh
        if s > 1:
            Hp = (H // s) * s
            Wp = (W // s) * s
            x = identity[:, :Hp, :Wp]
            x = x.reshape(C, Hp // s, s, Wp // s, s).mean(axis=(2, 4))
            if x.shape[-2:] != (H2, W2):
                x = _center_crop_or_pad_nchw(x, H2, W2)
            return x
        return identity
    return _center_crop_or_pad_nchw(identity, H2, W2)


class BasicBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    down_conv: eqx.nn.Conv2d | None
    down_bn: eqx.nn.BatchNorm | None
    stride: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True, default=1.0)

    def __init__(self, cin: int, cout: int, stride: int, *, key, alpha: float = 1.0):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(cin, cout, 3, stride=stride, padding=1, key=k1)
        self.bn1 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.conv2 = eqx.nn.Conv2d(cout, cout, 3, padding=1, key=k2)
        self.bn2 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        if stride != 1 or cin != cout:
            self.down_conv = eqx.nn.Conv2d(cin, cout, 1, stride=stride, key=k3)
            self.down_bn = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        else:
            self.down_conv = None
            self.down_bn = None
        self.stride = stride
        object.__setattr__(self, "alpha", float(alpha))

    def __call__(self, x, key, state):
        k1, k2, k3 = jr.split(key, 3)
        identity = x
        out = self.conv1(x, key=k1)
        out, state = self.bn1(out, state)
        out = jax.nn.relu(out)
        out = self.conv2(out, key=k2)
        out, state = self.bn2(out, state)

        if self.down_conv is not None:
            identity = self.down_conv(identity, key=k3)
            identity, state = self.down_bn(identity, state)
        if identity.shape != out.shape:
            identity = _downsample_skip_to(identity, out)

        out = jax.nn.relu(identity + self.alpha * out)
        return out, state


class Bottleneck(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    conv3: eqx.nn.Conv2d
    bn3: eqx.nn.BatchNorm
    down_conv: eqx.nn.Conv2d | None
    down_bn: eqx.nn.BatchNorm | None
    stride: int = eqx.field(static=True)
    expansion: int = eqx.field(static=True, default=4)
    alpha: float = eqx.field(static=True, default=1.0)

    def __init__(self, cin: int, cout: int, stride: int, *, key, alpha: float = 1.0):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(cin, cout, 1, key=k1)
        self.bn1 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.conv2 = eqx.nn.Conv2d(cout, cout, 3, stride=stride, padding=1, key=k2)
        self.bn2 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.conv3 = eqx.nn.Conv2d(cout, cout * 4, 1, key=k3)
        self.bn3 = eqx.nn.BatchNorm(cout * 4, axis_name="batch", mode="batch")
        if stride != 1 or cin != cout * 4:
            self.down_conv = eqx.nn.Conv2d(cin, cout * 4, 1, stride=stride, key=k4)
            self.down_bn = eqx.nn.BatchNorm(cout * 4, axis_name="batch", mode="batch")
        else:
            self.down_conv = None
            self.down_bn = None
        self.stride = stride
        object.__setattr__(self, "alpha", float(alpha))

    def __call__(self, x, key, state):
        k1, k2, k3, k4 = jr.split(key, 4)
        identity = x
        out = self.conv1(x, key=k1)
        out, state = self.bn1(out, state)
        out = jax.nn.relu(out)
        out = self.conv2(out, key=k2)
        out, state = self.bn2(out, state)
        out = jax.nn.relu(out)
        out = self.conv3(out, key=k3)
        out, state = self.bn3(out, state)

        if self.down_conv is not None:
            identity = self.down_conv(identity, key=k4)
            identity, state = self.down_bn(identity, state)
        if identity.shape != out.shape:
            identity = _downsample_skip_to(identity, out)

        out = jax.nn.relu(identity + self.alpha * out)
        return out, state


_RESNET_SPECS = {
    "resnet18": dict(
        block=BasicBlock, layers=[2, 2, 2, 2], channels=[64, 64, 128, 256, 512]
    ),
    "resnet34": dict(
        block=BasicBlock, layers=[3, 4, 6, 3], channels=[64, 64, 128, 256, 512]
    ),
    "resnet50": dict(
        block=Bottleneck, layers=[3, 4, 6, 3], channels=[64, 256, 512, 1024, 2048]
    ),
}


class ResNetEncoder(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    pool: eqx.nn.MaxPool2d
    layers1: Tuple[eqx.Module, ...]
    layers2: Tuple[eqx.Module, ...]
    layers3: Tuple[eqx.Module, ...]
    layers4: Tuple[eqx.Module, ...]
    spec_name: str = eqx.field(static=True)
    residual_alpha: Tuple[float, float, float, float] = eqx.field(static=True)

    def __init__(
        self,
        backbone: str,
        *,
        key,
        residual_alpha: float | Tuple[float, float, float, float] = 1.0,
    ):
        if backbone not in _RESNET_SPECS:
            raise ValueError(f"Unknown backbone '{backbone}'.")
        spec = _RESNET_SPECS[backbone]
        Block = spec["block"]
        layer_sizes = spec["layers"]

        # normalize α to per-stage tuple
        if isinstance(residual_alpha, (tuple, list)):
            if len(residual_alpha) != 4:
                raise ValueError("residual_alpha tuple must be length 4 (stages 1..4).")
            alphas = tuple(float(a) for a in residual_alpha)
        else:
            alphas = (float(residual_alpha),) * 4
        object.__setattr__(self, "residual_alpha", alphas)

        num_blocks = sum(layer_sizes)
        ks = list(jr.split(key, 2 + num_blocks))  # conv1/pool + blocks

        self.conv1 = eqx.nn.Conv2d(3, 64, 7, stride=2, padding=3, key=ks[0])
        self.bn1 = eqx.nn.BatchNorm(64, axis_name="batch", mode="batch")
        self.pool = eqx.nn.MaxPool2d(3, 2, padding=1)

        def _make_layer(cin, cout, blocks, stride, key_iter, alpha):
            mods = []
            c_in = cin
            for i in range(blocks):
                s = stride if i == 0 else 1
                mods.append(Block(c_in, cout, s, key=next(key_iter), alpha=alpha))
                c_in = cout * (4 if Block is Bottleneck else 1)
            return tuple(mods), c_in

        k_iter = iter(ks[1:])
        self.layers1, ch1 = _make_layer(64, 64, layer_sizes[0], 1, k_iter, alphas[0])
        self.layers2, ch2 = _make_layer(ch1, 128, layer_sizes[1], 2, k_iter, alphas[1])
        self.layers3, ch3 = _make_layer(ch2, 256, layer_sizes[2], 2, k_iter, alphas[2])
        self.layers4, _ = _make_layer(ch3, 512, layer_sizes[3], 2, k_iter, alphas[3])
        self.spec_name = backbone

    def __call__(self, x, key, state):
        k0, key = jr.split(key)
        x = self.conv1(x, key=k0)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        conv1_out = x
        x = self.pool(x)

        skips = [conv1_out]
        for layer in [self.layers1, self.layers2, self.layers3, self.layers4]:
            for block in layer:
                k, key = jr.split(key)
                x, state = block(x, key=k, state=state)
            skips.append(x)
        return (skips[0], skips[1], skips[2], skips[3], skips[4]), state


class ConvBlock(eqx.Module):
    c1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    c2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm

    def __init__(self, cin: int, cout: int, *, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.c1 = eqx.nn.Conv2d(cin, cout, 3, padding=1, key=k1)
        self.bn1 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.c2 = eqx.nn.Conv2d(cout, cout, 3, padding=1, key=k3)
        self.bn2 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")

    def __call__(self, x, key, state):
        k1, k2 = jr.split(key, 2)
        x, state = self.bn1(self.c1(x, key=k1), state)
        x = jax.nn.relu(x)
        x, state = self.bn2(self.c2(x, key=k2), state)
        x = jax.nn.relu(x)
        return x, state


class Up(eqx.Module):
    up: eqx.nn.ConvTranspose2d
    conv: ConvBlock

    def __init__(self, cin: int, skip: int, cout: int, *, key):
        k1, k2 = jr.split(key, 2)
        self.up = eqx.nn.ConvTranspose2d(cin, cout, 2, stride=2, key=k1)
        self.conv = ConvBlock(cout + skip, cout, key=k2)

    def __call__(self, x, skip, key, state):
        k1, k2 = jr.split(key, 2)
        x = self.up(x, key=k1)
        x, skip = _match(x, skip), _match(skip, x)
        x = jnp.concatenate([skip, x], axis=0)
        x, state = self.conv(x, key=k2, state=state)
        return x, state


class UNetBackbone(eqx.Module):
    encoder: ResNetEncoder
    d1: Up
    d2: Up
    d3: Up
    d4: Up
    out_conv: eqx.nn.Conv2d
    backbone_name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        out_ch: int = 1,
        backbone: str = "resnet34",
        residual_alpha: float | Tuple[float, float, float, float] = 1.0,
        key,
    ):
        k_enc, *k_rest = jr.split(key, 10)
        self.encoder = ResNetEncoder(backbone, key=k_enc, residual_alpha=residual_alpha)

        chans = _RESNET_SPECS[backbone]["channels"]  # [conv1,l1,l2,l3,l4]
        c1, c2, c3, c4, c5 = chans
        self.d1 = Up(c5, c4, c4, key=k_rest[0])
        self.d2 = Up(c4, c3, c3, key=k_rest[1])
        self.d3 = Up(c3, c2, c2, key=k_rest[2])
        self.d4 = Up(c2, c1, c1, key=k_rest[3])
        self.out_conv = eqx.nn.Conv2d(c1, out_ch, 1, key=k_rest[4])
        self.backbone_name = backbone

    def __call__(self, x, key, state):
        k_enc, k1, k2, k3, k4, k_out = jr.split(key, 6)
        (conv1, l1, l2, l3, l4), state = self.encoder(x, key=k_enc, state=state)

        d1, state = self.d1(l4, l3, key=k1, state=state)
        d2, state = self.d2(d1, l2, key=k2, state=state)
        d3, state = self.d3(d2, l1, key=k3, state=state)
        d4, state = self.d4(d3, conv1, key=k4, state=state)

        logits = self.out_conv(d4, key=k_out)
        logits = jax.image.resize(
            logits,
            (logits.shape[0], x.shape[1], x.shape[2]),
            method="bilinear",
        )
        return logits, state


if __name__ == "__main__":
    """
    Synthetic segmentation-pipeline smoke-test.
    Runs CPU-only in <10 s per backbone.
    Replace the fake data with a real dataset in actual experiments.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import augmax
    from augmax import InputType
    import optax
    import equinox as eqx
    from quantbayes.stochax import (
        train,
        make_augmax_augment,
        make_dice_bce_loss,
    )

    rng = np.random.RandomState(0)
    N, C, H, W, OUT_CH = 10, 3, 128, 128, 1
    X_np = rng.rand(N, C, H, W).astype("float32")
    y_np = rng.randint(0, 2, size=(N, OUT_CH, H, W)).astype("float32")

    split = int(0.8 * N)
    X_train, X_val = X_np[:split], X_np[split:]
    y_train, y_val = y_np[:split], y_np[split:]

    transform = augmax.Chain(
        augmax.HorizontalFlip(),
        augmax.Rotate(angle_range=15),
        input_types=[InputType.IMAGE, InputType.MASK],
    )
    augment_fn = make_augmax_augment(transform)

    backbones = ["resnet18", "resnet34", "resnet50"]
    master_key = jr.PRNGKey(42)

    for i, backbone in enumerate(backbones):
        print(f"\n=== Smoke-testing UNet with {backbone} encoder ===")
        model_key, train_key = jr.split(jr.fold_in(master_key, i))
        model, state = eqx.nn.make_with_state(UNetBackbone)(
            out_ch=OUT_CH,
            backbone=backbone,
            key=model_key,
        )

        lr_sched = optax.cosine_decay_schedule(1e-3, decay_steps=200)
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
            loss_fn=make_dice_bce_loss(),
            X_train=jnp.array(X_train),
            y_train=jnp.array(y_train),
            X_val=jnp.array(X_val),
            y_val=jnp.array(y_val),
            batch_size=32,
            num_epochs=5,
            patience=2,
            key=train_key,
            augment_fn=augment_fn,
            lambda_spec=0.0,
        )

        plt.figure()
        plt.plot(tr_loss, label="train")
        plt.plot(va_loss, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title(f"Backbone {backbone}")
        plt.show()
