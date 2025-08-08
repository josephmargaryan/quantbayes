"""
Equinox PSPNet with selectable ResNet encoder backbone (NCHW, single-sample API).

Backbones: "resnet18", "resnet34", "resnet50"
Author: Joseph Margaryan

Key points:
- Tracer-safe adaptive avg pooling (no Python int casts; uses reduce_window + counts).
- Correct NCHW<->NHWC handling for jax.image.resize.
- PPM branches: 1x1 -> BatchNorm(mode="batch") -> ReLU, then upsample.
- BatchNorm configured for Equinox version without `axis=` (uses axis_name="batch", mode="batch").
"""

from __future__ import annotations
from typing import Tuple, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.image

from quantbayes.stochax.vision_segmentation.models.unet_backbone import (
    ResNetEncoder,
    _RESNET_SPECS,
)

# ---------- Utilities ----------


def _resize_nchw(
    x: jnp.ndarray, out_h: int, out_w: int, *, method: str = "bilinear"
) -> jnp.ndarray:
    """Resize a [C,H,W] tensor via channels-last, then move axes back."""
    c, _, _ = x.shape
    x_hwc = jnp.moveaxis(x, 0, -1)  # [H,W,C]
    y_hwc = jax.image.resize(
        x_hwc, (out_h, out_w, c), method=method
    )  # channels-last API
    return jnp.moveaxis(y_hwc, -1, 0)  # [C,H,W]


def _adaptive_avg_pool(x: jnp.ndarray, out_size: int) -> jnp.ndarray:
    """
    Tracer-safe adaptive average pooling for [C,H,W] -> [C,out_size,out_size].
    Uses windowed sums with zero-padding + windowed counts, then divides.
    No Python `int(...)` on traced values; fully JAX-compatible under jit/vmap.
    """
    C, H, W = x.shape

    # ceil divisions as JAX ops (static for fixed shapes)
    k_h = (H + out_size - 1) // out_size
    k_w = (W + out_size - 1) // out_size
    s_h = k_h
    s_w = k_w

    pad_h = k_h * out_size - H
    pad_w = k_w * out_size - W
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    pads = (
        (0, 0),  # C
        (pad_top, pad_bottom),  # H
        (pad_left, pad_right),  # W
    )
    x_pad = jnp.pad(x, pads, mode="constant", constant_values=0.0)  # [C,H',W']

    # Sum over pooling windows
    win = (1, k_h, k_w)
    strides = (1, s_h, s_w)
    pooled_sum = jax.lax.reduce_window(
        x_pad,
        0.0,
        jax.lax.add,
        window_dimensions=win,
        window_strides=strides,
        padding="VALID",
    )  # [C,out_size,out_size]

    # Count valid elements per window (avoid zero-padding bias)
    ones = jnp.ones((1, x_pad.shape[1], x_pad.shape[2]), dtype=x.dtype)  # [1,H',W']
    counts = jax.lax.reduce_window(
        ones,
        0.0,
        jax.lax.add,
        window_dimensions=win,
        window_strides=strides,
        padding="VALID",
    )  # [1,out_size,out_size]
    pooled = pooled_sum / counts  # broadcasts over channel dim

    return pooled  # [C,out_size,out_size]


# ---------- Modules ----------


class PPM(eqx.Module):
    convs: Tuple[eqx.nn.Conv2d, ...]
    bns: Tuple[eqx.nn.BatchNorm, ...]
    out_ch: int = eqx.field(static=True)
    bins: Tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        cin: int,
        *,
        bins: Sequence[int] = (1, 2, 3, 6),
        reduction: int = 4,
        key,
        axis_name: str | None = "batch",
    ):
        ks = jr.split(key, len(bins))
        cout = max(1, cin // reduction)
        self.convs = tuple(eqx.nn.Conv2d(cin, cout, 1, key=k) for k in ks)
        self.bns = tuple(
            eqx.nn.BatchNorm(cout, axis_name=axis_name, mode="batch") for _ in bins
        )
        self.bins = tuple(bins)
        self.out_ch = cin + len(bins) * cout

    def __call__(self, x: jnp.ndarray, *, key, state):
        outs = [x]
        for bin_size, conv, bn in zip(self.bins, self.convs, self.bns):
            y = conv(_adaptive_avg_pool(x, bin_size))  # [C',b,b]
            y, state = bn(y, state)
            y = jax.nn.relu(y)
            y = _resize_nchw(y, x.shape[1], x.shape[2], method="bilinear")
            outs.append(y)
        return jnp.concatenate(outs, axis=0), state


class PSPHead(eqx.Module):
    conv3: eqx.nn.Conv2d
    bn3: eqx.nn.BatchNorm
    drop: eqx.nn.Dropout
    out_conv: eqx.nn.Conv2d
    dropout_p: float = eqx.field(static=True)

    def __init__(
        self,
        cin: int,
        cout: int,
        *,
        key,
        dropout_p: float = 0.1,
        axis_name: str | None = "batch",
    ):
        k1, k2 = jr.split(key, 2)
        mid = max(1, cin // 4)
        self.conv3 = eqx.nn.Conv2d(cin, mid, 3, padding=1, key=k1)
        self.bn3 = eqx.nn.BatchNorm(mid, axis_name=axis_name, mode="batch")
        self.drop = eqx.nn.Dropout(p=dropout_p)
        self.out_conv = eqx.nn.Conv2d(mid, cout, 1, key=k2)
        self.dropout_p = dropout_p

    def __call__(self, x: jnp.ndarray, *, key, state):
        x, state = self.bn3(self.conv3(x), state)
        x = jax.nn.relu(x)
        x = self.drop(x, key=key)
        logits = self.out_conv(x)
        return logits, state


class AuxHead(eqx.Module):
    conv: eqx.nn.Conv2d
    bn: eqx.nn.BatchNorm
    out_conv: eqx.nn.Conv2d

    def __init__(self, cin: int, cout: int, *, key, axis_name: str | None = "batch"):
        k1, k2 = jr.split(key, 2)
        mid = max(1, cin // 4)
        self.conv = eqx.nn.Conv2d(cin, mid, 3, padding=1, key=k1)
        self.bn = eqx.nn.BatchNorm(mid, axis_name=axis_name, mode="batch")
        self.out_conv = eqx.nn.Conv2d(mid, cout, 1, key=k2)

    def __call__(self, x, *, key, state):
        x, state = self.bn(self.conv(x), state)
        x = jax.nn.relu(x)
        logits = self.out_conv(x)
        return logits, state


class PSPNetResNet(eqx.Module):
    encoder: ResNetEncoder
    ppm: PPM
    head: PSPHead
    aux_head: AuxHead | None

    num_classes: int = eqx.field(static=True)
    aux: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int,
        backbone: str = "resnet50",
        aux: bool = True,
        ppm_bins: Sequence[int] = (1, 2, 3, 6),
        key,
        axis_name: str | None = "batch",
    ):
        if backbone not in _RESNET_SPECS:
            raise ValueError(f"Unknown backbone '{backbone}'")
        k_enc, k_ppm, k_head, k_aux = jr.split(key, 4)

        # Prefer a dilated encoder (OS=8/16) if your ResNet exposes it.
        self.encoder = ResNetEncoder(backbone, key=k_enc)

        c_deep = _RESNET_SPECS[backbone]["channels"][-1]
        c_aux = _RESNET_SPECS[backbone]["channels"][-2]

        self.ppm = PPM(c_deep, bins=ppm_bins, key=k_ppm, axis_name=axis_name)
        self.head = PSPHead(
            self.ppm.out_ch, num_classes, key=k_head, axis_name=axis_name
        )

        self.aux = aux
        self.aux_head = (
            AuxHead(c_aux, num_classes, key=k_aux, axis_name=axis_name) if aux else None
        )
        self.num_classes = num_classes

    def __call__(self, x, key, state):
        """
        x: [C,H,W] single sample (batch externally via vmap with axis_name='batch').
        Returns:
            logits, state                       if aux == False
            (main_logits, aux_logits), state    if aux == True
        """
        k_enc, k_ppm, k_head, k_aux = jr.split(key, 4)

        feats, state = self.encoder(x, key=k_enc, state=state)
        penult, deepest = feats[-2], feats[-1]

        y, state = self.ppm(deepest, key=k_ppm, state=state)
        y, state = self.head(y, key=k_head, state=state)
        y = _resize_nchw(y, x.shape[1], x.shape[2], method="bilinear")

        if not self.aux:
            return y, state

        aux_y, state = self.aux_head(penult, key=k_aux, state=state)
        aux_y = _resize_nchw(aux_y, x.shape[1], x.shape[2], method="bilinear")
        return (y, aux_y), state


# ---------- Synthetic smoke test (unchanged API) ----------

if __name__ == "__main__":
    """
    Synthetic segmentation-pipeline smoke-test.
    Runs CPU-only in <10 s. Swap the fake data for CIFAR-10, Cityscapes, etc.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import augmax
    from augmax import InputType

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import equinox as eqx

    from quantbayes.stochax import (
        train,
        predict,
        make_augmax_augment,
        make_dice_bce_loss,
    )

    rng = np.random.RandomState(0)
    N, C, H, W, OUT_CH = 10, 3, 128, 128, 1  # binary mask example

    X_np = rng.rand(N, C, H, W).astype("float32")  # images: N×C×H×W
    y_np = rng.randint(0, 2, size=(N, OUT_CH, H, W)).astype(
        "float32"
    )  # masks: N×OUT_CH×H×W

    split = int(0.8 * N)
    X_train, X_val = X_np[:split], X_np[split:]
    y_train, y_val = y_np[:split], y_np[split:]

    transform = augmax.Chain(
        augmax.HorizontalFlip(),
        augmax.Rotate(angle_range=15),
        input_types=[InputType.IMAGE, InputType.MASK],
    )
    augment_fn = make_augmax_augment(transform)

    master_key = jr.PRNGKey(42)
    model_key, train_key = jr.split(master_key)
    model, state = eqx.nn.make_with_state(PSPNetResNet)(
        num_classes=OUT_CH,
        backbone="resnet50",
        aux=True,
        key=model_key,
    )

    lr_sched = optax.cosine_decay_schedule(1e-3, decay_steps=500)
    optimizer = optax.adamw(
        learning_rate=lr_sched, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-4
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    best_model, best_state, tr_loss, va_loss = train(
        model=model,
        state=state,
        opt_state=opt_state,
        optimizer=optimizer,
        loss_fn=make_dice_bce_loss(),  # BCE-with-logits over 1-channel masks
        X_train=jnp.array(X_train),  # (N,C,H,W)
        y_train=jnp.array(y_train),  # (N,1,H,W)
        X_val=jnp.array(X_val),
        y_val=jnp.array(y_val),
        batch_size=32,
        num_epochs=15,
        patience=4,
        key=train_key,
        augment_fn=augment_fn,
        lambda_spec=0.0,
    )

    plt.plot(tr_loss, label="train")
    plt.plot(va_loss, label="val")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Synthetic PSPNet smoke-test")
    plt.show()
