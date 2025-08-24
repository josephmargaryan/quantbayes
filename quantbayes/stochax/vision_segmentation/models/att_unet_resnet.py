"""
Attention-UNet with selectable ResNet-{18,34,50} backbone.
* Shape convention: single sample, (C, H, W).  No batch dim.
* All __call__ signatures are (self, x, key, state) → (logits, state)
* BN uses mode="batch" (no EMA).
Author: <you>
"""

from __future__ import annotations
from typing import Any, Tuple
import equinox as eqx
import jax, jax.numpy as jnp, jax.random as jr
from quantbayes.stochax.vision_segmentation.models.unet_backbone import (
    _match,
    ConvBlock,
    ResNetEncoder,
    _RESNET_SPECS,
)


def _match(x: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    """Pad/crop so spatial dims of `x` match `ref`."""
    h, w = x.shape[-2:]
    H, W = ref.shape[-2:]
    dh, dw = H - h, W - w
    if dh > 0 or dw > 0:
        pads = [(0, 0)] * (x.ndim - 2) + [
            (dh // 2, dh - dh // 2),
            (dw // 2, dw - dw // 2),
        ]
        x = jnp.pad(x, pads)
    if dh < 0 or dw < 0:
        sh, sw = (-dh) // 2, (-dw) // 2
        x = x[(..., slice(sh, sh + H), slice(sw, sw + W))]
    return x


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

    def __call__(self, x, *, key, state):
        k1, k2 = jr.split(key, 2)
        x, state = self.bn1(self.c1(x, key=k1), state)
        x = jax.nn.relu(x)
        x, state = self.bn2(self.c2(x, key=k2), state)
        x = jax.nn.relu(x)
        return x, state


class AttentionBlock(eqx.Module):
    W_g: eqx.nn.Conv2d
    W_x: eqx.nn.Conv2d
    psi: eqx.nn.Conv2d

    def __init__(self, F_g: int, F_l: int, F_int: int, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.W_g = eqx.nn.Conv2d(F_g, F_int, 1, key=k1)
        self.W_x = eqx.nn.Conv2d(F_l, F_int, 1, key=k2)
        self.psi = eqx.nn.Conv2d(F_int, 1, 1, key=k3)

    def __call__(self, g, x):
        psi = jax.nn.relu(self.W_g(g) + self.W_x(x))
        psi = jax.nn.sigmoid(self.psi(psi))
        return x * psi


class UpAtt(eqx.Module):
    up: eqx.nn.ConvTranspose2d
    att: AttentionBlock
    conv: ConvBlock

    def __init__(self, cin: int, skip: int, cout: int, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.up = eqx.nn.ConvTranspose2d(cin, cout, 2, stride=2, key=k1)
        self.att = AttentionBlock(cout, skip, cout // 2, key=k2)
        self.conv = ConvBlock(cout + skip, cout, key=k3)

    def __call__(self, x, skip, *, key, state):
        k1, k2 = jr.split(key, 2)
        x = self.up(x, key=k1)
        x, skip = _match(x, skip), _match(skip, x)
        skip = self.att(x, skip)
        x_cat = jnp.concatenate([skip, x], axis=0)
        x_out, state = self.conv(x_cat, key=k2, state=state)
        return x_out, state


class AttentionUNetResNet(eqx.Module):
    encoder: ResNetEncoder
    d1: UpAtt
    d2: UpAtt
    d3: UpAtt
    d4: UpAtt
    out_conv: eqx.nn.Conv2d
    backbone_name: str = eqx.field(static=True)

    def __init__(self, *, out_ch: int = 1, backbone: str = "resnet34", key):
        k_enc, *ks = jr.split(key, 10)
        self.encoder = ResNetEncoder(backbone, key=k_enc)
        c1, c2, c3, c4, c5 = _RESNET_SPECS[backbone]["channels"]

        self.d1 = UpAtt(c5, c4, c4, key=ks[0])
        self.d2 = UpAtt(c4, c3, c3, key=ks[1])
        self.d3 = UpAtt(c3, c2, c2, key=ks[2])
        self.d4 = UpAtt(c2, c1, c1, key=ks[3])
        self.out_conv = eqx.nn.Conv2d(c1, out_ch, 1, key=ks[4])
        self.backbone_name = backbone

    def __call__(self, x, key, state):
        k_enc, k1, k2, k3, k4, k_out = jr.split(key, 6)
        (conv1, l1, l2, l3, l4), state = self.encoder(x, key=k_enc, state=state)

        d1, state = self.d1(l4, l3, key=k1, state=state)
        d2, state = self.d2(d1, l2, key=k2, state=state)
        d3, state = self.d3(d2, l1, key=k3, state=state)
        d4, state = self.d4(d3, conv1, key=k4, state=state)

        logits = self.out_conv(d4)
        logits = jax.image.resize(
            logits,
            (logits.shape[0], x.shape[1], x.shape[2]),
            method="linear",
        )
        return logits, state


if __name__ == "__main__":
    """
    Synthetic segmentation-pipeline smoke-test.
    Runs CPU-only in <10 s. Swap the fake data for CIFAR-10, Cityscapes,
    etc. in real experiments.
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
    N, C, H, W, OUT_CH = 10, 3, 128, 128, 1

    # images: N×C×H×W
    X_np = rng.rand(N, C, H, W).astype("float32")

    # masks:  N×OUT_CH×H×W
    y_np = rng.randint(0, 2, size=(N, OUT_CH, H, W)).astype("float32")

    # train/val split
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
    model, state = eqx.nn.make_with_state(AttentionUNetResNet)(
        out_ch=OUT_CH,
        backbone="resnet50",  # or "resnet18" / "resnet34" / "resnet50"
        key=model_key,
    )

    lr_sched = optax.cosine_decay_schedule(1e-3, decay_steps=500)
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
    plt.title("Synthetic TransUNet smoke-test")
    plt.show()
