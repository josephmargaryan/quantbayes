"""
Equinox LinkNet with selectable ResNet encoder backbone
Backbones: "resnet18", "resnet34", "resnet50"

Author: Joseph Margaryan
"""

from __future__ import annotations
from typing import Tuple, List

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import jax.image


from quantbayes.stochax.vision_segmentation.models.unet_backbone import (
    _match,
    ResNetEncoder,
    _RESNET_SPECS,
)


class DecoderBlock(eqx.Module):
    up: eqx.nn.ConvTranspose2d
    bn1: eqx.nn.BatchNorm
    conv1x1: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm

    def __init__(self, cin: int, cout: int, *, key):
        k_up, k_conv = jr.split(key, 2)
        self.up = eqx.nn.ConvTranspose2d(
            cin, cout, 3, stride=2, padding=1, output_padding=1, key=k_up
        )
        self.bn1 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.conv1x1 = eqx.nn.Conv2d(cout, cout, 1, key=k_conv)
        self.bn2 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")

    def __call__(self, x, key, state):
        k1, k2 = jr.split(key, 2)
        x = self.up(x, key=k1)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)

        x = self.conv1x1(x, key=k2)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)
        return x, state


class LinkNetResNet(eqx.Module):
    encoder: ResNetEncoder
    d4: DecoderBlock
    d3: DecoderBlock
    d2: DecoderBlock
    d1: DecoderBlock
    out_conv: eqx.nn.Conv2d

    backbone: str = eqx.field(static=True)
    out_ch: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        out_ch: int = 1,
        backbone: str = "resnet34",
        key,
    ):
        if backbone not in _RESNET_SPECS:
            raise ValueError(f"Unknown backbone '{backbone}'")

        k_enc, *k_dec, k_out = jr.split(key, 7)
        self.encoder = ResNetEncoder(backbone, key=k_enc)

        chans: List[int] = _RESNET_SPECS[backbone]["channels"]
        c1, c2, c3, c4, c5 = chans

        self.d4 = DecoderBlock(c5, c4, key=k_dec[0])
        self.d3 = DecoderBlock(c4, c3, key=k_dec[1])
        self.d2 = DecoderBlock(c3, c2, key=k_dec[2])
        self.d1 = DecoderBlock(c2, c1, key=k_dec[3])

        self.out_conv = eqx.nn.Conv2d(c1, out_ch, 1, key=k_out)

        self.backbone = backbone
        self.out_ch = out_ch

    def __call__(self, x, key, state):
        """
        Returns:
            logits, state   — logits are resized to full input resolution
        """
        k_enc, k4, k3, k2, k1, k_out = jr.split(key, 6)

        (f0, f1, f2, f3, f4), state = self.encoder(x, key=k_enc, state=state)

        d4, state = self.d4(f4, key=k4, state=state)
        d4 = _match(d4, f3)
        d4 = d4 + f3

        d3, state = self.d3(d4, key=k3, state=state)
        d3 = _match(d3, f2)
        d3 = d3 + f2

        d2, state = self.d2(d3, key=k2, state=state)
        d2 = _match(d2, f1)
        d2 = d2 + f1

        d1, state = self.d1(d2, key=k1, state=state)
        d1 = _match(d1, f0)
        d1 = d1 + f0

        logits = self.out_conv(d1, key=k_out)
        logits = jax.image.resize(
            logits,
            (logits.shape[0], x.shape[1], x.shape[2]),
            method="bilinear",
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
        train,  # training loop
        predict,
        make_augmax_augment,
        make_dice_bce_loss,
    )

    rng = np.random.RandomState(0)
    N, C, H, W, OUT_CH = 10, 3, 128, 128, 1  # now (N, C, H, W)

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
    model, state = eqx.nn.make_with_state(LinkNetResNet)(
        out_ch=OUT_CH,  # ← number of channels in your target mask
        backbone="resnet50",
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
        batch_size=32,  # smaller than 256 => fits CPU
        num_epochs=15,
        patience=4,
        key=train_key,
        augment_fn=augment_fn,  # our NCHW ↔ NHWC/HW wrapper
        lambda_spec=0.0,  # if you use spectral-norm reg
    )

    plt.plot(tr_loss, label="train")
    plt.plot(va_loss, label="val")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Synthetic TransUNet smoke-test")
    plt.show()
