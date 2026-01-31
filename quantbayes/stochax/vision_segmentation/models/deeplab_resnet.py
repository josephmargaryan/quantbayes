""" "

Author: Joseph Margaryan
Date: 2024-01-15

Example usage:
```python

import equinox as eqx
import jax.random as jr

from utils_pretrain   import load_imagenet_resnet50          # 18 / 34 / 50
from deeplabv3_resnet import DeepLabV3PlusResNet, ResNetEncoder

key = jr.key(0)
model, state = eqx.nn.make_with_state(DeepLabV3PlusResNet)(
    backbone="resnet50",           # "resnet18" or "resnet34" also work
    out_ch=1,                      # number of output channels (classes)
    key=key,
)

enc_loaded = load_imagenet_resnet50(model.encoder, "resnet50_imagenet.npz")

model = eqx.tree_at(
    lambda m: m.encoder,
    model,
    enc_loaded,
    is_leaf=lambda x: isinstance(x, ResNetEncoder),
)

print("✓ ResNet-50 backbone initialised from ImageNet")
```
"""

from __future__ import annotations
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.image
import jax.random as jr


from quantbayes.stochax.vision_segmentation.models.unet_backbone import (
    _match,
    ConvBlock,
    ResNetEncoder,
    _RESNET_SPECS,
)


class ConvBnRelu(eqx.Module):
    conv: eqx.nn.Conv2d
    bn: eqx.nn.BatchNorm

    def __init__(
        self,
        cin: int,
        cout: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        key,
    ):
        if padding is None:
            padding = dilation * (kernel_size // 2)
        k1, k2 = jr.split(key, 2)
        self.conv = eqx.nn.Conv2d(
            cin,
            cout,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            key=k1,
        )
        self.bn = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")

    def __call__(self, x, key, state: Any):
        x = self.conv(x, key=key)
        x, state = self.bn(x, state)
        x = jax.nn.relu(x)
        return x, state


class ASPP(eqx.Module):
    b1: ConvBnRelu
    b2: ConvBnRelu
    b3: ConvBnRelu
    b4: ConvBnRelu
    pool_proj: ConvBnRelu
    project: eqx.nn.Conv2d
    bn_proj: eqx.nn.BatchNorm

    def __init__(self, cin: int, cout: int, *, key):
        ks = jr.split(key, 6)
        self.b1 = ConvBnRelu(cin, cout, kernel_size=1, key=ks[0])
        self.b2 = ConvBnRelu(cin, cout, kernel_size=3, dilation=6, key=ks[1])
        self.b3 = ConvBnRelu(cin, cout, kernel_size=3, dilation=12, key=ks[2])
        self.b4 = ConvBnRelu(cin, cout, kernel_size=3, dilation=18, key=ks[3])
        self.pool_proj = ConvBnRelu(cin, cout, kernel_size=1, key=ks[4])
        self.project = eqx.nn.Conv2d(cout * 5, cout, 1, key=ks[5])
        self.bn_proj = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")

    def __call__(self, x, key, state):
        k1, k2, k3, k4, k5, k6 = jr.split(key, 6)

        y1, state = self.b1(x, key=k1, state=state)
        y2, state = self.b2(x, key=k2, state=state)
        y3, state = self.b3(x, key=k3, state=state)
        y4, state = self.b4(x, key=k4, state=state)

        gp = jnp.mean(x, axis=(1, 2), keepdims=True)
        gp, state = self.pool_proj(gp, key=k5, state=state)
        gp = jax.image.resize(
            gp, (gp.shape[0], x.shape[1], x.shape[2]), method="bilinear"
        )

        cat = jnp.concatenate([y1, y2, y3, y4, gp], axis=0)
        proj = self.project(cat, key=k6)
        proj, state = self.bn_proj(proj, state)
        proj = jax.nn.relu(proj)
        return proj, state


class DeepLabV3PlusResNet(eqx.Module):
    encoder: ResNetEncoder
    aspp: ASPP
    low_proj: eqx.nn.Conv2d
    low_bn: eqx.nn.BatchNorm
    dec1: ConvBnRelu
    dec2: ConvBnRelu
    out: eqx.nn.Conv2d
    backbone_name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        backbone: str = "resnet50",  # "resnet18" / "resnet34" / "resnet50"
        out_ch: int = 1,
        key,
    ):
        k_enc, *ks = jr.split(key, 8)

        self.encoder = ResNetEncoder(backbone, key=k_enc)
        chans = _RESNET_SPECS[backbone]["channels"]
        c_low = chans[1]
        c_high = chans[-1]
        #  Atrous Convolution ASPP Chen et al. 2017
        self.aspp = ASPP(c_high, 256, key=ks[0])

        self.low_proj = eqx.nn.Conv2d(c_low, 48, 1, key=ks[1])
        self.low_bn = eqx.nn.BatchNorm(48, axis_name="batch", mode="batch")
        self.dec1 = ConvBnRelu(256 + 48, 256, key=ks[2])
        self.dec2 = ConvBnRelu(256, 256, key=ks[3])

        self.out = eqx.nn.Conv2d(256, out_ch, 1, key=ks[4])

        self.backbone_name = backbone

    def __call__(self, x, key, state):
        k_enc, k_aspp, k_lp, k_d1, k_d2 = jr.split(key, 5)

        (conv1, l1, l2, l3, l4), state = self.encoder(x, key=k_enc, state=state)

        low = l1
        high = l4

        out_aspp, state = self.aspp(high, key=k_aspp, state=state)

        low = self.low_proj(low, key=k_lp)
        low, state = self.low_bn(low, state)
        low = jax.nn.relu(low)

        out_aspp_up = jax.image.resize(
            out_aspp,
            (out_aspp.shape[0], low.shape[1], low.shape[2]),
            method="bilinear",
        )

        cat = jnp.concatenate([out_aspp_up, low], axis=0)
        d1, state = self.dec1(cat, key=k_d1, state=state)
        d2, state = self.dec2(d1, key=k_d2, state=state)

        logits_low = self.out(d2)
        logits = jax.image.resize(
            logits_low,
            (logits_low.shape[0], x.shape[1], x.shape[2]),
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
    model, state = eqx.nn.make_with_state(DeepLabV3PlusResNet)(
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
