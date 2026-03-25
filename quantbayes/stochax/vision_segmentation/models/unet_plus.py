"""
Equinox U-Net++ with selectable ResNet encoder backbone.

Backbones supported: "resnet18", "resnet34", "resnet50".
Nested skip-connection decoder with optional deep supervision.

Author: Joseph Margaryan
"""

from __future__ import annotations
from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.image

from quantbayes.stochax.vision_segmentation.models.unet_backbone import (
    _match,
    ConvBlock,
    ResNetEncoder,
    _RESNET_SPECS,
)


def _upsample_like(x: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    """Resize `x` to spatial size of `ref` with bilinear interpolation."""
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return jax.image.resize(
        x,
        (x.shape[0], ref.shape[-2], ref.shape[-1]),
        method="bilinear",
    )


class UNetPPResNet(eqx.Module):
    encoder: ResNetEncoder

    convs: Tuple[Tuple[ConvBlock, ...], ...]
    out_conv: eqx.nn.Conv2d
    sup_convs: Tuple[eqx.nn.Conv2d, ...] | None

    deep_supervision: bool = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        out_ch: int = 1,
        backbone: str = "resnet34",
        deep_supervision: bool = True,
        key,
    ):
        if backbone not in _RESNET_SPECS:
            raise ValueError(f"Unknown backbone '{backbone}'.")
        k_encoder, *k_rest = jr.split(key, 1 + 1000)
        self.encoder = ResNetEncoder(backbone, key=k_encoder)

        chans: List[int] = _RESNET_SPECS[backbone]["channels"]
        self.depth = len(chans)

        conv_rows = []
        k_iter = iter(k_rest)
        for i in range(self.depth - 1):
            row = []
            for j in range(self.depth - 1 - i):
                cin = (j + 1) * chans[i] + chans[i + 1]
                row.append(ConvBlock(cin, chans[i], key=next(k_iter)))
            conv_rows.append(tuple(row))
        self.convs = tuple(conv_rows)

        self.out_conv = eqx.nn.Conv2d(chans[0], out_ch, 1, key=next(k_iter))

        self.deep_supervision = deep_supervision
        if deep_supervision:
            sup_heads = []
            for _ in range(1, self.depth):
                sup_heads.append(eqx.nn.Conv2d(chans[0], out_ch, 1, key=next(k_iter)))
            self.sup_convs = tuple(sup_heads)
        else:
            self.sup_convs = None

    def __call__(self, x, key, state):
        """
        Returns:
            logits, state                if deep_supervision=False
            (logits_list), state         if deep_supervision=True
        """

        k_enc, *k_rest = jr.split(
            key,
            1
            + len(self.convs) * (self.depth - 1)
            + (self.depth if self.deep_supervision else 1),
        )
        enc_feats, state = self.encoder(x, key=k_enc, state=state)
        X = [[None] * self.depth for _ in range(self.depth)]
        for i, feat in enumerate(enc_feats):
            X[i][0] = feat

        k_iter = iter(k_rest)
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                prevs = [X[i][k] for k in range(j)]
                up = _upsample_like(X[i + 1][j - 1], X[i][0])
                cat = jnp.concatenate(prevs + [up], axis=0)
                X[i][j], state = self.convs[i][j - 1](
                    cat, key=next(k_iter), state=state
                )

        def _predict(feat, head, k):
            logit = head(feat, key=k)
            return _upsample_like(logit, x)

        if not self.deep_supervision:
            logits = _predict(X[0][self.depth - 1], self.out_conv, next(k_iter))
            return logits, state
        else:
            logits_list = []
            for j, head in enumerate(self.sup_convs, start=1):
                logits_list.append(_predict(X[0][j], head, next(k_iter)))
            return logits_list, state


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
        make_augmax_augment,  # optional inference util
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
    model, state = eqx.nn.make_with_state(UNetPPResNet)(
        out_ch=OUT_CH,
        deep_supervision=True,
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
