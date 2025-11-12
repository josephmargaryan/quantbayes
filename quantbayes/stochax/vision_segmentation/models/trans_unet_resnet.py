"""
TransUNet with ResNet-{18,34,50} encoder + ViT bottleneck.
Channel-first (C, H, W); single-sample forward.
Author: <you>
"""

from __future__ import annotations
from typing import List, Any
import einops, equinox as eqx, jax, jax.numpy as jnp, jax.random as jr
from quantbayes.stochax.vision_segmentation.models.unet_backbone import (
    _match,
    ConvBlock,
    ResNetEncoder,
    _RESNET_SPECS,
)


def _match(x, ref):
    h, w = x.shape[-2:]  # ints
    H, W = ref.shape[-2:]  # ints
    dh, dw = H - h, W - w
    if dh or dw:
        pads = [(0, 0)] * (x.ndim - 2) + [
            (dh // 2, dh - dh // 2),
            (dw // 2, dw - dw // 2),
        ]
        x = (
            jnp.pad(x, pads)
            if dh > 0 or dw > 0
            else x[
                (
                    ...,
                    slice(-dh // 2, -dh // 2 + ref.shape[-2]),
                    slice(-dw // 2, -dw // 2 + ref.shape[-1]),
                )
            ]
        )
    return x


class ConvBlock(eqx.Module):
    c1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    c2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm

    def __init__(self, cin, cout, *, key):
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


class Up(eqx.Module):
    up: eqx.nn.ConvTranspose2d
    conv: ConvBlock

    def __init__(self, cin, skip, cout, *, key):
        k1, k2 = jr.split(key, 2)
        self.up = eqx.nn.ConvTranspose2d(cin, cout, 2, stride=2, key=k1)
        self.conv = ConvBlock(cout + skip, cout, key=k2)

    def __call__(self, x, skip, *, key, state):
        k1, k2 = jr.split(key, 2)
        x = self.up(x, key=k1)
        x, skip = _match(x, skip), _match(skip, x)
        x = jnp.concatenate([skip, x], axis=0)
        x, state = self.conv(x, key=k2, state=state)
        return x, state


class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Linear
    patch: int

    def __init__(self, in_ch, embed, patch, *, key):
        self.patch = patch
        self.linear = eqx.nn.Linear(patch**2 * in_ch, embed, key=key)

    def __call__(self, x):  # (C,H,W) → (N,embed)
        return jax.vmap(self.linear)(
            einops.rearrange(
                x, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=self.patch, pw=self.patch
            )
        )


class TransformerBlock(eqx.Module):
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    drop1: eqx.nn.Dropout
    drop2: eqx.nn.Dropout

    def __init__(self, dim, mlp, heads, drop, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.ln1 = eqx.nn.LayerNorm(dim)
        self.ln2 = eqx.nn.LayerNorm(dim)
        self.attn = eqx.nn.MultiheadAttention(heads, dim, key=k1)
        self.fc1 = eqx.nn.Linear(dim, mlp, key=k2)
        self.fc2 = eqx.nn.Linear(mlp, dim, key=k3)
        self.drop1 = eqx.nn.Dropout(drop)
        self.drop2 = eqx.nn.Dropout(drop)

    def __call__(self, x, *, key):
        x_ = jax.vmap(self.ln1)(x)
        x = x + self.attn(x_, x_, x_)
        h = jax.vmap(self.fc1)(jax.vmap(self.ln2)(x))
        h = jax.nn.gelu(h)
        k1, k2 = jr.split(key, 2)
        h = self.drop2(jax.vmap(self.fc2)(self.drop1(h, key=k1)), key=k2)
        return x + h


class TransUNetResNet(eqx.Module):
    encoder: ResNetEncoder
    b_proj: eqx.nn.Conv2d
    patch_embed: PatchEmbedding
    vit_blocks: List[TransformerBlock]
    proj_back: eqx.nn.Linear
    d1: Up
    d2: Up
    d3: Up
    d4: Up
    out_conv: eqx.nn.Conv2d
    pos_embed: jnp.ndarray
    patch: int = eqx.field(static=True)
    backbone_name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        out_ch: int = 1,
        backbone: str = "resnet34",
        patch: int = 2,
        vit_dim: int = 256,
        vit_depth: int = 4,
        vit_heads: int = 4,
        vit_mlp: int = 512,
        dropout: float = 0.1,
        key,
    ):
        k_enc, *ks = jr.split(key, 7 + vit_depth)
        self.encoder = ResNetEncoder(backbone, key=k_enc)
        c1, c2, c3, c4, c5 = _RESNET_SPECS[backbone]["channels"]

        self.b_proj = eqx.nn.Conv2d(c5, vit_dim, 1, key=ks[0])
        self.patch = patch
        self.patch_embed = PatchEmbedding(vit_dim, vit_dim, patch, key=ks[1])
        self.pos_embed = jr.normal(ks[2], (10_000, vit_dim))

        self.vit_blocks = [
            TransformerBlock(vit_dim, vit_mlp, vit_heads, dropout, key=ks[3 + i])
            for i in range(vit_depth)
        ]
        self.proj_back = eqx.nn.Linear(vit_dim, c5, key=ks[3 + vit_depth])

        self.d1 = Up(c5, c4, c4, key=ks[-3])
        self.d2 = Up(c4, c3, c3, key=ks[-2])
        self.d3 = Up(c3, c2, c2, key=ks[-1])
        self.d4 = Up(c2, c1, c1, key=jr.split(key, 1)[0])  # fresh split
        self.out_conv = eqx.nn.Conv2d(c1, out_ch, 1, key=jr.split(key, 1)[0])

        self.backbone_name = backbone

    def __call__(self, x, key, state):
        k_enc, k_bproj, k_vit, k_dec, k_out = jr.split(key, 5)

        # 1) ResNet encoder
        (conv1, l1, l2, l3, l4), state = self.encoder(x, key=k_enc, state=state)

        # 2) CNN -> ViT tokens
        b = self.b_proj(l4, key=k_bproj)
        C, H, W = b.shape
        assert H % self.patch == 0 and W % self.patch == 0

        tokens = self.patch_embed(b)
        tokens += self.pos_embed[: tokens.shape[0]]

        # 3) ViT blocks
        for blk, bk in zip(self.vit_blocks, jr.split(k_vit, len(self.vit_blocks))):
            tokens = blk(tokens, key=bk)
        tokens = jax.vmap(self.proj_back)(tokens)
        h_s, w_s = H // self.patch, W // self.patch
        b_small = einops.rearrange(tokens, "(h w) c -> c h w", h=h_s, w=w_s)
        b_vit = einops.repeat(
            b_small, "c h w -> c (h ph) (w pw)", ph=self.patch, pw=self.patch
        )
        b_vit = _match(b_vit, l4)

        # 4) Decoder
        ks_dec = jr.split(k_dec, 4)
        d1, state = self.d1(b_vit, l3, key=ks_dec[0], state=state)
        d2, state = self.d2(d1, l2, key=ks_dec[1], state=state)
        d3, state = self.d3(d2, l1, key=ks_dec[2], state=state)
        d4, state = self.d4(d3, conv1, key=ks_dec[3], state=state)

        # 5) Classifier + resize
        logits = self.out_conv(d4, key=k_out)
        logits = jax.image.resize(
            logits, (logits.shape[0], x.shape[1], x.shape[2]), method="linear"
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
        make_augmax_augment,  # optional inference util
        make_dice_bce_loss,
    )

    rng = np.random.RandomState(0)
    N, C, H, W, OUT_CH = 10, 3, 128, 128, 1  # now (N, C, H, W)

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

    master_key = jr.PRNGKey(42)
    model_key, train_key = jr.split(master_key)
    model, state = eqx.nn.make_with_state(TransUNetResNet)(
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
