from __future__ import annotations
from typing import List, Tuple, Any

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


def _match(x: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    """Pad/crop so spatial dims of `x` match those of `ref`."""
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

    def __call__(self, x: jnp.ndarray, *, key, state):
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

    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray, *, key, state):
        k1, k2 = jr.split(key, 2)
        x = self.up(x, key=k1)
        x, skip = _match(x, skip), _match(skip, x)
        x = jnp.concatenate([skip, x], axis=0)
        x, state = self.conv(x, key=k2, state=state)
        return x, state


class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Linear
    patch_size: int

    def __init__(self, in_ch: int, embed_dim: int, patch_size: int, *, key):
        self.patch_size = patch_size
        self.linear = eqx.nn.Linear(patch_size**2 * in_ch, embed_dim, key=key)

    def __call__(self, x: Float[Array, "C H W"]) -> Float[Array, "N D"]:
        x = einops.rearrange(
            x,
            "c (h ph) (w pw) -> (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return jax.vmap(self.linear)(x)


class TransformerBlock(eqx.Module):
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    drop1: eqx.nn.Dropout
    drop2: eqx.nn.Dropout

    def __init__(self, dim: int, mlp_dim: int, heads: int, drop: float, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.ln1 = eqx.nn.LayerNorm(dim)
        self.ln2 = eqx.nn.LayerNorm(dim)
        self.attn = eqx.nn.MultiheadAttention(heads, dim, key=k1)
        self.fc1 = eqx.nn.Linear(dim, mlp_dim, key=k2)
        self.fc2 = eqx.nn.Linear(mlp_dim, dim, key=k3)
        self.drop1 = eqx.nn.Dropout(drop)
        self.drop2 = eqx.nn.Dropout(drop)

    def __call__(self, x: Float[Array, "N D"], *, key):
        x_norm = jax.vmap(self.ln1)(x)
        x = x + self.attn(x_norm, x_norm, x_norm)
        x_norm = jax.vmap(self.ln2)(x)
        h = jax.vmap(self.fc1)(x_norm)
        h = jax.nn.gelu(h)
        k1, k2 = jr.split(key, 2)
        h = self.drop1(h, key=k1)
        h = jax.vmap(self.fc2)(h)
        h = self.drop2(h, key=k2)
        return x + h


class TransUNet(eqx.Module):
    # encoder
    e1: ConvBlock
    e2: ConvBlock
    e3: ConvBlock
    e4: ConvBlock
    pool: eqx.nn.MaxPool2d
    # bottleneck
    b: ConvBlock
    patch_embed: PatchEmbedding
    pos_embed: jnp.ndarray
    vit_blocks: List[TransformerBlock]
    proj_back: eqx.nn.Linear
    # decoder
    d1: Up
    d2: Up
    d3: Up
    d4: Up
    out_conv: eqx.nn.Conv2d
    # static
    patch_size: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_ch: int = 3,
        out_ch: int = 1,
        base: int = 8,
        patch_size: int = 2,
        vit_dim: int = 256,
        vit_depth: int = 4,
        vit_heads: int = 4,
        vit_mlp_dim: int = 512,
        dropout: float = 0.1,
        key: PRNGKeyArray,
    ):
        k = list(jr.split(key, 14 + vit_depth))
        # encoder
        self.e1 = ConvBlock(in_ch, base, key=k[0])
        self.e2 = ConvBlock(base, base * 2, key=k[1])
        self.e3 = ConvBlock(base * 2, base * 4, key=k[2])
        self.e4 = ConvBlock(base * 4, base * 8, key=k[3])
        self.pool = eqx.nn.MaxPool2d(2, 2)
        # bottleneck
        self.b = ConvBlock(base * 8, base * 16, key=k[4])
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(base * 16, vit_dim, patch_size, key=k[5])
        self.pos_embed = jr.normal(k[6], (10_000, vit_dim))
        self.vit_blocks = [
            TransformerBlock(vit_dim, vit_mlp_dim, vit_heads, dropout, key=k[7 + i])
            for i in range(vit_depth)
        ]
        self.proj_back = eqx.nn.Linear(vit_dim, base * 16, key=k[7 + vit_depth])
        # decoder
        self.d1 = Up(base * 16, base * 8, base * 8, key=k[-6])
        self.d2 = Up(base * 8, base * 4, base * 4, key=k[-5])
        self.d3 = Up(base * 4, base * 2, base * 2, key=k[-4])
        self.d4 = Up(base * 2, base, base, key=k[-3])
        self.out_conv = eqx.nn.Conv2d(base, out_ch, 1, key=k[-2])

    # forward
    def __call__(self, x: jnp.ndarray, key, state):
        k_enc = jr.split(key, 5)
        e1, state = self.e1(x, key=k_enc[0], state=state)
        p1 = self.pool(e1)
        e2, state = self.e2(p1, key=k_enc[1], state=state)
        p2 = self.pool(e2)
        e3, state = self.e3(p2, key=k_enc[2], state=state)
        p3 = self.pool(e3)
        e4, state = self.e4(p3, key=k_enc[3], state=state)
        p4 = self.pool(e4)
        b, state = self.b(p4, key=k_enc[4], state=state)

        # ViT bottleneck
        C, H, W = b.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        tokens = self.patch_embed(b)
        tokens += self.pos_embed[: tokens.shape[0]]
        k_vit = jr.split(key, len(self.vit_blocks) + 1)[1:]
        for blk, bk in zip(self.vit_blocks, k_vit):
            tokens = blk(tokens, key=bk)
        tokens_cnn = jax.vmap(self.proj_back)(tokens)

        h_small, w_small = H // self.patch_size, W // self.patch_size
        b_small = einops.rearrange(tokens_cnn, "(h w) c -> c h w", h=h_small, w=w_small)
        b_vit = einops.repeat(
            b_small,
            "c h w -> c (h ph) (w pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )

        b_vit = _match(b_vit, b)

        # decoder
        k_dec = jr.split(key, 4)
        d1, state = self.d1(b_vit, e4, key=k_dec[0], state=state)
        d2, state = self.d2(d1, e3, key=k_dec[1], state=state)
        d3, state = self.d3(d2, e2, key=k_dec[2], state=state)
        d4, state = self.d4(d3, e1, key=k_dec[3], state=state)
        logits = self.out_conv(d4)
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
    N, C, H, W, OUT_CH = 2048, 3, 128, 128, 1  # now (N, C, H, W)

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
    model, state = eqx.nn.make_with_state(TransUNet)(
        in_ch=C,
        out_ch=OUT_CH,
        base=8,  # small for CPU speed; bump for real runs
        patch_size=2,
        vit_dim=128,
        vit_depth=2,
        vit_heads=4,
        vit_mlp_dim=256,
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
