"""
Equinox Vision Transformer (ViT) with Torchvision weight loader.

- Single-sample forward with channel-first inputs [C, H, W]
- __call__(self, x, key, state) -> (logits, state)
- Learnable [CLS] and 1D positional embedding (+ robust 2D resize when seq length differs)
- Variants supported via loader: vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
- Custom Multi-Head Self-Attention with explicit q/k/v/out linears (easy, reliable weight mapping)

Torchvision weights
-------------------
1) Save torchvision ViT weights once (like your other models):
   ----------------------------------------------------------------
   # save_torchvision_vits.py
   from pathlib import Path
   import numpy as np
   from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14

   CHECKPOINTS = {
       "vit_b_16": (vit_b_16, "IMAGENET1K_V1"),
       "vit_b_32": (vit_b_32, "IMAGENET1K_V1"),
       "vit_l_16": (vit_l_16, "IMAGENET1K_V1"),
       "vit_l_32": (vit_l_32, "IMAGENET1K_V1"),
       # vit_h_14 uses SWAG weights in torchvision:
       "vit_h_14": (vit_h_14, "IMAGENET1K_SWAG_E2E_V1"),
   }

   def main():
       for name, (builder, weights_name) in CHECKPOINTS.items():
           print(f"⇢ downloading {name} …")
           model = builder(weights=weights_name)
           ckpt_path = Path(f"{name}_imagenet.npz")
           print(f"↳ saving → {ckpt_path}")
           np.savez(ckpt_path, **{k: v.cpu().numpy() for k, v in model.state_dict().items()})
           print(f"✓ done {ckpt_path}\\n")

   if __name__ == "__main__":
       main()
   ----------------------------------------------------------------

2) Initialize ViT and load weights:
   ----------------------------------------------------------------
   import equinox as eqx, jax.random as jr
   from quantbayes.stochax.vision_classification.models.vit import (
       VisionTransformer,
       load_imagenet_vit_b_16,   # or vit_b_32 / vit_l_16 / vit_l_32 / vit_h_14
   )

   H = W = 224
   patch = 16
   num_patches = (H // patch) * (W // patch)

   key = jr.PRNGKey(0)
   model, state = eqx.nn.make_with_state(VisionTransformer)(
       embedding_dim=768, hidden_dim=768*4, num_heads=12, num_layers=12,
       dropout_rate=0.1, patch_size=patch, num_patches=num_patches,
       num_classes=1000, channels=3, key=key,
   )

   # If your num_classes != 1000, set strict_fc=False to keep features and skip the head.
   model = load_imagenet_vit_b_16(model, "vit_b_16_imagenet.npz", strict_fc=True)
   ----------------------------------------------------------------
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import math
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.image as jimg
from jaxtyping import Array, Float, PRNGKeyArray


# --------------------------- Patch Embedding --------------------------- #
class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Linear
    patch_size: int
    in_ch: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        patch_size: int,
        key: PRNGKeyArray,
    ):
        self.patch_size = patch_size
        self.in_ch = input_channels
        self.out_dim = output_dim
        self.linear = eqx.nn.Linear(
            patch_size**2 * input_channels,
            output_dim,
            key=key,
        )

    def __call__(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "num_patches embedding_dim"]:
        ps = self.patch_size
        x = einops.rearrange(
            x,
            "c (h ph) (w pw) -> (h w) (c ph pw)",
            ph=ps,
            pw=ps,
        )
        x = jax.vmap(self.linear)(x)
        return x


# ------------------- Multi-Head Self-Attention (custom) ------------------- #
class MultiheadSelfAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, embed_dim: int, num_heads: int, *, key: PRNGKeyArray):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        kq, kk, kv, ko = jr.split(key, 4)
        self.q_proj = eqx.nn.Linear(embed_dim, embed_dim, key=kq)
        self.k_proj = eqx.nn.Linear(embed_dim, embed_dim, key=kk)
        self.v_proj = eqx.nn.Linear(embed_dim, embed_dim, key=kv)
        self.out_proj = eqx.nn.Linear(embed_dim, embed_dim, key=ko)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N_tokens, D]
        D = self.embed_dim
        H = self.num_heads
        hd = self.head_dim

        # Projections
        q = jax.vmap(self.q_proj)(x)  # [N, D]
        k = jax.vmap(self.k_proj)(x)  # [N, D]
        v = jax.vmap(self.v_proj)(x)  # [N, D]

        # Reshape to heads
        q = q.reshape(-1, H, hd).transpose(1, 0, 2)  # [H, N, hd]
        k = k.reshape(-1, H, hd).transpose(1, 2, 0)  # [H, hd, N]
        v = v.reshape(-1, H, hd).transpose(1, 0, 2)  # [H, N, hd]

        # Attention
        scale = 1.0 / math.sqrt(hd)
        scores = jnp.matmul(q * scale, k)  # [H, N, N]
        attn = jax.nn.softmax(scores, axis=-1)  # [H, N, N]
        ctx = jnp.matmul(attn, v)  # [H, N, hd]

        # Merge heads
        ctx = ctx.transpose(1, 0, 2).reshape(-1, D)  # [N, D]

        # Final projection
        out = jax.vmap(self.out_proj)(ctx)  # [N, D]
        return out


# --------------------------- Transformer Block --------------------------- #
class AttentionBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    attention: MultiheadSelfAttention
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
        key: PRNGKeyArray,
    ):
        key1, key2, key3 = jr.split(key, 3)
        self.layer_norm1 = eqx.nn.LayerNorm(embed_dim)
        self.layer_norm2 = eqx.nn.LayerNorm(embed_dim)
        self.attention = MultiheadSelfAttention(embed_dim, num_heads, key=key1)
        self.linear1 = eqx.nn.Linear(embed_dim, hidden_dim, key=key2)
        self.linear2 = eqx.nn.Linear(hidden_dim, embed_dim, key=key3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x: jnp.ndarray, key: PRNGKeyArray) -> jnp.ndarray:
        # Self-attn branch
        x_norm = jax.vmap(self.layer_norm1)(x)  # [N, D]
        attn_out = self.attention(x_norm)  # [N, D]
        x = x + attn_out

        # MLP branch
        x_norm = jax.vmap(self.layer_norm2)(x)
        mlp_hidden = jax.vmap(self.linear1)(x_norm)
        mlp_hidden = jax.nn.gelu(mlp_hidden)

        k1, k2 = jr.split(key, 2)
        mlp_hidden = self.dropout1(mlp_hidden, key=k1)
        mlp_out = jax.vmap(self.linear2)(mlp_hidden)
        mlp_out = self.dropout2(mlp_out, key=k2)

        x = x + mlp_out
        return x


# ------------------------------- ViT Model ------------------------------- #
class VisionTransformer(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray  # [1+N, D]
    cls_token: jnp.ndarray  # [1, D]
    attention_blocks: Tuple[AttentionBlock, ...]
    dropout: eqx.nn.Dropout
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    num_layers: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        patch_size: int,
        num_patches: int,
        num_classes: int,
        key: PRNGKeyArray,
        channels: int = 3,
    ):
        k1, k2, k3, k4, k5 = jr.split(key, 5)

        self.patch_embedding = PatchEmbedding(channels, embedding_dim, patch_size, k1)

        # +1 for CLS token; keep an explicit [1+N, D] param for positional embedding
        self.positional_embedding = jr.normal(k2, (num_patches + 1, embedding_dim))
        self.cls_token = jr.normal(k3, (1, embedding_dim))
        self.num_layers = num_layers
        self.embed_dim = embedding_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.channels = channels

        blocks = []
        block_keys = jr.split(k4, num_layers)
        for kb in block_keys:
            blocks.append(
                AttentionBlock(embedding_dim, hidden_dim, num_heads, dropout_rate, kb)
            )
        self.attention_blocks = tuple(blocks)

        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.norm = eqx.nn.LayerNorm(embedding_dim)
        self.head = eqx.nn.Linear(embedding_dim, num_classes, key=k5)

    def __call__(
        self, x: Float[Array, "channels height width"], key: PRNGKeyArray, state
    ):
        # Embed patches
        x = self.patch_embedding(x)  # [N_patches, D]

        # Prepend CLS and add pos
        x = jnp.concatenate((self.cls_token, x), axis=0)  # [1+N, D]
        pos = self.positional_embedding[: x.shape[0]]  # safe slice
        x = x + pos

        # Transformer
        keys = jr.split(key, self.num_layers + 1)
        x = self.dropout(x, key=keys[0])
        for block, k in zip(self.attention_blocks, keys[1:]):
            x = block(x, key=k)

        # CLS -> norm -> head
        x = self.norm(x[0])  # [D]
        logits = self.head(x)  # [C]
        return logits, state


# -------------------------- Weight Loading Utils -------------------------- #
def _resize_pos_embedding(tv_pos: jnp.ndarray, target_len: int) -> jnp.ndarray:
    """
    tv_pos: [1, L_tv, D]; returns [1, L_target, D]
    Assumes square grid for patch tokens; keeps CLS at index 0.
    """
    B, L_tv, D = tv_pos.shape
    assert B == 1
    cls_tok = tv_pos[:, :1, :]  # [1,1,D]
    seq = tv_pos[:, 1:, :]  # [1, L_tv-1, D]

    if (1 + seq.shape[1]) == target_len:
        return tv_pos

    # Infer old/new grid sizes
    old_n = seq.shape[1]
    new_n = target_len - 1
    old_hw = int(round(math.sqrt(old_n)))
    new_hw = int(round(math.sqrt(new_n)))
    if old_hw * old_hw != old_n or new_hw * new_hw != new_n:
        # Fallback: simple linear resize over length (rare); reshape as [1, old_n, D]
        seq_resized = jimg.resize(seq, (1, new_n, D), method="linear")
        return jnp.concatenate([cls_tok, seq_resized], axis=1)

    # 2D grid → resize → flatten
    seq_2d = seq.reshape(1, old_hw, old_hw, D)  # [1, H, W, D]
    seq_resized = jimg.resize(seq_2d, (1, new_hw, new_hw, D), method="linear")
    seq_resized = seq_resized.reshape(1, new_hw * new_hw, D)
    return jnp.concatenate([cls_tok, seq_resized], axis=1)


def load_torchvision_vit(
    model: VisionTransformer, npz_path: str, *, strict_fc: bool = True
) -> VisionTransformer:
    """
    Load a torchvision ViT .npz (from state_dict()) into this model.
    Handles:
      - conv_proj (patchify) → flatten into our Linear patch embed
      - class_token / encoder.pos_embedding → cls_token / positional_embedding (with 2D resize as needed)
      - per-layer ln_1/ln_2, self_attention.{in_proj,out_proj}, mlp.fc{1,2}
      - final encoder.ln and heads.head
    """
    import numpy as np

    raw = dict(np.load(npz_path))

    pt: Dict[str, jnp.ndarray] = {}

    # 1) Patch embedding: conv_proj -> linear
    if "conv_proj.weight" in raw:
        W = jnp.asarray(raw["conv_proj.weight"])  # [E, C, ph, pw]
        E, C, ph, pw = W.shape
        W_lin = W.reshape(E, C * ph * pw)  # [E, C*ph*pw]
        pt["patch_embedding.linear.weight"] = W_lin
        if "conv_proj.bias" in raw:
            pt["patch_embedding.linear.bias"] = jnp.asarray(raw["conv_proj.bias"])

    # 2) CLS + positional embedding
    if "class_token" in raw:
        cls = jnp.asarray(raw["class_token"]).reshape(1, -1)  # [1, D]
        pt["cls_token"] = cls
    if "encoder.pos_embedding" in raw:
        pos = jnp.asarray(raw["encoder.pos_embedding"])  # [1, L_tv, D]
        L_target = model.positional_embedding.shape[0]
        pos = _resize_pos_embedding(pos, L_target).squeeze(0)  # [L_target, D]
        pt["positional_embedding"] = pos

    # 3) Transformer blocks
    n_layers = len(model.attention_blocks)
    D = model.embed_dim
    for i in range(n_layers):
        # LN1
        w = raw.get(f"encoder.layers.{i}.ln_1.weight")
        b = raw.get(f"encoder.layers.{i}.ln_1.bias")
        if w is not None:
            pt[f"attention_blocks.{i}.layer_norm1.weight"] = jnp.asarray(w)
        if b is not None:
            pt[f"attention_blocks.{i}.layer_norm1.bias"] = jnp.asarray(b)

        # Self-attn QKV
        W_qkv = raw.get(f"encoder.layers.{i}.self_attention.in_proj_weight")
        b_qkv = raw.get(f"encoder.layers.{i}.self_attention.in_proj_bias")
        if W_qkv is not None:
            W_qkv = jnp.asarray(W_qkv)  # [3D, D]
            Wq, Wk, Wv = jnp.split(W_qkv, 3, axis=0)
            pt[f"attention_blocks.{i}.attention.q_proj.weight"] = Wq
            pt[f"attention_blocks.{i}.attention.k_proj.weight"] = Wk
            pt[f"attention_blocks.{i}.attention.v_proj.weight"] = Wv
        if b_qkv is not None:
            b_qkv = jnp.asarray(b_qkv)  # [3D]
            bq, bk, bv = jnp.split(b_qkv, 3, axis=0)
            pt[f"attention_blocks.{i}.attention.q_proj.bias"] = bq
            pt[f"attention_blocks.{i}.attention.k_proj.bias"] = bk
            pt[f"attention_blocks.{i}.attention.v_proj.bias"] = bv

        # Self-attn out proj
        W_o = raw.get(f"encoder.layers.{i}.self_attention.out_proj.weight")
        b_o = raw.get(f"encoder.layers.{i}.self_attention.out_proj.bias")
        if W_o is not None:
            pt[f"attention_blocks.{i}.attention.out_proj.weight"] = jnp.asarray(W_o)
        if b_o is not None:
            pt[f"attention_blocks.{i}.attention.out_proj.bias"] = jnp.asarray(b_o)

        # LN2
        w = raw.get(f"encoder.layers.{i}.ln_2.weight")
        b = raw.get(f"encoder.layers.{i}.ln_2.bias")
        if w is not None:
            pt[f"attention_blocks.{i}.layer_norm2.weight"] = jnp.asarray(w)
        if b is not None:
            pt[f"attention_blocks.{i}.layer_norm2.bias"] = jnp.asarray(b)

        # MLP
        w = raw.get(f"encoder.layers.{i}.mlp.fc1.weight")
        b = raw.get(f"encoder.layers.{i}.mlp.fc1.bias")
        if w is not None:
            pt[f"attention_blocks.{i}.linear1.weight"] = jnp.asarray(w)
        if b is not None:
            pt[f"attention_blocks.{i}.linear1.bias"] = jnp.asarray(b)

        w = raw.get(f"encoder.layers.{i}.mlp.fc2.weight")
        b = raw.get(f"encoder.layers.{i}.mlp.fc2.bias")
        if w is not None:
            pt[f"attention_blocks.{i}.linear2.weight"] = jnp.asarray(w)
        if b is not None:
            pt[f"attention_blocks.{i}.linear2.bias"] = jnp.asarray(b)

    # 4) Final norm + head
    if "encoder.ln.weight" in raw:
        pt["norm.weight"] = jnp.asarray(raw["encoder.ln.weight"])
    if "encoder.ln.bias" in raw:
        pt["norm.bias"] = jnp.asarray(raw["encoder.ln.bias"])

    if "heads.head.weight" in raw and "heads.head.bias" in raw:
        W = jnp.asarray(raw["heads.head.weight"])
        B = jnp.asarray(raw["heads.head.bias"])
        want_out, want_in = model.head.weight.shape
        have_out, have_in = W.shape
        if (want_out, want_in) != (have_out, have_in):
            if strict_fc:
                raise ValueError(
                    f"FC shape mismatch: want {(want_out, want_in)} vs have {(have_out, have_in)}. "
                    f"Set strict_fc=False to skip loading the head."
                )
        else:
            pt["head.weight"] = W
            pt["head.bias"] = B

    # Copy into pytree
    return _copy_into_tree(model, pt, prefix="")


def _copy_into_tree(obj, pt: Dict[str, jnp.ndarray], prefix: str = ""):
    """Recursively copy Linear / LayerNorm params (and ndarray leaves) into an Equinox pytree."""
    if isinstance(obj, eqx.Module):
        for name, attr in vars(obj).items():
            full = f"{prefix}{name}"

            # nn.Linear
            if isinstance(attr, eqx.nn.Linear):
                new_attr = attr
                w_key, b_key = f"{full}.weight", f"{full}.bias"
                if w_key in pt:
                    new_attr = eqx.tree_at(lambda m: m.weight, new_attr, pt[w_key])
                if b_key in pt:
                    new_attr = eqx.tree_at(lambda m: m.bias, new_attr, pt[b_key])
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, new_attr)
                continue

            # LayerNorm
            if isinstance(attr, eqx.nn.LayerNorm):
                w_key, b_key = f"{full}.weight", f"{full}.bias"
                w_val = pt.get(w_key, getattr(attr, "weight"))
                b_val = pt.get(b_key, getattr(attr, "bias"))
                obj = eqx.tree_at(
                    lambda m: (getattr(m, name).weight, getattr(m, name).bias),
                    obj,
                    (w_val, b_val),
                )
                continue

            # Tuples of submodules (e.g., attention_blocks)
            if isinstance(attr, tuple):
                new_tuple = []
                for i, child in enumerate(attr):
                    child_full = f"{full}.{i}"
                    new_tuple.append(_copy_into_tree(child, pt, prefix=child_full))
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, tuple(new_tuple))
                continue

            # Raw ndarray leaves (cls_token, positional_embedding)
            if isinstance(attr, jnp.ndarray) and full in pt:
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, pt[full])
                continue

            # Other stateless parts (Dropout) -> skip
        return obj

    if isinstance(obj, tuple):
        return tuple(_copy_into_tree(x, pt, prefix=prefix) for x in obj)

    return obj


# --------------- Convenience per-arch loader wrappers (TV) --------------- #
def load_imagenet_vit_b_16(
    model: VisionTransformer, npz="vit_b_16_imagenet.npz", strict_fc: bool = True
) -> VisionTransformer:
    return load_torchvision_vit(model, npz, strict_fc=strict_fc)


def load_imagenet_vit_b_32(
    model: VisionTransformer, npz="vit_b_32_imagenet.npz", strict_fc: bool = True
) -> VisionTransformer:
    return load_torchvision_vit(model, npz, strict_fc=strict_fc)


def load_imagenet_vit_l_16(
    model: VisionTransformer, npz="vit_l_16_imagenet.npz", strict_fc: bool = True
) -> VisionTransformer:
    return load_torchvision_vit(model, npz, strict_fc=strict_fc)


def load_imagenet_vit_l_32(
    model: VisionTransformer, npz="vit_l_32_imagenet.npz", strict_fc: bool = True
) -> VisionTransformer:
    return load_torchvision_vit(model, npz, strict_fc=strict_fc)


def load_imagenet_vit_h_14(
    model: VisionTransformer, npz="vit_h_14_imagenet.npz", strict_fc: bool = True
) -> VisionTransformer:
    return load_torchvision_vit(model, npz, strict_fc=strict_fc)


# ------------------------------- Smoke test ------------------------------- #
if __name__ == "__main__":
    """
    Synthetic classification smoke test for ViT (B/16-like).
    Replace with a real dataset in practice.
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
    N, C, H, W, NUM_CLASSES = 512, 3, 224, 224, 10
    PATCH = 16
    assert H % PATCH == 0 and W % PATCH == 0, "H/W must be multiples of patch size."
    NUM_PATCHES = (H // PATCH) * (W // PATCH)

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

    model, state = eqx.nn.make_with_state(VisionTransformer)(
        embedding_dim=768,  # ViT-B
        hidden_dim=768 * 4,  # MLP ratio 4
        num_heads=12,
        num_layers=12,
        dropout_rate=0.1,
        patch_size=PATCH,
        num_patches=NUM_PATCHES,
        num_classes=NUM_CLASSES,
        channels=3,
        key=model_key,
    )

    # Optional pretrained load (skips head if shapes mismatch)
    # model = load_imagenet_vit_b_16(model, "vit_b_16_imagenet.npz", strict_fc=False)

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
        batch_size=32,
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
    plt.title("ViT-B/16 smoke test")
    plt.show()
