# vit_dino_eqx.py
"""
Equinox DINO(v2) ViT encoder + thin classifier/segmentation wrappers.

- Single-sample forward, channel-first input x: [C,H,W].
- __call__(self, x, key, state) -> (out, state).
- Encoder returns ((cls, grid), state):
    cls: [D] or None (if use_cls=False)
    grid: [D, H_p, W_p] patch-grid features (H_p = H//patch).
- Supports DINOv2 "register" tokens (num_registers in {0,4,8} typically).
- Positional embeddings stored as a learned 2D grid (pretrain resolution)
  and bilinearly interpolated at run time for any input size.

Best practices:
- No BatchNorm/stateful layers; state is threaded but unused.
- Minimal, faithful ViT (pre-norm, GELU MLP, fused QKV attention).
- No Dropout/DropPath inside (DINO uses strong augmentation/multicrop during pretrain).
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# ------------------------- specs ------------------------- #
DINOV2_SPECS: Dict[str, Dict] = {
    # DINOv2 variants, patch size 14
    "vits14": dict(patch=14, dim=384, depth=12, heads=6),
    "vitb14": dict(patch=14, dim=768, depth=12, heads=12),
    "vitl14": dict(patch=14, dim=1024, depth=24, heads=16),
    "vitg14": dict(patch=14, dim=1536, depth=40, heads=24),
}


# ------------------------- helpers ------------------------- #
def _to_patches(
    x: jnp.ndarray, conv: eqx.nn.Conv2d
) -> Tuple[jnp.ndarray, Tuple[int, int]]:
    # x: [C,H,W] → conv: [D, H', W'] → tokens [H'*W', D]
    z = conv(x, key=None)
    D, Hh, Ww = z.shape
    tokens = jnp.reshape(jnp.moveaxis(z, 0, -1), (Hh * Ww, D))
    return tokens, (Hh, Ww)


def _interp_pos_grid(pos_grid: jnp.ndarray, Hh: int, Ww: int) -> jnp.ndarray:
    # pos_grid: [Gh, Gw, D] → bilinear to [Hh, Ww, D]
    Gh, Gw, D = pos_grid.shape
    ys = jnp.linspace(0, Gh - 1, Hh)
    xs = jnp.linspace(0, Gw - 1, Ww)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    y0 = jnp.floor(yy).astype(jnp.int32)
    x0 = jnp.floor(xx).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, Gh - 1)
    x1 = jnp.minimum(x0 + 1, Gw - 1)
    wy = yy - y0
    wx = xx - x0
    p00 = pos_grid[y0, x0]  # [Hh,Ww,D]
    p10 = pos_grid[y1, x0]
    p01 = pos_grid[y0, x1]
    p11 = pos_grid[y1, x1]
    top = (1 - wx)[..., None] * p00 + wx[..., None] * p01
    bot = (1 - wx)[..., None] * p10 + wx[..., None] * p11
    return (1 - wy)[..., None] * top + wy[..., None] * bot  # [Hh,Ww,D]


# ------------------------- core blocks ------------------------- #
class Attention(eqx.Module):
    qkv: eqx.nn.Linear  # fused [D -> 3D]
    proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def __init__(self, dim: int, num_heads: int, *, key):
        k1, k2 = jr.split(key, 2)
        self.qkv = eqx.nn.Linear(dim, 3 * dim, key=k1)
        self.proj = eqx.nn.Linear(dim, dim, key=k2)
        object.__setattr__(self, "num_heads", int(num_heads))
        head_dim = dim // num_heads
        object.__setattr__(self, "scale", 1.0 / (head_dim**0.5))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [N, D]
        N, D = x.shape
        H = self.num_heads
        Dh = D // H

        qkv = self.qkv(x)  # [N, 3D]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # [N, D] each

        def _reshape(a):  # [N, D] → [H, N, Dh]
            return a.reshape(N, H, Dh).transpose(1, 0, 2)

        q, k, v = map(_reshape, (q, k, v))

        attn = jax.nn.softmax(jnp.einsum("hnd,hmd->hnm", q * self.scale, k), axis=-1)
        out = jnp.einsum("hnm,hmd->hnd", attn, v)  # [H,N,Dh]
        out = out.transpose(1, 0, 2).reshape(N, D)  # [N,D]
        return self.proj(out)


class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim: int, mlp_ratio: float, *, key):
        k1, k2 = jr.split(key, 2)
        hidden = int(dim * mlp_ratio)
        self.fc1 = eqx.nn.Linear(dim, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, dim, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.fc2(jax.nn.gelu(self.fc1(x)))


class ViTBlock(eqx.Module):
    ln1: eqx.nn.LayerNorm
    attn: Attention
    ln2: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(self, dim: int, heads: int, mlp_ratio: float, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.ln1 = eqx.nn.LayerNorm(dim, elementwise_affine=True)
        self.attn = Attention(dim, heads, key=k1)
        self.ln2 = eqx.nn.LayerNorm(dim, elementwise_affine=True)
        self.mlp = MLP(dim, mlp_ratio, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ------------------------- encoder ------------------------- #
class DINOViTEncoder(eqx.Module):
    # patch embed
    patch_embed: eqx.nn.Conv2d  # 3→D, kernel=stride=patch
    # tokens
    cls_token: jnp.ndarray  # [D] or empty
    reg_tokens: Optional[jnp.ndarray]  # [R, D] or None
    # learned positional grid at pretrain resolution
    pos_grid: jnp.ndarray  # [Gh, Gw, D]
    # transformer
    blocks: Tuple[ViTBlock, ...]
    norm: eqx.nn.LayerNorm

    # statics
    dim: int = eqx.field(static=True)
    patch: int = eqx.field(static=True)
    num_registers: int = eqx.field(static=True)
    use_cls: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        image_size: int = 518,  # DINOv2 pretrain crop; grid = (image_size // patch)
        patch: int = 14,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        num_registers: int = 0,  # 0/4/8 for DINOv2
        use_cls: bool = True,
        key,
    ):
        k_conv, k_tok, k_blocks = jr.split(key, 3)
        self.patch_embed = eqx.nn.Conv2d(
            3, dim, kernel_size=patch, stride=patch, padding=0, key=k_conv
        )

        Gh = max(1, image_size // patch)
        Gw = max(1, image_size // patch)
        self.pos_grid = jr.normal(k_tok, (Gh, Gw, dim)) * 0.02

        self.cls_token = (
            jr.normal(jr.fold_in(k_tok, 1), (dim,)) * 0.02
            if use_cls
            else jnp.zeros((0,), dtype=jnp.float32)
        )
        self.reg_tokens = (
            jr.normal(jr.fold_in(k_tok, 2), (num_registers, dim)) * 0.02
            if num_registers > 0
            else None
        )

        ks = jr.split(k_blocks, depth)
        self.blocks = tuple(
            ViTBlock(dim, heads, mlp_ratio, key=ks[i]) for i in range(depth)
        )
        self.norm = eqx.nn.LayerNorm(dim, elementwise_affine=True)

        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "patch", patch)
        object.__setattr__(self, "num_registers", int(num_registers))
        object.__setattr__(self, "use_cls", bool(use_cls))

    def __call__(self, x: jnp.ndarray, key, state):
        # x: [C,H,W]; returns (cls, grid) where grid is [D,H//patch,W//patch]
        tokens, (Hh, Ww) = _to_patches(x, self.patch_embed)  # [N, D]
        pos = _interp_pos_grid(self.pos_grid, Hh, Ww).reshape(Hh * Ww, self.dim)
        seq = tokens + pos

        if self.num_registers > 0:
            seq = jnp.concatenate([self.reg_tokens, seq], axis=0)
        if self.use_cls:
            seq = jnp.concatenate([self.cls_token[None, :], seq], axis=0)

        for blk in self.blocks:
            seq = blk(seq)

        seq = self.norm(seq)

        off = 0
        cls = None
        if self.use_cls:
            cls = seq[0]
            off = 1
        if self.num_registers > 0:
            off += self.num_registers
        patch_tok = seq[off:]  # [Hh*Ww, D]
        grid = jnp.moveaxis(
            jnp.reshape(patch_tok, (Hh, Ww, self.dim)), -1, 0
        )  # [D,Hh,Ww]
        return (cls, grid), state


# ------------------------- thin wrappers ------------------------- #
class DINOClassifier(eqx.Module):
    """Linear/head-on-CLS classifier. Freeze encoder for linear probe, or fine-tune."""

    encoder: DINOViTEncoder
    head: eqx.nn.Linear  # linear probe on CLS

    def __init__(self, num_classes: int, *, encoder: DINOViTEncoder, key):
        if not encoder.use_cls:
            raise ValueError("DINOClassifier requires use_cls=True in encoder.")
        self.encoder = encoder
        self.head = eqx.nn.Linear(encoder.dim, num_classes, key=key)

    def __call__(self, x, key, state):
        (cls, _), state = self.encoder(x, key=key, state=state)
        logits = self.head(cls)
        return logits, state


class DINODenseHead(eqx.Module):
    """Minimal dense head for segmentation: 1x1 conv on patch grid + bilinear upsample."""

    proj: eqx.nn.Conv2d

    def __init__(self, in_ch: int, out_ch: int, *, key):
        self.proj = eqx.nn.Conv2d(in_ch, out_ch, kernel_size=1, key=key)

    def __call__(self, grid: jnp.ndarray, image_hw: Tuple[int, int], *, key=None):
        # grid: [C,Hh,Ww] -> logits: [out_ch,H,W]
        H, W = image_hw
        logits = self.proj(grid, key=key)
        return jax.image.resize(logits, (logits.shape[0], H, W), method="bilinear")


class DINOSegmenter(eqx.Module):
    """Encoder + simple dense head wrapper with your standard call signature."""

    encoder: DINOViTEncoder
    head: DINODenseHead

    def __init__(self, out_ch: int, *, encoder: DINOViTEncoder, key):
        if encoder.use_cls:
            raise ValueError("DINOSegmenter expects use_cls=False for dense tasks.")
        self.encoder = encoder
        self.head = DINODenseHead(encoder.dim, out_ch, key=key)

    def __call__(self, x, key, state):
        (_, grid), state = self.encoder(x, key=key, state=state)
        logits = self.head(grid, image_hw=x.shape[-2:], key=None)
        return logits, state


# ------------------------- convenience builder ------------------------- #
def build_dinov2_encoder(
    name: str,
    *,
    image_size: int = 518,
    num_registers: int = 0,
    use_cls: bool = True,
    key,
) -> DINOViTEncoder:
    if name not in DINOV2_SPECS:
        raise ValueError(f"Unknown spec '{name}'. Valid: {list(DINOV2_SPECS.keys())}")
    spec = DINOV2_SPECS[name]
    return DINOViTEncoder(
        image_size=image_size,
        patch=spec["patch"],
        dim=spec["dim"],
        depth=spec["depth"],
        heads=spec["heads"],
        num_registers=num_registers,
        use_cls=use_cls,
        key=key,
    )
