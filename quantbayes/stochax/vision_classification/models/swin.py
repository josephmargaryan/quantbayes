"""
Equinox Swin Transformer (Tiny/Small/Base) with Torchvision weight loader.

- Single-sample CHW input
- __call__(self, x, key, state) -> (logits, state)
- Implements PatchEmbed, BasicLayer with SwinTransformerBlock (W-MSA / SW-MSA),
  relative position bias, PatchMerging, MLP, LayerNorm, DropPath (stochastic depth)
- Torchvision weights: save state_dict() to .npz and load here (features + head)

Torchvision weights
-------------------
1) Save torchvision weights once (exactly like your other models):
   ----------------------------------------------------------------
   # save_torchvision_swins.py
   from pathlib import Path
   import numpy as np
   from torchvision.models import swin_t, swin_s, swin_b

   CHECKPOINTS = {
       "swin_t": swin_t,
       "swin_s": swin_s,
       "swin_b": swin_b,
   }

   def main():
       for name, builder in CHECKPOINTS.items():
           print(f"⇢ downloading {name} …")
           model = builder(weights="IMAGENET1K_V1")
           ckpt_path = Path(f"{name}_imagenet.npz")
           print(f"↳ saving → {ckpt_path}")
           np.savez(ckpt_path, **{k: v.cpu().numpy() for k, v in model.state_dict().items()})
           print(f"✓ done {ckpt_path}\\n")

   if __name__ == "__main__":
       main()
   ----------------------------------------------------------------

2) Initialize and load:
   ----------------------------------------------------------------
   import equinox as eqx, jax.random as jr
   from quantbayes.stochax.vision_classification.models.swin import (
       SwinTransformer, load_imagenet_swin_t,  # ..._s / ..._b
   )

   key = jr.PRNGKey(0)
   model, state = eqx.nn.make_with_state(SwinTransformer)(
       arch="swin_t",            # "swin_t", "swin_s", "swin_b"
       num_classes=1000,
       key=key,
   )

   # If your num_classes != 1000, set strict_fc=False to keep features and skip the head.
   model = load_imagenet_swin_t(model, "swin_t_imagenet.npz", strict_fc=True)
   ----------------------------------------------------------------
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# --------------------------- utilities --------------------------- #
class DropPath(eqx.Module):
    """Stochastic depth: drops residual branch with prob=rate (single-sample)."""

    rate: float = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray, *, key: jnp.ndarray) -> jnp.ndarray:
        if self.rate <= 0.0:
            return x
        keep = 1.0 - self.rate
        mask = jr.bernoulli(key, p=keep)
        return x * (mask.astype(x.dtype) / keep)


class LayerNorm2d(eqx.Module):
    """LayerNorm over channel dim for CHW tensors (channels-last semantics)."""

    weight: jnp.ndarray
    bias: jnp.ndarray
    eps: float = eqx.field(static=True)

    def __init__(self, channels: int, eps: float = 1e-5):
        self.weight = jnp.ones((channels,), dtype=jnp.float32)
        self.bias = jnp.zeros((channels,), dtype=jnp.float32)
        self.eps = float(eps)

    def __call__(self, x: jnp.ndarray, key=None, state=None):
        # x: [C,H,W] -> [H,W,C]
        x_hwc = jnp.moveaxis(x, 0, -1)
        mean = jnp.mean(x_hwc, axis=-1, keepdims=True)
        var = jnp.var(x_hwc, axis=-1, keepdims=True)
        x_norm = (x_hwc - mean) / jnp.sqrt(var + self.eps)
        x_norm = x_norm * self.weight + self.bias
        y = jnp.moveaxis(x_norm, -1, 0)
        return y, state


def _window_partition(x_hwc: jnp.ndarray, ws: int) -> jnp.ndarray:
    """[H,W,C] -> [nW, ws*ws, C]"""
    H, W, C = x_hwc.shape
    x = x_hwc.reshape(H // ws, ws, W // ws, ws, C)
    x = jnp.transpose(x, (0, 2, 1, 3, 4))  # [nH, nW, ws, ws, C]
    return x.reshape(-1, ws * ws, C)


def _window_unpartition(windows: jnp.ndarray, ws: int, H: int, W: int) -> jnp.ndarray:
    """[nW, ws*ws, C] -> [H,W,C]"""
    C = windows.shape[-1]
    x = windows.reshape(H // ws, W // ws, ws, ws, C)
    x = jnp.transpose(x, (0, 2, 1, 3, 4))  # [nH, ws, nW, ws, C]
    return x.reshape(H, W, C)


def _pad_to_window_size(
    x_chw: jnp.ndarray, ws: int
) -> Tuple[jnp.ndarray, Tuple[int, int]]:
    """Zero-pad CHW so H and W are multiples of ws. Returns padded tensor and (pad_h, pad_w)."""
    C, H, W = x_chw.shape
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h == 0 and pad_w == 0:
        return x_chw, (0, 0)
    pads = [(0, 0), (0, pad_h), (0, pad_w)]
    return jnp.pad(x_chw, pads), (pad_h, pad_w)


def _compute_attn_mask(H: int, W: int, ws: int, shift: int) -> jnp.ndarray:
    """Create attention mask for SW-MSA on padded HxW; returns [nW, ws*ws, ws*ws] with 0/-inf."""
    if shift == 0:
        return jnp.zeros(((H // ws) * (W // ws), ws * ws, ws * ws), dtype=jnp.float32)

    img_mask = jnp.zeros((1, H, W, 1), dtype=jnp.int32)
    cnt = 0
    h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    mask = img_mask
    for h in h_slices:
        for w in w_slices:
            pad = jnp.full((1, mask.shape[1], mask.shape[2], 1), cnt, dtype=jnp.int32)
            # apply only to region
            mask = mask.at[:, h, w, :].set(pad[:, h, w, :])
            cnt += 1
    mask_windows = _window_partition(mask[0, :, :, :], ws)  # [nW, ws*ws, 1]
    mask_windows = mask_windows.squeeze(-1)  # [nW, ws*ws]
    attn_mask = (
        mask_windows[:, None, :] - mask_windows[:, :, None]
    )  # [nW, ws*ws, ws*ws]
    attn_mask = jnp.where(attn_mask == 0, 0.0, -1e9)  # use large negative for masking
    return attn_mask


# --------------------------- building blocks --------------------------- #
class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    drop1: eqx.nn.Dropout
    drop2: eqx.nn.Dropout

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0, *, key):
        k1, k2 = jr.split(key, 2)
        hidden = int(dim * mlp_ratio)
        self.fc1 = eqx.nn.Linear(dim, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, dim, key=k2)
        self.drop1 = eqx.nn.Dropout(drop)
        self.drop2 = eqx.nn.Dropout(drop)

    def __call__(self, x_tok: jnp.ndarray, key: jnp.ndarray):
        # x_tok: [N, C] tokens
        k1, k2 = jr.split(key, 2)
        x = jax.vmap(self.fc1)(x_tok)
        x = jax.nn.gelu(x)
        x = self.drop1(x, key=k1)
        x = jax.vmap(self.fc2)(x)
        x = self.drop2(x, key=k2)
        return x


class WindowAttention(eqx.Module):
    """Window based multi-head self-attention with relative position bias."""

    dim: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    attn_drop: eqx.nn.Dropout
    proj_drop: eqx.nn.Dropout
    relative_position_bias_table: jnp.ndarray  # [(2*ws-1)^2, num_heads]
    relative_position_index: jnp.ndarray  # [ws*ws, ws*ws]; not trained

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        *,
        key,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv = eqx.nn.Linear(dim, 3 * dim, key=k1)
        self.proj = eqx.nn.Linear(dim, dim, key=k2)
        self.attn_drop = eqx.nn.Dropout(attn_drop)
        self.proj_drop = eqx.nn.Dropout(proj_drop)

        # relative position bias
        ws = window_size
        self.relative_position_bias_table = jr.normal(
            k3, ((2 * ws - 1) * (2 * ws - 1), num_heads)
        )

        # precompute pair-wise relative position index for each token inside the window
        coords_h = jnp.arange(ws)
        coords_w = jnp.arange(ws)
        coords = jnp.stack(
            jnp.meshgrid(coords_h, coords_w, indexing="ij")
        )  # [2, ws, ws]
        coords_flatten = coords.reshape(2, -1)  # [2, ws*ws]
        rel_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, ws*ws, ws*ws]
        rel_coords = rel_coords.transpose(1, 2, 0)  # [ws*ws, ws*ws, 2]
        rel_coords = rel_coords + jnp.array([ws - 1, ws - 1])
        rel_coords = rel_coords.at[:, :, 0].multiply(2 * ws - 1)
        rel_position_index = rel_coords.sum(-1)  # [ws*ws, ws*ws]
        self.relative_position_index = rel_position_index

    def __call__(
        self, x_win: jnp.ndarray, attn_mask: jnp.ndarray | None, key: jnp.ndarray
    ):
        """
        x_win: [nW, N, C]  (N = ws*ws)
        attn_mask: [nW, N, N] or None
        """
        nW, N, C = x_win.shape
        H = self.num_heads
        head_dim = C // H
        scale = 1.0 / math.sqrt(head_dim)

        # qkv
        qkv = jax.vmap(jax.vmap(self.qkv))(x_win)  # [nW, N, 3C]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each [nW, N, C]

        # reshape to heads
        def to_heads(t):
            t = t.reshape(nW, N, H, head_dim).transpose(0, 2, 1, 3)  # [nW, H, N, hd]
            return t

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        attn = jnp.einsum("whnd,whmd->whnm", q * scale, k)  # [nW, H, N, N]

        # relative position bias
        idx = self.relative_position_index.reshape(-1)  # [N*N]
        bias = self.relative_position_bias_table[idx]  # [N*N, H]
        bias = bias.reshape(N, N, H).transpose(2, 0, 1)  # [H, N, N]
        attn = attn + bias[None, :, :, :]  # broadcast over nW

        if attn_mask is not None:
            # add mask (broadcast to heads)
            attn = attn + attn_mask[:, None, :, :]  # [nW,H,N,N]

        attn = jax.nn.softmax(attn, axis=-1)
        k1, k2 = jr.split(key, 2)
        attn = self.attn_drop(attn, key=k1)

        out = jnp.einsum("whnm,whmd->whnd", attn, v)  # [nW,H,N,hd]
        out = out.transpose(0, 2, 1, 3).reshape(nW, N, C)
        out = jax.vmap(self.proj)(out)
        out = self.proj_drop(out, key=k2)
        return out


class SwinTransformerBlock(eqx.Module):
    dim: int = eqx.field(static=True)
    input_resolution: Tuple[int, int] | None = eqx.field(static=True, default=None)
    num_heads: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    shift_size: int = eqx.field(static=True)
    mlp_ratio: float = eqx.field(static=True)

    norm1: LayerNorm2d
    attn: WindowAttention
    drop_path: DropPath
    norm2: LayerNorm2d
    mlp: MLP

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        sd_prob: float,
        *,
        key,
    ):
        k_attn, k_mlp = jr.split(key, 2)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = LayerNorm2d(dim, eps=1e-5)
        self.attn = WindowAttention(
            dim, window_size, num_heads, attn_drop=0.0, proj_drop=0.0, key=k_attn
        )
        self.drop_path = DropPath(sd_prob)
        self.norm2 = LayerNorm2d(dim, eps=1e-5)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=0.0, key=k_mlp)

    def __call__(self, x_chw: jnp.ndarray, key: jnp.ndarray, state):
        """
        x_chw: [C,H,W]
        """
        C, H, W = x_chw.shape
        ws = self.window_size
        shift = self.shift_size

        # Norm1
        x_norm, state = self.norm1(x_chw, state=state)

        # Cyclic shift
        if shift > 0:
            x_shift = jnp.roll(x_norm, shift=(-shift, -shift), axis=(1, 2))
        else:
            x_shift = x_norm

        # Pad to multiples of ws
        x_shift, (pad_h, pad_w) = _pad_to_window_size(x_shift, ws)
        Hp, Wp = x_shift.shape[-2], x_shift.shape[-1]

        # Partition windows
        x_hwc = jnp.moveaxis(x_shift, 0, -1)  # [H,W,C]
        x_windows = _window_partition(x_hwc, ws)  # [nW, ws*ws, C]

        # Attention mask for SW-MSA
        attn_mask = _compute_attn_mask(Hp, Wp, ws, shift) if shift > 0 else None

        # Attention
        out = self.attn(x_windows, attn_mask, key=key)  # [nW, ws*ws, C]

        # Merge windows
        x_hwc = _window_unpartition(out, ws, Hp, Wp)  # [H',W',C]
        x_merge = jnp.moveaxis(x_hwc, -1, 0)  # [C,H',W']

        # Unpad
        if pad_h or pad_w:
            x_merge = x_merge[:, :H, :W]

        # Reverse cyclic shift
        if shift > 0:
            x_merge = jnp.roll(x_merge, shift=(shift, shift), axis=(1, 2))

        # Residual + DropPath
        k1, k2 = jr.split(key, 2)
        x = x_chw + self.drop_path(x_merge, key=k1)

        # Norm2 + MLP (token-wise)
        x2, state = self.norm2(x, state=state)
        # tokens [H*W, C]
        x2_tok = einops.rearrange(x2, "c h w -> (h w) c")
        x2_tok = self.mlp(x2_tok, key=k2)
        x2 = einops.rearrange(x2_tok, "(h w) c -> c h w", h=H, w=W)

        x = x + self.drop_path(x2, key=k1)
        return x, state


class PatchEmbed(eqx.Module):
    """Patchify with Conv(patch=4, stride=4) + LayerNorm over channels."""

    proj: eqx.nn.Conv2d
    norm: LayerNorm2d
    patch_size: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int = 4, *, key):
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = eqx.nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            use_bias=True,
            key=key,
        )
        self.norm = LayerNorm2d(embed_dim, eps=1e-5)

    def __call__(self, x, key, state):
        x = self.proj(x, key=key)  # [C', H/4, W/4]
        x, state = self.norm(x, state=state)
        return x, state


class PatchMerging(eqx.Module):
    """Downsample: concat 2x2 neighbors (4*C) -> LN -> Linear to 2*C."""

    in_dim: int = eqx.field(static=True)
    norm: eqx.nn.LayerNorm
    reduction: eqx.nn.Linear

    def __init__(self, in_dim: int, *, key):
        self.in_dim = in_dim
        self.norm = eqx.nn.LayerNorm(4 * in_dim)
        self.reduction = eqx.nn.Linear(4 * in_dim, 2 * in_dim, key=key)

    def __call__(self, x_chw, key, state):
        C, H, W = x_chw.shape
        # pad if odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x_chw = jnp.pad(x_chw, [(0, 0), (0, pad_h), (0, pad_w)])
            H += pad_h
            W += pad_w

        x = jnp.moveaxis(x_chw, 0, -1)  # [H,W,C]
        x0 = x[0::2, 0::2, :]
        x1 = x[1::2, 0::2, :]
        x2 = x[0::2, 1::2, :]
        x3 = x[1::2, 1::2, :]
        x_cat = jnp.concatenate([x0, x1, x2, x3], axis=-1)  # [H/2,W/2,4C]
        x_tok = x_cat.reshape(-1, 4 * C)  # [N, 4C]
        x_tok = jax.vmap(self.norm)(x_tok)  # LN token-wise
        x_tok = jax.vmap(self.reduction)(x_tok)  # [N, 2C]
        x_out = x_tok.reshape(H // 2, W // 2, 2 * C)
        x_out = jnp.moveaxis(x_out, -1, 0)  # [2C, H/2, W/2]
        return x_out, state


class BasicLayer(eqx.Module):
    """A stage: multiple Swin blocks; optional PatchMerging downsample."""

    blocks: Tuple[SwinTransformerBlock, ...]
    downsample: PatchMerging | None

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        sd_probs: List[float],
        *,
        key,
    ):
        keys = iter(jr.split(key, depth + 1))
        blocks: List[SwinTransformerBlock] = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    sd_prob=sd_probs[i],
                    key=next(keys),
                )
            )
        self.blocks = tuple(blocks)
        # downsample at stage end except the last stage (handled by caller via dim=None sentinel)
        self.downsample = None  # set by caller when wiring stages

    def __call__(self, x, key, state):
        k_run = key

        def split():
            nonlocal k_run
            k1, k_run = jr.split(k_run)
            return k1

        for blk in self.blocks:
            x, state = blk(x, key=split(), state=state)
        if self.downsample is not None:
            x, state = self.downsample(x, key=split(), state=state)
        return x, state


# --------------------------- architecture configs --------------------------- #
_VARIANTS: Dict[str, Dict[str, Any]] = {
    # embed_dim, depths, num_heads, window_size, drop_path
    "swin_t": dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path=0.2,
    ),
    "swin_s": dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path=0.3,
    ),
    "swin_b": dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path=0.5,
    ),
}
_MLP_RATIO = 4.0


# ------------------------------- Swin model ------------------------------- #
class SwinTransformer(eqx.Module):
    features: Tuple[
        Any, ...
    ]  # (PatchEmbed, BasicLayer, BasicLayer, BasicLayer, BasicLayer)
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    arch: str = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)

    def __init__(self, *, arch: str = "swin_t", num_classes: int = 1000, key):
        if arch not in _VARIANTS:
            raise ValueError(f"Unknown arch '{arch}'.")
        cfg = _VARIANTS[arch]
        embed_dim = cfg["embed_dim"]
        depths: List[int] = cfg["depths"]
        num_heads: List[int] = cfg["num_heads"]
        ws = cfg["window_size"]
        drop_path = cfg["drop_path"]

        # Keys
        big_keys = list(jr.split(key, 4096))
        k_it = iter(big_keys)

        feats: List[Any] = []
        # Patch embed
        feats.append(PatchEmbed(3, embed_dim, patch_size=4, key=next(k_it)))

        # Total blocks for sd schedule
        total_blocks = sum(depths)
        bid = 0
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        # Stages
        stages: List[BasicLayer] = []
        for stage_idx, depth in enumerate(depths):
            dim = dims[stage_idx]
            heads = num_heads[stage_idx]
            # per-block sd probs (linear ramp across entire network)
            sd_probs = [
                0.0 if total_blocks <= 1 else drop_path * (bid / (total_blocks - 1.0))
                for _ in range(depth)
            ]
            bid += depth
            layer = BasicLayer(
                dim, depth, heads, ws, _MLP_RATIO, sd_probs, key=next(k_it)
            )
            stages.append(layer)
        # Attach downsamples between stages (except after the last)
        for i in range(len(stages) - 1):
            stages[i] = eqx.tree_at(
                lambda l: l.downsample, stages[i], PatchMerging(dims[i], key=next(k_it))
            )
        feats.extend(stages)

        self.features = tuple(feats)
        last_dim = dims[-1]
        self.norm = eqx.nn.LayerNorm(last_dim)
        self.head = eqx.nn.Linear(last_dim, num_classes, key=next(k_it))

        self.arch = arch
        self.num_classes = num_classes

    def _check_input(self, x: jnp.ndarray):
        if x.ndim != 3:
            raise ValueError(
                f"SwinTransformer expects single sample [C,H,W]; got {tuple(x.shape)}."
            )
        if x.shape[0] != 3:
            raise ValueError(f"Expected 3 input channels; got {x.shape[0]}.")

    def __call__(self, x, key, state):
        self._check_input(x)
        k_run = key

        def split():
            nonlocal k_run
            k1, k_run = jr.split(k_run)
            return k1

        # features
        for i, feat in enumerate(self.features):
            x, state = feat(x, key=split(), state=state)

        # global pooling + norm + head
        C, H, W = x.shape
        x_tok = einops.rearrange(x, "c h w -> (h w) c")
        x_tok = jax.vmap(self.norm)(x_tok)  # LN token-wise as in TV
        x_vec = jnp.mean(x_tok, axis=0)  # [C]
        logits = self.head(x_vec)
        return logits, state


# ---------------------------- Weight Loading ---------------------------- #
def _rename_pt_key(k: str) -> str | None:
    """
    Map torchvision Swin state_dict keys to our module tree.
    We mirror TV's structure so most keys pass through unchanged.
    We only remap MLP linears from sequential indices to names.
      - *.mlp.0.(weight|bias) -> *.mlp.fc1.(weight|bias)
      - *.mlp.3.(weight|bias) -> *.mlp.fc2.(weight|bias)
    Drop non-parameter/buffer-ish keys if any appear.
    """
    if not (
        k.endswith(".weight")
        or k.endswith(".bias")
        or k.endswith("relative_position_bias_table")
    ):
        return None

    # MLP remap if TV uses Sequential indices
    if ".mlp.0." in k:
        return k.replace(".mlp.0.", ".mlp.fc1.")
    if ".mlp.3." in k:
        return k.replace(".mlp.3.", ".mlp.fc2.")

    return k


def _copy_into_tree(obj, pt: Dict[str, jnp.ndarray], prefix: str = ""):
    """Recursively copy Linear/Conv/LayerNorm (and ndarray leaves) into an Equinox pytree."""
    if isinstance(obj, eqx.Module):
        for name, attr in vars(obj).items():
            full = f"{prefix}{name}"

            # Conv2d
            if isinstance(attr, eqx.nn.Conv2d):
                new_attr = attr
                if f"{full}.weight" in pt:
                    new_attr = eqx.tree_at(
                        lambda m: m.weight, new_attr, pt[f"{full}.weight"]
                    )
                if f"{full}.bias" in pt:
                    new_attr = eqx.tree_at(
                        lambda m: m.bias, new_attr, pt[f"{full}.bias"]
                    )
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, new_attr)
                continue

            # Linear
            if isinstance(attr, eqx.nn.Linear):
                new_attr = attr
                if f"{full}.weight" in pt:
                    new_attr = eqx.tree_at(
                        lambda m: m.weight, new_attr, pt[f"{full}.weight"]
                    )
                if f"{full}.bias" in pt:
                    new_attr = eqx.tree_at(
                        lambda m: m.bias, new_attr, pt[f"{full}.bias"]
                    )
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, new_attr)
                continue

            # LayerNorm2d or eqx.nn.LayerNorm
            if isinstance(attr, LayerNorm2d):
                w_key, b_key = f"{full}.weight", f"{full}.bias"
                w_val = pt.get(w_key, getattr(attr, "weight"))
                b_val = pt.get(b_key, getattr(attr, "bias"))
                obj = eqx.tree_at(
                    lambda m: (getattr(m, name).weight, getattr(m, name).bias),
                    obj,
                    (w_val, b_val),
                )
                continue
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

            # Tuples (features, blocks)
            if isinstance(attr, tuple):
                new_tuple = []
                for i, child in enumerate(attr):
                    child_full = f"{full}.{i}"
                    new_tuple.append(_copy_into_tree(child, pt, prefix=child_full))
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, tuple(new_tuple))
                continue

            # Relative position bias table (ndarray param)
            if isinstance(attr, jnp.ndarray) and f"{full}" in pt:
                obj = eqx.tree_at(
                    lambda m: getattr(m, name), obj, jnp.asarray(pt[f"{full}"])
                )
                continue

            # Stateless (DropPath), computed indices/masks -> skip
        return obj

    if isinstance(obj, tuple):
        return tuple(_copy_into_tree(x, pt, prefix=prefix) for x in obj)

    return obj


def load_torchvision_swin(
    model: SwinTransformer, npz_path: str, *, strict_fc: bool = True
) -> SwinTransformer:
    """Load a torchvision Swin .npz (from state_dict()) into this model."""
    import numpy as np

    raw = dict(np.load(npz_path))
    pt: Dict[str, jnp.ndarray] = {}
    for k, v in raw.items():
        nk = _rename_pt_key(k)
        if nk is None:
            continue
        pt[nk] = jnp.asarray(v)

    # Handle final head shape mismatch
    if "head.weight" in pt and "head.bias" in pt:
        want_out, want_in = model.head.weight.shape
        have_out, have_in = pt["head.weight"].shape
        if (want_out != have_out) or (want_in != have_in):
            if strict_fc:
                raise ValueError(
                    f"FC shape mismatch: want {(want_out, want_in)} vs have {(have_out, have_in)}. "
                    f"Set strict_fc=False to skip loading the head."
                )
            pt.pop("head.weight", None)
            pt.pop("head.bias", None)

    return _copy_into_tree(model, pt, prefix="")


# Convenience wrappers
def load_imagenet_swin_t(
    model: SwinTransformer, npz="swin_t_imagenet.npz", strict_fc: bool = True
) -> SwinTransformer:
    return load_torchvision_swin(model, npz, strict_fc=strict_fc)


def load_imagenet_swin_s(
    model: SwinTransformer, npz="swin_s_imagenet.npz", strict_fc: bool = True
) -> SwinTransformer:
    return load_torchvision_swin(model, npz, strict_fc=strict_fc)


def load_imagenet_swin_b(
    model: SwinTransformer, npz="swin_b_imagenet.npz", strict_fc: bool = True
) -> SwinTransformer:
    return load_torchvision_swin(model, npz, strict_fc=strict_fc)


# ------------------------------ Smoke test ------------------------------ #
if __name__ == "__main__":
    """
    Synthetic classification smoke test for Swin Transformer (Tiny).
    Uses 224×224 RGB; replace with real data in experiments.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import augmax
    from augmax import InputType
    import optax
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jr

    # Your training utilities
    from quantbayes.stochax import (
        train,
        predict,
        make_augmax_augment,
        multiclass_loss,
    )

    rng = np.random.RandomState(0)
    N, C, H, W, NUM_CLASSES = 384, 3, 224, 224, 10
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

    model, state = eqx.nn.make_with_state(SwinTransformer)(
        arch="swin_t",
        num_classes=NUM_CLASSES,
        key=model_key,
    )
    # Optional pretrained load (skips head if shapes mismatch)
    # model = load_imagenet_swin_t(model, "swin_t_imagenet.npz", strict_fc=False)

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
    plt.title("Swin-Tiny smoke test")
    plt.show()
