"""RFFT-Swin: only square D→D attention output projections become RFFTCirculant1D.

This mirrors the successful RFFT-ViT idea more conservatively than the older layer-replacement
approach: qkv stays dense because it is D→3D, the MLP stays dense because it is D→4D→D,
and patch merging stays dense because it is 4D→2D. The only square projection in standard
Swin is the attention output projection, so this variant is intentionally lightweight.
"""

from __future__ import annotations

from typing import Any, List, Tuple
import math

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.vision_classification.models.swin import (
    DropPath,
    LayerNorm2d,
    MLP,
    PatchEmbed,
    PatchMerging,
    _VARIANTS,
    _MLP_RATIO,
    _window_partition,
    _window_unpartition,
    _pad_to_window_size,
    _compute_attn_mask,
)

try:
    from quantbayes.stochax.layers import RFFTCirculant1D
except Exception:
    try:
        from quantbayes.stochax.layers.spectral_layers import RFFTCirculant1D
    except (
        Exception
    ) as e:  # pragma: no cover - only hit when spectral layers are absent
        RFFTCirculant1D = None
        _RFFT_IMPORT_ERROR = e


def make_linear_or_spectral(
    in_features: int,
    out_features: int,
    *,
    use_spectral: bool,
    key,
) -> eqx.Module:
    if use_spectral and (in_features == out_features):
        if RFFTCirculant1D is None:
            raise RuntimeError(
                "RFFTCirculant1D is unavailable. Make sure "
                "quantbayes.stochax.layers.spectral_layers is on the PYTHONPATH."
            )
        return RFFTCirculant1D(
            in_features=in_features,
            padded_dim=out_features,
            key=key,
        )
    return eqx.nn.Linear(in_features, out_features, key=key)


class WindowAttention(eqx.Module):
    dim: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    qkv: eqx.nn.Linear
    proj: eqx.Module
    attn_drop: eqx.nn.Dropout
    proj_drop: eqx.nn.Dropout
    relative_position_bias_table: jnp.ndarray
    relative_position_index: jnp.ndarray

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        *,
        key,
        use_spectral_proj: bool = False,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv = eqx.nn.Linear(dim, 3 * dim, key=k1)
        self.proj = make_linear_or_spectral(
            dim, dim, use_spectral=use_spectral_proj, key=k2
        )
        self.attn_drop = eqx.nn.Dropout(attn_drop)
        self.proj_drop = eqx.nn.Dropout(proj_drop)

        ws = window_size
        self.relative_position_bias_table = jr.normal(
            k3, ((2 * ws - 1) * (2 * ws - 1), num_heads)
        )

        coords_h = jnp.arange(ws)
        coords_w = jnp.arange(ws)
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)
        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rel_coords = rel_coords.transpose(1, 2, 0)
        rel_coords = rel_coords + jnp.array([ws - 1, ws - 1])
        rel_coords = rel_coords.at[:, :, 0].multiply(2 * ws - 1)
        self.relative_position_index = rel_coords.sum(-1)

    def __call__(
        self, x_win: jnp.ndarray, attn_mask: jnp.ndarray | None, key: jnp.ndarray
    ):
        nW, N, C = x_win.shape
        H = self.num_heads
        head_dim = C // H
        scale = 1.0 / math.sqrt(head_dim)

        qkv = jax.vmap(jax.vmap(self.qkv))(x_win)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def to_heads(t):
            return t.reshape(nW, N, H, head_dim).transpose(0, 2, 1, 3)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        attn = jnp.einsum("whnd,whmd->whnm", q * scale, k)

        idx = self.relative_position_index.reshape(-1)
        bias = self.relative_position_bias_table[idx]
        bias = bias.reshape(N, N, H).transpose(2, 0, 1)
        attn = attn + bias[None, :, :, :]

        if attn_mask is not None:
            attn = attn + attn_mask[:, None, :, :]

        attn = jax.nn.softmax(attn, axis=-1)
        k1, k2 = jr.split(key, 2)
        attn = self.attn_drop(attn, key=k1)

        out = jnp.einsum("whnm,whmd->whnd", attn, v)
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
        use_spectral_proj: bool = False,
    ):
        k_attn, k_mlp = jr.split(key, 2)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = LayerNorm2d(dim, eps=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size,
            num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            key=k_attn,
            use_spectral_proj=use_spectral_proj,
        )
        self.drop_path = DropPath(sd_prob)
        self.norm2 = LayerNorm2d(dim, eps=1e-5)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=0.0, key=k_mlp)

    def __call__(self, x_chw: jnp.ndarray, key: jnp.ndarray, state):
        C, H, W = x_chw.shape
        ws = self.window_size
        shift = self.shift_size

        x_norm, state = self.norm1(x_chw, state=state)
        x_shift = (
            jnp.roll(x_norm, shift=(-shift, -shift), axis=(1, 2))
            if shift > 0
            else x_norm
        )

        x_shift, (pad_h, pad_w) = _pad_to_window_size(x_shift, ws)
        Hp, Wp = x_shift.shape[-2], x_shift.shape[-1]

        x_hwc = jnp.moveaxis(x_shift, 0, -1)
        x_windows = _window_partition(x_hwc, ws)
        attn_mask = _compute_attn_mask(Hp, Wp, ws, shift) if shift > 0 else None
        out = self.attn(x_windows, attn_mask, key=key)

        x_hwc = _window_unpartition(out, ws, Hp, Wp)
        x_merge = jnp.moveaxis(x_hwc, -1, 0)
        if pad_h or pad_w:
            x_merge = x_merge[:, :H, :W]
        if shift > 0:
            x_merge = jnp.roll(x_merge, shift=(shift, shift), axis=(1, 2))

        k1, k2 = jr.split(key, 2)
        x = x_chw + self.drop_path(x_merge, key=k1)

        x2, state = self.norm2(x, state=state)
        x2_tok = einops.rearrange(x2, "c h w -> (h w) c")
        x2_tok = self.mlp(x2_tok, key=k2)
        x2 = einops.rearrange(x2_tok, "(h w) c -> c h w", h=H, w=W)

        x = x + self.drop_path(x2, key=k1)
        return x, state


class BasicLayer(eqx.Module):
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
        use_spectral_proj: bool = False,
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
                    use_spectral_proj=use_spectral_proj,
                )
            )
        self.blocks = tuple(blocks)
        self.downsample = None

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


class RFFTSwinTransformer(eqx.Module):
    features: Tuple[Any, ...]
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    arch: str = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    use_spectral_proj: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        arch: str = "swin_t",
        num_classes: int = 1000,
        key,
        use_spectral_proj: bool = True,
    ):
        if arch not in _VARIANTS:
            raise ValueError(f"Unknown arch {arch!r}.")
        cfg = _VARIANTS[arch]
        embed_dim = cfg["embed_dim"]
        depths: List[int] = cfg["depths"]
        num_heads: List[int] = cfg["num_heads"]
        ws = cfg["window_size"]
        drop_path = cfg["drop_path"]

        big_keys = list(jr.split(key, 4096))
        k_it = iter(big_keys)

        feats: List[Any] = [PatchEmbed(3, embed_dim, patch_size=4, key=next(k_it))]
        total_blocks = sum(depths)
        bid = 0
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        stages: List[BasicLayer] = []
        for stage_idx, depth in enumerate(depths):
            dim = dims[stage_idx]
            heads = num_heads[stage_idx]
            sd_probs = [
                0.0 if total_blocks <= 1 else drop_path * (bid / (total_blocks - 1.0))
                for _ in range(depth)
            ]
            bid += depth
            layer = BasicLayer(
                dim,
                depth,
                heads,
                ws,
                _MLP_RATIO,
                sd_probs,
                key=next(k_it),
                use_spectral_proj=use_spectral_proj,
            )
            stages.append(layer)
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
        self.use_spectral_proj = bool(use_spectral_proj)

    def _check_input(self, x: jnp.ndarray):
        if x.ndim != 3:
            raise ValueError(
                f"RFFTSwinTransformer expects single sample [C,H,W]; got {tuple(x.shape)}."
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

        for feat in self.features:
            x, state = feat(x, key=split(), state=state)

        C, H, W = x.shape
        x_tok = einops.rearrange(x, "c h w -> (h w) c")
        x_tok = jax.vmap(self.norm)(x_tok)
        x_vec = jnp.mean(x_tok, axis=0)
        logits = self.head(x_vec)
        return logits, state


__all__ = ["RFFTSwinTransformer"]
