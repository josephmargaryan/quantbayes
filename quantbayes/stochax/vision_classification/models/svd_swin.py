from __future__ import annotations

from typing import Any, List, Tuple, Literal
import math

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.vision_classification.models.swin import (
    DropPath,
    LayerNorm2d,
    PatchEmbed,
    _VARIANTS,
    _MLP_RATIO,
    _window_partition,
    _window_unpartition,
    _pad_to_window_size,
    _compute_attn_mask,
)

try:
    from quantbayes.stochax.layers import SVDDense
except Exception:
    from quantbayes.stochax.layers.spectral_layers import SVDDense  # type: ignore


SVDMode = Literal["none", "attn_only", "attn_mlp", "all_linear"]


def _resolve_rank(
    in_features: int,
    out_features: int,
    *,
    rank: int | None,
    rank_ratio: float,
    min_rank: int,
    max_rank: int | None,
) -> int:
    full = min(int(in_features), int(out_features))
    if rank is not None:
        r = int(rank)
    else:
        r = max(int(min_rank), int(round(full * float(rank_ratio))))
    if max_rank is not None:
        r = min(r, int(max_rank))
    return max(1, min(full, r))


def make_linear_or_svd(
    in_features: int,
    out_features: int,
    *,
    use_svd: bool,
    key,
    rank: int | None = None,
    rank_ratio: float = 0.25,
    min_rank: int = 16,
    max_rank: int | None = None,
    alpha_init: float = 1.0,
) -> eqx.Module:
    dense = eqx.nn.Linear(in_features, out_features, key=key)
    if not use_svd:
        return dense
    r = _resolve_rank(
        in_features,
        out_features,
        rank=rank,
        rank_ratio=rank_ratio,
        min_rank=min_rank,
        max_rank=max_rank,
    )
    return SVDDense.from_linear(dense, rank=r, alpha_init=alpha_init)


class MLP(eqx.Module):
    fc1: eqx.Module
    fc2: eqx.Module
    drop1: eqx.nn.Dropout
    drop2: eqx.nn.Dropout

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        *,
        key,
        use_svd: bool = False,
        svd_rank: int | None = None,
        svd_rank_ratio: float = 0.25,
        svd_min_rank: int = 16,
        svd_rank_cap: int | None = None,
        alpha_init: float = 1.0,
    ):
        k1, k2 = jr.split(key, 2)
        hidden = int(dim * mlp_ratio)
        self.fc1 = make_linear_or_svd(
            dim,
            hidden,
            use_svd=use_svd,
            key=k1,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.fc2 = make_linear_or_svd(
            hidden,
            dim,
            use_svd=use_svd,
            key=k2,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.drop1 = eqx.nn.Dropout(drop)
        self.drop2 = eqx.nn.Dropout(drop)

    def __call__(self, x_tok: jnp.ndarray, key: jnp.ndarray):
        k1, k2 = jr.split(key, 2)
        x = jax.vmap(self.fc1)(x_tok)
        x = jax.nn.gelu(x)
        x = self.drop1(x, key=k1)
        x = jax.vmap(self.fc2)(x)
        x = self.drop2(x, key=k2)
        return x


class WindowAttention(eqx.Module):
    dim: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    qkv: eqx.Module
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
        use_svd_qkv: bool = False,
        use_svd_proj: bool = False,
        svd_rank: int | None = None,
        svd_rank_ratio: float = 0.25,
        svd_min_rank: int = 16,
        svd_rank_cap: int | None = None,
        alpha_init: float = 1.0,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        self.qkv = make_linear_or_svd(
            dim,
            3 * dim,
            use_svd=use_svd_qkv,
            key=k1,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.proj = make_linear_or_svd(
            dim,
            dim,
            use_svd=use_svd_proj,
            key=k2,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.attn_drop = eqx.nn.Dropout(attn_drop)
        self.proj_drop = eqx.nn.Dropout(proj_drop)

        ws = self.window_size
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


class PatchMerging(eqx.Module):
    in_dim: int = eqx.field(static=True)
    norm: eqx.nn.LayerNorm
    reduction: eqx.Module

    def __init__(
        self,
        in_dim: int,
        *,
        key,
        use_svd: bool = False,
        svd_rank: int | None = None,
        svd_rank_ratio: float = 0.25,
        svd_min_rank: int = 16,
        svd_rank_cap: int | None = None,
        alpha_init: float = 1.0,
    ):
        self.in_dim = int(in_dim)
        self.norm = eqx.nn.LayerNorm(4 * in_dim)
        self.reduction = make_linear_or_svd(
            4 * in_dim,
            2 * in_dim,
            use_svd=use_svd,
            key=key,
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )

    def __call__(self, x_chw, key, state):
        C, H, W = x_chw.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x_chw = jnp.pad(x_chw, [(0, 0), (0, pad_h), (0, pad_w)])
            H += pad_h
            W += pad_w

        x = jnp.moveaxis(x_chw, 0, -1)
        x0 = x[0::2, 0::2, :]
        x1 = x[1::2, 0::2, :]
        x2 = x[0::2, 1::2, :]
        x3 = x[1::2, 1::2, :]
        x_cat = jnp.concatenate([x0, x1, x2, x3], axis=-1)
        x_tok = x_cat.reshape(-1, 4 * C)
        x_tok = jax.vmap(self.norm)(x_tok)
        x_tok = jax.vmap(self.reduction)(x_tok)
        x_out = x_tok.reshape(H // 2, W // 2, 2 * C)
        x_out = jnp.moveaxis(x_out, -1, 0)
        return x_out, state


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
        use_svd_qkv: bool = False,
        use_svd_proj: bool = False,
        use_svd_mlp: bool = False,
        svd_rank: int | None = None,
        svd_rank_ratio: float = 0.25,
        svd_min_rank: int = 16,
        svd_rank_cap: int | None = None,
        alpha_init: float = 1.0,
    ):
        k_attn, k_mlp = jr.split(key, 2)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.mlp_ratio = float(mlp_ratio)

        self.norm1 = LayerNorm2d(dim, eps=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size,
            num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            key=k_attn,
            use_svd_qkv=use_svd_qkv,
            use_svd_proj=use_svd_proj,
            svd_rank=svd_rank,
            svd_rank_ratio=svd_rank_ratio,
            svd_min_rank=svd_min_rank,
            svd_rank_cap=svd_rank_cap,
            alpha_init=alpha_init,
        )
        self.drop_path = DropPath(sd_prob)
        self.norm2 = LayerNorm2d(dim, eps=1e-5)
        self.mlp = MLP(
            dim,
            mlp_ratio=mlp_ratio,
            drop=0.0,
            key=k_mlp,
            use_svd=use_svd_mlp,
            svd_rank=svd_rank,
            svd_rank_ratio=svd_rank_ratio,
            svd_min_rank=svd_min_rank,
            svd_rank_cap=svd_rank_cap,
            alpha_init=alpha_init,
        )

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
        use_svd_qkv: bool = False,
        use_svd_proj: bool = False,
        use_svd_mlp: bool = False,
        use_svd_downsample: bool = False,
        svd_rank: int | None = None,
        svd_rank_ratio: float = 0.25,
        svd_min_rank: int = 16,
        svd_rank_cap: int | None = None,
        alpha_init: float = 1.0,
    ):
        keys = iter(jr.split(key, depth + 1))
        self.blocks = tuple(
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0 if (i % 2 == 0) else window_size // 2),
                mlp_ratio=mlp_ratio,
                sd_prob=sd_probs[i],
                key=next(keys),
                use_svd_qkv=use_svd_qkv,
                use_svd_proj=use_svd_proj,
                use_svd_mlp=use_svd_mlp,
                svd_rank=svd_rank,
                svd_rank_ratio=svd_rank_ratio,
                svd_min_rank=svd_min_rank,
                svd_rank_cap=svd_rank_cap,
                alpha_init=alpha_init,
            )
            for i in range(depth)
        )
        self.downsample = None
        self._downsample_config = (
            dim,
            use_svd_downsample,
            svd_rank,
            svd_rank_ratio,
            svd_min_rank,
            svd_rank_cap,
            alpha_init,
        )

    def attach_downsample(self, key):
        (
            dim,
            use_svd_downsample,
            svd_rank,
            svd_rank_ratio,
            svd_min_rank,
            svd_rank_cap,
            alpha_init,
        ) = self._downsample_config
        self.downsample = PatchMerging(
            dim,
            key=key,
            use_svd=use_svd_downsample,
            svd_rank=svd_rank,
            svd_rank_ratio=svd_rank_ratio,
            svd_min_rank=svd_min_rank,
            svd_rank_cap=svd_rank_cap,
            alpha_init=alpha_init,
        )

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


class SVDSwinTransformer(eqx.Module):
    features: Tuple[Any, ...]
    norm: eqx.nn.LayerNorm
    head: eqx.Module

    arch: str = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    svd_mode: SVDMode = eqx.field(static=True)

    def __init__(
        self,
        *,
        arch: str = "swin_t",
        num_classes: int = 1000,
        key,
        svd_mode: SVDMode = "attn_mlp",
        svd_rank: int | None = None,
        svd_rank_ratio: float = 0.25,
        svd_min_rank: int = 16,
        svd_rank_cap: int | None = None,
        alpha_init: float = 1.0,
    ):
        if arch not in _VARIANTS:
            raise ValueError(f"Unknown arch {arch!r}.")
        cfg = _VARIANTS[arch]
        embed_dim = cfg["embed_dim"]
        depths: List[int] = cfg["depths"]
        num_heads: List[int] = cfg["num_heads"]
        ws = cfg["window_size"]
        drop_path = cfg["drop_path"]

        use_svd_qkv = svd_mode in {"attn_only", "attn_mlp", "all_linear"}
        use_svd_proj = svd_mode in {"attn_only", "attn_mlp", "all_linear"}
        use_svd_mlp = svd_mode in {"attn_mlp", "all_linear"}
        use_svd_downsample = svd_mode == "all_linear"
        use_svd_head = svd_mode == "all_linear"

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
                use_svd_qkv=use_svd_qkv,
                use_svd_proj=use_svd_proj,
                use_svd_mlp=use_svd_mlp,
                use_svd_downsample=use_svd_downsample,
                svd_rank=svd_rank,
                svd_rank_ratio=svd_rank_ratio,
                svd_min_rank=svd_min_rank,
                svd_rank_cap=svd_rank_cap,
                alpha_init=alpha_init,
            )
            stages.append(layer)

        for i in range(len(stages) - 1):
            stages[i].attach_downsample(next(k_it))

        feats.extend(stages)
        self.features = tuple(feats)

        last_dim = dims[-1]
        self.norm = eqx.nn.LayerNorm(last_dim)
        self.head = make_linear_or_svd(
            last_dim,
            num_classes,
            use_svd=use_svd_head,
            key=next(k_it),
            rank=svd_rank,
            rank_ratio=svd_rank_ratio,
            min_rank=svd_min_rank,
            max_rank=svd_rank_cap,
            alpha_init=alpha_init,
        )

        self.arch = arch
        self.num_classes = int(num_classes)
        self.svd_mode = svd_mode

    def _check_input(self, x: jnp.ndarray):
        if x.ndim != 3:
            raise ValueError(
                f"SVDSwinTransformer expects [C,H,W]; got {tuple(x.shape)}."
            )
        if x.shape[0] != 3:
            raise ValueError(f"Expected 3 input channels; got {x.shape[0]}.")

    def forward_features(self, x, key, state):
        self._check_input(x)
        k_run = key

        def split():
            nonlocal k_run
            k1, k_run = jr.split(k_run)
            return k1

        for feat in self.features:
            x, state = feat(x, key=split(), state=state)

        x_tok = einops.rearrange(x, "c h w -> (h w) c")
        x_tok = jax.vmap(self.norm)(x_tok)
        feat = jnp.mean(x_tok, axis=0)
        return feat, state

    def forward_head(self, feat: jnp.ndarray) -> jnp.ndarray:
        return self.head(feat)

    def __call__(self, x, key, state):
        feat, state = self.forward_features(x, key, state)
        logits = self.forward_head(feat)
        return logits, state
