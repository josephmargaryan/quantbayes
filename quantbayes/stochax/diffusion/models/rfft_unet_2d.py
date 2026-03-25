# quantbayes/stochax/diffusion/models/rfft_unet_2d.py
from __future__ import annotations
from typing import Iterable, Optional, Sequence, Set

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# ---- pull in your spectral layers ----
from quantbayes.stochax.layers.spectral_layers import (
    RFFTCirculant2D,
    SpectralConv2d,
)


# ---------------- Time embedding (same shape as your UNet) ---------------- #
class SinusoidalPosEmb(eqx.Module):
    emb: jax.Array

    def __init__(self, dim: int):
        half_dim = dim // 2
        emb = jnp.log(10000.0) / jnp.maximum(half_dim - 1, 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is scalar (log sigma or VP time)
        x = x * self.emb
        return jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)


# ---------- scheduled RFFT conv (no mutation; mask built per-call) -------- #
class RFFTConv2dScheduled(eqx.Module):
    """Wraps a fixed RFFTCirculant2D but builds a soft radial mask m(t) on-the-fly.

    K_rad(t) = kmin + (kmax-kmin) * sigmoid(-a*(t-b))   (t is log sigma by default)
    Large t (big noise) -> smaller passband; small t -> larger passband.
    """

    base: RFFTCirculant2D
    kmin: float
    kmax: float
    a: float
    b: float
    time_is_log_sigma: bool
    use_soft_mask: bool
    mask_steepness: float

    def __init__(
        self,
        base: RFFTCirculant2D,
        *,
        kmin: float = 0.15,
        kmax: float = 1.0,
        a: float = 0.5,
        b: float = -1.0,
        time_is_log_sigma: bool = True,
        use_soft_mask: bool = True,
        mask_steepness: float = 20.0,
    ):
        self.base = base
        self.kmin = float(kmin)
        self.kmax = float(kmax)
        self.a = float(a)
        self.b = float(b)
        self.time_is_log_sigma = bool(time_is_log_sigma)
        self.use_soft_mask = bool(use_soft_mask)
        self.mask_steepness = float(mask_steepness)

    def _k_rad(self, t: jax.Array) -> jax.Array:
        # t is scalar (log sigma if EDM; otherwise VP time — still monotone)
        # We keep the same schedule either way; works fine in practice.
        s = jax.nn.sigmoid(-self.a * (t - self.b))
        return self.kmin + (self.kmax - self.kmin) * s

    def _radial_grid(self, H_pad: int, W_pad: int, dtype=jnp.float32):
        u = jnp.fft.fftfreq(H_pad).astype(dtype) * H_pad
        v = jnp.fft.rfftfreq(W_pad).astype(dtype) * W_pad
        U, V = jnp.meshgrid(u, v, indexing="ij")
        R = jnp.sqrt(U**2 + V**2)
        Rn = R / jnp.maximum(jnp.max(R), 1.0)
        return Rn  # (H_pad, W_half)

    def _mask(self, t: jax.Array) -> jax.Array:
        k = self._k_rad(t)
        Rn = self._radial_grid(
            self.base.H_pad, self.base.W_pad, dtype=self.base.K_half.real.dtype
        )
        if self.use_soft_mask:
            m = jax.nn.sigmoid(self.mask_steepness * (k - Rn))
        else:
            m = (Rn <= k).astype(Rn.dtype)
        return m  # (H_pad, W_half)

    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        """x: (C,H,W) or (B,C,H,W)"""
        single = x.ndim == 3
        if single:
            x = x[None, ...]
        # Pad/crop like the base layer
        pad_h = max(0, self.base.H_pad - x.shape[-2])
        pad_w = max(0, self.base.W_pad - x.shape[-1])
        x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))[
            ..., : self.base.H_pad, : self.base.W_pad
        ]

        Xf = jnp.fft.rfft2(x_pad, axes=(-2, -1), norm="ortho")  # (B,Cin,H_pad,W_half)
        # scheduled mask
        m = self._mask(t)  # (H_pad, W_half)
        # clean half-plane like base
        K = self.base.K_half

        # enforce self-conjugate bins real
        def clean_half(Kh):
            Kh = Kh.at[..., 0, 0].set(jnp.real(Kh[..., 0, 0]) + 0.0j)
            if self.base.H_pad % 2 == 0:
                Kh = Kh.at[..., self.base.H_pad // 2, 0].set(
                    jnp.real(Kh[..., self.base.H_pad // 2, 0]) + 0.0j
                )
            if self.base.W_pad % 2 == 0:
                Kh = Kh.at[..., 0, self.base.W_half - 1].set(
                    jnp.real(Kh[..., 0, self.base.W_half - 1]) + 0.0j
                )
                if self.base.H_pad % 2 == 0:
                    Kh = Kh.at[..., self.base.H_pad // 2, self.base.W_half - 1].set(
                        jnp.real(Kh[..., self.base.H_pad // 2, self.base.W_half - 1])
                        + 0.0j
                    )
            return Kh

        Kh = clean_half(K) * m[None, None, :, :]  # (Cout,Cin,H_pad,W_half)
        Yf = jnp.einsum("oihw,bihw->bohw", Kh, Xf)
        y = jnp.fft.irfft2(
            Yf, s=(self.base.H_pad, self.base.W_pad), axes=(-2, -1), norm="ortho"
        ).real
        y = y + self.base.bias[None, :, None, None]

        if self.base.crop_output and (
            self.base.H_pad != self.base.H_in or self.base.W_pad != self.base.W_in
        ):
            y = y[..., : self.base.H_in, : self.base.W_in]
        return y[0] if single else y


# ------------------------- UNet building blocks --------------------------- #
def upsample_2d(y: jax.Array, factor: int = 2) -> jax.Array:
    C, H, W = y.shape
    y = jnp.reshape(y, [C, H, 1, W, 1])
    y = jnp.tile(y, [1, 1, factor, 1, factor])
    return jnp.reshape(y, [C, H * factor, W * factor])


class ResnetBlock(eqx.Module):
    dim_in: int
    dim_out: int
    time_mlp: eqx.nn.MLP
    gn1: eqx.nn.GroupNorm
    gn2: eqx.nn.GroupNorm
    conv1_rfft: Optional[RFFTConv2dScheduled]
    conv1_svd: Optional[SpectralConv2d]
    conv2_svd: SpectralConv2d
    res_conv: Optional[eqx.nn.Conv2d]

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        time_dim: int,
        *,
        use_rfft_first: bool,
        img_h: int,
        img_w: int,
        kernel_size: int = 3,
        key,
    ):
        k1, k2, k3, k4, k5 = jr.split(key, 5)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.time_mlp = eqx.nn.MLP(time_dim, dim_out, 4 * dim_out, depth=1, key=k1)
        self.gn1 = eqx.nn.GroupNorm(min(dim_in // 4, 32) if dim_in >= 4 else 1, dim_in)
        self.gn2 = eqx.nn.GroupNorm(
            min(dim_out // 4, 32) if dim_out >= 4 else 1, dim_out
        )

        if use_rfft_first:
            base = RFFTCirculant2D(
                C_in=dim_in,
                C_out=dim_out,
                H_in=img_h,
                W_in=img_w,
                H_pad=img_h,
                W_pad=img_w,
                alpha_init=1.0,
                crop_output=True,
                use_soft_mask=True,
                mask_steepness=20.0,
                key=k2,
            )
            self.conv1_rfft = RFFTConv2dScheduled(
                base,
                kmin=0.15,
                kmax=1.0,
                a=0.5,
                b=-1.0,
                time_is_log_sigma=True,
                use_soft_mask=True,
                mask_steepness=20.0,
            )
            self.conv1_svd = None
        else:
            self.conv1_rfft = None
            self.conv1_svd = SpectralConv2d(
                C_in=dim_in,
                C_out=dim_out,
                H_k=kernel_size,
                W_k=kernel_size,
                padding="SAME",
                key=k2,
            )

        self.conv2_svd = SpectralConv2d(
            C_in=dim_out,
            C_out=dim_out,
            H_k=kernel_size,
            W_k=kernel_size,
            padding="SAME",
            key=k3,
        )
        self.res_conv = (
            eqx.nn.Conv2d(dim_in, dim_out, kernel_size=1, key=k4)
            if (dim_in != dim_out)
            else None
        )

    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        # x: (C,H,W)
        h = jax.nn.silu(self.gn1(x))
        if self.conv1_rfft is not None:
            h = self.conv1_rfft(t, h)
        else:
            h = self.conv1_svd(h)  # type: ignore

        # FiLM from time
        temb = self.time_mlp(SinusoidalPosEmb(self.time_mlp.in_size)(t))
        h = h + temb[:, None, None]

        h = jax.nn.silu(self.gn2(h))
        h = self.conv2_svd(h)

        if self.res_conv is not None:
            x = self.res_conv(x)
        return (h + x) / jnp.sqrt(2.0)


# ----------------------------- The UNet ----------------------------------- #
class RFFTSpectralUNet2d(eqx.Module):
    """Hybrid UNet with scheduled RFFTCirculant2D in early convs.

    Args:
      data_shape: (C,H,W).
      dim_mults:  channel multipliers per level.
      hidden_size: base channels.
      num_res_blocks: blocks per level.
      rfft_levels: indices in {0..L-1} where the *first* block uses RFFT.
      time_is_log_sigma: interpret t as log sigma (EDM).
    """

    # stem
    stem: eqx.nn.Conv2d

    # time
    time_pos: SinusoidalPosEmb
    time_mlp: eqx.nn.MLP

    # down / mid / up
    downs: list[list[ResnetBlock]]
    downs_downsample: list[eqx.nn.Conv2d]
    mid1: ResnetBlock
    mid2: ResnetBlock
    ups: list[list[ResnetBlock]]
    ups_upsample: list[eqx.nn.ConvTranspose2d]

    # head
    head_norm: eqx.nn.GroupNorm
    head_conv: eqx.nn.Conv2d

    # meta
    levels: int
    time_is_log_sigma: bool
    img_hs: list[int]
    img_ws: list[int]

    def __init__(
        self,
        data_shape: tuple[int, int, int],
        *,
        dim_mults: Sequence[int] = (1, 2, 4),
        hidden_size: int = 64,
        num_res_blocks: int = 2,
        rfft_levels: Optional[Iterable[int]] = (0, 1),
        time_is_log_sigma: bool = True,
        key,
    ):
        C, H, W = data_shape
        self.levels = len(dim_mults)
        self.time_is_log_sigma = bool(time_is_log_sigma)

        k_all = jr.split(key, 5 + 2 * self.levels + self.levels * num_res_blocks * 2)
        k = iter(k_all)

        # stem
        self.stem = eqx.nn.Conv2d(C, hidden_size, kernel_size=3, padding=1, key=next(k))

        # time
        self.time_pos = SinusoidalPosEmb(hidden_size)
        self.time_mlp = eqx.nn.MLP(
            hidden_size, hidden_size, 4 * hidden_size, depth=1, key=next(k)
        )

        # precompute per-level H,W
        img_hs = [H]
        img_ws = [W]
        for _ in range(self.levels - 1):
            img_hs.append(img_hs[-1] // 2)
            img_ws.append(img_ws[-1] // 2)
        self.img_hs = img_hs
        self.img_ws = img_ws

        # Down path
        self.downs = []
        self.downs_downsample = []
        in_ch = hidden_size
        rfft_set: Set[int] = set(rfft_levels or [])
        for lvl, m in enumerate(dim_mults):
            out_ch = hidden_size * m
            blocks = []
            for b in range(num_res_blocks):
                use_rfft_first = (b == 0) and (lvl in rfft_set)
                blk = ResnetBlock(
                    dim_in=in_ch if b == 0 else out_ch,
                    dim_out=out_ch,
                    time_dim=hidden_size,
                    use_rfft_first=use_rfft_first,
                    img_h=self.img_hs[lvl],
                    img_w=self.img_ws[lvl],
                    key=next(k),
                )
                blocks.append(blk)
            self.downs.append(blocks)

            if lvl < self.levels - 1:
                # stride-2 conv (keep it plain, cheap)
                self.downs_downsample.append(
                    eqx.nn.Conv2d(
                        out_ch, out_ch, kernel_size=3, stride=2, padding=1, key=next(k)
                    )
                )
            in_ch = out_ch

        # Mid blocks (use SVD convs inside)
        self.mid1 = ResnetBlock(
            dim_in=in_ch,
            dim_out=in_ch,
            time_dim=hidden_size,
            use_rfft_first=False,
            img_h=self.img_hs[-1],
            img_w=self.img_ws[-1],
            key=next(k),
        )
        self.mid2 = ResnetBlock(
            dim_in=in_ch,
            dim_out=in_ch,
            time_dim=hidden_size,
            use_rfft_first=False,
            img_h=self.img_hs[-1],
            img_w=self.img_ws[-1],
            key=next(k),
        )

        # Up path
        self.ups = []
        self.ups_upsample = []
        for lvl, m in reversed(list(enumerate(dim_mults))):
            out_ch = hidden_size * m
            blocks = []
            for b in range(num_res_blocks):
                # input channels double because of skip connection concat, except first at top of level
                dim_in_blk = in_ch + out_ch if b == 0 else in_ch
                blk = ResnetBlock(
                    dim_in=dim_in_blk,
                    dim_out=out_ch,
                    time_dim=hidden_size,
                    use_rfft_first=False,  # keep high-freq detail with spatial SVD conv here
                    img_h=self.img_hs[lvl],
                    img_w=self.img_ws[lvl],
                    key=next(k),
                )
                blocks.append(blk)
                in_ch = out_ch
            self.ups.append(blocks)
            if lvl > 0:
                self.ups_upsample.append(
                    eqx.nn.ConvTranspose2d(
                        out_ch, out_ch, kernel_size=4, stride=2, padding=1, key=next(k)
                    )
                )

        self.head_norm = eqx.nn.GroupNorm(
            min(hidden_size // 4, 32) if hidden_size >= 4 else 1, hidden_size
        )
        self.head_conv = eqx.nn.Conv2d(hidden_size, C, kernel_size=1, key=next(k))

    # ---- forward (single) ----
    def _forward_single(self, t: jax.Array, y: jax.Array) -> jax.Array:
        # t: scalar (log σ by default); y: (C,H,W)
        temb = self.time_mlp(self.time_pos(t))

        h = self.stem(y)
        skips = []

        # Down
        cur = h
        for lvl, blocks in enumerate(self.downs):
            for b, blk in enumerate(blocks):
                cur = blk(t, cur)  # RFFT applies inside when configured
                skips.append(cur)
            if lvl < self.levels - 1:
                cur = self.downs_downsample[lvl](cur)

        # Mid
        cur = self.mid1(t, cur)
        cur = self.mid2(t, cur)

        # Up
        for lvl, blocks in enumerate(self.ups):
            for b, blk in enumerate(blocks):
                # concat skip (reverse order)
                cur = jnp.concatenate([cur, skips.pop()], axis=0) if b == 0 else cur
                cur = blk(t, cur)
            if lvl < len(self.ups_upsample):
                cur = self.ups_upsample[lvl](cur)

        # Head
        cur = self.head_norm(cur)
        cur = jax.nn.silu(cur)
        cur = self.head_conv(cur)
        return cur

    def __call__(
        self, t: jax.Array, y: jax.Array, *, key=None, train: bool | None = None
    ) -> jax.Array:
        # matches your UNet signature; key/train are unused (stateless)
        if y.ndim == 4:  # (B,C,H,W)
            return (
                jax.vmap(lambda ti, yi: self._forward_single(ti, yi))(t, y)
                if t.ndim == 1
                else jax.vmap(lambda yi: self._forward_single(t, yi))(y)
            )
        else:
            return self._forward_single(t, y)
