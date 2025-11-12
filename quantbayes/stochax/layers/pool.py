import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
from typing import Optional, Tuple


class SpectralPool2d(eqx.Module):
    """
    FFT → centered crop with feathering → IFFT (real).
    If stochastic=True, randomly sample output size in a range at train time.
    """

    out_hw: Tuple[int, int] = eqx.field(static=True)
    feather: float = eqx.field(static=True)  # pixels over which to taper
    stochastic: bool = eqx.field(static=True)
    jitter: int = eqx.field(static=True)  # +/- range for random crop

    def __init__(
        self,
        out_hw: int | Tuple[int, int],
        *,
        feather: float = 2.0,
        stochastic: bool = False,
        jitter: int = 0,
    ):
        if isinstance(out_hw, int):
            out_hw = (out_hw, out_hw)
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        self.feather = float(feather)
        self.stochastic = bool(stochastic)
        self.jitter = int(jitter)

    def __call__(self, x, *, key: Optional[jr.KeyArray] = None):
        # x: (..., H, W)
        H_out, W_out = self.out_hw
        H, W = x.shape[-2], x.shape[-1]

        if self.stochastic and key is not None and self.jitter > 0:
            dh = jr.randint(key, (), -self.jitter, self.jitter + 1)
            dw = jr.randint(key, (), -self.jitter, self.jitter + 1)
            H_out = int(jnp.clip(H_out + dh, 1, H))
            W_out = int(jnp.clip(W_out + dw, 1, W))

        Xf = jnp.fft.fftn(x, axes=(-2, -1), norm="ortho")
        Xf = jnp.fft.fftshift(Xf, axes=(-2, -1))

        h0 = (H - H_out) // 2
        w0 = (W - W_out) // 2
        Xc = Xf[..., h0 : h0 + H_out, w0 : w0 + W_out]

        # feather mask to reduce ringing (cosine taper on the border)
        if self.feather > 0:
            fh = int(min(self.feather, H_out // 2))
            fw = int(min(self.feather, W_out // 2))
            win_h = jnp.ones((H_out,))
            win_w = jnp.ones((W_out,))
            t_h = 0.5 - 0.5 * jnp.cos(jnp.linspace(0, jnp.pi, 2 * fh + 1))
            t_w = 0.5 - 0.5 * jnp.cos(jnp.linspace(0, jnp.pi, 2 * fw + 1))
            win_h = (
                win_h.at[: fh + 1]
                .multiply(t_h[: fh + 1])
                .at[-(fh + 1) :]
                .multiply(t_h[::-1])
            )
            win_w = (
                win_w.at[: fw + 1]
                .multiply(t_w[: fw + 1])
                .at[-(fw + 1) :]
                .multiply(t_w[::-1])
            )
            mask = win_h[:, None] * win_w[None, :]
            Xc = Xc * mask

        Xc = jnp.fft.ifftshift(Xc, axes=(-2, -1))
        x_pooled = jnp.fft.ifftn(Xc, axes=(-2, -1), norm="ortho").real
        return x_pooled
