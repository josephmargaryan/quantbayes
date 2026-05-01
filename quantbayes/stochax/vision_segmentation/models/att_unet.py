"""
Equinox AttentionUNet — RGB → 1-logit. VMAP-friendly (state threaded through).
Attention gates at each skip-connection.
BatchNorm uses mode="batch" (no EMA state, fast).
Default `base=8` to match your train script.
"""

from typing import Tuple, Any
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx


def _match(x: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    """Center-pad / crop spatial dims so `x.shape[-2:] == ref.shape[-2:]`."""
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

    def __call__(self, x: jnp.ndarray, *, key, state: Any) -> Tuple[jnp.ndarray, Any]:
        k1, k2 = jr.split(key, 2)
        x, state = self.bn1(self.c1(x, key=k1), state)
        x = jax.nn.relu(x)
        x, state = self.bn2(self.c2(x, key=k2), state)
        x = jax.nn.relu(x)
        return x, state


class AttentionBlock(eqx.Module):
    """
    Attention gate: takes gating signal g and skip connection x,
    returns attended skip: x * sigmoid(ψ(ReLU(W_g(g) + W_x(x)))).
    """

    W_g: eqx.nn.Conv2d
    W_x: eqx.nn.Conv2d
    psi: eqx.nn.Conv2d

    def __init__(self, F_g: int, F_l: int, F_int: int, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.W_g = eqx.nn.Conv2d(F_g, F_int, 1, key=k1)
        self.W_x = eqx.nn.Conv2d(F_l, F_int, 1, key=k2)
        self.psi = eqx.nn.Conv2d(F_int, 1, 1, key=k3)

    def __call__(self, g: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = jax.nn.relu(g1 + x1)
        psi = self.psi(psi)
        psi = jax.nn.sigmoid(psi)
        return x * psi


class UpAtt(eqx.Module):
    up: eqx.nn.ConvTranspose2d
    att: AttentionBlock
    conv: ConvBlock

    def __init__(self, cin: int, skip: int, cout: int, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.up = eqx.nn.ConvTranspose2d(cin, cout, 2, stride=2, key=k1)
        self.att = AttentionBlock(cout, skip, cout // 2, key=k2)
        self.conv = ConvBlock(cout + skip, cout, key=k3)

    def __call__(
        self, x: jnp.ndarray, skip: jnp.ndarray, *, key, state: Any
    ) -> Tuple[jnp.ndarray, Any]:
        k1, k2 = jr.split(key, 2)
        x_up = self.up(x, key=k1)
        x_up, skip = _match(x_up, skip), _match(skip, x_up)
        skip_att = self.att(x_up, skip)
        x_cat = jnp.concatenate([skip_att, x_up], axis=0)
        x_out, state = self.conv(x_cat, key=k2, state=state)
        return x_out, state


class AttentionUNet(eqx.Module):
    e1: ConvBlock
    e2: ConvBlock
    e3: ConvBlock
    e4: ConvBlock
    b: ConvBlock
    d1: UpAtt
    d2: UpAtt
    d3: UpAtt
    d4: UpAtt
    out: eqx.nn.Conv2d
    pool: eqx.nn.MaxPool2d

    def __init__(self, *, in_ch: int = 3, out_ch: int = 1, base: int = 8, key):
        ks = jr.split(key, 14)
        self.e1 = ConvBlock(in_ch, base, key=ks[0])
        self.e2 = ConvBlock(base, base * 2, key=ks[1])
        self.e3 = ConvBlock(base * 2, base * 4, key=ks[2])
        self.e4 = ConvBlock(base * 4, base * 8, key=ks[3])
        self.b = ConvBlock(base * 8, base * 16, key=ks[4])

        self.d1 = UpAtt(base * 16, base * 8, base * 8, key=ks[5])
        self.d2 = UpAtt(base * 8, base * 4, base * 4, key=ks[6])
        self.d3 = UpAtt(base * 4, base * 2, base * 2, key=ks[7])
        self.d4 = UpAtt(base * 2, base, base, key=ks[8])

        self.out = eqx.nn.Conv2d(base, out_ch, 1, key=ks[9])
        self.pool = eqx.nn.MaxPool2d(2, 2)

    def __call__(self, x: jnp.ndarray, key, state: Any) -> Tuple[jnp.ndarray, Any]:
        ks = jr.split(key, 9)
        e1, state = self.e1(x, key=ks[0], state=state)
        p1 = self.pool(e1)
        e2, state = self.e2(p1, key=ks[1], state=state)
        p2 = self.pool(e2)
        e3, state = self.e3(p2, key=ks[2], state=state)
        p3 = self.pool(e3)
        e4, state = self.e4(p3, key=ks[3], state=state)
        p4 = self.pool(e4)
        b, state = self.b(p4, key=ks[4], state=state)
        d1, state = self.d1(b, e4, key=ks[5], state=state)
        d2, state = self.d2(d1, e3, key=ks[6], state=state)
        d3, state = self.d3(d2, e2, key=ks[7], state=state)
        d4, state = self.d4(d3, e1, key=ks[8], state=state)
        logits = self.out(d4)
        return logits, state
