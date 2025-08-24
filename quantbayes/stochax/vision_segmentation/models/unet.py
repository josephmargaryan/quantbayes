from typing import Tuple, Any
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx


def _match(x: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    """Centre-pad / crop spatial dims so `x.shape[-2:] == ref.shape[-2:]`."""
    h, w = x.shape[-2:]
    H, W = ref.shape[-2:]
    dh, dw = H - h, W - w

    if dh > 0 or dw > 0:  # pad
        pads = [(0, 0)] * (x.ndim - 2) + [
            (dh // 2, dh - dh // 2),
            (dw // 2, dw - dw // 2),
        ]
        x = jnp.pad(x, pads)
    if dh < 0 or dw < 0:  # crop
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


class Up(eqx.Module):
    up: eqx.nn.ConvTranspose2d
    conv: ConvBlock

    def __init__(self, cin: int, skip: int, cout: int, *, key):
        k1, k2 = jr.split(key, 2)
        self.up = eqx.nn.ConvTranspose2d(cin, cout, 2, stride=2, key=k1)
        self.conv = ConvBlock(cout + skip, cout, key=k2)

    def __call__(
        self, x: jnp.ndarray, skip: jnp.ndarray, *, key, state: Any
    ) -> Tuple[jnp.ndarray, Any]:
        k1, k2 = jr.split(key, 2)
        x = self.up(x, key=k1)
        x, skip = _match(x, skip), _match(skip, x)
        x = jnp.concatenate([skip, x], axis=0)  # channels first
        x, state = self.conv(x, key=k2, state=state)
        return x, state


class UNet(eqx.Module):
    e1: ConvBlock
    e2: ConvBlock
    e3: ConvBlock
    e4: ConvBlock
    b: ConvBlock
    d1: Up
    d2: Up
    d3: Up
    d4: Up
    out: eqx.nn.Conv2d
    pool: eqx.nn.MaxPool2d

    def __init__(self, *, in_ch: int = 3, out_ch: int = 1, base: int = 8, key):
        k = jr.split(key, 10)
        self.e1 = ConvBlock(in_ch, base, key=k[0])
        self.e2 = ConvBlock(base, base * 2, key=k[1])
        self.e3 = ConvBlock(base * 2, base * 4, key=k[2])
        self.e4 = ConvBlock(base * 4, base * 8, key=k[3])
        self.b = ConvBlock(base * 8, base * 16, key=k[4])

        self.d1 = Up(base * 16, base * 8, base * 8, key=k[5])
        self.d2 = Up(base * 8, base * 4, base * 4, key=k[6])
        self.d3 = Up(base * 4, base * 2, base * 2, key=k[7])
        self.d4 = Up(base * 2, base, base, key=k[8])

        self.out = eqx.nn.Conv2d(base, out_ch, 1, key=k[9])
        self.pool = eqx.nn.MaxPool2d(2, 2)

    def __call__(self, x: jnp.ndarray, key, state: Any) -> Tuple[jnp.ndarray, Any]:
        k = jr.split(key, 9)
        e1, state = self.e1(x, key=k[0], state=state)
        p1 = self.pool(e1)
        e2, state = self.e2(p1, key=k[1], state=state)
        p2 = self.pool(e2)
        e3, state = self.e3(p2, key=k[2], state=state)
        p3 = self.pool(e3)
        e4, state = self.e4(p3, key=k[3], state=state)
        p4 = self.pool(e4)

        b, state = self.b(p4, key=k[4], state=state)

        d1, state = self.d1(b, e4, key=k[5], state=state)
        d2, state = self.d2(d1, e3, key=k[6], state=state)
        d3, state = self.d3(d2, e2, key=k[7], state=state)
        d4, state = self.d4(d3, e1, key=k[8], state=state)

        logits = self.out(d4)
        return logits, state
