# quantbayes/stochax/diffusion/objectives/flow_matching.py
from __future__ import annotations
import jax, jax.numpy as jnp, jax.random as jr


def _alpha_sigma(t):
    return 1.0 - t, t


def _dalpha_dsigma(t):
    return -jnp.ones_like(t), jnp.ones_like(t)


def fm_batch_loss(model, data, *, key, t_min=0.0, t_max=1.0):
    b = data.shape[0]
    kt, kz = jr.split(key)
    t = jr.uniform(kt, (b,), minval=t_min, maxval=t_max)
    z = jr.normal(kz, data.shape)

    # broadcast scalars to data shape
    if data.ndim == 4:
        shp = (b, 1, 1, 1)
    elif data.ndim == 2:
        shp = (b, 1)
    else:
        shp = (b,) + (1,) * (data.ndim - 1)

    a, s = _alpha_sigma(t)
    da, ds = _dalpha_dsigma(t)
    a, s, da, ds = a.reshape(shp), s.reshape(shp), da.reshape(shp), ds.reshape(shp)

    x_t = a * data + s * z
    u_t = da * data + ds * z
    v_pred = jax.vmap(lambda ti, xi: model(ti, xi))(t, x_t)
    return jnp.mean((v_pred - u_t) ** 2)
