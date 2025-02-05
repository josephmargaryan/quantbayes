# score_diffusion/sde/sde_utils.py

import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax as dfx

def int_beta_linear(t):
    return t  # simple integral of beta(t)=1 => t

def weight_fn(t):
    return 1 - jnp.exp(-t)

def single_loss_fn(model, weight, int_beta, data, t, key):
    mean = data * jnp.exp(-0.5 * int_beta(t))
    var = jnp.maximum(1 - jnp.exp(-int_beta(t)), 1e-5)
    std = jnp.sqrt(var)
    noise = jr.normal(key, data.shape)
    y = mean + std * noise
    pred = model(t, y)
    return weight(t) * jnp.mean((pred + noise / std) ** 2)

def batch_loss_fn(model, weight, int_beta, data, t1, key):
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskeys = jr.split(losskey, batch_size)
    # Low-discrepancy sampling over t
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1/batch_size)
    t = t + (t1/batch_size)*jnp.arange(batch_size)
    loss_fn = ft.partial(single_loss_fn, model, weight, int_beta)
    loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0))
    # shape => [batch_size]
    losses = loss_fn(data, t, losskeys)
    return jnp.mean(losses)

@jax.jit
def single_sample_fn(model, int_beta, data_shape, dt0, t1, key):
    """
    The reverse-time SDE as ODE approach for sampling (using Diffrax).
    """
    def drift(t, y, args):
        # derivative of int_beta wrt t is beta(t)=1 for int_beta(t)=t
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * (y + model(t, y))

    term = dfx.ODETerm(drift)
    solver = dfx.Tsit5()
    t0 = 0
    y1 = jr.normal(key, data_shape)
    # reverse time
    sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, y1)
    return sol.ys[0]
