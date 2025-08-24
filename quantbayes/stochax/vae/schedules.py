# quantbayes/stochax/vae/schedules.py
import jax
import jax.numpy as jnp

__all__ = [
    "beta_linear",
    "beta_cosine",
    "beta_sigmoid",
    "make_beta_schedule",
    "make_beta",  # back-compat alias
]


def _as_f32(x):
    return jnp.asarray(x, dtype=jnp.float32)


def beta_linear(step, warmup_steps):
    step = _as_f32(step)
    ws = jnp.maximum(_as_f32(warmup_steps), _as_f32(1.0))
    return jnp.clip(step / ws, 0.0, 1.0)


def beta_cosine(step, warmup_steps):
    step = _as_f32(step)
    ws = jnp.maximum(_as_f32(warmup_steps), _as_f32(1.0))
    t = jnp.clip(step / ws, 0.0, 1.0)
    return 0.5 - 0.5 * jnp.cos(jnp.pi * t)


def beta_sigmoid(step, warmup_steps, k=10.0):
    step = _as_f32(step)
    ws = jnp.maximum(_as_f32(warmup_steps), _as_f32(1.0))
    k = _as_f32(k)
    t = (step / ws - 0.5) * k
    return jax.nn.sigmoid(t)


def make_beta_schedule(name="linear", warmup_steps=1000, **kwargs):
    """Factory returning a callable beta(step) that outputs a JAX scalar."""
    name = (name or "linear").lower()
    if name == "linear":
        return lambda s: beta_linear(s, warmup_steps)
    if name == "cosine":
        return lambda s: beta_cosine(s, warmup_steps)
    if name == "sigmoid":
        k = kwargs.get("k", 10.0)
        return lambda s: beta_sigmoid(s, warmup_steps, k)
    raise ValueError(f"Unknown beta schedule: {name!r}")


# ---- Backwards-compat alias ----
make_beta = make_beta_schedule
