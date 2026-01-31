# quantbayes/stochax/diffusion/conditioning/cfg.py
import jax.numpy as jnp


def mix_cfg(uncond: jnp.ndarray, cond: jnp.ndarray, scale: float) -> jnp.ndarray:
    return uncond + scale * (cond - uncond)


def rescale_cfg(
    uncond: jnp.ndarray, guided: jnp.ndarray, rescale: float = 0.7
) -> jnp.ndarray:
    """
    Rescale the guided prediction so its per-sample std matches uncond's std,
    then blend with 'rescale' factor. Helps tame saturation at large CFG.
    """
    axes = tuple(range(1, guided.ndim))
    std_u = jnp.std(uncond, axis=axes, keepdims=True) + 1e-6
    std_g = jnp.std(guided, axis=axes, keepdims=True) + 1e-6
    guided_norm = guided * (std_u / std_g)
    return (
        uncond + rescale * (guided_norm - uncond) + (1.0 - rescale) * (guided - uncond)
    )
