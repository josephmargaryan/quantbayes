# quantbayes/stochax/utils/lip_upper.py
from typing import Callable, Literal, Optional, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
from quantbayes.stochax.utils.regularizers import network_lipschitz_upper


def make_lipschitz_upper_fn(
    *,
    conv_mode: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    conv_tn_iters: int = 8,
    conv_gram_iters: int = 5,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    conv_input_shape: Optional[Tuple[int, int]] = None,
    conv_circ_embed_candidates: Optional[Tuple[Tuple[int, int], ...]] = None,
    allow_exact_hints_for: Tuple[str, ...] = (
        "RFFTCirculant1D",
        "RFFTCirculant2D",
        "SVDDense",
        "SpectralDense",
    ),
) -> Callable[[object, Optional[object]], jnp.ndarray]:
    """Builds a JIT'd callable L(model, state) that returns a certified global
    Lipschitz upper bound under ℓ2, using the selected convolution certificate.
    """

    def _L_params(params, static, state):
        m = eqx.combine(params, static)
        return network_lipschitz_upper(
            m,
            state=state,  # BN-aware
            conv_mode=conv_mode,
            conv_tn_iters=conv_tn_iters,
            conv_gram_iters=conv_gram_iters,
            conv_fft_shape=conv_fft_shape,
            conv_input_shape=conv_input_shape,
            conv_circ_embed_candidates=conv_circ_embed_candidates,
            allow_exact_hints_for=allow_exact_hints_for,
        )

    _L_params = eqx.filter_jit(_L_params)

    def L(m, state=None):
        params, static = eqx.partition(m, eqx.is_inexact_array)
        val = _L_params(params, static, state)
        # Force a concrete scalar array back to Python
        return jax.device_get(jnp.asarray(val, jnp.float32))

    return L
