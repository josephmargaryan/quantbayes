from typing import Callable, Literal, Optional, Tuple
import jax.numpy as jnp
import equinox as eqx
from quantbayes.stochax.utils.regularizers import network_lipschitz_upper


def make_lipschitz_upper_fn(
    *,
    conv_mode: Literal[
        "tn", "circular_fft", "circular_gram", "min_tn_circ_embed", "circ_plus_lr"
    ] = "tn",
    conv_tn_iters: int = 8,
    conv_gram_iters: int = 5,  # <<< NEW: expose Gram-iteration steps
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    conv_input_shape: Optional[Tuple[int, int]] = None,
    allow_exact_hints_for: Tuple[str, ...] = (
        "RFFTCirculant1D",
        "RFFTCirculant2D",
        "SVDDense",
        "SpectralDense",
    ),
) -> Callable[[object, Optional[object]], jnp.ndarray]:
    """Builds a JIT'd callable L(model, state) that returns a certified global
    Lipschitz upper bound under ℓ2, using the selected convolution certificate.

    Notes
    -----
    • Pass `conv_fft_shape=(H,W)` only for exact circular models.
    • Pass `conv_input_shape=(H,W)` for grid-aware certificates
      ("min_tn_circ_embed", "circ_plus_lr"); the network routine will propagate
      (H,W) layer-by-layer to tighten the bound.
    • BN-aware: the `state` argument is forwarded so BatchNorm uses running stats.
    """

    def _L_params(params, static, state):
        m = eqx.combine(params, static)
        return network_lipschitz_upper(
            m,
            state=state,  # <-- critical for BN
            conv_mode=conv_mode,
            conv_tn_iters=conv_tn_iters,
            conv_gram_iters=conv_gram_iters,  # <<< NEW
            conv_fft_shape=conv_fft_shape,
            conv_input_shape=conv_input_shape,
            allow_exact_hints_for=allow_exact_hints_for,
        )

    _L_params = eqx.filter_jit(_L_params)

    def _contains_allowed(m) -> bool:
        found = False

        def visit(x):
            nonlocal found
            if type(x).__name__ in allow_exact_hints_for:
                found = True
            if isinstance(x, eqx.Module):
                for v in vars(x).values():
                    visit(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    visit(v)
            elif isinstance(x, dict):
                for v in x.values():
                    visit(v)

        visit(m)
        return found

    def L(m, state=None):
        params, static = eqx.partition(m, eqx.is_inexact_array)
        val = _L_params(params, static, state)
        # Fallback if we somehow got 1.0 despite having exact spectral layers
        if _contains_allowed(m) and jnp.asarray(val).item() == 1.0:
            return network_lipschitz_upper(
                m,
                state=state,  # <-- also thread here
                conv_mode=conv_mode,
                conv_tn_iters=conv_tn_iters,
                conv_gram_iters=conv_gram_iters,  # <<< NEW
                conv_fft_shape=conv_fft_shape,
                conv_input_shape=conv_input_shape,
                allow_exact_hints_for=allow_exact_hints_for,
            )
        return val

    return L
