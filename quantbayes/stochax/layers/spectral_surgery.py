from __future__ import annotations
from typing import Callable, Any, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from equinox import tree_at

from quantbayes.stochax.layers.spectral_layers import SVDDense, SpectralConv2d

""" 
# After you constructed and loaded your ResNetClassifier with torchvision weights:
from spectral_surgery import spectralize_resnet_classifier

key = jr.PRNGKey(0)
spec_model = spectralize_resnet_classifier(model, which_convs="3x3+fc", alpha_init=1.0, key=key)

# Train as usual; your global_spectral_penalty() will now act on s (and any other spectral layers).

"""


# ---------- Exact warm-starts from pretrained weights ----------


def linear_to_svddense(lin: eqx.nn.Linear, alpha_init=1.0) -> SVDDense:
    W = lin.weight  # (out,in)
    U, s, Vh = jnp.linalg.svd(W, full_matrices=False)
    V = Vh.T
    b = lin.bias if lin.bias is not None else jnp.zeros((W.shape[0],), W.dtype)
    return SVDDense(U, V, s, b, alpha_init=alpha_init)


def conv2d_to_spectralconv2d(conv: eqx.nn.Conv2d, *, alpha_init=1.0, key=None):
    """Return SpectralConv2d with U,V,s set so W is exactly reconstructed."""
    from quantbayes.stochax.layers import SpectralConv2d  # your class

    C_out, C_in, Hk, Wk = conv.weight.shape
    W_mat = conv.weight.reshape(C_out, C_in * Hk * Wk)
    U, s, Vh = jnp.linalg.svd(W_mat, full_matrices=False)
    V = Vh.T
    # Build layer with identical stride/padding.
    strides = (
        conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
    )
    padding = conv.padding  # pass through (int/tuple/"SAME" all OK in your impl)
    key = jr.PRNGKey(0) if key is None else key
    lyr = SpectralConv2d(
        C_in=C_in,
        C_out=C_out,
        H_k=Hk,
        W_k=Wk,
        strides=strides,
        padding=padding,
        alpha_init=alpha_init,
        key=key,
        init_scale=1e-6,
        bias_scale=0.0,  # small init; we replace immediately
    )
    d = lyr.s.shape[0]  # = min(C_out, C_in*Hk*Wk)
    lyr = tree_at(lambda m: m.U, lyr, U[:, :d])
    lyr = tree_at(lambda m: m.V, lyr, V[:, :d])
    lyr = tree_at(lambda m: m.s, lyr, s[:d])
    b = conv.bias if conv.bias is not None else jnp.zeros((C_out,), W_mat.dtype)
    lyr = tree_at(lambda m: m.bias, lyr, b)
    return lyr


# ---------- Optional: initialise a circulant-kernel from a 3×3 conv ----------
def conv2d_to_circulant2d(conv: eqx.nn.Conv2d, H_pad: int, W_pad: int, *, key=None):
    """Create a SpectralCirculantLayer2d that approximates a 3×3 conv at resolution (H_pad,W_pad).
    Note: circular boundary ≠ zero-padding; use for the 'spectral-architecture' ablations.
    """
    from quantbayes.stochax.layers import SpectralCirculantLayer2d, _enforce_hermitian

    C_out, C_in, Hk, Wk = conv.weight.shape
    if (Hk, Wk) != (3, 3):
        raise ValueError("This helper assumes a 3×3 kernel; generalise if needed.")
    # Embed 3×3 in a big H_pad×W_pad kernel centered at (0,0) in circular conv sense:
    k_full = jnp.zeros((C_out, C_in, H_pad, W_pad), conv.weight.dtype)
    # place 3×3 at top-left; circular conv interprets it as centered
    k_full = k_full.at[:, :, :3, :3].set(conv.weight)
    K = jnp.fft.fftn(k_full, axes=(-2, -1))
    K = _enforce_hermitian(K)
    w_real, w_imag = K.real, K.imag
    bias = (
        conv.bias if conv.bias is not None else jnp.zeros((C_out,), conv.weight.dtype)
    )
    key = jr.PRNGKey(0) if key is None else key
    # Build layer and replace params
    lyr = SpectralCirculantLayer2d(
        C_in=C_in,
        C_out=C_out,
        H_in=H_pad,
        W_in=W_pad,
        H_pad=H_pad,
        W_pad=W_pad,
        alpha_init=1.0,
        key=key,
        init_scale=1e-6,
        bias_scale=0.0,
    )
    lyr = tree_at(lambda m: m.w_real, lyr, w_real)
    lyr = tree_at(lambda m: m.w_imag, lyr, w_imag)
    lyr = tree_at(
        lambda m: m.bias,
        lyr,
        jnp.broadcast_to(bias[:, None, None], (C_out, H_pad, W_pad)),
    )
    return lyr


# ---------- Generic pytree surgery (replace matching modules) ----------


def replace_modules(
    obj: Any, predicate: Callable[[Any], bool], convert: Callable[[Any], Any]
) -> Any:
    """Recursively traverse an Equinox pytree and replace any leaf module
    satisfying `predicate` with `convert(module)`."""
    # Leaf replacement
    if predicate(obj):
        return convert(obj)

    # Recurse into modules
    if isinstance(obj, eqx.Module):
        for name, attr in vars(obj).items():
            new_attr = replace_modules(attr, predicate, convert)
            if new_attr is not attr:
                obj = tree_at(lambda m: getattr(m, name), obj, new_attr)
        return obj

    # Containers
    if isinstance(obj, tuple):
        new = tuple(replace_modules(x, predicate, convert) for x in obj)
        if new != obj:  # replace if changed
            return new
        return obj
    if isinstance(obj, list):
        new = [replace_modules(x, predicate, convert) for x in obj]
        return new
    if isinstance(obj, dict):
        return {k: replace_modules(v, predicate, convert) for k, v in obj.items()}

    # Leaves
    return obj


# ---------- Ready-made “policies” ----------


def spectralize_resnet_classifier(
    model, *, which_convs: str = "3x3+fc", alpha_init=1.0, key=jr.PRNGKey(0)
):
    """Replace (a) all 3×3 convs with SpectralConv2d, and (b) the final fc with SVDDense."""
    # 3×3 convs
    if "3x3" in which_convs:

        def is_3x3(m):
            return isinstance(m, eqx.nn.Conv2d) and m.weight.shape[-2:] == (3, 3)

        kiter = iter(jr.split(key, 10_000))
        model = replace_modules(
            model,
            is_3x3,
            lambda c: conv2d_to_spectralconv2d(
                c, alpha_init=alpha_init, key=next(kiter)
            ),
        )
    # fc
    if "fc" in which_convs:

        def is_fc(m):
            return isinstance(m, eqx.nn.Linear)

        model = replace_modules(
            model, is_fc, lambda l: linear_to_svddense(l, alpha_init=alpha_init)
        )
    return model


def spectralize_vit(model, *, which: str = "all_linear", alpha_init=1.0):
    """Replace every eqx.nn.Linear in ViT with SVDDense (Q/K/V/out + MLP + patch embed)."""

    def is_lin(m):
        return isinstance(m, eqx.nn.Linear)

    return replace_modules(
        model, is_lin, lambda l: linear_to_svddense(l, alpha_init=alpha_init)
    )
