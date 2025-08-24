# quantbayes/stochax/utils/regularizers.py
# -------------------------------------------------------------------------
# Unified math-first regularizers and norm utilities for dense/conv/spectral
# layers in JAX/Equinox. Everything here is JIT-safe and differentiable.
#
# Contents
#   • global_frobenius_penalty          (skip biases)                [classic]
#   • global_spectral_penalty           (back-compat hook aggregator)
#   • global_spectral_norm_penalty      (Σ per-layer operator norms; exact/upper bound)
#   • ocnn_spectral_penalty_circular2d  (||K^*K − I||₂ via FFT; circular stride=1)
#   • conv_ratio_penalty                (√(hw)·||K||σ / ||K||F)     [Grishina §6.3]
#   • sobolev_jacobian_penalty          (E ||∂f/∂x||F² via Hutchinson/JVP)
#   • sobolev_kernel_smoothness         (∑ ||∇K||² over spatial kernel)
#   • network_lipschitz_upper           (∏ layer-wise σ̂ upper bounds)
#   • lip_product_penalty               (τ · log Lip_upper)
#
# References
#   • Spectral norm regularization (power iteration):
#       T. Miyato & Y. Yoshida, *Spectral Norm Regularization for Improving the
#       Generalizability of Deep Learning* (2019).
#   • Exact σ for circular 2D conv via per-frequency SVD:
#       H. Sedghi, V. Gupta, P. Long, *The Singular Values of Convolutional Layers*,
#       ICLR 2019.
#   • Tight upper bound for (general) conv via tensor spectral norm:
#       E. Grishina, M. Gorbunov, M. Rakhuba,
#       *Tight and Efficient Upper Bound on Spectral Norm of Convolutional Layers*,
#       ECCV 2024.
#   • Jacobian Sobolev (trace estimation):
#       Hutchinson (1990); Pearlmutter (1994, JVP).
#   • Orthogonality for convs (discussion/variants):
#       See Grishina §6.3 and citations therein.
# -------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Callable, Literal, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _isinstance_by_name(obj, names: Sequence[str]) -> bool:
    return any(type(obj).__name__ == n for n in names)


def _is_trainable_array(x):
    """Heuristic used by Frobenius penalty:
    - includes inexact arrays with ndim >= 2
    - excludes things marked as `_frozen`
    - excludes biases (0D/1D)
    """
    return (
        eqx.is_inexact_array(x)
        and (getattr(x, "_frozen", False) is False)
        and x.ndim >= 2
    )


def _sigma_conv_circulant_embed_upper_2d(
    K: jnp.ndarray,  # (Cout, Cin, kH, kW)
    in_hw: Tuple[int, int],  # (H_in, W_in) seen by this layer
) -> jnp.ndarray:
    """
    Upper bound for zero/reflect-like convs via circulant embedding on the
    grid (H_in+kH-1, W_in+kW-1). Uses unitary FFT; multiply by sqrt(area).
    """
    Co, Ci, kH, kW = map(int, K.shape)
    H_in, W_in = map(int, in_hw)
    Hf = H_in + kH - 1
    Wf = W_in + kW - 1

    pad_h = max(0, Hf - kH)
    pad_w = max(0, Wf - kW)
    Kpad = jnp.pad(K, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))  # (Co, Ci, Hf, Wf)

    # unitary FFT -> eigenvalues are sqrt(Hf*Wf) * FFT(Kpad)
    Kf = jnp.fft.rfft2(Kpad, s=(Hf, Wf), axes=(-2, -1), norm="ortho")
    mats = jnp.transpose(Kf, (2, 3, 0, 1)).reshape(-1, Co, Ci)
    smax = jax.vmap(lambda M: jnp.linalg.svd(M, compute_uv=False)[0])(mats)
    return (jnp.sqrt(Hf * Wf) * jnp.max(jnp.real(smax))).astype(jnp.float32)


# -------------------------------------------------------------------------
# Classic weight penalties
# -------------------------------------------------------------------------


def global_frobenius_penalty(model: Any) -> jnp.ndarray:
    """
    ∑ ||W||² over all trainable arrays with ndim ≥ 2 (skips biases).

    Exactly the "weight decay" / L2 norm on matrices and conv kernels. This
    intentionally excludes 0D/1D arrays so biases are not penalized.
    """
    params = eqx.filter(model, _is_trainable_array)
    leaves = jax.tree_util.tree_leaves(params)
    total = jnp.array(
        0.0,
        dtype=jnp.result_type(
            *[getattr(p, "dtype", jnp.float32) for p in leaves] or [jnp.float32]
        ),
    )
    for p in leaves:
        total = total + jnp.sum(jnp.abs(p) ** 2)  # complex-safe
    return total


def global_spectral_penalty(model: Any) -> jnp.ndarray:
    """
    Backward-compat aggregator: if a module exposes `__spectral_penalty__()`
    (your custom spectral layers), sum those; also includes any `delta_alpha`.

    Use `global_spectral_norm_penalty` for general (per-layer σ) penalties.
    """
    total = jnp.array(0.0, dtype=jnp.float32)

    if hasattr(model, "__spectral_penalty__"):
        total = total + model.__spectral_penalty__()
        if hasattr(model, "delta_alpha"):
            total = total + jnp.sum(model.delta_alpha**2)

    if isinstance(model, eqx.Module):
        for v in vars(model).values():
            total = total + global_spectral_penalty(v)
    elif isinstance(model, (list, tuple)):
        for v in model:
            total = total + global_spectral_penalty(v)
    elif isinstance(model, dict):
        for v in model.values():
            total = total + global_spectral_penalty(v)
    return total


# -------------------------------------------------------------------------
# Linear/Conv per-layer spectral norms (exact or upper bounds)
# -------------------------------------------------------------------------


def _top_sigma_mat(M: jnp.ndarray) -> jnp.ndarray:
    """Largest singular value of a 2D matrix (exact, JAX SVD)."""
    return jnp.linalg.svd(M, compute_uv=False)[0]


def _sigma_linear_weight(
    W: jnp.ndarray,
    method: Literal["svd", "pi"] = "svd",
    pi_steps: int = 1,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    σ_max for dense/linear weight.
    - 'svd' is exact (recommended for research).
    - 'pi' is a cheap power-iteration proxy (useful if you need speed).
    """
    W2 = W.reshape(W.shape[0], -1)
    if method == "svd":
        return _top_sigma_mat(W2)

    # PI (stateless, few steps)
    v = jnp.ones((W2.shape[1],), dtype=W2.dtype) / jnp.sqrt(W2.shape[1])
    u = W2 @ v
    u = u / (jnp.linalg.norm(u) + eps)

    def body(c, _):
        u, v = c
        v = W2.T @ u
        v = v / (jnp.linalg.norm(v) + eps)
        u = W2 @ v
        u = u / (jnp.linalg.norm(u) + eps)
        return (u, v), None

    (u, v), _ = jax.lax.scan(body, (u, v), None, length=max(0, pi_steps - 1))
    # Rayleigh quotient
    return jnp.vdot(u, W2 @ v).real


def _sigma_conv_kernel_flat(
    K: jnp.ndarray, method: Literal["svd", "pi"] = "svd"
) -> jnp.ndarray:
    """
    Miyato/Yoshida proxy for conv: flatten K to (Cout, Cin*kH*kW) and take σ_max.
    Looser than frequency-domain or TN bounds, but simple and fast.
    """
    Co, Ci, kH, kW = K.shape
    M = K.reshape(Co, Ci * kH * kW)
    return (
        _top_sigma_mat(M) if method == "svd" else _sigma_linear_weight(M, method="pi")
    )


# --- Tensor spectral norm of the 4D kernel (HOPM) -------------------------


def _tensor_spectral_norm_K_4d(
    K: jnp.ndarray, *, iters: int = 10, key=None
) -> jnp.ndarray:
    """
    ||K||_σ for K ∈ R^{Cout×Cin×H×W} via Higher-Order Power Method (HOPM).

    Uses complex unit vectors as required by the tight bound proofs (see Grishina
    §5 + Remark 1). Complexity O(Cout*Cin*H*W*iters). Deterministic if key fixed.
    """
    Co, Ci, H, W = K.shape
    Kc = K.astype(jnp.complex64 if K.dtype == jnp.float32 else jnp.complex128)
    if key is None:
        key = jr.PRNGKey(0)
    k1, k2, k3, k4 = jr.split(key, 4)

    def _unit(rng, n, dtype):
        v = jr.normal(rng, (n,), dtype=dtype) + 1j * jr.normal(rng, (n,), dtype=dtype)
        return v / (jnp.linalg.norm(v) + 1e-12)

    u1 = _unit(k1, Co, Kc.dtype)
    u2 = _unit(k2, Ci, Kc.dtype)
    u3 = _unit(k3, H, Kc.dtype)
    u4 = _unit(k4, W, Kc.dtype)

    def body(c, _):
        a1, a2, a3, a4 = c
        v1 = jnp.einsum("oihw,i,h,w->o", Kc, a2, a3, a4)
        a1 = jnp.conj(v1) / (jnp.linalg.norm(v1) + 1e-12)
        v2 = jnp.einsum("oihw,o,h,w->i", Kc, a1, a3, a4)
        a2 = jnp.conj(v2) / (jnp.linalg.norm(v2) + 1e-12)
        v3 = jnp.einsum("oihw,o,i,w->h", Kc, a1, a2, a4)
        a3 = jnp.conj(v3) / (jnp.linalg.norm(v3) + 1e-12)
        v4 = jnp.einsum("oihw,o,i,h->w", Kc, a1, a2, a3)
        a4 = jnp.conj(v4) / (jnp.linalg.norm(v4) + 1e-12)
        return (a1, a2, a3, a4), None

    (u1, u2, u3, u4), _ = jax.lax.scan(body, (u1, u2, u3, u4), None, length=int(iters))
    val = jnp.einsum("oihw,o,i,h,w->", Kc, u1, u2, u3, u4)
    return jnp.abs(val).real


def _sigma_conv_circ_plus_lr_upper_2d(
    K: jnp.ndarray,
    in_hw: Tuple[int, int],  # (H_in, W_in)
    *,
    strides: Tuple[int, int] = (1, 1),
    rhs_dilation: Tuple[int, int] = (1, 1),
) -> jnp.ndarray:
    """
    Certified UB for zero/reflect-like padding:
        ||T||_2 ≤ ||C||_2 + sqrt(B) * ||K||_F

    where:
      • C is the CIRCULAR (stride=1) convolution on the (H_in, W_in) grid with
        dilation=rhs_dilation, whose norm is exact via FFT (Sedghi-style).
      • B upper-bounds #boundary-affected outputs given stride/dilation.
    """
    H_in, W_in = int(in_hw[0]), int(in_hw[1])
    kH, kW = int(K.shape[-2]), int(K.shape[-1])
    sh, sw = int(strides[0]), int(strides[1])
    dh, dw = int(rhs_dilation[0]), int(rhs_dilation[1])

    # 1) exact circular term with dilation
    L_circ = _sigma_conv_circular_fft_exact_2d(K, (H_in, W_in), rhs_dilation=(dh, dw))

    # 2) perimeter-style B (dilation-aware)
    rho_h = (kH - 1) * dh
    rho_w = (kW - 1) * dw
    H_out = (H_in + sh - 1) // sh
    W_out = (W_in + sw - 1) // sw
    ceil_div_h = (rho_h + sh - 1) // sh
    ceil_div_w = (rho_w + sw - 1) // sw
    Bh = 2 * W_out * ceil_div_h
    Bw = 2 * H_out * ceil_div_w
    B = Bh + Bw

    L_rem = jnp.sqrt(jnp.asarray(B, jnp.float32)) * jnp.linalg.norm(K)
    return (L_circ + L_rem).astype(jnp.float32)


def _pack_Q_strided(
    K: jnp.ndarray,
    strides: Tuple[int, int] = (1, 1),
    rhs_dilation: Tuple[int, int] = (1, 1),
) -> jnp.ndarray:
    """Build the Q tensor for strided/dilated conv as in Grishina et al. Thm. 3.

    Args
    ----
    K : (C_out, C_in, h, w) real kernel.
    strides : (s_h, s_w). If an int is passed upstream we convert to tuple.
    rhs_dilation : (d_h, d_w) dilation of the kernel taps (a.k.a. 'kernel dilation').

    Returns
    -------
    Q : (C_out, C_in * s_h * s_w, ceil(h_eff/s_h), ceil(w_eff/s_w)),
        where h_eff = (h-1)*d_h + 1 and similarly for w_eff.
    """
    Co, Ci, H, W = map(int, K.shape)
    sh, sw = int(strides[0]), int(strides[1])
    dh, dw = int(rhs_dilation[0]), int(rhs_dilation[1])

    # Effective support with dilation (insert zeros between taps)
    Heff = (H - 1) * dh + 1
    Weff = (W - 1) * dw + 1

    # Spatial size of the stride cosets
    Hs = (Heff + sh - 1) // sh
    Ws = (Weff + sw - 1) // sw

    # Dilate the kernel if needed by placing original taps on a (dh,dw)-grid
    if (dh, dw) != (1, 1):
        Kh = jnp.zeros((Co, Ci, Heff, Weff), dtype=K.dtype)
        Kh = Kh.at[..., ::dh, ::dw].set(K)
    else:
        Kh = K

    parts = []
    for p in range(sh):
        for q in range(sw):
            # Take the (p,q) stride coset
            sub = Kh[
                ..., p:Heff:sh, q:Weff:sw
            ]  # (Co, Ci, Hs', Ws') possibly smaller at borders
            # Pad to (Hs, Ws) so all parts align
            pad_h = Hs - sub.shape[-2]
            pad_w = Ws - sub.shape[-1]
            if pad_h > 0 or pad_w > 0:
                sub = jnp.pad(sub, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
            parts.append(sub)  # (Co, Ci, Hs, Ws)

    # Concatenate the s_h*s_w cosets along the input-channel axis
    Q = jnp.concatenate(parts, axis=1)  # (Co, Ci*sh*sw, Hs, Ws)
    return Q


def _sigma_conv_TN_upper(
    K: jnp.ndarray,
    *,
    strides: Tuple[int, int] = (1, 1),
    rhs_dilation: Tuple[int, int] = (1, 1),
    iters: int = 10,
    key=None,
) -> jnp.ndarray:
    """Grishina-style tight/efficient upper bound for strided/dilated conv.

    Bound:
        ||T_s||_2  ≤  sqrt( ceil(h_eff/s_h) * ceil(w_eff/s_w) ) * ||Q||_σ

    where Q is the 4th-order tensor constructed by stride cosets (and dilation),
    and ||Q||_σ is the tensor spectral norm (computed with HOPM).

    Args
    ----
    K : (C_out, C_in, h, w)
    strides : (s_h, s_w)
    rhs_dilation : (d_h, d_w)
    iters : HOPM iterations (10–20 is usually plenty)
    key : optional PRNGKey if your HOPM initializer uses randomness

    Returns
    -------
    ub : scalar upper bound on the operator norm of the convolution Jacobian.
    """
    Q = _pack_Q_strided(
        K, strides=strides, rhs_dilation=rhs_dilation
    )  # (Co, Ci*sh*sw, Hs, Ws)
    Hs, Ws = int(Q.shape[-2]), int(Q.shape[-1])
    # _tensor_spectral_norm_K_4d must already exist in your file (HOPM-based)
    return jnp.sqrt(Hs * Ws) * _tensor_spectral_norm_K_4d(Q, iters=iters, key=key)


# --- Exact σ for circular, stride=1 2D conv via per-frequency SVD ---------


def _sigma_conv_circular_fft_exact_2d(
    K: jnp.ndarray,
    fft_shape: Tuple[int, int],
    rhs_dilation: Tuple[int, int] = (1, 1),
) -> jnp.ndarray:
    """
    Exact ||Conv_circ(K; dilation, stride=1)||_2 via per-frequency SVD.
    IMPORTANT: with unitary FFT we must scale by sqrt(H*W).
    """
    Co, Ci, kH, kW = K.shape
    Hf, Wf = int(fft_shape[0]), int(fft_shape[1])
    dh, dw = int(rhs_dilation[0]), int(rhs_dilation[1])

    # modulo-scatter taps onto the Hf×Wf grid (dilation-aware)
    Kgrid = jnp.zeros((Co, Ci, Hf, Wf), dtype=K.dtype)
    for p in range(kH):
        i = (p * dh) % Hf
        for q in range(kW):
            j = (q * dw) % Wf
            Kgrid = Kgrid.at[..., i, j].add(K[..., p, q])

    Kf = jnp.fft.rfft2(Kgrid, s=(Hf, Wf), axes=(-2, -1), norm="ortho")
    mats = jnp.transpose(Kf, (2, 3, 0, 1)).reshape(-1, Co, Ci)
    smax = jax.vmap(lambda M: jnp.linalg.svd(M, compute_uv=False)[0])(mats)
    return (jnp.sqrt(Hf * Wf) * jnp.max(jnp.real(smax))).astype(jnp.float32)


# -------------------------------------------------------------------------
# Aggregated per-model spectral norm regularizer (Σ σ_i)
# -------------------------------------------------------------------------


def global_spectral_norm_penalty(
    model: Any,
    *,
    apply_to: Sequence[str] = ("spectral", "linear", "conv"),
    conv_mode: Literal[
        "tn", "circular_fft", "kernel_flat", "min_tn_circ_embed", "circ_plus_lr"
    ] = "tn",
    conv_tn_iters: int = 10,
    conv_fft_shape: Optional[Tuple[int, int]] = None,  # for exact circular stride=1
    conv_input_shape: Optional[Tuple[int, int]] = None,  # for circ-embed / circ_plus_lr
) -> jnp.ndarray:
    """
    Sum of per-layer operator norms (spectral norms) across the model.

    Dense (Linear): exact via SVD of weight reshaped to (out, -1).
    Convolution:
      - "tn":            tight/efficient Grishina upper bound (stride/dilation aware).
      - "circular_fft":  exact CIRCULAR conv for stride=1 and any dilation (needs conv_fft_shape);
                         otherwise falls back to TN.
      - "kernel_flat":   Miyato proxy: σ_max of flattened kernel matrix (not conv operator).
      - "min_tn_circ_embed": min(TN, circulant-embed) if input size is given.
      - "circ_plus_lr":  certified UB: ||T|| ≤ ||C||_2 + sqrt(B)·||K||_F,
                         requires conv_input_shape (H_in,W_in).
    """
    total = jnp.array(0.0, dtype=jnp.float32)
    nn = getattr(eqx, "nn", None)
    PARAM_SVD_CONV = {"SpectralConv2d", "AdaptiveSpectralConv2d"}

    def _svd_top_sigma_flat(W: jnp.ndarray) -> jnp.ndarray:
        W2 = W.reshape(W.shape[0], -1)
        return jnp.linalg.svd(W2, compute_uv=False)[0].astype(jnp.float32)

    def visit(x, acc):
        name = type(x).__name__

        # 0) custom hint (not for param-SVD convs)
        if (
            "spectral" in apply_to
            and hasattr(x, "__operator_norm_hint__")
            and name not in PARAM_SVD_CONV
        ):
            try:
                val = x.__operator_norm_hint__()
                if val is not None:
                    return acc + jnp.asarray(val, jnp.float32)
            except Exception:
                pass

        if name == "SpectralNorm" and "spectral" in apply_to:
            return acc + jnp.array(1.0, dtype=jnp.float32)

        # 1) Dense (Linear)
        if nn is not None and isinstance(x, getattr(nn, "Linear", ())):
            if "linear" in apply_to and hasattr(x, "weight"):
                return acc + _svd_top_sigma_flat(x.weight)

        # 2) Convs
        if nn is not None and isinstance(x, getattr(nn, "Conv", ())):
            if "conv" in apply_to and hasattr(x, "weight"):
                num_dims = getattr(x, "num_spatial_dims", x.weight.ndim - 2)
                s = getattr(x, "stride", getattr(x, "strides", (1,) * num_dims))
                if isinstance(s, int):
                    s = (s,) * num_dims
                d = (
                    getattr(x, "rhs_dilation", None)
                    or getattr(x, "dilation", None)
                    or getattr(x, "kernel_dilation", (1,) * num_dims)
                )
                if isinstance(d, int):
                    d = (d,) * num_dims

                if num_dims != 2:
                    return acc + _sigma_conv_kernel_flat(x.weight, method="svd")

                sh, sw = int(s[0]), int(s[1])
                dh, dw = int(d[0]), int(d[1])

                if conv_mode == "kernel_flat":
                    return acc + _sigma_conv_kernel_flat(x.weight, method="svd")

                elif conv_mode == "circular_fft":
                    # exact for stride=1, any dilation if fft_shape given; else fallback to TN
                    if (sh, sw) == (1, 1) and (conv_fft_shape is not None):
                        return acc + _sigma_conv_circular_fft_exact_2d(
                            x.weight, conv_fft_shape, rhs_dilation=(dh, dw)
                        )
                    return acc + _sigma_conv_TN_upper(
                        x.weight,
                        strides=(sh, sw),
                        rhs_dilation=(dh, dw),
                        iters=conv_tn_iters,
                    )

                elif conv_mode == "min_tn_circ_embed" and conv_input_shape is not None:
                    tn = _sigma_conv_TN_upper(
                        x.weight,
                        strides=(sh, sw),
                        rhs_dilation=(dh, dw),
                        iters=conv_tn_iters,
                    )
                    ce = _sigma_conv_circulant_embed_upper_2d(
                        x.weight, in_hw=conv_input_shape
                    )
                    return acc + jnp.minimum(tn, ce)

                elif conv_mode == "circ_plus_lr" and conv_input_shape is not None:
                    return acc + _sigma_conv_circ_plus_lr_upper_2d(
                        x.weight,
                        in_hw=conv_input_shape,
                        strides=(sh, sw),
                        rhs_dilation=(dh, dw),
                    )

                else:  # "tn"
                    return acc + _sigma_conv_TN_upper(
                        x.weight,
                        strides=(sh, sw),
                        rhs_dilation=(dh, dw),
                        iters=conv_tn_iters,
                    )

        # 3) Param-SVD convs → compute conv bound (not hint)
        if name in PARAM_SVD_CONV and "conv" in apply_to:
            W_mat = x.U @ (x.s[:, None] * x.V.T)
            W = jnp.reshape(W_mat, (x.C_out, x.C_in, x.H_k, x.W_k))
            s = getattr(x, "strides", (1, 1))
            if isinstance(s, int):
                s = (s, s)
            sh, sw = int(s[0]), int(s[1])

            if conv_mode == "kernel_flat":
                return acc + _sigma_conv_kernel_flat(W, method="svd")

            elif conv_mode == "circular_fft":
                if (sh, sw) == (1, 1) and (conv_fft_shape is not None):
                    return acc + _sigma_conv_circular_fft_exact_2d(
                        W,
                        conv_fft_shape,
                        rhs_dilation=(1, 1),  # these layers use no dilation
                    )
                return acc + _sigma_conv_TN_upper(
                    W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                )

            elif conv_mode == "min_tn_circ_embed" and conv_input_shape is not None:
                tn = _sigma_conv_TN_upper(
                    W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                )
                ce = _sigma_conv_circulant_embed_upper_2d(W, in_hw=conv_input_shape)
                return acc + jnp.minimum(tn, ce)

            elif conv_mode == "circ_plus_lr" and conv_input_shape is not None:
                return acc + _sigma_conv_circ_plus_lr_upper_2d(
                    W, in_hw=conv_input_shape, strides=(sh, sw), rhs_dilation=(1, 1)
                )

            else:
                return acc + _sigma_conv_TN_upper(
                    W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                )

        # 4) Recurse
        if isinstance(x, eqx.Module):
            for v in vars(x).values():
                acc = visit(v, acc)
            return acc
        if isinstance(x, (list, tuple)):
            for v in x:
                acc = visit(v, acc)
            return acc
        if isinstance(x, dict):
            for v in x.values():
                acc = visit(v, acc)
            return acc

        return acc

    return visit(model, total)


# -------------------------------------------------------------------------
# Orthogonality-style penalties for convs (circular, stride=1)
# -------------------------------------------------------------------------


def ocnn_spectral_penalty_circular2d(
    model: Any,
    *,
    fft_shape: Tuple[int, int],
) -> jnp.ndarray:
    """
    ∑_conv max_ω || (√(HW)·K̂(ω))^* (√(HW)·K̂(ω)) − I ||₂   (circular, stride=1)
    """
    total = jnp.array(0.0)

    def visit(x, acc):
        if (
            hasattr(eqx.nn, "Conv")
            and isinstance(x, eqx.nn.Conv)
            and x.num_spatial_dims == 2
        ):
            Co, Ci, kH, kW = x.weight.shape
            Hf, Wf = int(fft_shape[0]), int(fft_shape[1])

            pad_h = max(0, Hf - kH)
            pad_w = max(0, Wf - kW)
            Kpad = jnp.pad(x.weight, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
            Kf = jnp.fft.rfft2(Kpad, s=(Hf, Wf), axes=(-2, -1), norm="ortho")
            Kf = jnp.sqrt(Hf * Wf) * Kf  # <<< critical scaling

            mats = jnp.transpose(Kf, (2, 3, 0, 1)).reshape(-1, Co, Ci)

            def per_freq(M):
                s = jnp.linalg.svd(M, compute_uv=False)  # singular values of √(HW)·K̂
                smax = s[0]
                smin = s[-1] if s.shape[0] >= Ci else 0.0
                return jnp.maximum(jnp.abs(smax**2 - 1.0), jnp.abs(smin**2 - 1.0))

            vals = jax.vmap(per_freq)(mats)
            return acc + jnp.max(vals).real

        if isinstance(x, eqx.Module):
            for v in vars(x).values():
                acc = visit(v, acc)
        elif isinstance(x, (list, tuple)):
            for v in x:
                acc = visit(v, acc)
        elif isinstance(x, dict):
            for v in x.values():
                acc = visit(v, acc)
        return acc

    return visit(model, total)


def conv_ratio_penalty(
    model: Any,
    *,
    tn_iters: int = 8,
    eps: float = 1e-12,
    reduction: Literal["sum", "mean"] = "sum",
) -> jnp.ndarray:
    """
    ∑ over conv layers of  √(h*w)·||K||σ  /  (||K||F + eps)   (Grishina §6.3).
    This is minimized when singular values of the (Jacobian) are equal.

    Uses the TN estimator for ||K||σ. Applies to eqx.nn.Conv* only.
    """
    vals = []

    def visit(x):
        if hasattr(eqx.nn, "Conv") and isinstance(x, eqx.nn.Conv):
            K = x.weight
            num_dims = getattr(x, "num_spatial_dims", K.ndim - 2)
            if num_dims != 2:
                s_val = jnp.sqrt(K.shape[-2] * K.shape[-1]) * _sigma_conv_kernel_flat(
                    K, method="svd"
                )
            else:
                s = getattr(x, "stride", getattr(x, "strides", (1, 1)))
                d = (
                    getattr(x, "rhs_dilation", None)
                    or getattr(x, "dilation", None)
                    or getattr(x, "kernel_dilation", (1, 1))
                )
                if isinstance(s, int):
                    s = (s, s)
                if isinstance(d, int):
                    d = (d, d)
                s_val = _sigma_conv_TN_upper(
                    K,
                    strides=(int(s[0]), int(s[1])),
                    rhs_dilation=(int(d[0]), int(d[1])),
                    iters=tn_iters,
                )
            f = jnp.linalg.norm(K) + eps
            vals.append(s_val / f)
        else:
            if isinstance(x, eqx.Module):
                for v in vars(x).values():
                    visit(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    visit(v)
            elif isinstance(x, dict):
                for v in x.values():
                    visit(v)

    visit(model)
    if not vals:
        return jnp.array(0.0)
    arr = jnp.stack(vals)
    return jnp.sum(arr) if reduction == "sum" else jnp.mean(arr)


# -------------------------------------------------------------------------
# Sobolev-style penalties
# -------------------------------------------------------------------------


def sobolev_jacobian_penalty(
    model: Any,
    state: Any,
    x: jnp.ndarray,
    key: jr.PRNGKey,
    f_apply: Callable[[Any, Any, jnp.ndarray, jr.PRNGKey], jnp.ndarray],
    *,
    num_samples: int = 1,
    rv: Literal["rademacher", "gaussian"] = "rademacher",
    reduce: Literal["mean", "sum"] = "mean",
) -> jnp.ndarray:
    """
    Estimates E ||J(x)||_F^2 via Hutchinson with JVPs:
      E_u ||J u||^2 = trace(J^T J) = ||J||_F^2

    Args:
      f_apply: function (model, state, x, key) -> output (batched x).
      num_samples: # of probe vectors per batch (1 is often enough).
    """

    def one_sample(k):
        k1, k2 = jr.split(k)
        if rv == "rademacher":
            # JAX provides a native rademacher sampler for float dtypes.
            try:
                u = jr.rademacher(k2, x.shape, dtype=x.dtype)
            except Exception:
                u = (jr.randint(k2, x.shape, 0, 2, dtype=jnp.int32) * 2 - 1).astype(
                    x.dtype
                )
        else:
            u = jr.normal(k2, x.shape, dtype=x.dtype)

        def f_on_x(inp):
            return f_apply(model, state, inp, k1)

        _, Ju = jax.jvp(f_on_x, (x,), (u,))
        sq = jnp.sum(Ju**2, axis=tuple(range(1, Ju.ndim)))  # per-example
        return jnp.mean(sq)

    ks = jr.split(key, num_samples)
    est = jax.vmap(one_sample)(ks)
    val = jnp.mean(est)
    return val if reduce == "mean" else val * x.shape[0]


def sobolev_kernel_smoothness(model: Any) -> jnp.ndarray:
    """
    Discrete Sobolev seminorm on conv kernels: ∑ ||∇K||₂² over spatial dims.
    Applies to eqx.nn.Conv*. For your spectral-circulant 2D convs, extend by
    mapping spectral parameters to spatial kernels then reusing this routine.
    """
    total = jnp.array(0.0)

    def smooth(K: jnp.ndarray) -> jnp.ndarray:
        val = jnp.array(0.0, dtype=K.dtype)
        if K.shape[-2] > 1:
            dh = K[..., 1:, :] - K[..., :-1, :]
            val = val + jnp.sum(dh**2)
        if K.shape[-1] > 1:
            dw = K[..., :, 1:] - K[..., :, :-1]
            val = val + jnp.sum(dw**2)
        return val

    def visit(x, acc):
        if hasattr(eqx.nn, "Conv") and isinstance(x, eqx.nn.Conv):
            return acc + smooth(x.weight)

        # If you want to support your own spectral convs here, uncomment
        # and provide a way to map to spatial kernels:
        # if type(x).__name__ in ["SpectralCirculantLayer2d", "AdaptiveSpectralCirculantLayer2d"]:
        #     K_fft = x.get_fft_kernel()                   # complex spectrum
        #     K_spatial = jnp.fft.ifftn(K_fft, axes=(-2, -1), norm="ortho").real
        #     return acc + smooth(K_spatial)

        if isinstance(x, eqx.Module):
            for v in vars(x).values():
                acc = visit(v, acc)
        elif isinstance(x, (list, tuple)):
            for v in x:
                acc = visit(v, acc)
        elif isinstance(x, dict):
            for v in x.values():
                acc = visit(v, acc)
        return acc

    return visit(model, total)


# -------------------------------------------------------------------------
# Network Lipschitz: product of per-layer σ̂ (upper bound)
# -------------------------------------------------------------------------


def network_lipschitz_upper(
    model: Any,
    *,
    conv_mode: Literal[
        "tn", "circular_fft", "min_tn_circ_embed", "circ_plus_lr"
    ] = "tn",
    conv_tn_iters: int = 8,
    conv_fft_shape: Optional[Tuple[int, int]] = None,  # exact circular stride=1
    conv_input_shape: Optional[Tuple[int, int]] = None,  # circ-embed / circ_plus_lr
) -> jnp.ndarray:
    """
    Upper bound on Lip(f) by multiplying certified per-layer upper bounds.

    Convs:
      - "tn":            Grishina TN (stride/dilation aware).
      - "circular_fft":  exact circular for stride=1, any dilation (needs conv_fft_shape);
                         else TN.
      - "min_tn_circ_embed": min(TN, circulant-embed) if H×W known.
      - "circ_plus_lr":  ||T|| ≤ ||C||_2 + sqrt(B)·||K||_F, needs conv_input_shape.
    """
    log_sum = jnp.array(0.0, dtype=jnp.float32)
    nn = getattr(eqx, "nn", None)
    PARAM_SVD_CONV = {"SpectralConv2d", "AdaptiveSpectralConv2d"}

    def add_log(acc, sig):
        sig = jnp.asarray(sig, jnp.float32)
        return acc + jnp.log(jnp.clip(sig, 1e-12, 1e12))

    def visit(x, acc):
        name = type(x).__name__

        if name == "SpectralNorm":
            return acc

        if nn is not None and isinstance(x, getattr(nn, "Linear", ())):
            W = x.weight.reshape(x.weight.shape[0], -1)
            sig = jnp.linalg.svd(W, compute_uv=False)[0]
            return add_log(acc, sig)

        if nn is not None and isinstance(x, getattr(nn, "Conv", ())):
            num_dims = getattr(x, "num_spatial_dims", x.weight.ndim - 2)
            s = getattr(x, "stride", getattr(x, "strides", (1,) * num_dims))
            if isinstance(s, int):
                s = (s,) * num_dims
            d = (
                getattr(x, "rhs_dilation", None)
                or getattr(x, "dilation", None)
                or getattr(x, "kernel_dilation", (1,) * num_dims)
            )
            if isinstance(d, int):
                d = (d,) * num_dims

            if num_dims != 2:
                sig = _sigma_conv_kernel_flat(x.weight, method="svd")
                return add_log(acc, sig)

            sh, sw = int(s[0]), int(s[1])
            dh, dw = int(d[0]), int(d[1])

            if conv_mode == "circular_fft":
                if (sh, sw) == (1, 1) and (conv_fft_shape is not None):
                    sig = _sigma_conv_circular_fft_exact_2d(
                        x.weight, conv_fft_shape, rhs_dilation=(dh, dw)
                    )
                else:
                    sig = _sigma_conv_TN_upper(
                        x.weight,
                        strides=(sh, sw),
                        rhs_dilation=(dh, dw),
                        iters=conv_tn_iters,
                    )

            elif conv_mode == "min_tn_circ_embed" and conv_input_shape is not None:
                tn = _sigma_conv_TN_upper(
                    x.weight,
                    strides=(sh, sw),
                    rhs_dilation=(dh, dw),
                    iters=conv_tn_iters,
                )
                ce = _sigma_conv_circulant_embed_upper_2d(
                    x.weight, in_hw=conv_input_shape
                )
                sig = jnp.minimum(tn, ce)

            elif conv_mode == "circ_plus_lr" and conv_input_shape is not None:
                sig = _sigma_conv_circ_plus_lr_upper_2d(
                    x.weight,
                    in_hw=conv_input_shape,
                    strides=(sh, sw),
                    rhs_dilation=(dh, dw),
                )

            else:
                sig = _sigma_conv_TN_upper(
                    x.weight,
                    strides=(sh, sw),
                    rhs_dilation=(dh, dw),
                    iters=conv_tn_iters,
                )

            return add_log(acc, sig)

        if name in PARAM_SVD_CONV:
            W_mat = x.U @ (x.s[:, None] * x.V.T)
            W = jnp.reshape(W_mat, (x.C_out, x.C_in, x.H_k, x.W_k))
            s = getattr(x, "strides", (1, 1))
            if isinstance(s, int):
                s = (s, s)
            sh, sw = int(s[0]), int(s[1])

            if (
                conv_mode == "circular_fft"
                and (sh, sw) == (1, 1)
                and (conv_fft_shape is not None)
            ):
                sig = _sigma_conv_circular_fft_exact_2d(
                    W, conv_fft_shape, rhs_dilation=(1, 1)
                )

            elif conv_mode == "min_tn_circ_embed" and conv_input_shape is not None:
                tn = _sigma_conv_TN_upper(
                    W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                )
                ce = _sigma_conv_circulant_embed_upper_2d(W, in_hw=conv_input_shape)
                sig = jnp.minimum(tn, ce)

            elif conv_mode == "circ_plus_lr" and conv_input_shape is not None:
                sig = _sigma_conv_circ_plus_lr_upper_2d(
                    W, in_hw=conv_input_shape, strides=(sh, sw), rhs_dilation=(1, 1)
                )

            else:
                sig = _sigma_conv_TN_upper(
                    W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                )

            return add_log(acc, sig)

        if isinstance(x, eqx.Module):
            for v in vars(x).values():
                acc = visit(v, acc)
            return acc
        if isinstance(x, (list, tuple)):
            for v in x:
                acc = visit(v, acc)
            return acc
        if isinstance(x, dict):
            for v in x.values():
                acc = visit(v, acc)
            return acc

        return acc

    return jnp.exp(visit(model, log_sum))


def lip_product_penalty(
    model: Any,
    *,
    tau: float = 1e-4,
    conv_mode: Literal["tn", "circular_fft"] = "tn",
    conv_tn_iters: int = 8,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    τ · log( Lip_upper(model) + eps ) — a gentle, numerically stable penalty that
    discourages large products of layer norms. Often redundant if you already use
    Σσ(layer) penalties; keep τ very small if enabled.
    """
    lip = network_lipschitz_upper(
        model,
        conv_mode=conv_mode,
        conv_tn_iters=conv_tn_iters,
        conv_fft_shape=conv_fft_shape,
    )
    return tau * jnp.log(lip + eps)
