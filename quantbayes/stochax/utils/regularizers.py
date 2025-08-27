# quantbayes/stochax/utils/regularizers.py
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
import math
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


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


def _tensor_sigma_upper_via_unfoldings(A: jnp.ndarray) -> jnp.ndarray:
    """Certified UB on ||A||_σ via unfoldings (Fantastic-Four style)."""
    assert A.ndim == 4
    a, b, c, d = A.shape
    # (1;234)
    M1 = A.reshape(a, b * c * d)
    # (2;134)
    M2 = jnp.moveaxis(A, 1, 0).reshape(b, a * c * d)
    # (3;124)
    M3 = jnp.moveaxis(A, 2, 0).reshape(c, a * b * d)
    # (4;123)
    M4 = jnp.moveaxis(A, 3, 0).reshape(d, a * b * c)

    def top_sigma(M):
        return jnp.linalg.svd(M, compute_uv=False)[0].astype(jnp.float32)

    return jnp.min(
        jnp.stack([top_sigma(M1), top_sigma(M2), top_sigma(M3), top_sigma(M4)])
    )


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


def _as_2tuple(v):
    """Return (h, w) from an int, (h,w), or ((hl,hr),(wl,wr)) padding-style tuple."""
    if isinstance(v, tuple):
        if len(v) == 2 and all(isinstance(x, tuple) for x in v):
            (hl, hr), (wl, wr) = v
            return int(hl + hr), int(wl + wr)
        if len(v) == 2:
            return int(v[0]), int(v[1])
    try:
        i = int(v)
        return i, i
    except Exception:
        return 1, 1


def _conv_out_hw(
    hw: Optional[Tuple[int, int]],
    k: Tuple[int, int],
    s: Tuple[int, int],
    d: Tuple[int, int],
    padding: Union[
        str, int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]
    ] = "SAME",
) -> Optional[Tuple[int, int]]:
    """Output (H,W) of a forward conv given input (H,W), kernel (kH,kW),
    strides (sH,sW), dilations (dH,dW), and padding.
    Returns None if hw is None. Uses the usual floor/ceil conventions.
    """
    if hw is None:
        return None
    H, W = map(int, hw)
    kH, kW = map(int, k)
    sH, sW = map(int, s)
    dH, dW = map(int, d)
    He = (kH - 1) * dH + 1
    We = (kW - 1) * dW + 1

    # Normalize padding → (PH, PW) total padding added along each spatial dim.
    if isinstance(padding, (str, type(None))) or padding is None:
        # "SAME" behavior: output is ceil(H/s)
        Hout = int(math.ceil(H / max(1, sH)))
        Wout = int(math.ceil(W / max(1, sW)))
        return (Hout, Wout)

    if isinstance(padding, tuple):
        if len(padding) == 2 and all(isinstance(x, tuple) for x in padding):
            (hl, hr), (wl, wr) = padding
            PH, PW = int(hl + hr), int(wl + wr)
        else:
            PH, PW = map(int, padding)
    else:
        PH = PW = int(padding)

    Hout = (H + PH - He) // max(1, sH) + 1
    Wout = (W + PW - We) // max(1, sW) + 1
    return (max(0, Hout), max(0, Wout))


def _convT_out_hw(
    hw: Optional[Tuple[int, int]],
    k: Tuple[int, int],
    s: Tuple[int, int],
    d: Tuple[int, int],
    padding: Union[
        str, int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]
    ] = "SAME",
    output_padding: Tuple[int, int] = (0, 0),
) -> Optional[Tuple[int, int]]:
    """Output (H,W) for a ConvTranspose (PyTorch-style formula with output_padding).
    Falls back to SAME-like growth if padding is string/None. Returns None if hw is None.
    """
    if hw is None:
        return None
    H, W = map(int, hw)
    kH, kW = map(int, k)
    sH, sW = map(int, s)
    dH, dW = map(int, d)
    oH, oW = map(int, output_padding)

    if isinstance(padding, (str, type(None))) or padding is None:
        # SAME-like: enlarge roughly by stride
        Hout = (H - 1) * sH + 1
        Wout = (W - 1) * sW + 1
        return (Hout, Wout)

    if isinstance(padding, tuple):
        if len(padding) == 2 and all(isinstance(x, tuple) for x in padding):
            (pl, pr), (ql, qr) = padding
            PH, PW = int(pl + pr), int(ql + qr)
        else:
            PH, PW = map(int, padding)
    else:
        PH = PW = int(padding)

    He = (kH - 1) * dH + 1
    We = (kW - 1) * dW + 1
    Hout = (H - 1) * sH - PH + He + oH
    Wout = (W - 1) * sW - PW + We + oW
    return (max(1, Hout), max(1, Wout))


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
    Bc = 4 * ceil_div_h * ceil_div_w  # corner overlap
    B = jnp.maximum(0, Bh + Bw - Bc)

    L_rem = jnp.sqrt(B.astype(jnp.float32)) * jnp.linalg.norm(K)
    return (L_circ + L_rem).astype(jnp.float32)


def _gram_iter_upper_matrix(
    W: jnp.ndarray, iters: int = 5, eps: float = 1e-12
) -> jnp.ndarray:
    """Delattre-style Gram iteration upper bound on ||W||_2 (matrix)."""
    r = jnp.array(0.0, dtype=W.dtype.real.dtype)
    A = W
    for _ in range(int(iters)):
        nf = jnp.linalg.norm(A) + eps  # Frobenius
        r = 2.0 * (r + jnp.log(nf))
        A = (A / nf).conj().T @ (A / nf)  # Gram map
    # UB after t iterations
    return (jnp.linalg.norm(A) * jnp.exp(r)) ** (
        2.0 ** (-iters)
    )  # Frobenius in both places


def _sigma_conv_circular_gram_upper_2d(
    K: jnp.ndarray,
    fft_shape: Tuple[int, int],
    rhs_dilation: Tuple[int, int] = (1, 1),
    iters: int = 5,
) -> jnp.ndarray:
    """
    Delattre-style Gram-iteration UB for CIRCULAR conv (stride=1), any dilation.
    We build the frequency blocks by modulo-scattering the (possibly dilated)
    taps onto the (Hf,Wf) grid, apply a unitary FFT, then upper-bound each
    per-frequency operator norm via Gram iteration. Scale by sqrt(Hf*Wf).
    """
    Co, Ci, kH, kW = map(int, K.shape)
    Hf, Wf = map(int, fft_shape)
    dh, dw = map(int, rhs_dilation)

    # modulo-scatter taps with dilation
    Kgrid = jnp.zeros((Co, Ci, Hf, Wf), dtype=K.dtype)
    for p in range(kH):
        i = (p * dh) % Hf
        for q in range(kW):
            j = (q * dw) % Wf
            Kgrid = Kgrid.at[..., i, j].add(K[..., p, q])

    Kf = jnp.fft.rfft2(Kgrid, s=(Hf, Wf), axes=(-2, -1), norm="ortho")
    mats = jnp.transpose(Kf, (2, 3, 0, 1)).reshape(-1, Co, Ci)  # (freqs, Co, Ci)
    ub_per_freq = jax.vmap(lambda M: _gram_iter_upper_matrix(M, iters=iters))(mats)
    return (jnp.sqrt(Hf * Wf) * jnp.max(ub_per_freq)).astype(jnp.float32)


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
    strides=(1, 1),
    rhs_dilation=(1, 1),
    iters: int = 10,
    key=None,
    certify: bool = True,  # <<< NEW: default to certified
) -> jnp.ndarray:
    Q = _pack_Q_strided(
        K, strides=strides, rhs_dilation=rhs_dilation
    )  # (Co, Ci*sh*sw, Hs, Ws)
    Hs, Ws = int(Q.shape[-2]), int(Q.shape[-1])
    if certify:
        sig_Q = _tensor_sigma_upper_via_unfoldings(Q)  # certified UB on ||Q||_σ
    else:
        sig_Q = _tensor_spectral_norm_K_4d(
            Q, iters=iters, key=key
        )  # fast, not certified
    return jnp.sqrt(Hs * Ws) * sig_Q


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
        "tn",
        "circular_fft",
        "circular_gram",
        "kernel_flat",
        "min_tn_circ_embed",
        "circ_plus_lr",
    ] = "tn",
    conv_tn_iters: int = 10,
    conv_gram_iters: int = 5,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    conv_input_shape: Optional[Tuple[int, int]] = None,
) -> jnp.ndarray:
    """
    Sum of per-layer operator norms (spectral norms) across the model.
    This version threads dilation into circular_gram and leaves other modes intact.
    """
    total = jnp.array(0.0, dtype=jnp.float32)
    nn = getattr(eqx, "nn", None)
    PARAM_SVD_CONV = {"SpectralConv2d", "AdaptiveSpectralConv2d"}

    def _svd_top_sigma_flat(W: jnp.ndarray) -> jnp.ndarray:
        W2 = W.reshape(W.shape[0], -1)
        return jnp.linalg.svd(W2, compute_uv=False)[0].astype(jnp.float32)

    def visit(x, acc):
        name = type(x).__name__

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

        if nn is not None and isinstance(x, getattr(nn, "Linear", ())):
            if "linear" in apply_to and hasattr(x, "weight"):
                return acc + _svd_top_sigma_flat(x.weight)

        if nn is not None and isinstance(x, getattr(nn, "Conv", ())):
            if "conv" in apply_to and hasattr(x, "weight"):
                num_dims = getattr(x, "num_spatial_dims", x.weight.ndim - 2)
                s = getattr(x, "stride", getattr(x, "strides", (1,) * num_dims))
                s = (s, s) if isinstance(s, int) else s
                d = (
                    getattr(x, "rhs_dilation", None)
                    or getattr(x, "dilation", None)
                    or getattr(x, "kernel_dilation", (1,) * num_dims)
                )
                d = (d, d) if isinstance(d, int) else d

                if num_dims != 2:
                    return acc + _sigma_conv_kernel_flat(x.weight, method="svd")

                sh, sw = int(s[0]), int(s[1])
                dh, dw = int(d[0]), int(d[1])

                if conv_mode == "kernel_flat":
                    return acc + _sigma_conv_kernel_flat(x.weight, method="svd")

                elif conv_mode == "circular_fft":
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

                elif conv_mode == "circular_gram":
                    if (sh, sw) == (1, 1) and (conv_fft_shape is not None):
                        return acc + _sigma_conv_circular_gram_upper_2d(
                            x.weight,
                            conv_fft_shape,
                            rhs_dilation=(dh, dw),
                            iters=conv_gram_iters,
                        )
                    return acc + _sigma_conv_TN_upper(
                        x.weight,
                        strides=(sh, sw),
                        rhs_dilation=(dh, dw),
                        iters=conv_tn_iters,
                    )

                elif conv_mode == "min_tn_circ_embed" and conv_input_shape is not None:
                    if (sh, sw) == (1, 1) and (dh, dw) == (1, 1):
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
                    return acc + _sigma_conv_TN_upper(
                        x.weight,
                        strides=(sh, sw),
                        rhs_dilation=(dh, dw),
                        iters=conv_tn_iters,
                    )

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

        if name in PARAM_SVD_CONV and "conv" in apply_to:
            W_mat = x.U @ (x.s[:, None] * x.V.T)
            W = jnp.reshape(W_mat, (x.C_out, x.C_in, x.H_k, x.W_k))
            s = getattr(x, "strides", (1, 1))
            s = (s, s) if isinstance(s, int) else s
            sh, sw = int(s[0]), int(s[1])

            if conv_mode == "kernel_flat":
                return acc + _sigma_conv_kernel_flat(W, method="svd")

            elif conv_mode == "circular_fft":
                if (sh, sw) == (1, 1) and (conv_fft_shape is not None):
                    return acc + _sigma_conv_circular_fft_exact_2d(
                        W, conv_fft_shape, rhs_dilation=(1, 1)
                    )
                return acc + _sigma_conv_TN_upper(
                    W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                )

            elif conv_mode == "circular_gram":
                if (sh, sw) == (1, 1) and (conv_fft_shape is not None):
                    return acc + _sigma_conv_circular_gram_upper_2d(
                        W, conv_fft_shape, rhs_dilation=(1, 1), iters=conv_gram_iters
                    )
                return acc + _sigma_conv_TN_upper(
                    W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                )

            elif conv_mode == "min_tn_circ_embed" and conv_input_shape is not None:
                if (sh, sw) == (1, 1):
                    tn = _sigma_conv_TN_upper(
                        W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                    )
                    ce = _sigma_conv_circulant_embed_upper_2d(W, in_hw=conv_input_shape)
                    return acc + jnp.minimum(tn, ce)
                return acc + _sigma_conv_TN_upper(
                    W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                )

            elif conv_mode == "circ_plus_lr" and conv_input_shape is not None:
                return acc + _sigma_conv_circ_plus_lr_upper_2d(
                    W, in_hw=conv_input_shape, strides=(sh, sw), rhs_dilation=(1, 1)
                )

            else:
                return acc + _sigma_conv_TN_upper(
                    W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                )

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
                # M: (Co, Ci) block at a given frequency for √(HW)·K̂
                s = jnp.linalg.svd(
                    M, compute_uv=False
                )  # singular values (length = min(Co, Ci))
                dev = jnp.max(
                    jnp.abs(s**2 - 1.0)
                )  # eigenvalue deviations of M^*M - I_Ci (on its range)
                # If Ci > Co, K^*K has (Ci - rank) zeros -> (K^*K - I) has -1 eigenvalues. Account for that.
                dev = jnp.where(M.shape[1] > M.shape[0], jnp.maximum(dev, 1.0), dev)
                return dev

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
    state: Optional[Any] = None,
    conv_mode: Literal[
        "tn", "circular_fft", "circular_gram", "min_tn_circ_embed", "circ_plus_lr"
    ] = "tn",
    conv_tn_iters: int = 8,
    conv_gram_iters: int = 5,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    conv_input_shape: Optional[Tuple[int, int]] = None,
    allow_exact_hints_for: Tuple[str, ...] = (
        "RFFTCirculant1D",
        "RFFTCirculant2D",
        "SVDDense",
        "SpectralDense",
    ),
    resize_l2_factor: Optional[float] = None,
    return_log: bool = False,  # <<< new (useful for debugging / analysis)
) -> jnp.ndarray:
    """
    Product upper bound on Lip(f) under ℓ₂ using per-layer/operator norms.
    Numerically robust: accumulates in float64 log-space and clamps before exp.
    Grid-aware: carries a per-layer (H,W) when `conv_input_shape` is provided,
    which tightens CE / C+R bounds.
    """
    LOGDT = jnp.float64
    log_sum = jnp.array(0.0, dtype=LOGDT)
    nn = getattr(eqx, "nn", None)
    PARAM_SVD_CONV = {"SpectralConv2d", "AdaptiveSpectralConv2d"}

    # ---- per-layer (H,W) propagation for grid-aware modes ----
    current_hw: Optional[Tuple[int, int]] = (
        tuple(map(int, conv_input_shape)) if conv_input_shape is not None else None
    )

    def add_log(acc, sig):
        sig = jnp.asarray(sig, LOGDT)
        tiny = jnp.finfo(LOGDT).tiny
        huge = jnp.finfo(LOGDT).max
        return acc + jnp.log(jnp.clip(sig, tiny, huge))

    def _conv_sigma(weight, strides, dilations) -> jnp.ndarray:
        if conv_mode == "circular_fft":
            if strides == (1, 1) and (conv_fft_shape is not None):
                return _sigma_conv_circular_fft_exact_2d(
                    weight, conv_fft_shape, rhs_dilation=dilations
                )
            return _sigma_conv_TN_upper(
                weight, strides=strides, rhs_dilation=dilations, iters=conv_tn_iters
            )

        if conv_mode == "circular_gram":
            if strides == (1, 1) and (conv_fft_shape is not None):
                return _sigma_conv_circular_gram_upper_2d(
                    weight,
                    conv_fft_shape,
                    rhs_dilation=dilations,
                    iters=conv_gram_iters,
                )
            return _sigma_conv_TN_upper(
                weight, strides=strides, rhs_dilation=dilations, iters=conv_tn_iters
            )

        if conv_mode == "min_tn_circ_embed" and (current_hw is not None):
            if strides == (1, 1) and dilations == (1, 1):
                tn = _sigma_conv_TN_upper(
                    weight, strides=strides, rhs_dilation=dilations, iters=conv_tn_iters
                )
                ce = _sigma_conv_circulant_embed_upper_2d(weight, in_hw=current_hw)
                return jnp.minimum(tn, ce)
            return _sigma_conv_TN_upper(
                weight, strides=strides, rhs_dilation=dilations, iters=conv_tn_iters
            )

        if conv_mode == "circ_plus_lr" and (current_hw is not None):
            return _sigma_conv_circ_plus_lr_upper_2d(
                weight, in_hw=current_hw, strides=strides, rhs_dilation=dilations
            )

        # default: TN UB (stride/dilation-aware, size-free)
        return _sigma_conv_TN_upper(
            weight, strides=strides, rhs_dilation=dilations, iters=conv_tn_iters
        )

    def _lip_seq(mods: Sequence[Any]) -> jnp.ndarray:
        lacc = jnp.array(0.0, dtype=LOGDT)
        for m in mods:
            if m is None:
                continue
            lacc = visit(m, lacc)
        return lacc

    def visit(x, acc):
        nonlocal current_hw  # we advance it as we traverse the graph
        name = type(x).__name__

        # ---------- UNet backbone (kept from your original; grid updates flow through visit) ----------
        if (
            name == "UNetBackbone"
            and hasattr(x, "encoder")
            and all(hasattr(x, a) for a in ("d1", "d2", "d3", "d4", "out_conv"))
        ):
            enc = x.encoder
            L_conv1_bn = _lip_seq([enc.conv1, enc.bn1])
            acc = acc + L_conv1_bn
            acc = visit(enc.pool, acc)  # pool factor + grid update

            def L_layer(layer_tuple):
                return _lip_seq(list(layer_tuple))

            acc = acc + L_layer(enc.layers1)
            acc = acc + L_layer(enc.layers2)
            acc = acc + L_layer(enc.layers3)
            acc = acc + L_layer(enc.layers4)

            # Decoder stages (up -> concat -> conv)
            def up_stage(up_mod):
                l_before = visit(up_mod.up, jnp.array(0.0, dtype=LOGDT))
                l_after = visit(up_mod.conv, jnp.array(0.0, dtype=LOGDT))
                # concat: sqrt(a^2 + b^2); we just add log after computing both branches upstream
                return l_before + l_after

            acc = acc + up_stage(x.d1)
            acc = acc + up_stage(x.d2)
            acc = acc + up_stage(x.d3)
            acc = acc + up_stage(x.d4)
            acc = visit(x.out_conv, acc)

            if resize_l2_factor is not None:
                acc = add_log(acc, jnp.asarray(resize_l2_factor, LOGDT))
            return acc

        # ---------- Residual blocks via explicit hints ----------
        if hasattr(x, "__residual_hint__"):
            try:
                branch_mods, skip_mods, alpha = x.__residual_hint__()
                l_branch = _lip_seq(branch_mods)
                l_skip = (
                    _lip_seq(skip_mods)
                    if skip_mods is not None
                    else jnp.array(0.0, dtype=LOGDT)
                )
                return add_log(
                    acc,
                    jnp.exp(l_skip)
                    + jnp.abs(jnp.asarray(alpha, LOGDT)) * jnp.exp(l_branch),
                )
            except Exception:
                pass

        # ---------- Spectral dense parametrizations ----------
        if name in ("SVDDense", "SpectralDense") and hasattr(x, "s"):
            alpha = jnp.asarray(getattr(x, "alpha", 1.0), LOGDT)
            sig = jnp.max(jnp.abs(x.s)) * jnp.abs(alpha)
            return add_log(acc, sig)

        # Exact operator-norm hint for a few safe classes
        if (name in allow_exact_hints_for) and hasattr(x, "__operator_norm_hint__"):
            try:
                sig = x.__operator_norm_hint__()
                return add_log(acc, sig)
            except Exception:
                pass

        # Dropout
        if hasattr(nn, "Dropout") and isinstance(x, nn.Dropout):
            p = jnp.asarray(getattr(x, "p", 0.0), LOGDT)
            q = 1.0 - p
            infer = bool(getattr(x, "inference", False))
            s = jnp.where(infer, jnp.array(1.0, LOGDT), 1.0 / jnp.clip(q, 1e-6, 1.0))
            return add_log(acc, s)

        # 1-Lip activations (module forms)
        if name in ("ReLU", "Tanh", "Softplus", "Identity"):
            return acc
        if name in ("LeakyReLU", "PReLU"):
            slope = float(getattr(x, "negative_slope", getattr(x, "slope", 1.0)))
            return add_log(acc, jnp.asarray(max(1.0, abs(slope)), LOGDT))
        if name == "ELU":
            alpha = float(getattr(x, "alpha", 1.0))
            return add_log(acc, jnp.asarray(max(1.0, alpha), LOGDT))
        if name in ("SiLU", "Swish"):
            return add_log(acc, jnp.asarray(1.10, LOGDT))
        if name == "GELU":
            return add_log(acc, jnp.asarray(1.13, LOGDT))

        # Pooling (update grid too)
        if hasattr(nn, "MaxPool2d") and isinstance(x, nn.MaxPool2d):
            kH, kW = _as_2tuple(getattr(x, "kernel_size", 2))
            raw_stride = getattr(x, "stride", (kH, kW)) or (kH, kW)
            sH, sW = _as_2tuple(raw_stride)
            rho = int(math.ceil(kH / max(1, sH))) * int(math.ceil(kW / max(1, sW)))
            acc = add_log(acc, jnp.sqrt(jnp.asarray(rho, LOGDT)))
            current_hw = _conv_out_hw(current_hw, (kH, kW), (sH, sW), (1, 1), "SAME")
            return acc

        if hasattr(nn, "AvgPool2d") and isinstance(x, nn.AvgPool2d):
            kH, kW = _as_2tuple(getattr(x, "kernel_size", 2))
            raw_stride = getattr(x, "stride", (kH, kW)) or (kH, kW)
            sH, sW = _as_2tuple(raw_stride)
            rho = int(math.ceil(kH / max(1, sH))) * int(math.ceil(kW / max(1, sW)))
            denom = max(kH * kW, 1)
            acc = add_log(acc, jnp.sqrt(jnp.asarray(rho / denom, LOGDT)))
            current_hw = _conv_out_hw(current_hw, (kH, kW), (sH, sW), (1, 1), "SAME")
            return acc

        # SpectralNorm wrapper
        if name == "SpectralNorm":
            return acc

        # Linear (exact)
        if nn is not None and isinstance(x, getattr(nn, "Linear", ())):
            W = x.weight.reshape(x.weight.shape[0], -1)
            sig = jnp.linalg.svd(W, compute_uv=False)[0]
            return add_log(acc, sig)

        # ConvTranspose (adjoint norm = forward norm) + grid update
        is_ct = False
        for ct_name in ("ConvTranspose2d", "ConvTranspose"):
            if hasattr(nn, ct_name) and isinstance(x, getattr(nn, ct_name)):
                is_ct = True
                break
        if is_ct:
            num_dims = getattr(x, "num_spatial_dims", x.weight.ndim - 2)
            s = getattr(x, "stride", getattr(x, "strides", (1,) * num_dims))
            s = (s, s) if isinstance(s, int) else s
            d = (
                getattr(x, "rhs_dilation", None)
                or getattr(x, "dilation", None)
                or getattr(x, "kernel_dilation", (1,) * num_dims)
            )
            d = (d, d) if isinstance(d, int) else d

            W = x.weight
            Wc = (
                jnp.transpose(W, (1, 0, 2, 3))
                if (W.ndim >= 4 and W.shape[0] < W.shape[1])
                else W
            )
            sig = _conv_sigma(Wc, (int(s[0]), int(s[1])), (int(d[0]), int(d[1])))
            acc = add_log(acc, sig)

            # advance grid (convT output size)
            kH, kW = int(W.shape[-2]), int(W.shape[-1])
            padding = getattr(x, "padding", "SAME")
            out_pad = getattr(x, "output_padding", (0, 0))
            out_pad = (out_pad, out_pad) if isinstance(out_pad, int) else out_pad
            current_hw = _convT_out_hw(
                current_hw,
                (kH, kW),
                (int(s[0]), int(s[1])),
                (int(d[0]), int(d[1])),
                padding,
                out_pad,
            )
            return acc

        # Convolution + grid update
        if nn is not None and isinstance(x, getattr(nn, "Conv", ())):
            num_dims = getattr(x, "num_spatial_dims", x.weight.ndim - 2)
            s = getattr(x, "stride", getattr(x, "strides", (1,) * num_dims))
            s = (s, s) if isinstance(s, int) else s
            d = (
                getattr(x, "rhs_dilation", None)
                or getattr(x, "dilation", None)
                or getattr(x, "kernel_dilation", (1,) * num_dims)
            )
            d = (d, d) if isinstance(d, int) else d

            if num_dims != 2:
                sig = _sigma_conv_kernel_flat(x.weight, method="svd")
                acc = add_log(acc, sig)
                return acc

            sig = _conv_sigma(x.weight, (int(s[0]), int(s[1])), (int(d[0]), int(d[1])))
            acc = add_log(acc, sig)

            # advance grid
            kH, kW = int(x.weight.shape[-2]), int(x.weight.shape[-1])
            padding = getattr(x, "padding", "SAME")
            current_hw = _conv_out_hw(
                current_hw,
                (kH, kW),
                (int(s[0]), int(s[1])),
                (int(d[0]), int(d[1])),
                padding,
            )
            return acc

        # Param-SVD convs → reconstruct kernel then treat as conv
        if name in PARAM_SVD_CONV:
            W_mat = x.U @ (x.s[:, None] * x.V.T)
            W = jnp.reshape(W_mat, (x.C_out, x.C_in, x.H_k, x.W_k))
            s = getattr(x, "strides", (1, 1))
            s = (s, s) if isinstance(s, int) else s
            sig = _conv_sigma(W, (int(s[0]), int(s[1])), (1, 1))
            acc = add_log(acc, sig)

            # advance grid
            kH, kW = int(W.shape[-2]), int(W.shape[-1])
            padding = getattr(x, "padding", "SAME")
            current_hw = _conv_out_hw(
                current_hw, (kH, kW), (int(s[0]), int(s[1])), (1, 1), padding
            )
            return acc

        # Normalization layers (unchanged logic, still BN-aware)
        if hasattr(nn, "BatchNorm") and isinstance(x, nn.BatchNorm):
            eps = float(getattr(x, "eps", 1e-5))
            gamma = (
                jnp.abs(x.weight)
                if getattr(x, "channelwise_affine", True) and (x.weight is not None)
                else jnp.array(1.0, dtype=jnp.float32)
            )
            if state is not None:
                if x.mode == "ema" and (x.ema_state_index is not None):
                    _, var = state.get(x.ema_state_index)
                    s_val = jnp.max(gamma / jnp.sqrt(var + eps))
                    return add_log(acc, s_val)
                if (
                    x.mode == "batch"
                    and (x.batch_state_index is not None)
                    and (x.batch_counter is not None)
                ):
                    counter = state.get(x.batch_counter)
                    _, hidden_var = state.get(x.batch_state_index)
                    scale = jnp.where(counter == 0, 1.0, 1.0 - x.momentum**counter)
                    var = hidden_var / scale
                    s_val = jnp.max(gamma / jnp.sqrt(var + eps))
                    return add_log(acc, s_val)
            s_val = jnp.max(gamma) / jnp.sqrt(jnp.asarray(eps, jnp.float32))
            return add_log(acc, s_val)

        for name_norm in ("LayerNorm", "GroupNorm", "RMSNorm"):
            if hasattr(nn, name_norm) and isinstance(x, getattr(nn, name_norm)):
                eps = float(getattr(x, "eps", 1e-5))
                if getattr(x, "use_weight", True) and (
                    getattr(x, "weight", None) is not None
                ):
                    s_val = jnp.max(jnp.abs(x.weight)) / jnp.sqrt(
                        jnp.asarray(eps, jnp.float32)
                    )
                else:
                    s_val = 1.0 / jnp.sqrt(jnp.asarray(eps, jnp.float32))
                return add_log(acc, s_val)

        # Recurse
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

    out_log = visit(model, log_sum)
    if return_log:
        return out_log

    MAXLOG = jnp.log(jnp.finfo(LOGDT).max) - 1.0
    L = jnp.where(out_log > MAXLOG, jnp.inf, jnp.exp(out_log))
    return L.astype(jnp.float32)


def lip_product_penalty(
    model: Any,
    *,
    state: Optional[Any] = None,  # <-- NEW: thread state for BN
    tau: float = 1e-4,
    conv_mode: Literal[
        "tn", "circular_fft", "circular_gram", "min_tn_circ_embed", "circ_plus_lr"
    ] = "tn",
    conv_tn_iters: int = 8,
    conv_gram_iters: int = 5,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    conv_input_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    τ · log( Lip_upper(model) + eps ) — a gentle, numerically stable penalty that
    discourages large products of layer norms.
    """
    lip = network_lipschitz_upper(
        model,
        state=state,
        conv_mode=conv_mode,
        conv_tn_iters=conv_tn_iters,
        conv_gram_iters=conv_gram_iters,
        conv_fft_shape=conv_fft_shape,
        conv_input_shape=conv_input_shape,
    )
    return tau * jnp.log(lip + eps)
