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
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, Union, List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

Pad2D = Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]], str]


# ---------- Helper: explicit border count B(H,W,k,s,d) ----------
def _ceil_div(a: int, b: int) -> int:
    b = max(1, int(b))
    return (int(a) + b - 1) // b


def _border_B(H, W, kH, kW, sH=1, sW=1, dH=1, dW=1, *, padding: str = "SAME") -> int:
    eH = (kH - 1) * dH + 1
    eW = (kW - 1) * dW + 1
    if padding.upper() == "SAME":
        Hout = _ceil_div(H, sH)
        Wout = _ceil_div(W, sW)
    else:
        if H < eH or W < eW:
            return 0
        Hout = (H - eH) // sH + 1
        Wout = (W - eW) // sW + 1
    tH = _ceil_div(max(0, eH - 1), sH)
    tW = _ceil_div(max(0, eW - 1), sW)
    return int(max(0, 2 * Wout * tH + 2 * Hout * tW - 4 * tH * tW))


# ---------- ACE (Adaptive Circulant Embedding) UB ----------
def _sigma_conv_circ_embed_opt_upper_2d(
    K: jnp.ndarray,
    in_hw: Tuple[int, int],
    *,
    rhs_dilation: Tuple[int, int] = (1, 1),
    candidates: Optional[Sequence[Tuple[int, int]]] = None,
) -> jnp.ndarray:
    """
    Certified UB via inf over admissible circulant embeddings (ACE).
    JIT-safe: all shape math uses pure Python integers.
    """
    H, W = int(in_hw[0]), int(in_hw[1])
    Co, Ci, kH, kW = int(K.shape[0]), int(K.shape[1]), int(K.shape[2]), int(K.shape[3])
    dH, dW = int(rhs_dilation[0]), int(rhs_dilation[1])

    eH = (kH - 1) * dH + 1
    eW = (kW - 1) * dW + 1

    Hc_min = H + eH - 1
    Wc_min = W + eW - 1

    def next_pow2_int(x: int) -> int:
        # Pure Python: avoids tracers; works under jit
        return 1 if x <= 1 else 1 << ((x - 1).bit_length())

    if candidates is None:
        cand: List[Tuple[int, int]] = [
            (Hc_min, Wc_min),
            (Hc_min + (eH - 1), Wc_min + (eW - 1)),
            (next_pow2_int(Hc_min), next_pow2_int(Wc_min)),
        ]
    else:
        # Ensure Python ints
        cand = [(int(a), int(b)) for (a, b) in candidates]

    best = jnp.asarray(jnp.inf, jnp.float32)
    for Hc, Wc in cand:
        # dilate kernel on the torus (pure JAX arrays; shapes are Python ints)
        Kd = jnp.zeros((Co, Ci, eH, eW), dtype=K.dtype)
        Kd = Kd.at[..., ::dH, ::dW].set(K)

        # unitary FFT; s uses Python ints → static shapes
        Kf = jnp.fft.rfft2(Kd, s=(Hc, Wc), axes=(-2, -1), norm="ortho")
        mats = jnp.transpose(Kf, (2, 3, 0, 1)).reshape(-1, Co, Ci)
        smax = jax.vmap(lambda M: jnp.linalg.svd(M, compute_uv=False)[0])(mats)
        ub = jnp.sqrt(Hc * Wc) * jnp.max(jnp.real(smax))
        best = jnp.minimum(best, ub.astype(jnp.float32))

    return best


# ---------- Improved C+R UB (unitary FFT) ----------
def _sigma_conv_circ_plus_lr_upper_2d(
    K: jnp.ndarray,
    in_hw: Tuple[int, int],
    *,
    strides: Tuple[int, int] = (1, 1),
    rhs_dilation: Tuple[int, int] = (1, 1),
) -> jnp.ndarray:
    """Certified UB: ||T|| <= ||C|| + sqrt(B) ||K||_F with explicit B(H,W,k,s,d)."""
    H, W = map(int, in_hw)
    Co, Ci, kH, kW = map(int, K.shape)
    sH, sW = map(int, strides)
    dH, dW = map(int, rhs_dilation)

    eH = (kH - 1) * dH + 1
    eW = (kW - 1) * dW + 1
    Hc, Wc = H + eH - 1, W + eW - 1

    # exact circular on (Hc,Wc)
    Kd = jnp.zeros((Co, Ci, eH, eW), dtype=K.dtype)
    Kd = Kd.at[..., ::dH, ::dW].set(K)
    Kf = jnp.fft.rfft2(Kd, s=(Hc, Wc), axes=(-2, -1), norm="ortho")
    mats = jnp.transpose(Kf, (2, 3, 0, 1)).reshape(-1, Co, Ci)
    smax = jax.vmap(lambda M: jnp.linalg.svd(M, compute_uv=False)[0])(mats)
    circ = jnp.sqrt(Hc * Wc) * jnp.max(jnp.real(smax))

    B = _border_B(H, W, kH, kW, sH, sW, dH, dW)
    frob = jnp.linalg.norm(K)
    return (circ + jnp.sqrt(jnp.asarray(B, K.dtype)) * frob).astype(jnp.float32)


# ---------- (Optional) per-layer bracket via power-method LB ----------
def _conv2d_apply(x, K, stride=(1, 1), dilation=(1, 1), padding="VALID"):
    # x: (N, Cin, H, W), K: (Cout, Cin, kH, kW)
    dn = jax.lax.conv_dimension_numbers(
        lhs_shape=x.shape, rhs_shape=K.shape, dimension_numbers=("NCHW", "OIHW", "NCHW")
    )
    return jax.lax.conv_general_dilated(
        lhs=x,
        rhs=K,
        window_strides=tuple(map(int, stride)),
        padding=padding,  # "VALID"/"SAME" or ((pt,pb),(pl,pr))
        rhs_dilation=tuple(map(int, dilation)),
        dimension_numbers=dn,
        feature_group_count=1,
    )


# ---------- Transpose conv (adjoint of the above) ----------
def _conv2d_transpose_apply(y, K, stride=(1, 1), dilation=(1, 1), padding="VALID"):
    # y: (N, Cout, H_out, W_out), K: (Cout, Cin, kH, kW)
    dn = jax.lax.conv_dimension_numbers(
        lhs_shape=y.shape, rhs_shape=K.shape, dimension_numbers=("NCHW", "OIHW", "NCHW")
    )
    return jax.lax.conv_transpose(
        y,
        K,  # same OIHW filter
        strides=tuple(map(int, stride)),
        padding=padding,
        rhs_dilation=tuple(map(int, dilation)),
        dimension_numbers=dn,
        transpose_kernel=True,  # let JAX handle kernel transpose
    )


def _conv_power_lower_bound(
    K: jnp.ndarray,
    Cin: int,
    Hin: int,
    Win: int,
    *,
    stride=(1, 1),
    dilation=(1, 1),
    padding="VALID",
    iters=20,
    key=None,
    eps=1e-12,
) -> jnp.ndarray:
    """Monotone LB on ||T|| via power method on T^*T using conv/conv^T."""
    key = jr.PRNGKey(0) if key is None else key
    x = jr.normal(key, (1, Cin, Hin, Win), dtype=K.dtype)
    x = x / (jnp.linalg.norm(x) + eps)
    lb = jnp.array(0.0, dtype=jnp.float32)
    for _ in range(int(iters)):
        y = _conv2d_apply(x, K, stride=stride, dilation=dilation, padding=padding)
        z = _conv2d_transpose_apply(
            y, K, stride=stride, dilation=dilation, padding=padding
        )
        num = jnp.vdot(x, z).real
        den = jnp.vdot(x, x).real + eps
        lb = jnp.sqrt(jnp.maximum(lb**2, (num / den).astype(jnp.float32)))
        x = z / (jnp.linalg.norm(z) + eps)
    return lb


def conv_norm_bracket(
    K: jnp.ndarray,
    Cin: int,
    Hin: int,
    Win: int,
    *,
    strides=(1, 1),
    rhs_dilation=(1, 1),
    in_hw: Tuple[int, int],
    padding: Any = "VALID",
    tol: float = 1e-2,
    max_iters: int = 50,
) -> Tuple[float, float, int]:
    """Return (LB,UB,iters) for one conv layer using power-LB and ACE/C+R UB."""
    # UB = min( ACE , C+R )
    ub_ace = _sigma_conv_circ_embed_opt_upper_2d(K, in_hw, rhs_dilation=rhs_dilation)
    ub_cr = _sigma_conv_circ_plus_lr_upper_2d(
        K, in_hw, strides=strides, rhs_dilation=rhs_dilation
    )
    ub = jnp.minimum(ub_ace, ub_cr)
    lb = _conv_power_lower_bound(
        K,
        Cin,
        Hin,
        Win,
        stride=strides,
        dilation=rhs_dilation,
        padding=padding,
        iters=1,
    )
    it = 1
    while (ub / jnp.maximum(lb, 1e-12) > (1.0 + tol)) and (it < max_iters):
        it *= 2
        lb = _conv_power_lower_bound(
            K,
            Cin,
            Hin,
            Win,
            stride=strides,
            dilation=rhs_dilation,
            padding=padding,
            iters=it,
        )
    return float(lb), float(ub), int(it)


def _is_trainable_array(x, *, include_bias: bool = False):
    """Heuristic used by Frobenius penalty:
    - includes inexact arrays with ndim >= 2 (default)
    - if include_bias=True, also includes 0D/1D arrays (biases / vector params)
    - excludes things marked as `_frozen`
    """
    return (
        eqx.is_inexact_array(x)
        and (getattr(x, "_frozen", False) is False)
        and (include_bias or x.ndim >= 2)
    )


def _sigma_conv_circulant_embed_upper_2d(
    K: jnp.ndarray,  # (Cout, Cin, kH, kW)
    in_hw: Tuple[int, int],  # (H_in, W_in) seen by this layer
) -> jnp.ndarray:
    """
    Upper bound for zero/reflect-like convs via circulant embedding on the
    grid (H_in+kH−1, W_in+kW−1). Uses unitary FFT; multiply by √(area).
    """
    Co, Ci, kH, kW = map(int, K.shape)
    H_in, W_in = map(int, in_hw)
    Hf, Wf = H_in + kH - 1, W_in + kW - 1

    Kf = jnp.fft.rfft2(K, s=(Hf, Wf), axes=(-2, -1), norm="ortho")
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


def global_frobenius_penalty(model: Any, *, include_bias: bool = False) -> jnp.ndarray:
    """
    ∑ ||p||² over trainable arrays.

    Default (include_bias=False): only ndim ≥ 2 (classic weight decay on matrices/kernels).
    If include_bias=True: also includes 0D/1D arrays (biases, vector params like w in convex heads).
    """
    params = eqx.filter(
        model, lambda x: _is_trainable_array(x, include_bias=include_bias)
    )
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
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = "tn",
    conv_tn_iters: int = 10,
    conv_gram_iters: int = 5,
    conv_fft_shape: Optional[Tuple[int, int]] = None,
    conv_input_shape: Optional[Tuple[int, int]] = None,
) -> jnp.ndarray:
    """
    Σ of per-layer operator norms across the model (dense exact; conv via certified certificates).
    Certified-only version: no kernel_flat; non-2D convs are not supported and will raise.

    Conv certificates:
      - "tn": stride/dilation-aware, resolution-free UB
      - "circular_fft": exact for circular stride=1 (any dilation); requires conv_fft_shape
      - "circular_gram": circular stride=1 UB (any dilation); requires conv_fft_shape
      - "min_tn_circ_embed": min(TN, circulant-embed UB); requires conv_input_shape; stride=1, no dilation
      - "circ_plus_lr": exact circular + low-rank border UB; requires conv_input_shape; any stride/dilation
    """
    total = jnp.array(0.0, dtype=jnp.float32)
    nn = getattr(eqx, "nn", None)
    PARAM_SVD_CONV = {"SpectralConv2d", "AdaptiveSpectralConv2d"}

    def _svd_top_sigma_flat(W: jnp.ndarray) -> jnp.ndarray:
        W2 = W.reshape(W.shape[0], -1)
        return jnp.linalg.svd(W2, compute_uv=False)[0].astype(jnp.float32)

    def _canonize_K(module, K: jnp.ndarray) -> jnp.ndarray:
        outc = getattr(module, "out_channels", None)
        inc = getattr(module, "in_channels", None)
        if outc is not None and inc is not None:
            if K.shape[0] == outc and K.shape[1] == inc:
                return K
            if K.shape[0] == inc and K.shape[1] == outc:
                return jnp.swapaxes(K, 0, 1)
        if "ConvTranspose" in type(module).__name__:
            return jnp.swapaxes(K, 0, 1)
        return K

    def visit(x, acc):
        name = type(x).__name__

        # Prefer exact/certified operator-norm hints from layers that expose them.
        if (
            ("spectral" in apply_to)
            and hasattr(x, "__operator_norm_hint__")
            and name not in PARAM_SVD_CONV
        ):
            try:
                val = x.__operator_norm_hint__()
                if val is not None:
                    return acc + jnp.asarray(val, jnp.float32)
            except Exception:
                pass

        # SpectralNorm wrapper: certified cap
        if name == "SpectralNorm" and "spectral" in apply_to:
            tgt = getattr(x, "target", 1.0)
            return acc + jnp.asarray(tgt, jnp.float32)

        # Dense (exact)
        if nn is not None and isinstance(x, getattr(nn, "Linear", ())):
            if "linear" in apply_to and hasattr(x, "weight"):
                return acc + _svd_top_sigma_flat(x.weight)

        # Convolution / ConvTranspose (2D only; certified)
        if nn is not None and isinstance(
            x,
            tuple(
                t
                for t in (getattr(nn, "Conv", None), getattr(nn, "ConvTranspose", None))
                if t is not None
            ),
        ):
            if "conv" in apply_to and hasattr(x, "weight"):
                K = x.weight
                num_dims = getattr(x, "num_spatial_dims", K.ndim - 2)
                if int(num_dims) != 2:
                    raise ValueError(
                        "Certified spectral-norm penalty supports only 2D convs."
                    )
                # stride/dilation
                s = getattr(x, "stride", getattr(x, "strides", (1, 1)))
                s = (s, s) if isinstance(s, int) else s
                d = (
                    getattr(x, "rhs_dilation", None)
                    or getattr(x, "dilation", None)
                    or getattr(x, "kernel_dilation", (1, 1))
                )
                d = (d, d) if isinstance(d, int) else d

                # 2D path via certified certificate
                K = _canonize_K(x, K)
                sh, sw = int(s[0]), int(s[1])
                dh, dw = int(d[0]), int(d[1])

                if conv_mode == "circular_fft":
                    if (sh, sw) == (1, 1) and (conv_fft_shape is not None):
                        return acc + _sigma_conv_circular_fft_exact_2d(
                            K, conv_fft_shape, rhs_dilation=(dh, dw)
                        )
                    return acc + _sigma_conv_TN_upper(
                        K, strides=(sh, sw), rhs_dilation=(dh, dw), iters=conv_tn_iters
                    )

                if conv_mode == "circ_embed_opt":
                    if conv_input_shape is None:
                        raise ValueError(
                            "circ_embed_opt requires conv_input_shape=(H,W)."
                        )
                    return acc + _sigma_conv_circ_embed_opt_upper_2d(
                        K, in_hw=conv_input_shape, rhs_dilation=(dh, dw)
                    )

                if conv_mode == "circular_gram":
                    if (sh, sw) == (1, 1) and (conv_fft_shape is not None):
                        return acc + _sigma_conv_circular_gram_upper_2d(
                            K,
                            conv_fft_shape,
                            rhs_dilation=(dh, dw),
                            iters=conv_gram_iters,
                        )
                    return acc + _sigma_conv_TN_upper(
                        K, strides=(sh, sw), rhs_dilation=(dh, dw), iters=conv_tn_iters
                    )

                if conv_mode == "min_tn_circ_embed":
                    if conv_input_shape is None:
                        raise ValueError(
                            "min_tn_circ_embed requires conv_input_shape=(H,W)."
                        )
                    if (sh, sw) == (1, 1) and (dh, dw) == (1, 1):
                        tn = _sigma_conv_TN_upper(
                            K, strides=(1, 1), rhs_dilation=(1, 1), iters=conv_tn_iters
                        )
                        ce = _sigma_conv_circulant_embed_upper_2d(
                            K, in_hw=conv_input_shape
                        )
                        return acc + jnp.minimum(tn, ce)
                    return acc + _sigma_conv_TN_upper(
                        K, strides=(sh, sw), rhs_dilation=(dh, dw), iters=conv_tn_iters
                    )

                if conv_mode == "circ_plus_lr":
                    if conv_input_shape is None:
                        raise ValueError(
                            "circ_plus_lr requires conv_input_shape=(H,W)."
                        )
                    return acc + _sigma_conv_circ_plus_lr_upper_2d(
                        K,
                        in_hw=conv_input_shape,
                        strides=(sh, sw),
                        rhs_dilation=(dh, dw),
                    )

                # default: TN
                return acc + _sigma_conv_TN_upper(
                    K, strides=(sh, sw), rhs_dilation=(dh, dw), iters=conv_tn_iters
                )

        # Parametric SVD-convs: reconstruct kernel then apply certified conv bound
        if name in PARAM_SVD_CONV and "conv" in apply_to:
            W_mat = x.U @ (x.s[:, None] * x.V.T)
            W = jnp.reshape(W_mat, (x.C_out, x.C_in, x.H_k, x.W_k))
            s = getattr(x, "strides", (1, 1))
            s = (s, s) if isinstance(s, int) else s
            sh, sw = int(s[0]), int(s[1])

            if conv_mode == "circular_fft":
                if (sh, sw) == (1, 1) and (conv_fft_shape is not None):
                    return acc + _sigma_conv_circular_fft_exact_2d(
                        W, conv_fft_shape, rhs_dilation=(1, 1)
                    )
                return acc + _sigma_conv_TN_upper(
                    W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                )

            if conv_mode == "circular_gram":
                if (sh, sw) == (1, 1) and (conv_fft_shape is not None):
                    return acc + _sigma_conv_circular_gram_upper_2d(
                        W, conv_fft_shape, rhs_dilation=(1, 1), iters=conv_gram_iters
                    )
                return acc + _sigma_conv_TN_upper(
                    W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                )

            if conv_mode == "min_tn_circ_embed":
                if conv_input_shape is None:
                    raise ValueError(
                        "min_tn_circ_embed requires conv_input_shape=(H,W)."
                    )
                if (sh, sw) == (1, 1):
                    tn = _sigma_conv_TN_upper(
                        W, strides=(1, 1), rhs_dilation=(1, 1), iters=conv_tn_iters
                    )
                    ce = _sigma_conv_circulant_embed_upper_2d(W, in_hw=conv_input_shape)
                    return acc + jnp.minimum(tn, ce)
                return acc + _sigma_conv_TN_upper(
                    W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
                )

            if conv_mode == "circ_plus_lr":
                if conv_input_shape is None:
                    raise ValueError("circ_plus_lr requires conv_input_shape=(H,W).")
                return acc + _sigma_conv_circ_plus_lr_upper_2d(
                    W, in_hw=conv_input_shape, strides=(sh, sw), rhs_dilation=(1, 1)
                )

            return acc + _sigma_conv_TN_upper(
                W, strides=(sh, sw), rhs_dilation=(1, 1), iters=conv_tn_iters
            )

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
    ∑ over 2D conv layers of  √(h*w)·||K||σ  /  (||K||F + eps)   (Grishina §6.3).
    Certified-only: raises if non-2D convs are present.
    """
    vals = []

    def visit(x):
        if hasattr(eqx.nn, "Conv") and isinstance(x, eqx.nn.Conv):
            K = x.weight
            num_dims = getattr(x, "num_spatial_dims", K.ndim - 2)
            if int(num_dims) != 2:
                raise ValueError("conv_ratio_penalty supports only 2D convs.")
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
    allow_exact_hints_for: Tuple[str, ...] = (
        "RFFTCirculant1D",
        "RFFTCirculant2D",
        "SVDDense",
        "SpectralDense",
    ),
    conv_circ_embed_candidates: Optional[Tuple[Tuple[int, int], ...]] = None,  # << NEW
    return_log: bool = False,
) -> jnp.ndarray:
    """
    Certified global ℓ2 Lipschitz upper bound:
      - Dense: exact SVD
      - Conv2D: certified bound per conv_mode (TN, circular_fft, circular_gram, min_tn_circ_embed, circ_plus_lr)
      - ConvTranspose2D: same as corresponding conv (adjoint equality)
      - Pooling & activations: certified factors
      - BatchNorm: inference-mode running stats
      - Residual: uses __residual_hint__()
      - Concatenation: uses __concat_hint__() → sqrt(sum L_i^2)
    Raises on non-2D convolutions (certified regime).
    """
    LOGDT = jnp.float32
    log_sum = jnp.array(0.0, dtype=LOGDT)
    nn = getattr(eqx, "nn", None)
    PARAM_SVD_CONV = {"SpectralConv2d", "AdaptiveSpectralConv2d"}

    current_hw: Optional[Tuple[int, int]] = (
        tuple(map(int, conv_input_shape)) if conv_input_shape is not None else None
    )

    def add_log(acc, sig):
        sig = jnp.asarray(sig, LOGDT)
        lo = jnp.asarray(1e-12, LOGDT)
        hi = jnp.asarray(1e12, LOGDT)
        return acc + jnp.log(jnp.clip(sig, lo, hi))

    def _pool_out_hw(hw, strides):
        if hw is None:
            return None
        H, W = hw
        sH, sW = strides
        return (int(math.ceil(H / max(1, sH))), int(math.ceil(W / max(1, sW))))

    def _conv_out_hw(hw, k, s, d, padding="SAME"):
        if hw is None:
            return None
        H, W = hw
        sH, sW = s
        return (int(math.ceil(H / max(1, sH))), int(math.ceil(W / max(1, sW))))

    def _convT_out_hw(hw, k, s, d, padding="SAME", output_padding=(0, 0)):
        if hw is None:
            return None
        H, W = hw
        sH, sW = s
        return (int(H * max(1, sH)), int(W * max(1, sW)))

    def _conv_sigma(weight, strides, dilations) -> jnp.ndarray:
        if conv_mode == "circular_fft":
            if strides == (1, 1) and (conv_fft_shape is not None):
                return _sigma_conv_circular_fft_exact_2d(
                    weight, conv_fft_shape, rhs_dilation=dilations
                )
            return _sigma_conv_TN_upper(
                weight, strides=strides, rhs_dilation=dilations, iters=conv_tn_iters
            )

        if conv_mode == "circ_embed_opt":
            if current_hw is None:
                raise ValueError("circ_embed_opt requires conv_input_shape=(H,W).")
            return _sigma_conv_circ_embed_opt_upper_2d(
                weight,
                in_hw=current_hw,
                rhs_dilation=dilations,
                candidates=conv_circ_embed_candidates,  # << pass through
            )
        if conv_mode == "circular_gram":
            if (strides == (1, 1)) and (conv_fft_shape is not None):
                return _sigma_conv_circular_gram_upper_2d(
                    weight,
                    conv_fft_shape,
                    rhs_dilation=dilations,
                    iters=conv_gram_iters,
                )
            return _sigma_conv_TN_upper(
                weight, strides=strides, rhs_dilation=dilations, iters=conv_tn_iters
            )
        if conv_mode == "min_tn_circ_embed":
            if current_hw is None:
                raise ValueError("min_tn_circ_embed requires conv_input_shape=(H,W).")
            if (strides == (1, 1)) and (dilations == (1, 1)):
                tn = _sigma_conv_TN_upper(
                    weight, strides=strides, rhs_dilation=dilations, iters=conv_tn_iters
                )
                ce = _sigma_conv_circulant_embed_upper_2d(weight, in_hw=current_hw)
                return jnp.minimum(tn, ce)
            return _sigma_conv_TN_upper(
                weight, strides=strides, rhs_dilation=dilations, iters=conv_tn_iters
            )
        if conv_mode == "circ_plus_lr":
            if current_hw is None:
                raise ValueError("circ_plus_lr requires conv_input_shape=(H,W).")
            return _sigma_conv_circ_plus_lr_upper_2d(
                weight, in_hw=current_hw, strides=strides, rhs_dilation=dilations
            )
        return _sigma_conv_TN_upper(
            weight, strides=strides, rhs_dilation=dilations, iters=conv_tn_iters
        )

    def visit(x, acc):
        nonlocal current_hw
        name = type(x).__name__

        # Residual (explicit hint)
        if hasattr(x, "__residual_hint__"):
            branch_mods, skip_mods, alpha = x.__residual_hint__()
            l_branch = jnp.array(0.0, dtype=LOGDT)
            for m in branch_mods:
                l_branch = visit(m, l_branch)
            l_skip = jnp.array(0.0, dtype=LOGDT)
            if skip_mods is not None:
                for m in skip_mods:
                    l_skip = visit(m, l_skip)
            return add_log(
                acc,
                jnp.exp(l_skip)
                + jnp.abs(jnp.asarray(alpha, LOGDT)) * jnp.exp(l_branch),
            )

        # Concatenation (explicit hint): L = sqrt(sum_i L_i^2)
        if hasattr(x, "__concat_hint__"):
            parts = x.__concat_hint__()
            logs = [visit(m, jnp.array(0.0, dtype=LOGDT)) for m in parts]
            vals = jnp.stack([jnp.exp(l) for l in logs])
            s = jnp.sqrt(jnp.sum(vals**2))
            return add_log(acc, s)

        # Exact per-layer operator-norm hint
        if (name in allow_exact_hints_for) and hasattr(x, "__operator_norm_hint__"):
            sig = x.__operator_norm_hint__()
            return add_log(acc, sig)

        # Dropout
        if hasattr(nn, "Dropout") and isinstance(x, nn.Dropout):
            # In inference, factor is 1.0
            return add_log(acc, jnp.asarray(1.0, LOGDT))

        # Activations (certified sup |φ'|)
        if name in ("ReLU", "Tanh", "Hardtanh", "Identity"):
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

        # Pooling (ℓ2 bounds)
        if hasattr(nn, "MaxPool2d") and isinstance(x, nn.MaxPool2d):
            kH, kW = _as_2tuple(getattr(x, "kernel_size", 2))
            raw_stride = getattr(x, "stride", (kH, kW)) or (kH, kW)
            sH, sW = _as_2tuple(raw_stride)
            rho = int(math.ceil(kH / max(1, sH))) * int(math.ceil(kW / max(1, sW)))
            acc = add_log(acc, jnp.sqrt(jnp.asarray(rho, LOGDT)))
            current_hw = _pool_out_hw(current_hw, (sH, sW))
            return acc

        if hasattr(nn, "AvgPool2d") and isinstance(x, nn.AvgPool2d):
            kH, kW = _as_2tuple(getattr(x, "kernel_size", 2))
            raw_stride = getattr(x, "stride", (kH, kW)) or (kH, kW)
            sH, sW = _as_2tuple(raw_stride)
            rho = int(math.ceil(kH / max(1, sH))) * int(math.ceil(kW / max(1, sW)))
            denom = max(kH * kW, 1)
            acc = add_log(acc, jnp.sqrt(jnp.asarray(rho / denom, LOGDT)))
            current_hw = _pool_out_hw(current_hw, (sH, sW))
            return acc

        # SpectralNorm wrapper: at most 'target'
        if name == "SpectralNorm":
            tgt = getattr(x, "target", 1.0)
            return add_log(acc, jnp.asarray(tgt, LOGDT))

        # Linear (exact)
        if nn is not None and isinstance(x, getattr(nn, "Linear", ())):
            W = x.weight.reshape(x.weight.shape[0], -1)
            sig = jnp.linalg.svd(W, compute_uv=False)[0]
            return add_log(acc, sig)

        # ConvTranspose2d (2D only)
        if hasattr(nn, "ConvTranspose2d") and isinstance(x, nn.ConvTranspose2d):
            num_dims = getattr(x, "num_spatial_dims", x.weight.ndim - 2)
            if int(num_dims) != 2:
                raise ValueError("Certified bound supports only 2D ConvTranspose.")
            s = getattr(x, "stride", getattr(x, "strides", (1, 1)))
            s = (s, s) if isinstance(s, int) else s
            d = (
                getattr(x, "rhs_dilation", None)
                or getattr(x, "dilation", None)
                or getattr(x, "kernel_dilation", (1, 1))
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
            current_hw = _convT_out_hw(
                current_hw,
                (int(W.shape[-2]), int(W.shape[-1])),
                (int(s[0]), int(s[1])),
                (int(d[0]), int(d[1])),
            )
            return acc

        # Conv2D (certified)
        if nn is not None and isinstance(x, getattr(nn, "Conv", ())):
            num_dims = getattr(x, "num_spatial_dims", x.weight.ndim - 2)
            if int(num_dims) != 2:
                raise ValueError("Certified bound supports only 2D convs.")
            s = getattr(x, "stride", getattr(x, "strides", (1, 1)))
            s = (s, s) if isinstance(s, int) else s
            d = (
                getattr(x, "rhs_dilation", None)
                or getattr(x, "dilation", None)
                or getattr(x, "kernel_dilation", (1, 1))
            )
            d = (d, d) if isinstance(d, int) else d
            sig = _conv_sigma(x.weight, (int(s[0]), int(s[1])), (int(d[0]), int(d[1])))
            acc = add_log(acc, sig)
            current_hw = _conv_out_hw(
                current_hw,
                (int(x.weight.shape[-2]), int(x.weight.shape[-1])),
                (int(s[0]), int(s[1])),
                (int(d[0]), int(d[1])),
            )
            return acc

        # Param-SVD convs (reconstruct kernel)
        if name in PARAM_SVD_CONV:
            W_mat = x.U @ (x.s[:, None] * x.V.T)
            W = jnp.reshape(W_mat, (x.C_out, x.C_in, x.H_k, x.W_k))
            s = getattr(x, "strides", (1, 1))
            s = (s, s) if isinstance(s, int) else s
            sig = _conv_sigma(W, (int(s[0]), int(s[1])), (1, 1))
            acc = add_log(acc, sig)
            current_hw = _conv_out_hw(
                current_hw,
                (int(W.shape[-2]), int(W.shape[-1])),
                (int(s[0]), int(s[1])),
                (1, 1),
            )
            return acc

        # Normalization layers
        if hasattr(nn, "BatchNorm") and isinstance(x, nn.BatchNorm):
            eps = jnp.asarray(getattr(x, "eps", 1e-5), LOGDT)
            gamma = (
                jnp.abs(x.weight)
                if getattr(x, "channelwise_affine", True) and (x.weight is not None)
                else jnp.array(1.0, dtype=LOGDT)
            )
            if (
                state is not None
                and x.mode == "ema"
                and (x.ema_state_index is not None)
            ):
                _, var = state.get(x.ema_state_index)
                s_val = jnp.max(gamma / jnp.sqrt(var + eps))
                return add_log(acc, s_val)
            s_val = jnp.max(gamma) / jnp.sqrt(eps)
            return add_log(acc, s_val)

        for name_norm in ("LayerNorm", "GroupNorm", "RMSNorm"):
            if hasattr(nn, name_norm) and isinstance(x, getattr(nn, name_norm)):
                eps = jnp.asarray(getattr(x, "eps", 1e-5), LOGDT)
                if getattr(x, "use_weight", True) and (
                    getattr(x, "weight", None) is not None
                ):
                    s_val = jnp.max(jnp.abs(x.weight)) / jnp.sqrt(eps)
                else:
                    s_val = 1.0 / jnp.sqrt(eps)
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

    ls = visit(model, log_sum)
    if return_log:
        return ls
    max_log = jnp.log(jnp.asarray(jnp.finfo(LOGDT).max, LOGDT)) - jnp.asarray(
        1.0, LOGDT
    )
    ls = jnp.minimum(ls, max_log)
    return jnp.exp(ls)


def lip_product_penalty(
    model: Any,
    *,
    state: Optional[Any] = None,
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
