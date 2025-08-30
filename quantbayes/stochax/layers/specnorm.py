# quantbayes/stochax/layers/spectral_norm.py
from __future__ import annotations
from typing import Any, Literal, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import tree_at
from equinox.nn import StatefulLayer
from equinox.nn._misc import named_scope
from equinox.nn._stateful import State, StateIndex

# =============================================================================
# Import your certified/faithful estimators (regularizers.py)
# =============================================================================
# Prefer the stochax path; fall back to a plain quantbayes path if needed.
from quantbayes.stochax.utils.regularizers import (
    _sigma_conv_kernel_flat,
    _sigma_conv_circular_fft_exact_2d,
    _sigma_conv_circular_gram_upper_2d,
    _sigma_conv_circulant_embed_upper_2d,
    _sigma_conv_TN_upper,
    _sigma_conv_circ_plus_lr_upper_2d,
)


# =============================================================================
# Small utilities
# =============================================================================


def _is_name(x, names: Sequence[str]) -> bool:
    return any(type(x).__name__ == n for n in names)


def _as_pair(v) -> Tuple[int, int]:
    if isinstance(v, (tuple, list)):
        assert len(v) == 2
        return int(v[0]), int(v[1])
    iv = int(v)
    return iv, iv


def _flatten_matrix(W: jnp.ndarray) -> jnp.ndarray:
    return W.reshape(W.shape[0], -1)


def _top_sigma_matrix_svd(W2: jnp.ndarray) -> jnp.ndarray:
    # Exact σ_max via SVD (JAX)
    return jnp.linalg.svd(W2, compute_uv=False)[0].astype(jnp.float32)


def _power_iter_step(W: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, eps: float):
    # One bi-directional power-iteration step on a (possibly complex) matrix
    u = W @ v
    u = u / (jnp.linalg.norm(u) + eps)
    v = (W.conj().T if jnp.iscomplexobj(W) else W.T) @ u
    v = v / (jnp.linalg.norm(v) + eps)
    return u, v


def _max_abs(z: jnp.ndarray) -> jnp.ndarray:
    return jnp.max(jnp.abs(z))


# =============================================================================
# The Spectral Normalization wrapper
# =============================================================================


class SpectralNorm(StatefulLayer):
    """
    Research-grade spectral normalization for Equinox modules and your custom spectral layers.

    Per call:
      1) Estimate an operator norm σ̂ for the wrapped layer using a method appropriate
         to that layer (exact / certified UB / PI).
      2) Scale the layer parameters by 1 / scale, where:
            scale = max(1, safety_factor * σ̂ / target)     if mode='clip'
                    max(1e-12, safety_factor * σ̂ / target) if mode='force'
         (Optionally `stop_gradient` through the scale for stability.)

    Supported modules (matching your codebase):
      • eqx.nn.Linear
      • eqx.nn.Conv*, eqx.nn.ConvTranspose*  (2D gets SoTA: 'tn', 'circular_fft',
        'circular_gram', 'kernel_flat', 'min_tn_circ_embed', 'circ_plus_lr')
      • SVDDense, SpectralDense, AdaptiveSpectralDense (exact via max|s|)
      • SpectralConv2d, AdaptiveSpectralConv2d (reconstruct kernel → conv estimator)
      • RFFTCirculant1D (exact: max|H_half|)
      • RFFTCirculant2D (choose 'svd' exact per-frequency SVD or 'pi-sampled')
      • SpectralTokenMixer (exact with gate, +1 if residual enabled)

    Notes
    -----
    • Stateful when using:
        - 'pi_state' for Linear (persistent PI vectors), or
        - 'pi-sampled' for spectral 2D (u,v per sampled frequency).
      Otherwise it is stateless beyond the `inference` flag.
    • For 2D convs:
        - 'circular_fft' is exact for circular padding + stride=1 (any dilation).
        - 'circular_gram' is a tight UB for circular stride=1.
        - 'tn' is the certified UB for general strided/dilated convs.
        - 'min_tn_circ_embed' and 'circ_plus_lr' are grid-aware UBs (require input size).
    • For 1D/3D convs, we conservatively use 'kernel_flat'.
    • This wrapper handles only normalization. If you also want regularization penalties,
      use your `regularizers.py` independently in the loss.

    Parameters
    ----------
    layer : eqx.Module
        Module to wrap.
    target : float
        Desired per-layer Lipschitz cap (default 1.0).
    mode : {'clip','force'}
        'clip'  → no change if σ̂ <= target; otherwise scales down to target.
        'force' → always scales to target (ignoring whether σ̂ <= target).
    detach_scale : bool
        Stop gradients through σ̂ (recommended).
    safety_factor : float
        Cushion to offset underestimation (PI/sampling). Default 1.05.
    inference : bool
        Freeze internal PI updates if True.

    Method selection
    ----------------
    linear_method : {'svd','pi','pi_state'}
    conv2d_method : {'tn','circular_fft','circular_gram','kernel_flat',
                     'min_tn_circ_embed','circ_plus_lr'}
    convND_method : {'kernel_flat','tn'}      # used for Conv1d/Conv3d; default 'kernel_flat'
    param_svd_conv2d_method : same set as conv2d_method
    spectral2d_method : {'svd','pi-sampled'}  # for RFFTCirculant2D & similar

    Conv settings
    -------------
    conv_tn_iters : int
    conv_tn_certified : bool                # pass-through to _sigma_conv_TN_upper
    conv_gram_iters : int
    conv_fft_shape : Optional[(Hf,Wf)]      # for 'circular_fft'/'circular_gram'
    conv_input_shape : Optional[(Hin,Win)]  # for 'min_tn_circ_embed'/'circ_plus_lr'

    PI/sampling settings
    --------------------
    num_power_iterations : int     # Linear PI & spectral 2D PI-sampled
    num_freq_samples : Optional[int]  # None = all half-plane frequencies
    eps : float
    key : PRNGKey (for state init)
    """

    # Wrapped layer & global knobs
    layer: eqx.Module
    target: float = eqx.field(static=True)
    mode: Literal["clip", "force"] = eqx.field(static=True)
    detach_scale: bool = eqx.field(static=True)
    safety_factor: float = eqx.field(static=True)
    inference: bool

    # Method selection
    linear_method: Literal["svd", "pi", "pi_state"] = eqx.field(static=True)
    conv2d_method: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "kernel_flat",
        "min_tn_circ_embed",
        "circ_plus_lr",
    ] = eqx.field(static=True)
    convND_method: Literal["kernel_flat", "tn"] = eqx.field(static=True)
    param_svd_conv2d_method: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "kernel_flat",
        "min_tn_circ_embed",
        "circ_plus_lr",
    ] = eqx.field(static=True)
    spectral2d_method: Literal["svd", "pi-sampled"] = eqx.field(static=True)

    # Conv hyper-params
    conv_tn_iters: int = eqx.field(static=True)
    conv_tn_certified: bool = eqx.field(static=True)
    conv_gram_iters: int = eqx.field(static=True)
    conv_fft_shape: Optional[Tuple[int, int]] = eqx.field(static=True)
    conv_input_shape: Optional[Tuple[int, int]] = eqx.field(static=True)

    # PI/sampling hyper-params
    num_power_iterations: int = eqx.field(static=True)
    num_freq_samples: Optional[int] = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    # State (optional)
    _uv_linear_index: Optional[StateIndex[Tuple[jnp.ndarray, jnp.ndarray]]] = None
    _freq_idx: Optional[jnp.ndarray] = eqx.field(
        static=True, default=None
    )  # (M,2) int32
    _uv_freq_index: Optional[StateIndex[Tuple[jnp.ndarray, jnp.ndarray]]] = None

    def __init__(
        self,
        layer: eqx.Module,
        target: float = 1.0,
        mode: Literal["clip", "force"] = "clip",
        detach_scale: bool = True,
        safety_factor: float = 1.05,
        inference: bool = False,
        # method choices
        linear_method: Literal["svd", "pi", "pi_state"] = "svd",
        conv2d_method: Literal[
            "tn",
            "circular_fft",
            "circular_gram",
            "kernel_flat",
            "min_tn_circ_embed",
            "circ_plus_lr",
        ] = "tn",
        convND_method: Literal["kernel_flat", "tn"] = "kernel_flat",
        param_svd_conv2d_method: Literal[
            "tn",
            "circular_fft",
            "circular_gram",
            "kernel_flat",
            "min_tn_circ_embed",
            "circ_plus_lr",
        ] = "tn",
        spectral2d_method: Literal["svd", "pi-sampled"] = "svd",
        # conv settings
        conv_tn_iters: int = 10,
        conv_tn_certified: bool = True,
        conv_gram_iters: int = 5,
        conv_fft_shape: Optional[Tuple[int, int]] = None,
        conv_input_shape: Optional[Tuple[int, int]] = None,
        # PI/sampling settings
        num_power_iterations: int = 1,
        num_freq_samples: Optional[int] = 64,
        eps: float = 1e-12,
        *,
        key: Any,
    ):
        self.layer = layer
        self.target = float(target)
        self.mode = mode
        self.detach_scale = bool(detach_scale)
        self.safety_factor = float(safety_factor)
        self.inference = bool(inference)

        self.linear_method = linear_method
        self.conv2d_method = conv2d_method
        self.convND_method = convND_method
        self.param_svd_conv2d_method = param_svd_conv2d_method
        self.spectral2d_method = spectral2d_method

        self.conv_tn_iters = int(conv_tn_iters)
        self.conv_tn_certified = bool(conv_tn_certified)
        self.conv_gram_iters = int(conv_gram_iters)
        self.conv_fft_shape = conv_fft_shape
        self.conv_input_shape = conv_input_shape

        self.num_power_iterations = int(num_power_iterations)
        self.num_freq_samples = num_freq_samples
        self.eps = float(eps)

        self._uv_linear_index = None
        self._freq_idx = None
        self._uv_freq_index = None

        # ---- Initialize persistent Linear PI (if requested) ----
        if (
            self.linear_method == "pi_state"
            and hasattr(eqx.nn, "Linear")
            and isinstance(layer, eqx.nn.Linear)
        ):
            W = getattr(layer, "weight")
            if W.ndim < 2:
                raise ValueError("Linear weight must be at least 2D.")
            W2 = W.reshape(W.shape[0], -1)
            k1, k2 = jr.split(key, 2)
            u0 = jr.normal(k1, (W2.shape[0],), dtype=W2.dtype)
            v0 = jr.normal(k2, (W2.shape[1],), dtype=W2.dtype)
            # Warm start
            for _ in range(15):
                u0, v0 = _power_iter_step(W2, u0, v0, self.eps)
            self._uv_linear_index = StateIndex((u0, v0))

        # ---- Initialize spectral-2D PI-sampled state (if requested) ----
        if self.spectral2d_method == "pi-sampled":
            if _is_name(layer, ["RFFTCirculant2D"]):
                Co, Ci = int(layer.C_out), int(layer.C_in)
                H_pad, W_half = int(layer.H_pad), int(layer.W_half)
                self._init_freq_pi_state(Co, Ci, H_pad, W_half, key)
            elif _is_name(
                layer, ["SpectralCirculantLayer2d", "AdaptiveSpectralCirculantLayer2d"]
            ):
                # Expect a full complex FFT kernel
                K_full = layer.get_fft_kernel()  # (Co, Ci, H_pad, W_pad), complex
                Co, Ci, H_pad, W_pad = map(int, K_full.shape)
                W_half = W_pad // 2 + 1
                self._init_freq_pi_state(Co, Ci, H_pad, W_half, key)

    # -------------------------------------------------------------------------
    # State init helpers (frequency-sampled PI for spectral 2D)
    # -------------------------------------------------------------------------

    def _init_freq_pi_state(self, Co: int, Ci: int, H_pad: int, W_half: int, key: Any):
        # Always include DC and Nyquist edges
        special = [(0, 0)]
        if H_pad % 2 == 0:
            special.append((H_pad // 2, 0))
        if W_half > 1:
            special.append((0, W_half - 1))
            if H_pad % 2 == 0:
                special.append((H_pad // 2, W_half - 1))
        special = jnp.array(special, dtype=jnp.int32)

        # Pool of all half-plane indices, excluding special ones
        UU, VV = jnp.meshgrid(
            jnp.arange(H_pad, dtype=jnp.int32),
            jnp.arange(W_half, dtype=jnp.int32),
            indexing="ij",
        )
        all_idx = jnp.stack([UU.reshape(-1), VV.reshape(-1)], axis=1)

        def not_special(idx):
            eqs = jnp.all(idx[None, :] == special[:, :], axis=1)
            return ~jnp.any(eqs)

        mask = jax.vmap(not_special)(all_idx)
        pool = all_idx[mask]

        Mmax = H_pad * W_half
        M = (
            Mmax
            if (self.num_freq_samples is None)
            else int(min(self.num_freq_samples, Mmax))
        )
        rest_needed = max(0, M - special.shape[0])
        k1, k2, k3 = jr.split(key, 3)
        perm = (
            jr.permutation(k1, pool.shape[0])
            if pool.shape[0] > 0
            else jnp.arange(0, dtype=jnp.int32)
        )
        chosen_rest = pool[perm[:rest_needed]] if rest_needed > 0 else pool[:0]
        chosen = (
            jnp.concatenate([special, chosen_rest], axis=0)
            if rest_needed > 0
            else special
        )
        self._freq_idx = chosen  # (M,2)

        # Initialise PI vectors per sampled frequency
        u0 = jr.normal(k2, (chosen.shape[0], Co))
        v0 = jr.normal(k3, (chosen.shape[0], Ci))
        u0 = u0 / (jnp.linalg.norm(u0, axis=1, keepdims=True) + self.eps)
        v0 = v0 / (jnp.linalg.norm(v0, axis=1, keepdims=True) + self.eps)
        self._uv_freq_index = StateIndex((u0, v0))

    # -------------------------------------------------------------------------
    # σ̂ estimators for each family
    # -------------------------------------------------------------------------

    # --- Linear-like matrices -------------------------------------------------
    def _sigma_linear(
        self, lin: eqx.nn.Linear, state: State
    ) -> Tuple[jnp.ndarray, State]:
        W2 = _flatten_matrix(lin.weight)
        if self.linear_method == "svd":
            sig = _top_sigma_matrix_svd(W2)
            return sig, state

        if self.linear_method == "pi":
            # Stateless few-step PI
            v = jnp.ones((W2.shape[1],), dtype=W2.dtype) / jnp.sqrt(W2.shape[1])
            u = W2 @ v
            u = u / (jnp.linalg.norm(u) + self.eps)
            for _ in range(max(0, self.num_power_iterations - 1)):
                v = W2.T @ u
                v = v / (jnp.linalg.norm(v) + self.eps)
                u = W2 @ v
                u = u / (jnp.linalg.norm(u) + self.eps)
            sig = jnp.vdot(u, W2 @ v).real
            return jnp.asarray(jnp.abs(sig), jnp.float32), state

        # persistent PI ('pi_state')
        if self._uv_linear_index is None:
            # fallback to stateless PI
            v = jnp.ones((W2.shape[1],), dtype=W2.dtype) / jnp.sqrt(W2.shape[1])
            u = W2 @ v
            u = u / (jnp.linalg.norm(u) + self.eps)
            for _ in range(max(0, self.num_power_iterations - 1)):
                v = W2.T @ u
                v = v / (jnp.linalg.norm(v) + self.eps)
                u = W2 @ v
                u = u / (jnp.linalg.norm(u) + self.eps)
            sig = jnp.vdot(u, W2 @ v).real
            return jnp.asarray(jnp.abs(sig), jnp.float32), state

        u, v = state.get(self._uv_linear_index)
        if not self.inference and self.num_power_iterations > 0:
            Wsg = jax.lax.stop_gradient(W2)
            for _ in range(self.num_power_iterations):
                u, v = _power_iter_step(Wsg, u, v, self.eps)
            state = state.set(self._uv_linear_index, (u, v))
        sig = jnp.vdot(u, W2 @ v).real
        return jnp.asarray(jnp.abs(sig), jnp.float32), state

    # --- Conv/ConvTranspose helpers ------------------------------------------
    def _conv_common_params(
        self, conv
    ) -> Tuple[jnp.ndarray, int, Tuple[int, int], Tuple[int, int]]:
        W = conv.weight
        num_dims = getattr(conv, "num_spatial_dims", W.ndim - 2)
        stride = getattr(conv, "stride", getattr(conv, "strides", (1,) * num_dims))
        if isinstance(stride, int):
            stride = (stride,) * num_dims
        dilation = (
            getattr(conv, "rhs_dilation", None)
            or getattr(conv, "dilation", None)
            or getattr(conv, "kernel_dilation", (1,) * num_dims)
        )
        if isinstance(dilation, int):
            dilation = (dilation,) * num_dims
        # Return only the first two components for stride/dilation (2D paths use them)
        if num_dims == 1:
            stride = (int(stride[0]), 1)
            dilation = (int(dilation[0]), 1)
        elif num_dims >= 2:
            stride = (int(stride[0]), int(stride[1]))
            dilation = (int(dilation[0]), int(dilation[1]))
        return W, int(num_dims), stride, dilation

    def _sigma_conv2d_weight(
        self, W: jnp.ndarray, stride: Tuple[int, int], dilation: Tuple[int, int]
    ) -> jnp.ndarray:
        sh, sw = stride
        dh, dw = dilation
        method = self.conv2d_method

        if method == "kernel_flat":
            return jnp.asarray(_sigma_conv_kernel_flat(W, method="svd"), jnp.float32)

        if method == "circular_fft":
            if (sh, sw) == (1, 1) and (self.conv_fft_shape is not None):
                return _sigma_conv_circular_fft_exact_2d(
                    W, self.conv_fft_shape, rhs_dilation=(dh, dw)
                )
            # fallback: TN UB
            return _sigma_conv_TN_upper(
                W,
                strides=(sh, sw),
                rhs_dilation=(dh, dw),
                iters=self.conv_tn_iters,
                certify=self.conv_tn_certified,
            )

        if method == "circular_gram":
            if (sh, sw) == (1, 1) and (self.conv_fft_shape is not None):
                return _sigma_conv_circular_gram_upper_2d(
                    W,
                    self.conv_fft_shape,
                    rhs_dilation=(dh, dw),
                    iters=self.conv_gram_iters,
                )
            return _sigma_conv_TN_upper(
                W,
                strides=(sh, sw),
                rhs_dilation=(dh, dw),
                iters=self.conv_tn_iters,
                certify=self.conv_tn_certified,
            )

        if method == "min_tn_circ_embed":
            if self.conv_input_shape is None:
                return _sigma_conv_TN_upper(
                    W,
                    strides=(sh, sw),
                    rhs_dilation=(dh, dw),
                    iters=self.conv_tn_iters,
                    certify=self.conv_tn_certified,
                )
            if (sh, sw) == (1, 1) and (dh, dw) == (1, 1):
                tn = _sigma_conv_TN_upper(
                    W,
                    strides=(1, 1),
                    rhs_dilation=(1, 1),
                    iters=self.conv_tn_iters,
                    certify=self.conv_tn_certified,
                )
                ce = _sigma_conv_circulant_embed_upper_2d(
                    W, in_hw=self.conv_input_shape
                )
                return jnp.minimum(tn, ce).astype(jnp.float32)
            return _sigma_conv_TN_upper(
                W,
                strides=(sh, sw),
                rhs_dilation=(dh, dw),
                iters=self.conv_tn_iters,
                certify=self.conv_tn_certified,
            )

        if method == "circ_plus_lr":
            if self.conv_input_shape is None:
                return _sigma_conv_TN_upper(
                    W,
                    strides=(sh, sw),
                    rhs_dilation=(dh, dw),
                    iters=self.conv_tn_iters,
                    certify=self.conv_tn_certified,
                )
            return _sigma_conv_circ_plus_lr_upper_2d(
                W, in_hw=self.conv_input_shape, strides=(sh, sw), rhs_dilation=(dh, dw)
            )

        # default: "tn"
        return _sigma_conv_TN_upper(
            W,
            strides=(sh, sw),
            rhs_dilation=(dh, dw),
            iters=self.conv_tn_iters,
            certify=self.conv_tn_certified,
        )

    def _sigma_conv(self, conv, state: State) -> Tuple[jnp.ndarray, State]:
        W, nd, stride, dilation = self._conv_common_params(conv)
        if nd != 2:
            # 1D/3D conservative fallback
            sig = _sigma_conv_kernel_flat(W, method="svd")
            return jnp.asarray(sig, jnp.float32), state
        sig = self._sigma_conv2d_weight(W, stride, dilation)
        return jnp.asarray(sig, jnp.float32), state

    def _sigma_convT(self, convT, state: State) -> Tuple[jnp.ndarray, State]:
        # Treat ConvTranspose via channel-swapped conv weight
        W = convT.weight
        if W.ndim >= 4 and W.shape[0] < W.shape[1]:
            Wc = jnp.transpose(W, (1, 0, 2, 3))
        else:
            Wc = W
        stride = getattr(convT, "stride", getattr(convT, "strides", (1, 1)))
        if isinstance(stride, int):
            stride = (stride, stride)
        dilation = (
            getattr(convT, "rhs_dilation", None)
            or getattr(convT, "dilation", None)
            or getattr(convT, "kernel_dilation", (1, 1))
        )
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        sig = self._sigma_conv2d_weight(
            Wc, (int(stride[0]), int(stride[1])), (int(dilation[0]), int(dilation[1]))
        )
        return jnp.asarray(sig, jnp.float32), state

    # --- Param-SVD convs: reconstruct kernel then call conv estimator ---------
    def _sigma_param_svd_conv2d(self, mod, state: State) -> Tuple[jnp.ndarray, State]:
        # U:(Co,r), V:(Cin*Kh*Kw,r), s:(r,)
        Co, Ci, Kh, Kw = int(mod.C_out), int(mod.C_in), int(mod.H_k), int(mod.W_k)
        W_mat = mod.U @ (mod.s[:, None] * mod.V.T)
        W = W_mat.reshape(Co, Ci, Kh, Kw)
        stride = getattr(mod, "strides", (1, 1))
        if isinstance(stride, int):
            stride = (stride, stride)
        # Temporarily use the chosen method for param-SVD convs
        save = self.conv2d_method
        try:
            self.conv2d_method = self.param_svd_conv2d_method
            sig = self._sigma_conv2d_weight(W, (int(stride[0]), int(stride[1])), (1, 1))
        finally:
            self.conv2d_method = save
        return jnp.asarray(sig, jnp.float32), state

    # --- RFFTCirculant1D: exact ---------------------------------------------
    def _sigma_rfft_circ1d(self, mod) -> jnp.ndarray:
        return _max_abs(mod.H_half).astype(jnp.float32)

    # --- SpectralTokenMixer: exact (+1 if residual) ---------------------------
    def _sigma_token_mixer(self, mod) -> jnp.ndarray:
        H = mod.H_half  # (G, k_half)
        s_g = jnp.max(jnp.abs(H), axis=1)  # (G,)
        G = H.shape[0]
        if hasattr(mod, "gate"):
            C = mod.gate.shape[0]
            ch_per_g = C // G
            gate = jnp.abs(mod.gate).reshape(G, ch_per_g)
            g_g = jnp.max(gate, axis=1)
        else:
            g_g = jnp.ones_like(s_g)
        base = jnp.max(s_g * g_g)
        return (
            (base + 1.0).astype(jnp.float32)
            if getattr(mod, "use_residual", False)
            else base.astype(jnp.float32)
        )

    # --- RFFTCirculant2D & similar: exact SVD or PI-sampled over half-plane ---
    def _sigma_rfft_circ2d_svd(self, K_half: jnp.ndarray) -> jnp.ndarray:
        # K_half: (Co, Ci, Hf, W_half)
        mats = jnp.transpose(K_half, (2, 3, 0, 1)).reshape(
            -1, K_half.shape[0], K_half.shape[1]
        )
        svals = jax.vmap(lambda M: jnp.linalg.svd(M, compute_uv=False)[0])(mats)
        return jnp.max(jnp.real(svals)).astype(jnp.float32)

    def _sigma_rfft_circ2d_pi(
        self, K_half: jnp.ndarray, state: State
    ) -> Tuple[jnp.ndarray, State]:
        assert (
            self._freq_idx is not None and self._uv_freq_index is not None
        ), "PI state not initialised."
        u, v = state.get(self._uv_freq_index)  # (M, Co), (M, Ci)
        freq_idx = self._freq_idx  # (M,2)

        def gather(idx):
            u0, v0 = int(idx[0]), int(idx[1])
            return K_half[:, :, u0, v0]  # (Co, Ci)

        mats = jax.vmap(gather)(freq_idx)  # (M, Co, Ci)

        if not self.inference and self.num_power_iterations > 0:
            Mstop = jax.lax.stop_gradient(mats)

            def step(uu, vv):
                uu, vv = jax.vmap(_power_iter_step, in_axes=(0, 0, 0, None))(
                    Mstop, uu, vv, self.eps
                )
                return uu, vv

            uu, vv = u, v
            for _ in range(self.num_power_iterations):
                uu, vv = step(uu, vv)
            state = state.set(self._uv_freq_index, (uu, vv))
            u, v = uu, vv

        est = jax.vmap(lambda M, uu, vv: jnp.vdot(uu, M @ vv))(mats, u, v)  # (M,)
        sigma = jnp.max(jnp.abs(est)).astype(jnp.float32)
        return sigma, state

    # -------------------------------------------------------------------------
    # Build a scaled "view" of the layer (pure; no in-place mutation)
    # -------------------------------------------------------------------------

    def _scale_layer(self, lyr: eqx.Module, scale: jnp.ndarray) -> eqx.Module:
        if self.detach_scale:
            scale = jax.lax.stop_gradient(scale)

        name = type(lyr).__name__

        # --- Equinox built-ins ---
        if hasattr(eqx.nn, "Linear") and isinstance(lyr, eqx.nn.Linear):
            return tree_at(lambda m: m.weight, lyr, lyr.weight / scale)

        if hasattr(eqx.nn, "Conv") and isinstance(lyr, eqx.nn.Conv):
            return tree_at(lambda m: m.weight, lyr, lyr.weight / scale)

        if hasattr(eqx.nn, "ConvTranspose") and isinstance(lyr, eqx.nn.ConvTranspose):
            return tree_at(lambda m: m.weight, lyr, lyr.weight / scale)

        # --- Your spectral parametrisations ---
        if _is_name(lyr, ["SVDDense", "SpectralDense", "AdaptiveSpectralDense"]):
            return tree_at(lambda m: m.s, lyr, lyr.s / scale)

        if _is_name(lyr, ["SpectralConv2d", "AdaptiveSpectralConv2d"]):
            return tree_at(lambda m: m.s, lyr, lyr.s / scale)

        if _is_name(lyr, ["RFFTCirculant1D"]):
            return tree_at(lambda m: m.H_half, lyr, lyr.H_half / scale)

        if _is_name(lyr, ["RFFTCirculant2D"]):
            return tree_at(lambda m: m.K_half, lyr, lyr.K_half / scale)

        if _is_name(lyr, ["SpectralTokenMixer"]):
            return tree_at(lambda m: m.H_half, lyr, lyr.H_half / scale)

        # Optional: generic spectral-circulant variants with explicit real/imag
        if _is_name(lyr, ["SpectralCirculantLayer", "AdaptiveSpectralCirculantLayer"]):
            new = tree_at(lambda m: m.w_real, lyr, lyr.w_real / scale)
            new = tree_at(lambda m: m.w_imag, new, lyr.w_imag / scale)
            return new

        if _is_name(
            lyr, ["SpectralCirculantLayer2d", "AdaptiveSpectralCirculantLayer2d"]
        ):
            new = tree_at(lambda m: m.w_real, lyr, lyr.w_real / scale)
            new = tree_at(lambda m: m.w_imag, new, lyr.w_imag / scale)
            return new

        raise ValueError(f"Unsupported layer type for SpectralNorm: {name}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def __operator_norm_hint__(self) -> float | None:
        """
        Optional diagnostic (not used on the JIT path). If the wrapped layer
        exposes its own hint, forward it; otherwise return None.
        """
        core = getattr(self, "layer", None)
        if core is not None and hasattr(core, "__operator_norm_hint__"):
            try:
                v = core.__operator_norm_hint__()
                if v is not None:
                    return float(v)
            except Exception:
                pass
        return None

    @named_scope("quantbayes.nn.SpectralNorm")
    def __call__(
        self,
        x: jnp.ndarray,
        state: State,
        *,
        key: Any | None = None,
        inference: Optional[bool] = None,
    ) -> Tuple[jnp.ndarray, State]:
        if inference is not None:
            self.inference = bool(inference)

        lyr = self.layer
        name = type(lyr).__name__

        # 1) Compute σ̂ as a rank-0 JAX array
        if hasattr(eqx.nn, "Linear") and isinstance(lyr, eqx.nn.Linear):
            sigma, state = self._sigma_linear(lyr, state)

        elif hasattr(eqx.nn, "Conv") and isinstance(lyr, eqx.nn.Conv):
            sigma, state = self._sigma_conv(lyr, state)

        elif hasattr(eqx.nn, "ConvTranspose") and isinstance(lyr, eqx.nn.ConvTranspose):
            sigma, state = self._sigma_convT(lyr, state)

        elif _is_name(lyr, ["SVDDense", "SpectralDense", "AdaptiveSpectralDense"]):
            sigma = jnp.max(jnp.abs(lyr.s)).astype(jnp.float32)

        elif _is_name(lyr, ["SpectralConv2d", "AdaptiveSpectralConv2d"]):
            sigma, state = self._sigma_param_svd_conv2d(lyr, state)

        elif _is_name(lyr, ["RFFTCirculant1D"]):
            sigma = self._sigma_rfft_circ1d(lyr)

        elif _is_name(lyr, ["SpectralTokenMixer"]):
            sigma = self._sigma_token_mixer(lyr)

        elif _is_name(lyr, ["RFFTCirculant2D"]):
            if self.spectral2d_method == "svd":
                sigma = self._sigma_rfft_circ2d_svd(lyr.K_half)
            else:
                sigma, state = self._sigma_rfft_circ2d_pi(lyr.K_half, state)

        elif _is_name(
            lyr, ["SpectralCirculantLayer2d", "AdaptiveSpectralCirculantLayer2d"]
        ):
            K_full = lyr.get_fft_kernel()  # (Co, Ci, H_pad, W_pad), complex
            W_half = (K_full.shape[-1] // 2) + 1
            K_half = K_full[..., :W_half]
            if self.spectral2d_method == "svd":
                sigma = self._sigma_rfft_circ2d_svd(K_half)
            else:
                sigma, state = self._sigma_rfft_circ2d_pi(K_half, state)

        elif _is_name(
            lyr, ["SpectralCirculantLayer", "AdaptiveSpectralCirculantLayer"]
        ):
            if hasattr(lyr, "get_fourier_coeffs"):
                sigma = _max_abs(lyr.get_fourier_coeffs()).astype(jnp.float32)
            else:
                raise ValueError(
                    f"{name} missing get_fourier_coeffs() for norm estimate."
                )

        else:
            raise ValueError(f"Unsupported layer type for SpectralNorm: {name}")

        # 2) Compute scaling with cushion
        sigma_adj = sigma * jnp.asarray(self.safety_factor, jnp.float32)
        if self.mode == "clip":
            scale = jnp.maximum(1.0, sigma_adj / jnp.asarray(self.target, jnp.float32))
        else:  # 'force'
            scale = jnp.maximum(
                1e-12, sigma_adj / jnp.asarray(self.target, jnp.float32)
            )

        # 3) Scaled "view" and forward
        eff = self._scale_layer(lyr, scale)
        try:
            out = eff(x, key=key)  # many eqx layers accept key=...
        except TypeError:
            out = eff(x)

        return out, state


# =============================================================================
# Minimal sanity test
# =============================================================================
if __name__ == "__main__":
    # Basic smoke test: a tiny MLP with SpectralNorm on the first layer.
    import jax.random as jr
    import numpy as np
    import matplotlib.pyplot as plt
    import optax
    from sklearn.model_selection import train_test_split
    from quantbayes.stochax import train, predict, BoundLogger, binary_loss
    from quantbayes.stochax.utils.lip_upper import make_lipschitz_upper_fn
    from quantbayes.fake_data import generate_binary_classification_data

    df = generate_binary_classification_data()
    X, y = df.drop(columns=["target"]).values, df["target"].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_epochs = 100
    batch_size = 32

    class Toy(eqx.Module):
        l1: SpectralNorm
        l2: eqx.nn.Linear

        def __init__(self, key):
            k1, k2, k3 = jr.split(key, 3)
            core = eqx.nn.Linear(5, 10, key=k1)
            self.l1 = SpectralNorm(
                core,
                target=1.0,
                mode="clip",
                linear_method="svd",  # or 'pi' / 'pi_state'
                conv2d_method="tn",  # irrelevant here
                spectral2d_method="svd",  # irrelevant here
                key=k2,
            )
            self.l2 = eqx.nn.Linear(10, 1, key=k3)

        def __call__(self, x, key, state):
            x, state = self.l1(x, state, key=key)
            x = jax.nn.relu(x)
            y = self.l2(x)
            return y, state

    master_key = jr.PRNGKey(42)
    model_key, train_key = jr.split(master_key)

    model, state = eqx.nn.make_with_state(Toy)(model_key)
    steps_per_epoch = int(jnp.ceil(X_train.shape[0] / batch_size))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(100, steps_per_epoch)  # ~1 epoch or 100 steps

    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,  # your peak LR
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=1e-5,
    )
    optimizer = optax.adan(
        learning_rate=lr_sched,
        b1=0.95,
        b2=0.99,
        eps=1e-8,
        weight_decay=1e-4,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    brec = BoundLogger()
    best_model, best_state, tr_loss, va_loss = train(
        model,
        state,
        opt_state,
        optimizer,
        binary_loss,
        jnp.array(X_train),
        jnp.array(y_train),
        jnp.array(X_val),
        jnp.array(y_val),
        batch_size=batch_size,
        num_epochs=num_epochs,
        patience=5,
        key=train_key,
        lambda_spec=0.0,
        log_global_bound_every=1,  # log every epoch (raise if slow)
        bound_conv_mode="min_tn_circ_embed",  # tight, certified
        bound_tn_iters=8,
        bound_input_shape=(28, 28),  # REQUIRED for circ-embed flavors
        bound_recorder=brec,
    )
    print("Training complete.")
    print("First few bound logs:", brec.data[:3])

    # ---------------- final tightest certified bound (paper number) -----------
    L_min_tn_circ = float(
        make_lipschitz_upper_fn(
            conv_mode="min_tn_circ_embed", conv_tn_iters=8, conv_input_shape=(28, 28)
        )(best_model)
    )
    L_circ_plus_lr = float(
        make_lipschitz_upper_fn(
            conv_mode="circ_plus_lr", conv_tn_iters=8, conv_input_shape=(28, 28)
        )(best_model)
    )
    L_final = min(L_min_tn_circ, L_circ_plus_lr)
    print(
        f"Final certified L: min(min_tn_circ_embed={L_min_tn_circ:.6g}, "
        f"circ_plus_lr={L_circ_plus_lr:.6g}) = {L_final:.6g}"
    )

    # ---------------- sanity: predictions shape ----------------
    logits = predict(best_model, best_state, jnp.array(X_val), jr.PRNGKey(0))
    print("Predictions shape:", logits.shape)

    # ---------------- plot losses + Lipschitz curve on twin axis -------------
    epochs = np.arange(1, len(tr_loss) + 1)
    # brec logs only at chosen cadence; we plot eval (EMA if enabled)
    bepochs = np.array([r["epoch"] for r in brec.data], int)
    # L_eval = np.array([r["L_eval"] for r in brec.data], float)
    L_raw = np.array([r["L_raw"] for r in brec.data], float)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs, tr_loss, label="train loss")
    ax1.plot(epochs, va_loss, label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    # ax2.plot(bepochs, L_eval, linestyle="--", marker="o", label="L_eval (certified)")
    ax2.plot(bepochs, L_raw, linestyle=":", marker="x", label="L_raw (certified)")
    ax2.set_ylabel("global Lipschitz upper bound")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
