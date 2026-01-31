# quantbayes/stochax/layers/specnorm.py
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
    _sigma_conv_circular_fft_exact_2d,
    _sigma_conv_circular_gram_upper_2d,
    _sigma_conv_circulant_embed_upper_2d,
    _sigma_conv_TN_upper,
    _sigma_conv_circ_plus_lr_upper_2d,
    _sigma_conv_circ_embed_opt_upper_2d,
)


# =============================================================================
# Small utilities
# =============================================================================


def _is_name(x, names: Sequence[str]) -> bool:
    return any(type(x).__name__ == n for n in names)


def _flatten_matrix(W: jnp.ndarray) -> jnp.ndarray:
    return W.reshape(W.shape[0], -1)


def _top_sigma_matrix_svd(W2: jnp.ndarray) -> jnp.ndarray:
    # Exact σ_max via SVD (JAX)
    return jnp.linalg.svd(W2, compute_uv=False)[0].astype(jnp.float32)


def _max_abs(z: jnp.ndarray) -> jnp.ndarray:
    return jnp.max(jnp.abs(z))


# =============================================================================
# The Spectral Normalization wrapper
# =============================================================================


class SpectralNorm(StatefulLayer):
    """
    CERTIFIED-ONLY spectral normalization.

    Per call:
      1) Compute a certified operator norm σ for the wrapped layer:
           • eqx.nn.Linear                → exact (top singular value via SVD)
           • eqx.nn.Conv2d/ConvTranspose2d
                 - 'tn'                  → certified TN upper bound (any stride/dilation)
                 - 'circular_fft'        → exact for circular padding, stride=1 (any dilation) with fft_shape
                 - 'circular_gram'       → certified Gram UB for circular, stride=1 with fft_shape
                 - 'min_tn_circ_embed'   → min{TN, Circulant-Embed UB} (stride=1, dilation=1) with input_shape
                 - 'circ_plus_lr'        → certified circulant+low-rank UB with input_shape (stride/dilation allowed)
           • SVDDense/SpectralDense/AdaptiveSpectralDense → exact: max|s|
           • SpectralConv2d/AdaptiveSpectralConv2d        → reconstruct kernel → chosen conv2d estimator
           • RFFTCirculant1D                              → exact: max|H_half|
           • RFFTCirculant2D                              → exact: per-frequency SVD over half-plane
           • SpectralTokenMixer                           → exact with gate; +1 if residual
      2) Scale layer parameters by 1 / scale, where
            scale = max(1, safety_factor * σ / target)     if mode='clip'
                    max(1e-12, safety_factor * σ / target) if mode='force'

    No PI, no sampling, no heuristics, no silent fallbacks.

    Parameters
    ----------
    layer : eqx.Module
    target : float                (default 1.0)
    mode : {'clip','force'}       (default 'clip')
    detach_scale : bool           (default True)
    safety_factor : float         (default 1.0)
    inference : bool              (default False)  # passed through to wrapped layer if it accepts it

    conv2d_method : {'tn','circular_fft','circular_gram','min_tn_circ_embed','circ_plus_lr'}
    param_svd_conv2d_method : same as conv2d_method (used by SpectralConv2d*)

    conv_tn_iters : int           (default 10) — used by TN bound
    conv_gram_iters : int         (default 5)
    conv_fft_shape : Optional[(Hf,Wf)]  — required for 'circular_fft' / 'circular_gram'
    conv_input_shape : Optional[(Hin,Win)] — required for 'min_tn_circ_embed' / 'circ_plus_lr'
    """

    # Wrapped layer & global knobs
    layer: eqx.Module
    target: float = eqx.field(static=True)
    mode: Literal["clip", "force"] = eqx.field(static=True)
    detach_scale: bool = eqx.field(static=True)
    safety_factor: float = eqx.field(static=True)
    inference: bool

    # Method selection (certified-only)
    conv2d_method: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = eqx.field(static=True)
    param_svd_conv2d_method: Literal[
        "tn",
        "circular_fft",
        "circular_gram",
        "min_tn_circ_embed",
        "circ_plus_lr",
        "circ_embed_opt",
    ] = eqx.field(static=True)

    # Conv hyper-params
    conv_tn_iters: int = eqx.field(static=True)
    conv_gram_iters: int = eqx.field(static=True)
    conv_fft_shape: Optional[Tuple[int, int]] = eqx.field(static=True)
    conv_input_shape: Optional[Tuple[int, int]] = eqx.field(static=True)

    def __init__(
        self,
        layer: eqx.Module,
        *,
        target: float = 1.0,
        mode: Literal["clip", "force"] = "clip",
        detach_scale: bool = True,
        safety_factor: float = 1.0,
        inference: bool = False,
        conv2d_method: Literal[
            "tn",
            "circular_fft",
            "circular_gram",
            "min_tn_circ_embed",
            "circ_plus_lr",
            "circ_embed_opt",
        ] = "tn",
        param_svd_conv2d_method: Literal[
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
    ):
        # Store
        self.layer = layer
        self.target = float(target)
        self.mode = mode
        self.detach_scale = bool(detach_scale)
        self.safety_factor = float(safety_factor)
        self.inference = bool(inference)

        self.conv2d_method = conv2d_method
        self.param_svd_conv2d_method = param_svd_conv2d_method

        self.conv_tn_iters = int(conv_tn_iters)
        self.conv_gram_iters = int(conv_gram_iters)
        self.conv_fft_shape = conv_fft_shape
        self.conv_input_shape = conv_input_shape

        # Hard requirements (no silent fallback)
        if (
            self.conv2d_method in {"circular_fft", "circular_gram"}
            and self.conv_fft_shape is None
        ):
            raise ValueError(f"{self.conv2d_method} requires conv_fft_shape=(Hf, Wf).")
        if (
            self.conv2d_method in {"min_tn_circ_embed", "circ_plus_lr"}
            and self.conv_input_shape is None
        ):
            raise ValueError(
                f"{self.conv2d_method} requires conv_input_shape=(Hin, Win)."
            )

        if (
            self.param_svd_conv2d_method in {"circular_fft", "circular_gram"}
            and self.conv_fft_shape is None
        ):
            raise ValueError(
                f"param_svd_conv2d_method={self.param_svd_conv2d_method} requires conv_fft_shape=(Hf, Wf)."
            )
        if (
            self.param_svd_conv2d_method in {"min_tn_circ_embed", "circ_plus_lr"}
            and self.conv_input_shape is None
        ):
            raise ValueError(
                f"param_svd_conv2d_method={self.param_svd_conv2d_method} requires conv_input_shape=(Hin, Win)."
            )

    # --- Linear (exact) -------------------------------------------------------
    def _sigma_linear_exact(self, lin: eqx.nn.Linear) -> jnp.ndarray:
        W = _flatten_matrix(lin.weight)
        return _top_sigma_matrix_svd(W).astype(jnp.float32)

    # --- Conv helpers ---------------------------------------------------------
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

        if method == "tn":
            return _sigma_conv_TN_upper(
                W,
                strides=(sh, sw),
                rhs_dilation=(dh, dw),
                iters=self.conv_tn_iters,
                certify=True,
            ).astype(jnp.float32)

        if method == "circ_embed_opt":
            if self.conv_input_shape is None:
                raise ValueError("circ_embed_opt requires conv_input_shape=(Hin, Win).")
            return _sigma_conv_circ_embed_opt_upper_2d(
                W, in_hw=self.conv_input_shape, rhs_dilation=(dh, dw)
            ).astype(jnp.float32)

        if method == "circular_fft":
            if (sh, sw) != (1, 1):
                raise ValueError(
                    "circular_fft is exact only for stride=(1,1). Use 'tn' or 'circ_plus_lr'."
                )
            if self.conv_fft_shape is None:
                raise ValueError("circular_fft requires conv_fft_shape=(Hf, Wf).")
            return _sigma_conv_circular_fft_exact_2d(
                W, self.conv_fft_shape, rhs_dilation=(dh, dw)
            ).astype(jnp.float32)

        if method == "circular_gram":
            if (sh, sw) != (1, 1):
                raise ValueError(
                    "circular_gram is supported for stride=(1,1). Use 'tn' or 'circ_plus_lr'."
                )
            if self.conv_fft_shape is None:
                raise ValueError("circular_gram requires conv_fft_shape=(Hf, Wf).")
            return _sigma_conv_circular_gram_upper_2d(
                W,
                self.conv_fft_shape,
                rhs_dilation=(dh, dw),
                iters=self.conv_gram_iters,
            ).astype(jnp.float32)

        if method == "min_tn_circ_embed":
            if self.conv_input_shape is None:
                raise ValueError(
                    "min_tn_circ_embed requires conv_input_shape=(Hin, Win)."
                )
            if (sh, sw) != (1, 1) or (dh, dw) != (1, 1):
                raise ValueError(
                    "min_tn_circ_embed is defined for stride=(1,1), dilation=(1,1)."
                )
            tn = _sigma_conv_TN_upper(
                W,
                strides=(1, 1),
                rhs_dilation=(1, 1),
                iters=self.conv_tn_iters,
                certify=True,
            )
            ce = _sigma_conv_circulant_embed_upper_2d(W, in_hw=self.conv_input_shape)
            return jnp.minimum(tn, ce).astype(jnp.float32)

        # method == "circ_plus_lr"
        if self.conv_input_shape is None:
            raise ValueError("circ_plus_lr requires conv_input_shape=(Hin, Win).")
        return _sigma_conv_circ_plus_lr_upper_2d(
            W, in_hw=self.conv_input_shape, strides=(sh, sw), rhs_dilation=(dh, dw)
        ).astype(jnp.float32)

    def _sigma_conv(self, conv) -> jnp.ndarray:
        W, nd, stride, dilation = self._conv_common_params(conv)
        if nd != 2:
            raise ValueError("This SpectralNorm enforces certified convs for 2D only.")
        return self._sigma_conv2d_weight(W, stride, dilation)

    def _sigma_convT(self, convT) -> jnp.ndarray:
        W = convT.weight
        # Channel-swap for transpose conv
        Wc = (
            jnp.transpose(W, (1, 0, 2, 3))
            if (W.ndim >= 4 and W.shape[0] < W.shape[1])
            else W
        )
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
        return self._sigma_conv2d_weight(
            Wc, (int(stride[0]), int(stride[1])), (int(dilation[0]), int(dilation[1]))
        )

    # --- Param-SVD convs: reconstruct kernel then call conv estimator ---------
    def _sigma_param_svd_conv2d(self, mod) -> jnp.ndarray:
        Co, Ci, Kh, Kw = int(mod.C_out), int(mod.C_in), int(mod.H_k), int(mod.W_k)
        W_mat = mod.U @ (mod.s[:, None] * mod.V.T)  # (Co, Ci*Kh*Kw)
        W = W_mat.reshape(Co, Ci, Kh, Kw)  # (Co, Ci, Kh, Kw)
        stride = getattr(mod, "strides", (1, 1))
        if isinstance(stride, int):
            stride = (stride, stride)
        # Temporarily switch method for param-SVD convs
        save = self.conv2d_method
        try:
            self.conv2d_method = self.param_svd_conv2d_method
            sig = self._sigma_conv2d_weight(W, (int(stride[0]), int(stride[1])), (1, 1))
        finally:
            self.conv2d_method = save
        return jnp.asarray(sig, jnp.float32)

    # --- RFFTCirculant1D: exact ----------------------------------------------
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

    # --- RFFTCirculant2D: exact per-frequency SVD over the half-plane ---------
    def _sigma_rfft_circ2d_svd(self, K_half: jnp.ndarray) -> jnp.ndarray:
        # K_half: (Co, Ci, Hf, W_half)
        mats = jnp.transpose(K_half, (2, 3, 0, 1)).reshape(
            -1, K_half.shape[0], K_half.shape[1]
        )  # (Hf*W_half, Co, Ci)
        svals = jax.vmap(lambda M: jnp.linalg.svd(M, compute_uv=False)[0])(
            mats
        )  # (Hf*W_half,)
        return jnp.max(jnp.real(svals)).astype(jnp.float32)

    # --- Scaling --------------------------------------------------------------
    def _scale_layer(self, lyr: eqx.Module, scale: jnp.ndarray) -> eqx.Module:
        if self.detach_scale:
            scale = jax.lax.stop_gradient(scale)

        name = type(lyr).__name__

        # Equinox
        if hasattr(eqx.nn, "Linear") and isinstance(lyr, eqx.nn.Linear):
            return tree_at(lambda m: m.weight, lyr, lyr.weight / scale)

        if hasattr(eqx.nn, "Conv") and isinstance(lyr, eqx.nn.Conv):
            return tree_at(lambda m: m.weight, lyr, lyr.weight / scale)

        if hasattr(eqx.nn, "ConvTranspose") and isinstance(lyr, eqx.nn.ConvTranspose):
            return tree_at(lambda m: m.weight, lyr, lyr.weight / scale)

        # Your spectral parametrisations (exact via their s/FFT parameters)
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

    # --- Public API -----------------------------------------------------------
    def __operator_norm_hint__(self) -> jnp.ndarray | None:
        """
        Advertise a certified per-layer cap to the global Lipschitz aggregator.

        Returns target IFF the wrapped layer+method combination is proven-correct
        under our config; otherwise returns None so the aggregator won’t use a hint.
        """
        # Only advertise a cap when it’s actually a cap
        if not (self.target > 0 and self.safety_factor >= 1.0):
            return None

        lyr = self.layer

        # Exact linear
        if hasattr(eqx.nn, "Linear") and isinstance(lyr, eqx.nn.Linear):
            return jnp.asarray(self.target, jnp.float32)

        # Helpers for conv stride/dilation
        def _stride_dilation(_conv):
            W = getattr(_conv, "weight", None)
            nd = getattr(
                _conv, "num_spatial_dims", (W.ndim - 2) if W is not None else 2
            )
            stride = getattr(_conv, "stride", getattr(_conv, "strides", (1, 1)))
            if isinstance(stride, int):
                stride = (stride, stride)
            dilation = (
                getattr(_conv, "rhs_dilation", None)
                or getattr(_conv, "dilation", None)
                or getattr(_conv, "kernel_dilation", (1, 1))
            )
            if isinstance(dilation, int):
                dilation = (dilation, dilation)
            return (
                int(nd),
                (int(stride[0]), int(stride[1])),
                (int(dilation[0]), int(dilation[1])),
            )

        # Conv2d (only nd==2) — method-specific guards
        if hasattr(eqx.nn, "Conv") and isinstance(lyr, eqx.nn.Conv):
            nd, stride, dilation = _stride_dilation(lyr)
            if nd != 2:
                return None
            if self.conv2d_method == "tn":
                return jnp.asarray(self.target, jnp.float32)
            if self.conv2d_method == "circular_fft":
                return (
                    jnp.asarray(self.target, jnp.float32)
                    if stride == (1, 1) and self.conv_fft_shape is not None
                    else None
                )
            if self.conv2d_method == "circular_gram":
                return (
                    jnp.asarray(self.target, jnp.float32)
                    if stride == (1, 1) and self.conv_fft_shape is not None
                    else None
                )
            if self.conv2d_method == "min_tn_circ_embed":
                ok = (
                    (stride == (1, 1))
                    and (dilation == (1, 1))
                    and (self.conv_input_shape is not None)
                )
                return jnp.asarray(self.target, jnp.float32) if ok else None
            if self.conv2d_method == "circ_plus_lr":
                return (
                    jnp.asarray(self.target, jnp.float32)
                    if self.conv_input_shape is not None
                    else None
                )
            return None

        # ConvTranspose2d — same guards
        if hasattr(eqx.nn, "ConvTranspose") and isinstance(lyr, eqx.nn.ConvTranspose):
            nd, stride, dilation = _stride_dilation(lyr)
            if nd != 2:
                return None
            if self.conv2d_method == "tn":
                return jnp.asarray(self.target, jnp.float32)
            if self.conv2d_method == "circular_fft":
                return (
                    jnp.asarray(self.target, jnp.float32)
                    if stride == (1, 1) and self.conv_fft_shape is not None
                    else None
                )
            if self.conv2d_method == "circular_gram":
                return (
                    jnp.asarray(self.target, jnp.float32)
                    if stride == (1, 1) and self.conv_fft_shape is not None
                    else None
                )
            if self.conv2d_method == "min_tn_circ_embed":
                ok = (
                    (stride == (1, 1))
                    and (dilation == (1, 1))
                    and (self.conv_input_shape is not None)
                )
                return jnp.asarray(self.target, jnp.float32) if ok else None
            if self.conv2d_method == "circ_plus_lr":
                return (
                    jnp.asarray(self.target, jnp.float32)
                    if self.conv_input_shape is not None
                    else None
                )
            return None

        # Exact spectral parametrizations (all exact in your codebase)
        if _is_name(
            lyr,
            [
                "SVDDense",
                "SpectralDense",
                "AdaptiveSpectralDense",
                "RFFTCirculant1D",
                "RFFTCirculant2D",
                "SpectralTokenMixer",
                "SpectralCirculantLayer",
                "AdaptiveSpectralCirculantLayer",
                "SpectralCirculantLayer2d",
                "AdaptiveSpectralCirculantLayer2d",
            ],
        ):
            return jnp.asarray(self.target, jnp.float32)

        # Forward a hint from the wrapped module if it has one (cast to JAX scalar)
        if hasattr(lyr, "__operator_norm_hint__"):
            try:
                v = lyr.__operator_norm_hint__()
                return None if v is None else jnp.asarray(v, jnp.float32)
            except Exception:
                return None

        return None

    @named_scope("quantbayes.nn.SpectralNorm")
    def __call__(
        self,
        x: jnp.ndarray,
        state: "State",
        *,
        key: Any | None = None,
        inference: Optional[bool] = None,
    ):
        if inference is not None:
            self.inference = bool(inference)

        lyr = self.layer

        # 1) Certified σ
        if hasattr(eqx.nn, "Linear") and isinstance(lyr, eqx.nn.Linear):
            sigma = self._sigma_linear_exact(lyr)

        elif hasattr(eqx.nn, "Conv") and isinstance(lyr, eqx.nn.Conv):
            sigma = self._sigma_conv(lyr)

        elif hasattr(eqx.nn, "ConvTranspose") and isinstance(lyr, eqx.nn.ConvTranspose):
            sigma = self._sigma_convT(lyr)

        elif _is_name(lyr, ["SVDDense", "SpectralDense", "AdaptiveSpectralDense"]):
            sigma = jnp.max(jnp.abs(lyr.s)).astype(jnp.float32)

        elif _is_name(lyr, ["SpectralConv2d", "AdaptiveSpectralConv2d"]):
            sigma = self._sigma_param_svd_conv2d(lyr)

        elif _is_name(lyr, ["RFFTCirculant1D"]):
            sigma = self._sigma_rfft_circ1d(lyr)

        elif _is_name(lyr, ["SpectralTokenMixer"]):
            sigma = self._sigma_token_mixer(lyr)

        elif _is_name(lyr, ["RFFTCirculant2D"]):
            sigma = self._sigma_rfft_circ2d_svd(lyr.K_half)

        elif _is_name(
            lyr, ["SpectralCirculantLayer2d", "AdaptiveSpectralCirculantLayer2d"]
        ):
            K_full = lyr.get_fft_kernel()  # (Co, Ci, H_pad, W_pad), complex
            W_half = (K_full.shape[-1] // 2) + 1
            K_half = K_full[..., :W_half]
            sigma = self._sigma_rfft_circ2d_svd(K_half)

        elif _is_name(
            lyr, ["SpectralCirculantLayer", "AdaptiveSpectralCirculantLayer"]
        ):
            if hasattr(lyr, "get_fourier_coeffs"):
                sigma = _max_abs(lyr.get_fourier_coeffs()).astype(jnp.float32)
            else:
                raise ValueError(f"{type(lyr).__name__} missing get_fourier_coeffs().")

        else:
            raise ValueError(
                f"Unsupported layer type for SpectralNorm: {type(lyr).__name__}"
            )

        # 2) Scaling
        sigma_adj = sigma * jnp.asarray(self.safety_factor, jnp.float32)
        if self.mode == "clip":
            scale = jnp.maximum(1.0, sigma_adj / jnp.asarray(self.target, jnp.float32))
        else:  # 'force'
            scale = jnp.maximum(
                1e-12, sigma_adj / jnp.asarray(self.target, jnp.float32)
            )

        # 3) Apply and forward
        eff = self._scale_layer(lyr, scale)
        try:
            out = eff(x, key=key)
        except TypeError:
            out = eff(x)

        return out, state  # state is passed through untouched


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
            self.l1 = SpectralNorm(core, conv2d_method="circ_embed_opt")
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
