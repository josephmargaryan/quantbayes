# quantbayes/stochax/layers/specnorm.py

from __future__ import annotations

import inspect
from typing import Any, Literal, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import tree_at

from quantbayes.stochax.utils.regularizers import (
    _sigma_conv_TN_upper,
    _sigma_conv_circ_embed_opt_upper_2d,
    _sigma_conv_circ_plus_lr_upper_2d,
    _sigma_conv_circular_fft_exact_2d,
    _sigma_conv_circular_gram_upper_2d,
    _sigma_conv_circulant_embed_upper_2d,
)

_STATE_SENTINEL = object()

_DENSE_S_NAMES = {"SVDDense", "SpectralDense", "AdaptiveSpectralDense"}
_PARAM_SVD_CONV_NAMES = {"SpectralConv2d", "AdaptiveSpectralConv2d"}
_RFFT_CIRC_1D_NAMES = {"RFFTCirculant1D"}
_RFFT_CIRC_2D_NAMES = {"RFFTCirculant2D"}
_TOKEN_MIXER_NAMES = {"SpectralTokenMixer"}
_SPECTRAL_CIRC_1D_NAMES = {
    "SpectralCirculantLayer",
    "AdaptiveSpectralCirculantLayer",
}
_SPECTRAL_CIRC_2D_NAMES = {
    "SpectralCirculantLayer2d",
    "AdaptiveSpectralCirculantLayer2d",
}
_SCALE_BY_S_NAMES = _DENSE_S_NAMES | _PARAM_SVD_CONV_NAMES
_SCALE_BY_H_HALF_NAMES = _RFFT_CIRC_1D_NAMES | _TOKEN_MIXER_NAMES
_SCALE_BY_K_HALF_NAMES = _RFFT_CIRC_2D_NAMES
_SCALE_BY_W_COMPLEX_NAMES = _SPECTRAL_CIRC_1D_NAMES | _SPECTRAL_CIRC_2D_NAMES


def _accepts_keyword_arg(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    return (name in params) or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _call_layer(layer, x, *, key=None, state=_STATE_SENTINEL):
    fn = layer.__call__ if hasattr(layer, "__call__") else layer
    kwargs = {}
    if state is not _STATE_SENTINEL and _accepts_keyword_arg(fn, "state"):
        kwargs["state"] = state
    if key is not None and _accepts_keyword_arg(fn, "key"):
        kwargs["key"] = key
    out = layer(x, **kwargs) if kwargs else layer(x)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    if state is _STATE_SENTINEL:
        return out
    return out, state


def _type_name(x) -> str:
    return type(x).__name__


def _is_name(x, names: Sequence[str]) -> bool:
    return _type_name(x) in set(names)


def _flatten_matrix(W: jnp.ndarray) -> jnp.ndarray:
    return W.reshape(W.shape[0], -1)


def _top_sigma_matrix_svd(W2: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.svd(W2, compute_uv=False)[0].astype(jnp.float32)


def _max_abs(z: jnp.ndarray) -> jnp.ndarray:
    return jnp.max(jnp.abs(z))


def _is_eqx_linear(x) -> bool:
    return hasattr(eqx.nn, "Linear") and isinstance(x, eqx.nn.Linear)


def _is_eqx_conv(x) -> bool:
    return hasattr(eqx.nn, "Conv") and isinstance(x, eqx.nn.Conv)


def _is_eqx_convT(x) -> bool:
    return hasattr(eqx.nn, "ConvTranspose") and isinstance(x, eqx.nn.ConvTranspose)


def _is_param_svd_conv(x) -> bool:
    return _type_name(x) in _PARAM_SVD_CONV_NAMES


def _needs_conv_fft_shape(method: str) -> bool:
    return method in {"circular_fft", "circular_gram"}


def _needs_conv_input_shape(method: str) -> bool:
    return method in {"min_tn_circ_embed", "circ_plus_lr", "circ_embed_opt"}


def _stride_dilation_2d(mod):
    W = getattr(mod, "weight", None)
    nd = getattr(mod, "num_spatial_dims", (W.ndim - 2) if W is not None else 2)

    stride = getattr(mod, "stride", getattr(mod, "strides", (1, 1)))
    if isinstance(stride, int):
        stride = (stride, stride)

    dilation = (
        getattr(mod, "rhs_dilation", None)
        or getattr(mod, "dilation", None)
        or getattr(mod, "kernel_dilation", (1, 1))
    )
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    return (
        int(nd),
        (int(stride[0]), int(stride[1])),
        (int(dilation[0]), int(dilation[1])),
    )


class SpectralNorm(eqx.Module):
    """Certified spectral projection wrapper.

    This wrapper does not rescale during the forward pass. Instead, call
    `.project()` explicitly after optimizer updates to keep the wrapped
    parameters inside the certified operator-norm ball.

    Forward passes are transparent: they delegate to the wrapped layer and
    preserve optional `(output, state)` calling conventions for interoperability.
    """

    layer: eqx.Module
    target: float = eqx.field(static=True)
    mode: Literal["clip", "force"] = eqx.field(static=True)
    detach_scale: bool = eqx.field(static=True)
    safety_factor: float = eqx.field(static=True)

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

    conv_tn_iters: int = eqx.field(static=True)
    conv_gram_iters: int = eqx.field(static=True)
    conv_fft_shape: Optional[tuple[int, int]] = eqx.field(static=True)
    conv_input_shape: Optional[tuple[int, int]] = eqx.field(static=True)

    def __init__(
        self,
        layer: eqx.Module,
        *,
        target: float = 1.0,
        mode: Literal["clip", "force"] = "clip",
        detach_scale: bool = True,
        safety_factor: float = 1.0,
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
        conv_fft_shape: Optional[tuple[int, int]] = None,
        conv_input_shape: Optional[tuple[int, int]] = None,
        inference: Optional[bool] = None,
        enforce_on_forward: Optional[bool] = None,
    ):
        if enforce_on_forward not in (None, False):
            raise ValueError(
                "SpectralNorm no longer supports enforce_on_forward=True. "
                "Apply `.project()` after optimizer updates instead."
            )
        if inference not in (None, False, True):
            raise ValueError("inference must be True, False, or None.")

        self.layer = layer
        self.target = float(target)
        self.mode = mode
        self.detach_scale = bool(detach_scale)
        self.safety_factor = float(safety_factor)

        self.conv2d_method = conv2d_method
        self.param_svd_conv2d_method = param_svd_conv2d_method
        self.conv_tn_iters = int(conv_tn_iters)
        self.conv_gram_iters = int(conv_gram_iters)
        self.conv_fft_shape = (
            None
            if conv_fft_shape is None
            else (int(conv_fft_shape[0]), int(conv_fft_shape[1]))
        )
        self.conv_input_shape = (
            None
            if conv_input_shape is None
            else (int(conv_input_shape[0]), int(conv_input_shape[1]))
        )

        lyr = self.layer
        if _is_eqx_conv(lyr) or _is_eqx_convT(lyr):
            if (
                _needs_conv_fft_shape(self.conv2d_method)
                and self.conv_fft_shape is None
            ):
                raise ValueError(
                    f"{self.conv2d_method} requires conv_fft_shape=(Hf, Wf) for conv layers."
                )
            if (
                _needs_conv_input_shape(self.conv2d_method)
                and self.conv_input_shape is None
            ):
                raise ValueError(
                    f"{self.conv2d_method} requires conv_input_shape=(Hin, Win) for conv layers."
                )

        if _is_param_svd_conv(lyr):
            if (
                _needs_conv_fft_shape(self.param_svd_conv2d_method)
                and self.conv_fft_shape is None
            ):
                raise ValueError(
                    f"param_svd_conv2d_method={self.param_svd_conv2d_method} requires conv_fft_shape=(Hf, Wf)."
                )
            if (
                _needs_conv_input_shape(self.param_svd_conv2d_method)
                and self.conv_input_shape is None
            ):
                raise ValueError(
                    f"param_svd_conv2d_method={self.param_svd_conv2d_method} requires conv_input_shape=(Hin, Win)."
                )

    def _sigma_linear_exact(self, lin: eqx.nn.Linear) -> jnp.ndarray:
        return _top_sigma_matrix_svd(_flatten_matrix(lin.weight))

    def _conv_common_params(
        self, conv
    ) -> tuple[jnp.ndarray, int, tuple[int, int], tuple[int, int]]:
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

    def _resolve_input_hw(
        self, method: str, runtime_hw: Optional[tuple[int, int]] = None
    ) -> tuple[int, int]:
        hw = self.conv_input_shape if self.conv_input_shape is not None else runtime_hw
        if hw is None:
            raise ValueError(f"{method} requires conv_input_shape=(Hin, Win).")
        return (int(hw[0]), int(hw[1]))

    def _resolve_fft_hw(
        self, method: str, runtime_hw: Optional[tuple[int, int]] = None
    ) -> tuple[int, int]:
        hw = self.conv_fft_shape if self.conv_fft_shape is not None else runtime_hw
        if hw is None:
            raise ValueError(f"{method} requires conv_fft_shape=(Hf, Wf).")
        return (int(hw[0]), int(hw[1]))

    def _sigma_conv2d_weight(
        self,
        W: jnp.ndarray,
        stride: tuple[int, int],
        dilation: tuple[int, int],
        *,
        method: Optional[str] = None,
        runtime_hw: Optional[tuple[int, int]] = None,
    ) -> jnp.ndarray:
        sh, sw = stride
        dh, dw = dilation
        method = self.conv2d_method if method is None else method

        if method == "tn":
            return _sigma_conv_TN_upper(
                W,
                strides=(sh, sw),
                rhs_dilation=(dh, dw),
                iters=self.conv_tn_iters,
                certify=True,
            ).astype(jnp.float32)

        if method == "circ_embed_opt":
            if (sh, sw) != (1, 1):
                raise ValueError(
                    "circ_embed_opt is only valid for stride=(1,1). Use 'tn' or 'circ_plus_lr' for strided convs."
                )
            in_hw = self._resolve_input_hw(method, runtime_hw)
            return _sigma_conv_circ_embed_opt_upper_2d(
                W, in_hw=in_hw, rhs_dilation=(dh, dw)
            ).astype(jnp.float32)

        if method == "circular_fft":
            if (sh, sw) != (1, 1):
                raise ValueError(
                    "circular_fft is exact only for stride=(1,1). Use 'tn' or 'circ_plus_lr'."
                )
            fft_hw = self._resolve_fft_hw(method, runtime_hw)
            return _sigma_conv_circular_fft_exact_2d(
                W, fft_hw, rhs_dilation=(dh, dw)
            ).astype(jnp.float32)

        if method == "circular_gram":
            if (sh, sw) != (1, 1):
                raise ValueError(
                    "circular_gram is supported only for stride=(1,1). Use 'tn' or 'circ_plus_lr'."
                )
            fft_hw = self._resolve_fft_hw(method, runtime_hw)
            return _sigma_conv_circular_gram_upper_2d(
                W,
                fft_hw,
                rhs_dilation=(dh, dw),
                iters=self.conv_gram_iters,
            ).astype(jnp.float32)

        if method == "min_tn_circ_embed":
            if (sh, sw) != (1, 1) or (dh, dw) != (1, 1):
                raise ValueError(
                    "min_tn_circ_embed is only valid for stride=(1,1), dilation=(1,1)."
                )
            in_hw = self._resolve_input_hw(method, runtime_hw)
            tn = _sigma_conv_TN_upper(
                W,
                strides=(1, 1),
                rhs_dilation=(1, 1),
                iters=self.conv_tn_iters,
                certify=True,
            )
            ce = _sigma_conv_circulant_embed_upper_2d(W, in_hw=in_hw)
            return jnp.minimum(tn, ce).astype(jnp.float32)

        if method == "circ_plus_lr":
            in_hw = self._resolve_input_hw(method, runtime_hw)
            return _sigma_conv_circ_plus_lr_upper_2d(
                W, in_hw=in_hw, strides=(sh, sw), rhs_dilation=(dh, dw)
            ).astype(jnp.float32)

        raise ValueError(f"Unknown conv2d spectral certificate method: {method}")

    def _sigma_conv(
        self, conv, *, runtime_hw: Optional[tuple[int, int]] = None
    ) -> jnp.ndarray:
        W, nd, stride, dilation = self._conv_common_params(conv)
        if nd != 2:
            raise ValueError("This SpectralNorm supports certified convs for 2D only.")
        return self._sigma_conv2d_weight(W, stride, dilation, runtime_hw=runtime_hw)

    def _sigma_convT(
        self, convT, *, runtime_hw: Optional[tuple[int, int]] = None
    ) -> jnp.ndarray:
        W = convT.weight
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
            Wc,
            (int(stride[0]), int(stride[1])),
            (int(dilation[0]), int(dilation[1])),
            runtime_hw=runtime_hw,
        )

    def _sigma_param_svd_conv2d(
        self, mod, *, runtime_hw: Optional[tuple[int, int]] = None
    ) -> jnp.ndarray:
        Co, Ci, Kh, Kw = int(mod.C_out), int(mod.C_in), int(mod.H_k), int(mod.W_k)
        W_mat = mod.U @ (mod.s[:, None] * mod.V.T)
        W = W_mat.reshape(Co, Ci, Kh, Kw)
        stride = getattr(mod, "strides", (1, 1))
        if isinstance(stride, int):
            stride = (stride, stride)
        return jnp.asarray(
            self._sigma_conv2d_weight(
                W,
                (int(stride[0]), int(stride[1])),
                (1, 1),
                method=self.param_svd_conv2d_method,
                runtime_hw=runtime_hw,
            ),
            jnp.float32,
        )

    def _sigma_rfft_circ1d(self, mod) -> jnp.ndarray:
        return _max_abs(mod.H_half).astype(jnp.float32)

    def _sigma_token_mixer(self, mod) -> jnp.ndarray:
        H = mod.H_half
        s_g = jnp.max(jnp.abs(H), axis=1)
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

    def _sigma_rfft_circ2d_svd(self, K_half: jnp.ndarray) -> jnp.ndarray:
        mats = jnp.transpose(K_half, (2, 3, 0, 1)).reshape(
            -1, K_half.shape[0], K_half.shape[1]
        )
        svals = jax.vmap(lambda M: jnp.linalg.svd(M, compute_uv=False)[0])(mats)
        return jnp.max(jnp.real(svals)).astype(jnp.float32)

    def _certified_sigma(
        self,
        lyr: Optional[eqx.Module] = None,
        *,
        runtime_hw: Optional[tuple[int, int]] = None,
    ) -> jnp.ndarray:
        lyr = self.layer if lyr is None else lyr
        name = _type_name(lyr)

        if _is_eqx_linear(lyr):
            return self._sigma_linear_exact(lyr)
        if _is_eqx_conv(lyr):
            return self._sigma_conv(lyr, runtime_hw=runtime_hw)
        if _is_eqx_convT(lyr):
            return self._sigma_convT(lyr, runtime_hw=runtime_hw)
        if name in _DENSE_S_NAMES:
            return jnp.max(jnp.abs(lyr.s)).astype(jnp.float32)
        if name in _PARAM_SVD_CONV_NAMES:
            return self._sigma_param_svd_conv2d(lyr, runtime_hw=runtime_hw)
        if name in _RFFT_CIRC_1D_NAMES:
            return self._sigma_rfft_circ1d(lyr)
        if name in _TOKEN_MIXER_NAMES:
            return self._sigma_token_mixer(lyr)
        if name in _RFFT_CIRC_2D_NAMES:
            return self._sigma_rfft_circ2d_svd(lyr.K_half)
        if name in _SPECTRAL_CIRC_2D_NAMES:
            K_full = lyr.get_fft_kernel()
            W_half = (K_full.shape[-1] // 2) + 1
            K_half = K_full[..., :W_half]
            return self._sigma_rfft_circ2d_svd(K_half)
        if name in _SPECTRAL_CIRC_1D_NAMES:
            if hasattr(lyr, "get_fourier_coeffs"):
                return _max_abs(lyr.get_fourier_coeffs()).astype(jnp.float32)
            raise ValueError(f"{name} missing get_fourier_coeffs().")
        raise ValueError(f"Unsupported layer type for SpectralNorm: {name}")

    def _compute_scale(self, sigma: jnp.ndarray) -> jnp.ndarray:
        sigma_adj = sigma * jnp.asarray(self.safety_factor, jnp.float32)
        if self.mode == "clip":
            return jnp.maximum(1.0, sigma_adj / jnp.asarray(self.target, jnp.float32))
        return jnp.maximum(1e-12, sigma_adj / jnp.asarray(self.target, jnp.float32))

    def _scale_layer(self, lyr: eqx.Module, scale: jnp.ndarray) -> eqx.Module:
        if self.detach_scale:
            scale = jax.lax.stop_gradient(scale)

        name = _type_name(lyr)

        if _is_eqx_linear(lyr) or _is_eqx_conv(lyr) or _is_eqx_convT(lyr):
            return tree_at(lambda m: m.weight, lyr, lyr.weight / scale)
        if name in _SCALE_BY_S_NAMES:
            return tree_at(lambda m: m.s, lyr, lyr.s / scale)
        if name in _SCALE_BY_H_HALF_NAMES:
            return tree_at(lambda m: m.H_half, lyr, lyr.H_half / scale)
        if name in _SCALE_BY_K_HALF_NAMES:
            return tree_at(lambda m: m.K_half, lyr, lyr.K_half / scale)
        if name in _SCALE_BY_W_COMPLEX_NAMES:
            new = tree_at(lambda m: m.w_real, lyr, lyr.w_real / scale)
            new = tree_at(lambda m: m.w_imag, new, lyr.w_imag / scale)
            return new
        raise ValueError(f"Unsupported layer type for SpectralNorm: {name}")

    def project(self):
        projected_layer = self._scale_layer(
            self.layer,
            self._compute_scale(self._certified_sigma(self.layer)),
        )
        return tree_at(lambda m: m.layer, self, projected_layer)

    def __operator_norm_hint__(self) -> jnp.ndarray | None:
        if not (self.target > 0 and self.safety_factor >= 1.0):
            return None

        cap = jnp.asarray(self.target / self.safety_factor, jnp.float32)
        lyr = self.layer

        def _ok(
            method: str, stride: tuple[int, int], dilation: tuple[int, int]
        ) -> bool:
            if method == "tn":
                return True
            if method in {"circular_fft", "circular_gram"}:
                return stride == (1, 1) and (self.conv_fft_shape is not None)
            if method == "min_tn_circ_embed":
                return (
                    stride == (1, 1)
                    and dilation == (1, 1)
                    and (self.conv_input_shape is not None)
                )
            if method == "circ_plus_lr":
                return self.conv_input_shape is not None
            if method == "circ_embed_opt":
                return stride == (1, 1) and (self.conv_input_shape is not None)
            return False

        if _is_eqx_linear(lyr):
            return cap

        if _is_eqx_conv(lyr) or _is_eqx_convT(lyr):
            nd, stride, dilation = _stride_dilation_2d(lyr)
            if nd != 2:
                return None
            return cap if _ok(self.conv2d_method, stride, dilation) else None

        if _is_param_svd_conv(lyr):
            stride = getattr(lyr, "strides", (1, 1))
            if isinstance(stride, int):
                stride = (stride, stride)
            stride = (int(stride[0]), int(stride[1]))
            return cap if _ok(self.param_svd_conv2d_method, stride, (1, 1)) else None

        if _type_name(lyr) in (
            _DENSE_S_NAMES
            | _RFFT_CIRC_1D_NAMES
            | _RFFT_CIRC_2D_NAMES
            | _TOKEN_MIXER_NAMES
            | _SPECTRAL_CIRC_1D_NAMES
            | _SPECTRAL_CIRC_2D_NAMES
        ):
            return cap

        if hasattr(lyr, "__operator_norm_hint__"):
            try:
                v = lyr.__operator_norm_hint__()
                return None if v is None else jnp.asarray(v, jnp.float32)
            except Exception:
                return None

        return None

    def __call__(
        self,
        x: jnp.ndarray,
        state: Any = _STATE_SENTINEL,
        *,
        key: Any | None = None,
        inference: Optional[bool] = None,
    ):
        del inference
        return _call_layer(self.layer, x, key=key, state=state)


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
