# quantbayes/stochax/vision_common/spectral_surgery.py
from __future__ import annotations
from typing import Any, Callable, Dict, Tuple, Optional

import jax
import equinox as eqx
from equinox import tree_at
import jax.random as jr
import jax.numpy as jnp

try:
    from quantbayes.stochax.layers.spectral_layers import (
        SpectralConv2d,
        RFFTCirculant2D,
        SVDDense,
    )

    _HAS_SPEC = True
except Exception:
    _HAS_SPEC = False


def _safe_setattr(module: eqx.Module, name: str, value: Any) -> eqx.Module:
    """Try tree_at first (when field is part of pytree); if that fails, set statically."""
    try:
        return tree_at(lambda m: getattr(m, name), module, value)
    except TypeError:
        # Field is likely static=True or otherwise non-pytree — set it directly.
        object.__setattr__(module, name, value)
        return module


def _as_2tuple(v):
    """
    Return a (h, w) pair of ints. Handles:
      - int                     -> (v, v)
      - (h, w)                  -> (h, w)
      - ((hl, hr), (wl, wr))    -> (hl+hr, wl+wr)  (per-side padding totals)
    """
    if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, tuple) for x in v):
        hpair, wpair = v
        return (int(hpair[0]) + int(hpair[1]), int(wpair[0]) + int(wpair[1]))
    if (
        isinstance(v, tuple)
        and len(v) == 2
        and all(not isinstance(x, tuple) for x in v)
    ):
        return (int(v[0]), int(v[1]))
    try:
        i = int(v)
        return (i, i)
    except Exception:
        return (1, 1)


def _conv2d_out(h, k, s, p):  # dilation=1
    """
    Output size for one spatial dim.
    If p is a 2-tuple (pl, pr), use pl+pr. If p is int, use 2*p.
    """
    if isinstance(p, tuple):
        pad_total = int(p[0]) + int(p[1])
    else:
        pad_total = 2 * int(p)
    return (int(h) + pad_total - int(k)) // int(s) + 1


def infer_rfft_spatial_map_generic(model: eqx.Module, input_hw: Tuple[int, int]):
    """
    Walk the model and infer (H_in, W_in) seen by each 3x3 s=1 Conv2d.
    Handles Conv2d + MaxPool2d/AvgPool2d stride/padding (including nested per-side tuples).
    """
    H0, W0 = input_hw
    spatial: Dict[str, Tuple[int, int]] = {}

    def walk(obj, pfx, H, W):
        # Conv2d
        if isinstance(obj, eqx.nn.Conv2d):
            kH, kW = obj.weight.shape[-2:]
            sH, sW = _as_2tuple(getattr(obj, "stride", getattr(obj, "strides", (1, 1))))
            pad = getattr(obj, "padding", 0)

            if isinstance(pad, str) and pad.upper() == "SAME":
                H_out = (H + sH - 1) // sH
                W_out = (W + sW - 1) // sW
                pH, pW = (0, 0)
            else:
                pH, pW = _as_2tuple(pad)
                H_out = _conv2d_out(H, int(kH), int(sH), pH)
                W_out = _conv2d_out(W, int(kW), int(sW), pW)

            if (int(kH), int(kW)) == (3, 3) and (int(sH), int(sW)) == (1, 1):
                spatial[pfx] = (H, W)
            return H_out, W_out

        # Pooling
        if isinstance(obj, (eqx.nn.MaxPool2d, eqx.nn.AvgPool2d)):
            kH, kW = _as_2tuple(getattr(obj, "kernel_size", 2))
            raw_stride = getattr(obj, "stride", (kH, kW))
            if raw_stride is None:
                raw_stride = (kH, kW)
            sH, sW = _as_2tuple(raw_stride)
            pH, pW = _as_2tuple(getattr(obj, "padding", 0))
            H_out = _conv2d_out(H, int(kH), int(sH), pH)
            W_out = _conv2d_out(W, int(kW), int(sW), pW)
            return H_out, W_out

        # Containers
        if isinstance(obj, eqx.Module):
            curH, curW = H, W
            for name, child in vars(obj).items():
                curH, curW = walk(child, f"{pfx}.{name}" if pfx else name, curH, curW)
            return curH, curW
        if isinstance(obj, (list, tuple)):
            curH, curW = H, W
            for i, child in enumerate(obj):
                curH, curW = walk(child, f"{pfx}.{i}" if pfx else str(i), curH, curW)
            return curH, curW
        if isinstance(obj, dict):
            curH, curW = H, W
            for k, child in obj.items():
                curH, curW = walk(child, f"{pfx}.{k}" if pfx else k, curH, curW)
            return curH, curW

        return H, W

    walk(model, "", H0, W0)
    return spatial


def replace_modules(
    obj: Any,
    predicate: Callable[[Any, str], bool],
    build: Callable[[Any, str], Any],
    prefix: str = "",
) -> Any:
    """Generic pytree replacer with path prefixes. Robust to static fields and containers."""
    # If this node itself should be replaced, do it immediately.
    if isinstance(obj, eqx.Module) and predicate(obj, prefix):
        return build(obj, prefix)

    if isinstance(obj, eqx.Module):
        new_obj = obj
        for name, child in vars(obj).items():
            full = name if prefix == "" else f"{prefix}.{name}"
            # Try to replace the child directly
            if predicate(child, full):
                new_child = build(child, full)
            else:
                new_child = replace_modules(child, predicate, build, full)
            if new_child is not child:
                new_obj = _safe_setattr(new_obj, name, new_child)
        return new_obj

    if isinstance(obj, tuple):
        new_elems = []
        for i, c in enumerate(obj):
            full = f"{prefix}.{i}" if prefix else str(i)
            if predicate(c, full):
                new_elems.append(build(c, full))
            else:
                new_elems.append(replace_modules(c, predicate, build, full))
        return tuple(new_elems)

    if isinstance(obj, list):
        new_list = []
        for i, c in enumerate(obj):
            full = f"{prefix}.{i}" if prefix else str(i)
            if predicate(c, full):
                new_list.append(build(c, full))
            else:
                new_list.append(replace_modules(c, predicate, build, full))
        return new_list

    if isinstance(obj, dict):
        new_dict = {}
        for k, c in obj.items():
            full = f"{prefix}.{k}" if prefix else k
            if predicate(c, full):
                new_dict[k] = build(c, full)
            else:
                new_dict[k] = replace_modules(c, predicate, build, full)
        return new_dict

    return obj


def _linear_to_svddense(
    fc: eqx.nn.Linear,
    *,
    rank: int | None,
    key: jr.PRNGKey,
    alpha_init: float = 1.0,
    rank_cap: int = 512,
):
    """
    Memory-safe SVDDense init:
      - Use THIN (reduced) QR on tall Gaussian matrices to get U∈R^{out×r}, V∈R^{in×r}
      - Cap rank at `rank_cap` to avoid OOM on huge layers (e.g., VGG fc1: 25088×4096)
    """
    W = fc.weight  # (out, in)
    out_features = int(W.shape[0])
    in_features = int(W.shape[1])

    r_target = min(out_features, in_features)
    if rank is not None:
        r_target = min(r_target, int(rank))
    r = min(r_target, int(rank_cap))  # final economical rank

    kU, kV, ks = jr.split(key, 3)

    # THIN QR: shapes are (out, r) and (in, r) — not (out,out)/(in,in)!
    # jnp.linalg.qr(..., mode="reduced") returns Q with shape (m, k)
    U0 = jr.normal(kU, (out_features, r), W.dtype)
    V0 = jr.normal(kV, (in_features, r), W.dtype)
    U = jnp.linalg.qr(U0, mode="reduced")[0]
    V = jnp.linalg.qr(V0, mode="reduced")[0]
    U = jax.lax.stop_gradient(U)
    V = jax.lax.stop_gradient(V)

    # Small singular values; loader can warm-start later if shapes match.
    s = jr.normal(ks, (r,), W.dtype) * 0.01

    # Carry over bias if present; else zeros.
    if getattr(fc, "bias", None) is not None:
        b = fc.bias
    else:
        b = jnp.zeros((out_features,), W.dtype)

    return SVDDense(U=U, V=V, s=s, bias=b, alpha_init=alpha_init)


# ---------- Targeted policies ----------
def spectralize_resnet_3x3_to_svdconv(
    model: eqx.Module, *, alpha_init: float = 1.0, key: jr.PRNGKey = jr.PRNGKey(0)
) -> eqx.Module:
    """
    Replace 3×3 eqx.nn.Conv2d (any stride) with SpectralConv2d (same stride/padding/channels).
    Warm-start from ImageNet later with spectral_warmstart='svd'.
    """
    if not _HAS_SPEC:
        raise RuntimeError("SpectralConv2d not available.")
    kiter = iter(jr.split(key, 50_000))

    def pred(x, _):
        if not isinstance(x, eqx.nn.Conv2d):
            return False
        kH, kW = x.weight.shape[-2:]
        return (kH, kW) == (3, 3)

    def _as_2tuple(val):
        if isinstance(val, tuple):
            return val
        try:
            iv = int(val)
            return (iv, iv)
        except Exception:
            return (1, 1)

    def build(conv, pfx):
        C_out, C_in, kH, kW = conv.weight.shape
        stride = _as_2tuple(getattr(conv, "stride", getattr(conv, "strides", (1, 1))))
        padding = getattr(conv, "padding", "SAME")
        if hasattr(SpectralConv2d, "from_conv2d"):
            spec = SpectralConv2d.from_conv2d(conv, alpha_init=alpha_init, key=next(kiter))  # type: ignore
        else:
            spec = SpectralConv2d(  # type: ignore
                C_in=C_in,
                C_out=C_out,
                H_k=kH,
                W_k=kW,
                strides=stride,
                padding=padding,
                alpha_init=alpha_init,
                key=next(kiter),
            )
        # tag the module with its path for nicer diagnostics later
        try:
            object.__setattr__(spec, "_name_path", pfx)
        except Exception:
            pass
        return spec

    return replace_modules(model, pred, build)


def _infer_stage_hw(prefix: str, input_hw: Tuple[int, int]) -> Tuple[int, int]:
    """Heuristic for ImageNet-like ResNets (conv1 s=2, pool s=2, then /4,/8,/16,/32)."""
    H, W = input_hw
    if prefix.startswith("conv1"):
        return (H, W)
    if prefix.startswith(("bn1", "pool", "layers1")):
        return (H // 4, W // 4)
    if prefix.startswith("layers2"):
        return (H // 8, W // 8)
    if prefix.startswith("layers3"):
        return (H // 16, W // 16)
    if prefix.startswith("layers4"):
        return (H // 32, W // 32)
    return (H, W)


def spectralize_resnet_3x3_stride1_to_rfft(
    model: eqx.Module,
    *,
    input_hw: Tuple[int, int] = (224, 224),
    key: jr.PRNGKey = jr.PRNGKey(0),
) -> Tuple[eqx.Module, Dict[str, Tuple[int, int]]]:
    """
    Replace 3×3 stride=1 convs with RFFTCirculant2D, recording spatial sizes for FFT warm-start.
    Returns (new_model, rfft_spatial_map) — pass the map into the loader to do FFT warm-start.
    """
    if not _HAS_SPEC:
        raise RuntimeError("RFFTCirculant2D not available.")
    kiter = iter(jr.split(key, 50_000))
    spatial: Dict[str, Tuple[int, int]] = {}

    def pred(x, pfx):
        if not isinstance(x, eqx.nn.Conv2d):
            return False
        kH, kW = x.weight.shape[-2:]
        st = getattr(x, "stride", getattr(x, "strides", (1, 1)))
        st = st if isinstance(st, tuple) else (int(st), int(st))
        return (kH, kW) == (3, 3) and st == (1, 1)

    def build(conv, pfx):
        C_out, C_in, _, _ = conv.weight.shape
        H, W = _infer_stage_hw(pfx, input_hw)
        spatial[pfx] = (H, W)
        if hasattr(RFFTCirculant2D, "from_conv2d"):
            return RFFTCirculant2D.from_conv2d(
                conv,
                H_in=H,
                W_in=W,
                H_pad=H,
                W_pad=W,
                crop_output=True,
                use_soft_mask=True,
                mask_steepness=15.0,
                key=next(kiter),
            )  # type: ignore
        return RFFTCirculant2D(
            C_in=C_in,
            C_out=C_out,
            H_in=H,
            W_in=W,
            H_pad=H,
            W_pad=W,
            crop_output=True,
            use_soft_mask=True,
            mask_steepness=15.0,
            key=next(kiter),
        )  # type: ignore

    new_model = replace_modules(model, pred, build)
    return new_model, spatial


def spectralize_resnet_linear_to_svddense(
    model: eqx.Module,
    *,
    alpha_init: float = 1.0,
    key: jr.PRNGKey = jr.PRNGKey(0),
    rank: int | None = None,  # NEW
    rank_cap: int = 512,  # NEW
) -> eqx.Module:
    """
    Replace every eqx.nn.Linear with SVDDense using the safe internal builder.
    r = min(in_features, out_features, rank_cap) by default, unless `rank` is set.
    """
    if not _HAS_SPEC:
        raise RuntimeError("SVDDense not available.")

    kiter = iter(jr.split(key, 50_000))

    def pred(x, _):
        return isinstance(x, eqx.nn.Linear)

    def build(lin: eqx.nn.Linear, _):
        return _linear_to_svddense(
            lin,
            rank=rank,  # <- pass explicit rank if provided
            key=next(kiter),
            alpha_init=alpha_init,
            rank_cap=rank_cap,
        )

    return replace_modules(model, pred, build)


def spectralize_cnn_3x3_stride1_to_rfft(
    model: eqx.Module,
    *,
    input_hw: Tuple[int, int] = (224, 224),
    key: jr.PRNGKey = jr.PRNGKey(0),
) -> Tuple[eqx.Module, Dict[str, Tuple[int, int]]]:
    """
    CNN-agnostic: replace 3×3 stride=1 Conv2d with RFFTCirculant2D using auto-inferred (H,W).
    Returns (new_model, spatial_map) for optional FFT warm-start.
    """
    if not _HAS_SPEC:
        raise RuntimeError("RFFTCirculant2D not available.")
    spatial = infer_rfft_spatial_map_generic(model, input_hw)
    kiter = iter(jr.split(key, 50_000))

    def pred(x, pfx):
        if not isinstance(x, eqx.nn.Conv2d):
            return False
        kH, kW = x.weight.shape[-2:]
        sH, sW = _as_2tuple(getattr(x, "stride", getattr(x, "strides", (1, 1))))
        return (int(kH), int(kW)) == (3, 3) and (int(sH), int(sW)) == (1, 1)

    def build(conv, pfx):
        C_out, C_in, _, _ = conv.weight.shape
        H, W = spatial.get(pfx, input_hw)  # robust fallback
        if hasattr(RFFTCirculant2D, "from_conv2d"):
            return RFFTCirculant2D.from_conv2d(
                conv,
                H_in=H,
                W_in=W,
                H_pad=H,
                W_pad=W,
                crop_output=True,
                use_soft_mask=True,
                mask_steepness=15.0,
                key=next(kiter),
            )
        return RFFTCirculant2D(
            C_in=C_in,
            C_out=C_out,
            H_in=H,
            W_in=W,
            H_pad=H,
            W_pad=W,
            crop_output=True,
            use_soft_mask=True,
            mask_steepness=15.0,
            key=next(kiter),
        )

    new_model = replace_modules(model, pred, build)
    return new_model, spatial


def spectralize_cnn_3x3_stride1_to_rfft_and_warmstart(
    vanilla_model: eqx.Module,
    *,
    input_hw: Tuple[int, int] = (224, 224),
    key: jr.PRNGKey = jr.PRNGKey(0),
):
    """
    1) CNN-agnostic RFFT replacement
    2) Warm-start each RFFT conv from the *vanilla* conv at the same path via FFT
    Returns (spec_model, report_dict, spatial_map) + prints coverage among 3×3 s=1.
    """
    spec, spatial = spectralize_cnn_3x3_stride1_to_rfft(
        vanilla_model, input_hw=input_hw, key=key
    )
    spec, rep = warmstart_rfft_from_vanilla(vanilla_model, spec, rfft_spatial=spatial)

    total_3x3s1 = 0
    for _, node in _iter_with_paths(vanilla_model):
        if isinstance(node, eqx.nn.Conv2d):
            kH, kW = node.weight.shape[-2:]
            sH, sW = _as_2tuple(
                getattr(node, "stride", getattr(node, "strides", (1, 1)))
            )
            if (int(kH), int(kW)) == (3, 3) and (int(sH), int(sW)) == (1, 1):
                total_3x3s1 += 1
    replaced = sum(
        1 for _, n in _iter_with_paths(spec) if isinstance(n, RFFTCirculant2D)
    )
    rep = {
        **rep,
        "replaced": replaced,
        "total_candidates": total_3x3s1,
        "coverage": (replaced / total_3x3s1) if total_3x3s1 > 0 else 0.0,
    }
    print(
        f"[surgery:rfft:generic] 3x3 s=1 convs: total={total_3x3s1} "
        f"replaced={replaced} coverage={rep['coverage']:.1%}"
    )
    return spec, rep, spatial


def _flatten_conv_OIHW_to_mat(W):  # (C_out, C_in, H, W) -> (C_out, C_in*H*W)
    Co, Ci, H, Wk = W.shape
    return W.reshape(Co, Ci * H * Wk)


def _svd_truncate(Wmat, r=None):
    U, s, Vh = jnp.linalg.svd(Wmat, full_matrices=False)
    if r is not None:
        r = int(min(r, s.shape[0]))
        U, s, Vh = U[:, :r], s[:r], Vh[:r, :]
    return U, s, Vh.T  # return V (CiHW, r)


# ---------- tree walking utilities ----------


def _iter_with_paths(obj, prefix: str = ""):
    """Yield (path, node) over modules/containers; paths use '.' and integer indices."""
    import equinox as eqx

    if isinstance(obj, eqx.Module):
        yield prefix, obj
        for k, v in vars(obj).items():
            p = f"{prefix}.{k}" if prefix else k
            yield from _iter_with_paths(v, p)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            p = f"{prefix}.{i}" if prefix else str(i)
            yield from _iter_with_paths(v, p)
    elif isinstance(obj, dict):
        for k in obj:
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from _iter_with_paths(obj[k], p)


def _get_by_path(root, path: str):
    """Follow a '.' path through attributes / list/tuple indices / dict keys."""
    cur = root
    if not path:
        return cur
    for tok in path.split("."):
        if tok == "":
            continue
        if isinstance(cur, (list, tuple)):
            cur = cur[int(tok)]
        elif isinstance(cur, dict):
            cur = cur[tok]
        else:
            cur = getattr(cur, tok)
    return cur


def _count_3x3_convs(m) -> int:
    import equinox as eqx

    n = 0
    for _, node in _iter_with_paths(m):
        if isinstance(node, eqx.nn.Conv2d):
            kH, kW = node.weight.shape[-2:]
            if (kH, kW) == (3, 3):
                n += 1
    return n


def _count_spectral_convs(m) -> int:
    n = 0
    for _, node in _iter_with_paths(m):
        if isinstance(node, SpectralConv2d):
            n += 1
    return n


def _fft2_kernel_from_conv_weight(
    W: jnp.ndarray, H_pad: int, W_pad: int
) -> jnp.ndarray:
    """(Cout,Cin,Hk,Wk) -> RFFT2 half-plane at (H_pad,W_pad), with circular centering."""
    Cout, Cin, Hk, Wk = map(int, W.shape)
    k = jnp.zeros((Cout, Cin, H_pad, W_pad), W.dtype)
    k = k.at[:, :, :Hk, :Wk].set(W)
    # center kernel for circular conv convention
    k = jnp.roll(k, shift=(-Hk // 2, -Wk // 2), axis=(-2, -1))
    return jnp.fft.rfft2(k, s=(H_pad, W_pad), axes=(-2, -1), norm="ortho")


def warmstart_rfft_from_vanilla(
    model_vanilla: eqx.Module,
    model_rfft: eqx.Module,
    *,
    rfft_spatial: Optional[Dict[str, Tuple[int, int]]] = None,
):
    """
    For each RFFTCirculant2D in model_rfft, pull the vanilla Conv2d at the same path,
    build an RFFT2 kernel from its spatial weights (using (H,W) from rfft_spatial if
    provided, else from the layer’s own H_pad/W_pad), and copy bias.
    Prints a short report and returns (model_rfft, report_dict).
    """
    tried = warmed = 0
    for path, spec in _iter_with_paths(model_rfft):
        if not isinstance(spec, RFFTCirculant2D):
            continue
        try:
            vanilla = _get_by_path(model_vanilla, path)
        except Exception:
            continue
        if not isinstance(vanilla, eqx.nn.Conv2d):
            continue

        W = getattr(vanilla, "weight", None)
        b = getattr(vanilla, "bias", None)
        if W is None or b is None:
            continue

        # decide FFT size
        if (rfft_spatial is not None) and (path in rfft_spatial):
            H, Wp = rfft_spatial[path]
        else:
            H, Wp = int(getattr(spec, "H_pad")), int(getattr(spec, "W_pad"))

        K = _fft2_kernel_from_conv_weight(W, H, Wp).astype(spec.K_half.dtype)
        object.__setattr__(spec, "K_half", K)
        if hasattr(spec, "bias"):
            object.__setattr__(spec, "bias", b.astype(spec.bias.dtype))

        tried += 1
        warmed += 1

    rep = {"tried": tried, "warmed": warmed, "skipped": tried - warmed}
    print(
        f"[rfft warmstart:path] candidates={tried} warmed={warmed} skipped={rep['skipped']}"
    )
    return model_rfft, rep


# ---------- robust, path-based SVD warm-start ----------


def warmstart_svd_from_vanilla(model_vanilla, model_spectral, *, eps=1e-6):
    """
    For each SpectralConv2d in the spectralized model, fetch the *vanilla* layer at the same path
    and set (U, s, V, bias) from the SVD of the vanilla conv weight if it's a 3x3.
    Returns (model_spectral, {"tried":..., "warmed":..., "skipped":...}).
    """
    import equinox as eqx

    tried = warmed = 0
    for path, spec in _iter_with_paths(model_spectral):
        if not isinstance(spec, SpectralConv2d):
            continue

        # locate vanilla node at identical path
        try:
            vanilla = _get_by_path(model_vanilla, path)
        except Exception:
            continue
        if not isinstance(vanilla, eqx.nn.Conv2d):
            continue

        W = getattr(vanilla, "weight", None)
        b = getattr(vanilla, "bias", None)
        if W is None or b is None:
            continue
        Co, Ci, kH, kW = W.shape
        if (kH, kW) != (3, 3):
            continue

        # strict dim check
        if not (
            Ci == spec.C_in and Co == spec.C_out and kH == spec.H_k and kW == spec.W_k
        ):
            continue

        tried += 1
        Wmat = _flatten_conv_OIHW_to_mat(W)
        r_use = int(getattr(spec, "rank", min(Wmat.shape)))
        U, s, V = _svd_truncate(Wmat, r_use)
        s = jnp.where(s < eps, eps, s)

        # in-place set on the spectral layer
        object.__setattr__(spec, "U", U.astype(spec.U.dtype))
        object.__setattr__(spec, "s", s.astype(spec.s.dtype))
        object.__setattr__(spec, "V", V.astype(spec.V.dtype))
        object.__setattr__(spec, "bias", b.astype(spec.bias.dtype))
        warmed += 1

    rep = {"tried": tried, "warmed": warmed, "skipped": tried - warmed}
    print(
        f"[svd warmstart:path] candidates={tried} warmed={warmed} skipped={rep['skipped']}"
    )
    return model_spectral, rep


def spectralize_resnet_3x3_to_svdconv_and_warmstart(
    vanilla_model: eqx.Module,
    *,
    alpha_init: float = 1.0,
    key: jr.PRNGKey = jr.PRNGKey(0),
):
    """
    1) replace all 3×3 Conv2d with SpectralConv2d
    2) warm-start each spectral conv from the *vanilla* conv at the same path.
    Also reports replacement coverage.
    """
    total_3x3 = _count_3x3_convs(vanilla_model)
    spec = spectralize_resnet_3x3_to_svdconv(
        vanilla_model, alpha_init=alpha_init, key=key
    )
    replaced = _count_spectral_convs(spec)
    coverage = (replaced / total_3x3) if total_3x3 > 0 else 0.0
    print(
        f"[surgery] 3x3 convs: total={total_3x3} replaced={replaced} coverage={coverage:.1%}"
    )

    spec, rep = warmstart_svd_from_vanilla(vanilla_model, spec)
    rep = {
        **rep,
        "coverage": coverage,
        "replaced": replaced,
        "total_candidates": total_3x3,
    }
    return spec, rep


def spectralize_resnet_3x3_stride1_to_rfft_and_warmstart(
    vanilla_model: eqx.Module,
    *,
    input_hw: Tuple[int, int] = (224, 224),
    key: jr.PRNGKey = jr.PRNGKey(0),
):
    """
    1) replace all 3×3 stride=1 Conv2d with RFFTCirculant2D (H_pad=W_pad=stage size),
    2) warm-start each RFFT conv from the *vanilla* conv at the same path using FFT.

    Returns (spec_model, report_dict, spatial_map).
    """
    spec, spatial = spectralize_resnet_3x3_stride1_to_rfft(
        vanilla_model, input_hw=input_hw, key=key
    )
    spec, rep = warmstart_rfft_from_vanilla(vanilla_model, spec, rfft_spatial=spatial)

    # coverage report (only among 3×3 stride=1)
    total_3x3s1 = 0
    for _, node in _iter_with_paths(vanilla_model):
        if isinstance(node, eqx.nn.Conv2d):
            kH, kW = node.weight.shape[-2:]
            st = getattr(node, "stride", getattr(node, "strides", (1, 1)))
            st = st if isinstance(st, tuple) else (int(st), int(st))
            if (kH, kW) == (3, 3) and st == (1, 1):
                total_3x3s1 += 1
    replaced = sum(
        1 for _, n in _iter_with_paths(spec) if isinstance(n, RFFTCirculant2D)
    )
    rep = {
        **rep,
        "replaced": replaced,
        "total_candidates": total_3x3s1,
        "coverage": (replaced / total_3x3s1) if total_3x3s1 > 0 else 0.0,
    }
    print(
        f"[surgery:rfft] 3x3 s=1 convs: total={total_3x3s1} replaced={replaced} "
        f"coverage={rep['coverage']:.1%}"
    )
    return spec, rep, spatial
