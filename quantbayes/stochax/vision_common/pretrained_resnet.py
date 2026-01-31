# quantbayes/stochax/vision_common/pretrained_resnet.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Literal, Optional

import numpy as np
import jax.numpy as jnp
import equinox as eqx
from equinox import tree_at


# ---- Optional spectral imports (if present in your repo) ----
_HAS_SPECTRAL = False
SpectralConv2d = SVDDense = RFFTCirculant2D = object  # placeholders

try:
    from quantbayes.stochax.layers.spectral_layers import SpectralConv2d, SVDDense

    _HAS_SPECTRAL = True
except Exception:
    pass

try:
    from quantbayes.stochax.layers.spectral_layers import RFFTCirculant2D

    _HAS_SPECTRAL = True
except Exception:
    pass


# ---- torchvision -> equinox key mapping ----
def _rename_pt_key_resnet(k: str) -> str:
    k = k.replace("downsample.0.", "down_conv.")
    k = k.replace("downsample.1.", "down_bn.")
    k = k.replace("layer1.", "layers1.")
    k = k.replace("layer2.", "layers2.")
    k = k.replace("layer3.", "layers3.")
    k = k.replace("layer4.", "layers4.")
    return k  # conv1., bn1., fc.* unchanged


def _keep_weight_bias(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        k: v for k, v in raw.items() if k.endswith(".weight") or k.endswith(".bias")
    }


def _build_keymap(npz_path: str) -> Dict[str, jnp.ndarray]:
    raw = dict(np.load(npz_path))
    raw = _keep_weight_bias(raw)
    return {_rename_pt_key_resnet(k): jnp.asarray(v) for k, v in raw.items()}


# ---- spectral warm-start helpers ----
def _svd_linear(W: jnp.ndarray):
    U, s, Vh = jnp.linalg.svd(W, full_matrices=False)
    return U, s, Vh.T


def _warmstart_svddense(mod, W, b):
    import jax

    U, s, V = _svd_linear(W)
    r = min(len(s), getattr(mod, "s").shape[0])
    U = jax.lax.stop_gradient(U[:, :r].astype(mod.U.dtype))
    V = jax.lax.stop_gradient(V[:, :r].astype(mod.V.dtype))
    s = s[:r].astype(mod.s.dtype)

    new = tree_at(lambda m: m.U, mod, U)
    new = tree_at(lambda m: m.V, new, V)
    new = tree_at(lambda m: m.s, new, s)
    if hasattr(new, "bias") and (b is not None):
        new = tree_at(lambda m: m.bias, new, b.astype(new.bias.dtype))
    return new


def _warmstart_svdconv(mod: Any, W: jnp.ndarray, b: Optional[jnp.ndarray], report):
    """Warm-start a SpectralConv2d from a vanilla conv weight ONLY if shapes match.

    We require:  W.shape == (mod.C_out, mod.C_in, mod.H_k, mod.W_k).
    Otherwise we record a skip and return the module unchanged.
    """
    import jax

    want = (
        int(getattr(mod, "C_out")),
        int(getattr(mod, "C_in")),
        int(getattr(mod, "H_k")),
        int(getattr(mod, "W_k")),
    )
    have = tuple(int(x) for x in W.shape)
    if want != have:
        report["spectral_skipped"].append(("svdconv", have, want))
        return mod

    Cout, Cin, Hk, Wk = have
    U, s, Vh = jnp.linalg.svd(W.reshape(Cout, Cin * Hk * Wk), full_matrices=False)
    r = min(len(s), getattr(mod, "s").shape[0])

    # Freeze orthonormal bases: exactness & stability
    U = jax.lax.stop_gradient(U[:, :r].astype(mod.U.dtype))
    V = jax.lax.stop_gradient(Vh.T[:, :r].astype(mod.V.dtype))
    s = s[:r].astype(mod.s.dtype)

    new = tree_at(lambda m: m.U, mod, U)
    new = tree_at(lambda m: m.V, new, V)
    new = tree_at(lambda m: m.s, new, s)
    if hasattr(new, "bias") and (b is not None):
        new = tree_at(lambda m: m.bias, new, b.astype(new.bias.dtype))

    report["spectral_warmstarted"] += 1
    report["n_loaded"] += _numel(W) + (_numel(b) if (b is not None) else 0)
    return new


def _fft_warmstart_kernel(W: jnp.ndarray, H_pad: int, W_pad: int) -> jnp.ndarray:
    """Take spatial conv W (Cout,Cin,Hk,Wk) -> rfft2 half-plane at (H_pad,W_pad)."""
    Cout, Cin, Hk, Wk = map(int, W.shape)
    k = jnp.zeros((Cout, Cin, H_pad, W_pad), W.dtype)
    k = k.at[:, :, :Hk, :Wk].set(W)
    k = jnp.roll(k, shift=(-Hk // 2, -Wk // 2), axis=(-2, -1))
    return jnp.fft.rfft2(k, s=(H_pad, W_pad), axes=(-2, -1), norm="ortho")


def _numel(a) -> int:
    try:
        return int(a.size)
    except Exception:
        return int(np.prod(a.shape))


# ---- recursive copier (Conv/Linear/BN + spectral warm-start) ----
def _copy_into(
    obj: Any,
    pt: Dict[str, jnp.ndarray],
    *,
    prefix: str,
    strict_fc: bool,
    spectral_warmstart: Literal["skip", "svd", "fft"],
    rfft_spatial: Optional[Dict[str, Tuple[int, int]]],
    report: Dict[str, Any],
) -> Any:
    w_key, b_key = f"{prefix}.weight", f"{prefix}.bias"

    # -------------------- eqx leaves: Conv2d --------------------
    if isinstance(obj, eqx.nn.Conv2d):
        if w_key in pt and tuple(pt[w_key].shape) == tuple(obj.weight.shape):
            new = tree_at(lambda m: m.weight, obj, pt[w_key])
            report["used"].add(w_key)
            report["n_loaded"] += _numel(pt[w_key])
            if hasattr(obj, "bias") and obj.bias is not None and (b_key in pt):
                new = tree_at(lambda m: m.bias, new, pt[b_key])
                report["used"].add(b_key)
                report["n_loaded"] += _numel(pt[b_key])
            return new
        # mismatch → leave unchanged
        return obj

    # -------------------- eqx leaves: Linear --------------------
    if isinstance(obj, eqx.nn.Linear):
        if w_key in pt:
            W = pt[w_key]
            if tuple(W.shape) == tuple(obj.weight.shape):
                new = tree_at(lambda m: m.weight, obj, W)
                report["used"].add(w_key)
                report["n_loaded"] += _numel(W)
                if (b_key in pt) and obj.bias is not None:
                    new = tree_at(lambda m: m.bias, new, pt[b_key])
                    report["used"].add(b_key)
                    report["n_loaded"] += _numel(pt[b_key])
                return new
            if ("fc.weight" in w_key) and (not strict_fc):
                report["skipped_fc"].append(
                    (w_key, tuple(W.shape), tuple(obj.weight.shape))
                )
                return obj
            report["mismatch"].append((w_key, tuple(W.shape), tuple(obj.weight.shape)))
        return obj

    # -------------------- eqx leaves: BatchNorm (affine only) --------------------
    if isinstance(obj, eqx.nn.BatchNorm):
        new = obj
        if w_key in pt:
            new = tree_at(lambda m: m.weight, new, pt[w_key])
            report["used"].add(w_key)
            report["n_loaded"] += _numel(pt[w_key])
        if b_key in pt:
            new = tree_at(lambda m: m.bias, new, pt[b_key])
            report["used"].add(b_key)
            report["n_loaded"] += _numel(pt[b_key])
        return new

    # -------------------- spectral leaves (top-level, non-nested) --------------------
    if _HAS_SPECTRAL:
        # SVDDense (linear spectral) with SVD warm-start
        if isinstance(obj, SVDDense) and spectral_warmstart == "svd" and (w_key in pt):
            W = pt[w_key]
            have = tuple(int(x) for x in W.shape)
            # infer wanted (out,in) from U,V shapes
            try:
                want = (
                    int(getattr(obj, "U").shape[0]),
                    int(getattr(obj, "V").shape[0]),
                )
            except Exception:
                want = have  # fallback; try anyway
            if have != want:
                report["spectral_skipped"].append(("svddense", have, want))
                report["used"].add(w_key)
                if b_key in pt:
                    report["used"].add(b_key)
                return obj
            try:
                new = _warmstart_svddense(obj, W, pt.get(b_key))
                report["used"].add(w_key)
                if b_key in pt:
                    report["used"].add(b_key)
                    report["n_loaded"] += _numel(pt[b_key])
                report["spectral_warmstarted"] += 1
                report["n_loaded"] += _numel(W)
                return new
            except Exception as e:
                report["spectral_errors"].append((prefix, repr(e)))
                return obj

        # SpectralConv2d (conv spectral) with SVD warm-start
        if (
            isinstance(obj, SpectralConv2d)
            and spectral_warmstart == "svd"
            and (w_key in pt)
        ):
            try:
                new = _warmstart_svdconv(obj, pt[w_key], pt.get(b_key), report)
                # mark keys used regardless of warm-start success
                report["used"].add(w_key)
                if b_key in pt:
                    report["used"].add(b_key)
                return new
            except Exception as e:
                report["spectral_errors"].append((prefix, repr(e)))
                return obj

        # RFFTCirculant2D (circulant spectral) with FFT warm-start
        if (
            isinstance(obj, RFFTCirculant2D)
            and spectral_warmstart == "fft"
            and (w_key in pt)
        ):
            if rfft_spatial is None or prefix not in rfft_spatial:
                report["spectral_errors"].append((prefix, "missing rfft_spatial entry"))
                return obj
            H, W = rfft_spatial[prefix]
            try:
                K = _fft_warmstart_kernel(pt[w_key], H, W)
                new = tree_at(lambda m: m.K_half, obj, K)
                report["used"].add(w_key)
                report["n_loaded"] += _numel(pt[w_key])
                if b_key in pt and hasattr(obj, "bias"):
                    new = tree_at(lambda m: m.bias, new, pt[b_key])
                    report["used"].add(b_key)
                    report["n_loaded"] += _numel(pt[b_key])
                report["spectral_warmstarted"] += 1
                return new
            except Exception as e:
                report["spectral_errors"].append((prefix, repr(e)))
                return obj

    # -------------------- recurse containers / nested modules --------------------
    if isinstance(obj, eqx.Module):
        new_obj = obj
        for name, child in vars(obj).items():
            child_prefix = name if prefix == "" else f"{prefix}.{name}"
            new_child = _copy_into(
                child,
                pt,
                prefix=child_prefix,
                strict_fc=strict_fc,
                spectral_warmstart=spectral_warmstart,
                rfft_spatial=rfft_spatial,
                report=report,
            )
            if new_child is not child:
                # Prefer tree_at; fall back to setattr for static/non-pytree fields.
                try:
                    new_obj = tree_at(lambda m: getattr(m, name), new_obj, new_child)
                except TypeError:
                    object.__setattr__(new_obj, name, new_child)
        return new_obj

    if isinstance(obj, tuple):
        return tuple(
            _copy_into(
                x,
                pt,
                prefix=(f"{prefix}.{i}" if prefix else str(i)),
                strict_fc=strict_fc,
                spectral_warmstart=spectral_warmstart,
                rfft_spatial=rfft_spatial,
                report=report,
            )
            for i, x in enumerate(obj)
        )
    if isinstance(obj, list):
        return [
            _copy_into(
                x,
                pt,
                prefix=(f"{prefix}.{i}" if prefix else str(i)),
                strict_fc=strict_fc,
                spectral_warmstart=spectral_warmstart,
                rfft_spatial=rfft_spatial,
                report=report,
            )
            for i, x in enumerate(obj)
        ]
    if isinstance(obj, dict):
        return {
            k: _copy_into(
                v,
                pt,
                prefix=f"{prefix}.{k}" if prefix else k,
                strict_fc=strict_fc,
                spectral_warmstart=spectral_warmstart,
                rfft_spatial=rfft_spatial,
                report=report,
            )
            for k, v in obj.items()
        }
    return obj


# ---- public API ----
def load_imagenet_resnet(
    eqx_tree: eqx.Module,
    npz_path: str,
    *,
    strict_fc: bool = False,
    spectral_warmstart: Literal["skip", "svd", "fft"] = "skip",
    rfft_spatial: Optional[Dict[str, Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any]]:
    """
    Unified loader for both classifiers and UNet encoders (or whole UNets).
      - Works with Conv2d/Linear/BatchNorm directly.
      - If your tree contains SpectralConv2d/SVDDense, set spectral_warmstart="svd".
      - If it contains RFFTCirculant2D, set spectral_warmstart="fft" and provide rfft_spatial[prefix]=(H,W).
      - If classifier head dims differ, set strict_fc=False to skip fc.*.
    """
    pt = _build_keymap(npz_path)
    report = {
        "used": set(),
        "unused_keys": [],
        "mismatch": [],
        "skipped_fc": [],  # list of tuples: (key, pt_shape, model_shape)
        "spectral_warmstarted": 0,
        "spectral_skipped": [],  # <— NEW
        "spectral_errors": [],
        "n_loaded": 0,
        "n_total_pt": sum(_numel(v) for v in pt.values()),
    }

    new_tree = _copy_into(
        eqx_tree,
        pt,
        prefix="",
        strict_fc=strict_fc,
        spectral_warmstart=spectral_warmstart,
        rfft_spatial=rfft_spatial,
        report=report,
    )
    report["unused_keys"] = sorted(set(pt.keys()) - report["used"])
    report["coverage"] = float(report["n_loaded"]) / max(1, report["n_total_pt"])

    if verbose:
        print(
            f"[resnet loader] coverage ~{100*report['coverage']:.2f}% "
            f"| spectral warm-started: {report['spectral_warmstarted']}"
        )
        if report.get("spectral_skipped"):
            print(
                f"[resnet loader] spectral skipped (shape-mismatch): {len(report['spectral_skipped'])}"
            )

        # --- FC head messages (clear, shape-specific) ---
        # Case 1: strict_fc=False (we skipped loading FC)
        if report["skipped_fc"]:
            printed = False
            for k, have, want in report["skipped_fc"]:
                if (
                    isinstance(k, str)
                    and k.endswith("fc.weight")
                    and len(have) == 2
                    and len(want) == 2
                ):
                    have_out, have_in = have
                    want_out, want_in = want
                    print(
                        "[resnet loader] skipped FC classifier head due to shape mismatch: "
                        f"PT {have_out}×{have_in} vs model {want_out}×{want_in} "
                        f"(strict_fc={strict_fc})."
                    )
                    printed = True
            if not printed:
                # Fallback generic line (e.g., if only bias was recorded)
                print(
                    f"[resnet loader] skipped fc due to mismatch (strict_fc={strict_fc})."
                )

        # Case 2: strict_fc=True (FC landed in general mismatches)
        if strict_fc and report["mismatch"]:
            for k, have, want in report["mismatch"]:
                if (
                    isinstance(k, str)
                    and k.endswith("fc.weight")
                    and len(have) == 2
                    and len(want) == 2
                ):
                    have_out, have_in = have
                    want_out, want_in = want
                    print(
                        "[resnet loader] FC classifier head mismatch: "
                        f"PT {have_out}×{have_in} vs model {want_out}×{want_in} "
                        "(strict_fc=True). Consider strict_fc=False for transfer tasks."
                    )
                    break  # one detailed FC line is enough

        if report["mismatch"]:
            print(f"[resnet loader] mismatches: {len(report['mismatch'])}.")
        if report["unused_keys"]:
            print(
                f"[resnet loader] unused PT keys: {len(report['unused_keys'])} (mostly BN buffers)."
            )

    return new_tree, report


# Convenience wrappers
def load_imagenet_resnet18(m, npz="resnet18_imagenet.npz", **kw):
    return load_imagenet_resnet(m, npz, **kw)


def load_imagenet_resnet34(m, npz="resnet34_imagenet.npz", **kw):
    return load_imagenet_resnet(m, npz, **kw)


def load_imagenet_resnet50(m, npz="resnet50_imagenet.npz", **kw):
    return load_imagenet_resnet(m, npz, **kw)


def load_imagenet_resnet101(m, npz="resnet101_imagenet.npz", **kw):
    return load_imagenet_resnet(m, npz, **kw)


def load_imagenet_resnet152(m, npz="resnet152_imagenet.npz", **kw):
    return load_imagenet_resnet(m, npz, **kw)
