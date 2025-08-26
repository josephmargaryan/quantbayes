# quantbayes/stochax/vision_common/pretrained_vgg.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Literal, Optional

import numpy as np
import jax.numpy as jnp
import equinox as eqx
from equinox import tree_at

# ---- Optional spectral imports (same as ResNet loader) ----
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


# ---- torchvision -> equinox key mapping (VGG) ----
def _rename_pt_key_vgg(k: str) -> str:
    # Map torchvision VGG classifier to your Equinox fields.
    # classifier.0 -> fc1, classifier.3 -> fc2, classifier.6 -> fc3
    if k.startswith("classifier.0."):
        return k.replace("classifier.0.", "fc1.")
    if k.startswith("classifier.3."):
        return k.replace("classifier.3.", "fc2.")
    if k.startswith("classifier.6."):
        return k.replace("classifier.6.", "fc3.")
    # features.* stays identical (your `features` is a flat tuple matching torchvision)
    return k


def _keep_weight_bias(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        k: v for k, v in raw.items() if k.endswith(".weight") or k.endswith(".bias")
    }


def _build_keymap(npz_path: str) -> Dict[str, jnp.ndarray]:
    raw = dict(np.load(npz_path))
    raw = _keep_weight_bias(raw)
    return {_rename_pt_key_vgg(k): jnp.asarray(v) for k, v in raw.items()}


# ---- spectral warm-start helpers (identical to ResNet loader) ----
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

    # Freeze orthonormal bases
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

    # Conv2d
    if isinstance(obj, eqx.nn.Conv2d):
        if w_key in pt and tuple(pt[w_key].shape) == tuple(obj.weight.shape):
            new = tree_at(lambda m: m.weight, obj, pt[w_key])
            report["used"].add(w_key)
            report["n_loaded"] += _numel(pt[w_key])
            if getattr(obj, "bias", None) is not None and (b_key in pt):
                new = tree_at(lambda m: m.bias, new, pt[b_key])
                report["used"].add(b_key)
                report["n_loaded"] += _numel(pt[b_key])
            return new
        return obj

    # Linear
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
            if ("classifier." in w_key) and (not strict_fc):
                report["skipped_fc"].append(
                    (w_key, tuple(W.shape), tuple(obj.weight.shape))
                )
                return obj
            report["mismatch"].append((w_key, tuple(W.shape), tuple(obj.weight.shape)))
        return obj

    # BatchNorm (affine only)
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

    # Spectral leaves
    if _HAS_SPECTRAL:
        if isinstance(obj, SVDDense) and spectral_warmstart == "svd" and (w_key in pt):
            W = pt[w_key]
            try:
                want = (
                    int(getattr(obj, "U").shape[0]),
                    int(getattr(obj, "V").shape[0]),
                )
            except Exception:
                want = tuple(int(x) for x in W.shape)
            have = tuple(int(x) for x in W.shape)
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

        if (
            isinstance(obj, SpectralConv2d)
            and spectral_warmstart == "svd"
            and (w_key in pt)
        ):
            try:
                new = _warmstart_svdconv(obj, pt[w_key], pt.get(b_key), report)
                report["used"].add(w_key)
                if b_key in pt:
                    report["used"].add(b_key)
                return new
            except Exception as e:
                report["spectral_errors"].append((prefix, repr(e)))
                return obj

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

    # Recurse containers
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
def load_imagenet_vgg(
    eqx_tree: eqx.Module,
    npz_path: str,
    *,
    strict_fc: bool = False,
    spectral_warmstart: Literal["skip", "svd", "fft"] = "skip",
    rfft_spatial: Optional[Dict[str, Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any]]:
    """
    Loader for VGG-style trees (features + classifier), with optional spectral warm-start.
      - Works with Conv2d/Linear/BatchNorm directly.
      - If the tree contains SpectralConv2d/SVDDense, set spectral_warmstart="svd".
      - If it contains RFFTCirculant2D, set spectral_warmstart="fft" and provide rfft_spatial[prefix]=(H,W).
      - If classifier head dims differ, set strict_fc=False to skip classifier.*.
    """
    # Build raw PT keymap (weight/bias only) and apply renaming
    pt_all = _build_keymap(npz_path)

    # --- Sanity warn: BN/non-BN mismatch between checkpoint and model ---
    # Heuristic: BN checkpoints have feature BN weights (e.g., features.1.weight).
    is_bn_ckpt = any(
        k.startswith("features.")
        and k.endswith(".weight")
        and ".running_" not in k
        and k.split(".")[1].isdigit()
        and int(k.split(".")[1]) % 3 == 1
        for k in pt_all.keys()
    ) or any(".bn" in k for k in pt_all.keys())
    # Model-side: just check whether any feature is an Equinox BatchNorm
    try:
        feats = getattr(eqx_tree, "features", ())
        is_bn_model = any(
            isinstance(m, eqx.nn.BatchNorm)
            for m in (feats if isinstance(feats, (list, tuple)) else ())
        )
    except Exception:
        is_bn_model = False

    if verbose and (is_bn_ckpt != is_bn_model):
        print(
            "[vgg loader] WARNING: BatchNorm mismatch between checkpoint and model "
            f"(ckpt BN={is_bn_ckpt}, model BN={is_bn_model}). Coverage will be poor."
        )

    # --- If strict_fc=False, exclude the head from loading AND from coverage ---
    if not strict_fc:
        dropped = [
            k
            for k in pt_all.keys()
            if k.startswith("classifier.")
            or k.startswith("fc1.")
            or k.startswith("fc2.")
            or k.startswith("fc3.")
        ]
        pt = {k: v for k, v in pt_all.items() if k not in set(dropped)}
    else:
        pt = pt_all

    report = {
        "used": set(),
        "unused_keys": [],
        "mismatch": [],
        "skipped_fc": [],
        "spectral_warmstarted": 0,
        "spectral_skipped": [],
        "spectral_errors": [],
        "n_loaded": 0,
        "n_total_pt": sum(
            _numel(v) for v in pt.values()
        ),  # denominator AFTER head filter
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
            f"[vgg loader] coverage ~{100*report['coverage']:.2f}% "
            f"| spectral warm-started: {report['spectral_warmstarted']}"
        )
        if report.get("spectral_skipped"):
            print(
                f"[vgg loader] spectral skipped (shape-mismatch): {len(report['spectral_skipped'])}"
            )

        if report["skipped_fc"]:
            printed = False
            for k, have, want in report["skipped_fc"]:
                if (
                    isinstance(k, str)
                    and k.endswith("weight")
                    and len(have) == 2
                    and len(want) == 2
                ):
                    print(
                        "[vgg loader] skipped classifier head due to shape mismatch: "
                        f"PT {have[0]}×{have[1]} vs model {want[0]}×{want[1]} "
                        f"(strict_fc={strict_fc})."
                    )
                    printed = True
            if not printed:
                print(
                    f"[vgg loader] skipped classifier due to mismatch (strict_fc={strict_fc})."
                )

        if report["mismatch"]:
            print(f"[vgg loader] mismatches: {len(report['mismatch'])}.")
        if report["unused_keys"]:
            print(
                f"[vgg loader] unused PT keys: {len(report['unused_keys'])} (mostly BN buffers)."
            )

    return new_tree, report


# Convenience wrappers
def load_imagenet_vgg11(m, npz="vgg11_imagenet.npz", **kw):
    return load_imagenet_vgg(m, npz, **kw)


def load_imagenet_vgg13(m, npz="vgg13_imagenet.npz", **kw):
    return load_imagenet_vgg(m, npz, **kw)


def load_imagenet_vgg16(m, npz="vgg16_imagenet.npz", **kw):
    return load_imagenet_vgg(m, npz, **kw)


def load_imagenet_vgg19(m, npz="vgg19_imagenet.npz", **kw):
    return load_imagenet_vgg(m, npz, **kw)


def load_imagenet_vgg11_bn(m, npz="vgg11_bn_imagenet.npz", **kw):
    return load_imagenet_vgg(m, npz, **kw)


def load_imagenet_vgg13_bn(m, npz="vgg13_bn_imagenet.npz", **kw):
    return load_imagenet_vgg(m, npz, **kw)


def load_imagenet_vgg16_bn(m, npz="vgg16_bn_imagenet.npz", **kw):
    return load_imagenet_vgg(m, npz, **kw)


def load_imagenet_vgg19_bn(m, npz="vgg19_bn_imagenet.npz", **kw):
    return load_imagenet_vgg(m, npz, **kw)
