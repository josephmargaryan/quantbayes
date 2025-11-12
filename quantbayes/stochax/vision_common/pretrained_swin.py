from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, Literal
import re
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from equinox import tree_at

_HAS_SPECTRAL = False
SVDDense = object
try:
    from quantbayes.stochax.layers.spectral_layers import SVDDense  # type: ignore

    _HAS_SPECTRAL = True
except Exception:
    pass


def _numel(a) -> int:
    try:
        return int(a.size)
    except Exception:
        return int(np.prod(a.shape))


def _as_jnp(x):
    return jnp.asarray(x)


def _is_ln2d(obj) -> bool:
    if isinstance(obj, (eqx.nn.Linear, eqx.nn.Conv2d)):
        return False
    w = getattr(obj, "weight", None)
    b = getattr(obj, "bias", None)
    return (
        w is not None
        and b is not None
        and hasattr(obj, "eps")
        and isinstance(w, jnp.ndarray)
        and w.ndim == 1
    )


def _svd_warmstart_svddense(mod, W, b, report):
    import jax

    U, s, Vh = jnp.linalg.svd(W, full_matrices=False)
    r = min(len(s), getattr(mod, "s").shape[0])
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


def _copy_into(
    obj,
    pt: Dict[str, jnp.ndarray],
    *,
    prefix: str,
    spectral: bool,
    report: Dict[str, Any],
):
    w_key, b_key = f"{prefix}.weight", f"{prefix}.bias"

    if isinstance(obj, jnp.ndarray):
        if prefix in pt and tuple(pt[prefix].shape) == tuple(obj.shape):
            report["used"].add(prefix)
            report["n_loaded"] += _numel(pt[prefix])
            return pt[prefix]
        return obj

    if _HAS_SPECTRAL and isinstance(obj, SVDDense) and spectral:
        if w_key in pt:
            try:
                new = _svd_warmstart_svddense(obj, pt[w_key], pt.get(b_key), report)
                report["used"].add(w_key)
                if b_key in pt:
                    report["used"].add(b_key)
                return new
            except Exception as e:
                report["spectral_errors"].append((prefix, repr(e)))
                report["used"].add(w_key)
                if b_key in pt:
                    report["used"].add(b_key)
                return obj

    if isinstance(obj, eqx.nn.Linear):
        if w_key in pt and tuple(pt[w_key].shape) == tuple(obj.weight.shape):
            new = tree_at(lambda m: m.weight, obj, pt[w_key])
            report["used"].add(w_key)
            report["n_loaded"] += _numel(pt[w_key])
            if getattr(obj, "bias", None) is not None and b_key in pt:
                new = tree_at(lambda m: m.bias, new, pt[b_key])
                report["used"].add(b_key)
                report["n_loaded"] += _numel(pt[b_key])
            return new
        return obj

    if isinstance(obj, eqx.nn.Conv2d):
        if w_key in pt and tuple(pt[w_key].shape) == tuple(obj.weight.shape):
            new = tree_at(lambda m: m.weight, obj, pt[w_key])
            report["used"].add(w_key)
            report["n_loaded"] += _numel(pt[w_key])
            if getattr(obj, "bias", None) is not None and b_key in pt:
                new = tree_at(lambda m: m.bias, new, pt[b_key])
                report["used"].add(b_key)
                report["n_loaded"] += _numel(pt[b_key])
            return new
        return obj

    if _is_ln2d(obj) or isinstance(obj, eqx.nn.LayerNorm):
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

    if isinstance(obj, eqx.Module):
        new_obj = obj
        for name, child in vars(obj).items():
            child_prefix = name if prefix == "" else f"{prefix}.{name}"
            new_child = _copy_into(
                child, pt, prefix=child_prefix, spectral=spectral, report=report
            )
            if new_child is not child:
                try:
                    new_obj = tree_at(lambda m: getattr(m, name), new_obj, new_child)
                except TypeError:
                    object.__setattr__(new_obj, name, new_child)
        return new_obj

    if isinstance(obj, (list, tuple)):
        seq = []
        for i, c in enumerate(obj):
            pfx = f"{prefix}.{i}" if prefix else str(i)
            seq.append(_copy_into(c, pt, prefix=pfx, spectral=spectral, report=report))
        return type(obj)(seq)

    if isinstance(obj, dict):
        return {
            k: _copy_into(
                v,
                pt,
                prefix=f"{prefix}.{k}" if prefix else k,
                spectral=spectral,
                report=report,
            )
            for k, v in obj.items()
        }

    return obj


# ---- map torch->eqx paths ----
def load_imagenet_swin(
    eqx_tree: eqx.Module,
    npz_path: str,
    *,
    strict_fc: bool = True,
    spectral_warmstart: Literal["skip", "svd"] = "skip",
    verbose: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any]]:
    raw_np = dict(np.load(npz_path, allow_pickle=False))
    raw = {k.replace("module.", ""): _as_jnp(v) for k, v in raw_np.items()}
    pt: Dict[str, jnp.ndarray] = {}

    # Patch embed
    for k in ("patch_embed.proj.weight", "features.0.proj.weight"):
        if k in raw:
            pt["features.0.proj.weight"] = raw[k]
            break
    for k in ("patch_embed.proj.bias", "features.0.proj.bias"):
        if k in raw:
            pt["features.0.proj.bias"] = raw[k]
            break
    for k in ("patch_embed.norm.weight", "features.0.norm.weight"):
        if k in raw:
            pt["features.0.norm.weight"] = raw[k]
            break
    for k in ("patch_embed.norm.bias", "features.0.norm.bias"):
        if k in raw:
            pt["features.0.norm.bias"] = raw[k]
            break

    # stages: layers.i.blocks.j.*
    # Equinox: features[1..4] are BasicLayer; each has blocks[j]; each block has norm1, attn(qkv/proj), norm2, mlp(fc1,fc2)
    for i in range(4):
        # count blocks present in eqx
        layer = (
            getattr(eqx_tree.features, str(i + 1))
            if isinstance(eqx_tree.features, dict)
            else eqx_tree.features[i + 1]
        )
        nblocks = len(getattr(layer, "blocks"))
        for j in range(nblocks):
            base_tv = f"layers.{i}.blocks.{j}"
            base_eq = f"features.{i+1}.blocks.{j}"
            # norm1
            if f"{base_tv}.norm1.weight" in raw:
                pt[f"{base_eq}.norm1.weight"] = raw[f"{base_tv}.norm1.weight"]
            if f"{base_tv}.norm1.bias" in raw:
                pt[f"{base_eq}.norm1.bias"] = raw[f"{base_tv}.norm1.bias"]
            # attn qkv/proj
            if f"{base_tv}.attn.qkv.weight" in raw:
                pt[f"{base_eq}.attn.qkv.weight"] = raw[f"{base_tv}.attn.qkv.weight"]
            if f"{base_tv}.attn.qkv.bias" in raw:
                pt[f"{base_eq}.attn.qkv.bias"] = raw[f"{base_tv}.attn.qkv.bias"]
            if f"{base_tv}.attn.proj.weight" in raw:
                pt[f"{base_eq}.attn.proj.weight"] = raw[f"{base_tv}.attn.proj.weight"]
            if f"{base_tv}.attn.proj.bias" in raw:
                pt[f"{base_eq}.attn.proj.bias"] = raw[f"{base_tv}.attn.proj.bias"]
            # rel bias table
            if f"{base_tv}.attn.relative_position_bias_table" in raw:
                pt[f"{base_eq}.attn.relative_position_bias_table"] = raw[
                    f"{base_tv}.attn.relative_position_bias_table"
                ]
            # norm2
            if f"{base_tv}.norm2.weight" in raw:
                pt[f"{base_eq}.norm2.weight"] = raw[f"{base_tv}.norm2.weight"]
            if f"{base_tv}.norm2.bias" in raw:
                pt[f"{base_eq}.norm2.bias"] = raw[f"{base_tv}.norm2.bias"]
            # mlp
            if f"{base_tv}.mlp.fc1.weight" in raw:
                pt[f"{base_eq}.mlp.fc1.weight"] = raw[f"{base_tv}.mlp.fc1.weight"]
            if f"{base_tv}.mlp.fc1.bias" in raw:
                pt[f"{base_eq}.mlp.fc1.bias"] = raw[f"{base_tv}.mlp.fc1.bias"]
            if f"{base_tv}.mlp.fc2.weight" in raw:
                pt[f"{base_eq}.mlp.fc2.weight"] = raw[f"{base_tv}.mlp.fc2.weight"]
            if f"{base_tv}.mlp.fc2.bias" in raw:
                pt[f"{base_eq}.mlp.fc2.bias"] = raw[f"{base_tv}.mlp.fc2.bias"]

        # downsample
        ds_tv = f"layers.{i}.downsample"
        ds_eq = f"features.{i+1}.downsample"
        if f"{ds_tv}.reduction.weight" in raw:
            pt[f"{ds_eq}.reduction.weight"] = raw[f"{ds_tv}.reduction.weight"]
        if f"{ds_tv}.reduction.bias" in raw:
            pt[f"{ds_eq}.reduction.bias"] = raw[f"{ds_tv}.reduction.bias"]
        if f"{ds_tv}.norm.weight" in raw:
            pt[f"{ds_eq}.norm.weight"] = raw[f"{ds_tv}.norm.weight"]
        if f"{ds_tv}.norm.bias" in raw:
            pt[f"{ds_eq}.norm.bias"] = raw[f"{ds_tv}.norm.bias"]

    # final norm + head
    if "norm.weight" in raw:
        pt["norm.weight"] = raw["norm.weight"]
    if "norm.bias" in raw:
        pt["norm.bias"] = raw["norm.bias"]
    if "head.weight" in raw:
        pt["head.weight"] = raw["head.weight"]
    if "head.bias" in raw:
        pt["head.bias"] = raw["head.bias"]

    report = {
        "used": set(),
        "unused_keys": [],
        "mismatch": [],
        "spectral_warmstarted": 0,
        "spectral_errors": [],
        "n_loaded": 0,
        "n_total_pt": sum(_numel(v) for v in pt.values()),
    }
    new_tree = _copy_into(
        eqx_tree, pt, prefix="", spectral=(spectral_warmstart == "svd"), report=report
    )
    report["unused_keys"] = sorted(set(pt.keys()) - report["used"])
    report["coverage"] = float(report["n_loaded"]) / max(1, report["n_total_pt"])

    if strict_fc and ("head.weight" in pt):
        want = tuple(new_tree.head.weight.shape)
        have = tuple(pt["head.weight"].shape)
        if want != have:
            raise ValueError(
                f"Head mismatch PT {have} vs model {want} (strict_fc=True)."
            )

    if verbose:
        print(
            f"[swin loader] coverage ~{100*report['coverage']:.2f}% | spectral-warmstarted: {report['spectral_warmstarted']}"
        )
        if report["unused_keys"]:
            print(f"[swin loader] mapped-but-unused: {len(report['unused_keys'])}")
        if report["mismatch"]:
            print(f"[swin loader] mismatches: {len(report['mismatch'])}")
    return new_tree, report
