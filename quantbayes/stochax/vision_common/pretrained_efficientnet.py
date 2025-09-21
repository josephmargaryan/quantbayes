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


def _is_bn(obj) -> bool:
    return isinstance(obj, eqx.nn.BatchNorm)


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

    if _is_bn(obj):
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


# ----- mapping helpers -----
def _collect_blocks(raw: Dict[str, jnp.ndarray]) -> list[str]:
    # candidate prefixes ending with ".block"
    bases = set()
    for k in raw.keys():
        m = re.match(r"(features\.\d+\.\d+)\.block\.", k)
        if m:
            bases.add(m.group(1) + ".block")
    return sorted(bases, key=lambda s: [int(x) for x in re.findall(r"\d+", s)])


def _stem_keys(raw):
    # TV stem: features.0.0 (conv), features.0.1 (bn)
    sc = raw.get("features.0.0.weight")
    sbw = raw.get("features.0.1.weight")
    sbb = raw.get("features.0.1.bias")
    return sc, sbw, sbb


def load_imagenet_efficientnet(
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

    # Stem (ConvBNActTV)
    sc, sbw, sbb = _stem_keys(raw)
    if sc is not None:
        pt["features.0.c0.weight"] = sc
    if sbw is not None:
        pt["features.0.c1.weight"] = sbw
    if sbb is not None:
        pt["features.0.c1.bias"] = sbb

    # Gather Equinox MBConv paths in order
    eqx_blocks: list[str] = []
    for idx, feat in enumerate(eqx_tree.features):
        if isinstance(feat, tuple):
            for j, _ in enumerate(feat):
                eqx_blocks.append(f"features.{idx}.{j}")

    bases = _collect_blocks(raw)
    n = min(len(eqx_blocks), len(bases))
    eqx_blocks, bases = eqx_blocks[:n], bases[:n]

    for eqx_p, ptb in zip(eqx_blocks, bases):
        # Expand (optional)
        # expand conv: block.0.0 ; bn: block.0.1
        if f"{ptb}.0.0.weight" in raw:
            pt[f"{eqx_p}.block.0.c0.weight"] = raw[f"{ptb}.0.0.weight"]
            if f"{ptb}.0.1.weight" in raw:
                pt[f"{eqx_p}.block.0.c1.weight"] = raw[f"{ptb}.0.1.weight"]
            if f"{ptb}.0.1.bias" in raw:
                pt[f"{eqx_p}.block.0.c1.bias"] = raw[f"{ptb}.0.1.bias"]
            dw_prefix = f"{ptb}.1"  # DWConvBN
            se_prefix = f"{ptb}.2"
            proj_prefix = f"{ptb}.3"
            se_in_suffix = ".0"
            se_out_suffix = ".1"
        else:
            # no expand → DW is block.0; SE is block.1; proj is block.2
            dw_prefix = f"{ptb}.0"
            se_prefix = f"{ptb}.1"
            proj_prefix = f"{ptb}.2"
            se_in_suffix = ".0"
            se_out_suffix = ".1"

        # Depthwise
        if f"{dw_prefix}.0.weight" in raw:
            pt[f"{eqx_p}.block.{1 if f'{ptb}.0.0.weight' in raw else 0}.c0.weight"] = (
                raw[f"{dw_prefix}.0.weight"]
            )
        if f"{dw_prefix}.1.weight" in raw:
            pt[f"{eqx_p}.block.{1 if f'{ptb}.0.0.weight' in raw else 0}.c1.weight"] = (
                raw[f"{dw_prefix}.1.weight"]
            )
        if f"{dw_prefix}.1.bias" in raw:
            pt[f"{eqx_p}.block.{1 if f'{ptb}.0.0.weight' in raw else 0}.c1.bias"] = raw[
                f"{dw_prefix}.1.bias"
            ]

        # SE reduce/expand (conv1x1)
        if f"{se_prefix}{se_in_suffix}.weight" in raw:
            pt[f"{eqx_p}.block.{2 if f'{ptb}.0.0.weight' in raw else 1}.c0.weight"] = (
                raw[f"{se_prefix}{se_in_suffix}.weight"]
            )
            pt[
                f"{eqx_p}.block.{2 if f'{ptb}.0.0.weight' in raw else 1}.c0.bias"
            ] = raw.get(f"{se_prefix}{se_in_suffix}.bias", None) or jnp.zeros_like(
                pt[f"{eqx_p}.block.{2 if f'{ptb}.0.0.weight' in raw else 1}.c0.weight"][
                    ..., 0, 0
                ]
            )
        if f"{se_prefix}{se_out_suffix}.weight" in raw:
            pt[f"{eqx_p}.block.{2 if f'{ptb}.0.0.weight' in raw else 1}.c1.weight"] = (
                raw[f"{se_prefix}{se_out_suffix}.weight"]
            )
            pt[
                f"{eqx_p}.block.{2 if f'{ptb}.0.0.weight' in raw else 1}.c1.bias"
            ] = raw.get(f"{se_prefix}{se_out_suffix}.bias", None) or jnp.zeros_like(
                pt[f"{eqx_p}.block.{2 if f'{ptb}.0.0.weight' in raw else 1}.c1.weight"][
                    ..., 0, 0
                ]
            )

        # Project
        if f"{proj_prefix}.0.weight" in raw:
            # figure index of project block inside our tuple:
            proj_idx = 3 if f"{ptb}.0.0.weight" in raw else 2
            pt[f"{eqx_p}.block.{proj_idx}.c0.weight"] = raw[f"{proj_prefix}.0.weight"]
        if f"{proj_prefix}.1.weight" in raw:
            proj_idx = 3 if f"{ptb}.0.0.weight" in raw else 2
            pt[f"{eqx_p}.block.{proj_idx}.c1.weight"] = raw[f"{proj_prefix}.1.weight"]
        if f"{proj_prefix}.1.bias" in raw:
            proj_idx = 3 if f"{ptb}.0.0.weight" in raw else 2
            pt[f"{eqx_p}.block.{proj_idx}.c1.bias"] = raw[f"{proj_prefix}.1.bias"]

    # Head (ConvBNActTV + classifier)
    # last feature is head conv (k=1)
    if "features.7.0.weight" in raw:
        # common TV idx for B0
        pt["features.-1.c0.weight"] = raw["features.7.0.weight"]
        if "features.7.1.weight" in raw:
            pt["features.-1.c1.weight"] = raw["features.7.1.weight"]
        if "features.7.1.bias" in raw:
            pt["features.-1.c1.bias"] = raw["features.7.1.bias"]
    # classifier
    if "classifier.1.weight" in raw:
        pt["fc.weight"] = raw["classifier.1.weight"]
        if "classifier.1.bias" in raw:
            pt["fc.bias"] = raw["classifier.1.bias"]

    # copy
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

    if strict_fc and ("fc.weight" in pt):
        want = tuple(new_tree.fc.weight.shape)
        have = tuple(pt["fc.weight"].shape)
        if want != have:
            raise ValueError(
                f"Head mismatch PT {have} vs model {want} (strict_fc=True)."
            )

    if verbose:
        print(
            f"[efficientnet loader] coverage ~{100*report['coverage']:.2f}% | spectral-warmstarted: {report['spectral_warmstarted']}"
        )
        if report["unused_keys"]:
            print(
                f"[efficientnet loader] mapped-but-unused: {len(report['unused_keys'])}"
            )
        if report["mismatch"]:
            print(f"[efficientnet loader] mismatches: {len(report['mismatch'])}")
    return new_tree, report
