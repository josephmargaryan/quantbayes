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

    if isinstance(obj, eqx.nn.Conv2d):
        if w_key in pt and tuple(pt[w_key].shape) == tuple(obj.weight.shape):
            new = tree_at(lambda m: m.weight, obj, pt[w_key])
            report["used"].add(w_key)
            report["n_loaded"] += _numel(pt[w_key])
            # bias may exist (stem / downsample convs use_bias=True)
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

    if _is_ln2d(obj):
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


# ---- helpers to extract PT prefixes ----
def _find_stem(raw: Dict[str, jnp.ndarray]):
    conv = None
    lnw = lnb = None
    for k, v in raw.items():
        if (
            k.endswith(".weight")
            and v.ndim == 4
            and v.shape[1] == 3
            and v.shape[-2:] == (4, 4)
        ):
            # candidate conv stem
            if (
                ("features.0.0.weight" in k)
                or ("stem.0.weight" in k)
                or ("downsample_layers.0.0.weight" in k)
            ):
                conv = v
                break
    if conv is None:
        # fallback: first 4x4 conv with in=3
        for k, v in raw.items():
            if (
                k.endswith(".weight")
                and v.ndim == 4
                and v.shape[1] == 3
                and v.shape[-2:] == (4, 4)
            ):
                conv = v
                break
    # LN likely next to it
    for cand in ("features.0.1", "stem.1", "downsample_layers.0.1", "features.stem.1"):
        if f"{cand}.weight" in raw:
            lnw = raw[f"{cand}.weight"]
            lnb = raw.get(f"{cand}.bias", None)
            break
    return conv, lnw, lnb


def _collect_block_prefixes(raw: Dict[str, jnp.ndarray]) -> list[str]:
    bases = set()
    for k in raw.keys():
        m = re.match(r"(features\.\d+\.\d+)\.block\.0\.weight$", k)
        if m:
            bases.add(m.group(1) + ".block")
        m = re.match(r"(stages\.\d+\.\d+)\.block\.0\.weight$", k)
        if m:
            bases.add(m.group(1) + ".block")
    return sorted(bases, key=lambda s: [int(x) for x in re.findall(r"\d+", s)])


def _collect_downsample_prefixes(raw: Dict[str, jnp.ndarray]) -> list[str]:
    cand = []
    for k in raw.keys():
        if re.match(r"features\.\d+\.downsample\.\d+\.weight$", k):
            cand.append(k.rsplit(".", 2)[0])  # 'features.{i}.downsample'
        if re.match(r"downsample_layers\.\d+\.\d+\.weight$", k):
            cand.append(k.rsplit(".", 2)[0])  # 'downsample_layers.{i}'
    return sorted(set(cand), key=lambda s: [int(x) for x in re.findall(r"\d+", s)])


# ---- public API ----
def load_imagenet_convnext(
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

    # Stem
    conv, lnw, lnb = _find_stem(raw)
    if conv is not None:
        pt["features.0.conv.weight"] = conv
    if lnw is not None:
        pt["features.0.norm.weight"] = lnw
    if lnb is not None:
        pt["features.0.norm.bias"] = lnb

    # Blocks
    bases = _collect_block_prefixes(raw)
    # map onto your features: [Stem(0), Stage0(1), Down(2), Stage1(3), Down(4), Stage2(5), Down(6), Stage3(7)]
    eqx_block_paths: list[str] = []
    for idx, node in enumerate(eqx_tree.features):
        if isinstance(node, tuple):  # a stage
            for j, _ in enumerate(node):
                eqx_block_paths.append(f"features.{idx}.{j}")
    if len(eqx_block_paths) != len(bases):
        # still try by trunc/min
        n = min(len(eqx_block_paths), len(bases))
        eqx_block_paths, bases = eqx_block_paths[:n], bases[:n]

    for eqx_base, pt_base in zip(eqx_block_paths, bases):
        # dwconv
        if f"{pt_base}.0.weight" in raw:
            pt[f"{eqx_base}.dwconv.weight"] = raw[f"{pt_base}.0.weight"]
        # norm
        if f"{pt_base}.1.weight" in raw:
            pt[f"{eqx_base}.norm.weight"] = raw[f"{pt_base}.1.weight"]
        if f"{pt_base}.1.bias" in raw:
            pt[f"{eqx_base}.norm.bias"] = raw[f"{pt_base}.1.bias"]
        # linear1 (block.2)
        if f"{pt_base}.2.weight" in raw:
            pt[f"{eqx_base}.linear1.weight"] = raw[f"{pt_base}.2.weight"]
        if f"{pt_base}.2.bias" in raw:
            pt[f"{eqx_base}.linear1.bias"] = raw[f"{pt_base}.2.bias"]
        # linear2 (block.4)
        if f"{pt_base}.4.weight" in raw:
            pt[f"{eqx_base}.linear2.weight"] = raw[f"{pt_base}.4.weight"]
        if f"{pt_base}.4.bias" in raw:
            pt[f"{eqx_base}.linear2.bias"] = raw[f"{pt_base}.4.bias"]
        # layer scale gamma (block.5.gamma or block.5.weight) -> (C,1,1)
        g = raw.get(f"{pt_base}.5.gamma", raw.get(f"{pt_base}.5.weight", None))
        if g is not None:
            pt[f"{eqx_base}.gamma"] = g.reshape(-1, 1, 1)

    # Downsamples
    ds_pt = _collect_downsample_prefixes(raw)
    ds_eqx: list[tuple[str, int]] = [
        (f"features.{i}", i)
        for i, n in enumerate(eqx_tree.features)
        if n.__class__.__name__ == "Downsample"
    ]
    n = min(len(ds_pt), len(ds_eqx))
    for (eqx_p, _), pt_p in zip(ds_eqx[:n], ds_pt[:n]):
        # some repos store norm at .1 and conv at .0; others inverse
        for a, b in ((1, 0), (0, 1)):
            wN, bN = raw.get(f"{pt_p}.{a}.weight"), raw.get(f"{pt_p}.{a}.bias")
            wC, bC = raw.get(f"{pt_p}.{b}.weight"), raw.get(f"{pt_p}.{b}.bias")
            if wN is not None and wN.ndim == 1 and wC is not None and wC.ndim == 4:
                pt[f"{eqx_p}.norm.weight"] = wN
                if bN is not None:
                    pt[f"{eqx_p}.norm.bias"] = bN
                pt[f"{eqx_p}.conv.weight"] = wC
                if bC is not None:
                    pt[f"{eqx_p}.conv.bias"] = bC
                break

    # classifier norm + fc (LayerNorm2d over [C,1,1])
    if "classifier.2.weight" in raw or "classifier.1.weight" in raw:
        # torchvision head: (AdaptivePool) -> LayerNorm2d -> Linear
        # search LN
        for k in ("classifier.1.weight", "classifier.2.weight"):
            if k in raw and raw[k].ndim == 1:
                pt["classifier_norm.weight"] = raw[k]
                b = k.replace(".weight", ".bias")
                if b in raw:
                    pt["classifier_norm.bias"] = raw[b]
                break
        # search Linear
        for k in ("classifier.4.weight", "classifier.3.weight", "classifier.1.weight"):
            if k in raw and raw[k].ndim == 2:
                pt["fc.weight"] = raw[k]
                b = k.replace(".weight", ".bias")
                if b in raw:
                    pt["fc.bias"] = raw[b]
                break
    else:
        # older TorchVision has head as norm+linear directly
        if "norm.weight" in raw:
            pt["classifier_norm.weight"] = raw["norm.weight"]
        if "norm.bias" in raw:
            pt["classifier_norm.bias"] = raw["norm.bias"]
        if "head.weight" in raw:
            pt["fc.weight"] = raw["head.weight"]
        if "head.bias" in raw:
            pt["fc.bias"] = raw["head.bias"]

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
            f"[convnext loader] coverage ~{100*report['coverage']:.2f}% | spectral-warmstarted: {report['spectral_warmstarted']}"
        )
        if report["unused_keys"]:
            print(f"[convnext loader] mapped-but-unused: {len(report['unused_keys'])}")
        if report["mismatch"]:
            print(f"[convnext loader] mismatches: {len(report['mismatch'])}")
    return new_tree, report
