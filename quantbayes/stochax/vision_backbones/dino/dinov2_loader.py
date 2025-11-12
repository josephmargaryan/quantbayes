from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, Literal
import re, math, numpy as np
import jax, jax.numpy as jnp
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


def _strip_known_prefixes(k: str) -> str:
    for p in ("module.", "model.", "backbone.", "trunk."):
        if k.startswith(p):
            return k[len(p) :]
    return k


def _resize_pos(tv: jnp.ndarray, target_len: int) -> jnp.ndarray:
    if tv.ndim == 2:
        tv = tv[None, ...]
    cls, seq = tv[:, :1, :], tv[:, 1:, :]
    L_tgt = int(target_len - 1)
    if seq.shape[1] == L_tgt:
        return tv
    old = int(seq.shape[1])
    oh = int(round(math.sqrt(old)))
    nh = int(round(math.sqrt(L_tgt)))
    if oh * oh == old and nh * nh == L_tgt:
        seq2 = jax.image.resize(
            seq.reshape(1, oh, oh, seq.shape[-1]),
            (1, nh, nh, seq.shape[-1]),
            method="linear",
        ).reshape(1, nh * nh, seq.shape[-1])
    else:
        seq2 = jax.image.resize(seq, (1, L_tgt, seq.shape[-1]), method="linear")
    return jnp.concatenate([cls, seq2], axis=1)


def _svd_warmstart_svddense(mod, W, b, report):
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
            if b_key in pt and (obj.bias is not None):
                new = tree_at(lambda m: m.bias, new, pt[b_key])
                report["used"].add(b_key)
                report["n_loaded"] += _numel(pt[b_key])
            return new
        return obj

    if isinstance(obj, eqx.nn.LayerNorm):
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


def load_dinov2(
    eqx_tree: eqx.Module,
    npz_path: str,
    *,
    strict_fc: bool = False,
    spectral_warmstart: Literal["skip", "svd"] = "skip",
    verbose: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any]]:
    raw = dict(np.load(npz_path, allow_pickle=False))
    # unify prefixes (module./model./backbone./trunk.)
    raw = {_strip_known_prefixes(k): _as_jnp(v) for k, v in raw.items()}

    pt: Dict[str, jnp.ndarray] = {}

    # patch embed conv -> linear
    Wpe = raw.get("patch_embed.proj.weight", None)
    if Wpe is None:
        Wpe = raw.get("embedding.conv.weight", None)
    if Wpe is not None:
        E, C, ph, pw = map(int, Wpe.shape)
        pt["patch_embedding.linear.weight"] = Wpe.reshape(E, C * ph * pw)
        Bpe = raw.get("patch_embed.proj.bias", raw.get("embedding.conv.bias", None))
        if Bpe is not None:
            pt["patch_embedding.linear.bias"] = Bpe

    # tokens/pos
    if "cls_token" in raw:
        pt["cls_token"] = raw["cls_token"].reshape(-1, raw["cls_token"].shape[-1])
    if "register_tokens" in raw:
        reg = raw["register_tokens"]
        pt["register_tokens"] = reg.reshape(-1, reg.shape[-1]) if reg.ndim == 3 else reg
    if "pos_embed" in raw:
        L_target = int(getattr(eqx_tree, "positional_embedding").shape[0])
        pt["positional_embedding"] = _resize_pos(raw["pos_embed"], L_target).squeeze(0)

    # final norm
    if "norm.weight" in raw:
        pt["norm.weight"] = raw["norm.weight"]
    if "norm.bias" in raw:
        pt["norm.bias"] = raw["norm.bias"]

    # blocks: try a few prefix schemes automatically
    depth = len(getattr(eqx_tree, "attention_blocks"))

    def find_block_prefix(i: int) -> Optional[str]:
        cand_roots = ["blocks", "encoder.block", "encoder.layers"]
        for root in cand_roots:
            p = f"{root}.{i}."
            if any(k.startswith(p) for k in raw.keys()):
                return f"{root}.{i}"
        # also allow chunked blocks like "blocks.0.{i}."
        m = [k for k in raw.keys() if re.search(rf"blocks\.\d+\.{i}\.", k)]
        if m:
            return re.sub(rf"(blocks\.\d+\.{i}).*", r"\1", m[0])
        # last resort: model.blocks.* was stripped; try direct qkv key probe
        for k in raw.keys():
            if re.search(rf"(.*blocks\.){i}\.attn\.qkv\.weight$", k):
                return re.sub(r"\.attn\..*$", "", k)
        return None

    for i in range(depth):
        pref = find_block_prefix(i)
        if pref is None:
            continue

        # ln1/ln2 (DINOv2 uses norm1/norm2)
        for src, dst in (("norm1", "norm1"), ("norm2", "norm2")):
            w = f"{pref}.{src}.weight"
            b = f"{pref}.{src}.bias"
            if w in raw:
                pt[f"attention_blocks.{i}.{dst}.weight"] = raw[w]
            if b in raw:
                pt[f"attention_blocks.{i}.{dst}.bias"] = raw[b]

        # qkv fused → split
        for qkv_w in (
            f"{pref}.attn.qkv.weight",
            f"{pref}.self_attention.qkv.weight",
            f"{pref}.attn.in_proj_weight",
        ):
            if qkv_w in raw:
                W = raw[qkv_w]
                Wq, Wk, Wv = jnp.split(W, 3, axis=0)
                pt[f"attention_blocks.{i}.attn.q_proj.weight"] = Wq
                pt[f"attention_blocks.{i}.attn.k_proj.weight"] = Wk
                pt[f"attention_blocks.{i}.attn.v_proj.weight"] = Wv
                break
        for qkv_b in (
            f"{pref}.attn.qkv.bias",
            f"{pref}.self_attention.qkv.bias",
            f"{pref}.attn.in_proj_bias",
        ):
            if qkv_b in raw:
                bq, bk, bv = jnp.split(raw[qkv_b], 3, axis=0)
                pt[f"attention_blocks.{i}.attn.q_proj.bias"] = bq
                pt[f"attention_blocks.{i}.attn.k_proj.bias"] = bk
                pt[f"attention_blocks.{i}.attn.v_proj.bias"] = bv
                break

        # out proj
        for ow in (
            f"{pref}.attn.proj.weight",
            f"{pref}.attention.proj.weight",
            f"{pref}.attn.out_proj.weight",
        ):
            if ow in raw:
                pt[f"attention_blocks.{i}.attn.out_proj.weight"] = raw[ow]
                break
        for ob in (
            f"{pref}.attn.proj.bias",
            f"{pref}.attention.proj.bias",
            f"{pref}.attn.out_proj.bias",
        ):
            if ob in raw:
                pt[f"attention_blocks.{i}.attn.out_proj.bias"] = raw[ob]
                break

        # mlp (fc1/fc2 or 0/3)
        for w1 in (f"{pref}.mlp.fc1.weight", f"{pref}.mlp.0.weight"):
            if w1 in raw:
                pt[f"attention_blocks.{i}.mlp1.weight"] = raw[w1]
                break
        for b1 in (f"{pref}.mlp.fc1.bias", f"{pref}.mlp.0.bias"):
            if b1 in raw:
                pt[f"attention_blocks.{i}.mlp1.bias"] = raw[b1]
                break
        for w2 in (f"{pref}.mlp.fc2.weight", f"{pref}.mlp.3.weight"):
            if w2 in raw:
                pt[f"attention_blocks.{i}.mlp2.weight"] = raw[w2]
                break
        for b2 in (f"{pref}.mlp.fc2.bias", f"{pref}.mlp.3.bias"):
            if b2 in raw:
                pt[f"attention_blocks.{i}.mlp2.bias"] = raw[b2]
                break

        # LayerScale gamma/grandma
        for tag in ("ls1", "ls2"):
            g = raw.get(f"{pref}.{tag}.gamma", raw.get(f"{pref}.{tag}.grandma", None))
            if g is not None:
                pt[f"attention_blocks.{i}.{tag}"] = g

    # head (often identity)
    if "head.weight" in raw and "weight" in vars(eqx_tree.head):
        pt["head.weight"] = raw["head.weight"]
    if "head.bias" in raw and getattr(eqx_tree.head, "bias", None) is not None:
        pt["head.bias"] = raw["head.bias"]

    # copy into tree
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
        want = tuple(getattr(new_tree.head, "weight").shape)
        have = tuple(pt["head.weight"].shape)
        if want != have:
            raise ValueError(
                f"Head mismatch: PT {have} vs model {want} (strict_fc=True)."
            )

    if verbose:
        print(
            f"[dinov2 loader] coverage ~{100*report['coverage']:.2f}% | spectral-warmstarted: {report['spectral_warmstarted']}"
        )
        if report["unused_keys"]:
            print(
                f"[dinov2 loader] mapped-but-unused params: {len(report['unused_keys'])}"
            )
        if report["spectral_errors"]:
            print(
                f"[dinov2 loader] spectral warmstart errors: {report['spectral_errors'][:3]} ..."
            )
    return new_tree, report
