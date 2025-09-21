# quantbayes/stochax/vision_backbones/dino/dinov2_loader.py
import jax
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, Literal

import re
import math
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from equinox import tree_at


_HAS_SPECTRAL = False
SVDDense = object  # placeholder
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


def _resize_pos_embed_2d(tv_pos: jnp.ndarray, target_len: int) -> jnp.ndarray:
    """tv_pos: [1,L_tv,D] or [L_tv,D] → [1, L_target, D]; keeps CLS at 0; 2D-aware if square."""
    if tv_pos.ndim == 2:
        tv_pos = tv_pos[None, ...]
    cls = tv_pos[:, :1, :]
    seq = tv_pos[:, 1:, :]  # [1, L-1, D]
    L_tv = int(seq.shape[1])
    L_tgt = int(target_len - 1)
    if L_tv == L_tgt:
        return tv_pos
    old_hw = int(round(math.sqrt(L_tv)))
    new_hw = int(round(math.sqrt(L_tgt)))
    if old_hw * old_hw == L_tv and new_hw * new_hw == L_tgt:
        # 2D resize
        seq2 = seq.reshape(1, old_hw, old_hw, seq.shape[-1])
        seq2 = jax.image.resize(
            seq2, (1, new_hw, new_hw, seq.shape[-1]), method="linear"
        )
        seq2 = seq2.reshape(1, new_hw * new_hw, seq.shape[-1])
    else:
        # length-only fallback
        seq2 = jax.image.resize(seq, (1, L_tgt, seq.shape[-1]), method="linear")
    return jnp.concatenate([cls, seq2], axis=1)


def _infer_block_prefixes(keys: Dict[str, Any], depth: int) -> Dict[int, str]:
    """
    DINOv2 checkpoints sometimes chunk blocks: e.g. 'blocks.0.3.attn.qkv.weight'.
    Return a dict i->prefix like 'blocks.{i}' or 'blocks.0.{i}' that actually exists.
    """
    kset = set(keys)
    out = {}
    for i in range(depth):
        candidates = [
            f"blocks.{i}",
            f"blocks.0.{i}",
            f"backbone.blocks.{i}",
            f"backbone.blocks.0.{i}",
            f"model.backbone.blocks.{i}",
            f"encoder.block.{i}",
            f"encoder.layers.{i}",
        ]
        for p in candidates:
            if any(k.startswith(p + ".") for k in kset):
                out[i] = p
                break
        if i not in out:
            # try regex
            for k in kset:
                m = re.match(r"(.*blocks\.\d+\.)" + str(i) + r"\.", k)
                if m:
                    out[i] = m.group(1) + str(i)
                    break
    return out


def _copy_svddense(mod, W, b, report):
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
    spectral_warmstart: bool,
    report: Dict[str, Any],
):
    w_key, b_key = f"{prefix}.weight", f"{prefix}.bias"

    if isinstance(obj, jnp.ndarray):
        if prefix in pt and tuple(pt[prefix].shape) == tuple(obj.shape):
            report["used"].add(prefix)
            report["n_loaded"] += _numel(pt[prefix])
            return pt[prefix]
        return obj

    if _HAS_SPECTRAL and isinstance(obj, SVDDense) and spectral_warmstart:
        if w_key in pt:
            try:
                new = _copy_svddense(obj, pt[w_key], pt.get(b_key), report)
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
                child,
                pt,
                prefix=child_prefix,
                spectral_warmstart=spectral_warmstart,
                report=report,
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
            seq.append(
                _copy_into(
                    c,
                    pt,
                    prefix=pfx,
                    spectral_warmstart=spectral_warmstart,
                    report=report,
                )
            )
        return type(obj)(seq)

    if isinstance(obj, dict):
        return {
            k: _copy_into(
                v,
                pt,
                prefix=f"{prefix}.{k}" if prefix else k,
                spectral_warmstart=spectral_warmstart,
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
    """
    Load a DINO/DINOv2 checkpoint saved as .npz (see save scripts).
    - Maps conv patch_embed to Linear
    - Handles fused qkv → (q,k,v) split
    - Resizes positional embedding to current num_patches (CLS+patches only)
    - Copies register_tokens if present
    - Optionally SVD-warm-starts SVDDense
    """
    raw = dict(np.load(npz_path, allow_pickle=False))
    # jnp-ify
    for k in list(raw.keys()):
        raw[k] = _as_jnp(raw[k])

    # build param table in terms of *your* module paths
    pt: Dict[str, jnp.ndarray] = {}

    # ---- Patch embedding conv -> linear
    Wpe = None
    for k in (
        "patch_embed.proj.weight",
        "backbone.patch_embed.proj.weight",
        "model.patch_embed.proj.weight",
    ):
        if k in raw:
            Wpe = raw[k]
            break
    if Wpe is not None:
        E, C, ph, pw = map(int, Wpe.shape)
        pt["patch_embedding.linear.weight"] = Wpe.reshape(E, C * ph * pw)
        b_key = None
        for k in (
            "patch_embed.proj.bias",
            "backbone.patch_embed.proj.bias",
            "model.patch_embed.proj.bias",
        ):
            if k in raw:
                b_key = k
                break
        if b_key is not None:
            pt["patch_embedding.linear.bias"] = raw[b_key]

    # ---- Tokens: cls / register / mask / pos_embed
    for k in ("cls_token", "backbone.cls_token", "model.cls_token"):
        if k in raw:
            v = raw[k].reshape(-1, raw[k].shape[-1])  # [1,D]
            pt["cls_token"] = v
            break
    # register tokens [1,R,D] or [R,D]
    for k in ("register_tokens", "backbone.register_tokens", "model.register_tokens"):
        if k in raw:
            v = raw[k]
            v = v.reshape(-1, v.shape[-1]) if v.ndim == 3 else v
            pt["register_tokens"] = v
            break
    # pos_embed [1,1+N_tv,D] -> [1+N_cur,D]
    for k in ("pos_embed", "backbone.pos_embed", "model.pos_embed"):
        if k in raw:
            tv = raw[k]
            # eqx tree carries target num_patches on attribute
            L_target = int(
                getattr(eqx_tree, "positional_embedding").shape[0]
            )  # [1+N, D]
            pos = _resize_pos_embed_2d(tv, L_target).squeeze(0)
            pt["positional_embedding"] = pos
            break

    # ---- Final norm
    for wname in ("norm.weight", "backbone.norm.weight", "model.norm.weight"):
        if wname in raw:
            pt["norm.weight"] = raw[wname]
            break
    for bname in ("norm.bias", "backbone.norm.bias", "model.norm.bias"):
        if bname in raw:
            pt["norm.bias"] = raw[bname]
            break

    # ---- Blocks
    depth = len(getattr(eqx_tree, "attention_blocks"))
    prefixes = _infer_block_prefixes(raw.keys(), depth)

    for i in range(depth):
        pref = prefixes.get(i, f"blocks.{i}")
        # LayerNorms
        for suf_src, suf_dst in (("norm1", "layer_norm1"), ("norm2", "layer_norm2")):
            w = f"{pref}.{suf_src}.weight"
            b = f"{pref}.{suf_src}.bias"
            if w in raw:
                pt[f"attention_blocks.{i}.{suf_dst}.weight"] = raw[w]
            if b in raw:
                pt[f"attention_blocks.{i}.{suf_dst}.bias"] = raw[b]

        # qkv fused
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
        # mlp
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
        # LayerScale (gamma or grandma)
        for tag, dst in (("ls1", "ls1"), ("ls2", "ls2")):
            g = None
            for gk in (f"{pref}.{tag}.gamma", f"{pref}.{tag}.grandma"):
                if gk in raw:
                    g = raw[gk]
                    break
            if g is not None:
                pt[f"attention_blocks.{i}.{dst}"] = g

    # ---- Head (often identity) — load only if shapes match or strict_fc=True
    for wname in ("head.weight", "backbone.head.weight", "model.head.weight"):
        if wname in raw:
            pt["head.weight"] = raw[wname]
            break
    for bname in ("head.bias", "backbone.head.bias", "model.head.bias"):
        if bname in raw:
            pt["head.bias"] = raw[bname]
            break

    # ---- now copy into tree
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
        eqx_tree,
        pt,
        prefix="",
        spectral_warmstart=(spectral_warmstart == "svd"),
        report=report,
    )
    report["unused_keys"] = sorted(set(pt.keys()) - report["used"])
    report["coverage"] = float(report["n_loaded"]) / max(1, report["n_total_pt"])

    # head shape sanity when strict
    if strict_fc and ("head.weight" in pt):
        want = tuple(getattr(new_tree.head, "weight").shape)
        have = tuple(pt["head.weight"].shape)
        if have != want:
            raise ValueError(
                f"Head shape mismatch PT {have} vs model {want} (strict_fc=True)."
            )

    if verbose:
        print(
            f"[dinov2 loader] coverage ~{100*report['coverage']:.2f}% | spectral-warmstarted: {report['spectral_warmstarted']}"
        )
        if report["unused_keys"]:
            print(
                f"[dinov2 loader] note: {len(report['unused_keys'])} mapped params unused (expected if layers absent)."
            )
        if report["mismatch"]:
            print(f"[dinov2 loader] mismatches: {len(report['mismatch'])}.")
    return new_tree, report
