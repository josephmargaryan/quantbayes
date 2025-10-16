# quantbayes/stochax/vision_common/pretrained_vit.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Literal, Optional

import math
import numpy as np
import jax.numpy as jnp
import jax.image as jimg
import equinox as eqx
from equinox import tree_at

# ---- Optional spectral imports (if present) ----
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


def _svd_linear(W: jnp.ndarray):
    U, s, Vh = jnp.linalg.svd(W, full_matrices=False)
    return U, s, Vh.T


def _kget_global(raw: Dict[str, np.ndarray], candidates):
    """Return the first existing key (as jnp array) among candidates, else None."""
    for k in candidates:
        v = raw.get(k, None)
        if v is not None:
            return jnp.asarray(v)
    return None


def _kget_layer(raw: Dict[str, np.ndarray], i: int, suffixes):
    """
    Layer-wise key lookup that is robust to multiple naming schemes:
      - torchvision-ish:  encoder.layers.{i}.*
      - your file:        encoder.layers.encoder_layer_{i}.*
      - timm-ish:         blocks.{i}.*
      - HF-ish:           encoder.block.{i}.*
    """
    prefixes = [
        f"encoder.layers.{i}",
        f"encoder.layers.encoder_layer_{i}",
        f"blocks.{i}",
        f"encoder.block.{i}",
    ]
    for pref in prefixes:
        for suf in suffixes:
            k = f"{pref}.{suf}"
            v = raw.get(k, None)
            if v is not None:
                return jnp.asarray(v)
    return None


def _warmstart_svddense(mod, W, b, report):
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
    report["spectral_warmstarted"] += 1
    report["n_loaded"] += _numel(W) + (_numel(b) if (b is not None) else 0)
    return new


def _resize_pos_embedding(tv_pos: jnp.ndarray, target_len: int) -> jnp.ndarray:
    """tv_pos: [1, L_tv, D] -> [1, L_target, D] (2D aware, keeps CLS at 0)."""
    assert tv_pos.ndim == 3 and tv_pos.shape[0] == 1
    cls = tv_pos[:, :1, :]
    seq = tv_pos[:, 1:, :]  # [1, L-1, D]
    if 1 + seq.shape[1] == target_len:
        return tv_pos
    old_n = int(seq.shape[1])
    new_n = int(target_len - 1)
    old_hw = int(round(math.sqrt(old_n)))
    new_hw = int(round(math.sqrt(new_n)))
    if old_hw * old_hw != old_n or new_hw * new_hw != new_n:
        # length-only resize fallback
        seq_resized = jimg.resize(seq, (1, new_n, seq.shape[-1]), method="linear")
        return jnp.concatenate([cls, seq_resized], axis=1)
    # 2D resize
    seq_2d = seq.reshape(1, old_hw, old_hw, seq.shape[-1])
    seq_resized = jimg.resize(
        seq_2d, (1, new_hw, new_hw, seq.shape[-1]), method="linear"
    )
    seq_resized = seq_resized.reshape(1, new_hw * new_hw, seq.shape[-1])
    return jnp.concatenate([cls, seq_resized], axis=1)


def _infer_linear_want_shape(obj) -> Optional[Tuple[int, int]]:
    """Return (out_features, in_features) for Linear or SVDDense; else None."""
    try:
        if isinstance(obj, eqx.nn.Linear):
            w = obj.weight
            return (int(w.shape[0]), int(w.shape[1]))
        if _HAS_SPECTRAL and isinstance(obj, SVDDense):
            # Prefer statics; fall back to U/V shapes if needed.
            out_f = int(getattr(obj, "out_features", obj.U.shape[0]))
            in_f = int(getattr(obj, "in_features", obj.V.shape[0]))
            return (out_f, in_f)
    except Exception:
        pass
    return None


def _copy_into(
    obj: Any,
    pt: Dict[str, jnp.ndarray],
    *,
    prefix: str,
    spectral_warmstart: Literal["skip", "svd"],
    strict_fc: bool,
    report: Dict[str, Any],
) -> Any:
    """Generic pytree copier for ViT: Linear, LayerNorm, ndarray leaves, SVDDense warm-start."""
    w_key, b_key = f"{prefix}.weight", f"{prefix}.bias"

    # Raw ndarray leaves (e.g., cls_token, positional_embedding)
    if isinstance(obj, jnp.ndarray):
        if prefix in pt and tuple(pt[prefix].shape) == tuple(obj.shape):
            report["used"].add(prefix)
            report["n_loaded"] += _numel(pt[prefix])
            return pt[prefix]
        return obj

    # --- Handle spectral dense FIRST to avoid touching .weight on SVDDense ---
    if _HAS_SPECTRAL and isinstance(obj, SVDDense) and spectral_warmstart == "svd":
        if w_key in pt:
            try:
                new = _warmstart_svddense(obj, pt[w_key], pt.get(b_key), report)
                # Mark keys as used on success (we already updated n_loaded inside warmstart)
                report["used"].add(w_key)
                if b_key in pt:
                    report["used"].add(b_key)
                return new
            except Exception as e:
                report["spectral_errors"].append((prefix, repr(e)))
                # Even on failure, we attempted to consume these keys.
                report["used"].add(w_key)
                if b_key in pt:
                    report["used"].add(b_key)
                return obj

    # eqx.nn.Linear
    if isinstance(obj, eqx.nn.Linear):
        if w_key in pt:
            W = pt[w_key]
            if tuple(W.shape) == tuple(obj.weight.shape):
                new = tree_at(lambda m: m.weight, obj, W)
                report["used"].add(w_key)
                report["n_loaded"] += _numel(W)
                if (b_key in pt) and (obj.bias is not None):
                    new = tree_at(lambda m: m.bias, new, pt[b_key])
                    report["used"].add(b_key)
                    report["n_loaded"] += _numel(pt[b_key])
                return new
            if ("head.weight" in w_key) and (not strict_fc):
                report["skipped_fc"].append(
                    (w_key, tuple(W.shape), tuple(obj.weight.shape))
                )
                return obj
            report["mismatch"].append((w_key, tuple(W.shape), tuple(obj.weight.shape)))
        return obj

    # eqx.nn.LayerNorm (affine)
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

    # Recurse containers / modules
    if isinstance(obj, eqx.Module):
        new_obj = obj
        for name, child in vars(obj).items():
            child_prefix = name if prefix == "" else f"{prefix}.{name}"
            new_child = _copy_into(
                child,
                pt,
                prefix=child_prefix,
                spectral_warmstart=spectral_warmstart,
                strict_fc=strict_fc,
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
                spectral_warmstart=spectral_warmstart,
                strict_fc=strict_fc,
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
                spectral_warmstart=spectral_warmstart,
                strict_fc=strict_fc,
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
                spectral_warmstart=spectral_warmstart,
                strict_fc=strict_fc,
                report=report,
            )
            for k, v in obj.items()
        }
    return obj


def load_imagenet_vit(
    eqx_tree: eqx.Module,
    npz_path: str,
    *,
    strict_fc: bool = True,
    spectral_warmstart: Literal["skip", "svd"] = "skip",
    verbose: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any]]:
    """
    Load torchvision/timm/HF-style ViT .npz into your Equinox ViT.
    - Handles conv_proj/patch_embed → Linear (patch embedding)
    - Resizes positional embedding if token count differs
    - Maps LN/attn/MLP across several naming schemes
    - SVD-warm-starts SVDDense if spectral_warmstart='svd'
    """
    raw = dict(np.load(npz_path))
    pt: Dict[str, jnp.ndarray] = {}

    # 1) Patch embedding: conv_proj (E,C,ph,pw) or patch_embed.proj
    W_patch = _kget_global(
        raw,
        [
            "conv_proj.weight",  # torchvision
            "patch_embed.proj.weight",  # timm
            "embedding.conv.weight",  # occasional alt
        ],
    )
    if W_patch is not None:
        E, C, ph, pw = W_patch.shape
        pt["patch_embedding.linear.weight"] = W_patch.reshape(E, C * ph * pw)
        B_patch = _kget_global(
            raw,
            [
                "conv_proj.bias",
                "patch_embed.proj.bias",
                "embedding.conv.bias",
            ],
        )
        if B_patch is not None:
            pt["patch_embedding.linear.bias"] = B_patch

    # 2) CLS + positional embedding (resize as needed)
    cls_tok = _kget_global(raw, ["class_token", "cls_token"])
    if cls_tok is not None:
        pt["cls_token"] = cls_tok.reshape(1, -1)

    pos_tv = _kget_global(raw, ["encoder.pos_embedding", "pos_embed"])
    if pos_tv is not None:
        pos_tv = jnp.asarray(pos_tv)  # [1, L_tv, D] or [L_tv, D]
        if pos_tv.ndim == 2:
            pos_tv = pos_tv[None, ...]
        L_target = int(eqx_tree.positional_embedding.shape[0])  # [L, D]
        pos_ours = _resize_pos_embedding(pos_tv, L_target).squeeze(0)  # [L, D]
        pt["positional_embedding"] = pos_ours

    # 3) Per-layer mappings (robust)
    n_layers = len(getattr(eqx_tree, "attention_blocks"))
    for i in range(n_layers):
        # LN1 / LN2
        w = _kget_layer(raw, i, ["ln_1.weight", "ln1.weight", "norm1.weight"])
        b = _kget_layer(raw, i, ["ln_1.bias", "ln1.bias", "norm1.bias"])
        if w is not None:
            pt[f"attention_blocks.{i}.layer_norm1.weight"] = w
        if b is not None:
            pt[f"attention_blocks.{i}.layer_norm1.bias"] = b

        w = _kget_layer(raw, i, ["ln_2.weight", "ln2.weight", "norm2.weight"])
        b = _kget_layer(raw, i, ["ln_2.bias", "ln2.bias", "norm2.bias"])
        if w is not None:
            pt[f"attention_blocks.{i}.layer_norm2.weight"] = w
        if b is not None:
            pt[f"attention_blocks.{i}.layer_norm2.bias"] = b

        # QKV: either fused qkv.* or in_proj_*
        W_qkv = _kget_layer(
            raw,
            i,
            [
                "self_attention.in_proj_weight",
                "attn.in_proj_weight",
                "attention.in_proj_weight",
                "self_attention.qkv.weight",
                "attn.qkv.weight",
                "attention.qkv.weight",
            ],
        )
        b_qkv = _kget_layer(
            raw,
            i,
            [
                "self_attention.in_proj_bias",
                "attn.in_proj_bias",
                "attention.in_proj_bias",
                "self_attention.qkv.bias",
                "attn.qkv.bias",
                "attention.qkv.bias",
            ],
        )
        if W_qkv is not None:
            W_qkv = jnp.asarray(W_qkv)
            # expected [3D, D]; timm also uses this layout for qkv.weight
            Wq, Wk, Wv = jnp.split(W_qkv, 3, axis=0)
            pt[f"attention_blocks.{i}.attention.q_proj.weight"] = Wq
            pt[f"attention_blocks.{i}.attention.k_proj.weight"] = Wk
            pt[f"attention_blocks.{i}.attention.v_proj.weight"] = Wv
        if b_qkv is not None:
            b_qkv = jnp.asarray(b_qkv)
            bq, bk, bv = jnp.split(b_qkv, 3, axis=0)
            pt[f"attention_blocks.{i}.attention.q_proj.bias"] = bq
            pt[f"attention_blocks.{i}.attention.k_proj.bias"] = bk
            pt[f"attention_blocks.{i}.attention.v_proj.bias"] = bv

        # Attention out projection: out_proj.* or proj.*
        W_o = _kget_layer(
            raw,
            i,
            [
                "self_attention.out_proj.weight",
                "attn.out_proj.weight",
                "attention.out_proj.weight",
                "attn.proj.weight",
                "attention.proj.weight",
            ],
        )
        b_o = _kget_layer(
            raw,
            i,
            [
                "self_attention.out_proj.bias",
                "attn.out_proj.bias",
                "attention.out_proj.bias",
                "attn.proj.bias",
                "attention.proj.bias",
            ],
        )
        if W_o is not None:
            pt[f"attention_blocks.{i}.attention.out_proj.weight"] = jnp.asarray(W_o)
        if b_o is not None:
            pt[f"attention_blocks.{i}.attention.out_proj.bias"] = jnp.asarray(b_o)

        # MLP: fc1/fc2 or 0/3
        w = _kget_layer(raw, i, ["mlp.fc1.weight", "mlp.0.weight"])
        b = _kget_layer(raw, i, ["mlp.fc1.bias", "mlp.0.bias"])
        if w is not None:
            pt[f"attention_blocks.{i}.linear1.weight"] = jnp.asarray(w)
        if b is not None:
            pt[f"attention_blocks.{i}.linear1.bias"] = jnp.asarray(b)

        w = _kget_layer(raw, i, ["mlp.fc2.weight", "mlp.3.weight"])
        b = _kget_layer(raw, i, ["mlp.fc2.bias", "mlp.3.bias"])
        if w is not None:
            pt[f"attention_blocks.{i}.linear2.weight"] = jnp.asarray(w)
        if b is not None:
            pt[f"attention_blocks.{i}.linear2.bias"] = jnp.asarray(b)

    # 4) final LN + head
    w_final_ln = _kget_global(
        raw, ["encoder.ln.weight", "encoder.norm.weight", "norm.weight"]
    )
    b_final_ln = _kget_global(
        raw, ["encoder.ln.bias", "encoder.norm.bias", "norm.bias"]
    )
    if w_final_ln is not None:
        pt["norm.weight"] = w_final_ln
    if b_final_ln is not None:
        pt["norm.bias"] = b_final_ln

    W_head = _kget_global(
        raw,
        [
            "heads.head.weight",
            "head.weight",
            "classifier.weight",
            "mlp_head.dense.weight",
        ],
    )
    B_head = _kget_global(
        raw, ["heads.head.bias", "head.bias", "classifier.bias", "mlp_head.dense.bias"]
    )
    if (W_head is not None) and (B_head is not None):
        want = _infer_linear_want_shape(
            eqx_tree.head
        )  # (out,in) for Linear or SVDDense; None if unknown
        have = tuple(W_head.shape)
        if want is None:
            if strict_fc:
                raise ValueError(
                    "Unsupported classifier head type for strict_fc=True; "
                    "set strict_fc=False to skip loading the head."
                )
        else:
            if have == want:
                pt["head.weight"] = W_head
                pt["head.bias"] = B_head
            elif strict_fc:
                raise ValueError(
                    f"Head shape mismatch: PT {have} vs model {want}. "
                    f"Set strict_fc=False to skip classifier head."
                )
            # else: skip head silently when not strict

    # ---- copy into tree (with optional SVDDense SVD warm-start) ----
    report = {
        "used": set(),
        "unused_keys": [],
        "mismatch": [],
        "skipped_fc": [],
        "spectral_warmstarted": 0,
        "spectral_errors": [],
        "n_loaded": 0,
        "n_total_pt": sum(_numel(v) for v in pt.values()),
    }

    new_tree = _copy_into(
        eqx_tree,
        pt,
        prefix="",
        spectral_warmstart=spectral_warmstart,
        strict_fc=strict_fc,
        report=report,
    )
    report["unused_keys"] = sorted(set(pt.keys()) - report["used"])
    report["coverage"] = float(report["n_loaded"]) / max(1, report["n_total_pt"])

    if verbose:
        print(
            f"[vit loader] coverage ~{100*report['coverage']:.2f}% "
            f"| spectral warm-started: {report['spectral_warmstarted']}"
        )
        if report["skipped_fc"]:
            print(
                f"[vit loader] skipped head due to shape mismatch (strict_fc={strict_fc})."
            )
        if report["mismatch"]:
            print(f"[vit loader] mismatches: {len(report['mismatch'])}.")
        if report["unused_keys"]:
            print(f"[vit loader] unused PT keys: {len(report['unused_keys'])}.")

    return new_tree, report


# Convenience wrappers per-arch (filenames you saved via your script)
def load_imagenet_vit_b_16(m, npz="vit_b_16_imagenet.npz", **kw):
    return load_imagenet_vit(m, npz, **kw)


def load_imagenet_vit_b_32(m, npz="vit_b_32_imagenet.npz", **kw):
    return load_imagenet_vit(m, npz, **kw)


def load_imagenet_vit_l_16(m, npz="vit_l_16_imagenet.npz", **kw):
    return load_imagenet_vit(m, npz, **kw)


def load_imagenet_vit_l_32(m, npz="vit_l_32_imagenet.npz", **kw):
    return load_imagenet_vit(m, npz, **kw)


def load_imagenet_vit_h_14(m, npz="vit_h_14_imagenet.npz", **kw):
    return load_imagenet_vit(m, npz, **kw)
