# dinov2_loader.py
"""
Load a DINOv2 ViT checkpoint saved as .npz into DINOViTEncoder (Equinox).

- Handles official FAIR/timm/HF-ish key patterns.
- Supports fused or split (q,k,v) attention projections.
- Converts 1D pos_embed (with/without CLS) to 2D learned grid.
- Verifies/register-token count when strict=True.

Usage:
    enc = build_dinov2_encoder("vitb14", image_size=518, num_registers=4, use_cls=True, key=...)
    enc = load_dinov2_npz(enc, "dinov2-base.npz", strict=True)
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import jax.numpy as jnp
import equinox as eqx

from .vit_dino_eqx import DINOViTEncoder


def _sqrt_int(n: int) -> Tuple[int, int]:
    r = int(round(n**0.5))
    if r * r == n:
        return r, r
    # Try a nearby factorization (rectangular grids)
    for a in range(max(1, r - 8), r + 9):
        if a > 0 and n % a == 0:
            return a, n // a
    # Fallback: nearest square
    return r, max(1, n // max(1, r))


def _strip_prefixes(k: str) -> str:
    for p in ("module.", "model.", "backbone.", "vit."):
        if k.startswith(p):
            return k[len(p) :]
    return k


def _rename_dinov2_key(k: str) -> str:
    """
    Map common DINOv2/timm/HF-ish keys to our Equinox tree names.

    Accepted endpoints after rename:
      - patch_embed.(weight|bias)
      - cls_token
      - reg_token  (or register_tokens)
      - pos_embed
      - blocks.{i}.(ln1|ln2).(weight|bias)
      - blocks.{i}.attn.(qkv.weight|qkv.bias|proj.weight|proj.bias)
      - blocks.{i}.mlp.(fc1|fc2).(weight|bias)
      - norm.(weight|bias)
    """
    k = _strip_prefixes(k)

    # Patch embed (HF: embeddings.patch_embeddings.projection.*)
    k = k.replace("embeddings.patch_embeddings.projection.", "patch_embed.")
    k = k.replace("patch_embed.proj.", "patch_embed.")
    k = k.replace("patch_embed.projection.", "patch_embed.")

    # Blocks
    k = k.replace("blocks.", "blocks.")
    k = k.replace(".norm1.", ".ln1.")
    k = k.replace(".norm2.", ".ln2.")

    # Attention (HF split paths)
    k = k.replace(".attention.attention.query.", ".attn.qkv_q.")
    k = k.replace(".attention.attention.key.", ".attn.qkv_k.")
    k = k.replace(".attention.attention.value.", ".attn.qkv_v.")
    k = k.replace(".attention.output.dense.", ".attn.proj.")

    # MLP (timm/HF)
    k = k.replace(".mlp.fc1.", ".mlp.fc1.")
    k = k.replace(".mlp.fc2.", ".mlp.fc2.")
    k = k.replace(".intermediate.dense.", ".mlp.fc1.")
    k = k.replace(".output.dense.", ".mlp.fc2.")

    # Final norm
    k = k.replace("norm.", "norm.")

    # Positional & tokens
    if k in (
        "pos_embed",
        "embeddings.position_embeddings",
        "embeddings.position_embedding",
    ):
        k = "pos_embed"
    if k in ("cls_token", "embeddings.cls_token"):
        k = "cls_token"
    if k in ("reg_token", "register_token", "register_tokens"):
        k = "reg_token"
    return k


def _copy_into_tree(obj, pt: Dict[str, jnp.ndarray], prefix: str = ""):
    """Recursively copy Conv2d/Linear/LayerNorm params (and ndarray leaves) into an Equinox pytree."""
    if isinstance(obj, eqx.Module):
        for name, attr in vars(obj).items():
            full = f"{prefix}{name}"

            # Conv2d
            if isinstance(attr, eqx.nn.Conv2d):
                new_attr = attr
                if f"{full}.weight" in pt:
                    new_attr = eqx.tree_at(
                        lambda m: m.weight, new_attr, pt[f"{full}.weight"]
                    )
                if f"{full}.bias" in pt:
                    new_attr = eqx.tree_at(
                        lambda m: m.bias, new_attr, pt[f"{full}.bias"]
                    )
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, new_attr)
                continue

            # Linear
            if isinstance(attr, eqx.nn.Linear):
                new_attr = attr
                if f"{full}.weight" in pt:
                    new_attr = eqx.tree_at(
                        lambda m: m.weight, new_attr, pt[f"{full}.weight"]
                    )
                if f"{full}.bias" in pt:
                    new_attr = eqx.tree_at(
                        lambda m: m.bias, new_attr, pt[f"{full}.bias"]
                    )
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, new_attr)
                continue

            # LayerNorm
            if isinstance(attr, eqx.nn.LayerNorm):
                w_key, b_key = f"{full}.weight", f"{full}.bias"
                if (w_key in pt) or (b_key in pt):
                    obj = eqx.tree_at(
                        lambda m: (getattr(m, name).weight, getattr(m, name).bias),
                        obj,
                        (
                            pt.get(w_key, getattr(attr, "weight")),
                            pt.get(b_key, getattr(attr, "bias")),
                        ),
                    )
                continue

            # Tuples of submodules
            if isinstance(attr, tuple):
                new_tuple = []
                for i, child in enumerate(attr):
                    child_full = f"{full}.{i}"
                    new_tuple.append(_copy_into_tree(child, pt, prefix=child_full))
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, tuple(new_tuple))
                continue

            # Raw ndarray leaves (tokens/pos_grid) handled explicitly elsewhere
        return obj
    return obj


def load_dinov2_npz(
    encoder: DINOViTEncoder, npz_path: str, *, strict: bool = True
) -> DINOViTEncoder:
    """
    Load a DINOv2-style ViT checkpoint saved as .npz into our DINOViTEncoder.

    Important expected mappings:
      - patch_embed.(weight|bias)
      - cls_token, reg_token (optional)
      - pos_embed: [1,N,D] or [Gh,Gw,D] or [N,D]  → pos_grid: [Gh,Gw,D]
      - blocks.{i}.(ln1|ln2).(weight|bias)
      - blocks.{i}.attn.(qkv.weight|qkv.bias|proj.weight|proj.bias)
      - blocks.{i}.mlp.(fc1|fc2).(weight|bias)
      - norm.(weight|bias)
    """
    raw = dict(np.load(npz_path, allow_pickle=False))
    # transform keys
    pt: Dict[str, jnp.ndarray] = {}
    for k, v in raw.items():
        new_k = _rename_dinov2_key(k)
        pt[new_k] = jnp.asarray(v)

    # ---- tokens & pos grid (explicit) ---- #
    # cls_token
    if "cls_token" in pt:
        ct = pt["cls_token"]
        if ct.ndim == 3:  # [1,1,D]
            ct = ct[0, 0]
        elif ct.ndim == 2:  # [1,D]
            ct = ct[0]
        assert ct.shape == (
            encoder.dim,
        ), f"cls_token dim mismatch: {ct.shape} vs {(encoder.dim,)}"
        encoder = eqx.tree_at(lambda m: m.cls_token, encoder, ct)

    # reg_token(s)
    if "reg_token" in pt:
        rt = pt["reg_token"]
        if rt.ndim == 3:  # [1,R,D]
            rt = rt[0]
        assert (
            rt.shape[-1] == encoder.dim
        ), f"reg_token D mismatch: {rt.shape} vs (*,{encoder.dim})"
        R = rt.shape[0]
        if encoder.num_registers != R:
            if strict:
                raise ValueError(
                    f"Checkpoint has {R} register tokens; encoder.num_registers={encoder.num_registers}."
                )
        encoder = eqx.tree_at(lambda m: m.reg_tokens, encoder, rt)

    # pos_embed → pos_grid [Gh,Gw,D] (strip cls if present)
    if "pos_embed" in pt:
        pe = pt["pos_embed"]
        if pe.ndim == 3 and pe.shape[0] == 1:
            pe = pe[0]  # [N,D] or [Gh,Gw,D]
        if pe.ndim == 2:  # [N, D] → try to infer grid
            N, D = pe.shape
            # If first token is CLS, drop it (common HF convention)
            if encoder.use_cls and (N - 1) in (
                encoder.pos_grid.shape[0] * encoder.pos_grid.shape[1],
            ):
                pe = pe[1:]
                N = N - 1
            Gh, Gw = _sqrt_int(N)
            pe = pe.reshape(Gh, Gw, D)
        assert (
            pe.shape[-1] == encoder.dim
        ), f"pos_embed dim mismatch: {pe.shape} vs (*,* ,{encoder.dim})"
        encoder = eqx.tree_at(lambda m: m.pos_grid, encoder, pe)

    # ---- fuse split q/k/v (HF) into qkv ---- #
    q_keys = [k for k in list(pt.keys()) if ".attn.qkv_q." in k]
    if q_keys:
        fused: Dict[str, jnp.ndarray] = {}
        for kq in q_keys:
            base = kq.replace(".attn.qkv_q.", ".attn.qkv.")
            kk = kq.replace("_q.", "_k.")
            kv = kq.replace("_q.", "_v.")
            Wq, Wk, Wv = pt[kq], pt[kk], pt[kv]
            fused[base + "weight"] = jnp.concatenate([Wq, Wk, Wv], axis=0)
            bq = pt.get(kq.replace("weight", "bias"))
            bk = pt.get(kk.replace("weight", "bias"))
            bv = pt.get(kv.replace("weight", "bias"))
            if bq is not None and bk is not None and bv is not None:
                fused[base + "bias"] = jnp.concatenate([bq, bk, bv], axis=0)
            # remove originals
            for kk_ in (
                kq,
                kk,
                kv,
                kq.replace("weight", "bias"),
                kk.replace("weight", "bias"),
                kv.replace("weight", "bias"),
            ):
                pt.pop(kk_, None)
        pt.update(fused)

    # ---- copy rest into the tree ---- #
    encoder = _copy_into_tree(encoder, pt, prefix="")

    return encoder
