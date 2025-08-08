"""
Equinox SegFormer (MiT-B0…B5) with HF weight loader and production-style test.

Highlights
----------
- Single-sample CHW inputs; strict `(x, key, state) -> (logits, state)`.
- Mix Transformer (MiT) encoder: OverlapPatchEmbeddings, Efficient Self-Attention
  with spatial reduction (sr_ratio), Mix-FFN (with DWConv), Stochastic Depth.
- Decode head: per-scale MLPs -> upsample-to-1/4 -> fuse (1x1 conv + BN + ReLU) -> classifier.
- Weight loader takes `.npz` saved from Hugging Face `state_dict()` and maps into Equinox.
  (Skips BN running stats; `strict_head` toggle for classifier shape mismatch.)

Where to get weights
--------------------
Hugging Face provides MiT backbones and full SegFormer checkpoints (B0…B5). Torchvision does not
provide SegFormer. Save them once as `.npz`:

# save_hf_segformer.py
# --------------------
from pathlib import Path
import numpy as np
import torch

# (A) Encoder-only checkpoints (MiT backbones)
ENCODERS = ["nvidia/mit-b0","nvidia/mit-b1","nvidia/mit-b2","nvidia/mit-b3","nvidia/mit-b4","nvidia/mit-b5"]

# (B) Full segmentation checkpoints (fine-tuned heads, e.g., ADE20k)
FULL = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "nvidia/segformer-b1-finetuned-ade-512-512",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b3-finetuned-ade-512-512",
    "nvidia/segformer-b4-finetuned-ade-512-512",
    "nvidia/segformer-b5-finetuned-ade-640-640",
]

def main():
    from transformers import SegformerModel, SegformerForSemanticSegmentation

    # Encoder-only (MiT)
    for repo in ENCODERS:
        print(f"⇢ downloading {repo} …")
        model = SegformerModel.from_pretrained(repo)
        out = Path(repo.split('/')[-1] + "_encoder.npz")
        print(f"↳ saving → {out}")
        np.savez(out, **{k: v.cpu().numpy() for k, v in model.state_dict().items()})
        print(f"✓ done {out}\n")

    # Full (encoder + decode head)
    for repo in FULL:
        print(f"⇢ downloading {repo} …")
        model = SegformerForSemanticSegmentation.from_pretrained(repo)
        out = Path(repo.split('/')[-1] + ".npz")
        print(f"↳ saving → {out}")
        np.savez(out, **{k: v.cpu().numpy() for k, v in model.state_dict().items()})
        print(f"✓ done {out}\n")

if __name__ == "__main__":
    main()


Usage
-----
import equinox as eqx, jax.random as jr, jax.numpy as jnp
from quantbayes.stochax.vision_segmentation.models.segformer import (
    SegFormer, load_hf_segformer_encoder, load_hf_segformer_full,
)

key = jr.PRNGKey(0)
model, state = eqx.nn.make_with_state(SegFormer)(
    arch="b0", num_classes=150, key=key   # 150 for ADE20K, e.g.
)

# Load encoder-only (MiT) pretrain
model = load_hf_segformer_encoder(model, "mit-b0_encoder.npz")

# Or load full (encoder + decode head). If you changed num_classes, set strict_head=False
# and fine-tune your classifier.
# model = load_hf_segformer_full(model, "segformer-b0-finetuned-ade-512-512.npz", strict_head=False)


Notes
-----
- Shapes: single-sample CHW everywhere.
- LayerNorm eps = 1e-6; default drops are zero (settable).
- BN in decode head uses mode="batch" (no EMA state), matching your codebase patterns.

References
----------
- Hugging Face SegFormer docs & configs (MiT depths, dims, heads, sr ratios, decoder dims).
- SegFormer paper (NeurIPS 2021).
- Torchvision model list (SegFormer not included).
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# --------------------------- small utils ---------------------------


class DropPath(eqx.Module):
    rate: float = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray, *, key: jnp.ndarray) -> jnp.ndarray:
        if self.rate <= 0.0:
            return x
        keep = 1.0 - self.rate
        mask = jr.bernoulli(key, p=keep)
        return x * (mask.astype(x.dtype) / keep)


# --------------------------- encoder: MiT ---------------------------


class SegformerOverlapPatchEmbeddings(eqx.Module):
    """OverlapPatchEmbeddings: conv k, stride s, padding=k//2, + LayerNorm (token-wise)."""

    proj: eqx.nn.Conv2d
    layer_norm: eqx.nn.LayerNorm
    patch_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)

    def __init__(self, in_ch: int, hidden: int, patch_size: int, stride: int, *, key):
        self.patch_size = patch_size
        self.stride = stride
        self.proj = eqx.nn.Conv2d(
            in_ch,
            hidden,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            use_bias=True,
            key=key,
        )
        self.layer_norm = eqx.nn.LayerNorm(hidden, eps=1e-6)

    def __call__(self, x_chw: jnp.ndarray):
        # conv
        x = self.proj(x_chw)  # [C', H', W']
        C, H, W = x.shape
        # tokens + LN (token-wise)
        tok = einops.rearrange(x, "c h w -> (h w) c")
        tok = jax.vmap(self.layer_norm)(tok)
        return tok, H, W


class SegformerEfficientSelfAttention(eqx.Module):
    """EMSA: q from tokens; k,v from spatially reduced tokens via sr conv + LN."""

    hidden_size: int = eqx.field(static=True)
    num_attention_heads: int = eqx.field(static=True)
    sr_ratio: int = eqx.field(static=True)

    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    sr: eqx.nn.Conv2d | None
    layer_norm: eqx.nn.LayerNorm  # for reduced tokens

    def __init__(
        self, hidden: int, heads: int, sr_ratio: int, attn_drop: float, *, key
    ):
        self.hidden_size = hidden
        self.num_attention_heads = heads
        self.sr_ratio = sr_ratio

        kq, kk, kv = jr.split(key, 3)
        self.query = eqx.nn.Linear(hidden, hidden, key=kq)
        self.key = eqx.nn.Linear(hidden, hidden, key=kk)
        self.value = eqx.nn.Linear(hidden, hidden, key=kv)
        self.dropout = eqx.nn.Dropout(attn_drop)

        if sr_ratio > 1:
            self.sr = eqx.nn.Conv2d(
                hidden,
                hidden,
                kernel_size=sr_ratio,
                stride=sr_ratio,
                use_bias=True,
                key=jr.fold_in(key, 7),
            )
        else:
            self.sr = None
        self.layer_norm = eqx.nn.LayerNorm(hidden, eps=1e-6)

    def __call__(self, tokens: jnp.ndarray, H: int, W: int, *, key: jnp.ndarray):
        # tokens: [N, C], N=H*W
        C = self.hidden_size
        Hh = self.num_attention_heads
        hd = C // Hh
        scale = 1.0 / jnp.sqrt(jnp.asarray(hd, dtype=jnp.float32))

        # q
        q = jax.vmap(self.query)(tokens)  # [N, C]
        q = einops.rearrange(q, "n (h d) -> h n d", h=Hh)

        # k,v
        if self.sr is None:
            red = tokens
        else:
            x = einops.rearrange(tokens, "(h w) c -> c h w", h=H, w=W)
            x = self.sr(x, key=jr.fold_in(key, 1))  # [C,H',W']
            Hr, Wr = x.shape[-2], x.shape[-1]
            red = einops.rearrange(x, "c h w -> (h w) c")
            red = jax.vmap(self.layer_norm)(red)

        k = jax.vmap(self.key)(red)  # [N' , C]
        v = jax.vmap(self.value)(red)
        k = einops.rearrange(k, "m (h d) -> h m d", h=Hh)
        v = einops.rearrange(v, "m (h d) -> h m d", h=Hh)

        # attention: [h, N, N']
        attn = jnp.einsum("hnd,hmd->hnm", q * scale, k)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, key=jr.fold_in(key, 2))

        out = jnp.einsum("hnm,hmd->hnd", attn, v)
        out = einops.rearrange(out, "h n d -> n (h d)")
        return out


class SegformerSelfOutput(eqx.Module):
    dense: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, hidden: int, drop: float, *, key):
        self.dense = eqx.nn.Linear(hidden, hidden, key=key)
        self.dropout = eqx.nn.Dropout(drop)

    def __call__(self, x: jnp.ndarray, *, key: jnp.ndarray):
        x = jax.vmap(self.dense)(x)
        x = self.dropout(x, key=key)
        return x


class SegformerAttention(eqx.Module):
    self_attn: SegformerEfficientSelfAttention
    output: SegformerSelfOutput

    def __init__(
        self,
        hidden: int,
        heads: int,
        sr_ratio: int,
        drop: float,
        attn_drop: float,
        *,
        key,
    ):
        k1, k2 = jr.split(key, 2)
        self.self_attn = SegformerEfficientSelfAttention(
            hidden, heads, sr_ratio, attn_drop, key=k1
        )
        self.output = SegformerSelfOutput(hidden, drop, key=k2)

    def __call__(self, tokens, H, W, *, key):
        y = self.self_attn(tokens, H, W, key=jr.fold_in(key, 0))
        y = self.output(y, key=jr.fold_in(key, 1))
        return y


class SegformerDWConv(eqx.Module):
    dwconv: eqx.nn.Conv2d

    def __init__(self, dim: int, *, key):
        try:
            # Depthwise: groups=dim (preferred)
            self.dwconv = eqx.nn.Conv2d(dim, dim, 3, padding=1, use_bias=True, key=key, groups=dim)  # type: ignore
        except TypeError:
            # Fallback if groups not supported in your eqx
            self.dwconv = eqx.nn.Conv2d(dim, dim, 3, padding=1, use_bias=True, key=key)

    def __call__(self, tokens: jnp.ndarray, H: int, W: int, *, key):
        x = einops.rearrange(tokens, "(h w) c -> c h w", h=H, w=W)
        x = self.dwconv(x, key=key)
        return einops.rearrange(x, "c h w -> (h w) c")


class SegformerMixFFN(eqx.Module):
    dense1: eqx.nn.Linear
    dwconv: SegformerDWConv
    dense2: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, dim: int, mlp_ratio: float, drop: float, *, key):
        k1, k2, k3 = jr.split(key, 3)
        hidden = int(dim * mlp_ratio)
        self.dense1 = eqx.nn.Linear(dim, hidden, key=k1)
        self.dwconv = SegformerDWConv(hidden, key=k2)
        self.dense2 = eqx.nn.Linear(hidden, dim, key=k3)
        self.dropout = eqx.nn.Dropout(drop)

    def __call__(self, tokens: jnp.ndarray, H: int, W: int, *, key):
        t = jax.vmap(self.dense1)(tokens)
        t = self.dwconv(t, H, W, key=jr.fold_in(key, 0))
        t = jax.nn.gelu(t)
        t = self.dropout(t, key=jr.fold_in(key, 1))
        t = jax.vmap(self.dense2)(t)
        t = self.dropout(t, key=jr.fold_in(key, 2))
        return t


class SegformerLayer(eqx.Module):
    layer_norm_1: eqx.nn.LayerNorm
    attention: SegformerAttention
    drop_path: DropPath
    layer_norm_2: eqx.nn.LayerNorm
    mlp: SegformerMixFFN

    def __init__(
        self,
        hidden: int,
        heads: int,
        sr_ratio: int,
        mlp_ratio: float,
        drop_path: float,
        drop: float,
        attn_drop: float,
        *,
        key,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.layer_norm_1 = eqx.nn.LayerNorm(hidden, eps=1e-6)
        self.attention = SegformerAttention(
            hidden, heads, sr_ratio, drop, attn_drop, key=k1
        )
        self.drop_path = DropPath(drop_path)
        self.layer_norm_2 = eqx.nn.LayerNorm(hidden, eps=1e-6)
        self.mlp = SegformerMixFFN(hidden, mlp_ratio, drop, key=k2)

    def __call__(self, tokens: jnp.ndarray, H: int, W: int, *, key: jnp.ndarray):
        k1, k2 = jr.split(key, 2)
        t = jax.vmap(self.layer_norm_1)(tokens)
        t = self.attention(t, H, W, key=k1)
        t = self.drop_path(t, key=k2)
        tokens = tokens + t

        t2 = jax.vmap(self.layer_norm_2)(tokens)
        t2 = self.mlp(t2, H, W, key=k1)
        t2 = self.drop_path(t2, key=k2)
        tokens = tokens + t2
        return tokens


class SegformerEncoder(eqx.Module):
    """4-stage hierarchical encoder; returns CHW features for each stage."""

    patch_embeddings: Tuple[SegformerOverlapPatchEmbeddings, ...]
    block: Tuple[Tuple[SegformerLayer, ...], ...]
    layer_norm: Tuple[eqx.nn.LayerNorm, ...]
    depths: Tuple[int, int, int, int] = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_ch: int,
        hidden_sizes: List[int],
        depths: List[int],
        heads: List[int],
        sr_ratios: List[int],
        patch_sizes: List[int],
        strides: List[int],
        drop_path_rate: float,
        drop: float,
        attn_drop: float,
        key,
    ):
        assert (
            len(hidden_sizes)
            == len(depths)
            == len(heads)
            == len(sr_ratios)
            == len(patch_sizes)
            == len(strides)
            == 4
        )
        self.depths = tuple(depths)

        keys = iter(jr.split(key, 4096))

        # patch embeddings
        pe = []
        in_c = in_ch
        for i in range(4):
            pe.append(
                SegformerOverlapPatchEmbeddings(
                    in_c, hidden_sizes[i], patch_sizes[i], strides[i], key=next(keys)
                )
            )
            in_c = hidden_sizes[i]
        self.patch_embeddings = tuple(pe)

        # stochastic depth schedule across all blocks
        total = sum(depths)
        sd = [(drop_path_rate * i) / max(1, total - 1) for i in range(total)]
        cur = 0
        blocks = []
        for i in range(4):
            layers = []
            for j in range(depths[i]):
                layers.append(
                    SegformerLayer(
                        hidden_sizes[i],
                        heads[i],
                        sr_ratios[i],
                        mlp_ratio=4.0,
                        drop_path=sd[cur],
                        drop=drop,
                        attn_drop=attn_drop,
                        key=next(keys),
                    )
                )
                cur += 1
            blocks.append(tuple(layers))
        self.block = tuple(blocks)

        # per-stage final LN
        self.layer_norm = tuple(eqx.nn.LayerNorm(h, eps=1e-6) for h in hidden_sizes)

    def __call__(self, x: jnp.ndarray, key, state):
        # x: [3,H,W]
        feats: List[jnp.ndarray] = []
        x_in = x
        k_run = key

        def split():
            nonlocal k_run
            k1, k_run = jr.split(k_run)
            return k1

        for i in range(4):
            # patchify current input
            tokens, H, W = self.patch_embeddings[i](x_in)
            # blocks
            for lyr in self.block[i]:
                tokens = lyr(tokens, H, W, key=split())
            # final stage LN and to CHW
            tokens = jax.vmap(self.layer_norm[i])(tokens)
            feat = einops.rearrange(tokens, "(h w) c -> c h w", h=H, w=W)
            feats.append(feat)
            x_in = feat  # next stage consumes CHW feature

        # returns [c1(1/4), c2(1/8), c3(1/16), c4(1/32)]
        return tuple(feats), state


# --------------------------- decode head ---------------------------


class SegformerMLP(eqx.Module):
    proj: eqx.nn.Linear

    def __init__(self, input_dim: int, out_dim: int, *, key):
        self.proj = eqx.nn.Linear(input_dim, out_dim, key=key)

    def __call__(self, x_chw: jnp.ndarray):
        # x_chw: [C,H,W] -> tokens -> linear -> [Out,H,W]
        C, H, W = x_chw.shape
        tok = einops.rearrange(x_chw, "c h w -> (h w) c")
        tok = jax.vmap(self.proj)(tok)
        return einops.rearrange(tok, "(h w) c -> c h w", h=H, w=W)


class SegformerDecodeHead(eqx.Module):
    linear_c: Tuple[SegformerMLP, ...]
    linear_fuse: eqx.nn.Conv2d
    batch_norm: eqx.nn.BatchNorm
    activation: Any  # ReLU via jax.nn.relu
    dropout: eqx.nn.Dropout
    classifier: eqx.nn.Conv2d
    decoder_hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_sizes: List[int],
        decoder_hidden_size: int,
        num_classes: int,
        *,
        key,
    ):
        keys = iter(jr.split(key, 256))
        self.linear_c = tuple(
            SegformerMLP(h, decoder_hidden_size, key=next(keys)) for h in hidden_sizes
        )
        self.linear_fuse = eqx.nn.Conv2d(
            decoder_hidden_size * 4,
            decoder_hidden_size,
            1,
            use_bias=False,
            key=next(keys),
        )
        self.batch_norm = eqx.nn.BatchNorm(
            decoder_hidden_size, axis_name="batch", mode="batch"
        )
        self.activation = None  # use jax.nn.relu
        self.dropout = eqx.nn.Dropout(0.1)
        self.classifier = eqx.nn.Conv2d(
            decoder_hidden_size, num_classes, 1, key=next(keys)
        )
        self.decoder_hidden_size = decoder_hidden_size

    def __call__(
        self,
        encoder_hidden_states: Tuple[
            jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
        ],
        key,
        state,
    ):
        # encoder_hidden_states: 4 CHW maps (1/4 .. 1/32)
        c1, c2, c3, c4 = encoder_hidden_states
        H0, W0 = c1.shape[-2], c1.shape[-1]

        # 1) per-scale MLP + upsample to 1/4
        outs = []
        for feat, mlp in zip((c1, c2, c3, c4), self.linear_c):
            y = mlp(feat)  # [Dh, H_i, W_i]
            if y.shape[-2:] != (H0, W0):
                y = jax.image.resize(y, (y.shape[0], H0, W0), method="linear")
            outs.append(y)

        # 2) fuse (concat in reverse order as in HF)
        x = jnp.concatenate(outs[::-1], axis=0)  # [4*Dh, H0, W0]
        x = self.linear_fuse(x, key=jr.fold_in(key, 0))
        x, state = self.batch_norm(x, state)
        x = jax.nn.relu(x)
        x = self.dropout(x, key=jr.fold_in(key, 1))
        logits = self.classifier(x, key=jr.fold_in(key, 2))  # [num_classes, H0, W0]
        return logits, state


# --------------------------- full model ---------------------------

_VARIANTS: Dict[str, Dict[str, Any]] = {
    # From HF docs/paper
    "b0": dict(
        depths=[2, 2, 2, 2], hidden=[32, 64, 160, 256], heads=[1, 2, 5, 8], decoder=256
    ),
    "b1": dict(
        depths=[2, 2, 2, 2], hidden=[64, 128, 320, 512], heads=[1, 2, 5, 8], decoder=256
    ),
    "b2": dict(
        depths=[3, 4, 6, 3], hidden=[64, 128, 320, 512], heads=[1, 2, 5, 8], decoder=768
    ),
    "b3": dict(
        depths=[3, 4, 18, 3],
        hidden=[64, 128, 320, 512],
        heads=[1, 2, 5, 8],
        decoder=768,
    ),
    "b4": dict(
        depths=[3, 8, 27, 3],
        hidden=[64, 128, 320, 512],
        heads=[1, 2, 5, 8],
        decoder=768,
    ),
    "b5": dict(
        depths=[3, 6, 40, 3],
        hidden=[64, 128, 320, 512],
        heads=[1, 2, 5, 8],
        decoder=768,
    ),
}
_SR = [8, 4, 2, 1]
_PATCH = [7, 3, 3, 3]
_STRIDE = [4, 2, 2, 2]


class SegFormer(eqx.Module):
    """SegFormer — MiT encoder + all-MLP decode head."""

    segformer: SegformerEncoder
    decode_head: SegformerDecodeHead
    num_classes: int = eqx.field(static=True)
    arch: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        arch: str = "b0",
        num_classes: int = 150,
        drop_path_rate: float = 0.1,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        key,
    ):
        if arch not in _VARIANTS:
            raise ValueError(f"Unknown arch '{arch}'. Choose one of {list(_VARIANTS)}.")
        cfg = _VARIANTS[arch]
        depths, hidden, heads, dec = (
            cfg["depths"],
            cfg["hidden"],
            cfg["heads"],
            cfg["decoder"],
        )

        k1, k2 = jr.split(key, 2)
        self.segformer = SegformerEncoder(
            in_ch=3,
            hidden_sizes=hidden,
            depths=depths,
            heads=heads,
            sr_ratios=_SR,
            patch_sizes=_PATCH,
            strides=_STRIDE,
            drop_path_rate=drop_path_rate,
            drop=drop,
            attn_drop=attn_drop,
            key=k1,
        )
        self.decode_head = SegformerDecodeHead(hidden, dec, num_classes, key=k2)

        self.num_classes = num_classes
        self.arch = arch

    def __call__(self, x: jnp.ndarray, key: jnp.ndarray, state):
        # x: [3, H, W]
        k_enc, k_dec = jr.split(key, 2)
        feats, state = self.segformer(x, key=k_enc, state=state)
        logits_1_4, state = self.decode_head(feats, key=k_dec, state=state)
        # upsample to input size
        logits = jax.image.resize(
            logits_1_4, (logits_1_4.shape[0], x.shape[1], x.shape[2]), method="linear"
        )
        return logits, state


# --------------------------- HF Weight Loading ---------------------------


def _rename_hf_key(k: str) -> str | None:
    """
    Map HF PyTorch keys -> our pytree paths.
    We mirrored HF's names, except:
      - 'attention.self.' (HF) -> 'attention.self_attn.' (ours)
    Skip non-parameter entries (e.g., BN running stats).
    """
    if not (k.endswith(".weight") or k.endswith(".bias")) and "relative" not in k:
        # keep linear/conv/bn/ln params only
        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            return None
        # also allow DWConv kernel/bias
        if ".dwconv." in k and (k.endswith(".weight") or k.endswith(".bias")):
            pass
        else:
            # non-param — skip
            return None

    k = k.replace("segformer.", "segformer.")
    k = k.replace("encoder.", "segformer.") if k.startswith("encoder.") else k
    k = k.replace("attention.self.", "attention.self_attn.")
    return k


def _copy_into(
    mod: eqx.Module, pt: Dict[str, jnp.ndarray], prefix: str = ""
) -> eqx.Module:
    """Recursively copy Linear/Conv/LayerNorm/BatchNorm params into the Equinox module."""
    for name, attr in vars(mod).items():
        full = f"{prefix}{name}"
        if isinstance(attr, eqx.nn.Conv2d):
            w, b = pt.get(f"{full}.weight"), pt.get(f"{full}.bias")
            if w is not None:
                mod = eqx.tree_at(
                    lambda m: getattr(m, name).weight, mod, jnp.asarray(w)
                )
            if b is not None:
                mod = eqx.tree_at(lambda m: getattr(m, name).bias, mod, jnp.asarray(b))
        elif isinstance(attr, eqx.nn.Linear):
            w, b = pt.get(f"{full}.weight"), pt.get(f"{full}.bias")
            if w is not None:
                mod = eqx.tree_at(
                    lambda m: getattr(m, name).weight, mod, jnp.asarray(w)
                )
            if b is not None:
                mod = eqx.tree_at(lambda m: getattr(m, name).bias, mod, jnp.asarray(b))
        elif isinstance(attr, eqx.nn.LayerNorm):
            w, b = pt.get(f"{full}.weight"), pt.get(f"{full}.bias")
            if w is not None and b is not None:
                mod = eqx.tree_at(
                    lambda m: (getattr(m, name).weight, getattr(m, name).bias),
                    mod,
                    (jnp.asarray(w), jnp.asarray(b)),
                )
        elif isinstance(attr, eqx.nn.BatchNorm):
            # only affine params
            w, b = pt.get(f"{full}.weight"), pt.get(f"{full}.bias")
            if w is not None:
                mod = eqx.tree_at(
                    lambda m: getattr(m, name).weight, mod, jnp.asarray(w)
                )
            if b is not None:
                mod = eqx.tree_at(lambda m: getattr(m, name).bias, mod, jnp.asarray(b))
        elif isinstance(attr, tuple):
            new = []
            for i, child in enumerate(attr):
                new.append(_copy_into(child, pt, prefix=f"{full}.{i}"))
            mod = eqx.tree_at(lambda m: getattr(m, name), mod, tuple(new))
        elif isinstance(attr, eqx.Module):
            mod = _copy_into(attr, pt, prefix=f"{full}.")
        else:
            # dropouts, activations, floats/ints
            pass
    return mod


def load_hf_segformer_encoder(model: SegFormer, npz_path: str) -> SegFormer:
    """Load MiT encoder weights saved from HF `SegformerModel` or `SegformerForSemanticSegmentation`."""
    import numpy as np

    raw = dict(np.load(npz_path))
    pt: Dict[str, jnp.ndarray] = {}
    for k, v in raw.items():
        nk = _rename_hf_key(k)
        if nk is None:
            continue
        # accept both "segformer.encoder.*" or "segformer.*" encoder keys
        if ".decode_head." in nk:
            continue
        pt[nk] = jnp.asarray(v)
    return _copy_into(model, pt, prefix="")  # keys include "segformer..."


def load_hf_segformer_full(
    model: SegFormer, npz_path: str, *, strict_head: bool = True
) -> SegFormer:
    """Load full SegFormer (encoder + decode head) from HF checkpoint .npz.

    If your `num_classes` differs from the checkpoint's classifier shape,
    set `strict_head=False` to skip loading `decode_head.classifier.*`.
    """
    import numpy as np

    raw = dict(np.load(npz_path))
    pt: Dict[str, jnp.ndarray] = {}
    for k, v in raw.items():
        nk = _rename_hf_key(k)
        if nk is None:
            continue
        pt[nk] = jnp.asarray(v)

    # Handle classifier mismatch
    cls_w = pt.get("decode_head.classifier.weight")
    cls_b = pt.get("decode_head.classifier.bias")
    if cls_w is not None:
        want_out, want_in = model.decode_head.classifier.weight.shape
        have_out, have_in = cls_w.shape
        if (want_out, want_in) != (have_out, have_in):
            if strict_head:
                raise ValueError(
                    f"classifier shape mismatch: want {(want_out, want_in)} vs have {(have_out, have_in)}. "
                    f"Set strict_head=False to load everything except the classifier."
                )
            pt.pop("decode_head.classifier.weight", None)
            pt.pop("decode_head.classifier.bias", None)

    return _copy_into(model, pt, prefix="")


# ------------------------------ Smoke test ------------------------------ #
if __name__ == "__main__":
    """
    Synthetic segmentation smoke-test for SegFormer (B0).
    Mirrors your other tests: CHW, single-sample models, BN(mode='batch').
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import augmax
    from augmax import InputType
    import optax
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jr

    from quantbayes.stochax import (
        train,
        predict,
        make_augmax_augment,
        make_dice_bce_loss,
    )

    rng = np.random.RandomState(0)
    N, C, H, W, OUT_CH = 10, 3, 128, 128, 1
    X_np = rng.rand(N, C, H, W).astype("float32")
    y_np = rng.randint(0, 2, size=(N, OUT_CH, H, W)).astype("float32")

    split = int(0.8 * N)
    X_train, X_val = X_np[:split], X_np[split:]
    y_train, y_val = y_np[:split], y_np[split:]

    transform = augmax.Chain(
        augmax.HorizontalFlip(),
        augmax.Rotate(angle_range=15),
        input_types=[InputType.IMAGE, InputType.MASK],
    )
    augment_fn = make_augmax_augment(transform)

    master_key = jr.PRNGKey(42)
    model_key, train_key = jr.split(master_key)

    model, state = eqx.nn.make_with_state(SegFormer)(
        arch="b0",
        num_classes=OUT_CH,
        key=model_key,
    )
    # Optional HF loading (encoder-only or full). Uncomment to use your saved .npz:
    # model = load_hf_segformer_encoder(model, "mit-b0_encoder.npz")
    # model = load_hf_segformer_full(model, "segformer-b0-finetuned-ade-512-512.npz", strict_head=False)

    lr_sched = optax.cosine_decay_schedule(1e-3, decay_steps=300)
    optimizer = optax.adamw(
        learning_rate=lr_sched,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=1e-4,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    best_model, best_state, tr_loss, va_loss = train(
        model=model,
        state=state,
        opt_state=opt_state,
        optimizer=optimizer,
        loss_fn=make_dice_bce_loss(),
        X_train=jnp.array(X_train),
        y_train=jnp.array(y_train),
        X_val=jnp.array(X_val),
        y_val=jnp.array(y_val),
        batch_size=32,
        num_epochs=8,
        patience=3,
        key=train_key,
        augment_fn=augment_fn,
        lambda_spec=0.0,
    )

    logits = predict(best_model, best_state, jnp.array(X_val), train_key)
    print("Predictions shape:", logits.shape)

    plt.figure()
    plt.plot(tr_loss, label="train")
    plt.plot(va_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("SegFormer-B0 smoke test")
    plt.show()
