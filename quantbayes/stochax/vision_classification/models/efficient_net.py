"""
Equinox EfficientNet (B0–B7) with torchvision weight loader.

- Single-sample forward with channel-first inputs [C, H, W]
- BatchNorm uses mode="batch" (no EMA state)
- __call__(self, x, key, state) -> (logits, state)
- Variants: efficientnet_b0 ... efficientnet_b7
- Implements MBConv + SE + Stochastic Depth (DropPath)
- Torchvision weights: save state_dict to .npz and load here (features + classifier)

Weight Loading (torchvision → Equinox):
---------------------------------------
1) Save torchvision weights once (exactly like your other models):
   ----------------------------------------------------------------
   # save_torchvision_efficientnets.py
   from pathlib import Path
   import numpy as np
   from torchvision.models import (
       efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
       efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
   )

   CHECKPOINTS = {
       "efficientnet_b0": efficientnet_b0,
       "efficientnet_b1": efficientnet_b1,
       "efficientnet_b2": efficientnet_b2,
       "efficientnet_b3": efficientnet_b3,
       "efficientnet_b4": efficientnet_b4,
       "efficientnet_b5": efficientnet_b5,
       "efficientnet_b6": efficientnet_b6,
       "efficientnet_b7": efficientnet_b7,
   }

   def main():
       for name, builder in CHECKPOINTS.items():
           print(f"⇢ downloading {name} …")
           model = builder(weights="IMAGENET1K_V1")
           ckpt_path = Path(f"{name}_imagenet.npz")
           print(f"↳ saving → {ckpt_path}")
           np.savez(ckpt_path, **{k: v.cpu().numpy() for k, v in model.state_dict().items()})
           print(f"✓ done {ckpt_path}\\n")

   if __name__ == "__main__":
       main()
   ----------------------------------------------------------------

2) Initialize EfficientNet and load weights:
   ----------------------------------------------------------------
   import equinox as eqx, jax.random as jr
   from quantbayes.stochax.vision_classification.models.efficientnet import (
       EfficientNet, load_imagenet_efficientnet_b0,   # ... b1 ... b7 helpers
   )

   key = jr.PRNGKey(0)
   model, state = eqx.nn.make_with_state(EfficientNet)(
       arch="efficientnet_b0",      # b0 ... b7
       num_classes=1000,
       key=key,
   )

   # If your num_classes != 1000, the loader will keep features and skip final FC when shapes mismatch.
   model = load_imagenet_efficientnet_b0(model, "efficientnet_b0_imagenet.npz", strict_fc=True)
   ----------------------------------------------------------------

Author: Joseph Margaryan (library conventions), JAX/Equinox port with robust loader.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# --------------------------- Small utilities --------------------------- #
class ReLU(eqx.Module):
    def __call__(self, x, key=None, state=None):
        return jax.nn.relu(x), state


class SiLU(eqx.Module):
    def __call__(self, x, key=None, state=None):
        return jax.nn.silu(x), state


class DropPath(eqx.Module):
    """Stochastic Depth. Drops residual branch with prob=rate."""

    rate: float = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray, *, key: jnp.ndarray) -> jnp.ndarray:
        if self.rate <= 0.0:
            return x
        keep = 1.0 - self.rate
        # Sample a single Bernoulli mask for the whole sample (single-sample model).
        mask = jr.bernoulli(key, p=keep)
        return x * (mask.astype(x.dtype) / keep)


class AdaptiveAvgPool2d(eqx.Module):
    """Exact AdaptiveAvgPool2d to (out_h, out_w) for single-sample CHW tensors."""

    out_h: int = eqx.field(static=True)
    out_w: int = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray):
        C, H, W = x.shape
        # Reduce H
        h_slices = []
        for i in range(self.out_h):
            h0 = int(math.floor(i * H / self.out_h))
            h1 = int(math.ceil((i + 1) * H / self.out_h))
            h1 = max(h1, h0 + 1)
            h_slices.append(jnp.mean(x[:, h0:h1, :], axis=1))  # [C, W]
        tmp = jnp.stack(h_slices, axis=1)  # [C, out_h, W]
        # Reduce W
        w_slices = []
        for j in range(self.out_w):
            w0 = int(math.floor(j * W / self.out_w))
            w1 = int(math.ceil((j + 1) * W / self.out_w))
            w1 = max(w1, w0 + 1)
            w_slices.append(jnp.mean(tmp[:, :, w0:w1], axis=2))  # [C, out_h]
        out = jnp.stack(w_slices, axis=2)  # [C, out_h, out_w]
        return out


# ------------------------- Conv/BN/Act blocks ------------------------- #
class ConvBNActTV(eqx.Module):
    """Conv -> BN -> (SiLU or ReLU). Matches torchvision ConvNormActivation param shape/naming."""

    c0: eqx.nn.Conv2d
    c1: eqx.nn.BatchNorm
    use_silu: bool = eqx.field(static=True)

    def __init__(
        self,
        cin: int,
        cout: int,
        k: int,
        stride: int,
        padding: int,
        *,
        use_silu: bool,
        key,
    ):
        (k1,) = jr.split(key, 1)
        self.c0 = eqx.nn.Conv2d(
            cin, cout, k, stride=stride, padding=padding, use_bias=False, key=k1
        )
        self.c1 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.use_silu = use_silu

    def __call__(self, x, key, state):
        x = self.c0(x, key=key)
        x, state = self.c1(x, state)
        if self.use_silu:
            x = jax.nn.silu(x)
        else:
            x = jax.nn.relu(x)
        return x, state


class DWConvBNActTV(eqx.Module):
    """Depthwise Conv -> BN -> SiLU"""

    c0: eqx.nn.Conv2d
    c1: eqx.nn.BatchNorm

    def __init__(self, ch: int, k: int, stride: int, padding: int, *, key):
        (k1,) = jr.split(key, 1)
        self.c0 = eqx.nn.Conv2d(
            ch, ch, k, stride=stride, padding=padding, groups=ch, use_bias=False, key=k1
        )
        self.c1 = eqx.nn.BatchNorm(ch, axis_name="batch", mode="batch")

    def __call__(self, x, key, state):
        x = self.c0(x, key=key)
        x, state = self.c1(x, state)
        x = jax.nn.silu(x)
        return x, state


class ConvBNTV(eqx.Module):
    """Conv -> BN (no activation)"""

    c0: eqx.nn.Conv2d
    c1: eqx.nn.BatchNorm

    def __init__(self, cin: int, cout: int, k: int, stride: int, padding: int, *, key):
        (k1,) = jr.split(key, 1)
        self.c0 = eqx.nn.Conv2d(
            cin, cout, k, stride=stride, padding=padding, use_bias=False, key=k1
        )
        self.c1 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")

    def __call__(self, x, key, state):
        x = self.c0(x, key=key)
        x, state = self.c1(x, state)
        return x, state


class SqueezeExcite(eqx.Module):
    """Squeeze-and-Excitation with two 1x1 convs (reduce/expand)."""

    c0: eqx.nn.Conv2d  # reduce
    c1: eqx.nn.Conv2d  # expand

    def __init__(self, ch: int, se_ratio: float, *, key):
        reduced = max(1, int(ch * se_ratio))
        k1, k2 = jr.split(key, 2)
        self.c0 = eqx.nn.Conv2d(ch, reduced, 1, use_bias=True, key=k1)
        self.c1 = eqx.nn.Conv2d(reduced, ch, 1, use_bias=True, key=k2)

    def __call__(self, x, key=None, state=None):
        # Global average pool to [C,1,1]
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        s = self.c0(s)  # [reduced,1,1]
        s = jax.nn.silu(s)
        s = self.c1(s)  # [C,1,1]
        s = jax.nn.sigmoid(s)
        return x * s, state


# ------------------------------ MBConv ------------------------------ #
class MBConvTV(eqx.Module):
    """Torchvision-compatible MBConv (with optional expansion), SE, and projection."""

    block: Tuple[eqx.Module, ...]  # mirrors torchvision: sequence of sub-layers
    use_residual: bool = eqx.field(static=True)
    droppath: DropPath

    def __init__(
        self,
        cin: int,
        cout: int,
        *,
        expand_ratio: int,
        kernel: int,
        stride: int,
        se_ratio: float,
        drop_rate: float,
        key,
    ):
        ks = list(jr.split(key, 5))
        use_expand = expand_ratio != 1
        exp_ch = cin * expand_ratio

        mods: List[eqx.Module] = []
        idx = 0

        if use_expand:
            mods.append(
                ConvBNActTV(
                    cin, exp_ch, k=1, stride=1, padding=0, use_silu=True, key=ks[idx]
                )
            )
            idx += 1
            mods.append(
                DWConvBNActTV(
                    exp_ch, k=kernel, stride=stride, padding=kernel // 2, key=ks[idx]
                )
            )
            idx += 1
        else:
            # No expand; depthwise runs on input channels
            mods.append(
                DWConvBNActTV(
                    cin, k=kernel, stride=stride, padding=kernel // 2, key=ks[idx]
                )
            )
            idx += 1

        # SE on the depthwise's channel-dim = exp_ch if expanded else cin
        se_ch = exp_ch if use_expand else cin
        mods.append(SqueezeExcite(se_ch, se_ratio, key=ks[idx]))
        idx += 1

        # Project to cout
        mods.append(ConvBNTV(se_ch, cout, k=1, stride=1, padding=0, key=ks[idx]))
        idx += 1

        self.block = tuple(mods)
        self.use_residual = (stride == 1) and (cin == cout)
        self.droppath = DropPath(drop_rate)

    def __call__(self, x, key, state):
        # Split keys for convs and droppath
        keys = jr.split(key, len(self.block) + 1)
        out = x
        for i, m in enumerate(self.block):
            if isinstance(m, (ConvBNActTV, DWConvBNActTV, ConvBNTV)):
                out, state = m(out, key=keys[i], state=state)
            elif isinstance(m, SqueezeExcite):
                out, state = m(out, state=state)
            else:
                out = m(out)  # shouldn't happen
        if self.use_residual:
            out = x + self.droppath(out, key=keys[-1])
        return out, state


# ---------------------------- Architecture ---------------------------- #
_BASE_BLOCKS = [
    # (expand_ratio, out_c, num_blocks, kernel, stride, se_ratio)
    (1, 16, 1, 3, 1, 0.25),
    (6, 24, 2, 3, 2, 0.25),
    (6, 40, 2, 5, 2, 0.25),
    (6, 80, 3, 3, 2, 0.25),
    (6, 112, 3, 5, 1, 0.25),
    (6, 192, 4, 5, 2, 0.25),
    (6, 320, 1, 3, 1, 0.25),
]

_VARIANTS: Dict[str, Dict[str, Any]] = {
    "efficientnet_b0": dict(
        width=1.0, depth=1.0, dropout=0.2, drop_path=0.2, stem=32, head=1280
    ),
    "efficientnet_b1": dict(
        width=1.0, depth=1.1, dropout=0.2, drop_path=0.2, stem=32, head=1280
    ),
    "efficientnet_b2": dict(
        width=1.1, depth=1.2, dropout=0.3, drop_path=0.3, stem=32, head=1280
    ),
    "efficientnet_b3": dict(
        width=1.2, depth=1.4, dropout=0.3, drop_path=0.3, stem=40, head=1280
    ),
    "efficientnet_b4": dict(
        width=1.4, depth=1.8, dropout=0.4, drop_path=0.4, stem=48, head=1280
    ),
    "efficientnet_b5": dict(
        width=1.6, depth=2.2, dropout=0.4, drop_path=0.4, stem=48, head=1280
    ),
    "efficientnet_b6": dict(
        width=1.8, depth=2.6, dropout=0.5, drop_path=0.5, stem=56, head=1280
    ),
    "efficientnet_b7": dict(
        width=2.0, depth=3.1, dropout=0.5, drop_path=0.5, stem=64, head=1280
    ),
}


def _round_filters(ch: int, width_mult: float, divisor: int = 8) -> int:
    ch *= width_mult
    new_ch = max(divisor, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return int(new_ch)


def _round_repeats(r: int, depth_mult: float) -> int:
    return int(math.ceil(depth_mult * r))


# ------------------------------ EfficientNet ------------------------------ #
class EfficientNet(eqx.Module):
    """EfficientNet-Bx (B0–B7) implemented in Equinox; torchvision-compatible features layout."""

    features: Tuple[
        Any, ...
    ]  # stem: ConvBNActTV, stages: tuple[MBConvTV,...], head: ConvBNActTV
    avgpool: AdaptiveAvgPool2d
    dr: eqx.nn.Dropout
    fc: eqx.nn.Linear

    arch: str = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)

    def __init__(self, *, arch: str = "efficientnet_b0", num_classes: int = 1000, key):
        if arch not in _VARIANTS:
            raise ValueError(f"Unknown arch '{arch}'.")
        cfg = _VARIANTS[arch]
        width, depth = cfg["width"], cfg["depth"]
        stem_c = _round_filters(cfg["stem"], width)
        head_c = _round_filters(cfg["head"], width)
        dropout = cfg["dropout"]
        drop_path_rate = cfg["drop_path"]

        # Keys
        # We need: 1 for stem, per MBConv (many), 1 for head, 1 for fc
        # We'll oversubdivide: split a generous number and consume as needed.
        big_keys = list(jr.split(key, 4096))
        k_iter = iter(big_keys)

        # Stem
        stem = ConvBNActTV(
            3, stem_c, k=3, stride=2, padding=1, use_silu=True, key=next(k_iter)
        )

        # Stages
        features: List[Any] = [stem]
        total_blocks = sum(
            _round_repeats(nb, depth) for _, _, nb, _, _, _ in _BASE_BLOCKS
        )
        block_id = 0

        in_c = stem_c
        for t, c, n, ksz, stride, se in _BASE_BLOCKS:
            out_c = _round_filters(c, width)
            reps = _round_repeats(n, depth)

            blocks: List[MBConvTV] = []
            for i in range(reps):
                s = stride if i == 0 else 1
                drop = drop_path_rate * (block_id / max(1, total_blocks - 1))
                blocks.append(
                    MBConvTV(
                        in_c,
                        out_c,
                        expand_ratio=t,
                        kernel=ksz,
                        stride=s,
                        se_ratio=se,
                        drop_rate=drop,
                        key=next(k_iter),
                    )
                )
                in_c = out_c
                block_id += 1
            features.append(tuple(blocks))

        # Head
        head = ConvBNActTV(
            in_c, head_c, k=1, stride=1, padding=0, use_silu=True, key=next(k_iter)
        )
        features.append(head)

        self.features = tuple(features)
        self.avgpool = AdaptiveAvgPool2d(1, 1)
        self.dr = eqx.nn.Dropout(dropout)
        self.fc = eqx.nn.Linear(head_c, num_classes, key=next(k_iter))

        self.arch = arch
        self.num_classes = num_classes

    def _check_input(self, x: jnp.ndarray):
        if x.ndim != 3:
            raise ValueError(
                f"EfficientNet expects single sample [C,H,W]; got {tuple(x.shape)}."
            )
        if x.shape[0] != 3:
            raise ValueError(f"Expected 3 input channels; got {x.shape[0]}.")

    def __call__(self, x, key, state):
        self._check_input(x)

        # Helper to walk features, which is [stem, (stage0 ...), (stage1 ...), ..., head]
        k_run = key

        def split():
            nonlocal k_run
            k1, k_run = jr.split(k_run)
            return k1

        for i, feat in enumerate(self.features):
            if isinstance(feat, ConvBNActTV):
                x, state = feat(x, key=split(), state=state)
            elif isinstance(feat, tuple):
                for block in feat:
                    x, state = block(x, key=split(), state=state)
            else:
                raise RuntimeError("Unknown feature element type.")

        # Global avg pool
        x = self.avgpool(x).reshape(-1)  # [C]
        # Classifier
        x = self.dr(x, key=split())
        logits = self.fc(x)
        return logits, state


# ---------------------------- Weight Loading ---------------------------- #
def _rename_pt_key(k: str) -> str | None:
    """
    Map torchvision EfficientNet state_dict keys to our module tree.

    We preserve the 'features' hierarchical indices (stem, stages, head).
    We drop running stats and num_batches_tracked.
    We map numeric submodules to our 'cN' names, and map classifier.1 -> fc.
    """
    if any(s in k for s in ("running_mean", "running_var", "num_batches_tracked")):
        return None

    if k.startswith("classifier.1."):
        return k.replace("classifier.1.", "fc.")
    if k.startswith("classifier.0."):
        return None  # dropout has no params

    # Numeric → cN inside any module chain (Conv/BN inside ConvNormActivation, DW, SE, project)
    # e.g., features.0.0.weight -> features.0.c0.weight
    k = k.replace(".0.", ".c0.")
    k = k.replace(".1.", ".c1.")
    k = k.replace(".2.", ".c2.")
    k = k.replace(".3.", ".c3.")
    k = k.replace(".4.", ".c4.")

    return k


def _copy_into_tree(obj, pt: Dict[str, jnp.ndarray], prefix: str = ""):
    """Recursively copy Conv/BN/Linear (and nested tuples) into an Equinox pytree."""
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

            # BatchNorm (affine only)
            if isinstance(attr, eqx.nn.BatchNorm):
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

            # Nested tuples (stages and MBConv sequences)
            if isinstance(attr, tuple):
                new_tuple = []
                for i, child in enumerate(attr):
                    child_full = f"{full}.{i}"
                    new_tuple.append(_copy_into_tree(child, pt, prefix=child_full))
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, tuple(new_tuple))
                continue

            # Stateless/others: DropPath, Dropout, activations, pools -> skip
        return obj

    if isinstance(obj, tuple):
        return tuple(_copy_into_tree(x, pt, prefix=prefix) for x in obj)
    return obj


def load_torchvision_efficientnet(
    model: EfficientNet, npz_path: str, *, strict_fc: bool = True
) -> EfficientNet:
    """Load a torchvision EfficientNet .npz (from state_dict()) into this model."""
    import numpy as np

    raw = dict(np.load(npz_path))
    pt: Dict[str, jnp.ndarray] = {}
    for k, v in raw.items():
        nk = _rename_pt_key(k)
        if nk is None:
            continue
        pt[nk] = jnp.asarray(v)

    # Handle final FC shape mismatch (e.g., custom num_classes)
    if "fc.weight" in pt and "fc.bias" in pt:
        want_out, want_in = model.fc.weight.shape
        have_out, have_in = pt["fc.weight"].shape
        if (want_out != have_out) or (want_in != have_in):
            if strict_fc:
                raise ValueError(
                    f"FC shape mismatch: want {(want_out, want_in)} vs have {(have_out, have_in)}. "
                    f"Set strict_fc=False to skip loading final FC."
                )
            pt.pop("fc.weight", None)
            pt.pop("fc.bias", None)

    return _copy_into_tree(model, pt, prefix="")


# Convenience per-arch loaders
def load_imagenet_efficientnet_b0(
    model: EfficientNet, npz="efficientnet_b0_imagenet.npz", strict_fc: bool = True
) -> EfficientNet:
    return load_torchvision_efficientnet(model, npz, strict_fc=strict_fc)


def load_imagenet_efficientnet_b1(
    model: EfficientNet, npz="efficientnet_b1_imagenet.npz", strict_fc: bool = True
) -> EfficientNet:
    return load_torchvision_efficientnet(model, npz, strict_fc=strict_fc)


def load_imagenet_efficientnet_b2(
    model: EfficientNet, npz="efficientnet_b2_imagenet.npz", strict_fc: bool = True
) -> EfficientNet:
    return load_torchvision_efficientnet(model, npz, strict_fc=strict_fc)


def load_imagenet_efficientnet_b3(
    model: EfficientNet, npz="efficientnet_b3_imagenet.npz", strict_fc: bool = True
) -> EfficientNet:
    return load_torchvision_efficientnet(model, npz, strict_fc=strict_fc)


def load_imagenet_efficientnet_b4(
    model: EfficientNet, npz="efficientnet_b4_imagenet.npz", strict_fc: bool = True
) -> EfficientNet:
    return load_torchvision_efficientnet(model, npz, strict_fc=strict_fc)


def load_imagenet_efficientnet_b5(
    model: EfficientNet, npz="efficientnet_b5_imagenet.npz", strict_fc: bool = True
) -> EfficientNet:
    return load_torchvision_efficientnet(model, npz, strict_fc=strict_fc)


def load_imagenet_efficientnet_b6(
    model: EfficientNet, npz="efficientnet_b6_imagenet.npz", strict_fc: bool = True
) -> EfficientNet:
    return load_torchvision_efficientnet(model, npz, strict_fc=strict_fc)


def load_imagenet_efficientnet_b7(
    model: EfficientNet, npz="efficientnet_b7_imagenet.npz", strict_fc: bool = True
) -> EfficientNet:
    return load_torchvision_efficientnet(model, npz, strict_fc=strict_fc)


# ------------------------------ Smoke test ------------------------------ #
if __name__ == "__main__":
    """
    Synthetic classification smoke test for EfficientNet (B0).
    Replace with a real dataset in practice.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import augmax
    from augmax import InputType
    import optax
    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jr

    # Your training utilities
    from quantbayes.stochax import (
        train,
        predict,
        make_augmax_augment,
        multiclass_loss,
    )

    rng = np.random.RandomState(0)
    N, C, H, W, NUM_CLASSES = 512, 3, 224, 224, 10
    X_np = rng.rand(N, C, H, W).astype("float32")
    y_np = rng.randint(0, NUM_CLASSES, size=(N,)).astype("int32")

    split = int(0.8 * N)
    X_train, X_val = X_np[:split], X_np[split:]
    y_train, y_val = y_np[:split], y_np[split:]

    transform = augmax.Chain(
        augmax.HorizontalFlip(),
        augmax.Rotate(angle_range=10),
        input_types=[InputType.IMAGE, InputType.METADATA],
    )
    augment_fn = make_augmax_augment(transform)

    master_key = jr.PRNGKey(42)
    model_key, train_key = jr.split(master_key)

    model, state = eqx.nn.make_with_state(EfficientNet)(
        arch="efficientnet_b0",
        num_classes=NUM_CLASSES,
        key=model_key,
    )
    # Optional: load pretrained features (skips fc if shapes mismatch)
    # model = load_imagenet_efficientnet_b0(model, "efficientnet_b0_imagenet.npz", strict_fc=False)

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
        loss_fn=multiclass_loss,
        X_train=jnp.array(X_train),
        y_train=jnp.array(y_train),
        X_val=jnp.array(X_val),
        y_val=jnp.array(y_val),
        batch_size=32,
        num_epochs=6,
        patience=2,
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
    plt.title("EfficientNet-B0 smoke test")
    plt.show()
