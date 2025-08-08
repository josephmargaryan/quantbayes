"""
Equinox ConvNeXt (Tiny/Small/Base/Large) + torchvision weight loader.

- Single-sample forward with channel-first inputs [C, H, W]
- LayerNorm uses channels-last semantics with eps=1e-6 (no EMA state)
- __call__(self, x, key, state) -> (logits, state)
- Variants: convnext_tiny, convnext_small, convnext_base, convnext_large
- Implements CNBlock: DWConv (7x7, depthwise) -> LN -> PW Linear (4x) -> GELU -> PW Linear -> LayerScale -> DropPath + Residual
- Torchvision weights: save state_dict() to .npz and load here (features + classifier)

Weight Loading (torchvision → Equinox)
--------------------------------------
1) Save torchvision weights once (same pattern as your other models):
   ----------------------------------------------------------------
   # save_torchvision_convnext.py
   from pathlib import Path
   import numpy as np
   from torchvision.models import (
       convnext_tiny, convnext_small, convnext_base, convnext_large
   )

   CHECKPOINTS = {
       "convnext_tiny": convnext_tiny,
       "convnext_small": convnext_small,
       "convnext_base": convnext_base,
       "convnext_large": convnext_large,
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

2) Initialize and load into Equinox:
   ----------------------------------------------------------------
   import equinox as eqx, jax.random as jr
   from quantbayes.stochax.vision_classification.models.convnext import (
       ConvNeXt, load_imagenet_convnext_tiny  # or small/base/large helpers below
   )

   key = jr.PRNGKey(0)
   model, state = eqx.nn.make_with_state(ConvNeXt)(
       arch="convnext_tiny",  # "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
       num_classes=1000,
       key=key,
   )

   # If your num_classes != 1000, set strict_fc=False to skip the final FC (features still load).
   model = load_imagenet_convnext_tiny(model, "convnext_tiny_imagenet.npz", strict_fc=True)
   ----------------------------------------------------------------
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# --------------------------- utilities --------------------------- #
class DropPath(eqx.Module):
    """Stochastic Depth: drops the residual branch with prob=rate (row-wise)."""

    rate: float = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray, *, key: jnp.ndarray) -> jnp.ndarray:
        if self.rate <= 0.0:
            return x
        keep = 1.0 - self.rate
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


class LayerNorm2d(eqx.Module):
    """LayerNorm applied channel-wise on CHW input (channels-last semantics)."""

    weight: jnp.ndarray
    bias: jnp.ndarray
    eps: float = eqx.field(static=True)

    def __init__(self, channels: int, eps: float = 1e-6):
        self.weight = jnp.ones((channels,), dtype=jnp.float32)
        self.bias = jnp.zeros((channels,), dtype=jnp.float32)
        self.eps = float(eps)

    def __call__(self, x: jnp.ndarray, key=None, state=None):
        # x: [C, H, W] -> HWC
        x_hwc = jnp.moveaxis(x, 0, -1)
        mean = jnp.mean(x_hwc, axis=-1, keepdims=True)
        var = jnp.var(x_hwc, axis=-1, keepdims=True)
        x_norm = (x_hwc - mean) / jnp.sqrt(var + self.eps)
        x_norm = x_norm * self.weight + self.bias
        y = jnp.moveaxis(x_norm, -1, 0)
        return y, state


# --------------------------- building blocks --------------------------- #
class Stem(eqx.Module):
    """Patchify: Conv(4x4, stride 4, bias=True) -> LayerNorm2d."""

    conv: eqx.nn.Conv2d
    norm: LayerNorm2d

    def __init__(self, cout: int, *, key):
        (k1,) = jr.split(key, 1)
        self.conv = eqx.nn.Conv2d(3, cout, 4, stride=4, use_bias=True, key=k1)
        self.norm = LayerNorm2d(cout, eps=1e-6)

    def __call__(self, x, key, state):
        x = self.conv(x, key=key)
        x, state = self.norm(x, state=state)
        return x, state


class Downsample(eqx.Module):
    """Downsample between stages: LayerNorm2d -> Conv2d(2x2, stride 2, bias=True)."""

    norm: LayerNorm2d
    conv: eqx.nn.Conv2d

    def __init__(self, cin: int, cout: int, *, key):
        (k1,) = jr.split(key, 1)
        self.norm = LayerNorm2d(cin, eps=1e-6)
        self.conv = eqx.nn.Conv2d(cin, cout, 2, stride=2, use_bias=True, key=k1)

    def __call__(self, x, key, state):
        x, state = self.norm(x, state=state)
        x = self.conv(x, key=key)
        return x, state


class CNBlock(eqx.Module):
    """ConvNeXt block (torchvision-compatible):
    DWConv(7x7 groups=C) -> LN -> PW Linear (4*C) -> GELU -> PW Linear (C) -> LayerScale -> DropPath + Residual
    """

    dwconv: eqx.nn.Conv2d
    norm: LayerNorm2d
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    gamma: jnp.ndarray
    droppath: DropPath

    def __init__(self, dim: int, layer_scale: float, sd_prob: float, *, key):
        kdw, k1, k2 = jr.split(key, 3)
        self.dwconv = eqx.nn.Conv2d(
            dim, dim, 7, padding=3, groups=dim, use_bias=True, key=kdw
        )
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.linear1 = eqx.nn.Linear(dim, 4 * dim, key=k1)
        self.linear2 = eqx.nn.Linear(4 * dim, dim, key=k2)
        self.gamma = jnp.ones((dim, 1, 1), dtype=jnp.float32) * layer_scale
        self.droppath = DropPath(sd_prob)

    def __call__(self, x, key, state):
        kdw, kdp = jr.split(key, 2)
        out = self.dwconv(x, key=kdw)
        out, state = self.norm(out, state=state)
        # channels-last PW MLP via vmaps over H and W
        out_hwc = jnp.moveaxis(out, 0, -1)  # [H, W, C]
        out_hwc = jax.vmap(jax.vmap(self.linear1))(out_hwc)  # [H, W, 4C]
        out_hwc = jax.nn.gelu(out_hwc)
        out_hwc = jax.vmap(jax.vmap(self.linear2))(out_hwc)  # [H, W, C]
        out = jnp.moveaxis(out_hwc, -1, 0)  # [C, H, W]
        out = out * self.gamma  # layer scale
        out = self.droppath(out, key=kdp)
        out = out + x
        return out, state


# --------------------------- architecture configs --------------------------- #
# Mirrors torchvision CNBlockConfig lists for each variant.
_VARIANTS: Dict[str, Dict[str, Any]] = {
    # (input_channels, out_channels, num_layers) per stage
    "convnext_tiny": dict(
        stages=[(96, 192, 3), (192, 384, 3), (384, 768, 9), (768, None, 3)],
        sd_prob=0.1,
    ),
    "convnext_small": dict(
        stages=[(96, 192, 3), (192, 384, 3), (384, 768, 27), (768, None, 3)],
        sd_prob=0.4,
    ),
    "convnext_base": dict(
        stages=[(128, 256, 3), (256, 512, 3), (512, 1024, 27), (1024, None, 3)],
        sd_prob=0.5,
    ),
    "convnext_large": dict(
        stages=[(192, 384, 3), (384, 768, 3), (768, 1536, 27), (1536, None, 3)],
        sd_prob=0.5,
    ),
}
_LAYER_SCALE_DEFAULT = 1e-6


# --------------------------- ConvNeXt model --------------------------- #
class ConvNeXt(eqx.Module):
    """ConvNeXt (Tiny/Small/Base/Large) in Equinox; torchvision-compatible for weight loading."""

    features: Tuple[
        Any, ...
    ]  # (Stem, Stage0(tuple[CNBlock]), Downsample, Stage1, Downsample, Stage2, Downsample, Stage3)
    avgpool: AdaptiveAvgPool2d
    classifier_norm: LayerNorm2d
    fc: eqx.nn.Linear

    arch: str = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)

    def __init__(self, *, arch: str = "convnext_tiny", num_classes: int = 1000, key):
        if arch not in _VARIANTS:
            raise ValueError(f"Unknown arch '{arch}'.")
        cfg = _VARIANTS[arch]
        stages_cfg: List[Tuple[int, int | None, int]] = cfg["stages"]
        sd_prob: float = cfg["sd_prob"]

        # Keys (oversubdivide and consume)
        big_keys = list(jr.split(key, 4096))
        k_it = iter(big_keys)

        features: List[Any] = []
        # Stem
        first_out = stages_cfg[0][0]
        features.append(Stem(first_out, key=next(k_it)))

        # Count total blocks for sd schedule
        total_blocks = sum(n for _, _, n in stages_cfg)
        block_id = 0

        # Stages + Downsamples
        in_c = first_out
        for cin, cout, num_layers in stages_cfg:
            assert cin == in_c, f"Config mismatch: expected cin={in_c}, got {cin}"

            stage_blocks: List[CNBlock] = []
            for _ in range(num_layers):
                # linear sd ramp
                sd = (
                    0.0
                    if total_blocks <= 1
                    else sd_prob * (block_id / (total_blocks - 1.0))
                )
                stage_blocks.append(
                    CNBlock(cin, _LAYER_SCALE_DEFAULT, sd, key=next(k_it))
                )
                block_id += 1
            features.append(tuple(stage_blocks))
            # Downsample if needed
            if cout is not None:
                features.append(Downsample(cin, cout, key=next(k_it)))
                in_c = cout

        last_c = in_c
        self.features = tuple(features)
        self.avgpool = AdaptiveAvgPool2d(1, 1)
        self.classifier_norm = LayerNorm2d(last_c, eps=1e-6)
        self.fc = eqx.nn.Linear(last_c, num_classes, key=next(k_it))

        self.arch = arch
        self.num_classes = num_classes

    def _check_input(self, x: jnp.ndarray):
        if x.ndim != 3:
            raise ValueError(
                f"ConvNeXt expects single sample [C,H,W]; got {tuple(x.shape)}."
            )
        if x.shape[0] != 3:
            raise ValueError(f"Expected 3 input channels; got {x.shape[0]}.")

    def __call__(self, x, key, state):
        self._check_input(x)
        k_run = key

        def split():
            nonlocal k_run
            k1, k_run = jr.split(k_run)
            return k1

        for feat in self.features:
            if isinstance(feat, Stem):
                x, state = feat(x, key=split(), state=state)
            elif isinstance(feat, tuple):
                for block in feat:
                    x, state = block(x, key=split(), state=state)
            elif isinstance(feat, Downsample):
                x, state = feat(x, key=split(), state=state)
            else:
                raise RuntimeError("Unknown feature element.")

        x = self.avgpool(x).reshape(-1)  # [C]
        x, state = self.classifier_norm(
            x.reshape(-1, 1, 1), state=state
        )  # reuse LN2d; treat as CHW
        x = x.reshape(-1)
        logits = self.fc(x)
        return logits, state


# --------------------------- Weight Loading --------------------------- #
def _rename_pt_key(k: str) -> str | None:
    """
    Map torchvision ConvNeXt state_dict keys to our module tree.

    Patterns we handle:
      Stem:
        features.0.0.(weight|bias)           -> features.0.conv.(weight|bias)
        features.0.1.(weight|bias)           -> features.0.norm.(weight|bias)

      Stages (CNBlock inside features.<stage_idx>.<block_idx>.block):
        ...block.0.(weight|bias)             -> ...dwconv.(weight|bias)
        ...block.2.(weight|bias)             -> ...norm.(weight|bias)
        ...block.3.(weight|bias)             -> ...linear1.(weight|bias)
        ...block.5.(weight|bias)             -> ...linear2.(weight|bias)
        ...layer_scale                       -> ...gamma

      Downsample sequences:
        features.<k>.0.(weight|bias)         -> features.<k>.norm.(weight|bias)
        features.<k>.1.(weight|bias)         -> features.<k>.conv.(weight|bias)

      Classifier:
        classifier.0.(weight|bias)           -> classifier_norm.(weight|bias)
        classifier.2.(weight|bias)           -> fc.(weight|bias)
    """
    # ConvNeXt uses LayerNorm (no running stats), so nothing to drop apart from non-param ops.
    # Keep only param tensors.
    if not any(k.endswith(suffix) for suffix in (".weight", ".bias", "layer_scale")):
        return None

    # Stem
    if k.startswith("features.0.0."):
        return k.replace("features.0.0.", "features.0.conv.")
    if k.startswith("features.0.1."):
        return k.replace("features.0.1.", "features.0.norm.")

    # Stage blocks
    if ".block.0." in k:
        return k.replace(".block.0.", ".dwconv.")
    if ".block.2." in k:
        return k.replace(".block.2.", ".norm.")
    if ".block.3." in k:
        return k.replace(".block.3.", ".linear1.")
    if ".block.5." in k:
        return k.replace(".block.5.", ".linear2.")
    if ".layer_scale" in k:
        return k.replace(".layer_scale", ".gamma")

    # Downsample
    # e.g., features.2.0.weight -> features.2.norm.weight
    parts = k.split(".")
    if (
        len(parts) >= 4
        and parts[0] == "features"
        and parts[2] in {"0", "1"}
        and "block" not in k
    ):
        if parts[2] == "0":
            return k.replace(".0.", ".norm.")
        else:
            return k.replace(".1.", ".conv.")

    # Classifier
    if k.startswith("classifier.0."):
        return k.replace("classifier.0.", "classifier_norm.")
    if k.startswith("classifier.2."):
        return k.replace("classifier.2.", "fc.")

    return None


def _copy_into_tree(obj, pt: Dict[str, jnp.ndarray], prefix: str = ""):
    """Recursively copy Conv2d/Linear/LayerNorm2d params (and gamma arrays) into an Equinox pytree."""
    if isinstance(obj, eqx.Module):
        for name, attr in vars(obj).items():
            full = f"{prefix}{name}"

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

            if isinstance(attr, LayerNorm2d):
                w_key, b_key = f"{full}.weight", f"{full}.bias"
                w_val = pt.get(w_key, getattr(attr, "weight"))
                b_val = pt.get(b_key, getattr(attr, "bias"))
                obj = eqx.tree_at(
                    lambda m: (getattr(m, name).weight, getattr(m, name).bias),
                    obj,
                    (w_val, b_val),
                )
                continue

            if isinstance(attr, tuple):
                new_tuple = []
                for i, child in enumerate(attr):
                    child_full = f"{full}.{i}"
                    new_tuple.append(_copy_into_tree(child, pt, prefix=child_full))
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, tuple(new_tuple))
                continue

            # gamma (layer scale) stored as jnp.ndarray
            if isinstance(attr, jnp.ndarray) and f"{full}" in pt:
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, pt[f"{full}"])
                continue

            # Stateless (DropPath), pools, etc. -> skip
        return obj

    if isinstance(obj, tuple):
        return tuple(_copy_into_tree(x, pt, prefix=prefix) for x in obj)
    return obj


def load_torchvision_convnext(
    model: ConvNeXt, npz_path: str, *, strict_fc: bool = True
) -> ConvNeXt:
    """Load a torchvision ConvNeXt .npz (from state_dict()) into this model."""
    import numpy as np

    raw = dict(np.load(npz_path))
    pt: Dict[str, jnp.ndarray] = {}
    for k, v in raw.items():
        nk = _rename_pt_key(k)
        if nk is None:
            continue
        pt[nk] = jnp.asarray(v)

    # Handle final FC shape mismatch (custom num_classes)
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


# Convenience helpers per arch
def load_imagenet_convnext_tiny(
    model: ConvNeXt, npz="convnext_tiny_imagenet.npz", strict_fc: bool = True
) -> ConvNeXt:
    return load_torchvision_convnext(model, npz, strict_fc=strict_fc)


def load_imagenet_convnext_small(
    model: ConvNeXt, npz="convnext_small_imagenet.npz", strict_fc: bool = True
) -> ConvNeXt:
    return load_torchvision_convnext(model, npz, strict_fc=strict_fc)


def load_imagenet_convnext_base(
    model: ConvNeXt, npz="convnext_base_imagenet.npz", strict_fc: bool = True
) -> ConvNeXt:
    return load_torchvision_convnext(model, npz, strict_fc=strict_fc)


def load_imagenet_convnext_large(
    model: ConvNeXt, npz="convnext_large_imagenet.npz", strict_fc: bool = True
) -> ConvNeXt:
    return load_torchvision_convnext(model, npz, strict_fc=strict_fc)


# --------------------------- Smoke test --------------------------- #
if __name__ == "__main__":
    """
    Synthetic classification smoke test for ConvNeXt.
    Uses 224×224 RGB; replace with real data for experiments.
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

    model, state = eqx.nn.make_with_state(ConvNeXt)(
        arch="convnext_tiny",
        num_classes=NUM_CLASSES,
        key=model_key,
    )
    # Optional pretrained load (skips fc when shapes mismatch)
    # model = load_imagenet_convnext_tiny(model, "convnext_tiny_imagenet.npz", strict_fc=False)

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
    plt.title("ConvNeXt-Tiny smoke test")
    plt.show()
