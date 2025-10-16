"""
Equinox VGG (VGG11/13/16/19) with optional BatchNorm and torchvision weight loader.

- Single-sample forward with channel-first inputs [C, H, W]
- BatchNorm uses mode="batch" (no EMA state carried)
- __call__(self, x, key, state) -> (logits, state)
- Variants: vgg11, vgg13, vgg16, vgg19, and *_bn variants
- Classifier matches torchvision: [512*7*7 -> 4096 -> 4096 -> num_classes] with Dropout

Weight Loading (torchvision → Equinox):
---------------------------------------
1) Save torchvision weights once:
   ----------------------------------------------------------------
   # save_torchvision_vggs.py
   from pathlib import Path
   import numpy as np
   from torchvision.models import (
       vgg11, vgg13, vgg16, vgg19,
       vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
   )

   CHECKPOINTS = {
       "vgg11": vgg11,
       "vgg13": vgg13,
       "vgg16": vgg16,
       "vgg19": vgg19,
       "vgg11_bn": vgg11_bn,
       "vgg13_bn": vgg13_bn,
       "vgg16_bn": vgg16_bn,
       "vgg19_bn": vgg19_bn,
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

2) Initialize VGG and load weights:
   ----------------------------------------------------------------
   import equinox as eqx, jax.random as jr
   from quantbayes.stochax.vision_classification.models.vgg import (
       VGG, load_imagenet_vgg16_bn,  # or any loader below
   )

   key = jr.PRNGKey(0)
   model, state = eqx.nn.make_with_state(VGG)(
       arch="vgg16_bn",      # one of: vgg11(_bn), vgg13(_bn), vgg16(_bn), vgg19(_bn)
       num_classes=1000,
       key=key,
   )

   # Load torchvision weights (.npz). If your num_classes != 1000,
   # set strict_fc=False to skip loading the final FC layer.
   model = load_imagenet_vgg16_bn(model, "vgg16_bn_imagenet.npz", strict_fc=True)
   ----------------------------------------------------------------

Author: Joseph Margaryan (library conventions), VGG port and loader adapted to Equinox.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# --------------------------- Small utility mods --------------------------- #
class ReLU(eqx.Module):
    """Stateless ReLU that preserves `(x, state)` calling convention."""

    def __call__(self, x, key=None, state=None):
        return jax.nn.relu(x), state


class AdaptiveAvgPool2d(eqx.Module):
    """Exact AdaptiveAvgPool2d to (out_h, out_w) for single-sample CHW tensors."""

    out_h: int = eqx.field(static=True)
    out_w: int = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray):
        # x: [C, H, W] -> [C, out_h, out_w]
        C, H, W = x.shape
        # Reduce H
        h_slices = []
        for i in range(self.out_h):
            h0 = int(jnp.floor(i * H / self.out_h))
            h1 = int(jnp.ceil((i + 1) * H / self.out_h))
            h1 = max(h1, h0 + 1)  # ensure non-empty
            h_slices.append(jnp.mean(x[:, h0:h1, :], axis=1))  # [C, W]
        tmp = jnp.stack(h_slices, axis=1)  # [C, out_h, W]
        # Reduce W
        w_slices = []
        for j in range(self.out_w):
            w0 = int(jnp.floor(j * W / self.out_w))
            w1 = int(jnp.ceil((j + 1) * W / self.out_w))
            w1 = max(w1, w0 + 1)
            w_slices.append(jnp.mean(tmp[:, :, w0:w1], axis=2))  # [C, out_h]
        out = jnp.stack(w_slices, axis=2)  # [C, out_h, out_w]
        return out


# ------------------------------ VGG configs ------------------------------ #
# Torchvision cfgs: 'M' denotes MaxPool.
_CFGS: Dict[str, List] = {
    # VGG11
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # VGG13
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # VGG16
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    # VGG19
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

_ARCH_TO_CFG = {
    "vgg11": ("A", False),
    "vgg13": ("B", False),
    "vgg16": ("D", False),
    "vgg19": ("E", False),
    "vgg11_bn": ("A", True),
    "vgg13_bn": ("B", True),
    "vgg16_bn": ("D", True),
    "vgg19_bn": ("E", True),
}


# ------------------------------- VGG model ------------------------------- #
class VGG(eqx.Module):
    """VGG backbone + classifier, torchvision-compatible structure."""

    # feature extractor as a flat sequence (matches torchvision.features indices)
    features: Tuple[eqx.Module, ...]
    # adaptive avgpool to (7,7) for classifier compatibility
    avgpool: AdaptiveAvgPool2d
    # classifier
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear
    dr1: eqx.nn.Dropout
    dr2: eqx.nn.Dropout
    act1: ReLU
    act2: ReLU

    arch: str = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    batch_norm: bool = eqx.field(static=True)

    def __init__(self, *, arch: str = "vgg16_bn", num_classes: int = 1000, key):
        if arch not in _ARCH_TO_CFG:
            raise ValueError(f"Unknown VGG arch '{arch}'.")
        cfg_name, use_bn = _ARCH_TO_CFG[arch]
        cfg = _CFGS[cfg_name]

        # keys
        # We need one key per Conv layer in features + 3 for classifier linears.
        num_convs = sum(isinstance(v, int) for v in cfg)
        k_feat, k_fc1, k_fc2, k_fc3 = jr.split(key, 4)
        conv_keys = list(jr.split(k_feat, num_convs))
        ckey_iter = iter(conv_keys)

        # Build features (matches torchvision's features indexing)
        feats: List[eqx.Module] = []
        in_ch = 3
        for v in cfg:
            if v == "M":
                feats.append(eqx.nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = eqx.nn.Conv2d(
                    in_ch, int(v), kernel_size=3, padding=1, key=next(ckey_iter)
                )
                feats.append(conv)
                if use_bn:
                    feats.append(
                        eqx.nn.BatchNorm(int(v), axis_name="batch", mode="batch")
                    )
                feats.append(ReLU())
                in_ch = int(v)
        self.features = tuple(feats)

        self.avgpool = AdaptiveAvgPool2d(7, 7)

        # Classifier (torchvision-compatible)
        self.fc1 = eqx.nn.Linear(512 * 7 * 7, 4096, key=k_fc1)
        self.act1 = ReLU()
        self.dr1 = eqx.nn.Dropout(0.5)
        self.fc2 = eqx.nn.Linear(4096, 4096, key=k_fc2)
        self.act2 = ReLU()
        self.dr2 = eqx.nn.Dropout(0.5)
        self.fc3 = eqx.nn.Linear(4096, num_classes, key=k_fc3)

        self.arch = arch
        self.num_classes = num_classes
        self.batch_norm = use_bn

    def _check_input(self, x: jnp.ndarray):
        if x.ndim != 3:
            raise ValueError(
                f"VGG expects single sample [C,H,W]; got shape {tuple(x.shape)}."
            )
        if x.shape[0] != 3:
            raise ValueError(f"Expected 3 input channels; got {x.shape[0]}.")

    def __call__(self, x, key, state):
        """
        x: [3, H, W] single-sample
        Returns: (logits [num_classes], state)
        """
        self._check_input(x)

        # helper to split keys on-demand for conv/dropout
        def next_key(k):
            k1, k2 = jr.split(k)
            return k1, k2

        # Forward features
        k_run = key
        for layer in self.features:
            if isinstance(layer, eqx.nn.Conv2d):
                k_use, k_run = next_key(k_run)
                x = layer(x, key=k_use)
            elif isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state)
            elif isinstance(layer, eqx.nn.MaxPool2d):
                x = layer(x)
            elif isinstance(layer, ReLU):
                x, state = layer(x, state=state)
            else:
                # Unknown layer (shouldn't happen)
                x = layer(x)

        # Adaptive avgpool to 7x7 (classifier expects this)
        x = self.avgpool(x)  # [C, 7, 7]
        x = x.reshape(-1)  # [C*7*7]

        # Classifier
        x = self.fc1(x)
        x, state = self.act1(x, state=state)
        k_use, k_run = next_key(k_run)
        x = self.dr1(x, key=k_use)

        x = self.fc2(x)
        x, state = self.act2(x, state=state)
        k_use, k_run = next_key(k_run)
        x = self.dr2(x, key=k_use)

        logits = self.fc3(x)
        return logits, state


# ---------------------------- Weight loading ---------------------------- #
def _rename_torch_key(k: str) -> str | None:
    """
    Map torchvision VGG state_dict keys to our module tree.
    We keep `features.*` as-is (indices must match).
    Classifier mapping:
        classifier.0.* -> fc1.*
        classifier.3.* -> fc2.*
        classifier.6.* -> fc3.*
    Drop others (ReLU/Dropout/avgpool/running stats/num_batches_tracked).
    """
    # Features: keep conv/bn weights; skip BN running stats and num_batches_tracked
    if k.startswith("features."):
        if any(s in k for s in ("running_mean", "running_var", "num_batches_tracked")):
            return None
        return k

    # Classifier: only linear layers 0,3,6 have params
    if k.startswith("classifier.0."):
        return k.replace("classifier.0.", "fc1.")
    if k.startswith("classifier.3."):
        return k.replace("classifier.3.", "fc2.")
    if k.startswith("classifier.6."):
        return k.replace("classifier.6.", "fc3.")

    # avgpool has no params; ReLUs and Dropouts have none.
    return None


def _copy_into_tree(obj, pt: Dict[str, jnp.ndarray], prefix: str = ""):
    """Recursively copy weights into an Equinox pytree (Conv/Linear/BatchNorm + tuples)."""
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

            if isinstance(attr, tuple):
                new_tuple = []
                for i, child in enumerate(attr):
                    child_full = f"{full}.{i}"
                    new_tuple.append(_copy_into_tree(child, pt, prefix=child_full))
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, tuple(new_tuple))
                continue

            # else: stateless layers (ReLU, Dropout, pools) or static fields
        return obj

    if isinstance(obj, tuple):
        return tuple(_copy_into_tree(x, pt, prefix=prefix) for x in obj)
    return obj


def load_torchvision_vgg(model: VGG, npz_path: str, *, strict_fc: bool = True) -> VGG:
    """Load torchvision VGG .npz into this model (features + classifier).
    If `strict_fc=False` and num_classes != 1000, skips final FC copy.
    """
    import numpy as np

    raw = dict(np.load(npz_path))
    pt = {}
    for k, v in raw.items():
        nk = _rename_torch_key(k)
        if nk is None:
            continue
        pt[nk] = jnp.asarray(v)

    # Handle fc shape mismatch
    if strict_fc and ("fc3.weight" in pt):
        want_out, want_in = model.fc3.weight.shape
        have_out, have_in = pt["fc3.weight"].shape
        if (want_out != have_out) or (want_in != have_in):
            raise ValueError(
                f"FC shape mismatch: want {(want_out, want_in)} vs have {(have_out, have_in)}. "
                f"Set strict_fc=False to skip loading final FC."
            )
    else:
        # Drop fc3 if shapes differ or strict disabled
        if "fc3.weight" in pt:
            if model.fc3.weight.shape != pt["fc3.weight"].shape:
                pt.pop("fc3.weight", None)
                pt.pop("fc3.bias", None)

    return _copy_into_tree(model, pt, prefix="")


# Convenience arch-specific loaders
def load_imagenet_vgg11(
    model: VGG, npz="vgg11_imagenet.npz", strict_fc: bool = True
) -> VGG:
    return load_torchvision_vgg(model, npz, strict_fc=strict_fc)


def load_imagenet_vgg13(
    model: VGG, npz="vgg13_imagenet.npz", strict_fc: bool = True
) -> VGG:
    return load_torchvision_vgg(model, npz, strict_fc=strict_fc)


def load_imagenet_vgg16(
    model: VGG, npz="vgg16_imagenet.npz", strict_fc: bool = True
) -> VGG:
    return load_torchvision_vgg(model, npz, strict_fc=strict_fc)


def load_imagenet_vgg19(
    model: VGG, npz="vgg19_imagenet.npz", strict_fc: bool = True
) -> VGG:
    return load_torchvision_vgg(model, npz, strict_fc=strict_fc)


def load_imagenet_vgg11_bn(
    model: VGG, npz="vgg11_bn_imagenet.npz", strict_fc: bool = True
) -> VGG:
    return load_torchvision_vgg(model, npz, strict_fc=strict_fc)


def load_imagenet_vgg13_bn(
    model: VGG, npz="vgg13_bn_imagenet.npz", strict_fc: bool = True
) -> VGG:
    return load_torchvision_vgg(model, npz, strict_fc=strict_fc)


def load_imagenet_vgg16_bn(
    model: VGG, npz="vgg16_bn_imagenet.npz", strict_fc: bool = True
) -> VGG:
    return load_torchvision_vgg(model, npz, strict_fc=strict_fc)


def load_imagenet_vgg19_bn(
    model: VGG, npz="vgg19_bn_imagenet.npz", strict_fc: bool = True
) -> VGG:
    return load_torchvision_vgg(model, npz, strict_fc=strict_fc)


# ------------------------------ Smoke test ------------------------------ #
if __name__ == "__main__":
    """
    Synthetic classification smoke test for VGG.
    Uses 224×224 RGB to match canonical VGG geometry; replace with real data when integrating.
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

    # Choose an arch to smoke-test
    model, state = eqx.nn.make_with_state(VGG)(
        arch="vgg16_bn",
        num_classes=NUM_CLASSES,
        key=model_key,
    )

    # Optional backbone+classifier weight load (requires ImageNet shapes)
    # model = load_imagenet_vgg16_bn(model, "vgg16_bn_imagenet.npz", strict_fc=False)

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
    plt.title("VGG16-BN smoke test")
    plt.show()
