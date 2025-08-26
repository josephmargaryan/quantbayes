"""
Equinox ResNet (ImageNet-style) classifier with torchvision weight loader.

Backbones supported: "resnet18", "resnet34", "resnet50", "resnet101", "resnet152".
- Uses BatchNorm with mode="batch" (no EMA state).
- Single-sample forward with channel-first inputs [C, H, W].
- __call__(self, x, key, state) -> (logits, state).

Weight loading:
    - Save torchvision weights once (see helper script at bottom).
    - Use `load_torchvision_resnet(model, "resnet34_imagenet.npz")`.

Author: Joseph Margaryan (structure & conventions), implementation adapted for classification.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# ------------------------- Blocks & Specs ------------------------- #
def _center_crop_or_pad_nchw(x: jnp.ndarray, H2: int, W2: int) -> jnp.ndarray:
    """Center-crop or pad a (C,H,W) tensor to exactly (H2,W2)."""
    C, h, w = x.shape
    dh, dw = H2 - h, W2 - w
    # pad if needed
    if dh > 0 or dw > 0:
        pad_h1 = max(0, dh) // 2
        pad_h2 = max(0, dh) - pad_h1
        pad_w1 = max(0, dw) // 2
        pad_w2 = max(0, dw) - pad_w1
        x = jnp.pad(x, ((0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2)))
        _, h, w = x.shape
        dh, dw = H2 - h, W2 - w
    # crop if needed
    if dh < 0 or dw < 0:
        sh = (-dh) // 2
        sw = (-dw) // 2
        x = x[:, sh : sh + H2, sw : sw + W2]
    return x


def _downsample_skip_to(identity: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    """
    Make the skip path spatially match `ref` (both are (C,H,W)).
    Prefer integer-factor average pooling; otherwise center crop/pad.
    """
    C, H, W = identity.shape
    C2, H2, W2 = ref.shape
    if (H, W) == (H2, W2):
        return identity
    if C != C2:
        raise ValueError(
            f"Skip/branch channel mismatch (C={C} vs C2={C2}) without a downsample conv."
        )

    # try integer-factor average pooling
    sh = H // H2 if H2 > 0 else 1
    sw = W // W2 if W2 > 0 else 1
    if (H2 * sh == H) and (W2 * sw == W) and (sh == sw) and (sh >= 1):
        s = sh
        if s > 1:
            Hp = (H // s) * s
            Wp = (W // s) * s
            x = identity[:, :Hp, :Wp]
            # reshape-based average pool (no params)
            x = x.reshape(C, Hp // s, s, Wp // s, s).mean(axis=(2, 4))
            if x.shape[-2:] != (H2, W2):
                x = _center_crop_or_pad_nchw(x, H2, W2)
            return x
        return identity  # s==1
    # fallback: center crop/pad
    return _center_crop_or_pad_nchw(identity, H2, W2)


# ------------------------- Blocks & Specs ------------------------- #
class BasicBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    down_conv: eqx.nn.Conv2d | None
    down_bn: eqx.nn.BatchNorm | None
    stride: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True, default=1.0)

    def __init__(self, cin: int, cout: int, stride: int, *, key, alpha: float = 1.0):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(cin, cout, 3, stride=stride, padding=1, key=k1)
        self.bn1 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.conv2 = eqx.nn.Conv2d(cout, cout, 3, padding=1, key=k2)
        self.bn2 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        if stride != 1 or cin != cout:
            self.down_conv = eqx.nn.Conv2d(cin, cout, 1, stride=stride, key=k3)
            self.down_bn = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        else:
            self.down_conv = None
            self.down_bn = None
        self.stride = stride
        object.__setattr__(self, "alpha", float(alpha))

    def __call__(self, x, key, state):
        k1, k2, k3 = jr.split(key, 3)
        identity = x

        out = self.conv1(x, key=k1)
        out, state = self.bn1(out, state)
        out = jax.nn.relu(out)

        out = self.conv2(out, key=k2)
        out, state = self.bn2(out, state)

        # ensure skip path matches
        if self.down_conv is not None:
            identity = self.down_conv(identity, key=k3)
            identity, state = self.down_bn(identity, state)
        if identity.shape != out.shape:
            identity = _downsample_skip_to(identity, out)

        out = jax.nn.relu(identity + self.alpha * out)
        return out, state


class Bottleneck(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    conv3: eqx.nn.Conv2d
    bn3: eqx.nn.BatchNorm
    down_conv: eqx.nn.Conv2d | None
    down_bn: eqx.nn.BatchNorm | None
    stride: int = eqx.field(static=True)
    expansion: int = eqx.field(static=True, default=4)
    alpha: float = eqx.field(static=True, default=1.0)

    def __init__(self, cin: int, cout: int, stride: int, *, key, alpha: float = 1.0):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(cin, cout, 1, key=k1)
        self.bn1 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.conv2 = eqx.nn.Conv2d(cout, cout, 3, stride=stride, padding=1, key=k2)
        self.bn2 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.conv3 = eqx.nn.Conv2d(cout, cout * 4, 1, key=k3)
        self.bn3 = eqx.nn.BatchNorm(cout * 4, axis_name="batch", mode="batch")
        if stride != 1 or cin != cout * 4:
            self.down_conv = eqx.nn.Conv2d(cin, cout * 4, 1, stride=stride, key=k4)
            self.down_bn = eqx.nn.BatchNorm(cout * 4, axis_name="batch", mode="batch")
        else:
            self.down_conv = None
            self.down_bn = None
        self.stride = stride
        object.__setattr__(self, "alpha", float(alpha))

    def __call__(self, x, key, state):
        k1, k2, k3, k4 = jr.split(key, 4)
        identity = x

        out = self.conv1(x, key=k1)
        out, state = self.bn1(out, state)
        out = jax.nn.relu(out)

        out = self.conv2(out, key=k2)
        out, state = self.bn2(out, state)
        out = jax.nn.relu(out)

        out = self.conv3(out, key=k3)
        out, state = self.bn3(out, state)

        # ensure skip path matches
        if self.down_conv is not None:
            identity = self.down_conv(identity, key=k4)
            identity, state = self.down_bn(identity, state)
        if identity.shape != out.shape:
            identity = _downsample_skip_to(identity, out)

        out = jax.nn.relu(identity + self.alpha * out)
        return out, state


# Mirrors torchvision layer configs; channels[-1] is the final feature dim.
_RESNET_SPECS: Dict[str, Dict[str, Any]] = {
    "resnet18": dict(
        block=BasicBlock, layers=[2, 2, 2, 2], channels=[64, 64, 128, 256, 512]
    ),
    "resnet34": dict(
        block=BasicBlock, layers=[3, 4, 6, 3], channels=[64, 64, 128, 256, 512]
    ),
    "resnet50": dict(
        block=Bottleneck, layers=[3, 4, 6, 3], channels=[64, 256, 512, 1024, 2048]
    ),
    "resnet101": dict(
        block=Bottleneck, layers=[3, 4, 23, 3], channels=[64, 256, 512, 1024, 2048]
    ),
    "resnet152": dict(
        block=Bottleneck, layers=[3, 8, 36, 3], channels=[64, 256, 512, 1024, 2048]
    ),
}


# ------------------------- Model ------------------------- #
class ResNetClassifier(eqx.Module):
    """Original ImageNet-style ResNet classifier."""

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    pool: eqx.nn.MaxPool2d
    layers1: Tuple[eqx.Module, ...]
    layers2: Tuple[eqx.Module, ...]
    layers3: Tuple[eqx.Module, ...]
    layers4: Tuple[eqx.Module, ...]
    fc: eqx.nn.Linear

    num_classes: int = eqx.field(static=True)
    backbone: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        backbone: str = "resnet34",
        num_classes: int = 1000,
        key,
        residual_alpha: float = 1.0,
    ):
        if backbone not in _RESNET_SPECS:
            raise ValueError(f"Unknown backbone '{backbone}'.")
        spec = _RESNET_SPECS[backbone]
        Block = spec["block"]
        layer_sizes = spec["layers"]
        final_dim = spec["channels"][-1]  # 512 for basic, 2048 for bottleneck

        # keys
        num_blocks = sum(layer_sizes)
        ks = list(jr.split(key, 2 + num_blocks + 1))  # conv1/pool + blocks + fc

        self.conv1 = eqx.nn.Conv2d(3, 64, 7, stride=2, padding=3, key=ks[0])
        self.bn1 = eqx.nn.BatchNorm(64, axis_name="batch", mode="batch")
        self.pool = eqx.nn.MaxPool2d(3, 2, padding=1)

        def _make_layer(
            cin: int, cout: int, blocks: int, stride: int, kiter
        ) -> Tuple[Tuple[eqx.Module, ...], int]:
            mods: List[eqx.Module] = []
            c_in = cin
            for i in range(blocks):
                s = stride if i == 0 else 1
                mods.append(
                    Block(c_in, cout, s, key=next(kiter), alpha=residual_alpha)
                )  # <— pass alpha
                c_in = cout * (4 if Block is Bottleneck else 1)
            return tuple(mods), c_in

        kiter = iter(ks[1 : 1 + num_blocks])

        self.layers1, ch1 = _make_layer(64, 64, layer_sizes[0], 1, kiter)
        self.layers2, ch2 = _make_layer(ch1, 128, layer_sizes[1], 2, kiter)
        self.layers3, ch3 = _make_layer(ch2, 256, layer_sizes[2], 2, kiter)
        self.layers4, _ = _make_layer(ch3, 512, layer_sizes[3], 2, kiter)

        # classifier
        self.fc = eqx.nn.Linear(final_dim, num_classes, key=ks[-1])

        self.num_classes = num_classes
        self.backbone = backbone

    def _check_input(self, x: jnp.ndarray):
        if x.ndim != 3:
            raise ValueError(
                f"ResNetClassifier expects single sample [C,H,W]; got shape {tuple(x.shape)}."
            )
        if x.shape[0] != 3:
            # Still allow non-RGB but warn via exception to be explicit.
            # Change to soft assumption if you want to support other channel counts.
            raise ValueError(f"Expected 3 input channels; got {x.shape[0]}.")

    def __call__(self, x, key, state):
        # x: [C,H,W] single sample
        self._check_input(x)

        k0, key = jr.split(key)
        x = self.conv1(x, key=k0)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        x = self.pool(x)

        for layer in (self.layers1, self.layers2, self.layers3, self.layers4):
            for block in layer:
                kb, key = jr.split(key)
                x, state = block(x, key=kb, state=state)

        # Global average pooling (ImageNet-style)
        x = jnp.mean(x, axis=(-2, -1))  # [C]

        logits = self.fc(x)  # [num_classes]
        return logits, state


# ------------------------- Torch weight loader ------------------------- #
# We map torchvision keys to our module tree:
#   conv1.*, bn1.*, layer{1..4}.{idx}.*, downsample.{0,1}.*  ->  conv1/bn1/layers{1..4}.{idx}.* with down_conv/down_bn
# We keep fc.* if num_classes==1000 (shape match); otherwise skip it.


def _rename_pt_key(k: str) -> str:
    # Keep stem + fc names; remap layer blocks; downsample -> (down_conv/down_bn)
    k = k.replace("downsample.0.", "down_conv.")
    k = k.replace("downsample.1.", "down_bn.")
    k = k.replace("layer1.", "layers1.")
    k = k.replace("layer2.", "layers2.")
    k = k.replace("layer3.", "layers3.")
    k = k.replace("layer4.", "layers4.")
    return k  # conv1., bn1., fc. left intact


def _copy_into_tree(obj, pt: Dict[str, jnp.ndarray], prefix: str = ""):
    """Recursively copy weights into an Equinox pytree.
    Supports eqx.Module, tuple/list/dict containers."""
    # eqx modules: iterate fields
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

            # BatchNorm: copy affine only (no EMA state in "batch" mode)
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

            # Containers: tuple/list/dict
            if isinstance(attr, tuple):
                new_tuple = []
                for i, child in enumerate(attr):
                    child_full = f"{full}.{i}"
                    new_tuple.append(_copy_into_tree(child, pt, prefix=child_full))
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, tuple(new_tuple))
                continue

            if isinstance(attr, list):
                new_list = []
                for i, child in enumerate(attr):
                    child_full = f"{full}.{i}"
                    new_list.append(_copy_into_tree(child, pt, prefix=child_full))
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, new_list)
                continue

            if isinstance(attr, dict):
                new_dict = {}
                for k, child in attr.items():
                    child_full = f"{full}.{k}"
                    new_dict[k] = _copy_into_tree(child, pt, prefix=child_full)
                obj = eqx.tree_at(lambda m: getattr(m, name), obj, new_dict)
                continue

            # else: static or None -> skip
        return obj

    # Containers outside modules (rare)
    if isinstance(obj, tuple):
        return tuple(_copy_into_tree(x, pt, prefix=prefix) for x in obj)
    if isinstance(obj, list):
        return [_copy_into_tree(x, pt, prefix=prefix) for x in obj]
    if isinstance(obj, dict):
        return {k: _copy_into_tree(v, pt, prefix=prefix) for k, v in obj.items()}

    return obj  # leaf


def load_torchvision_resnet(
    model: ResNetClassifier, npz_path: str, *, strict_fc: bool = True
) -> ResNetClassifier:
    """Load a torchvision ResNet .npz (saved from `state_dict()`) into this classifier."""
    import numpy as np  # local import to avoid hard dependency when unused

    raw = dict(np.load(npz_path))
    # Keep only weight/bias arrays (ignore running stats/num_batches_tracked automatically)
    pt = {}
    for k, v in raw.items():
        new_k = _rename_pt_key(k)
        pt[new_k] = jnp.asarray(v)

    # If strict_fc is False and shapes mismatch, drop fc.* keys
    fc_w_key, fc_b_key = "fc.weight", "fc.bias"
    if (fc_w_key in pt) and (fc_b_key in pt):
        want_out, want_in = model.fc.weight.shape
        have_out, have_in = pt[fc_w_key].shape
        if (want_out != have_out) or (want_in != have_in):
            if strict_fc:
                raise ValueError(
                    f"FC shape mismatch: want {(want_out, want_in)}, have {(have_out, have_in)}."
                )
            pt.pop(fc_w_key, None)
            pt.pop(fc_b_key, None)

    return _copy_into_tree(model, pt, prefix="")


def load_imagenet_resnet18(
    model: ResNetClassifier, npz="resnet18_imagenet.npz", strict_fc: bool = True
) -> ResNetClassifier:
    return load_torchvision_resnet(model, npz, strict_fc=strict_fc)


def load_imagenet_resnet34(
    model: ResNetClassifier, npz="resnet34_imagenet.npz", strict_fc: bool = True
) -> ResNetClassifier:
    return load_torchvision_resnet(model, npz, strict_fc=strict_fc)


def load_imagenet_resnet50(
    model: ResNetClassifier, npz="resnet50_imagenet.npz", strict_fc: bool = True
) -> ResNetClassifier:
    return load_torchvision_resnet(model, npz, strict_fc=strict_fc)


def load_imagenet_resnet101(
    model: ResNetClassifier, npz="resnet101_imagenet.npz", strict_fc: bool = True
) -> ResNetClassifier:
    return load_torchvision_resnet(model, npz, strict_fc=strict_fc)


def load_imagenet_resnet152(
    model: ResNetClassifier, npz="resnet152_imagenet.npz", strict_fc: bool = True
) -> ResNetClassifier:
    return load_torchvision_resnet(model, npz, strict_fc=strict_fc)


# ------------------------- Save-from-torch helper (optional) ------------------------- #
# Usage:
#   $ python save_torchvision_resnets.py
# Generates resnet{18,34,50,101,152}_imagenet.npz files in CWD.
SAVE_TORCH_HELPER = r"""
from pathlib import Path
import numpy as np
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152
)

CHECKPOINTS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}

def main():
    for name, builder in CHECKPOINTS.items():
        print(f"⇢ downloading {name} …")
        model = builder(weights="IMAGENET1K_V1") # or "IMAGENET1K_V2" for newer weights
        ckpt_path = Path(f"{name}_imagenet.npz")
        print(f"↳ saving → {ckpt_path}")
        np.savez(ckpt_path, **{k: v.cpu().numpy() for k, v in model.state_dict().items()})
        print(f"✓ done {ckpt_path}\\n")

if __name__ == "__main__":
    main()
"""


# ------------------------- Smoke test ------------------------- #
if __name__ == "__main__":
    """
    Synthetic classification smoke-test (single-sample models).
    Replace X/y with real data when integrating.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import augmax
    from augmax import InputType
    import optax

    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx

    # Your training utilities (assumed available per your examples)
    from quantbayes.stochax import (
        train,
        predict,
        make_augmax_augment,
        multiclass_loss,
    )

    rng = np.random.RandomState(0)
    N, C, H, W, NUM_CLASSES = 1024, 3, 96, 96, 10
    X_np = rng.rand(N, C, H, W).astype(
        "float32"
    )  # channel-first (single-sample at call time)
    y_np = rng.randint(0, NUM_CLASSES, size=(N,)).astype("int32")

    # train/val split
    split = int(0.8 * N)
    X_train, X_val = X_np[:split], X_np[split:]
    y_train, y_val = y_np[:split], y_np[split:]

    # a light augmentation chain (image + labels as metadata)
    transform = augmax.Chain(
        augmax.HorizontalFlip(),
        augmax.Rotate(angle_range=10),
        input_types=[InputType.IMAGE, InputType.METADATA],
    )
    augment_fn = make_augmax_augment(transform)

    backbones = ["resnet18", "resnet34", "resnet50"]  # extend to 101/152 if you like
    master_key = jr.PRNGKey(42)

    for i, backbone in enumerate(backbones):
        print(f"\n=== Smoke-testing ResNetClassifier {backbone} ===")
        model_key, train_key = jr.split(jr.fold_in(master_key, i))

        # init model+state
        model, state = eqx.nn.make_with_state(ResNetClassifier)(
            backbone=backbone,
            num_classes=NUM_CLASSES,
            key=model_key,
        )

        # optimizer & schedule
        lr_sched = optax.cosine_decay_schedule(1e-3, decay_steps=400)
        optimizer = optax.adamw(
            learning_rate=lr_sched,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=1e-4,
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        # train loop (your helper handles single-sample model semantics)
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
            batch_size=64,
            num_epochs=10,
            patience=3,
            key=train_key,
            augment_fn=augment_fn,
            lambda_spec=0.0,
        )

        # quick eval
        logits = predict(best_model, best_state, jnp.array(X_val), train_key)
        print("Predictions shape:", logits.shape)

        # curves
        plt.figure()
        plt.plot(tr_loss, label="train")
        plt.plot(va_loss, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title(f"ResNetClassifier {backbone}")
        plt.show()

    # If you need the torch-save helper as a separate script:
    # with open("save_torchvision_resnets.py", "w") as f:
    #     f.write(SAVE_TORCH_HELPER)
    # print("Wrote: save_torchvision_resnets.py")
