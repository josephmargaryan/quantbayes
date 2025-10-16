"""
HRNet-OCR for semantic segmentation (Equinox).

- Single-sample CHW input, returns [main_logits, aux_logits] for deep supervision.
- __call__(self, x, key, state) -> (logits_or_list, state)
- BatchNorm uses mode="batch" (no EMA), consistent with your codebase.

Backbone variants:
    "hrnet_w18_small_v2" (default), "hrnet_w18", "hrnet_w32", "hrnet_w48"

Notes:
- We implement HRNetV2-style multi-branch stages (Basic/Bottleneck), multi-scale fuse.
- OCR head: SpatialGather + ObjectAttention refinement + classifier; plus an auxiliary head.
- If you want pretrained backbone: export from timm, map into the backbone conv/bn names.
  (OCR head is usually task-specific; keep randomly initialized.)
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.image as jimg
import einops
import equinox as eqx


# ------------------------------ Residual blocks ------------------------------ #


class BasicBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    down: eqx.nn.Conv2d | None
    down_bn: eqx.nn.BatchNorm | None

    def __init__(self, cin: int, cout: int, stride: int = 1, *, key):
        k1, k2, k3 = jr.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(
            cin, cout, 3, stride=stride, padding=1, use_bias=False, key=k1
        )
        self.bn1 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.conv2 = eqx.nn.Conv2d(cout, cout, 3, padding=1, use_bias=False, key=k2)
        self.bn2 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        if stride != 1 or cin != cout:
            self.down = eqx.nn.Conv2d(
                cin, cout, 1, stride=stride, use_bias=False, key=k3
            )
            self.down_bn = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        else:
            self.down, self.down_bn = None, None

    def __call__(self, x, key, state):
        y = self.conv1(x, key=jr.fold_in(key, 1))
        y, state = self.bn1(y, state)
        y = jax.nn.relu(y)
        y = self.conv2(y, key=jr.fold_in(key, 2))
        y, state = self.bn2(y, state)

        if self.down is not None:
            skip = self.down(x, key=jr.fold_in(key, 3))
            skip, state = self.down_bn(skip, state)
        else:
            skip = x
        out = jax.nn.relu(y + skip)
        return out, state


class Bottleneck(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    conv3: eqx.nn.Conv2d
    bn3: eqx.nn.BatchNorm
    down: eqx.nn.Conv2d | None
    down_bn: eqx.nn.BatchNorm | None
    expansion: int = eqx.field(static=True, default=4)

    def __init__(self, cin: int, planes: int, stride: int = 1, *, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        cout = planes
        self.conv1 = eqx.nn.Conv2d(cin, cout, 1, use_bias=False, key=k1)
        self.bn1 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.conv2 = eqx.nn.Conv2d(
            cout, cout, 3, stride=stride, padding=1, use_bias=False, key=k2
        )
        self.bn2 = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")
        self.conv3 = eqx.nn.Conv2d(
            cout, cout * self.expansion, 1, use_bias=False, key=k3
        )
        self.bn3 = eqx.nn.BatchNorm(
            cout * self.expansion, axis_name="batch", mode="batch"
        )
        if stride != 1 or cin != cout * self.expansion:
            self.down = eqx.nn.Conv2d(
                cin, cout * self.expansion, 1, stride=stride, use_bias=False, key=k4
            )
            self.down_bn = eqx.nn.BatchNorm(
                cout * self.expansion, axis_name="batch", mode="batch"
            )
        else:
            self.down, self.down_bn = None, None

    def __call__(self, x, key, state):
        y = self.conv1(x, key=jr.fold_in(key, 1))
        y, state = self.bn1(y, state)
        y = jax.nn.relu(y)
        y = self.conv2(y, key=jr.fold_in(key, 2))
        y, state = self.bn2(y, state)
        y = jax.nn.relu(y)
        y = self.conv3(y, key=jr.fold_in(key, 3))
        y, state = self.bn3(y, state)
        if self.down is not None:
            skip = self.down(x, key=jr.fold_in(key, 4))
            skip, state = self.down_bn(skip, state)
        else:
            skip = x
        out = jax.nn.relu(y + skip)
        return out, state


# --------------------------- HRNet modules (stages) --------------------------- #


def _make_layer(block_ctor, cin, cout, blocks, *, key):
    ks = iter(jr.split(key, blocks))
    layers = []
    ch = cin
    for i in range(blocks):
        stride = 1
        b = block_ctor(ch, cout, stride, key=next(ks))
        ch = cout if block_ctor is BasicBlock else cout * b.expansion
        layers.append(b)
    return tuple(layers), ch


class HighResolutionModule(eqx.Module):
    branches: Tuple[Tuple[eqx.Module, ...], ...]
    fuse_layers: Tuple[Tuple[Tuple[eqx.Module | None, ...], ...], ...]
    num_branches: int = eqx.field(static=True)

    def __init__(
        self,
        num_branches: int,
        blocks: int,
        channels: Tuple[int, ...],
        block_type,
        *,
        key,
    ):
        """
        channels: per-branch output channels for this module (after its blocks)
        """
        assert num_branches == len(channels)
        self.num_branches = num_branches

        # branches
        branch_keys = iter(jr.split(key, num_branches + num_branches * num_branches))
        branches = []
        for i in range(num_branches):
            layers, _ = _make_layer(
                BasicBlock if block_type == "basic" else Bottleneck,
                cin=channels[i],
                cout=channels[i],
                blocks=blocks,
                key=next(branch_keys),
            )
            branches.append(layers)
        self.branches = tuple(branches)

        # fuse layers: for output i, sum transforms from each input j
        fuse = []
        for i in range(num_branches):
            row = []
            for j in range(num_branches):
                if i == j:
                    row.append(None)
                elif j > i:
                    # upsample: 1x1 conv to match channels
                    row.append(
                        eqx.nn.Conv2d(
                            channels[j],
                            channels[i],
                            1,
                            use_bias=False,
                            key=next(branch_keys),
                        )
                    )
                else:
                    # downsample: sequence of strided 3x3 convs
                    ops = []
                    in_c = channels[j]
                    for k in range(i - j):
                        out_c = channels[i] if k == i - j - 1 else in_c
                        ops.append(
                            eqx.nn.Conv2d(
                                in_c,
                                out_c,
                                3,
                                stride=2,
                                padding=1,
                                use_bias=False,
                                key=next(branch_keys),
                            )
                        )
                        in_c = out_c
                    row.append(tuple(ops))
            fuse.append(tuple(row))
        self.fuse_layers = tuple(fuse)

    def _apply_branch(self, x_chw, layers, key, state):
        k = key
        out = x_chw
        for b in layers:
            k, k1 = jr.split(k)
            out, state = b(out, key=k1, state=state)
        return out, state

    def __call__(
        self, xs: Tuple[jnp.ndarray, ...], key, state
    ) -> Tuple[Tuple[jnp.ndarray, ...], Any]:
        assert len(xs) == self.num_branches
        k = key

        # 1) run branches independently
        ys = []
        for i in range(self.num_branches):
            k, k1 = jr.split(k)
            y, state = self._apply_branch(xs[i], self.branches[i], k1, state)
            ys.append(y)

        # 2) fuse to produce outputs at all resolutions
        out = []
        for i in range(self.num_branches):
            y = ys[i]
            C, H, W = y.shape
            acc = y
            for j in range(self.num_branches):
                if j == i:
                    continue
                trans = self.fuse_layers[i][j]
                yj = ys[j]
                if isinstance(trans, eqx.nn.Conv2d):  # upsample path
                    yj = trans(yj, key=jr.fold_in(k, i * 10 + j))
                    yj = jimg.resize(yj, (yj.shape[0], H, W), method="linear")
                    acc = acc + yj
                else:
                    # downsample path: sequence of stride-2 convs
                    ops = trans
                    tmp = yj
                    for idx, op in enumerate(ops):
                        tmp = op(tmp, key=jr.fold_in(k, 1000 + i * 10 + j * 5 + idx))
                        # no BN here for simplicity (common impls do BN+ReLU). Keep lean.
                    acc = acc + tmp
            out.append(jax.nn.relu(acc))
        return tuple(out), state


class HRNetBackbone(eqx.Module):
    stem_conv1: eqx.nn.Conv2d
    stem_bn1: eqx.nn.BatchNorm
    stem_conv2: eqx.nn.Conv2d
    stem_bn2: eqx.nn.BatchNorm

    # stage1 (layer1)
    layer1: Tuple[eqx.Module, ...]
    # transitions to stages 2/3/4
    trans2: Tuple[eqx.Module, ...]
    stage2: HighResolutionModule

    trans3: Tuple[eqx.Module, ...]
    stage3: HighResolutionModule

    trans4: Tuple[eqx.Module, ...]
    stage4: HighResolutionModule

    widths: Tuple[int, int, int, int] = eqx.field(static=True)
    arch: str = eqx.field(static=True)

    def __init__(self, *, arch: str = "hrnet_w18_small_v2", key):
        """
        Supported:
          - hrnet_w18_small_v2: widths [18,36,72,144], blocks per module [2,2,2,2] (lite)
          - hrnet_w18:          widths [18,36,72,144], blocks per module [4,4,4,4]
          - hrnet_w32:          widths [32,64,128,256], blocks per module [4,4,4,4]
          - hrnet_w48:          widths [48,96,192,384], blocks per module [4,4,4,4]
        """
        cfg = {
            "hrnet_w18_small_v2": dict(widths=[18, 36, 72, 144], blocks=[2, 2, 2, 2]),
            "hrnet_w18": dict(widths=[18, 36, 72, 144], blocks=[4, 4, 4, 4]),
            "hrnet_w32": dict(widths=[32, 64, 128, 256], blocks=[4, 4, 4, 4]),
            "hrnet_w48": dict(widths=[48, 96, 192, 384], blocks=[4, 4, 4, 4]),
        }
        if arch not in cfg:
            raise ValueError(f"Unknown HRNet arch '{arch}'")
        widths = cfg[arch]["widths"]
        blocks = cfg[arch]["blocks"]
        self.widths = tuple(widths)
        self.arch = arch

        k_iter = iter(jr.split(key, 2048))

        # stem (1/4 resolution after two stride-2 convs)
        self.stem_conv1 = eqx.nn.Conv2d(
            3, 64, 3, stride=2, padding=1, use_bias=False, key=next(k_iter)
        )
        self.stem_bn1 = eqx.nn.BatchNorm(64, axis_name="batch", mode="batch")
        self.stem_conv2 = eqx.nn.Conv2d(
            64, 64, 3, stride=2, padding=1, use_bias=False, key=next(k_iter)
        )
        self.stem_bn2 = eqx.nn.BatchNorm(64, axis_name="batch", mode="batch")

        # layer1 (Bottleneck x 4; output 256 channels)
        self.layer1, c_after_l1 = _make_layer(
            Bottleneck, 64, 64, blocks=4, key=next(k_iter)
        )

        # transition to stage2 (2 branches)
        self.trans2 = (
            eqx.nn.Conv2d(
                c_after_l1, widths[0], 3, padding=1, use_bias=False, key=next(k_iter)
            ),
            eqx.nn.Conv2d(
                c_after_l1,
                widths[1],
                3,
                stride=2,
                padding=1,
                use_bias=False,
                key=next(k_iter),
            ),
        )
        self.stage2 = HighResolutionModule(
            num_branches=2,
            blocks=blocks[0],
            channels=(widths[0], widths[1]),
            block_type="basic",
            key=next(k_iter),
        )

        # transition to stage3 (3 branches)
        self.trans3 = (
            eqx.nn.Conv2d(
                widths[0], widths[0], 3, padding=1, use_bias=False, key=next(k_iter)
            ),  # keep
            eqx.nn.Conv2d(
                widths[1], widths[1], 3, padding=1, use_bias=False, key=next(k_iter)
            ),  # keep
            eqx.nn.Conv2d(
                widths[1],
                widths[2],
                3,
                stride=2,
                padding=1,
                use_bias=False,
                key=next(k_iter),
            ),  # new down
        )
        self.stage3 = HighResolutionModule(
            num_branches=3,
            blocks=blocks[1],
            channels=(widths[0], widths[1], widths[2]),
            block_type="basic",
            key=next(k_iter),
        )

        # transition to stage4 (4 branches)
        self.trans4 = (
            eqx.nn.Conv2d(
                widths[0], widths[0], 3, padding=1, use_bias=False, key=next(k_iter)
            ),  # keep
            eqx.nn.Conv2d(
                widths[1], widths[1], 3, padding=1, use_bias=False, key=next(k_iter)
            ),  # keep
            eqx.nn.Conv2d(
                widths[2], widths[2], 3, padding=1, use_bias=False, key=next(k_iter)
            ),  # keep
            eqx.nn.Conv2d(
                widths[2],
                widths[3],
                3,
                stride=2,
                padding=1,
                use_bias=False,
                key=next(k_iter),
            ),  # new down
        )
        self.stage4 = HighResolutionModule(
            num_branches=4,
            blocks=blocks[2],
            channels=(widths[0], widths[1], widths[2], widths[3]),
            block_type="basic",
            key=next(k_iter),
        )

    def _apply_layer(self, x, layers, key, state):
        k = key
        out = x
        for b in layers:
            k, k1 = jr.split(k)
            out, state = b(out, key=k1, state=state)
        return out, state

    def __call__(self, x, key, state):
        # stem
        x = self.stem_conv1(x, key=jr.fold_in(key, 1))
        x, state = self.stem_bn1(x, state)
        x = jax.nn.relu(x)
        x = self.stem_conv2(x, key=jr.fold_in(key, 2))
        x, state = self.stem_bn2(x, state)
        x = jax.nn.relu(x)
        # layer1
        x, state = self._apply_layer(x, self.layer1, jr.fold_in(key, 3), state)

        # stage2 inputs
        x0 = self.trans2[0](x, key=jr.fold_in(key, 4))
        x1 = self.trans2[1](x, key=jr.fold_in(key, 5))
        (x0, x1), state = self.stage2((x0, x1), key=jr.fold_in(key, 6), state=state)

        # stage3
        x0 = self.trans3[0](x0, key=jr.fold_in(key, 7))
        x1 = self.trans3[1](x1, key=jr.fold_in(key, 8))
        x2 = self.trans3[2](x1, key=jr.fold_in(key, 9))
        (x0, x1, x2), state = self.stage3(
            (x0, x1, x2), key=jr.fold_in(key, 10), state=state
        )

        # stage4
        x0 = self.trans4[0](x0, key=jr.fold_in(key, 11))
        x1 = self.trans4[1](x1, key=jr.fold_in(key, 12))
        x2 = self.trans4[2](x2, key=jr.fold_in(key, 13))
        x3 = self.trans4[3](x2, key=jr.fold_in(key, 14))
        feats, state = self.stage4(
            (x0, x1, x2, x3), key=jr.fold_in(key, 15), state=state
        )

        # feats: tuple of 4 feature maps at resolutions {1/4,1/8,1/16,1/32}
        return feats, state


# ------------------------------- OCR Head -------------------------------- #


class SpatialGather(eqx.Module):
    """Aggregate class-wise context vectors weighted by predicted probabilities."""

    def __call__(self, feats_chw: jnp.ndarray, class_logits_chw: jnp.ndarray):
        # feats: [Cf, H, W], logits: [K, H, W]
        Cf, H, W = feats_chw.shape
        K = class_logits_chw.shape[0]
        feats = einops.rearrange(feats_chw, "c h w -> (h w) c")  # [N, Cf]
        probs = jax.nn.softmax(
            einops.rearrange(class_logits_chw, "k h w -> (h w) k"), axis=-1
        )  # [N, K]
        weights = probs / jnp.clip(
            jnp.sum(probs, axis=0, keepdims=True), 1e-6
        )  # [N,K] normalized per class
        # context per class: [K, Cf] = (weights^T @ feats)
        context = jnp.einsum("nk, nc -> kc", weights, feats)
        return context  # [K, Cf]


class ObjectAttention(eqx.Module):
    """Pixel-object attention: queries from pixels, keys/values from class context."""

    query: eqx.nn.Conv2d
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    out: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        key_channels: int,
        value_channels: int,
        out_channels: int,
        *,
        key,
    ):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.query = eqx.nn.Conv2d(in_channels, key_channels, 1, use_bias=False, key=k1)
        self.key = eqx.nn.Linear(in_channels, key_channels, key=k2)
        self.value = eqx.nn.Linear(in_channels, value_channels, key=k3)
        self.out = eqx.nn.Conv2d(
            value_channels, out_channels, 1, use_bias=False, key=k4
        )

    def __call__(self, feats_chw: jnp.ndarray, context_kc: jnp.ndarray, key):
        # feats: [Cf, H, W]; context: [K, Cf]
        Cf, H, W = feats_chw.shape
        # queries from pixels
        q = self.query(feats_chw, key=jr.fold_in(key, 0))  # [Cq, H, W]
        q = einops.rearrange(q, "c h w -> (h w) c")  # [N, Cq]
        # keys/values from context
        k = jax.vmap(self.key)(context_kc)  # [K, Cq]
        v = jax.vmap(self.value)(context_kc)  # [K, Cv]

        # attention: [N,K]
        scale = 1.0 / jnp.sqrt(jnp.asarray(q.shape[-1], jnp.float32))
        attn = jax.nn.softmax(jnp.einsum("nc,kc->nk", q * scale, k), axis=-1)
        # aggregate values: [N, Cv]
        agg = jnp.einsum("nk,kc->nc", attn, v)
        agg_chw = einops.rearrange(agg, "(h w) c -> c h w", h=H, w=W)
        out = self.out(agg_chw, key=jr.fold_in(key, 1))  # [out_channels, H, W]
        return out


class OCRHead(eqx.Module):
    # pre-feature
    conv3x3: eqx.nn.Conv2d
    bn: eqx.nn.BatchNorm

    # aux head (deep supervision)
    aux_head: eqx.nn.Conv2d

    # OCR
    gather: SpatialGather
    obj_attn: ObjectAttention

    # classifier
    cls_head: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        key_channels: int,
        value_channels: int,
        num_classes: int,
        *,
        key,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.conv3x3 = eqx.nn.Conv2d(
            in_channels, 512, 3, padding=1, use_bias=False, key=k1
        )
        self.bn = eqx.nn.BatchNorm(512, axis_name="batch", mode="batch")
        self.aux_head = eqx.nn.Conv2d(512, num_classes, 1, key=jr.fold_in(key, 10))

        self.gather = SpatialGather()
        self.obj_attn = ObjectAttention(
            512, key_channels, value_channels, out_channels=512, key=k2
        )
        self.cls_head = eqx.nn.Conv2d(512, num_classes, 1, key=k3)

    def __call__(self, feats_concat: jnp.ndarray, key, state):
        x = self.conv3x3(feats_concat, key=jr.fold_in(key, 0))
        x, state = self.bn(x, state)
        x = jax.nn.relu(x)

        # auxiliary logits (same spatial size as x)
        aux_logits = self.aux_head(x, key=jr.fold_in(key, 1))

        # gather class context with aux logits
        context = self.gather(x, aux_logits)  # [K, 512]
        obj = self.obj_attn(x, context, key=jr.fold_in(key, 2))
        x = x + obj  # residual refine
        logits = self.cls_head(x, key=jr.fold_in(key, 3))
        return logits, aux_logits, state


# ------------------------------ Segmentation net ------------------------------ #


class HRNetOCR(eqx.Module):
    backbone: HRNetBackbone
    fuse_head: eqx.nn.Conv2d
    fuse_bn: eqx.nn.BatchNorm
    ocr: OCRHead
    num_classes: int = eqx.field(static=True)

    def __init__(self, *, arch: str = "hrnet_w18_small_v2", num_classes: int = 19, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.backbone = HRNetBackbone(arch=arch, key=k1)

        # Fuse all 4 branches to 1/4 res by upsampling and concatenation, then 1x1 -> 512
        widths = self.backbone.widths
        total_c = sum(widths)
        self.fuse_head = eqx.nn.Conv2d(total_c, 512, 1, use_bias=False, key=k2)
        self.fuse_bn = eqx.nn.BatchNorm(512, axis_name="batch", mode="batch")

        # OCR head
        self.ocr = OCRHead(
            in_channels=512,
            key_channels=256,
            value_channels=256,
            num_classes=num_classes,
            key=k3,
        )
        self.num_classes = num_classes

    def __call__(self, x, key, state):
        feats, state = self.backbone(x, key=jr.fold_in(key, 0), state=state)
        # feats[0] is 1/4 resolution; upsample others to match
        H, W = feats[0].shape[-2], feats[0].shape[-1]
        ups = [feats[0]]
        for f in feats[1:]:
            ups.append(jimg.resize(f, (f.shape[0], H, W), method="linear"))
        concat = jnp.concatenate(ups, axis=0)  # [sumC, H, W]
        y = self.fuse_head(concat, key=jr.fold_in(key, 1))
        y, state = self.fuse_bn(y, state)
        y = jax.nn.relu(y)

        logits_ocr, aux_logits, state = self.ocr(y, key=jr.fold_in(key, 2), state=state)
        # return [main, aux] for deep supervision; caller can weight heads.
        return [logits_ocr, aux_logits], state


# ------------------------------ Smoke test -------------------------------- #

if __name__ == "__main__":
    """
    Synthetic smoke test (small run). Replace with real dataset in practice.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import optax
    import equinox as eqx
    import jax.random as jr
    import jax.numpy as jnp

    from quantbayes.stochax import (
        train,
        predict,
        make_augmax_augment,
        make_dice_bce_loss,
    )
    import augmax
    from augmax import InputType

    rng = np.random.RandomState(0)
    N, C, H, W, OUT_CH = 12, 3, 128, 128, 1
    X_np = rng.rand(N, C, H, W).astype("float32")
    y_np = rng.randint(0, 2, size=(N, OUT_CH, H, W)).astype("float32")

    split = int(0.8 * N)
    X_train, X_val = X_np[:split], X_np[split:]
    y_train, y_val = y_np[:split], y_np[split:]

    transform = augmax.Chain(
        augmax.HorizontalFlip(),
        augmax.Rotate(angle_range=10),
        input_types=[InputType.IMAGE, InputType.MASK],
    )
    augment_fn = make_augmax_augment(transform)

    master_key = jr.PRNGKey(42)
    model_key, train_key = jr.split(master_key)

    model, state = eqx.nn.make_with_state(HRNetOCR)(
        arch="hrnet_w18_small_v2",
        num_classes=OUT_CH,
        key=model_key,
    )

    # Deep supervision: weight main more than aux (e.g., [0.8, 0.2])
    loss_fn = make_dice_bce_loss()
    loss_fn = loss_fn  # your trainer already supports multiple heads ⇒ weighted inside make_loss_fn if desired

    lr_sched = optax.cosine_decay_schedule(1e-3, decay_steps=400)
    optimizer = optax.adamw(learning_rate=lr_sched, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    best_model, best_state, tr_loss, va_loss = train(
        model=model,
        state=state,
        opt_state=opt_state,
        optimizer=optimizer,
        loss_fn=loss_fn,
        X_train=jnp.array(X_train),
        y_train=jnp.array(y_train),
        X_val=jnp.array(X_val),
        y_val=jnp.array(y_val),
        batch_size=16,
        num_epochs=8,
        patience=3,
        key=train_key,
        augment_fn=augment_fn,
        lambda_spec=0.0,
    )

    plt.plot(tr_loss, label="train")
    plt.plot(va_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("HRNet-OCR smoke test")
    plt.show()
