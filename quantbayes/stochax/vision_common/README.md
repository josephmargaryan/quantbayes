# Vision transformers: dense, SVDDense, and RFFT1D

This directory exposes one small public API for the workflows you described:

1. instantiate a model,
2. optionally load a pretrained checkpoint,
3. optionally replace dense linear layers with `SVDDense`,
4. choose whether to freeze only `U,V` or to train only the singular values `s`.

The intended import surface is:

```python
from quantbayes.stochax.vision_common import (
    available_arches,
    default_checkpoint_name,
    make_model,
    make_and_load_model,
    load_pretrained,
    replace_linears_with_svd,
    replace_attention_linears_with_svd,
    make_svd_basis_freeze_mask,
    make_s_only_freeze_mask,
)
```

## Supported attention families

The factory API supports the three attention families in this library:

- `family="vit"`
- `family="swin"`
- `family="dino"`

Each family exposes three variants:

- `variant="dense"`
- `variant="svd"`
- `variant="rfft"`

For SVD variants, use `svd_mode` in `{"attn_only", "attn_mlp", "all_linear"}`.

## 1. Instantiate a model

```python
import jax.random as jr

from quantbayes.stochax.vision_common import make_model

model, state = make_model(
    family="vit",
    arch="vit_b_16",
    variant="dense",
    image_size=224,
    num_classes=10,
    key=jr.PRNGKey(0),
)
```

The same pattern works for Swin and DINO:

```python
model, state = make_model("swin", "swin_t", variant="rfft", num_classes=10, key=jr.PRNGKey(0))

model, state = make_model(
    "dino",
    "vitb14",
    variant="svd",
    image_size=224,
    num_classes=10,
    registers=False,
    svd_mode="attn_mlp",
    svd_rank_ratio=0.25,
    key=jr.PRNGKey(0),
)
```

List available architecture names with:

```python
from quantbayes.stochax.vision_common import available_arches

print(available_arches("vit"))
print(available_arches("swin"))
print(available_arches("dino"))
```

## 2. Load pretrained checkpoints

### Smallest explicit path

```python
from quantbayes.stochax.vision_common import load_pretrained

model = load_pretrained(
    model,
    family="vit",
    npz_path="vit_b_16_imagenet.npz",
    strict_fc=False,
)
```

Use `strict_fc=False` whenever the checkpoint head shape does not match your downstream head.

### One-call construction + loading

```python
from quantbayes.stochax.vision_common import make_and_load_model

model, state = make_and_load_model(
    family="vit",
    arch="vit_b_16",
    variant="svd",
    image_size=224,
    num_classes=10,
    npz_path="vit_b_16_imagenet.npz",
    strict_fc=False,
    svd_mode="attn_mlp",
    svd_rank_ratio=0.25,
    key=jr.PRNGKey(0),
)
```

The standard checkpoint name for a family/architecture pair is available through:

```python
from quantbayes.stochax.vision_common import default_checkpoint_name

print(default_checkpoint_name("vit", "vit_b_16"))
print(default_checkpoint_name("swin", "swin_t"))
print(default_checkpoint_name("dino", "vitb14", registers=False))
```

## 3. Direct SVDDense workflow

This is the cleanest way to say: instantiate an SVD model first, then warm-start it from a dense checkpoint.

```python
model, state = make_model(
    family="vit",
    arch="vit_b_16",
    variant="svd",
    image_size=224,
    num_classes=10,
    svd_mode="attn_mlp",
    svd_rank_ratio=0.25,
    key=jr.PRNGKey(0),
)

model = load_pretrained(
    model,
    family="vit",
    npz_path="vit_b_16_imagenet.npz",
    strict_fc=False,
)
```

This works because the pretrained loaders detect `SVDDense` leaves and warm-start them by SVD.

## 4. Dense -> train -> replace with SVDDense -> fine-tune again

This is the missing workflow that is now exposed directly.

### Step A: train a dense model normally

```python
model, state = make_model(
    family="vit",
    arch="vit_b_16",
    variant="dense",
    image_size=224,
    num_classes=10,
    key=jr.PRNGKey(0),
)

# train model on your dataset D
```

### Step B: retrofit selected dense linears into `SVDDense`

```python
from quantbayes.stochax.vision_common import replace_attention_linears_with_svd

model_svd, report = replace_attention_linears_with_svd(
    model,
    family="vit",
    mode="attn_mlp",   # or "attn_only" / "all_linear"
    rank_ratio=1.0,     # 1.0 = exact transplant; <1.0 = truncated SVD
)

print(report["n_replaced"])
print(report["replaced"][:3])
```

For a completely generic rewrite of every `eqx.nn.Linear` leaf, use:

```python
from quantbayes.stochax.vision_common import replace_linears_with_svd

model_svd, report = replace_linears_with_svd(model, rank_ratio=1.0)
```

### Step C: freeze the bases and fine-tune again

If you want to keep the learned singular vectors fixed and continue training only the singular values and biases inside each `SVDDense`, build an Optax freeze mask:

```python
import optax

from quantbayes.stochax.utils.optim_util import OptimizerConfig, build_optimizer
from quantbayes.stochax.vision_common import make_svd_basis_freeze_mask

freeze_mask = make_svd_basis_freeze_mask(model_svd, freeze_alpha=True)

optimizer, opt_state, aux = build_optimizer(
    model_svd,
    OptimizerConfig(algorithm="adamw", lr=3e-4, weight_decay=0.0),
    prepend=optax.masked(optax.set_to_zero(), freeze_mask),
)
```

`make_svd_basis_freeze_mask(...)` freezes `U` and `V` (and optionally `alpha_raw`) but leaves the rest trainable.

If you want the strict "train only `s`" regime:

```python
from quantbayes.stochax.vision_common import make_s_only_freeze_mask

freeze_mask = make_s_only_freeze_mask(
    model_svd,
    train_bias=False,
    train_alpha=False,
)
```

That matches the literal theory statement "freeze the orthonormal bases and optimize only the singular values".

## 5. Pretrained dense -> fine-tune -> convert -> second-stage fine-tune

This is the two-stage transfer-learning workflow:

```python
model, state = make_and_load_model(
    family="vit",
    arch="vit_b_16",
    variant="dense",
    image_size=224,
    num_classes=10,
    npz_path="vit_b_16_imagenet.npz",
    strict_fc=False,
    key=jr.PRNGKey(0),
)

# stage 1: fine-tune the dense model on dataset D

model_svd, report = replace_attention_linears_with_svd(
    model,
    family="vit",
    mode="attn_mlp",
    rank_ratio=1.0,
)

freeze_mask = make_svd_basis_freeze_mask(model_svd, freeze_alpha=True)

# stage 2: fine-tune again with U,V frozen
```

The same pattern works for:

- `family="swin", arch="swin_t" | "swin_s" | "swin_b"`
- `family="dino", arch="vits14" | "vitb14" | "vitl14" | "vitg14"`

Only the checkpoint name and constructor arguments change.

## 6. Exactness vs truncation

For a dense -> SVD replacement, there are two regimes.

### Exact warm-start

Use full rank:

```python
rank_ratio=1.0
```

or pass `rank=min(out_features, in_features)`.

At initialization this reproduces the dense weight exactly up to numerical precision.

### Truncated warm-start

Use

```python
rank_ratio < 1.0
```

or a smaller explicit `rank`.

Then the initial `SVDDense` layer is the best rank-`r` approximation in the least-squares Frobenius sense, not an exact copy of the dense layer.

## 7. Recommended public API surface

For day-to-day use, treat `quantbayes.stochax.vision_common` as the only public import path.

- Construction: `make_model`, `make_and_load_model`
- Checkpoints: `load_pretrained`, `default_checkpoint_name`
- Dense -> SVD surgery: `replace_linears_with_svd`, `replace_attention_linears_with_svd`
- Theory-aligned freezing: `make_svd_basis_freeze_mask`, `make_s_only_freeze_mask`

That gives you one centralized entry point without requiring extra project-specific scripts.
