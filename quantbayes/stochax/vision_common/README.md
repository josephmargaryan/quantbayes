# Vision transformers: dense, SVDDense, and RFFT1D

This directory now exposes the two retrofit workflows discussed in the project:

1. instantiate a model,
2. optionally load a pretrained checkpoint,
3. optionally replace dense linear layers with either `SVDDense` or `RFFTCirculant1D`,
4. continue fine-tuning from that warm-started model.

In particular, the previously missing workflow

- pretrain / fine-tune a **dense** model,
- replace selected dense linears with **RFFT1D**,
- fine-tune again,

is now available through the public `vision_common` API.

## Public import surface

```python
from quantbayes.stochax.vision_common import (
    available_arches,
    default_checkpoint_name,
    make_model,
    make_and_load_model,
    load_pretrained,
    replace_linears_with_svd,
    replace_attention_linears_with_svd,
    replace_square_linears_with_rfft,
    replace_attention_linears_with_rfft,
    make_svd_basis_freeze_mask,
    make_s_only_freeze_mask,
)
```

## Supported attention families

The factory API supports the three attention families in this library:

- `family="vit"`
- `family="swin"`
- `family="dino"`

Each family exposes three constructor variants:

- `variant="dense"`
- `variant="svd"`
- `variant="rfft"`

For attention-family retrofits, use `mode` in `{"attn_only", "attn_mlp", "all_linear"}`.

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
model, state = make_model(
    family="swin",
    arch="swin_t",
    variant="rfft",
    num_classes=10,
    key=jr.PRNGKey(0),
)

model, state = make_model(
    family="dino",
    arch="vitb14",
    variant="svd",
    image_size=224,
    num_classes=10,
    registers=False,
    svd_mode="attn_mlp",
    svd_rank_ratio=0.25,
    key=jr.PRNGKey(0),
)
```

List available architectures with:

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
    variant="rfft",
    image_size=224,
    num_classes=10,
    npz_path="vit_b_16_imagenet.npz",
    strict_fc=False,
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

## 3. Direct RFFT workflow

This is the cleanest way to say: instantiate an RFFT model first, then warm-start it from a dense checkpoint.

```python
model, state = make_model(
    family="vit",
    arch="vit_b_16",
    variant="rfft",
    image_size=224,
    num_classes=10,
    key=jr.PRNGKey(0),
)

model = load_pretrained(
    model,
    family="vit",
    npz_path="vit_b_16_imagenet.npz",
    strict_fc=False,
)
```

This works because the pretrained loaders already detect `RFFTCirculant1D` leaves and warm-start them by projecting the corresponding dense checkpoint weight onto the nearest circulant matrix in Frobenius norm.

## 4. Dense -> train -> replace with RFFT -> fine-tune again

This is the newly documented retrofit workflow.

### Step A: train or fine-tune a dense model normally

```python
model, state = make_model(
    family="vit",
    arch="vit_b_16",
    variant="dense",
    image_size=224,
    num_classes=10,
    key=jr.PRNGKey(0),
)

# train or fine-tune model on your dataset D
```

### Step B: retrofit selected dense linears into `RFFTCirculant1D`

Use the attention-family helper when you want a clean architecture-aware rewrite.

```python
from quantbayes.stochax.vision_common import replace_attention_linears_with_rfft

model_rfft, report = replace_attention_linears_with_rfft(
    model,
    family="vit",
    mode="attn_only",     # or "attn_mlp" / "all_linear"
    warmstart=True,
)

print(report["n_replaced"])
print(report["replaced"][:3])
print(report["skipped"][:3])
```

For a completely generic rewrite of every eligible dense linear leaf, use:

```python
from quantbayes.stochax.vision_common import replace_square_linears_with_rfft

model_rfft, report = replace_square_linears_with_rfft(
    model,
    warmstart=True,
)
```

### What `warmstart=True` does

For each square dense weight matrix `W`, the helper:

1. computes the nearest circulant matrix in Frobenius norm,
2. extracts its first column,
3. converts that first column into the stored RFFT half-spectrum,
4. copies the dense bias exactly when the original dense layer has a vector bias.

So this is not random replacement: it is a deterministic dense -> circulant projection followed by RFFT reparameterization.

### Step C: fine-tune the retrofitted model

No special freeze mask is required for the standard RFFT workflow. Just build your optimizer and continue training.

```python
from quantbayes.stochax.utils.optim_util import OptimizerConfig, build_optimizer

optimizer, opt_state, aux = build_optimizer(
    model_rfft,
    OptimizerConfig(
        algorithm="adamw",
        lr=3e-4,
        weight_decay=0.05,
    ),
)
```

If you want random RFFT replacements instead of warm-starting from the dense model, pass `warmstart=False` and provide a PRNG key:

```python
model_rfft, report = replace_attention_linears_with_rfft(
    model,
    family="vit",
    mode="attn_only",
    warmstart=False,
    key=jr.PRNGKey(42),
)
```

## 5. Pretrained dense -> fine-tune -> convert -> second-stage fine-tune (RFFT)

This is the two-stage transfer-learning workflow that is usually most useful in practice.

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

model_rfft, report = replace_attention_linears_with_rfft(
    model,
    family="vit",
    mode="attn_only",
    warmstart=True,
)

# stage 2: fine-tune again from the dense -> circulant warm-start
```

The same pattern works for:

- `family="swin", arch="swin_t" | "swin_s" | "swin_b"`
- `family="dino", arch="vits14" | "vitb14" | "vitl14" | "vitg14"`

Only the checkpoint name and constructor arguments change.

## 6. Architecture-specific notes for RFFT retrofits

### ViT / DINO

In the current models, the attention projections

- `q_proj`
- `k_proj`
- `v_proj`
- `out_proj`

are square, so `mode="attn_only"` is the natural RFFT retrofit mode.

`mode="attn_mlp"` and `mode="all_linear"` are allowed, but many MLP projections are not square and will therefore be left unchanged.

### Swin

In Swin, `proj` is square but `qkv` has shape `(3d, d)` and is therefore **not** eligible for `RFFTCirculant1D` replacement.

So for Swin, the retrofit report is important:

```python
model_rfft, report = replace_attention_linears_with_rfft(
    model,
    family="swin",
    mode="attn_only",
    warmstart=True,
)

print(report["replaced"])  # typically attention proj leaves
print(report["skipped"])   # includes qkv as non-square
```

## 7. Direct SVDDense workflow

This remains unchanged. Instantiate an SVD model first, then warm-start it from a dense checkpoint.

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

## 8. Dense -> train -> replace with SVDDense -> fine-tune again

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

## 9. Exactness vs truncation

### Dense -> SVD

For a dense -> SVD replacement, there are two regimes.

#### Exact warm-start

Use full rank:

```python
rank_ratio=1.0
```

or pass `rank=min(out_features, in_features)`.

At initialization this reproduces the dense weight exactly up to numerical precision.

#### Truncated warm-start

Use

```python
rank_ratio < 1.0
```

or a smaller explicit `rank`.

Then the initial `SVDDense` layer is the best rank-`r` approximation in the least-squares Frobenius sense, not an exact copy of the dense layer.

### Dense -> RFFT

The RFFT replacement is **not** an exact dense copy in general. It is the nearest circulant approximation, reparameterized in the RFFT half-spectrum.

So the two-stage RFFT workflow should be read as

- dense fine-tune,
- project selected square dense layers onto the circulant subspace,
- continue fine-tuning inside that structured subspace.

## 10. Recommended public API surface

For day-to-day use, treat `quantbayes.stochax.vision_common` as the public import path.

- Construction: `make_model`, `make_and_load_model`
- Checkpoints: `load_pretrained`, `default_checkpoint_name`
- Dense -> SVD surgery: `replace_linears_with_svd`, `replace_attention_linears_with_svd`
- Dense -> RFFT surgery: `replace_square_linears_with_rfft`, `replace_attention_linears_with_rfft`
- Theory-aligned freezing: `make_svd_basis_freeze_mask`, `make_s_only_freeze_mask`

That gives you one centralized entry point without requiring extra project-specific scripts.
