# Vision pretrained loaders and embeddings

This directory exposes a small public API for two production paths:

1. **load pretrained weights** into the model you actually want,
2. **extract embeddings** from the penultimate representation.

The normal classifier API is unchanged. You can still train and predict exactly
as before.

## Public entry points

```python
from quantbayes.stochax.vision_common import (
    load_pretrained,
    load_pretrained_resnet,
    load_pretrained_vit,
    load_pretrained_swin,
    infer_pretrained_family,
    default_pretrained_checkpoint,
    as_feature_extractor,
    extract_embeddings,
    extract_embeddings_batched,
    infer_embedding_dim,
)
```

## Load pretrained weights

### Smallest plain usage

```python
import equinox as eqx
import jax.random as jr

from quantbayes.stochax.vision_classification.models import ResNetClassifier
from quantbayes.stochax.vision_common import load_pretrained

model, state = eqx.nn.make_with_state(ResNetClassifier)(
    backbone="resnet18",
    num_classes=10,
    key=jr.PRNGKey(0),
)

model = load_pretrained(
    model,
    npz_path="resnet18_imagenet.npz",
    strict_fc=False,
)
```

If the checkpoint name is standard and the model stores enough metadata, you
can omit `npz_path`:

```python
model = load_pretrained(model, strict_fc=False)
```

That works for model families where the variant is explicit on the instance, for
example ResNet (`backbone="resnet18"`) or Swin (`arch="swin_t"`).

## Embeddings

For a pretrained classifier, the helper returns the penultimate feature vector
`phi(x) in R^d` for an input image `x in R^{C x H x W}`.

### Direct embedding extraction

```python
import jax.random as jr

from quantbayes.stochax.vision_common import extract_embeddings

Z = extract_embeddings(
    model,
    state,
    X,                  # [B, C, H, W]
    key=jr.PRNGKey(123),
)
print(Z.shape)         # [B, d]
```

To L2-normalize the embeddings:

```python
Z = extract_embeddings(model, state, X, key=jr.PRNGKey(123), l2_normalize=True)
```

For large datasets:

```python
Z = extract_embeddings(
    model,
    state,
    X,
    key=jr.PRNGKey(123),
    batch_size=256,
)
```

### Reuse your normal `predict(...)` flow

```python
from quantbayes.stochax.vision_common import as_feature_extractor
from quantbayes.stochax.trainer.train import predict

feature_model = as_feature_extractor(model)
Z = predict(feature_model, state, X, jr.PRNGKey(123))
```

### Infer embedding size

```python
from quantbayes.stochax.vision_common import infer_embedding_dim

print(infer_embedding_dim(model))
```

Typical values:

- ResNet18/34: `512`
- ResNet50/101/152: `2048`
- ViT / RFFT-ViT / DINO: `embed_dim`
- Swin-T / Swin-S: `768`
- Swin-B: `1024`
- VGG: `4096`

## Family-specific helpers

Use the dedicated helpers when you want the call-site to stay explicit:

```python
from quantbayes.stochax.vision_common import (
    load_pretrained_resnet,
    load_pretrained_vit,
    load_pretrained_swin,
)
```

### ResNet

```python
model = load_pretrained_resnet(
    model,
    "resnet18_imagenet.npz",
    strict_fc=False,
)
```

### ViT

```python
model = load_pretrained_vit(
    model,
    "vit_b_16_imagenet.npz",
    strict_fc=True,
)
```

### Swin

```python
model = load_pretrained_swin(
    model,
    "swin_t_imagenet.npz",
    strict_fc=False,
)
```

## What is supported

The plain loader supports:

- dense models,
- models that explicitly use `SVDDense`,
- models that explicitly use `RFFTCirculant1D`.

That means the supported spectral path is now **explicit model design**, not
retrofitting arbitrary CNNs at load time.

## Classifier heads and transfer learning

- Keep `strict_fc=True` when checkpoint and classifier head should match.
- Use `strict_fc=False` when `num_classes != 1000` and you only want the
  backbone/features loaded.

## Standard checkpoint names

The code assumes torchvision-style `.npz` files such as:

- `resnet18_imagenet.npz`
- `vit_b_16_imagenet.npz`
- `swin_t_imagenet.npz`
- `convnext_tiny_imagenet.npz`

## Removed experimental path

The old layer-surgery utilities that rewrote models into spectral CNNs at load time have been removed from the public package surface. The recommended path is now:

- keep dense CNNs dense,
- use explicit RFFT/SVD model variants where they empirically make sense,
- load weights with `load_pretrained(...)`,
- extract features with `extract_embeddings(...)` when needed.
