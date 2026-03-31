# Vision pretrained loaders

This directory now exposes a **plain pretrained-loading API** for your vision
models. The common path is:

1. instantiate the model you actually want,
2. load a pretrained `.npz` checkpoint into it,
3. train or finetune.

You do **not** need to go through `replace_layers_api` unless you are
intentionally changing the architecture.

## Public entry points

```python
from quantbayes.stochax.vision_common import (
    load_pretrained,
    load_pretrained_resnet,
    load_pretrained_vit,
    load_pretrained_swin,
    infer_pretrained_family,
    default_pretrained_checkpoint,
)
```

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

If the checkpoint filename is standard and the model stores enough metadata, you
can omit `npz_path`:

```python
model = load_pretrained(model, strict_fc=False)
```

That works for model families where the variant is explicit on the instance, for
example ResNet (`backbone="resnet18"`) or Swin (`arch="swin_t"`).

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
from quantbayes.stochax.vision_common import load_pretrained_resnet

model = load_pretrained_resnet(
    model,
    "resnet18_imagenet.npz",
    strict_fc=False,
)
```

### ViT

```python
from quantbayes.stochax.vision_common import load_pretrained_vit

model = load_pretrained_vit(
    model,
    "vit_b_16_imagenet.npz",
    strict_fc=True,
)
```

### Swin

```python
from quantbayes.stochax.vision_common import load_pretrained_swin

model = load_pretrained_swin(
    model,
    "swin_t_imagenet.npz",
    strict_fc=False,
)
```

## Spectral models: what is still supported

The plain loader can still warm-start the spectral models that made empirical
sense in practice:

- **SVDDense** leaves in ViT / Swin / DINO-style models,
- **RFFTCirculant1D** leaves in `rfft_vit.py`,
- **RFFTCirculant1D** leaves in `rfft_swin.py`.

The loader detects those leaves automatically and applies the appropriate
warm-start when possible.

## About classifier heads

- Keep `strict_fc=True` when the checkpoint and model classifier head should
  match exactly.
- Use `strict_fc=False` for transfer learning when `num_classes != 1000` and you
  only want the backbone/features loaded.

## Naming convention

The code assumes torchvision-style `.npz` files such as:

- `resnet18_imagenet.npz`
- `vit_b_16_imagenet.npz`
- `swin_t_imagenet.npz`
- `convnext_tiny_imagenet.npz`

## Where the experimental API lives

The spectral surgery entry points still exist in:

```python
quantbayes.stochax.vision_common.replace_layers_api
```

Use that module only when you explicitly want to **replace layers** before
loading weights.
