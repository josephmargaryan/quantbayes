# Vision pretrained loaders, embeddings, and transformer checkpoint recipes

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
- ViT / RFFT-ViT / SVD-ViT / DINO: `embed_dim`
- Swin-T / Swin-S: `768`
- Swin-B: `1024`
- VGG: `4096`

## Explicit structured transformer variants

In addition to the standard dense models, the library supports explicit
structured variants of selected transformer families.

### ViT
- `VisionTransformer`
- `RFFTVisionTransformer`
- `SVDVisionTransformer`

### Swin
- `SwinTransformer`
- `RFFTSwinTransformer`
- `SVDSwinTransformer`

### DINO
- `DinoVisionTransformer`
- `RFFTDinoVisionTransformer`
- `SVDDinoVisionTransformer`

These variants use the same:

- training API
- inference API
- pretrained-loading API
- embedding extraction API

The workflow is unchanged:

1. instantiate the model class you want,
2. optionally load pretrained weights with `load_pretrained(...)`,
3. use `train(...)`, `predict(...)`, or `extract_embeddings(...)` as usual.

The checkpoint family still follows the base architecture:

- ViT variants use `family="vit"`
- Swin variants use `family="swin"`
- DINO variants use `family="dino"`

### Variant-specific constructor notes

The downstream API is the same, but the model class may take extra variant
arguments:

- RFFT variants typically expose `use_spectral_proj=True` or equivalent.
- SVD variants typically expose arguments such as `svd_mode` and
  `svd_rank_ratio`.

Example idea:

```python
model, state = eqx.nn.make_with_state(SVDVisionTransformer)(
    ..., 
    svd_mode="attn_mlp",
    svd_rank_ratio=0.25,
)

model = load_pretrained(model, family="vit", strict_fc=False)
```

## Checkpoint download recipes for ViT, Swin, and DINOv2

This is the part that is easiest to get wrong. The checkpoint filename, model
architecture, and any register-token setting must match.

### ViT checkpoints

Your ViT download script saves torchvision ViT checkpoints as `.npz` files.

Run:

```bash
python scripts/save_imagenet_vits.py
```

This script downloads **all configured ViT variants** and writes:

- `vit_b_16_imagenet.npz`
- `vit_b_32_imagenet.npz`
- `vit_l_16_imagenet.npz`
- `vit_l_32_imagenet.npz`
- `vit_h_14_imagenet.npz`

These correspond to:

- `vit_b_16`
- `vit_b_32`
- `vit_l_16`
- `vit_l_32`
- `vit_h_14`

Example loading call once you have instantiated a matching Equinox ViT model:

```python
from quantbayes.stochax.vision_common import load_pretrained_vit

model = load_pretrained_vit(
    model,
    "vit_b_16_imagenet.npz",
    strict_fc=False,
)
```

Use `strict_fc=False` when your downstream `num_classes` is not the original
classifier head size.

### Swin checkpoints

Your Swin download script saves torchvision Swin checkpoints as `.npz` files.

Examples:

```bash
python quantbayes/stochax/save_imagenet_swins.py --arch swin_t
python quantbayes/stochax/save_imagenet_swins.py --arch swin_s
python quantbayes/stochax/save_imagenet_swins.py --arch swin_b
```

These write:

- `swin_t_imagenet.npz`
- `swin_s_imagenet.npz`
- `swin_b_imagenet.npz`

Example loading call:

```python
from quantbayes.stochax.vision_common import load_pretrained_swin

model = load_pretrained_swin(
    model,
    "swin_t_imagenet.npz",
    strict_fc=False,
)
```

### DINOv2 checkpoints

Your DINO script downloads **official DINOv2 backbones** from PyTorch Hub and
saves them as `.npz` files for the Equinox DINO loader.

Examples without registers:

```bash
python quantbayes/stochax/save_dinov2_from_torch.py --arch vits14
python quantbayes/stochax/save_dinov2_from_torch.py --arch vitb14
python quantbayes/stochax/save_dinov2_from_torch.py --arch vitl14
python quantbayes/stochax/save_dinov2_from_torch.py --arch vitg14
```

Examples with registers:

```bash
python quantbayes/stochax/save_dinov2_from_torch.py --arch vits14 --registers
python quantbayes/stochax/save_dinov2_from_torch.py --arch vitb14 --registers
python quantbayes/stochax/save_dinov2_from_torch.py --arch vitl14 --registers
python quantbayes/stochax/save_dinov2_from_torch.py --arch vitg14 --registers
```

These write filenames like:

- `dinov2_vits14_noreg.npz`
- `dinov2_vitb14_noreg.npz`
- `dinov2_vitl14_noreg.npz`
- `dinov2_vitg14_noreg.npz`
- `dinov2_vits14_reg.npz`
- `dinov2_vitb14_reg.npz`
- `dinov2_vitl14_reg.npz`
- `dinov2_vitg14_reg.npz`

### What `--registers` means

DINOv2 has two variants:

- **without registers**: standard CLS + patch tokens,
- **with registers**: extra learned register tokens are inserted into the token
  sequence.

In your Equinox model this corresponds to:

- no registers: `n_register_tokens=0`
- with registers: `n_register_tokens=4`

This setting must match the checkpoint.

If you save a `*_reg.npz` checkpoint, instantiate the DINO model with
`n_register_tokens=4`.
If you save a `*_noreg.npz` checkpoint, instantiate the DINO model with
`n_register_tokens=0`.

### DINOv2 config helper

Unlike ResNet or Swin, the DINO constructor does not use a simple `arch="..."`
argument. It takes the low-level transformer dimensions, so it helps to use a
small config map.

```python
import equinox as eqx
import jax.random as jr

from quantbayes.stochax.vision_backbones.dino.vit_dino_eqx import DinoVisionTransformer
from quantbayes.stochax.vision_common import load_pretrained

DINO_CONFIGS = {
    "vits14": dict(embedding_dim=384, hidden_dim=1536, num_heads=6,  num_layers=12),
    "vitb14": dict(embedding_dim=768, hidden_dim=3072, num_heads=12, num_layers=12),
    "vitl14": dict(embedding_dim=1024, hidden_dim=4096, num_heads=16, num_layers=24),
    "vitg14": dict(embedding_dim=1536, hidden_dim=6144, num_heads=24, num_layers=40),
}


def make_dino_model(
    arch: str,
    *,
    image_size: int = 224,
    num_classes: int = 10,
    registers: bool = False,
    key=jr.PRNGKey(0),
):
    cfg = DINO_CONFIGS[arch]
    patch_size = 14
    num_patches = (image_size // patch_size) ** 2

    model, state = eqx.nn.make_with_state(DinoVisionTransformer)(
        **cfg,
        patch_size=patch_size,
        num_patches=num_patches,
        num_classes=num_classes,
        n_register_tokens=4 if registers else 0,
        dropout_rate=0.0,
        pool="cls",
        key=key,
    )
    return model, state
```

Example load:

```python
model, state = make_dino_model(
    "vitb14",
    image_size=224,
    num_classes=10,
    registers=True,
    key=jr.PRNGKey(0),
)

model = load_pretrained(
    model,
    family="dino",
    npz_path="dinov2_vitb14_reg.npz",
    strict_fc=False,
)
```

The same family rule applies to explicit DINO variants too:

```python
# Dense DINO
model = load_pretrained(model, family="dino", npz_path="dinov2_vitb14_reg.npz", strict_fc=False)

# RFFT DINO
model = load_pretrained(model, family="dino", npz_path="dinov2_vitb14_reg.npz", strict_fc=False)

# SVD DINO
model = load_pretrained(model, family="dino", npz_path="dinov2_vitb14_reg.npz", strict_fc=False)
```

### Important DINO note

The DINOv2 script saves **self-supervised backbone weights**, not an ImageNet
classifier head that should match your downstream `num_classes`.

So for downstream classification or transfer learning, the usual choice is:

```python
strict_fc=False
```

That keeps the backbone/features and leaves your downstream head free to train.

## Matching rules that avoid loading issues

### ViT

Make sure these match:

- checkpoint variant, e.g. `vit_b_16_imagenet.npz`
- model width/depth/head count for `vit_b_16`
- patch size, e.g. `16` for `vit_b_16`
- image size divisible by patch size

### Swin

Make sure these match:

- checkpoint variant, e.g. `swin_t_imagenet.npz`
- model `arch`, e.g. `arch="swin_t"`

### DINOv2

Make sure these match:

- checkpoint variant, e.g. `vitb14`
- register-token setting (`*_reg.npz` vs `*_noreg.npz`)
- model width/depth/head count for the chosen DINO architecture
- patch size `14`
- image size divisible by `14`

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
    strict_fc=False,
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

## Loading report / coverage

Use `load_pretrained(...)` by default.

For DINO specifically, use the lower-level loader only when you want diagnostics
such as load coverage, unused keys, or spectral warm-start reports.

```python
from quantbayes.stochax.vision_backbones.dino.dinov2_loader import load_dinov2

model, report = load_dinov2(
    model,
    "dinov2_vitb14_reg.npz",
    strict_fc=False,
    spectral_warmstart="skip",
    verbose=True,
)

print(f"coverage: {100 * report['coverage']:.2f}%")
print(f"loaded params: {report['n_loaded']}")
print(f"total mapped params: {report['n_total_pt']}")
print(f"unused mapped keys: {len(report['unused_keys'])}")
```

Use this only when you need loader diagnostics. For normal usage, prefer:

```python
model = load_pretrained(
    model,
    family="dino",
    npz_path="dinov2_vitb14_reg.npz",
    strict_fc=False,
)
```

## Classifier heads and transfer learning

- Keep `strict_fc=True` when checkpoint and classifier head should match.
- Use `strict_fc=False` when `num_classes != 1000` or when loading a DINOv2
  backbone for a downstream task.

## Standard checkpoint names

Common filenames in this repo are:

- `resnet18_imagenet.npz`
- `vit_b_16_imagenet.npz`
- `swin_t_imagenet.npz`
- `convnext_tiny_imagenet.npz`
- `dinov2_vitb14_noreg.npz`
- `dinov2_vitb14_reg.npz`
