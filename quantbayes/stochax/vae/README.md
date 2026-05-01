# VAE spectral workflows

Put this file at:

```text
quantbayes/stochax/vae/README.md
```

This README documents the intended end-to-end workflow for running VAE experiments with:

- ordinary dense models,
- dense models retrofitted to `SVDDense`,
- warm-start dense models converted to `SVDDense` and fine-tuned with only `s` trainable,
- dense models retrofitted to `RFFTCirculant1D` on square linear maps,
- optional spectral regularization.

The base VAE usage already exists in:

- `quantbayes/stochax/vae/test.py`

This README extends that end-to-end pattern to the spectral variants.

## What is assumed

This README assumes the spectral retrofit utilities have been added:

- `quantbayes/stochax/utils/linear_surgery.py`
- `quantbayes/stochax/vae/workflows.py`

Unlike diffusion, the VAE trainer needs a small patch if you want the full workflow symmetry:

- `optimizer=...` for warm-start `s`-only fine-tuning,
- `extra_loss_fn=...` for spectral regularization.

If you only want dense training or direct retrofit followed by ordinary training, the stock trainer is enough. If you want warm-start `s`-only fine-tuning and regularization, apply the trainer patch below.

## Minimal end-to-end example

```python
from __future__ import annotations

import optax
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.vae.components import ViT_VAE
from quantbayes.stochax.vae.train_vae import TrainConfig, train_vae

from quantbayes.stochax.vae.workflows import retrofit_vae_model
from quantbayes.stochax.utils.linear_surgery import make_s_only_freeze_mask
from quantbayes.stochax.utils.optim_util import OptimizerConfig, build_optimizer
from quantbayes.stochax.utils.regularizers import global_spectral_norm_penalty


def build_dense_vae(key):
    return ViT_VAE(
        image_size=28,
        channels=1,
        patch_size=4,
        embedding_dim=256,
        num_layers=2,
        num_heads=4,
        latent_dim=32,
        dropout_rate=0.1,
        key=key,
    )


def make_cfg():
    return TrainConfig(
        epochs=3,
        batch_size=16,
        learning_rate=1e-3,
        likelihood="gaussian",
        beta_warmup_steps=2000,
    )


def make_s_only_optimizer(model, lr=1e-4):
    freeze_mask = make_s_only_freeze_mask(
        model,
        train_bias=False,
        train_alpha=False,
    )
    tx, _, _ = build_optimizer(
        model,
        OptimizerConfig(
            algorithm="adamw",
            lr=lr,
            weight_decay=0.0,
            clip_global_norm=1.0,
        ),
        prepend=optax.masked(optax.set_to_zero(), freeze_mask),
    )
    return tx


def reconstruct(model, x, *, seed=0):
    recon, mu, logvar = model(x, jr.PRNGKey(seed), train=False)
    print("recon shape:", recon.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
    return recon, mu, logvar


def vae_reg(model):
    return 1e-5 * global_spectral_norm_penalty(model)


def main():
    # Replace this with real NCHW image data if desired.
    X = jr.uniform(jr.PRNGKey(0), shape=(128, 1, 28, 28))

    # ------------------------------------------------------------
    # 1) Dense baseline
    # ------------------------------------------------------------
    dense = build_dense_vae(jr.PRNGKey(1))
    dense = train_vae(dense, X, make_cfg())
    reconstruct(dense, X[:8], seed=10)

    # ------------------------------------------------------------
    # 2) SVD from scratch
    # ------------------------------------------------------------
    svd_model, svd_report = retrofit_vae_model(
        build_dense_vae(jr.PRNGKey(2)),
        variant="svd",
        mode="all_linear",
    )
    print("svd_report:", svd_report)
    svd_model = train_vae(svd_model, X, make_cfg())
    reconstruct(svd_model, X[:8], seed=20)

    # ------------------------------------------------------------
    # 3) Warm-start dense -> exact SVD transplant -> fine-tune only s
    # ------------------------------------------------------------
    warm_svd_model, warm_report = retrofit_vae_model(
        dense,
        variant="svd",
        mode="all_linear",
    )
    print("warm_report:", warm_report)

    s_only_tx = make_s_only_optimizer(warm_svd_model, lr=1e-4)
    warm_cfg = make_cfg()
    warm_cfg.epochs = 2
    warm_svd_model = train_vae(
        warm_svd_model,
        X,
        warm_cfg,
        optimizer=s_only_tx,
    )
    reconstruct(warm_svd_model, X[:8], seed=30)

    # ------------------------------------------------------------
    # 4) RFFT retrofit on square linear layers only
    # ------------------------------------------------------------
    rfft_model, rfft_report = retrofit_vae_model(
        build_dense_vae(jr.PRNGKey(3)),
        variant="rfft",
        mode="all_linear",
        warmstart=True,
        key=jr.PRNGKey(4),
    )
    print("rfft_report:", rfft_report)
    rfft_model = train_vae(rfft_model, X, make_cfg())
    reconstruct(rfft_model, X[:8], seed=40)

    # ------------------------------------------------------------
    # 5) Optional: spectral regularization
    # ------------------------------------------------------------
    reg_model, reg_report = retrofit_vae_model(
        build_dense_vae(jr.PRNGKey(5)),
        variant="svd",
        mode="all_linear",
    )
    print("reg_report:", reg_report)
    reg_model = train_vae(
        reg_model,
        X,
        make_cfg(),
        extra_loss_fn=vae_reg,
    )
    reconstruct(reg_model, X[:8], seed=50)


if __name__ == "__main__":
    main()
```

## How to adapt the example to a real dataset

Replace the synthetic array:

```python
X = jr.uniform(jr.PRNGKey(0), shape=(128, 1, 28, 28))
```

with your actual dataset tensor.

Common shapes are:

- tabular: `(N, D)`
- sequence: `(N, T, D)`
- image: `(N, C, H, W)`

The rest of the flow stays the same:

1. instantiate a base VAE,
2. optionally retrofit,
3. call `train_vae(...)`,
4. call the model on a batch with `train=False` for reconstructions.

## Retrofitting modes

The VAE workflow exposes:

- `mode="encoder_only"`
- `mode="decoder_only"`
- `mode="attention_only"`
- `mode="all_linear"`

Use:

- `encoder_only` to replace only encoder linears,
- `decoder_only` to replace only decoder linears,
- `attention_only` to target attention projections,
- `all_linear` to target every dense-like linear layer.

## Dense vs SVD vs warm-start vs RFFT

### Dense

```python
model = build_dense_vae(jr.PRNGKey(0))
model = train_vae(model, X, cfg)
recon, mu, logvar = model(X[:8], jr.PRNGKey(1), train=False)
```

### SVD from scratch

```python
model, report = retrofit_vae_model(build_dense_vae(jr.PRNGKey(0)), variant="svd")
model = train_vae(model, X, cfg)
```

### Warm-start SVD

```python
dense = train_vae(build_dense_vae(jr.PRNGKey(0)), X, cfg)
svd_model, report = retrofit_vae_model(dense, variant="svd")
s_only_tx = make_s_only_optimizer(svd_model)
svd_model = train_vae(svd_model, X, cfg, optimizer=s_only_tx)
```

### RFFT

```python
model, report = retrofit_vae_model(
    build_dense_vae(jr.PRNGKey(0)),
    variant="rfft",
    warmstart=True,
    key=jr.PRNGKey(1),
)
model = train_vae(model, X, cfg)
```

## About regularization

Once the trainer patch is applied, VAE regularization is straightforward.

Example:

```python
from quantbayes.stochax.utils.regularizers import global_spectral_norm_penalty


def extra_loss_fn(model):
    return 1e-5 * global_spectral_norm_penalty(model)


model = train_vae(
    model,
    X,
    cfg,
    extra_loss_fn=extra_loss_fn,
)
```

You can swap in other penalties, for example `global_frobenius_penalty(model)`, depending on whether you want to control the top singular mode or the whole spectrum.

## Which existing file to copy from

For the base VAE train/reconstruct idiom, copy from:

- `quantbayes/stochax/vae/test.py`

To add the spectral idea, insert:

1. a call to `retrofit_vae_model(...)`,
2. optionally a masked optimizer for `s`-only fine-tuning,
3. optionally `extra_loss_fn=...` for spectral regularization.

## Suggested experiment grid

A clean first ablation is:

1. Dense
2. Dense + spectral regularization
3. SVD from scratch
4. SVD from scratch + spectral regularization
5. Warm-start dense -> SVD, fine-tune only `s`
6. Warm-start dense -> SVD, fine-tune only `s` + regularization
7. RFFT
8. RFFT + regularization

That is the simplest way to separate:

- the effect of the parameterization,
- the effect of explicit norm control.
