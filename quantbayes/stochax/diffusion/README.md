# Diffusion spectral workflows

Put this file at:

```text
quantbayes/stochax/diffusion/README.md
```

This README documents the intended end-to-end workflow for running diffusion experiments with:

- ordinary dense models,
- dense models retrofitted to `SVDDense`,
- warm-start dense models converted to `SVDDense` and fine-tuned with only `s` trainable,
- dense models retrofitted to `RFFTCirculant1D` on square linear maps,
- optional spectral regularization.

The examples below are written to match the existing APIs already used in:

- `quantbayes/stochax/diffusion/testing/test_image.py`
- `quantbayes/stochax/diffusion/testing/test_tabular.py`
- `quantbayes/stochax/diffusion/testing/test_time_series.py`

## What is assumed

This README assumes the spectral retrofit utilities have been added:

- `quantbayes/stochax/utils/linear_surgery.py`
- `quantbayes/stochax/diffusion/workflows.py`

and that the layer imports expose `SVDDense` / `RFFTCirculant1D` as expected.

No trainer patch is required for diffusion. The existing trainer already supports:

- `optimizer=...` for masked or frozen-parameter fine-tuning,
- `custom_loss=...` for spectral regularization.

## Core idea

The workflow is:

1. build a normal diffusion model,
2. optionally retrofit its linear layers to `SVDDense` or `RFFTCirculant1D`,
3. train with `train_model(...)`,
4. sample with `sample_edm(...)` or `sample_edm_conditional(...)`.

For warm-start SVD experiments:

1. train a dense model,
2. retrofit the trained dense model to `SVDDense`,
3. freeze everything except `s`,
4. fine-tune.

## Minimal end-to-end image example

```python
from __future__ import annotations

import optax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from quantbayes.stochax.diffusion.dataloaders import (
    generate_synthetic_image_dataset,
    dataloader,
)
from quantbayes.stochax.diffusion.trainer import train_model
from quantbayes.stochax.diffusion.edm import edm_batch_loss
from quantbayes.stochax.diffusion import sample_edm, sample_edm_conditional
from quantbayes.stochax.diffusion.models.adaptive_DiT import DiT
from quantbayes.stochax.diffusion.models.wrappers import DiTWrapper

from quantbayes.stochax.diffusion.workflows import retrofit_diffusion_model
from quantbayes.stochax.utils.linear_surgery import make_s_only_freeze_mask
from quantbayes.stochax.utils.optim_util import OptimizerConfig, build_optimizer
from quantbayes.stochax.utils.regularizers import global_spectral_norm_penalty


def build_dense_dit(key):
    core = DiT(
        img_size=(1, 28, 28),
        patch_size=4,
        in_channels=1,
        embed_dim=192,
        depth=2,
        n_heads=6,
        mlp_ratio=4.0,
        dropout_rate=0.0,
        time_emb_dim=192,
        num_classes=10,
        learn_sigma=False,
        key=key,
    )
    return DiTWrapper(
        model=core,
        num_classes=10,
        time_mode="vp_t",
        null_label_index=None,
        cfg_rescale=0.7,
    )


def make_edm_loss(reg_strength: float = 0.0):
    def loss_fn(model, batch, key):
        base = edm_batch_loss(
            model,
            batch,
            key,
            sigma_data=0.5,
            rho_min=-1.2,
            rho_max=1.2,
            sample="uniform",
        )
        if reg_strength == 0.0:
            return base
        reg = reg_strength * global_spectral_norm_penalty(model)
        return base + reg

    return loss_fn


def train_edm(model, data, *, steps=100, seed=0, batch_size=32, optimizer=None, reg_strength=0.0):
    return train_model(
        model,
        dataset=data,
        t1=1.0,
        lr=3e-4,
        num_steps=steps,
        batch_size=batch_size,
        weight_fn=None,
        int_beta_fn=None,
        print_every=max(steps // 5, 1),
        seed=seed,
        data_loader_func=dataloader,
        loss_impl="edm",
        custom_loss=make_edm_loss(reg_strength=reg_strength),
        checkpoint_dir=None,
        optimizer=optimizer,
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


def sample_some(ema_model, *, seed=0, n=8):
    xs = sample_edm(
        ema_model=ema_model,
        num_samples=n,
        sample_shape=(1, 28, 28),
        key=jr.PRNGKey(seed),
        steps=20,
        sigma_min=0.002,
        sigma_max=1.0,
        sigma_data=0.5,
        rho=7.0,
    )
    print("sample shape:", xs.shape)
    return xs


def sample_conditional(ema_model, *, label_value=3, seed=0, n=8):
    labels = jnp.full((n,), label_value, dtype=jnp.int32)
    xs = sample_edm_conditional(
        ema_model=ema_model,
        label=labels,
        cfg_scale=3.0,
        num_samples=n,
        sample_shape=(1, 28, 28),
        key=jr.PRNGKey(seed),
        steps=20,
        sigma_min=0.002,
        sigma_max=1.0,
        sigma_data=0.5,
        rho=7.0,
    )
    print("conditional sample shape:", xs.shape)
    return xs


def main():
    data = generate_synthetic_image_dataset(
        num_samples=256,
        shape=(1, 28, 28),
        key=jr.PRNGKey(0),
    )

    # ------------------------------------------------------------
    # 1) Dense baseline
    # ------------------------------------------------------------
    dense = build_dense_dit(jr.PRNGKey(1))
    ema_dense = train_edm(dense, data, steps=100, seed=10)
    sample_some(ema_dense, seed=100)
    sample_conditional(ema_dense, seed=101)

    # ------------------------------------------------------------
    # 2) SVD from scratch
    # ------------------------------------------------------------
    svd_model, svd_report = retrofit_diffusion_model(
        build_dense_dit(jr.PRNGKey(2)),
        variant="svd",
        mode="all_linear",
    )
    print("svd_report:", svd_report)
    ema_svd = train_edm(svd_model, data, steps=100, seed=20)
    sample_some(ema_svd, seed=200)

    # ------------------------------------------------------------
    # 3) Warm-start dense -> exact SVD transplant -> fine-tune only s
    # ------------------------------------------------------------
    warm_svd_model, warm_report = retrofit_diffusion_model(
        ema_dense,
        variant="svd",
        mode="all_linear",
    )
    print("warm_report:", warm_report)

    s_only_tx = make_s_only_optimizer(warm_svd_model, lr=1e-4)
    ema_warm = train_edm(
        warm_svd_model,
        data,
        steps=50,
        seed=30,
        optimizer=s_only_tx,
    )
    sample_some(ema_warm, seed=300)

    # ------------------------------------------------------------
    # 4) RFFT retrofit on square linear layers only
    # ------------------------------------------------------------
    rfft_model, rfft_report = retrofit_diffusion_model(
        build_dense_dit(jr.PRNGKey(3)),
        variant="rfft",
        mode="all_linear",
        warmstart=True,
        key=jr.PRNGKey(4),
    )
    print("rfft_report:", rfft_report)
    ema_rfft = train_edm(rfft_model, data, steps=100, seed=40)
    sample_some(ema_rfft, seed=400)

    # ------------------------------------------------------------
    # 5) Optional: spectral regularization
    # ------------------------------------------------------------
    reg_svd_model, reg_report = retrofit_diffusion_model(
        build_dense_dit(jr.PRNGKey(5)),
        variant="svd",
        mode="all_linear",
    )
    print("reg_report:", reg_report)
    ema_reg = train_edm(
        reg_svd_model,
        data,
        steps=100,
        seed=50,
        reg_strength=1e-5,
    )
    sample_some(ema_reg, seed=500)


if __name__ == "__main__":
    main()
```

## How to adapt the example to a real dataset

The only thing you need to replace is the dataset construction.

Current synthetic line:

```python
data = generate_synthetic_image_dataset(
    num_samples=256,
    shape=(1, 28, 28),
    key=jr.PRNGKey(0),
)
```

Replace it with a real array of shape:

- images: `(N, C, H, W)`
- tabular: `(N, D)`
- time series: `(N, T, D)`

and keep the rest of the training loop unchanged.

## Retrofitting modes

The diffusion workflow exposes:

- `mode="attn_only"`
- `mode="attn_mlp"`
- `mode="all_linear"`

Use:

- `attn_only` if you want only attention projections replaced,
- `attn_mlp` if you want attention and MLP projections replaced,
- `all_linear` if you want every dense-like linear map replaced.

## Dense vs SVD vs warm-start vs RFFT

### Dense

No retrofit. Train normally.

```python
model = build_dense_dit(jr.PRNGKey(0))
ema = train_edm(model, data)
```

### SVD from scratch

Start from a dense architecture, immediately retrofit to `SVDDense`, then train.

```python
model, report = retrofit_diffusion_model(build_dense_dit(jr.PRNGKey(0)), variant="svd")
ema = train_edm(model, data)
```

### Warm-start SVD

Train dense first, then transplant to `SVDDense`, then fine-tune only `s`.

```python
ema_dense = train_edm(build_dense_dit(jr.PRNGKey(0)), data)
svd_model, report = retrofit_diffusion_model(ema_dense, variant="svd")
s_only_tx = make_s_only_optimizer(svd_model)
ema_svd = train_edm(svd_model, data, optimizer=s_only_tx)
```

### RFFT

Replace only square linear maps. Non-square maps are skipped and reported.

```python
model, report = retrofit_diffusion_model(
    build_dense_dit(jr.PRNGKey(0)),
    variant="rfft",
    warmstart=True,
    key=jr.PRNGKey(1),
)
ema = train_edm(model, data)
```

## About regularization

Diffusion is already complete for regularized training because `train_model(...)` accepts `custom_loss=...`.

Example with operator-norm regularization:

```python
from quantbayes.stochax.utils.regularizers import global_spectral_norm_penalty


def edm_loss_with_reg(model, batch, key):
    base = edm_batch_loss(
        model,
        batch,
        key,
        sigma_data=0.5,
        rho_min=-1.2,
        rho_max=1.2,
        sample="uniform",
    )
    reg = 1e-5 * global_spectral_norm_penalty(model)
    return base + reg
```

Then pass:

```python
ema = train_model(
    model,
    dataset=data,
    t1=1.0,
    lr=3e-4,
    num_steps=100,
    batch_size=32,
    weight_fn=None,
    int_beta_fn=None,
    print_every=20,
    seed=0,
    data_loader_func=dataloader,
    loss_impl="edm",
    custom_loss=edm_loss_with_reg,
)
```

## Which existing files to copy from

For the base training and sampling idiom, copy from:

- `quantbayes/stochax/diffusion/testing/test_image.py`
- `quantbayes/stochax/diffusion/testing/test_tabular.py`
- `quantbayes/stochax/diffusion/testing/test_time_series.py`

For the spectral idea, add exactly two things:

1. a call to `retrofit_diffusion_model(...)`,
2. optionally a masked optimizer and/or custom regularized loss.

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

That separation helps distinguish:

- effects from the parameterization itself,
- effects from explicit norm control.
