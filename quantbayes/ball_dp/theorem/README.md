# Theorem-backed nonconvex Ball-DP API

This package now has a **compact theory-scoped entry point**:

```python
from quantbayes.ball_dp.theorem import ...
```

The goal of this API is narrow and explicit:

1. instantiate exactly the one-hidden-layer tanh model families covered by the theorem files,
2. train them with the existing `fit_ball_sgd(...)` release path,
3. warm-start from dense checkpoints,
4. convert dense hidden layers into fixed-basis `SVDDense` layers with frozen `U, V`,
5. fine-tune either the default theorem-safe trainable set (`s`, hidden bias, output head) or only `s`,
6. keep the README self-contained so the workflow is obvious without hunting through demo scripts.

## Model families

All theorem families are represented by a single declarative spec:

```python
from quantbayes.ball_dp.theorem import TheoremModelSpec
```

Supported families:

- binary dense Frobenius: `task="binary", parameterization="dense", constraint="fro"`
- binary dense operator-norm: `task="binary", parameterization="dense", constraint="op"`
- binary fixed-basis SVD: `task="binary", parameterization="svd", constraint="op"`
- multiclass dense Frobenius: `task="multiclass", parameterization="dense", constraint="fro"`
- multiclass dense operator-norm: `task="multiclass", parameterization="dense", constraint="op"`
- multiclass fixed-basis SVD: `task="multiclass", parameterization="svd", constraint="op"`

The fixed-basis SVD family is intentionally the operator-norm theorem family, because the theorem controls
`||s||_∞ <= Lambda`, which is equivalent to the hidden operator norm when `U, V` are orthonormal.

## Public theorem quantities vs. training quantities

Keep these separate:

- `bounds.B` is the public input norm bound from the theorem.
- `bounds.A` bounds the output head norm.
- `bounds.S` is only for the dense Frobenius theorem family.
- `bounds.Lambda` is for the dense operator-norm and fixed-basis SVD theorem families.
- `train_cfg.radius` is the Ball-DP radius `r` used by the release mechanism.

## Core API

```python
from quantbayes.ball_dp.theorem import (
    TheoremBounds,
    TheoremModelSpec,
    TrainConfig,
    certified_constants,
    certified_lz,
    check_constraints,
    check_input_bound,
    fit_release,
    load_dense_checkpoint_as_svd,
    load_model_checkpoint,
    make_model,
    make_optimizer,
    make_projector,
    replace_dense_with_svd,
    save_model_checkpoint,
)
```

## Recommended defaults

The compact API uses privacy-safer defaults than the old exploratory demos:

- `TrainConfig.batch_sampler="poisson"`
- `TrainConfig.accountant_subsampling="match_sampler"`

This makes the public API line up with the accounting assumptions by default.

## 1) Binary dense model, trained directly

```python
import jax.random as jr
import numpy as np

from quantbayes.ball_dp.theorem import (
    TheoremBounds,
    TheoremModelSpec,
    TrainConfig,
    certified_lz,
    fit_release,
    make_model,
)

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int32)

spec = TheoremModelSpec(
    d_in=X_train.shape[1],
    hidden_dim=256,
    task="binary",
    parameterization="dense",
    constraint="fro",
)
bounds = TheoremBounds(B=1.0, A=4.0, S=8.0)
model = make_model(spec, key=jr.PRNGKey(0), init_project=True, bounds=bounds)

train_cfg = TrainConfig(
    radius=0.5,
    privacy="ball_dp",
    epsilon=3.0,
    delta=1e-6,
    num_steps=2000,
    batch_size=128,
    clip_norm=1.0,
    noise_multiplier=1.1,
    learning_rate=3e-3,
)

release = fit_release(
    model,
    spec,
    bounds,
    X_train,
    y_train,
    train_cfg=train_cfg,
)

print("L_z:", certified_lz(spec, bounds))
print("train accuracy:", release.utility_metrics.get("train_accuracy"))
```

## 2) Binary fixed-basis SVD model from scratch

This instantiates the SVD theorem model directly. The random initialization is created by sampling a dense hidden
layer and factorizing it once; `U, V` are then treated as fixed.

```python
spec = TheoremModelSpec(
    d_in=X_train.shape[1],
    hidden_dim=256,
    task="binary",
    parameterization="svd",
    constraint="op",
    rank=32,
)
bounds = TheoremBounds(B=1.0, A=4.0, Lambda=2.0)
model = make_model(spec, key=jr.PRNGKey(1), init_project=True, bounds=bounds)

release = fit_release(
    model,
    spec,
    bounds,
    X_train,
    y_train,
    train_cfg=train_cfg,
    trainable="default",   # freeze U,V; optimize s, hidden bias, output head
)
```

## 3) Public or pretrained dense checkpoint -> private dense fine-tuning

```python
from quantbayes.ball_dp.theorem import save_model_checkpoint, load_model_checkpoint

# Suppose release.payload is the public/pretrained dense model.
dense_model = release.payload
save_model_checkpoint(dense_model, spec, "./ckpt_dense")

dense_model_2, dense_spec, dense_state, metadata = load_model_checkpoint("./ckpt_dense")

dense_finetune = fit_release(
    dense_model_2,
    dense_spec,
    bounds,
    X_private,
    y_private,
    train_cfg=train_cfg,
)
```

## 4) Dense -> fixed-basis SVD conversion -> second-stage fine-tuning

This is the canonical theorem-aligned warm-start path:

1. start from a dense model,
2. convert the hidden layer to `SVDDense` via the SVD of the learned dense weight,
3. keep `U, V` fixed,
4. continue training the fixed-basis model.

```python
from quantbayes.ball_dp.theorem import replace_dense_with_svd

dense_spec = TheoremModelSpec(
    d_in=X_public.shape[1],
    hidden_dim=256,
    task="binary",
    parameterization="dense",
    constraint="fro",
)
dense_bounds = TheoremBounds(B=1.0, A=4.0, S=8.0)

dense_model = make_model(dense_spec, key=jr.PRNGKey(0), init_project=True, bounds=dense_bounds)
public_cfg = TrainConfig(
    radius=0.5,
    privacy="noiseless",
    num_steps=1000,
    batch_size=128,
    clip_norm=1.0,
    noise_multiplier=0.0,
    learning_rate=3e-3,
)
public_release = fit_release(
    dense_model,
    dense_spec,
    dense_bounds,
    X_public,
    y_public,
    train_cfg=public_cfg,
)

svd_spec = dense_spec.to_svd(rank=32)
svd_bounds = TheoremBounds(B=1.0, A=4.0, Lambda=2.0)
svd_model = replace_dense_with_svd(
    public_release.payload,
    svd_spec,
    init_project=True,
    bounds=svd_bounds,
)

private_release = fit_release(
    svd_model,
    svd_spec,
    svd_bounds,
    X_private,
    y_private,
    train_cfg=train_cfg,
    trainable="default",
)
```

## 5) Load a dense checkpoint and convert it directly to SVD

```python
from quantbayes.ball_dp.theorem import load_dense_checkpoint_as_svd

svd_bounds = TheoremBounds(B=1.0, A=4.0, Lambda=2.0)
svd_model, svd_spec, state, metadata = load_dense_checkpoint_as_svd(
    "./ckpt_dense",
    rank=32,
    bounds=svd_bounds,
    init_project=True,
)

private_release = fit_release(
    svd_model,
    svd_spec,
    svd_bounds,
    X_private,
    y_private,
    train_cfg=train_cfg,
    trainable="default",
)
```

## 6) Train only `s`

The fixed-basis SVD models expose two meaningful trainable modes:

- `trainable="default"`: freeze `U, V`, train `s`, hidden bias, and output head.
- `trainable="s_only"`: freeze everything except `hidden.s`.

```python
s_only_release = fit_release(
    svd_model,
    svd_spec,
    svd_bounds,
    X_private,
    y_private,
    train_cfg=train_cfg,
    trainable="s_only",
)
```

## 7) Multiclass dense and SVD models

The multiclass workflow is exactly the same except for `task` and `num_classes`.

```python
multiclass_spec = TheoremModelSpec(
    d_in=X_train.shape[1],
    hidden_dim=256,
    task="multiclass",
    num_classes=10,
    parameterization="dense",
    constraint="op",
)
multiclass_bounds = TheoremBounds(B=1.0, A=6.0, Lambda=2.0)
multiclass_model = make_model(
    multiclass_spec,
    key=jr.PRNGKey(2),
    init_project=True,
    bounds=multiclass_bounds,
)

multiclass_release = fit_release(
    multiclass_model,
    multiclass_spec,
    multiclass_bounds,
    X_train,
    y_train,
    train_cfg=train_cfg,
)

multiclass_svd_spec = multiclass_spec.to_svd(rank=64)
multiclass_svd_model = replace_dense_with_svd(
    multiclass_release.payload,
    multiclass_svd_spec,
    init_project=True,
    bounds=multiclass_bounds,
)
```

## 8) Explicit theorem constants

```python
consts = certified_constants(svd_spec, svd_bounds)
print(consts)
# {'kappa_tanh': ..., 'L_f': ..., 'G_f': ..., 'L_z': ...}
```

## 9) Constraint checks

```python
from quantbayes.ball_dp.theorem import check_constraints, check_input_bound

check_input_bound(X_train, bounds)
check_constraints(model, spec, bounds)
```

## Notes on warm-starts and privacy composition

- If a first-stage dense model is trained **nonprivately on the same private dataset**, the overall pipeline is not DP for that dataset.
- The clean theorem-facing use cases are:
  - public pretraining,
  - pretraining on disjoint data,
  - or a first stage that is itself privately trained and accounted.
