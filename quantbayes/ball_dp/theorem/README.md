# Theorem-only nonconvex API (dense + fixed-basis SVD)

This README is the focused guide for the compact theorem API:

```python
from quantbayes.ball_dp.theorem import ...
```

It is intentionally narrow:
- one-hidden-layer tanh families only;
- dense and fixed-basis SVD parameterizations;
- theorem-backed `L_z` certificates;
- direct compatibility with the Ball-SGD release and Ball-ReRo report paths.

---

## 1. Model families and theorem parameters

The declarative spec is

```python
from quantbayes.ball_dp.theorem import TheoremModelSpec, TheoremBounds
```

Supported families:
- binary dense Frobenius
- binary dense operator norm
- binary fixed-basis SVD
- multiclass dense Frobenius
- multiclass dense operator norm
- multiclass fixed-basis SVD

The fixed-basis SVD hidden layer has the form
$$
W \approx U\,\operatorname{diag}(s)\,V^\top,
$$
with frozen orthonormal $U,V$ and private optimization over $s$ (plus typically hidden bias and output head).

The theorem parameters are kept separate from the training configuration:
- `bounds.B` — public input norm bound;
- `bounds.A` — output-head norm bound;
- `bounds.S` — dense Frobenius constraint;
- `bounds.Lambda` — dense operator norm or fixed-basis diagonal/operator constraint;
- `train_cfg.radius` — Ball policy radius.

---

## 2. Rank invariance of the fixed-basis certificate

In the fixed-basis theorem used here, the certified Lipschitz-in-data constant
$$
L_z^{(\mathrm{fb})}
$$
depends on the public norm bound and the parameter constraints, but **not on the chosen rank**.

So if `B`, `A`, `Lambda`, and the hidden width are fixed, then changing rank should leave the theorem-backed privacy certificate unchanged.
This means the rank sweep is a study of:
- utility vs. rank,
- empirical attack behavior vs. rank,
- with a rank-invariant theorem-backed certificate.

---

## 3. Group 1: private-only dense training

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

X_priv = np.asarray(X_priv, dtype=np.float32)
y_priv = np.asarray(y_priv, dtype=np.int32)
X_priv_test = np.asarray(X_priv_test, dtype=np.float32)
y_priv_test = np.asarray(y_priv_test, dtype=np.int32)

num_classes = int(len(np.unique(y_priv)))
feature_dim = int(X_priv.shape[1])
B_all = float(max(np.linalg.norm(X_priv, axis=1).max(), np.linalg.norm(X_priv_test, axis=1).max()))

spec = TheoremModelSpec(
    d_in=feature_dim,
    hidden_dim=128,
    task="multiclass",
    parameterization="dense",
    constraint="op",
    num_classes=num_classes,
)
bounds = TheoremBounds(B=B_all, A=4.0, Lambda=4.0)
model = make_model(spec, key=jr.PRNGKey(0), init_project=True, bounds=bounds)

train_cfg = TrainConfig(
    radius=0.10,
    privacy="ball_rdp",
    delta=1e-6,
    num_steps=400,
    batch_size=min(128, len(X_priv)),
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
    clip_norm=1.0,
    noise_multiplier=1.0,
    learning_rate=3e-3,
    eval_every=50,
    normalize_noisy_sum_by="batch_size",
    seed=0,
)

release = fit_release(
    model,
    spec,
    bounds,
    X_priv,
    y_priv,
    X_eval=X_priv_test,
    y_eval=y_priv_test,
    train_cfg=train_cfg,
)

print("L_z:", certified_lz(spec, bounds))
print("utility:", release.utility_metrics)
```

---

## 4. Group 2: public dense pretrain $\to$ private dense fine-tune

Typical workflow:
1. train a dense theorem model on public data;
2. save the checkpoint;
3. load it back and continue private fine-tuning on private data.

Use the helpers:
- `save_model_checkpoint(...)`
- `load_model_checkpoint(...)`

---

## 5. Group 3: public dense pretrain $\to$ fixed-basis SVD private fine-tune

Typical workflow:
1. train a dense public model;
2. convert the dense hidden layer into a frozen-basis `SVDDense` layer;
3. choose ranks `r in {8, 16, 32, 64, ...}`;
4. fine-tune only the theorem-safe trainable set (`s`, hidden bias, output head) on private data.

Key helpers:
- `replace_dense_with_svd(...)`
- `load_dense_checkpoint_as_svd(...)`
- `fit_release(..., trainable="default")`

The rank sweep is expected to keep the theorem-backed certificate fixed while changing approximation/optimization behavior.

---

## 6. ReRo reporting after theorem-backed training

Once you have a theorem-backed private release, the recommended report path is the same as in the main notebook:
- finite-prior exact identification for the primary embedding result;
- `mode="rdp"` for the main nonconvex certificate;
- `mode="ball_sgd_direct"` as the direct transcript theorem.
