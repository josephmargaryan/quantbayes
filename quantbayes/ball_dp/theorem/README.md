# Theorem-only nonconvex API

This guide covers the compact theorem API:

```python
from quantbayes.ball_dp.theorem import ...
```

It is intentionally narrow:

- one-hidden-layer tanh models;
- dense and fixed-basis SVD parameterizations;
- theorem-backed $L_z$ certificates;
- direct compatibility with Poisson Ball-SGD releases and Ball-ReRo reports;
- compatibility with the shared finite-prior attack setup.

---

## 1. Model families and theorem parameters

The declarative model spec is:

```python
from quantbayes.ball_dp.theorem import TheoremModelSpec, TheoremBounds
```

Supported families include:

- binary dense Frobenius;
- binary dense operator norm;
- binary fixed-basis SVD;
- multiclass dense Frobenius;
- multiclass dense operator norm;
- multiclass fixed-basis SVD.

The theorem parameters are deliberately separate from the training configuration:

| Parameter | Meaning |
|---|---|
| `bounds.B` | public input norm bound; |
| `bounds.A` | output-head norm bound; |
| `bounds.S` | dense Frobenius constraint; |
| `bounds.Lambda` | dense operator norm or fixed-basis diagonal/operator constraint; |
| `train_cfg.radius` | Ball policy radius $r$; |
| `train_cfg.clip_norm` | gradient clipping norm $C_t$. |

---

## 2. Dense private training

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
X_eval = np.asarray(X_eval, dtype=np.float32)
y_eval = np.asarray(y_eval, dtype=np.int32)

num_classes = int(len(np.unique(y_priv)))
feature_dim = int(X_priv.shape[1])
B_public = float(max(np.linalg.norm(X_priv, axis=1).max(), np.linalg.norm(X_eval, axis=1).max()))

spec = TheoremModelSpec(
    d_in=feature_dim,
    hidden_dim=128,
    task="multiclass",
    parameterization="dense",
    constraint="op",
    num_classes=num_classes,
)

bounds = TheoremBounds(B=B_public, A=4.0, Lambda=4.0)
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
    X_eval=X_eval,
    y_eval=y_eval,
    train_cfg=train_cfg,
)

print("L_z:", certified_lz(spec, bounds))
print("utility:", release.utility_metrics)
```

---

## 3. Fixed-basis SVD private fine-tuning

The fixed-basis hidden layer has the form

$$
W \approx U\operatorname{diag}(s)V^\top,
$$

where $U,V$ are frozen public bases and the private optimization updates $s$ and the theorem-safe trainable parameters.

Typical workflow:

1. train a dense theorem model on public data;
2. save the dense public checkpoint;
3. convert the dense hidden layer to fixed-basis SVD;
4. choose ranks $r\in\{8,16,32,64,\dots\}$;
5. private fine-tune with `fit_release(..., trainable="default")`.

Key helpers:

```python
save_model_checkpoint(...)
load_model_checkpoint(...)
replace_dense_with_svd(...)
load_dense_checkpoint_as_svd(...)
fit_release(..., trainable="default")
```

---

## 4. Rank invariance of the fixed-basis certificate

In the fixed-basis theorem used here, the certified Lipschitz-in-data constant

$$
L_z^{(\mathrm{fb})}
$$

depends on the public norm bound and parameter constraints, but not on the chosen SVD rank. If $B$, $A$, $\Lambda$, and the hidden width are fixed, then changing rank should leave the theorem-backed privacy certificate unchanged.

A rank sweep is therefore a study of:

- utility vs. rank;
- empirical finite-prior attack behavior vs. rank;
- a rank-invariant theorem-backed privacy certificate.

---

## 5. Finite-prior attacks after theorem-backed training

Use the shared setup from `attacks.finite_prior_setup` before training the private release:

```python
from quantbayes.ball_dp.attacks.finite_prior_setup import (
    CandidateSource,
    find_feasible_replacement_banks,
    select_support_from_bank,
    make_replacement_trial,
)
```

Train the theorem release on `trial.X_full, trial.y_full`, then use the nonconvex wrapper:

```python
from quantbayes.ball_dp.api import attack_nonconvex_finite_prior_trial
```

For reporting, construct the finite-prior object from the same support:

```python
from quantbayes.ball_dp import make_finite_identification_prior, ball_rero

prior = make_finite_identification_prior(trial.support.X, weights=trial.support.weights)
report = ball_rero(release, prior=prior, eta_grid=(0.5,), mode="rdp")
```

---

## 6. Recommended reports

For theorem-backed nonconvex releases, report:

- utility on public/test data;
- $L_z$;
- Ball and standard RDP curves;
- RDP-to-ReRo finite-prior bound;
- direct revealed-inclusion bound when appropriate;
- hidden-inclusion bound when matching the unknown-inclusion attack;
- exact-ID attack success against $1/m$;
- support hash and source-ID bookkeeping.

---

## 7. Related tutorials

```text
examples/ball_dp/nonconvex_synthetic_transcript_ball_dp_end_to_end_demo.ipynb
```

The tutorial uses dense theorem models. Fixed-basis SVD experiments can reuse the same finite-prior setup and reporting cells.
