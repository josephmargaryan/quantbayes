# `quantbayes.ball_dp.nonconvex`

Usage-oriented documentation for the nonconvex DP-ERM path.
This file is meant to live at:

- `quantbayes/ball_dp/nonconvex/README.md`

This README covers two related workflows:

1. the **generic nonconvex trainer** based on clipped per-example gradients and noisy SGD, and
2. the **theorem-backed model families** in the sibling `quantbayes.ball_dp.theorem` API.

If you want a one-hidden-layer tanh family with a certified `L_z` already wired in, use the theorem API.
If you want to train an arbitrary Equinox model, use the generic trainer.

---

## 1. What this path does

The generic public entry point is:

```python
from quantbayes.ball_dp import fit_ball_sgd
```

This trainer takes:

- a user-supplied Equinox model,
- an Optax optimizer,
- a per-example classification loss,
- clipping / noise schedules,
- and optional public evaluation data.

Supported privacy modes are:

- `privacy="ball_dp"`
- `privacy="ball_rdp"`
- `privacy="standard_dp"`
- `privacy="standard_rdp"`
- `privacy="noiseless"`

The key distinction is:

- `ball_*` requires a valid user-supplied `lz`.
- `standard_*` does not.

So for arbitrary custom models:

- use `standard_dp` / `standard_rdp` unless you have a theorem-backed `L_z`, or
- use the theorem API, which computes `L_z` for the supported one-hidden-layer tanh families.

---

## 2. Crucial difference from the convex path

The convex path calibrates Gaussian output noise directly.
The nonconvex path does **not** do that.

In the nonconvex trainer:

- `noise_multiplier` is the actual SGD noise schedule used during training,
- `epsilon` and `delta` are only used to **verify / certify** the resulting privacy level.

So this is wrong:

```python
# Wrong mental model
# "I set epsilon=3.0, therefore the library picked the right noise for me."
```

What is actually true is:

```python
# Correct mental model
# "I chose a noise_multiplier schedule, and the library told me whether
#  that schedule satisfies the requested epsilon, delta target."
```

If you want automatic noise calibration, use the accountant helper first:

```python
from quantbayes.ball_dp.api import calibrate_ball_sgd_noise_multiplier
```

---

## 3. Model contract for the generic trainer

Your custom model must implement:

```python
__call__(x, *, key=None, state=None) -> (output, state)
```

The trainer expects:

- input: one example `x`
- output: a tuple `(logits_or_score, state)`
- `state` can be `None` if unused

When you rely on built-in public accuracy tracking, the effective `predict_fn` should return logits / scores compatible with classification.
The default `predict_fn` just returns the model output.

The built-in losses are:

- `loss_name="softmax_cross_entropy"` — multiclass classification
- `loss_name="binary_logistic"` — binary classification

Label conventions:

- multiclass: integer labels in `{0, ..., K-1}`
- binary: labels in `{0, 1}` or `{-1, +1}`

Stateful BatchNorm-style mutable state is **not** supported in the core trainer.
In particular, the trainer rejects Equinox models carrying `StateIndex` leaves.
Use stateless layers or read-only state instead.

---

## 4. Quick start A: custom model with standard DP-SGD

This is the easiest generic starting point because it does not require `lz`.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from sklearn.model_selection import train_test_split

from quantbayes.ball_dp import fit_ball_sgd
from quantbayes.ball_dp.api import (
    calibrate_ball_sgd_noise_multiplier,
    evaluate_release_classifier,
)


class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, d_in: int, hidden_dim: int, num_classes: int, *, key):
        k1, k2 = jr.split(key)
        self.fc1 = eqx.nn.Linear(d_in, hidden_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_dim, num_classes, key=k2)

    def __call__(self, x, *, key=None, state=None):
        h = jax.nn.tanh(self.fc1(x))
        logits = self.fc2(h)
        return logits, state


X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

num_steps = 1500
batch_size = 128
clip_norm = 1.0

cal = calibrate_ball_sgd_noise_multiplier(
    dataset_size=len(X_train),
    radius=1.0,                 # unused by the standard view, but accepted by the API
    lz=None,
    num_steps=num_steps,
    batch_size=batch_size,
    clip_norm=clip_norm,
    target_epsilon=3.0,
    delta=1e-6,
    privacy="standard_dp",
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
)
noise_multiplier = float(cal["noise_multiplier"])

model = MLP(X_train.shape[1], hidden_dim=128, num_classes=10, key=jr.PRNGKey(0))
optimizer = optax.adam(3e-3)

release = fit_ball_sgd(
    model,
    optimizer,
    X_train,
    y_train,
    X_eval=X_test,
    y_eval=y_test,
    privacy="standard_dp",
    radius=1.0,
    lz=None,
    epsilon=3.0,
    delta=1e-6,
    num_steps=num_steps,
    batch_size=batch_size,
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
    clip_norm=clip_norm,
    noise_multiplier=noise_multiplier,
    loss_name="softmax_cross_entropy",
    checkpoint_selection="best_public_eval_accuracy",
    eval_every=25,
    seed=0,
)

print(release.utility_metrics)
print(evaluate_release_classifier(release, X_test, y_test))
```

This is the right generic baseline when you want to debug a model architecture before moving into Ball-specific theorem families.

---

## 5. Quick start B: theorem-backed Ball-DP model

If you want actual Ball accounting for a nonconvex classifier and do not want to supply `lz` manually, use the theorem API:

```python
import jax.random as jr
import numpy as np

from quantbayes.ball_dp.api import (
    calibrate_ball_sgd_noise_multiplier,
    evaluate_release_classifier,
)
from quantbayes.ball_dp.theorem import (
    TheoremBounds,
    TheoremModelSpec,
    TrainConfig,
    certified_lz,
    fit_release,
    make_model,
)

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int64)

spec = TheoremModelSpec(
    d_in=X_train.shape[1],
    hidden_dim=128,
    task="multiclass",
    parameterization="dense",
    constraint="fro",
    num_classes=10,
)

bounds = TheoremBounds(B=1.0, A=4.0, S=8.0)
lz = certified_lz(spec, bounds)

cal = calibrate_ball_sgd_noise_multiplier(
    dataset_size=len(X_train),
    radius=0.10,
    lz=lz,
    num_steps=2000,
    batch_size=128,
    clip_norm=1.0,
    target_epsilon=3.0,
    delta=1e-6,
    privacy="ball_dp",
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
)
noise_multiplier = float(cal["noise_multiplier"])

model = make_model(spec, key=jr.PRNGKey(0), init_project=True, bounds=bounds)
train_cfg = TrainConfig(
    radius=0.10,
    privacy="ball_dp",
    epsilon=3.0,
    delta=1e-6,
    num_steps=2000,
    batch_size=128,
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
    clip_norm=1.0,
    noise_multiplier=noise_multiplier,
    learning_rate=3e-3,
    checkpoint_selection="best_public_eval_accuracy",
    eval_every=25,
    seed=0,
)

release = fit_release(
    model,
    spec,
    bounds,
    X_train,
    y_train,
    train_cfg=train_cfg,
    X_eval=X_test,
    y_eval=y_test,
)

print("L_z:", lz)
print(release.utility_metrics)
print(evaluate_release_classifier(release, X_test, y_test))
```

This is the cleanest theorem-facing Ball-DP workflow for the supported tanh networks.

`fit_release(...)` also checks the public input bound `||x||_2 <= B` from `TheoremBounds`.
So `bounds.B=1.0` is only appropriate when your embeddings are actually normalized to norm at most 1.

---

## 6. The theorem API: what is supported

The theorem entry point is:

```python
from quantbayes.ball_dp.theorem import ...
```

Theorem-backed families are specified with `TheoremModelSpec`.
Supported combinations:

- binary dense Frobenius
- binary dense operator-norm
- binary fixed-basis SVD operator-norm
- multiclass dense Frobenius
- multiclass dense operator-norm
- multiclass fixed-basis SVD operator-norm

Minimal objects:

- `TheoremModelSpec` — architecture family
- `TheoremBounds` — theorem-side public bounds
- `TrainConfig` — trainer configuration

Core helpers:

- `make_model(...)`
- `fit_release(...)`
- `certified_constants(...)`
- `certified_lz(...)`
- `check_input_bound(...)`
- `check_constraints(...)`
- `make_projector(...)`
- `replace_dense_with_svd(...)`

The theorem README already present at `quantbayes/ball_dp/theorem/README.md` is the narrow theorem-specific companion.
The current file is the broader nonconvex usage guide.

---

## 7. Dense -> SVD warm-start workflow

The supported SVD warm-start path is:

1. train a dense theorem model,
2. convert the hidden layer to a fixed-basis SVD layer,
3. continue training with `U, V` frozen.

```python
from quantbayes.ball_dp.theorem import replace_dense_with_svd

public_release = fit_release(
    model,
    spec,
    bounds,
    X_public,
    y_public,
    train_cfg=TrainConfig(
        radius=0.10,
        privacy="noiseless",
        num_steps=1000,
        batch_size=128,
        clip_norm=1.0,
        noise_multiplier=0.0,
        learning_rate=3e-3,
    ),
)

svd_spec = spec.to_svd(rank=32)
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

Trainable modes for SVD models:

- `trainable="default"` — freeze `U, V`, train `s`, hidden bias, output head
- `trainable="all"` — train everything
- `trainable="s_only"` — train only the singular-value vector `s`

---

## 8. Checkpointing theorem models

The theorem helper layer includes explicit checkpoint helpers:

```python
from quantbayes.ball_dp.theorem import (
    save_model_checkpoint,
    load_model_checkpoint,
    load_dense_checkpoint_as_svd,
)

save_model_checkpoint(release.payload, spec, "./ckpt_dense")
model2, spec2, state2, metadata2 = load_model_checkpoint("./ckpt_dense")

svd_model, svd_spec, svd_state, svd_metadata = load_dense_checkpoint_as_svd(
    "./ckpt_dense",
    rank=32,
    bounds=TheoremBounds(B=1.0, A=4.0, Lambda=2.0),
    init_project=True,
)
```

This is useful when you want a reproducible dense -> SVD fine-tuning pipeline.

---

## 9. What the `ReleaseArtifact` contains

All nonconvex trainers also return a `ReleaseArtifact`.
The most important fields are:

- `release.payload` — the released model itself
- `release.training_config` — full training schedule, clipping, noise, checkpointing, sampler settings
- `release.privacy.ball` / `release.privacy.standard` — privacy ledgers
- `release.sensitivity` — `lz`, per-step sensitivities, and summary tags
- `release.utility_metrics` — public evaluation metrics only
- `release.extra` — stored histories and model state

A few nonconvex-specific details matter a lot:

### 9.1 `train_accuracy` is not populated

The nonconvex trainer does **not** store train accuracy.
The stored evaluation metrics are public-eval metrics such as:

- `public_eval_loss`
- `public_eval_accuracy`
- `accuracy` (an alias for public eval accuracy)

So if you see:

```python
release.utility_metrics.get("train_accuracy") is None
```

that is expected.

### 9.2 Public curves and operator norms live in `release.extra`

```python
print(release.extra.keys())
```

Common entries:

- `public_curve_history`
- `operator_norm_history`
- `ball_regime`
- `selected_checkpoint_step`
- `resolved_batch_sampler`
- `resolved_accountant_subsampling`
- `model_state`

### 9.3 The payload is directly callable

For nonconvex releases, `release.payload` is the released Equinox model.
You normally run inference by putting it into inference mode.

---

## 10. Inference from a nonconvex release

### 10.1 Easiest path: built-in evaluator

```python
from quantbayes.ball_dp.api import evaluate_release_classifier

print(evaluate_release_classifier(release, X_train, y_train))
print(evaluate_release_classifier(release, X_test, y_test))
```

This is the fastest way to check whether the model ever learned.

### 10.2 Manual batch prediction

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

model = eqx.nn.inference_mode(release.payload, value=True)
state = release.extra.get("model_state", None)

@jax.jit
def predict_batch(xb, key):
    xb = jnp.asarray(xb, dtype=jnp.float32)
    keys = jr.split(key, xb.shape[0])
    logits = jax.vmap(
        lambda x, k: model(x, key=k, state=state)[0],
        in_axes=(0, 0),
    )(xb, keys)
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(probs, axis=-1)
    return logits, probs, preds

logits, probs, preds = predict_batch(X_test[:64], jr.PRNGKey(0))
print(np.asarray(preds))
print(np.asarray(probs[:3]))
```

For binary models, replace the softmax with a sigmoid threshold on the single logit.

---

## 11. Built-in diagnostics and visualizations

### 11.1 Public evaluation curve

```python
from quantbayes.ball_dp import plot_release_curves
from quantbayes.ball_dp.api import get_public_curve_history

history = get_public_curve_history(release)
print(history[:3])
plot_release_curves(release)
```

This is the most important built-in plot for private training.
If the best public accuracy happened early but the final model is worse, you should use public checkpoint selection.

### 11.2 Operator norm history

```python
from quantbayes.ball_dp import plot_operator_norm_history
from quantbayes.ball_dp.api import get_operator_norm_history

history = get_operator_norm_history(release)
print(history[:3])
plot_operator_norm_history(release)
```

To populate this history, fit with:

```python
record_operator_norms=True,
operator_norms_every=25,
```

### 11.3 Per-step privacy / mechanism table

```python
from quantbayes.ball_dp.api import get_release_step_table

rows = get_release_step_table(release)
print(rows[0].keys())
print(rows[:2])
```

This table contains, per step:

- batch size / sample rate
- clip norm
- noise multiplier
- effective noise standard deviation
- Ball sensitivity
- standard sensitivity
- Ball-to-standard sensitivity ratio
- `rho = (L_z r) / (2 C_t)` when defined

This is the easiest way to inspect whether Ball accounting is actually improving sensitivity for your schedule.

---

## 12. Sensitive debug history

You can ask the trainer to return stepwise gradient diagnostics:

```python
release, debug = fit_ball_sgd(
    model,
    optimizer,
    X_train,
    y_train,
    privacy="standard_dp",
    radius=1.0,
    lz=None,
    epsilon=3.0,
    delta=1e-6,
    num_steps=200,
    batch_size=128,
    clip_norm=1.0,
    noise_multiplier=noise_multiplier,
    loss_name="softmax_cross_entropy",
    return_debug_history=True,
)

print(debug["warning"])
print(debug["step_history"][:3])
```

This is useful for diagnosing clipping and noise, but the returned statistics are computed directly from private per-example gradients.
Treat them as internal-only debugging output unless you separately privatize / account for them.

Fields include:

- `mean_per_example_grad_norm`
- `max_per_example_grad_norm`
- `clip_fraction`
- realized batch size
- effective noise std on the stored private object

---

## 13. Noise calibration and privacy accounting

### 13.1 Calibrate a noise multiplier before training

```python
from quantbayes.ball_dp.api import calibrate_ball_sgd_noise_multiplier

cal = calibrate_ball_sgd_noise_multiplier(
    dataset_size=len(X_train),
    radius=0.10,
    lz=lz,                      # required for Ball accounting; can be None for standard
    num_steps=2000,
    batch_size=128,
    clip_norm=1.0,
    target_epsilon=3.0,
    delta=1e-6,
    privacy="ball_dp",
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
)
print(cal)
```

### 13.2 Account a candidate noise multiplier without training

```python
from quantbayes.ball_dp.api import account_ball_sgd_noise_multiplier

acct = account_ball_sgd_noise_multiplier(
    dataset_size=len(X_train),
    radius=0.10,
    lz=lz,
    num_steps=2000,
    batch_size=128,
    clip_norm=1.0,
    noise_multiplier=1.1,
    delta=1e-6,
    privacy="ball_dp",
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
)
print(acct)
```

### 13.3 Interpreting the Ball mechanism sensitivity

For the theorem-backed nonconvex path, each step uses:

```text
delta_ball_t = min(L_z * r, 2 C_t)
```

where:

- `L_z` is the certified theorem constant,
- `r` is the Ball radius,
- `C_t` is the clip norm at step `t`.

Ball accounting only helps when `L_z * r < 2 C_t`.
If `L_z * r >= 2 C_t` for all steps, the Ball view saturates to the standard clipped sensitivity.
Inspect this using `get_release_step_table(...)` or `release.extra["ball_regime"]`.

---

## 14. Utility debugging checklist

When utility is poor, the highest-value checks are usually:

1. **Run a noiseless baseline.**
   If the noiseless model cannot fit, privacy is not the primary problem.

2. **Use a public eval split and non-last checkpoint selection.**
   Private SGD often has a much better intermediate iterate than the final one.

3. **Calibrate noise instead of guessing it.**
   In this library, `epsilon` does not choose `noise_multiplier` for you.

4. **Compare train vs test with `evaluate_release_classifier(...)`.**
   That separates “never fit” from “fit but overfit.”

5. **Inspect clipping with debug history.**
   If `clip_fraction` is near 1.0 almost always, the clip norm is probably too small.

6. **Shrink the model if you are classifying good embeddings.**
   If a linear baseline already works, a large private MLP may just add needless noise burden.

7. **Check label shape and dtype.**
   For multiclass, you want shape `(N,)` and integer labels in `{0, ..., K-1}`.

A useful rule of thumb for multiclass classification: if

- `public_eval_loss ≈ log(K)`, and
- `public_eval_accuracy ≈ 1/K`,

then the model is behaving near chance level.

---

## 15. Public checkpoint selection

The trainer supports:

- `checkpoint_selection="last"`
- `checkpoint_selection="best_public_eval_loss"`
- `checkpoint_selection="best_public_eval_accuracy"`

Example:

```python
release = fit_ball_sgd(
    model,
    optimizer,
    X_train,
    y_train,
    X_eval=X_test,
    y_eval=y_test,
    privacy="standard_dp",
    radius=1.0,
    lz=None,
    epsilon=3.0,
    delta=1e-6,
    num_steps=2000,
    batch_size=128,
    clip_norm=1.0,
    noise_multiplier=noise_multiplier,
    loss_name="softmax_cross_entropy",
    checkpoint_selection="best_public_eval_accuracy",
    eval_every=25,
)
```

If you choose `best_public_eval_*`, you must provide a public evaluation dataset.

---

## 16. Radius selection for embeddings

If you are working in embedding space and want a data-driven Ball radius, use:

```python
from quantbayes.ball_dp import summarize_embedding_ball_radii, select_ball_radius

report = summarize_embedding_ball_radii(
    X_train,
    y_train,
    quantiles=(0.5, 0.8, 0.9, 0.95, 0.99, 1.0),
    max_exact_pairs=250_000,
    max_sampled_pairs=100_000,
    seed=0,
)

radius = select_ball_radius(
    report,
    strategy="max_labelwise_quantile",
    quantile=0.95,
)

print(radius)
print(report["candidate_radii"])
```

This is usually a better starting point than choosing `r` arbitrarily.

---

## 17. Manual confusion matrix / confidence plots

The library does not currently ship a built-in confusion-matrix helper, but it is easy to add around manual inference:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

logits, probs, preds = predict_batch(X_test, jr.PRNGKey(1))
preds = np.asarray(preds)
probs = np.asarray(probs)
conf = probs.max(axis=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, preds)).plot(ax=ax[0], colorbar=False)
ax[0].set_title("Confusion matrix")
ax[1].hist(conf, bins=30)
ax[1].set_title("Max softmax probability")
ax[1].set_xlabel("confidence")
plt.tight_layout()
plt.show()
```

For arbitrary embeddings, this is often more informative than plotting parameters directly.

---

## 18. Advanced attack / trace interfaces

The broader nonconvex stack includes advanced attack-oriented components in `quantbayes.ball_dp.api`, including:

- `make_trace_metadata_from_release(...)`
- `attack_nonconvex_ball_trace_finite_prior(...)`
- shadow / informed attack utilities such as `prepare_informed_attack_data(...)`, `build_attack_corpus(...)`, and `train_reconstructor(...)`

These are best treated as controlled experiment interfaces rather than standard training workflow tools.
If your immediate goal is understanding the release and getting good utility, start with the training, evaluation, and accounting sections above.

---

## 19. Common mistakes

### Mistake: using `privacy="ball_dp"` without `lz`

That will fail.
Ball accounting requires a valid `lz`.

### Mistake: assuming `epsilon` chooses the noise

It does not.
You still need to choose or calibrate `noise_multiplier`.

### Mistake: reading `train_accuracy` from `utility_metrics`

The trainer does not populate it.
Use `evaluate_release_classifier(release, X_train, y_train)` instead.

### Mistake: using a large MLP on already-good embeddings without a noiseless baseline

This is a common way to lose utility without learning anything about the privacy mechanism.
First compare:

- linear nonprivate baseline,
- small noiseless MLP,
- private MLP.

---

## 20. Suggested first experiment grid

If you are working with embeddings and want to understand what the nonconvex path can do, a good sequence is:

1. custom model, `privacy="noiseless"`
2. custom model, `privacy="standard_dp"`
3. theorem dense model, `privacy="ball_dp"`
4. theorem dense -> SVD warm-start, `privacy="ball_dp"`
5. compare:
   - utility metrics
   - selected checkpoint step
   - public curve shape
   - step table ratios `delta_ball / delta_standard`

That sequence usually tells you whether the main bottleneck is:

- architecture,
- optimization,
- clipping,
- noise,
- or a Ball radius / theorem-constant mismatch.
