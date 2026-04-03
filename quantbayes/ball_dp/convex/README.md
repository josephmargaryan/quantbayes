# `quantbayes.ball_dp.convex`

Usage-oriented documentation for the convex DP-ERM path.
This file is meant to live at:

- `quantbayes/ball_dp/convex/README.md`

The convex path solves a regularized ERM problem first, then releases a noisy version of the fitted parameters.
That is very different from the nonconvex DP-SGD path: here the privacy mechanism is **Gaussian output perturbation of the solved model**, not iterative noisy gradient descent.

---

## 1. What this path does

The public entry point is:

```python
from quantbayes.ball_dp import fit_convex
```

Supported convex model families:

- `softmax_logistic` — multiclass linear softmax regression
- `binary_logistic` — binary logistic regression
- `squared_hinge` — binary linear squared-hinge classifier
- `ridge_prototype` — one prototype vector per class with ridge shrinkage

The privacy modes exposed by `fit_convex(...)` are:

- `privacy="ball_dp"` — calibrate Gaussian output noise from `(epsilon, delta)` under Ball adjacency
- `privacy="ball_rdp"` — use a user-supplied `sigma` and attach RDP / converted DP ledgers
- `privacy="noiseless"` — no privacy noise; useful for baselines and convex reconstruction experiments

The convex public API does **not** expose `standard_dp` / `standard_rdp` as separate fit modes.
Instead, every release stores both:

- a Ball view (`release.privacy.ball`, `release.sensitivity.delta_ball`), and
- when available, a standard comparator view (`release.privacy.standard`, `release.sensitivity.delta_std`).

The standard comparator exists only if the code can infer a standard radius, usually from `embedding_bound`.

---

## 2. Conceptual parameters

There are four quantities that are easy to conflate:

- `radius` — the Ball radius `r` in feature / embedding space
- `embedding_bound` — a public norm bound `||x||_2 <= B` used to certify theorem-backed `L_z` bounds for the linear families
- `lam` — regularization strength in the ERM objective
- `sigma` — Gaussian output noise standard deviation applied to the released parameters

The convex path uses them as follows:

- `radius` determines Ball sensitivity.
- `embedding_bound` is required for theorem-backed `L_z` bounds for:
  - `softmax_logistic`
  - `binary_logistic`
  - `squared_hinge`
- `ridge_prototype` has its own exact sensitivity path and does **not** require `embedding_bound`.
- `lam` affects both optimization and sensitivity.
- For `privacy="ball_dp"`, the library calibrates `sigma` from `(epsilon, delta)`.
- For `privacy="ball_rdp"`, you must provide `sigma` explicitly.

If you already know a valid `L_z`, you may pass it via `lz=...` instead of using `embedding_bound`.

Do **not** set `embedding_bound` to a convenient number unless it really is a public upper bound on your feature norms.
If your embeddings are not normalized to `||x||_2 <= 1`, then `embedding_bound=1.0` is simply the wrong certificate input.

---

## 3. Quick start: multiclass softmax logistic on embeddings

```python
import numpy as np
from sklearn.model_selection import train_test_split

from quantbayes.ball_dp import fit_convex

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

release = fit_convex(
    X_train,
    y_train,
    X_eval=X_test,
    y_eval=y_test,
    model_family="softmax_logistic",
    privacy="ball_dp",
    radius=0.10,
    lam=1e-2,
    epsilon=3.0,
    delta=1e-6,
    embedding_bound=1.0,   # required unless you provide lz=...
    num_classes=10,
    solver="lbfgs_fullbatch",
    max_iter=200,
    early_stop=True,
    stop_rule="grad_only",
    min_iter=20,
    seed=0,
)

print("release kind:", release.release_kind)
print("utility:", release.utility_metrics)
print("sensitivity ball:", release.sensitivity.delta_ball)
print("sensitivity standard:", release.sensitivity.delta_std)

if release.privacy.ball.dp_certificates:
    cert = release.privacy.ball.dp_certificates[0]
    print("ball epsilon:", cert.epsilon)
    print("ball delta:", cert.delta)
```

What comes back:

- `release.payload` is the released model parameters.
- `release.utility_metrics` contains evaluation metrics on `X_eval, y_eval` if you supplied them.
- `release.privacy` contains Ball and standard-comparator ledgers.
- `release.sensitivity` contains Ball / standard sensitivities and the `L_z` source.
- `release.optimization` contains solver diagnostics for the iterative linear solvers.

---

## 4. Family-specific notes

### 4.1 `softmax_logistic`

Use this for multiclass linear classification on embeddings or features.
Labels should be integer-coded in `{0, ..., K-1}`.

Recommended when:

- your features are already good embeddings,
- you want the strongest convex baseline,
- you want easy, direct inference.

### 4.2 `binary_logistic`

Use this for binary classification.
Accepted labels are either `{0, 1}` or `{-1, +1}`.

### 4.3 `squared_hinge`

Binary linear large-margin model.
Accepted labels are either `{0, 1}` or `{-1, +1}`.
Internally, this path uses the `{-1, +1}` view.

### 4.4 `ridge_prototype`

This is not a linear weight vector model.
Instead, the release stores one prototype vector per class.
Prediction is nearest-prototype in squared Euclidean distance.

This family is especially useful when:

- you want a very interpretable class-representative release,
- you want the exact Ball sensitivity path implemented in the library,
- you want the exact prototype-space attacks included in the library.

---

## 5. What the `ReleaseArtifact` contains

All convex releases return a `ReleaseArtifact` dataclass:

```python
from dataclasses import asdict

print(release.release_kind)
print(release.model_family)
print(release.architecture)
print(release.training_config.keys())
print(release.dataset_metadata)
print(release.utility_metrics)

print(asdict(release.sensitivity))
if release.optimization is not None:
    print(asdict(release.optimization))
```

The most important fields are:

- `release.payload`
  - the actual released model parameters
  - type depends on `model_family`
- `release.training_config`
  - radius, lambda, solver, optimization config, Gaussian config, requested privacy inputs
- `release.privacy.ball`
  - Ball privacy ledger, certificates, RDP curve, `sigma`
- `release.privacy.standard`
  - standard comparator ledger when available
- `release.sensitivity`
  - `lz`, `lz_source`, `delta_ball`, `delta_std`, and whether the sensitivity is exact or an upper bound
- `release.optimization`
  - optimization diagnostics for iterative solvers
- `release.attack_metadata`
  - for noiseless convex releases, includes whether theorem-backed exact reconstruction is available
- `release.dataset_metadata`
  - feature shape, number of classes, label values
- `release.utility_metrics`
  - evaluation metrics on the public eval set only

Two subtle points:

1. `utility_metrics` are populated only if you pass `X_eval` and `y_eval`.
2. `release.optimization` can be `None`, especially for `ridge_prototype`, which uses a direct closed-form fit instead of an iterative solver.

---

## 6. Inference from a convex release

There is no single family-agnostic `predict(...)` wrapper in the public convex API.
You use the family-specific helpers from the corresponding model module.

### 6.1 Softmax logistic

```python
from quantbayes.ball_dp.convex.models.softmax_logistic import (
    softmax_logits,
    softmax_predictions,
)

logits = softmax_logits(release.payload, X_test)
preds = softmax_predictions(release.payload, X_test)
print(logits.shape, preds.shape)
```

### 6.2 Binary logistic

```python
from quantbayes.ball_dp.convex.models.binary_logistic import (
    binary_logits,
    binary_predictions,
)

scores = binary_logits(release.payload, X_test)
preds = binary_predictions(release.payload, X_test)
```

### 6.3 Squared hinge

```python
from quantbayes.ball_dp.convex.models.squared_hinge import (
    squared_hinge_scores,
    squared_hinge_predictions,
)

scores = squared_hinge_scores(release.payload, X_test)
preds = squared_hinge_predictions(release.payload, X_test)
```

### 6.4 Ridge prototype

```python
from quantbayes.ball_dp.convex.models.ridge_prototype import prototype_predict

preds = prototype_predict(release.payload, X_test)
```

### 6.5 One small dispatch helper

```python
def predict_from_convex_release(release, X):
    fam = release.model_family
    if fam == "softmax_logistic":
        from quantbayes.ball_dp.convex.models.softmax_logistic import (
            softmax_logits,
            softmax_predictions,
        )
        return {
            "logits": softmax_logits(release.payload, X),
            "predictions": softmax_predictions(release.payload, X),
        }
    if fam == "binary_logistic":
        from quantbayes.ball_dp.convex.models.binary_logistic import (
            binary_logits,
            binary_predictions,
        )
        return {
            "scores": binary_logits(release.payload, X),
            "predictions": binary_predictions(release.payload, X),
        }
    if fam == "squared_hinge":
        from quantbayes.ball_dp.convex.models.squared_hinge import (
            squared_hinge_scores,
            squared_hinge_predictions,
        )
        return {
            "scores": squared_hinge_scores(release.payload, X),
            "predictions": squared_hinge_predictions(release.payload, X),
        }
    if fam == "ridge_prototype":
        from quantbayes.ball_dp.convex.models.ridge_prototype import prototype_predict
        return {
            "predictions": prototype_predict(release.payload, X),
            "prototypes": release.payload.prototypes,
        }
    raise ValueError(f"Unsupported model_family={fam!r}")
```

---

## 7. Built-in visualizations

The convex path has a small but useful plot layer:

```python
from quantbayes.ball_dp import (
    plot_convex_model_parameters,
    plot_linear_model_weights,
    plot_ridge_prototypes,
)
```

### 7.1 Family-aware plotting

```python
plot_convex_model_parameters(release)
```

This dispatches automatically:

- linear families -> `plot_linear_model_weights(...)`
- `ridge_prototype` -> `plot_ridge_prototypes(...)`

### 7.2 Ridge prototypes directly

```python
plot_ridge_prototypes(release, class_names=[str(i) for i in range(10)])
```

### 7.3 Linear weights directly

```python
plot_linear_model_weights(release, class_names=[str(i) for i in range(10)])
```

Notes:

- These plots are most meaningful when the feature shape is image-like or otherwise spatially meaningful.
- If your inputs are generic embeddings, the raw parameter plots may be hard to interpret. In that case, confusion matrices, classwise accuracy, or nearest-neighbor diagnostics are often more informative.

---

## 8. Noiseless convex reconstruction attack

For noiseless convex releases, the library exposes exact / near-exact reconstruction attacks:

```python
from quantbayes.ball_dp import attack_convex

attack, d_minus, target = attack_convex(
    release,
    X_train,
    y_train,
    target_index=17,
    known_label=int(y_train[17]),   # optional but often helpful
    eta_grid=(0.1, 0.2, 0.5, 1.0),
)

print("status:", attack.status)
print("predicted label:", attack.y_hat)
print("metrics:", attack.metrics)
```

Plot the reconstruction if the features are image-like:

```python
from quantbayes.ball_dp import plot_convex_attack_result

plot_convex_attack_result(attack, target)
```

This path is most natural for `privacy="noiseless"`.
For Gaussian output perturbation, use the Ball-output attacks below instead.

---

## 9. Gaussian-output Ball attacks

For a noisy convex release with positive `sigma`, the library exposes two Ball-constrained attack families.

### 9.1 Continuous MAP attack with a Ball prior

Use one of the attack priors from `quantbayes.ball_dp.api`:

```python
import numpy as np

from quantbayes.ball_dp.api import (
    make_uniform_ball_attack_prior,
    attack_convex_ball_output,
    BallOutputMapAttackConfig,
)

center = np.asarray(X_train[y_train == y_train[17]].mean(axis=0), dtype=np.float32)
prior = make_uniform_ball_attack_prior(center=center, radius=0.10)

attack, d_minus, target = attack_convex_ball_output(
    release,
    X_train,
    y_train,
    target_index=17,
    prior=prior,
    cfg=BallOutputMapAttackConfig(
        optimizer="adam",
        num_steps=250,
        learning_rate=5e-2,
        num_restarts=3,
        seed=0,
    ),
    known_label=int(y_train[17]),
    eta_grid=(0.1, 0.2, 0.5, 1.0),
)

print(attack.status)
print(attack.metrics)
```

### 9.2 Exact finite-prior Bayes attack

```python
from quantbayes.ball_dp.api import attack_convex_ball_output_finite_prior

mask = (y_train == y_train[17])
X_candidates = np.asarray(X_train[mask], dtype=np.float32)
y_candidates = np.asarray(y_train[mask], dtype=np.int64)

attack, d_minus, target = attack_convex_ball_output_finite_prior(
    release,
    X_train,
    y_train,
    target_index=17,
    X_candidates=X_candidates,
    y_candidates=y_candidates,
    prior_weights=None,   # uniform prior over the finite support
    known_label=int(y_train[17]),
    eta_grid=(0.1, 0.2, 0.5, 1.0),
)

print(attack.status)
print(attack.metrics)
```

Use the finite-prior path when the theorem-aligned prior is genuinely discrete.
It is exact score-each-candidate Bayes classification, not projected gradient descent.

---

## 10. ReRo reporting

The library also exposes Ball-ReRo upper-bound reporting.
This uses **ReRo priors**, which are different from the attack priors above.

Construct a ReRo prior with one of:

- `make_uniform_ball_prior(...)`
- `make_empirical_ball_prior(...)`
- `make_finite_identification_prior(...)`

Example:

```python
from quantbayes.ball_dp import ball_rero, make_uniform_ball_prior, plot_rero_report

prior = make_uniform_ball_prior(center=center, radius=0.10)
report = ball_rero(
    release,
    prior=prior,
    eta_grid=(0.1, 0.2, 0.5, 1.0),
)

for point in report.points:
    print(point.eta, point.gamma_ball, point.gamma_standard)

plot_rero_report(report)
```

Do not confuse these two prior systems:

- `make_uniform_ball_prior(...)` is for ReRo reporting
- `make_uniform_ball_attack_prior(...)` is for continuous MAP attacks

---

## 11. Inspecting privacy and sensitivity

The convex release stores both the mechanism-side noise and the sensitivity-side theorem quantities.

```python
print("sigma:", release.privacy.ball.sigma)
print("ball delta:", release.sensitivity.delta_ball)
print("standard delta:", release.sensitivity.delta_std)
print("L_z:", release.sensitivity.lz)
print("L_z source:", release.sensitivity.lz_source)
print("exact or bound:", release.sensitivity.exact_vs_upper)
```

For `privacy="ball_dp"`, the stored Gaussian noise is the calibrated `sigma` used to satisfy the requested `(epsilon, delta)`.
For `privacy="ball_rdp"`, the stored Gaussian noise is exactly the `sigma` you supplied.

---

## 12. Optimization certificate: how to read it

For `softmax_logistic`, `binary_logistic`, and `squared_hinge`, the default public solvers are iterative:

- `lbfgs_fullbatch`
- `gd_fullbatch`

These produce an `OptimizationCertificate` inside `release.optimization`.
The key fields are:

- `converged`
- `n_iter`
- `grad_norm`
- `parameter_error_bound`
- `sensitivity_addon`
- `termination_reason`

Important interpretation:

- these are **solver diagnostics for the realized dataset**, not a theorem-backed global optimality proof;
- if `sensitivity_addon > 0`, the final sensitivity tag will reflect that the release used a local residual heuristic add-on.

If you need the cleanest reconstruction algebra for paper-style noiseless experiments, prefer the most exact path available and inspect:

```python
print(release.attack_metadata["theorem_backed_exact_reconstruction"])
```

---

## 13. Troubleshooting

### Error: missing `embedding_bound`

For the linear convex heads (`softmax_logistic`, `binary_logistic`, `squared_hinge`), the theorem-backed `L_z` bound needs either:

- `embedding_bound=...`, or
- `lz=...`

If you pass neither, fitting will fail.

### `ball_rdp` complains that `sigma` is missing

That is expected.
`privacy="ball_rdp"` requires an explicit `sigma`.
If you want the library to calibrate `sigma` from `(epsilon, delta)`, use `privacy="ball_dp"` instead.

### `utility_metrics` is empty

That usually just means you did not pass `X_eval` and `y_eval`.

### Parameter plot looks meaningless

That is common for embedding features.
The built-in parameter plots are most interpretable when features have image-like structure.

### Binary labels fail

For the binary families, labels must be either `{0, 1}` or `{-1, +1}`.

---

## 14. Minimal practical recipes

### Best convex baseline for pretrained embeddings

Use:

- `model_family="softmax_logistic"` for multiclass
- `model_family="binary_logistic"` for binary

### Most interpretable class representative release

Use:

- `model_family="ridge_prototype"`

### Noiseless reconstruction experiments

Use:

- `privacy="noiseless"`
- then call `attack_convex(...)`

### Private noisy output release with Ball attack experiments

Use:

- `privacy="ball_dp"` or `privacy="ball_rdp"`
- then call `attack_convex_ball_output(...)` or `attack_convex_ball_output_finite_prior(...)`

---

## 15. Saving a release

```python
from quantbayes.ball_dp.serialization import save_release_artifact

save_release_artifact(release, "./artifacts/convex_release")
```

This writes:

- a pickle with the full release artifact
- a JSON metadata summary

---

## 16. Suggested first experiments

If you are using embeddings and want a clean sequence of baselines:

1. `privacy="noiseless"`, `model_family="softmax_logistic"`
2. `privacy="ball_dp"`, same family and train/test split
3. compare `utility_metrics`, `delta_ball`, `delta_std`, and the calibrated `sigma`
4. if you care about reconstruction, add `attack_convex(...)` for the noiseless release and `attack_convex_ball_output(...)` for the noisy one

That sequence usually gives the clearest picture of what the convex Ball-DP path is buying you.
