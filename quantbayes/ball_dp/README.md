# quantbayes.ball_dp — Ball-DP Output Perturbation for Convex Heads

This module implements **Ball adjacency** DP for convex/strongly-convex “heads” trained on embeddings/features, plus **Gaussian output perturbation** calibrated from a clean ERM sensitivity bound:

$$
\Delta_2 \le \frac{L_z\, r}{\lambda\, n}
$$

You train a convex head normally (ideally full-batch LBFGS), then **release a private model** by adding calibrated Gaussian noise to the trained parameters.

---

## Core ideas (what you must provide)

To release a DP model you need:

- `n`: number of training records
- `lam`: L2 regularization strength used in the ERM objective (must match training!)
- `r`: Ball radius in your record metric (often chosen from within-class NN distances)
- `Lz`: a Lipschitz-in-data constant of per-example gradients (head-specific bound/estimate)
- `(eps, delta)`: DP budget
- Gaussian calibration method: `"analytic"` (tight) or `"classic"` (simple, conservative)

### Important: public norm bound for DP baselines
If you want a DP-valid “bounded replacement baseline” like `r_std = 2B`, then `B` must be a **public bound enforced by preprocessing**:
- L2-normalize rows (`B=1`), or
- L2-clip rows (`||x|| <= B`)

```
import numpy as np
from quantbayes.ball_dp.metrics import maybe_l2_normalize, clip_l2_rows

# Option A: normalize => B=1
Xtr = maybe_l2_normalize(Xtr.astype(np.float32), enabled=True)
Xte = maybe_l2_normalize(Xte.astype(np.float32), enabled=True)
B = 1.0

# Option B: clip => B is public clip norm
# B = 2.0
# Xtr = clip_l2_rows(Xtr.astype(np.float32), max_norm=B)
# Xte = clip_l2_rows(Xte.astype(np.float32), max_norm=B)
```
---

## Quickstart: choose `r` (radius policy)

```python
import numpy as np
from quantbayes.ball_dp.api import compute_radius_policy
from quantbayes.ball_dp.metrics import maybe_l2_normalize, clip_l2_rows

# X: (N,d) float, y: (N,) int
X = X.astype(np.float32)
y = y.astype(np.int64)

# Enforce a public bound B (recommended)
X = maybe_l2_normalize(X, enabled=True)   # => B=1

policy = compute_radius_policy(
    X, y,
    percentiles=(10, 25, 50, 75, 90),
    nn_sample_per_class=400,
    B_mode="public",
    B_public=1.0,               # because we normalized
)

# Pick a radius (example: median within-class NN distance)
r = policy.r_values[50.0]
# Or use the standard bounded-replacement baseline:
# r = policy.r_std  # = 2B
```
# 1) Equinox SVM (squared hinge, binary)
Use this bound: `lz_squared_hinge_bound(B=B, lam=lam, include_bias=True)`
```
n = int(Xtr.shape[0])
eps_dp, delta_dp = 2.0, 1e-6
y = (ytr.astype(np.int64) * 2 - 1).astype(np.float32) # Must be \{-1,1\}
Lz = float(lz_squared_hinge_bound(B=B, lam=lam, include_bias=True))

# flatten params
params, static = eqx.partition(model_trained, eqx.is_inexact_array)
theta_jax, unravel = ravel_pytree(params)
theta_hat = np.asarray(jax.device_get(theta_jax), dtype=np.float32)

out = dp_release_erm_params_gaussian(
    theta_hat,
    lz=Lz, r=float(r), lam=float(lam), n=n,
    eps=eps_dp, delta=delta_dp,
    sigma_method="analytic",
)

theta_private = out["params_noisy"]
params_private = unravel(jnp.asarray(theta_private, dtype=theta_jax.dtype))
model_private = eqx.combine(params_private, static)

yhat = predict_eqx_svm(model_private, Xte)
```
# 2) Torch SVM (squared hinge, binary)
Use this bound: `lz_squared_hinge_bound(B=B, lam=lam, include_bias=True)`
```
ytr_pm = (ytr.astype(np.int64) * 2 - 1).astype(np.int64) # must be in \{-1, 1\}
lam = 1e-2
cfg = SquaredHingeTorchConfig(lam=lam, optimizer="lbfgs", device="cuda")

model, metrics = train_squared_hinge_svm_torch(
    Xtr.astype(np.float32), ytr_pm, None, None, cfg=cfg
)

n = int(Xtr.shape[0])
eps_dp, delta_dp = 2.0, 1e-6
Lz = float(lz_squared_hinge_bound(B=B, lam=lam, include_bias=True))

dp = dp_release_torch_model_gaussian(
    model,
    lz=Lz, r=float(r), lam=float(lam), n=n,
    eps=eps_dp, delta=delta_dp,
    sigma_method="analytic",
)

model_private = dp["model_private"]
yhat = predict_squared_hinge_svm_torch(model_private, Xte.astype(np.float32))
```
# 3) Any Equinox convex model (example: logistic regression)
Use either `lz_logistic_binary_bound(B=B, lam=lam, include_bias=True)` or `lz_softmax_linear_bound(B=B, lam=lam, include_bias=True)`
If you trained with `stochax` using lambda_frob, then set
`lam = 2 * lambda_frob` when computing Lz and releasing DP noise. And remember ot turn on `include_bias` in the trainer when adding regularization
```
n = int(Xtr.shape[0])
eps_dp, delta_dp = 2.0, 1e-6
Lz = float(lz_logistic_binary_bound(B=B, lam=lam, include_bias=True))

dp = dp_release_eqx_model_gaussian(
    model_trained,
    lz=Lz, r=float(r), lam=float(lam), n=n,
    eps=eps_dp, delta=delta_dp,
    sigma_method="analytic",
)

model_private = dp["model_private"]
```
# 4) Any Torch convex model (example: logistic regression)
Use either `lz_logistic_binary_bound(B=B, lam=lam, include_bias=True)` or `lz_softmax_linear_bound(B=B, lam=lam, include_bias=True)`
```
cfg = LogRegTorchConfig(
    num_classes=K,
    lam=lam,
    optimizer="lbfgs",
    lbfgs_max_iter=500,
    device="cuda",
)

model, metrics = train_softmax_logreg_torch(
    Xtr.astype(np.float32), ytr.astype(np.int64), None, None, cfg=cfg
)

n = int(Xtr.shape[0])
eps_dp, delta_dp = 2.0, 1e-6
Lz = float(lz_softmax_linear_bound(B=B, lam=lam, include_bias=True))

dp = dp_release_torch_model_gaussian(
    model,
    lz=Lz, r=float(r), lam=float(lam), n=n,
    eps=eps_dp, delta=delta_dp,
    sigma_method="analytic",
)

model_private = dp["model_private"]
yhat = predict_softmax_logreg_torch(model_private, Xte.astype(np.float32))
```
# 5) NumPy prototypes (closed-form ridge prototypes)
Use this bound: `lz_prototypes_exact()` (exact = 2)
```
import numpy as np
from quantbayes.ball_dp.heads.prototypes import fit_ridge_prototypes, predict_nearest_prototype
from quantbayes.ball_dp.api import dp_release_ridge_prototypes_gaussian

K = int(np.max(ytr) + 1)
lam = 1e-1

mus, counts = fit_ridge_prototypes(
    Xtr.astype(np.float32),
    ytr.astype(np.int64),
    num_classes=K,
    lam=lam,
)

dp = dp_release_ridge_prototypes_gaussian(
    mus, counts,
    r=float(r),
    lam=float(lam),
    eps=2.0, delta=1e-6,
    sigma_method="analytic",
)

mus_private = dp["mus_noisy"]
yhat = predict_nearest_prototype(Xte.astype(np.float32), mus_private)
```

