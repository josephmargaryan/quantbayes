# Quickstart: Ball-Adjacency DP on NumPy embeddings (X, y)

This repo implements **Ball-Adjacency Differential Privacy (Ball-DP)** for embedding + head / retrieval pipelines.
You provide:
- `X` : numpy array of shape `(n, d)` containing **embeddings**
- `y` : numpy array of shape `(n,)` containing labels (integers recommended)

Ball-DP is standard (ε, δ)-DP but with a *local substitution adjacency*:
two datasets are neighbors if they differ in one record and the substituted records are within distance `r`.

---

## Install

```bash
%%capture
!pip install --upgrade pip
!pip install equinox diffrax numpyro augmax catboost optuna
!git clone https://github.com/josephmargaryan/quantbayes.git
%cd quantbayes
!pip install -e . --no-deps
```
## Minimal Assumptions / best practice
- If you want to compare against the standard bounded replacement baseline `r_std = 2B` you should enforce a public embedding norm bound `||x|| <= B` e.g. by L2-normalizing or L2-clipping.
Example: L2-normalize embeddings so that `B=1 `
```
import numpy as np

eps = 1e-12
X = X.astype(np.float32)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)  # now ||x|| ~ 1
B_public = 1.0
```
## 1) Choose Ball radii from within-class nearest neighbors
We recommend choosing `r` as a percentile of within-class nearest-neighbor distances.
This makes the threat model auditable and easy to communicate.
```
import numpy as np
from quantbayes.ball_dp.api import compute_radius_policy

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)

policy = compute_radius_policy(
    X, y,
    percentiles=(10, 25, 50, 75, 90),
    nn_sample_per_class=400,
    seed=0,
    # DP-valid bounded replacement baseline:
    B_mode="public",
    B_public=1.0,
)

print("Ball radii:", policy.r_values)
print("Bounded replacement baseline r_std=2B:", policy.r_std)
print("NN dist summary:", policy.nn_dists_summary)

# Pick a default policy radius (median NN distance):
r = policy.r_values[50.0]
```
## 2) DP release of a trained ERM parameter vector (Gaussian output perturbation)
If you trained any strongly convex ERM head (e.g., logistic regression / softmax linear head with L2 regularization),
and you have the trained parameter vector `theta_hat`, you can release it privately via:
```
import numpy as np
from quantbayes.ball_dp.api import dp_release_erm_params_gaussian

# Example inputs:
theta_hat = np.asarray(theta_hat, dtype=np.float32).reshape(-1)
n = X.shape[0]

# Ball-DP parameters
eps_dp = 1.0
delta_dp = 1e-5
lam = 1e-2     # must be > 0 (strong convexity from L2 regularization)

# Lipschitz-in-data gradient constant Lz depends on the head + embedding norm bound.
# For softmax linear classifier (multiclass) with L2-regularized weights+bias:
from quantbayes.ball_dp.lz import lz_softmax_linear_bound
Lz = lz_softmax_linear_bound(B=1.0, lam=lam, include_bias=True)

out = dp_release_erm_params_gaussian(
    theta_hat,
    lz=Lz,
    r=r,
    lam=lam,
    n=n,
    eps=eps_dp,
    delta=delta_dp,
    sigma_method="analytic",   # or "classic"
)

theta_private = out["params_noisy"]
print("Sensitivity Δ2:", out["Delta"])
print("Noise σ:", out["sigma"])
```
Important: The DP guarantee applies to the mechanism `M(D) = θ̂(D) + N(0, σ²I)`.
So `theta_hat` should correspond to the deterministic ERM solution (or a very accurate approximation).
## 3) DP release of ridge prototypes (closed-form)
If your head is a **prototype classifier** (one prototype per class), you can use the exact closed-form sensitivity.
``` 
import numpy as np
from quantbayes.ball_dp.heads.prototypes import fit_ridge_prototypes
from quantbayes.ball_dp.api import dp_release_ridge_prototypes_gaussian

K = int(np.max(y) + 1)
lam = 1e-2
eps_dp = 1.0
delta_dp = 1e-5

mus, counts = fit_ridge_prototypes(X, y, num_classes=K, lam=lam)

out = dp_release_ridge_prototypes_gaussian(
    mus, counts,
    r=r,
    lam=lam,
    eps=eps_dp,
    delta=delta_dp,
    sigma_method="analytic",
)

mus_private = out["mus_noisy"]
print("Exact sensitivity Δ2:", out["Delta"])
print("Noise σ:", out["sigma"])
```
## 4) Private top-k retrieval over a corpus embedding matrix
If you have a private corpus `V ∈ R^{m×d}` and you want to answer queries `Q ∈ R^{B×d}` with DP guarantees w.r.t.
substitution of one corpus record within distance `r`:
```
import numpy as np
from quantbayes.retrieval_dp.api import make_topk_retriever

V = np.asarray(V, dtype=np.float32)  # corpus
Q = np.asarray(Q, dtype=np.float32)  # query embeddings

retr = make_topk_retriever(
    V=V,
    score="neg_l2",          # sensitivity Δu <= r (independent of q)
    mechanism="gaussian",    # (ε,δ)-DP
    r=r,
    eps=1.0,
    delta=1e-5,
    sigma_method="analytic",
)

idx = retr.query_many(Q, k=10)  # returns (B,10) indices
print(idx[:2])
```

## For equinox models 
```
params, static = eqx.partition(trained_model, eqx.is_inexact_array)

# Flatten params -> 1D vector (JAX array) and keep an unravel function
theta_jax, unravel = ravel_pytree(params)

# Convert theta to numpy for quantbayes DP helper
theta_hat = np.asarray(jax.device_get(theta_jax), dtype=np.float32)

out = dp_release_erm_params_gaussian(
    theta_hat,
    lz=Lz,
    r=r,
    lam=lam,
    n=n,
    eps=eps_dp,
    delta=delta_dp,
    sigma_method="analytic",  # or "classic"
)

theta_private = out["params_noisy"]  # numpy vector

params_private = unravel(jnp.asarray(theta_private))
dp_model = eqx.combine(params_private, static) # Now we can predict with dp_model

print("DP release complete.")
print("Sensitivity Δ2:", out["Delta"])
print("Noise σ:", out["sigma"])
```
