# Convex Ball-DP path

This README is the focused guide for the convex path.
It assumes you are using one of the convex families supported by `fit_convex(...)`:
- `ridge-prototype`
- `softmax_logistic`
- `binary_logistic`
- `squared_hinge`

## 1. Theorem routes used in this path

For Gaussian output perturbation
$$
\widetilde\theta = \widehat\theta(D)+\xi,
\qquad
\xi\sim\mathcal N(0,\sigma^2 I),
$$
we use:

### 1.1 Generic Ball-DP $\Rightarrow$ Ball-ReRo
$$
p_{\mathrm{succ}}(\eta)
\le
 e^\varepsilon\kappa(\eta)+\delta.
$$

**`ball_rero(..., mode="dp")`**

### 1.2 Direct Gaussian Ball-ReRo
$$
p_{\mathrm{succ}}(\eta)
\le
\Phi\!\left(\Phi^{-1}(\kappa(\eta))+\frac{\Delta_2(r)}{\sigma}\right).
$$

**`ball_rero(..., mode="gaussian_direct")`**

For high-dimensional embeddings, the finite-prior exact-identification setting is usually the most interpretable primary result.

---

## 2. Train a convex Ball-DP release

```python
import numpy as np
from sklearn.model_selection import train_test_split
from quantbayes.ball_dp import fit_convex

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

B_public = float(np.max(np.linalg.norm(X_train, axis=1)))
num_classes = int(len(np.unique(y_train)))
radius = 0.10

release = fit_convex(
    X_train,
    y_train,
    X_eval=X_test,
    y_eval=y_test,
    model_family="ridge-prototype",
    privacy="ball_dp",
    radius=radius,
    lam=1e-2,
    epsilon=3.0,
    delta=1e-6,
    embedding_bound=B_public,
    num_classes=num_classes,
    solver="lbfgs_fullbatch",
    max_iter=200,
    early_stop=True,
    stop_rule="grad_only",
    min_iter=20,
    seed=0,
)

print("utility:", release.utility_metrics)
print("delta_ball:", release.sensitivity.delta_ball)
print("delta_std:", release.sensitivity.delta_std)
```

---

## 3. Primary embedding result: finite-prior exact identification

```python
from quantbayes.ball_dp import (
    attack_convex_ball_output_finite_prior,
    ball_rero,
    make_finite_identification_prior,
)

target_index = 17
target_label = int(y_train[target_index])

mask = (y_test == target_label)
X_candidates = np.asarray(X_test[mask], dtype=np.float32)
y_candidates = np.asarray(y_test[mask], dtype=np.int32)
X_candidates = np.concatenate([X_candidates, X_train[target_index:target_index+1]], axis=0)
y_candidates = np.concatenate([y_candidates, y_train[target_index:target_index+1]], axis=0)

prior = make_finite_identification_prior(X_candidates, weights=None)

attack, _, _ = attack_convex_ball_output_finite_prior(
    release,
    X_train,
    y_train,
    target_index=target_index,
    X_candidates=X_candidates,
    y_candidates=y_candidates,
    prior_weights=None,
    known_label=target_label,
    eta_grid=(0.5,),
)

report_dp = ball_rero(release, prior=prior, eta_grid=(0.5,), mode="dp")
report_dir = ball_rero(release, prior=prior, eta_grid=(0.5,), mode="gaussian_direct")

print("empirical exact-ID success:", attack.metrics.get("exact_identification_success"))
print("generic DP bound:", report_dp.points[0].gamma_ball)
print("direct Gaussian bound:", report_dir.points[0].gamma_ball)
```

---

## 4. Optional geometry diagnostic: continuous prior

For a continuous uniform Ball prior in dimension \$d\$,
$$
\kappa(\eta)=\left(\frac{\eta}{r}\right)^d
\qquad (\eta<r).
$$
This is useful for intuition but often not the best primary embedding result.

```python
from quantbayes.ball_dp import ball_rero, make_uniform_ball_prior

x_target = np.asarray(X_train[target_index], dtype=np.float32)
prior_cont = make_uniform_ball_prior(center=x_target, radius=radius)
eta_grid = tuple(float(radius * q) for q in (0.90, 0.95, 0.97, 0.98, 0.99, 0.995))

report_cont = ball_rero(
    release,
    prior=prior_cont,
    eta_grid=eta_grid,
    mode="gaussian_direct",
)

for point in report_cont.points:
    print(point.eta / radius, point.kappa, point.gamma_ball)
```
