# Nonconvex Poisson Ball-SGD path

This README is the focused guide for theorem-backed Poisson Ball-SGD releases and attacks.

## 1. Theorems used here

At step $t$, Poisson Ball-SGD releases
```math
\widetilde S_t(D)=\sum_{i\in B_t}\bar g_t(z_i)+\xi_t,
\qquad
\xi_t\sim\mathcal N(0,\nu_t^2 I).
```
The stepwise Ball sensitivity is
```math
\Delta_t(r)\le \min\{L_z r, 2C\}.
```
### 1.1 Ball-RDP → Ball-ReRo
If the transcript-level Ball-RDP curve is
```math
\alpha \mapsto \varepsilon_{1:T}^{\mathrm{ball}}(\alpha;r),
```
then the generic theorem gives
```math
p_{\mathrm{succ}}(\eta)
\le
\inf_\alpha
\min\left\{
1,
\exp\!\left(
\frac{\alpha-1}{\alpha}[\log \kappa(\eta)+\varepsilon_{1:T}^{\mathrm{ball}}(\alpha;r)]
\right)
\right\}.
```
**`ball_rero(..., mode="rdp")`**

### 1.2 Direct Poisson Ball-SGD → Ball-ReRo
The direct theorem composes
```math
\Gamma_t^{\mathrm{ball}}(\kappa;r)=\Psi_{\gamma_t,\Delta_t(r)/\nu_t}(\kappa)
```
into
```math
\Gamma_{1:T}^{\mathrm{ball}}(\kappa;r)
=
(\Gamma_1^{\mathrm{ball}}\circ\cdots\circ\Gamma_T^{\mathrm{ball}})(\kappa).
```
**`ball_rero(..., mode="ball_sgd_direct")`**

In long adaptive traces this direct route is often looser than the RDP-based one.

---

## 2. Calibrate the noise multiplier from the accountant

```python
import numpy as np
import jax.random as jr

from quantbayes.ball_dp.api import calibrate_ball_sgd_noise_multiplier
from quantbayes.ball_dp.theorem import TheoremBounds, TheoremModelSpec, TrainConfig, certified_lz, make_model, fit_release

orders = tuple(range(2, 13))
feature_dim = int(X_train.shape[1])
num_classes = int(len(np.unique(y_train)))
B_public = float(np.max(np.linalg.norm(X_train, axis=1)))
radius = 0.10

spec = TheoremModelSpec(
    d_in=feature_dim,
    hidden_dim=128,
    task="multiclass",
    parameterization="dense",
    constraint="op",
    num_classes=num_classes,
)
bounds = TheoremBounds(B=B_public, A=4.0, Lambda=4.0)
lz = certified_lz(spec, bounds)

cal = calibrate_ball_sgd_noise_multiplier(
    dataset_size=len(X_train),
    radius=radius,
    lz=lz,
    num_steps=400,
    batch_size=min(128, len(X_train)),
    clip_norm=1.0,
    target_epsilon=3.0,
    delta=1e-6,
    privacy="ball_dp",
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
    orders=orders,
)

noise_multiplier = float(cal["noise_multiplier"])
print("L_z:", lz)
print("noise multiplier:", noise_multiplier)
```

---

## 3. Train a theorem-backed release

```python
from quantbayes.ball_dp.attacks.gradient_based import DPSGDTraceRecorder

recorder = DPSGDTraceRecorder(
    capture_every=10,
    keep_models=True,
    keep_batch_indices=True,
)

model = make_model(spec, key=jr.PRNGKey(0), init_project=True, bounds=bounds)

release = fit_release(
    model,
    spec,
    bounds,
    X_train,
    y_train,
    X_eval=X_test,
    y_eval=y_test,
    train_cfg=TrainConfig(
        radius=radius,
        privacy="ball_rdp",
        delta=1e-6,
        num_steps=400,
        batch_size=min(128, len(X_train)),
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
        clip_norm=1.0,
        noise_multiplier=noise_multiplier,
        learning_rate=3e-3,
        eval_every=50,
        normalize_noisy_sum_by="batch_size",
        seed=0,
    ),
    orders=orders,
    trace_recorder=recorder,
)

print("utility:", release.utility_metrics)
```

---

## 4. Primary attack: unknown-inclusion finite prior

Unknown inclusion is the default theorem-aligned threat model for Poisson Ball-SGD.
The target inclusion bits are latent Bernoulli variables, so the attack uses all retained steps.

```python
from quantbayes.ball_dp import (
    ArrayDataset,
    attack_nonconvex_ball_trace_finite_prior,
    make_finite_identification_prior,
    make_trace_metadata_from_release,
)
from quantbayes.ball_dp.attacks.gradient_based import subtract_known_batch_gradients
from quantbayes.ball_dp.attacks.ball_policy import BallTraceMapAttackConfig

train_ds = ArrayDataset(X_train, y_train, name="train")

target_index = 17
target_label = int(y_train[target_index])
true_record = train_ds.record(target_index)

mask = (y_test == target_label)
X_candidates = np.asarray(X_test[mask], dtype=np.float32)
y_candidates = np.asarray(y_test[mask], dtype=np.int32)
X_candidates = np.concatenate([X_candidates, X_train[target_index:target_index+1]], axis=0)
y_candidates = np.concatenate([y_candidates, y_train[target_index:target_index+1]], axis=0)
prior = make_finite_identification_prior(X_candidates, weights=None)

trace = recorder.to_trace(
    state=None,
    loss_name=spec.loss_name,
    reduction="mean",
    metadata=make_trace_metadata_from_release(release, target_index=target_index),
)
residual_trace = subtract_known_batch_gradients(trace, train_ds, target_index=target_index)

attack = attack_nonconvex_ball_trace_finite_prior(
    residual_trace,
    X_candidates,
    y_candidates,
    prior_weights=None,
    cfg=BallTraceMapAttackConfig(mode="unknown_inclusion", seed=0),
    target_index=target_index,
    known_label=target_label,
    true_record=true_record,
    eta_grid=(0.5,),
)

print("status:", attack.status)
print("metrics:", attack.metrics)
```

---

## 5. Theorem-backed ReRo reports

```python
from quantbayes.ball_dp import ball_rero

report_rdp = ball_rero(release, prior=prior, eta_grid=(0.5,), mode="rdp")
report_dir = ball_rero(release, prior=prior, eta_grid=(0.5,), mode="ball_sgd_direct")

print("RDP->ReRo bound:", report_rdp.points[0].gamma_ball)
print("direct Ball-SGD bound:", report_dir.points[0].gamma_ball)
print("alpha_opt_ball:", report_rdp.points[0].alpha_opt_ball)
```

---

## 6. Runtime notes

The expensive step is usually
`subtract_known_batch_gradients(...)`, not the finite-prior score-each-candidate call.
The rough attack scale is
```math
(\#\text{retained steps})\times(\text{avg retained batch size})
+
(\#\text{retained steps})\times(\#\text{candidates}).
```
Good demo defaults:
- `batch_size = 128` or `256`
- `capture_every = 10` or `20`
- candidate support size `m = 8` to `16`

The **known-inclusion** attack is best interpreted as an oracle stress test, not the default threat model.

