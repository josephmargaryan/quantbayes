# Convex Ball-DP path

This guide covers convex Gaussian output perturbation and finite-prior exact-identification attacks. It assumes one of the convex families supported by `fit_convex(...)`:

- `ridge_prototype` / `ridge-prototype`;
- `softmax_logistic`;
- `binary_logistic`;
- `squared_hinge`.

The key update is that convex attacks should use the **shared finite-prior replacement setup** from `quantbayes.ball_dp.attacks.finite_prior_setup`.

---

## 1. Observation model and bounds

Convex Gaussian output perturbation releases

$$
\widetilde \theta = \widehat \theta(D) + \xi,
\qquad
\xi\sim \mathcal N(0,\sigma^2 I).
$$

The relevant Ball-ReRo routes are:

1. generic Ball-DP:

$$
p_{\mathrm{succ}}(\eta)
\le e^\varepsilon \kappa(\eta)+\delta;
$$

2. generic Ball-RDP:

$$
p_{\mathrm{succ}}(\eta)
\le
\inf_\alpha
\min\left\{1,
\left(\kappa(\eta)e^{\varepsilon_\alpha}\right)^{(\alpha-1)/\alpha}
\right\};
$$

3. direct Gaussian testing:

$$
p_{\mathrm{succ}}(\eta)
\le
\Phi\!\left(\Phi^{-1}(\kappa(\eta))+\frac{\Delta_2(r)}{\sigma}\right).
$$

For finite-prior exact identification with uniform support size $m$, $\kappa=1/m$.

---

## 2. Canonical finite-prior setup

Use the shared setup helpers. The support is public-only by default, same-label, and radius-feasible.

```python
import numpy as np
from quantbayes.ball_dp.attacks.finite_prior_setup import (
    CandidateSource,
    find_feasible_replacement_banks,
    select_support_from_bank,
    target_positions_for_support,
    make_replacement_trial,
)

public = CandidateSource("public", X_public, y_public)

bank = find_feasible_replacement_banks(
    X_train=X_train,
    y_train=y_train,
    candidate_sources=[public],
    radius=radius,
    min_support_size=m,
    num_banks=1,
    seed=seed,
    anchor_selection="large_bank",
    strict=True,
)[0]

support = select_support_from_bank(
    bank,
    m=m,
    selection="farthest",
    seed=seed,
    draw_index=0,
)

target_pos = target_positions_for_support(
    support,
    policy="sample",
    num_targets=1,
    seed=seed,
)[0]

trial = make_replacement_trial(
    X_train=X_train,
    y_train=y_train,
    support=support,
    target_support_position=target_pos,
)
```

The trained convex release should use:

```python
X_train=trial.X_full
y_train=trial.y_full
```

The adversary support is:

```python
trial.support.X
trial.support.y
trial.support.weights
```

---

## 3. Train convex noisy ERM

```python
from quantbayes.ball_dp import fit_convex

release = fit_convex(
    trial.X_full,
    trial.y_full,
    model_family="ridge_prototype",
    privacy="ball_dp",
    radius=float(radius),
    lam=1e-2,
    epsilon=4.0,
    delta=1e-6,
    embedding_bound=float(B_public),
    standard_radius=float(standard_radius),
    ridge_sensitivity_mode="global",
    num_classes=int(num_classes),
    orders=tuple(float(v) for v in range(2, 65)),
    max_iter=100,
    solver="lbfgs_fullbatch",
    seed=int(seed),
)
```

For a standard-DP comparator, train the same call with `radius=standard_radius` while keeping the same `trial`.

---

## 4. Run the finite-prior attack

Prefer the trial-object wrapper:

```python
from quantbayes.ball_dp.api import attack_convex_finite_prior_trial

attack = attack_convex_finite_prior_trial(
    release,
    trial,
    known_label=int(trial.support.center_y),
    eta_grid=(0.5,),
)

print("baseline:", attack.metrics["oblivious_kappa"])
print("source exact-ID:", attack.metrics["source_exact_identification_success"])
print("predicted source:", attack.diagnostics["predicted_source_id"])
print("target source:", attack.diagnostics["target_source_id"])
```

This wrapper adds source-ID exact-identification metadata on top of the underlying Gaussian finite-prior Bayes rule.

---

## 5. Compute bounds and diagnostics

```python
from quantbayes.ball_dp import ball_rero, make_finite_identification_prior
from quantbayes.ball_dp.api import diagnose_convex_ball_output_finite_prior

finite_prior = make_finite_identification_prior(
    trial.support.X,
    weights=trial.support.weights,
)

report_dp = ball_rero(release, prior=finite_prior, eta_grid=(0.5,), mode="dp")
report_rdp = ball_rero(release, prior=finite_prior, eta_grid=(0.5,), mode="rdp")
report_direct = ball_rero(
    release,
    prior=finite_prior,
    eta_grid=(0.5,),
    mode="gaussian_direct",
)

diag = diagnose_convex_ball_output_finite_prior(
    release,
    trial.X_full,
    trial.y_full,
    target_index=trial.target_index,
    X_candidates=trial.support.X,
    y_candidates=trial.support.y,
    prior_weights=trial.support.weights,
    known_label=int(trial.support.center_y),
    center_features=trial.support.center_x,
    center_label=int(trial.support.center_y),
)
```

The diagnostics explain why a support is easy or hard: model-space separations, finite Gaussian direct bounds, posterior concentration, and ridge-specific inverse-noise quantities.

---

## 6. Minimal reporting table

```python
row = {
    "support_hash": trial.support.support_hash,
    "m": trial.support.m,
    "baseline_kappa": trial.support.oblivious_kappa,
    "exact_id": attack.metrics.get("source_exact_identification_success"),
    "predicted_source_id": attack.diagnostics.get("predicted_source_id"),
    "target_source_id": attack.diagnostics.get("target_source_id"),
    "gamma_dp_ball": report_dp.points[0].gamma_ball,
    "gamma_rdp_ball": report_rdp.points[0].gamma_ball,
    "gamma_direct_ball": report_direct.points[0].gamma_ball,
    "gamma_direct_standard": report_direct.points[0].gamma_standard,
    "posterior_top1": attack.metrics.get("posterior_top1_probability"),
    "posterior_true": attack.metrics.get("posterior_true_probability"),
    "direct_instance_finite_opt": diag["bound_direct_instance_finite_opt"],
}
```

---

## 7. Tutorial and experiments

Recommended demo notebook in this repo:

```text
quantbayes/ball_dp/examples/convex_demo.ipynb
```

If using the thesis/demo split, the lightweight demo notebook is:

```text
quantbayes/ball_dp/examples/convex_single_workflow_demo.ipynb
```

Paper-scale finite-prior runs should use the experiment scripts rather than the demo notebook, for example:

```text
quantbayes/ball_dp/experiments/convex/run_attack_experiment.py
quantbayes/ball_dp/experiments/convex/run_thesis_experiment.py
```

Replace the synthetic data cell or loader arguments with real embeddings for paper-scale runs.
