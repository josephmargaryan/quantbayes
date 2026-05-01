# Nonconvex Poisson Ball-SGD path

This guide covers theorem-backed Poisson Ball-SGD releases, transcript attacks, and Ball-ReRo reports. The important repo-level policy is:

> Nonconvex transcript attacks use the same canonical finite-prior setup as convex attacks. Only the observation model and scorer differ.

Use `quantbayes.ball_dp.attacks.finite_prior_setup` to build `FinitePriorTrial` objects, then train the nonconvex release on `trial.X_full, trial.y_full`.

---

## 1. Observation model and theorem routes

At step $t$, Poisson Ball-SGD releases a noisy clipped gradient sum,

$$
\widetilde S_t(D)=\sum_{i\in B_t}\bar g_t(z_i)+\xi_t,
\qquad
\xi_t\sim\mathcal N(0,\nu_t^2 I).
$$

The stepwise Ball sensitivity is bounded by

$$
\Delta_t(r)\le \min\{L_z r,2C_t\}.
$$

The main reporting routes are:

- `ball_rero(..., mode="rdp")`: optimized Ball-RDP to Ball-ReRo;
- `ball_rero(..., mode="ball_sgd_direct")`: direct revealed-inclusion one-step profile composition;
- `ball_rero(..., mode="ball_sgd_hidden")`: hidden-inclusion product f-DP/ReRo numerical object;
- `ball_rero(..., mode="ball_sgd_hayes")`: theorem-backed global/product revealed-inclusion dominating-pair object;
- `ball_rero(..., mode="ball_sgd_mix_comp")`: exploratory hidden-mixture per-step composition, not a raw-transcript certificate without an extra domination proof.

For finite-prior exact identification with support size $m$, the baseline is $\kappa=1/m$ for uniform weights.

---

## 2. Canonical finite-prior setup

```python
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

For paper-scale nonconvex experiments, sample a small number of target positions per support. For small demos, enumeration with `policy="all"` is fine.

---

## 3. Calibrate and train a theorem-backed release

```python
import jax.random as jr
from quantbayes.ball_dp.api import calibrate_ball_sgd_noise_multiplier
from quantbayes.ball_dp.theorem import (
    TheoremBounds,
    TheoremModelSpec,
    TrainConfig,
    certified_lz,
    make_model,
    fit_release,
)
from quantbayes.ball_dp.attacks.gradient_based import DPSGDTraceRecorder

orders = tuple(range(2, 13))

spec = TheoremModelSpec(
    d_in=int(trial.X_full.shape[1]),
    hidden_dim=128,
    task="binary",       # or "multiclass"
    parameterization="dense",
    constraint="op",
    num_classes=int(num_classes),
)

bounds = TheoremBounds(B=float(B_public), A=4.0, Lambda=4.0)
lz = float(certified_lz(spec, bounds))

cal = calibrate_ball_sgd_noise_multiplier(
    dataset_size=len(trial.X_full),
    radius=float(radius),
    lz=lz,
    num_steps=num_steps,
    batch_size=batch_size,
    clip_norm=clip_norm,
    target_epsilon=target_epsilon,
    delta=delta,
    privacy="ball_rdp",
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
    orders=orders,
)
noise_multiplier = float(cal["noise_multiplier"])

recorder = DPSGDTraceRecorder(
    capture_every=capture_every,
    keep_models=True,
    keep_batch_indices=True,
)

model = make_model(spec, key=jr.PRNGKey(seed), init_project=True, bounds=bounds)

release = fit_release(
    model,
    spec,
    bounds,
    trial.X_full,
    trial.y_full,
    X_eval=X_public,
    y_eval=y_public,
    train_cfg=TrainConfig(
        radius=float(radius),
        privacy="ball_rdp",
        delta=float(delta),
        num_steps=int(num_steps),
        batch_size=int(batch_size),
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
        clip_norm=float(clip_norm),
        noise_multiplier=float(noise_multiplier),
        learning_rate=float(learning_rate),
        eval_every=eval_every,
        normalize_noisy_sum_by="batch_size",
        seed=int(seed),
    ),
    orders=orders,
    trace_recorder=recorder,
)
```

---

## 4. Build a residualized transcript

The finite-prior transcript attacks expect a residualized trace: known non-target batch gradients are subtracted, leaving the target contribution plus Gaussian noise or the hidden-mixture alternative.

```python
from quantbayes.ball_dp.api import make_trace_metadata_from_release
from quantbayes.ball_dp.attacks.gradient_based import subtract_known_batch_gradients
from quantbayes.ball_dp.types import ArrayDataset

metadata = make_trace_metadata_from_release(
    release,
    target_index=int(trial.target_index),
)

trace = recorder.to_trace(
    state=release.extra.get("model_state", None),
    loss_name=spec.loss_name,
    reduction="mean",
    metadata=metadata,
)

residual_trace = subtract_known_batch_gradients(
    trace,
    ArrayDataset(trial.X_full, trial.y_full, name="attack_train"),
    target_index=int(trial.target_index),
    loss_name=spec.loss_name,
    seed=0,
)
```

---

## 5. Run known- and unknown-inclusion attacks

Unknown inclusion is the default hidden-subsampling threat model. Known inclusion is an oracle stress test.

```python
from quantbayes.ball_dp.api import attack_nonconvex_finite_prior_trial
from quantbayes.ball_dp.attacks.ball_policy import BallTraceMapAttackConfig

unknown = attack_nonconvex_finite_prior_trial(
    residual_trace,
    trial,
    cfg=BallTraceMapAttackConfig(
        mode="unknown_inclusion",
        step_mode="all",
        seed=seed,
    ),
    known_label=int(trial.support.center_y),
    eta_grid=(0.5,),
)

known = attack_nonconvex_finite_prior_trial(
    residual_trace,
    trial,
    cfg=BallTraceMapAttackConfig(
        mode="known_inclusion",
        step_mode="present_steps",
        seed=seed,
    ),
    known_label=int(trial.support.center_y),
    eta_grid=(0.5,),
)
```

The wrappers add source-ID metrics:

```python
unknown.metrics["source_exact_identification_success"]
unknown.diagnostics["predicted_source_id"]
unknown.diagnostics["target_source_id"]
```

---

## 6. Theorem-backed ReRo reports

```python
from quantbayes.ball_dp import ball_rero, make_finite_identification_prior

finite_prior = make_finite_identification_prior(
    trial.support.X,
    weights=trial.support.weights,
)

report_rdp = ball_rero(release, prior=finite_prior, eta_grid=(0.5,), mode="rdp")
report_direct = ball_rero(release, prior=finite_prior, eta_grid=(0.5,), mode="ball_sgd_direct")
report_hidden = ball_rero(release, prior=finite_prior, eta_grid=(0.5,), mode="ball_sgd_hidden")
```

For final tables, avoid mixing known-inclusion attack success with hidden-inclusion bounds unless the observation models are clearly separated.

---

## 7. Runtime notes

The expensive step is usually `subtract_known_batch_gradients(...)`. The rough cost is

$$
(\#\text{retained steps})(\text{average retained batch size})
+
(\#\text{retained steps})(\#\text{candidates}).
$$

Good demo defaults:

- `capture_every = 5` to `20`;
- `m = 6` to `16`;
- `num_steps = 100` to `500` for notebooks;
- `target_policy="sample"` for expensive runs.

---

## 8. Tutorial

See:

```text
examples/ball_dp/nonconvex_synthetic_transcript_ball_dp_end_to_end_demo.ipynb
```

Replace only the synthetic data-loading cell for real experiments.
