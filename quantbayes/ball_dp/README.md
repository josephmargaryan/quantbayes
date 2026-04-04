# Ball-DP experiment recipes

This file is the current paper-facing usage guide.
It assumes your dataset is already loaded into memory as `X` and `y`.

The code blocks below are meant to be copied into notebook cells.

## 0. Common setup

```python
import numpy as np

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

seed = 0
rng = np.random.default_rng(seed)
perm = rng.permutation(len(X))

n_eval = max(1, int(0.1 * len(X)))
train_idx = perm[:-n_eval]
eval_idx = perm[-n_eval:]

X_train = X[train_idx]
y_train = y[train_idx]
X_eval = X[eval_idx]
y_eval = y[eval_idx]

feature_shape = tuple(int(v) for v in X_train.shape[1:])
num_classes = int(np.max(y_train)) + 1

# Pick the target record inside X_train.
target_index = 17
assert 0 <= target_index < len(X_train)

# Paper-side Ball policy.
radius = 1.0

# Example side-information center u:
# use a same-label class mean in embedding space.
target_label = int(y_train[target_index])
u = np.asarray(X_train[y_train == target_label].mean(axis=0), dtype=np.float32)

eta_grid = (0.1, 0.2, 0.5, 1.0)
```

## 0.5. Pick the Ball radius `r` in embedding space

Under the paper's label-preserving metric, cross-label distances are irrelevant:
only within-label Euclidean distances matter.
The helper below reports pooled within-label quantiles, worst-class quantiles,
and within-label maxima so you can choose `r` explicitly.

```python
from quantbayes.ball_dp.api import summarize_embedding_ball_radii, select_ball_radius

radius_report = summarize_embedding_ball_radii(
    X_train,
    y_train,
    quantiles=(0.5, 0.8, 0.9, 0.95, 0.99, 1.0),
    max_exact_pairs=250_000,
    max_sampled_pairs=100_000,
    seed=seed,
)

# Recommended default for one universal policy radius:
# take a conservative within-label quantile and then the worst class.
radius = select_ball_radius(
    radius_report,
    strategy="max_labelwise_quantile",
    quantile=0.95,
)

print("selected radius:", radius)
print("candidate radii:")
for k, v in radius_report["candidate_radii"].items():
    print(f"  {k}: {v:.6f}")

print("per-label summaries:")
for row in radius_report["per_label"]:
    print(
        row["label"],
        row["n_examples"],
        row["pair_sampling_mode"],
        row["quantiles"],
    )
```

Interpretation:
- `pooled_q...` means the corresponding quantile for a random same-label pair.
- `max_labelwise_q...` means: compute that quantile inside each label, then take the worst class.
  This is usually the better default for a single universal policy radius.
- `global_max_exact` is the within-label diameter. It is valid but usually too outlier-sensitive.

## 1. Convex exact equation-solving attack

```python
from quantbayes.ball_dp.api import fit_convex, attack_convex

convex_release = fit_convex(
    X_train,
    y_train,
    X_eval=X_eval,
    y_eval=y_eval,
    model_family="softmax-logreg",
    privacy="noiseless",
    solver="lbfgs_fullbatch",
    radius=radius,
    lam=0.01,
    epsilon=3.0,
    delta=1e-6,
    embedding_bound=None,
    seed=seed,
    max_iter=51,
    early_stop=True,
    stop_rule="grad_only",
    min_iter=50,
    grad_tol=1e-6,
    param_tol=None,
    objective_tol=None,
)

attack, d_minus, true_record = attack_convex(
    convex_release,
    X_train,
    y_train,
    target_index=target_index,
    known_label=target_label,
    eta_grid=eta_grid,
)

print("status:", attack.status)
print("predicted label:", attack.y_hat)
print("metrics:", attack.metrics)
```

## 2. Convex Ball-Attack I: Gaussian output MAP on a Ball prior

This is the continuous Ball-constrained MAP attack for Gaussian output perturbation.
The feasible set is the Euclidean ball only.

```python
from quantbayes.ball_dp.api import (
    fit_convex,
    attack_convex_ball_output,
    make_uniform_ball_attack_prior,
    BallOutputMapAttackConfig,
)

prior = make_uniform_ball_attack_prior(
    center=u,
    radius=radius,
)

convex_release = fit_convex(
    X_train,
    y_train,
    model_family="ridge-prototype",
    radius=radius,
    lam=1.0,
    privacy="ball-dp",
    epsilon=3.0,
    delta=1e-5,
    embedding_bound=None,
    num_classes=num_classes,
    seed=seed,
)

attack, d_minus, true_record = attack_convex_ball_output(
    convex_release,
    X_train,
    y_train,
    target_index=target_index,
    prior=prior,
    cfg=BallOutputMapAttackConfig(
        optimizer="adam",
        num_steps=250,
        learning_rate=5e-2,
        num_restarts=3,
        seed=seed,
    ),
    known_label=target_label,
    eta_grid=eta_grid,
)

print("status:", attack.status)
print("predicted label:", attack.y_hat)
print("metrics:", attack.metrics)
print("objective gap to truth:", attack.metrics.get("objective_gap_to_truth"))
```

## 3. Convex Ball-Attack I: exact finite-prior Bayes classifier

Use this when the theorem-aligned prior is a finite candidate set.
This is exact score-each-candidate Bayes classification, not projected gradient descent.

```python
from quantbayes.ball_dp.api import attack_convex_ball_output_finite_prior

# Example finite prior: same-label candidates from the evaluation split.
mask = (y_eval == target_label)
X_candidates = np.asarray(X_eval[mask], dtype=np.float32)
y_candidates = np.asarray(y_eval[mask])

# Ensure the true target is in the candidate support if you want exact-ID metrics.
X_candidates = np.concatenate([X_candidates, X_train[target_index:target_index+1]], axis=0)
y_candidates = np.concatenate([y_candidates, y_train[target_index:target_index+1]], axis=0)

attack, d_minus, true_record = attack_convex_ball_output_finite_prior(
    convex_release,
    X_train,
    y_train,
    target_index=target_index,
    X_candidates=X_candidates,
    y_candidates=y_candidates,
    prior_weights=None,   # uniform finite prior
    known_label=target_label,
    eta_grid=eta_grid,
)

print("status:", attack.status)
print("metrics:", attack.metrics)
print("oblivious_kappa:", attack.metrics.get("oblivious_kappa"))
print("exact_identification_success:", attack.metrics.get("exact_identification_success"))
```

## 4. Common nonconvex model definition

Replace this with your actual architecture.

```python
import equinox as eqx
import jax
import jax.random as jr

class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, key, in_dim: int, hidden_dim: int, out_dim: int):
        k1, k2 = jr.split(key)
        self.fc1 = eqx.nn.Linear(in_dim, hidden_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_dim, out_dim, key=k2)

    def __call__(self, x, *, key=None, state=None):
        x = jnp.asarray(x, dtype=jnp.float32).reshape(-1)
        h = jax.nn.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits, state


def build_model(seed: int):
    model = MLP(
        jr.PRNGKey(seed),
        int(np.prod(feature_shape)),
        128,
        num_classes,
    )
    return model, None

model, state = build_model(seed)
```

## 5. Main paper nonconvex release: matched Poisson Ball-SGD

This is the theorem-backed mainline.
Use Poisson subsampling and matched accounting in the actual paper runs.

```python
import optax

from quantbayes.ball_dp.api import fit_ball_sgd
from quantbayes.ball_dp.attacks.gradient_based import DPSGDTraceRecorder

# Replace lz with your theorem-backed value.
lz = 1.0
batch_size = 128
num_steps = 2000

recorder = DPSGDTraceRecorder(
    capture_every=1,
    keep_models=True,
    keep_batch_indices=True,
)

sgd_release = fit_ball_sgd(
    model,
    optax.adam(1e-3),
    X_train,
    y_train,
    X_eval=X_eval,
    y_eval=y_eval,
    privacy="ball_rdp",
    radius=radius,
    lz=lz,
    num_steps=num_steps,
    batch_size=batch_size,
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
    clip_norm=1.0,
    noise_multiplier=1.0,
    loss_name="softmax_cross_entropy",
    normalize_noisy_sum_by="batch_size",
    trace_recorder=recorder,
    state=state,
    seed=seed,
)
```

## 6. Baseline trace optimization attack (non-Ball baseline)

This is not the theorem-aligned Ball MAP attack. It is the older gradient-based baseline.

```python
from quantbayes.ball_dp.api import make_trace_metadata_from_release
from quantbayes.ball_dp.types import ArrayDataset
from quantbayes.ball_dp.attacks.gradient_based import (
    subtract_known_batch_gradients,
    TraceOptimizationAttackConfig,
    run_trace_optimization_attack,
)

trace = recorder.to_trace(
    state=state,
    loss_name="softmax_cross_entropy",
    reduction="mean",
    metadata=make_trace_metadata_from_release(
        sgd_release,
        target_index=target_index,
    ),
)

residual_trace = subtract_known_batch_gradients(
    trace,
    ArrayDataset(X_train, y_train, name="train"),
    target_index=target_index,
)

true_record = residual_trace.metadata.get("true_record", None)
if true_record is None:
    from quantbayes.ball_dp.types import Record
    true_record = Record(features=X_train[target_index], label=int(y_train[target_index]))

attack = run_trace_optimization_attack(
    residual_trace,
    cfg=TraceOptimizationAttackConfig(
        step_mode="present_steps",
        num_steps=800,
        learning_rate=1e-2,
        num_restarts=5,
        seed=seed,
    ),
    feature_shape=feature_shape,
    known_label=target_label,
    true_record=true_record,
    target_index=target_index,
    eta_grid=eta_grid,
)

print("status:", attack.status)
print("metrics:", attack.metrics)
```

## 7. Ball-Attack II: known-inclusion transcript MAP

This is the exact Ball posterior attack on the residualized stored sanitized trace.
Use a Ball prior and the theorem-aligned trace metadata helper.

```python
from quantbayes.ball_dp.api import make_uniform_ball_attack_prior
from quantbayes.ball_dp.attacks.ball_policy import (
    BallTraceMapAttackConfig,
    run_ball_trace_map_attack,
)

prior = make_uniform_ball_attack_prior(
    center=u,
    radius=radius,
)

attack = run_ball_trace_map_attack(
    residual_trace,
    prior=prior,
    cfg=BallTraceMapAttackConfig(
        mode="known_inclusion",
        optimizer="adam",
        num_steps=500,
        learning_rate=1e-2,
        num_restarts=3,
        step_mode="present_steps",
        seed=seed,
    ),
    target_index=target_index,
    known_label=target_label,
    true_record=true_record,
    eta_grid=eta_grid,
)

print("status:", attack.status)
print("metrics:", attack.metrics)
print("diagnostics:", attack.diagnostics)
```

## 8. Ball-Attack II: known-inclusion finite-prior exact Bayes classifier

```python
from quantbayes.ball_dp.api import attack_nonconvex_ball_trace_finite_prior
from quantbayes.ball_dp.attacks.ball_policy import BallTraceMapAttackConfig

mask = (y_eval == target_label)
X_candidates = np.asarray(X_eval[mask], dtype=np.float32)
y_candidates = np.asarray(y_eval[mask])

X_candidates = np.concatenate([X_candidates, X_train[target_index:target_index+1]], axis=0)
y_candidates = np.concatenate([y_candidates, y_train[target_index:target_index+1]], axis=0)

attack = attack_nonconvex_ball_trace_finite_prior(
    residual_trace,
    X_candidates,
    y_candidates,
    prior_weights=None,
    cfg=BallTraceMapAttackConfig(
        mode="known_inclusion",
        step_mode="present_steps",
        seed=seed,
    ),
    target_index=target_index,
    known_label=target_label,
    true_record=true_record,
    eta_grid=eta_grid,
)

print("status:", attack.status)
print("metrics:", attack.metrics)
print("oblivious_kappa:", attack.metrics.get("oblivious_kappa"))
```

## 9. Ball-Attack III: unknown-inclusion transcript MAP

This is the exact Bernoulli-mixture posterior attack.
When the trace metadata is produced via `make_trace_metadata_from_release(...)`, the per-step Poisson probabilities are already available to the attack.

```python
attack = run_ball_trace_map_attack(
    residual_trace,
    prior=prior,
    cfg=BallTraceMapAttackConfig(
        mode="unknown_inclusion",
        optimizer="adam",
        num_steps=500,
        learning_rate=1e-2,
        num_restarts=3,
        seed=seed,
    ),
    label_space=sorted(np.unique(y_train).tolist()),
    true_record=true_record,
    eta_grid=eta_grid,
)

print("status:", attack.status)
print("metrics:", attack.metrics)
print("diagnostics:", attack.diagnostics)
```

## 10. Ball-Attack III: unknown-inclusion finite-prior exact Bayes classifier

```python
attack = attack_nonconvex_ball_trace_finite_prior(
    residual_trace,
    X_candidates,
    y_candidates,
    prior_weights=None,
    cfg=BallTraceMapAttackConfig(
        mode="unknown_inclusion",
        seed=seed,
    ),
    known_label=None,
    true_record=true_record,
    eta_grid=eta_grid,
)

print("status:", attack.status)
print("metrics:", attack.metrics)
print("oblivious_kappa:", attack.metrics.get("oblivious_kappa"))
```

## 11. Theorem-aligned Ball-ReRo reporting

### 11a. Ball-RDP to Ball-ReRo conversion for a continuous Ball prior

```python
from quantbayes.ball_dp.api import ball_rero, make_uniform_ball_prior

continuous_prior = make_uniform_ball_prior(
    center=np.asarray(u, dtype=np.float32).reshape(-1),
    radius=radius,
)

report = ball_rero(
    sgd_release,
    prior=continuous_prior,
    eta_grid=eta_grid,
    mode="rdp",   # explicit theorem-backed Ball-RDP conversion
)

for point in report.points:
    print(
        "eta=", point.eta,
        "kappa=", point.kappa,
        "gamma_ball=", point.gamma_ball,
        "alpha_opt_ball=", point.alpha_opt_ball,
    )
```

### 11b. Direct Gaussian Ball-ReRo for convex Gaussian output perturbation

This direct mode is theorem-backed only for Gaussian output perturbation releases.
Do not use it for Ball-SGD releases; use `mode="ball_sgd_direct"` below instead.

```python
report_direct = ball_rero(
    convex_release,
    prior=continuous_prior,
    eta_grid=eta_grid,
    mode="gaussian_direct",
)

for point in report_direct.points:
    print("eta=", point.eta, "gamma_ball_dir=", point.gamma_ball)
```

### 11c. Direct Ball-ReRo for Poisson Ball-SGD

This is the theorem-backed direct bound from the Poisson Ball-SGD section.
Use it for nonconvex releases trained with:

- `batch_sampler="poisson"`
- no `fixed_batch_indices_schedule`
- `normalize_noisy_sum_by` in `{"batch_size", "none"}`
- strictly positive noise on every step

The report JSON contains the per-step direct-profile parameters in
`report.metadata["ball_step_profiles"]`.

```python
from quantbayes.ball_dp import plot_rero_report

report_direct_sgd = ball_rero(
    sgd_release,
    prior=continuous_prior,
    eta_grid=eta_grid,
    mode="ball_sgd_direct",
    out_path="results/ball_sgd_direct_rero.json",
)

plot_rero_report(report_direct_sgd, out_path="results/ball_sgd_direct_rero.png")

for point in report_direct_sgd.points:
    print(
        "eta=", point.eta,
        "kappa=", point.kappa,
        "gamma_ball_dir=", point.gamma_ball,
        "gamma_standard_dir=", point.gamma_standard,
    )

print(report_direct_sgd.metadata["ball_step_profiles"][0])
```

If you want to inspect the one-step ingredients directly, `get_release_step_table(...)` now also
includes the per-step direct-profile ratios and breakpoint locations:

```python
from quantbayes.ball_dp.api import get_release_step_table

rows = get_release_step_table(sgd_release)
print(rows[0]["direct_c_ball"], rows[0]["direct_kappa_left_ball"])
```

### 11d. Finite-prior exact-identification certificate

For exact identification under a finite prior, use the theorem-aligned discrete prior helper.
Choose any `eta < 1`.

```python
from quantbayes.ball_dp.api import make_finite_identification_prior

finite_prior = make_finite_identification_prior(X_candidates, weights=None)

report_finite = ball_rero(
    sgd_release,
    prior=finite_prior,
    eta_grid=(0.5,),
    mode="rdp",
)

report_finite_direct = ball_rero(
    sgd_release,
    prior=finite_prior,
    eta_grid=(0.5,),
    mode="ball_sgd_direct",
)

print(report_finite.points[0].kappa)
print(report_finite.points[0].gamma_ball)
print(report_finite_direct.points[0].gamma_ball)
```

## 12. Theorem-aligned empirical summaries across many targets

```python
from quantbayes.ball_dp.api import summarize_attack_trials

all_attacks = [attack]   # append one AttackResult per target / trial
summary = summarize_attack_trials(
    all_attacks,
    eta_grid=eta_grid,
    oblivious_kappa=all_attacks[0].metrics.get("oblivious_kappa"),
)
print(summary)
```

For exact finite-prior identification, the primary metric is
`exact_identification_success`.
For approximate reconstruction, use the thresholded success keys `p_succ@...` returned by `summarize_attack_trials(...)`.
Raw MSE is secondary.

## 13. Optional baselines

### 13a. Model-based reconstruction baseline

```python
import optax

from quantbayes.ball_dp.api import (
    prepare_informed_attack_data,
    train_released_model,
    build_attack_corpus,
    train_reconstructor,
    run_model_based_attack,
)
from quantbayes.ball_dp.config import ReconstructorTrainingConfig, ShadowCorpusConfig
from quantbayes.ball_dp.types import ArrayDataset

data = prepare_informed_attack_data(
    X_train,
    y_train,
    X_target=X_eval,
    y_target=y_eval,
    target_index=0,
    base_train_size=32,
    num_shadow=1,
    seed=seed,
)

target_label = int(data.target.label)
same_label_idx = np.where(np.asarray(y_train) == target_label)[0]
same_label_idx = same_label_idx[~np.isin(same_label_idx, data.base_indices)]

shadow_pool = ArrayDataset(
    X=np.asarray(X_train[same_label_idx], dtype=np.float32),
    y=np.asarray(y_train[same_label_idx]),
    name="same_label_shadow_pool",
)

release_model = train_released_model(
    data.d_minus,
    data.target,
    build_model=build_model,
    optimizer=optax.adam(1e-3),
    seed=seed,
    privacy="noiseless",
    num_steps=800,
    batch_size=len(data.d_minus) + 1,
    clip_norm=1e6,
    noise_multiplier=0.0,
    loss_name="softmax_cross_entropy",
    checkpoint_selection="last",
)

corpus = build_attack_corpus(
    data.d_minus,
    shadow_pool,
    build_model=build_model,
    optimizer=optax.adam(1e-3),
    corpus_cfg=ShadowCorpusConfig(
        num_trials=min(256, len(shadow_pool)),
        train_frac=0.85,
        val_frac=0.10,
        seed=seed,
        store_releases=False,
    ),
    shadow_seed_policy="fixed",
    fixed_shadow_seed=seed,
    seed=seed,
    privacy="noiseless",
    num_steps=800,
    batch_size=len(data.d_minus) + 1,
    clip_norm=1e6,
    noise_multiplier=0.0,
    loss_name="softmax_cross_entropy",
    checkpoint_selection="last",
)

reconstructor = train_reconstructor(
    corpus,
    cfg=ReconstructorTrainingConfig(
        hidden_dims=(512, 512),
        batch_size=64,
        num_epochs=80,
        patience=15,
        learning_rate=1e-3,
        seed=seed,
        target_normalization="l2",
        output_normalization="l2",
        loss_name="cosine",
    ),
)

attack = run_model_based_attack(
    release_model,
    data.d_minus,
    reconstructor=reconstructor,
    true_record=data.target,
    known_label=target_label,
)

print("status:", attack.status)
print("metrics:", attack.metrics)
```

### 13b. SPEAR one-step baseline

```python
import optax

from quantbayes.ball_dp.trace_setup import (
    prepare_targeted_trace_batch,
    make_target_inclusion_schedule,
)
from quantbayes.ball_dp.attacks.gradient_based import DPSGDTraceRecorder
from quantbayes.ball_dp.attacks.spear import (
    SpearAttackConfig,
    run_spear_trace_step_attack,
)

trace_batch = prepare_targeted_trace_batch(
    X_train,
    y_train,
    target_index=target_index,
    batch_size=2,
    seed=seed,
)

fixed_schedule = make_target_inclusion_schedule(
    num_steps=1,
    batch_indices=trace_batch.batch_indices,
    mode="all",
)

recorder_spear = DPSGDTraceRecorder(
    capture_every=1,
    keep_models=True,
    keep_batch_indices=True,
)

model_spear, state_spear = build_model(seed)

_ = fit_ball_sgd(
    model_spear,
    optax.sgd(0.0),
    X_train,
    y_train,
    privacy="noiseless",
    radius=radius,
    lz=lz,
    num_steps=1,
    batch_size=2,
    batch_sampler="without_replacement",
    clip_norm=1e6,
    noise_multiplier=0.0,
    loss_name="softmax_cross_entropy",
    normalize_noisy_sum_by="none",
    fixed_batch_indices_schedule=fixed_schedule,
    trace_recorder=recorder_spear,
    state=state_spear,
    seed=seed,
)

step0 = recorder_spear.steps[0]
true_batch = np.asarray(X_train[step0.batch_indices], dtype=np.float32).reshape(len(step0.batch_indices), -1)

result = run_spear_trace_step_attack(
    step0,
    layer_path=("fc1",),
    cfg=SpearAttackConfig(
        batch_size=2,
        random_seed=seed,
        noisy_mode=False,
    ),
    true_batch=true_batch,
)

print("status:", result.status)
print("metrics:", result.metrics)
print("best_gamma:", result.diagnostics.get("best_gamma"))
```

## 14. Minimal checklist for paper runs

Use these settings for theorem-backed Ball-SGD mainline runs:

```python
batch_sampler="poisson"
accountant_subsampling="match_sampler"
normalize_noisy_sum_by="batch_size"
```
### Sampler/accountant compatibility
- "`batch_sampler="poisson"` $\to$ use `accountant_subsampling="match_sampler"` (mainline/private runs)"
- "`batch_sampler="without_replacement"` $\to$ use `batch_sampler="without_replacement"` for noiseless/public debugging"
`match_sampler` is only valid with Poisson sampling in this library.

Keep the trace recorder metadata needed by the attacks:

```python
DPSGDTraceRecorder(keep_models=True, keep_batch_indices=True)
make_trace_metadata_from_release(sgd_release, target_index=...)
```

For finite-prior exact identification, report:
- `exact_identification_success`
- thresholded success probabilities `p_succ@...`
- `oblivious_kappa`
- MSE only as a secondary descriptive metric.
