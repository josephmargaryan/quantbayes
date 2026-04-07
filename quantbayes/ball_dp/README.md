# Ball-DP: paper-facing guide

This README mirrors the demonstration notebook style and is meant to be the **main entry point** for the project.
It emphasizes the theorem-backed workflows used in the thesis and then, at the end, shows how to run the older non-Ball baselines as well.

Throughout, records are $z=(x,y)$ with the label-preserving metric
```math
d\big((x,y),(x',y')\big)
=
\begin{cases}
\|x-x'\|_2, & y=y',\\
\infty, & y\neq y'.
\end{cases}
```

---

## 1. What the library can certify

The library can upper-bound
```math
p_{\mathrm{succ}}(\eta)
:=
\Pr\big[\rho(Z,\widehat Z)\le \eta\big]
```
through four theorem routes.

### 1.1 Generic Ball-DP → Ball-ReRo
If a mechanism is $(\varepsilon,\delta)$-Ball-DP, then
```math
p_{\mathrm{succ}}(\eta)
\le
 e^\varepsilon\,\kappa(\eta)+\delta.
```

**`ball_rero(..., mode="dp")`**

### 1.2 Generic Ball-RDP → Ball-ReRo
If a mechanism is $(\alpha,\varepsilon_\alpha)$-Ball-RDP, then
```math
p_{\mathrm{succ}}(\eta)
\le
\min\left\{1,\,(\kappa(\eta)e^{\varepsilon_\alpha})^{(\alpha-1)/\alpha}\right\}.
```
With a full RDP curve $\alpha\mapsto\varepsilon(\alpha)$, the library optimizes over $\alpha$.

**`ball_rero(..., mode="rdp")`**

### 1.3 Direct Gaussian Ball-ReRo
For Gaussian output perturbation
```math
\widetilde\theta = \widehat\theta(D)+\xi,
\qquad
\xi\sim\mathcal N(0,\sigma^2 I),
```
with Ball sensitivity $\Delta_2(r)$, the direct theorem gives
```math
p_{\mathrm{succ}}(\eta)
\le
\Phi\!\left(\Phi^{-1}(\kappa(\eta))+\frac{\Delta_2(r)}{\sigma}\right).
```

**`ball_rero(..., mode="gaussian_direct")`**

### 1.4 Direct Poisson Ball-SGD Ball-ReRo
For Poisson Ball-SGD, the theorem composes one-step profiles
```math
\Gamma_t^{\mathrm{ball}}(\kappa;r)=\Psi_{\gamma_t,\Delta_t(r)/\nu_t}(\kappa)
```
into
```math
\Gamma_{1:T}^{\mathrm{ball}}(\kappa;r)
=
(\Gamma_1^{\mathrm{ball}}\circ\cdots\circ\Gamma_T^{\mathrm{ball}})(\kappa).
```
Then
```math
p_{\mathrm{succ}}(\eta)
\le
\Gamma_{1:T}^{\mathrm{ball}}\big(\kappa(\eta);r\big).
```

**`ball_rero(..., mode="ball_sgd_direct")`**

**`ball_rero(..., mode="ball_sgd_mix_comp")`** — exploratory hidden-mixture reference quantity; not theorem-backed for the raw replacement transcript without an additional domination argument.

**`ball_rero(..., mode="ball_sgd_hayes")`** — theorem-backed global/product direct bound for the revealed-inclusion dominating pair (the alias `mode="ball_sgd_kaissis"` returns the same quantity in trade-off language).

### 1.5 Finite-prior exact identification
For a uniform finite prior
```math
\pi_S=\frac1m\sum_{i=1}^m \delta_{z_i}
```
and the exact-identification loss
```math
\rho_{0/1}(z,z')=\mathbf{1}_{\{z\neq z'\}},
```
all $\eta<1$ are equivalent and
```math
\Pr[\rho_{0/1}(Z,\widehat Z)\le \eta]=\Pr[\widehat Z=Z].
```
So the primary empirical quantity is
```math
\widehat p_{\mathrm{exact}}
=
\frac1N\sum_{j=1}^N \mathbf{1}_{\{\widehat Z_j=Z_j\}}.
```
For a uniform prior, $\kappa=1/m$.

**Practical takeaway:**
- for high-dimensional embeddings, the **finite-prior exact-ID** setting is usually the most interpretable primary result;
- the **continuous Ball prior** is still useful as a geometry diagnostic because
  ```math
  \kappa(\eta)=\left(\frac{\eta}{r}\right)^d
  \qquad (\eta<r)
  ```
  changes extremely quickly in large dimension.

---

## 2. Common setup

```python
import numpy as np
from sklearn.model_selection import train_test_split

from quantbayes.ball_dp import (
    ArrayDataset,
    make_finite_identification_prior,
    make_uniform_ball_prior,
    select_ball_radius,
    summarize_embedding_ball_radii,
)

X = np.asarray(X, dtype=np.float32)
if X.ndim != 2:
    X = X.reshape(len(X), -1)

y_raw = np.asarray(y)
label_values, y = np.unique(y_raw, return_inverse=True)
y = y.astype(np.int32)

seed = 0
preferred_target_index = 17
eta_grid = (0.1, 0.2, 0.5, 1.0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

train_ds = ArrayDataset(X_train, y_train, name="train")
test_ds = ArrayDataset(X_test, y_test, name="test")

num_classes = int(len(np.unique(y_train)))
feature_dim = int(X_train.shape[1])
feature_shape = tuple(int(v) for v in X_train.shape[1:])
B_public = float(np.max(np.linalg.norm(X_train, axis=1)))

target_index = int(min(preferred_target_index, len(X_train) - 1))
target_label = int(y_train[target_index])
x_target = np.asarray(X_train[target_index], dtype=np.float32)

radius_report = summarize_embedding_ball_radii(
    X_train,
    y_train,
    quantiles=(0.5, 0.8, 0.9, 0.95, 0.99, 1.0),
    max_exact_pairs=250_000,
    max_sampled_pairs=100_000,
    seed=seed,
)

radius = select_ball_radius(
    radius_report,
    strategy="max_labelwise_quantile",
    quantile=0.80,
)

print("feature_dim:", feature_dim)
print("feature_shape:", feature_shape)
print("num_classes:", num_classes)
print("public theorem bound B:", B_public)
print("chosen Ball radius r:", radius)
print("target_index:", target_index)
print("target_label:", target_label)
```

---

## 3. Convex Gaussian output perturbation

We release
```math
\widetilde\theta = \widehat\theta(D)+\xi,
\qquad
\xi\sim\mathcal N(0,\sigma^2 I).
```
The ERM sensitivity theorem gives
```math
\|\widehat\theta(D)-\widehat\theta(D')\|_2
\le
\Delta_2(r).
```
The two main theorem-backed ReRo routes here are:
- generic Ball-DP $\Rightarrow$ Ball-ReRo;
- direct Gaussian Ball-ReRo.

### 3.1 Train a convex Ball-DP release

```python
from quantbayes.ball_dp import fit_convex

release_convex = fit_convex(
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
    seed=seed,
)

print("release kind:", release_convex.release_kind)
print("utility:", release_convex.utility_metrics)
print("ball sensitivity:", release_convex.sensitivity.delta_ball)
print("standard sensitivity:", release_convex.sensitivity.delta_std)
print("sigma:", release_convex.privacy.ball.sigma)
```

### 3.2 Primary embedding result: finite-prior exact identification

```python
from quantbayes.ball_dp import attack_convex_ball_output_finite_prior, ball_rero

# Example: same-label held-out candidates + the true target.
target_index = int(min(preferred_target_index, len(X_train) - 1))
target_label = int(y_train[target_index])
x_target = np.asarray(X_train[target_index], dtype=np.float32)

mask = (y_test == target_label)
X_candidates = np.asarray(X_test[mask], dtype=np.float32)
y_candidates = np.asarray(y_test[mask], dtype=np.int32)
X_candidates = np.concatenate([X_candidates, x_target[None, :]], axis=0)
y_candidates = np.concatenate([y_candidates, np.asarray([target_label], dtype=np.int32)], axis=0)

finite_prior = make_finite_identification_prior(X_candidates, weights=None)

attack_convex_finite, _, true_record = attack_convex_ball_output_finite_prior(
    release_convex,
    X_train,
    y_train,
    target_index=target_index,
    X_candidates=X_candidates,
    y_candidates=y_candidates,
    prior_weights=None,
    known_label=target_label,
    eta_grid=(0.5,),
)

report_convex_dp = ball_rero(
    release_convex,
    prior=finite_prior,
    eta_grid=(0.5,),
    mode="dp",
)
report_convex_direct = ball_rero(
    release_convex,
    prior=finite_prior,
    eta_grid=(0.5,),
    mode="gaussian_direct",
)

p_dp = report_convex_dp.points[0]
p_dir = report_convex_direct.points[0]

print("m =", len(X_candidates))
print("empirical exact-ID success =", attack_convex_finite.metrics.get("exact_identification_success"))
print("generic DP Ball-ReRo bound (ball) =", p_dp.gamma_ball)
print("direct Gaussian Ball-ReRo bound (ball) =", p_dir.gamma_ball)
print("direct Gaussian comparator (standard) =", p_dir.gamma_standard)
```

### 3.3 Optional geometry diagnostic: continuous prior

```python
oracle_cont_prior = make_uniform_ball_prior(center=x_target, radius=radius)
continuous_eta_grid = tuple(float(radius * q) for q in (0.90, 0.95, 0.97, 0.98, 0.99, 0.995))

report_convex_cont = ball_rero(
    release_convex,
    prior=oracle_cont_prior,
    eta_grid=continuous_eta_grid,
    mode="gaussian_direct",
)

for point in report_convex_cont.points:
    print(
        "eta/r =", point.eta / radius,
        "kappa =", point.kappa,
        "gamma_ball =", point.gamma_ball,
    )
```

Use this section to explain the geometry of
```math
\kappa(\eta)=\left(\frac{\eta}{r}\right)^d
```
not as the main embedding result.

---

## 4. Nonconvex Poisson Ball-SGD

At step $t$, we sample a Poisson minibatch with rate $\gamma_t$, compute clipped per-example gradients, and release
```math
\widetilde S_t(D)=\sum_{i\in B_t}\bar g_t(z_i)+\xi_t,
\qquad
\xi_t\sim\mathcal N(0,\nu_t^2 I).
```
The stepwise Ball sensitivity is
```math
\Delta_t(r)
\le
\min\{L_z r,\ 2C\}.
```
The two main theorem-backed ReRo routes here are:
- optimized Ball-RDP $\Rightarrow$ Ball-ReRo;
- direct Poisson Ball-SGD $\Rightarrow$ Ball-ReRo.

### 4.1 Calibrate a noise multiplier from the accountant

```python
from quantbayes.ball_dp.api import calibrate_ball_sgd_noise_multiplier
from quantbayes.ball_dp.theorem import TheoremBounds, TheoremModelSpec, TrainConfig, certified_lz, make_model, fit_release
import jax.random as jr

orders = tuple(range(2, 13))

target_epsilon = 3.0
target_delta = 1e-6
hidden_dim = 128
num_steps = 400
batch_size = min(128, len(X_train))
clip_norm = 1.0
learning_rate = 3e-3

spec = TheoremModelSpec(
    d_in=feature_dim,
    hidden_dim=hidden_dim,
    task="multiclass",
    parameterization="dense",
    constraint="op",
    num_classes=num_classes,
)

bounds = TheoremBounds(B=B_public, A=4.0, Lambda=4.0)
lz = certified_lz(spec, bounds)

calibration = calibrate_ball_sgd_noise_multiplier(
    dataset_size=len(X_train),
    radius=radius,
    lz=lz,
    num_steps=num_steps,
    batch_size=batch_size,
    clip_norm=clip_norm,
    target_epsilon=target_epsilon,
    delta=target_delta,
    privacy="ball_dp",
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
    orders=orders,
)

noise_multiplier = float(calibration["noise_multiplier"])
print("L_z:", lz)
print("noise multiplier:", noise_multiplier)
```

### 4.2 Train a theorem-backed release

```python
from quantbayes.ball_dp.attacks.gradient_based import DPSGDTraceRecorder
from quantbayes.ball_dp import fit_ball_sgd
import optax

model = make_model(spec, key=jr.PRNGKey(seed), init_project=True, bounds=bounds)
recorder = DPSGDTraceRecorder(capture_every=10, keep_models=True, keep_batch_indices=True)

release_nonconvex = fit_release(
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
        delta=target_delta,
        num_steps=num_steps,
        batch_size=batch_size,
        batch_sampler="poisson",
        accountant_subsampling="match_sampler",
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        learning_rate=learning_rate,
        eval_every=50,
        normalize_noisy_sum_by="batch_size",
        seed=seed,
    ),
    orders=orders,
    trace_recorder=recorder,
    record_operator_norms=True,
    operator_norms_every=25,
)

print("release kind:", release_nonconvex.release_kind)
print("utility:", release_nonconvex.utility_metrics)
```

### 4.3 Primary nonconvex attack: unknown-inclusion finite prior

```python
from quantbayes.ball_dp import attack_nonconvex_ball_trace_finite_prior, make_trace_metadata_from_release
from quantbayes.ball_dp.attacks.gradient_based import subtract_known_batch_gradients
from quantbayes.ball_dp.attacks.ball_policy import BallTraceMapAttackConfig

trace = recorder.to_trace(
    state=None,
    loss_name=spec.loss_name,
    reduction="mean",
    metadata=make_trace_metadata_from_release(
        release_nonconvex,
        target_index=target_index,
    ),
)

residual_trace = subtract_known_batch_gradients(
    trace,
    train_ds,
    target_index=target_index,
)

true_record = residual_trace.metadata.get("true_record", None)
if true_record is None:
    from quantbayes.ball_dp.types import Record
    true_record = Record(
        features=np.asarray(X_train[target_index], dtype=np.float32),
        label=int(y_train[target_index]),
    )

attack_nonconvex_unknown = attack_nonconvex_ball_trace_finite_prior(
    residual_trace,
    X_candidates,
    y_candidates,
    prior_weights=None,
    cfg=BallTraceMapAttackConfig(
        mode="unknown_inclusion",
        seed=seed,
    ),
    target_index=target_index,
    known_label=target_label,
    true_record=true_record,
    eta_grid=(0.5,),
)

print("status:", attack_nonconvex_unknown.status)
print("metrics:", attack_nonconvex_unknown.metrics)
```

### 4.4 Theorem-backed nonconvex Ball-ReRo reports

```python
from quantbayes.ball_dp import ball_rero

report_nonconvex_rdp = ball_rero(
    release_nonconvex,
    prior=finite_prior,
    eta_grid=(0.5,),
    mode="rdp",
)
report_nonconvex_direct = ball_rero(
    release_nonconvex,
    prior=finite_prior,
    eta_grid=(0.5,),
    mode="ball_sgd_direct",
)

p_rdp = report_nonconvex_rdp.points[0]
p_dir = report_nonconvex_direct.points[0]

print("empirical exact-ID success =", attack_nonconvex_unknown.metrics.get("exact_identification_success"))
print("RDP -> Ball-ReRo bound (ball) =", p_rdp.gamma_ball)
print("RDP comparator (standard) =", p_rdp.gamma_standard)
print("alpha_opt_ball =", p_rdp.alpha_opt_ball)
print("direct Ball-SGD bound (ball) =", p_dir.gamma_ball)
print("direct comparator (standard) =", p_dir.gamma_standard)
```

**Expected behavior:** in long adaptive traces, the direct Poisson Ball-SGD bound is often looser than the RDP-based one.
That is normal, not a bug.

### 4.5 Runtime notes for the trace attacks

For finite-prior trace attacks, the expensive step is usually
`subtract_known_batch_gradients(...)`, not the final score-each-candidate call.
The rough attack scale is
```math
(\#\text{retained steps})\times(\text{avg retained batch size})
+
(\#\text{retained steps})\times(\#\text{candidates}).
```

Practical defaults for notebook demos:
- `batch_size = 128` or `256`
- `capture_every = 10` or `20`
- candidate support size `m = 8` to `16`

---

## 5. Theorem API and fixed-basis SVD fine-tuning

The compact theorem API lives in

```python
from quantbayes.ball_dp.theorem import ...
```

It is designed for the theorem-covered one-hidden-layer tanh models only.
A fixed-basis SVD hidden layer has the form
```math
W \approx U\,\operatorname{diag}(s)\,V^\top,
```
with frozen orthonormal $U,V$ and private optimization over $s$ (and typically the hidden bias and output head).

### What rank can and cannot change

Under the fixed-basis theorem currently implemented in the library, the certified Lipschitz-in-data constant
```math
L_z^{(\mathrm{fb})}
```
depends on the public norm bound and the parameter constraints, but **not on the rank**.
Therefore, if the theorem parameters are fixed, changing the SVD rank should leave the theorem-backed privacy certificate unchanged.

So the rank sweep is best interpreted as a study of:
- utility vs. rank,
- empirical attack behavior vs. rank,
- with a rank-invariant theorem-backed privacy certificate.

The theorem README contains the three recommended workflows:
1. private-only dense training;
2. public dense pretrain $\to$ private dense fine-tune;
3. public dense pretrain $\to$ fixed-basis SVD private fine-tune across ranks.

See **`quantbayes/ball_dp/theorem/README.md`** for the full code.

---

## 6. Recommended plotting/reporting in paper runs

For finite-prior exact identification, the primary quantities are:
- empirical exact-ID success;
- oblivious baseline $1/m$;
- theorem-backed Ball-ReRo upper bound;
- standard comparator bound.

For continuous-prior diagnostics, plot $\kappa(\eta)$ against $\eta/r$ and report direct Gaussian or direct Ball-SGD curves only as geometric intuition.

---

## 7. Other attacks / baselines (non-Ball, legacy, DeepMind-style)

The thesis notebook focuses on theorem-aligned Ball attacks. The package still includes the earlier non-Ball baselines and the SPEAR-style one-step attack.
This section shows how to run them.

### 7.0 Concrete model builder used by the legacy baselines

The model-based baseline and the SPEAR recipe below need a callable
`build_model(seed) -> (model, state)`.
If you already have your own architecture, use that instead. Otherwise the minimal example below restores the older notebook-style recipe and makes the later sections runnable as written.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
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
```

If you swap in a different architecture, update any architecture-specific settings below, especially SPEAR's `layer_path`.

### 7.1 Gradient trace optimization baseline (Hayes et al. 2023)

This is the older gradient-based baseline. It is **not** the Ball MAP attack.

```python
from quantbayes.ball_dp.api import make_trace_metadata_from_release
from quantbayes.ball_dp.attacks.gradient_based import (
    subtract_known_batch_gradients,
    TraceOptimizationAttackConfig,
    run_trace_optimization_attack,
)

trace = recorder.to_trace(
    state=None,
    loss_name=spec.loss_name,
    reduction="mean",
    metadata=make_trace_metadata_from_release(
        release_nonconvex,
        target_index=target_index,
    ),
)

residual_trace = subtract_known_batch_gradients(
    trace,
    train_ds,
    target_index=target_index,
)

true_record = residual_trace.metadata.get("true_record", None)
if true_record is None:
    from quantbayes.ball_dp.types import Record
    true_record = Record(
        features=np.asarray(X_train[target_index], dtype=np.float32),
        label=int(y_train[target_index]),
    )

attack_trace_baseline = run_trace_optimization_attack(
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

print("status:", attack_trace_baseline.status)
print("metrics:", attack_trace_baseline.metrics)
```

### 7.2 Model-based informed-attacker baseline (Balle et al. 2022)

This is the shadow-corpus / reconstructor baseline.

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

# Use the example build_model from Section 7.0, or replace it with your own
# build_model(seed) -> (model, state) function.

data = prepare_informed_attack_data(
    X_train,
    y_train,
    X_target=X_test,
    y_target=y_test,
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

attack_model_based = run_model_based_attack(
    release_model,
    data.d_minus,
    reconstructor=reconstructor,
    true_record=data.target,
    known_label=target_label,
)

print("status:", attack_model_based.status)
print("metrics:", attack_model_based.metrics)
```

### 7.3 SPEAR one-step baseline (Dimitrov et al. 2024)

This is the one-step batch attack implemented in `quantbayes.ball_dp.attacks.spear`.
It is a **single-step / single-batch** reconstruction baseline and should be understood separately from the Ball-local transcript MAP attacks.

```python
import optax
from quantbayes.ball_dp import fit_ball_sgd
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
    layer_path=("fc1",),   # matches the example MLP from Section 7.0
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

If you replace the example MLP with a different architecture, update `layer_path` accordingly.

---

## 8. Where to go next

- `quantbayes/ball_dp/convex/README.md` — convex releases, attacks, and convex ReRo reports.
- `quantbayes/ball_dp/nonconvex/README.md` — Poisson Ball-SGD, runtime notes, and theorem-backed nonconvex ReRo.
- `quantbayes/ball_dp/theorem/README.md` — theorem-only dense/SVD API and rank experiments.
- `quantbayes/ball_dp/decentralized/README.md` — observer-specific decentralized Ball-PN-RDP and exact MAP attacks.

