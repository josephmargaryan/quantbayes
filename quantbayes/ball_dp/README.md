# Ball-DP

Ball-DP is a privacy and reconstruction-risk toolkit for **local, label-preserving record changes**. The current codebase is organized around one shared finite-prior attack setup and several mechanism-specific observation models:

- convex Gaussian output perturbation;
- nonconvex Poisson Ball-SGD transcripts;
- decentralized linear-Gaussian observer views.

The central design rule is:

> Build the candidate support and hidden-record trial once, then pass the same `FinitePriorTrial` to the mechanism-specific attack and bound code.

This keeps empirical attacks, theorem-backed bounds, and paper/demo notebooks consistent.

---

## 1. Core objects and threat model

Records are $z=(x,y)$ with label-preserving metric

$$
d((x,y),(x',y'))=
\begin{cases}
\|x-x'\|_2, & y=y', \\
\infty, & y\ne y'.
\end{cases}
$$

A canonical finite-prior exact-identification experiment is:

1. choose a private training record $u=(x_u,y_u)$ as the center/anchor;
2. form $D^- = D \setminus \{u\}$;
3. build a public same-label candidate bank inside $B(u,r)$;
4. select a finite support $S=\{z_1,\dots,z_m\}\subset B(u,r)$;
5. draw or enumerate a hidden target $Z\in S$;
6. train/release on $D^-\cup\{Z\}$;
7. attack using the same support $S$;
8. report exact-ID success against the oblivious baseline $\kappa=\max_i \pi_i$, usually $1/m$.

The canonical setup lives in:

```python
from quantbayes.ball_dp.attacks.finite_prior_setup import (
    CandidateSource,
    find_feasible_replacement_banks,
    select_support_from_bank,
    make_replacement_trial,
)
```

The resulting `FinitePriorTrial` owns:

```text
D_minus_X, D_minus_y
X_full, y_full
target_index
target_support_position
target_source_id
support.X, support.y, support.weights, support.source_ids
support.center_x, support.center_y, support.radius
```

---

## 2. Quick canonical finite-prior setup

```python
import numpy as np
from sklearn.model_selection import train_test_split
from quantbayes.ball_dp.attacks.finite_prior_setup import (
    CandidateSource,
    find_feasible_replacement_banks,
    select_support_from_bank,
    make_replacement_trial,
    target_positions_for_support,
)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

X_train, X_public, y_train, y_public = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

radius = 1.0
m = 8
seed = 0

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

print("support hash:", trial.support.support_hash)
print("target source:", trial.target_source_id)
print("baseline kappa:", trial.support.oblivious_kappa)
```

Use `policy="all"` when the training cost is small and you want to enumerate every target $z_i\in S$. Use `policy="sample"` for expensive nonconvex/decentralized runs.

---

## 3. Mechanism-specific attack wrappers

### Convex Gaussian output perturbation

Use the same `trial`, but score it with the convex Gaussian-output likelihood:

```python
from quantbayes.ball_dp import fit_convex, ball_rero, make_finite_identification_prior
from quantbayes.ball_dp.api import attack_convex_finite_prior_trial

release = fit_convex(
    trial.X_full,
    trial.y_full,
    model_family="ridge_prototype",
    privacy="ball_dp",
    radius=radius,
    lam=1e-2,
    epsilon=4.0,
    delta=1e-6,
    embedding_bound=B_public,
    standard_radius=standard_radius,
    num_classes=num_classes,
    solver="lbfgs_fullbatch",
    max_iter=100,
    seed=seed,
)

attack = attack_convex_finite_prior_trial(
    release,
    trial,
    known_label=int(trial.support.center_y),
    eta_grid=(0.5,),
)

finite_prior = make_finite_identification_prior(trial.support.X, weights=trial.support.weights)
report = ball_rero(release, prior=finite_prior, eta_grid=(0.5,), mode="gaussian_direct")
```

For convex finite-prior diagnostics, use `diagnose_convex_ball_output_finite_prior(...)` with `trial.X_full`, `trial.y_full`, `trial.target_index`, `trial.support.X`, `trial.support.y`, and `center_features=trial.support.center_x`.

### Nonconvex Poisson Ball-SGD transcript

Train on `trial.X_full, trial.y_full`, build a residualized trace, then use:

```python
from quantbayes.ball_dp.api import attack_nonconvex_finite_prior_trial
from quantbayes.ball_dp.attacks.ball_policy import BallTraceMapAttackConfig

cfg = BallTraceMapAttackConfig(
    mode="unknown_inclusion",
    step_mode="all",
    seed=seed,
)

attack = attack_nonconvex_finite_prior_trial(
    residual_trace,
    trial,
    cfg=cfg,
    known_label=int(trial.support.center_y),
    eta_grid=(0.5,),
)
```

Known-inclusion attacks are useful oracle stress tests. Unknown-inclusion is the default threat model for hidden Poisson subsampling.

### Decentralized linear-Gaussian observer view

The decentralized path also reuses the same `trial`. The observation model changes:

```python
from quantbayes.ball_dp.decentralized.attacks import run_linear_gaussian_finite_prior_attack
from quantbayes.ball_dp.attacks.finite_prior_setup import enrich_attack_result_with_trial

attack = run_linear_gaussian_finite_prior_attack(
    observed_view=observed_view,
    candidate_features=trial.support.X,
    candidate_labels=trial.support.y,
    mean_fn=mean_fn,
    covariance=covariance,
    prior_weights=trial.support.weights,
    true_record=true_record,
    covariance_mode="kron_eye",
)
attack = enrich_attack_result_with_trial(attack, trial)
```

---

## 4. Theorem-backed Ball-ReRo routes

The main bound API is:

```python
from quantbayes.ball_dp import ball_rero
```

Common modes:

| Mode | Use case |
|---|---|
| `dp` | Generic $(\varepsilon,\delta)$ Ball-DP to Ball-ReRo. |
| `rdp` | Optimized Ball-RDP to Ball-ReRo. |
| `gaussian_direct` | Convex Gaussian output perturbation. |
| `ball_sgd_direct` | Direct revealed-inclusion Poisson Ball-SGD profile composition. |
| `ball_sgd_hidden` | Hidden-inclusion product f-DP/ReRo numerical object. |
| `ball_sgd_hayes` | Theorem-backed global/product revealed-inclusion dominating-pair object. |
| `ball_sgd_mix_comp` | Exploratory hidden-mixture per-step composition; do not treat as a raw-transcript certificate without the missing domination step. |

For finite-prior exact ID, use:

```python
finite_prior = make_finite_identification_prior(trial.support.X, weights=trial.support.weights)
```

For uniform finite support, $\kappa=1/m$.

---

## 5. Quantities to report

For every finite-prior attack table, report:

- `support_hash`;
- support size $m$;
- baseline $\kappa=\max_i\pi_i$;
- target source ID;
- predicted source ID;
- source-ID exact success;
- feature exact success, when meaningful;
- posterior top-1 probability or log-score gap, when available;
- theorem-backed Ball bound;
- standard comparator bound.

For utility/privacy figures, prefer:

1. utility vs. mechanism/noise;
2. attack success vs. baseline/bounds;
3. bound curves $\Gamma(\kappa)$ vs. $\kappa$;
4. support geometry diagnostics, especially finite-prior candidate separations.

---

## 6. Tutorials

The end-to-end synthetic tutorials are the recommended entry point:

```text
examples/ball_dp/convex_synthetic_ball_dp_end_to_end_demo.ipynb
examples/ball_dp/nonconvex_synthetic_transcript_ball_dp_end_to_end_demo.ipynb
examples/ball_dp/decentralized_synthetic_ball_dp_end_to_end_demo.ipynb
```

Each notebook uses synthetic data and the same canonical finite-prior setup. For real experiments, replace only the data-loading cell.

---

## 7. Focused guides

```text
quantbayes/ball_dp/convex/README.md
quantbayes/ball_dp/nonconvex/README.md
quantbayes/ball_dp/theorem/README.md
quantbayes/ball_dp/decentralized/README.md
examples/ball_dp/README.md
```