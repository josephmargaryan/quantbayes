# Decentralized Ball-DP helpers

This subpackage is intentionally **adapter-first**.
The uploaded repo does not contain an existing decentralized trainer, so the cleanest,
least-invasive implementation is:

1. keep your existing decentralized training code,
2. log only the quantities required by the theorems,
3. pass those logs into the helpers here.

That gives you theorem-aligned accounting and attacks without forcing a rewrite of your
legacy decentralized code.

## 1) Public-transcript decentralized Ball-SGD-RDP

Under the public-transcript theorem, the privacy cost against a target at node `j`
depends only on node `j`'s local Poisson-subsampled Gaussian schedule.
Consensus / gossip / message passing after those releases is post-processing.

```python
from quantbayes.ball_dp.decentralized import (
    LocalNodeSGDSchedule,
    account_public_transcript_node_local_rdp,
)

schedule = LocalNodeSGDSchedule.from_noise_multipliers(
    dataset_size=n_j,
    batch_sizes=[64] * T,
    clip_norms=[1.0] * T,
    noise_multipliers=[1.2] * T,
    radius=0.5,
    lz=3.7,
    orders=(2, 3, 4, 5, 8, 16, 32, 64, 128),
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
    dp_delta=1e-6,
)

result = account_public_transcript_node_local_rdp(
    schedule,
    attacked_node=j,
)

ball_curve = result.ledger.ball.rdp_curve
ball_dp = result.ledger.ball.dp_certificates[0]
```

If your decentralized code already computes a sharper per-step local sensitivity schedule
than `min(L_z r, 2 C_t)`, pass it through `step_delta_ball=`.

## 2) Observer-specific Ball-PN-RDP for a linear Gaussian view

When the observer view has the theorem form

\[
Y_A = c_A(D_{-j}) + (H_{A \leftarrow j} \otimes I_p) s_j(D) + \zeta_A,
\]

you can account it directly.

### Common efficient case: `Sigma_A = Sigma_time ⊗ I_p`

Pass `covariance` with shape `(d_A, d_A)`.

```python
from quantbayes.ball_dp.decentralized import account_linear_gaussian_observer

obs_result = account_linear_gaussian_observer(
    transfer_matrix=H,
    covariance=Sigma_time,
    block_sensitivities=delta_blocks,
    parameter_dim=p,
    orders=(2, 3, 4, 5, 8, 16, 32, 64, 128),
    radius=0.5,
    dp_delta=1e-6,
    attacked_node=j,
    observer=tuple(A),
)

curve = obs_result.rdp_curve
cert = obs_result.dp_certificate
```

Correctness notes:
- The code returns an **exact** sensitivity only in theorem-safe special cases.
- Otherwise it returns the theorem-backed **operator-norm upper bound**.
- The result object tells you which case was used through `exact` and `method`.

## 3) Gossip specialization

For the linear recursion

\[
x_{t+1} = W_t x_t + u_t + \xi_t,
\qquad
y_t = (S_A \otimes I_p) x_t,
\]

build the theorem transfer matrix as:

```python
from quantbayes.ball_dp.decentralized import selector_matrix, gossip_transfer_matrix

S_A = selector_matrix(A, num_nodes=m)
H = gossip_transfer_matrix(
    mixing_matrices=W_schedule,
    observer_selector=S_A,
    attacked_node=j,
)
```

The returned `H` stacks the observer views `(y_1, ..., y_T)` and the attacked-node
sensitive blocks `(u_0, ..., u_{T-1})`.

## 4) Observer-specific Ball-ReRo from the Ball-PN-RDP curve

```python
from quantbayes.ball_dp.decentralized import compute_ball_pn_rero_report
from quantbayes.ball_dp.evaluation.rero import FiniteExactIdentificationPrior

prior = FiniteExactIdentificationPrior(candidate_features, weights=None)
report = compute_ball_pn_rero_report(curve, prior, eta_grid=(0.25, 0.5, 1.0))
```

## 5) Exact MAP attacks on decentralized Gaussian views

### Finite prior (exact theorem-aligned Bayes classification)

```python
from quantbayes.ball_dp.decentralized import run_linear_gaussian_finite_prior_attack

attack = run_linear_gaussian_finite_prior_attack(
    observed_view=y_A,
    candidate_features=X_candidates,
    candidate_labels=y_candidates,
    mean_fn=mean_fn,
    covariance=Sigma_time,          # or full covariance on the flattened view
    true_record=true_record,
)
```

### Continuous Ball-constrained MAP

```python
from quantbayes.ball_dp.decentralized import (
    LinearGaussianMapAttackConfig,
    run_linear_gaussian_ball_map_attack,
)
from quantbayes.ball_dp.attacks.ball_priors import UniformBallAttackPrior

attack = run_linear_gaussian_ball_map_attack(
    observed_view=y_A,
    prior=UniformBallAttackPrior(center=u, radius=r),
    mean_fn=mean_fn,
    covariance=Sigma_time,
    cfg=LinearGaussianMapAttackConfig(
        optimizer="adam",
        num_steps=400,
        learning_rate=1e-2,
        num_restarts=5,
        seed=0,
    ),
    known_label=target_label,
    true_record=true_record,
)
```

## 6) Building `mean_fn` from the theorem decomposition

If you already have a function that computes the attacked node's stacked sensitive blocks
`s_j(z)` for a candidate record, use

```python
from quantbayes.ball_dp.decentralized import make_linear_gaussian_mean_fn

mean_fn = make_linear_gaussian_mean_fn(
    transfer_matrix=H,
    sensitive_blocks_fn=sensitive_blocks_fn,
    base_offset=c_A,
)
```

where `sensitive_blocks_fn(x, y)` returns the stacked blocks with shape `(q, p)`
or flat shape `(q * p,)`.
