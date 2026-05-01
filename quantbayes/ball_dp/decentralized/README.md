# Decentralized Ball privacy

This guide covers the decentralized path: observer-specific privacy, linear-Gaussian observer views, exact finite-prior MAP attacks, and the decentralized noisy-prototype utility benchmark.

The important repository-level update is:

> Decentralized attacks should reuse the same canonical finite-prior setup as convex and nonconvex attacks. The decentralized-specific part is the observer view and Gaussian likelihood, not the construction of $D^-$, $S$, or the hidden target.

---

## 1. Observer-specific privacy notion

Let node $j$ be the attacked node and $A$ the observer set. The observer sees

$$
\operatorname{View}_A(M(D)).
$$

The decentralized privacy notion is observer-specific Ball-PN-RDP:

$$
D_\alpha\!\left(
\operatorname{View}_A(M(D))
\,\|\,
\operatorname{View}_A(M(D'))
\right)
\le
\varepsilon_{A\leftarrow j}(\alpha;r),
\qquad
D\sim_{r,j}D'.
$$

This quantifies how much a Ball-local change at node $j$ reaches observer set $A$.

---

## 2. Linear-Gaussian observer theorem

When the observer view can be written as

$$
Y_A(D)
=
c_A(D_{-j}) + (H_{A\leftarrow j}\otimes I_p)s_j(D)+\zeta_A,
\qquad
\zeta_A\sim\mathcal N(0,\Sigma_A),
$$

with block sensitivities $\Delta_{j,\ell}(r)$, the theorem gives

$$
D_\alpha(Y_A(D)\|Y_A(D'))
\le
\frac{\alpha}{2}\Delta_{A\leftarrow j}(r)^2,
$$

where

$$
\Delta_{A\leftarrow j}(r)^2
=
\sup_{\|\delta_\ell\|\le \Delta_{j,\ell}(r)}
\left\|\Sigma_A^{-1/2}(H_{A\leftarrow j}\otimes I_p)\delta\right\|_2^2.
$$

Topology, gossip, and the observer set enter through $H_{A\leftarrow j}$ and $\Sigma_A$.

---

## 3. Reuse the canonical finite-prior setup

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

The decentralized attack uses:

```python
trial.support.X
trial.support.y
trial.support.weights
trial.target_source_id
```

The exact same `trial` object can also be used in convex and nonconvex demos.

---

## 4. Gossip transfer and observer covariance

```python
from quantbayes.ball_dp.decentralized.gossip import (
    make_graph_adjacency,
    metropolis_mixing_matrix,
    constant_mixing_matrices,
    selector_matrix,
    gossip_transfer_matrix,
    gossip_observer_noise_covariance,
)

num_nodes = 8
num_rounds = 6
attacked_node = 0
observer_nodes = [num_nodes - 1]

A = make_graph_adjacency("path", num_nodes=num_nodes)
W = metropolis_mixing_matrix(A, laziness=0.25)
Ws = constant_mixing_matrices(W, num_rounds=num_rounds)
S_A = selector_matrix(observer_nodes, num_nodes=num_nodes)

H = gossip_transfer_matrix(
    Ws,
    observer_selector=S_A,
    attacked_node=attacked_node,
)

covariance_time = gossip_observer_noise_covariance(
    Ws,
    observer_selector=S_A,
    state_noise_stds=noise_std,
    observation_noise_std=0.0,
    jitter=1e-8,
)
```

---

## 5. Observer-specific accounting

Use the linear-Gaussian accountant for the observer view:

```python
from quantbayes.ball_dp.decentralized.accounting import account_linear_gaussian_observer

acct = account_linear_gaussian_observer(
    transfer_matrix=H,
    covariance=covariance_time,
    block_sensitivities=block_sensitivities,
    parameter_dim=feature_dim,
    orders=orders,
    radius=radius,
    dp_delta=delta,
    attacked_node=attacked_node,
    observer=tuple(observer_nodes),
    covariance_mode="kron_eye",
    method="auto",
)

print("sensitivity_sq:", acct.sensitivity_sq)
print("exact sensitivity calculation:", acct.exact)
print("method:", acct.method)
```

The result is observer-specific. Changing the observer set or graph can change the privacy curve.

---

## 6. Exact finite-prior MAP attack on a Gaussian observer view

Build a deterministic candidate mean map:

```python
from quantbayes.ball_dp.decentralized.attacks import (
    make_linear_gaussian_mean_fn,
    run_linear_gaussian_finite_prior_attack,
)
from quantbayes.ball_dp.attacks.finite_prior_setup import (
    enrich_attack_result_with_trial,
    trial_true_record,
)

def sensitive_blocks_fn(x, y):
    # Example: every local round has the same clipped candidate contribution.
    # Replace this with your theorem-side s_j(z) for the decentralized mechanism.
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return np.repeat(x[None, :], repeats=H.shape[1], axis=0)

mean_fn = make_linear_gaussian_mean_fn(
    transfer_matrix=H,
    sensitive_blocks_fn=sensitive_blocks_fn,
    base_offset=base_offset,
)

attack = run_linear_gaussian_finite_prior_attack(
    observed_view=observed_view,
    candidate_features=trial.support.X,
    candidate_labels=trial.support.y,
    mean_fn=mean_fn,
    covariance=covariance_time,
    prior_weights=trial.support.weights,
    true_record=trial_true_record(trial),
    eta_grid=(0.5,),
    covariance_mode="kron_eye",
)

attack = enrich_attack_result_with_trial(attack, trial)
```

Report:

```python
attack.metrics["source_exact_identification_success"]
attack.diagnostics["predicted_source_id"]
attack.diagnostics["target_source_id"]
```

---

## 7. Public-transcript bridge accounting

For public-transcript decentralized algorithms where the full public transcript is post-processing of an attacked node's local noisy Poisson releases, use:

```python
from quantbayes.ball_dp.decentralized.accounting import (
    LocalNodeSGDSchedule,
    account_public_transcript_node_local_rdp,
)

schedule = LocalNodeSGDSchedule.from_noise_multipliers(
    dataset_size=local_dataset_size,
    batch_sizes=batch_sizes,
    clip_norms=clip_norms,
    noise_multipliers=noise_multiplier,
    radius=radius,
    lz=lz,
    orders=orders,
    batch_sampler="poisson",
    accountant_subsampling="match_sampler",
    dp_delta=delta,
)

public_transcript_acct = account_public_transcript_node_local_rdp(
    schedule,
    attacked_node=attacked_node,
)
```

This bridge applies to public transcripts and final states that are post-processing of the attacked node's local private releases.

---

## 8. Utility benchmark: noisy prototype gossip

The prototype benchmark is lightweight and NumPy-only:

```python
from quantbayes.ball_dp.decentralized.prototypes import run_noisy_prototype_gossip

utility = run_noisy_prototype_gossip(
    X_train,
    y_train,
    X_test,
    y_test,
    W=W,
    num_classes=num_classes,
    rounds=num_rounds,
    clip_norm=clip_norm,
    noise_std=noise_std,
    seed=seed,
)

print("mean accuracy:", utility.accuracy_mean)
print("min node accuracy:", utility.accuracy_min)
print("consensus disagreement:", utility.consensus_disagreement)
```

---

## 9. Quantities to report

For decentralized experiments, report:

- graph/topology;
- attacked node;
- observer set;
- $\Delta_{A\leftarrow j}(r)^2$;
- whether the sensitivity calculation is exact or an upper bound;
- observer-specific RDP/DP curve;
- finite-prior baseline $1/m$;
- exact-ID attack success;
- utility and consensus disagreement.

Useful figures:

1. heatmap of transferred sensitivity by attacked node and observer;
2. exact-ID success vs. observer distance;
3. bound curves $\Gamma_A(\kappa)$ by observer set;
4. utility vs. privacy/noise;
5. topology comparison plots.

---

## 10. Tutorial

See:

```text
examples/ball_dp/decentralized_synthetic_ball_dp_end_to_end_demo.ipynb
```

It uses synthetic data and the shared finite-prior setup. For real decentralized experiments, replace the data-loading and `sensitive_blocks_fn` cells.
