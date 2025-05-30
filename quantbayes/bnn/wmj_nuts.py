import math
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss
from quantbayes.pac_bayes_analysis.WMV import (
    PBLambdaCriterion,
    PBKLCriterion,
    TandemCriterion,
    PBBernsteinCriterion,
)


# ----------------------------
# 1) Fit a NUTS chain and return posterior samples
# ----------------------------
def fit_nuts(model, X, y, rng_key, num_warmup=200, num_samples=100):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False
    )
    mcmc.run(rng_key, X, y)
    return mcmc.get_samples()


# ----------------------------
# 2) Posterior predictive mean
# ----------------------------
def predictive_mean_nuts(model, posterior_samples, X, rng_key, num_samples=100):
    pred = Predictive(
        model, posterior_samples=posterior_samples, num_samples=num_samples
    )
    out = pred(rng_key, X, None)["obs"]  # shape (num_samples, N)
    return jnp.mean(out, axis=0)  # shape (N,)


# ----------------------------
# 3) PAC-Bayes weighting (0–1 for classification)
# ----------------------------
CRITERIA = {
    "pblambda": PBLambdaCriterion,
    "pbkl": PBKLCriterion,
    "tandem": TandemCriterion,
    "pbbernstein": PBBernsteinCriterion,
}


def compute_pacbayes_weights(preds_hold, y_hold, bound_type="pbkl", delta=0.05):
    N = len(y_hold)
    M = len(preds_hold)
    losses = np.array([zero_one_loss(y_hold, np.array(ph)) for ph in preds_hold])
    pi, rho = np.full(M, 1 / M), np.full(M, 1 / M)
    lam = max(1 / math.sqrt(N), 0.5)
    Crit = CRITERIA[bound_type]()
    prev_b = np.inf
    for _ in range(200):
        kl = float((rho * np.log(rho / pi)).sum())
        stat, bnd = Crit.compute(losses, rho, kl, N, delta, lam, N)
        if abs(prev_b - bnd) < 1e-6:
            break
        prev_b = bnd
        lam = 2.0 / (
            math.sqrt(1 + 2 * N * stat / (kl + math.log(2 * math.sqrt(N) / delta))) + 1
        )
        shift = losses.min()
        w = np.exp(-lam * N * (losses - shift))
        rho = w / w.sum()
    return rho, bnd


# ----------------------------
# 4) Full NUTS-ensemble pipeline
# ----------------------------
def nuts_ensemble_pacbayes(
    model,  # single numpyro model callable
    X_train,
    y_train,
    X_hold,
    y_hold,
    X_test,
    y_test,
    m,  # ensemble size
    rng_key,
    bound_type="pbkl",
    delta=0.05,
):
    # 1) Fit m NUTS chains
    keys = jr.split(rng_key, m)
    samples_list = [fit_nuts(model, X_train, y_train, k) for k in keys]

    # 2) Predict on hold-out
    preds_hold = [
        (
            predictive_mean_nuts(model, samples, X_hold, jr.fold_in(rng_key, i)) >= 0.5
        ).astype(int)
        for i, samples in enumerate(samples_list)
    ]

    # 3) PAC-Bayes weights
    rho, bound = compute_pacbayes_weights(
        preds_hold, np.array(y_hold), bound_type, delta
    )

    # 4) Predict on test set via weighted majority vote
    preds_test = [
        (
            predictive_mean_nuts(model, samples, X_test, jr.fold_in(rng_key, 100 + i))
            >= 0.5
        ).astype(int)
        for i, samples in enumerate(samples_list)
    ]
    P = np.stack(preds_test, axis=0)  # (m, N_test)
    P = np.where(P == 0, -1, +1)
    agg = (rho[:, None] * P).sum(axis=0)
    y_pred = (agg >= 0).astype(int)
    test_loss = zero_one_loss(np.array(y_test), y_pred)

    return rho, bound, test_loss, y_pred


# ----------------------------
# 5) A simple Bayesian logistic regression
# ----------------------------
def logistic_model(X, y=None):
    D = X.shape[1]
    w = numpyro.sample("w", dist.Normal(jnp.zeros(D), 1.0).to_event(1))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    logits = jnp.dot(X, w) + b
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


# ----------------------------
# 6) Example usage & visualization
# ----------------------------
if __name__ == "__main__":
    import jax.random as jr
    from sklearn.datasets import make_classification

    key = jr.PRNGKey(0)
    # Generate binary data
    Xc, yc = make_classification(1000, 20, random_state=1)
    split = int(0.6 * len(yc))
    X_train, X_hold, X_test = Xc[:split], Xc[split : split + 200], Xc[split + 200 :]
    y_train, y_hold, y_test = yc[:split], yc[split : split + 200], yc[split + 200 :]

    results = []
    for m in [1, 2, 4, 8]:
        rho, bound, test_loss, _ = nuts_ensemble_pacbayes(
            logistic_model,
            jnp.array(X_train),
            jnp.array(y_train),
            jnp.array(X_hold),
            jnp.array(y_hold),
            jnp.array(X_test),
            jnp.array(y_test),
            m,
            jr.fold_in(key, m),
            bound_type="pbkl",
            delta=0.05,
        )
        results.append({"m": m, "bound": bound, "test_loss": test_loss})

    # Summary
    print(" m |   bound   | test_loss")
    print("--------------------------")
    for r in results:
        print(f"{r['m']:2d} | {r['bound']:.4f} | {r['test_loss']:.4f}")

    # Plot
    ms = [r["m"] for r in results]
    bd = [r["bound"] for r in results]
    tl = [r["test_loss"] for r in results]
    plt.figure(figsize=(6, 4))
    plt.plot(ms, bd, "-o", label="PAC-Bayes bound")
    plt.plot(ms, tl, "-s", label="Test 0-1 loss")
    plt.xscale("log", base=2)
    plt.xlabel("Ensemble size m")
    plt.ylabel("Risk / Bound")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Demo predict
    _, _, _, y_pred = nuts_ensemble_pacbayes(
        logistic_model,
        jnp.array(X_train),
        jnp.array(y_train),
        jnp.array(X_hold),
        jnp.array(y_hold),
        jnp.array(X_test),
        jnp.array(y_test),
        m=8,
        rng_key=jr.fold_in(key, 42),
        bound_type="pbkl",
        delta=0.05,
    )
    print("Sample preds:", y_pred[:10])
