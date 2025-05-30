import math
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from numpyro.optim import Adam
from sklearn.metrics import zero_one_loss, mean_squared_error
from quantbayes.pac_bayes_analysis.WMV import (
    PBLambdaCriterion,
    PBKLCriterion,
    TandemCriterion,
    PBBernsteinCriterion,
)


# ----------------------------
# 1) SVI helper with AutoNormal guides
# ----------------------------
def fit_svi(model, optimizer, X, y, rng_key, num_steps=1000):
    guide = autoguide.AutoNormal(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    state = svi.init(rng_key, X, y)
    for i in range(1, num_steps + 1):
        state, loss = svi.update(state, X, y)
        if i % 100 == 0:
            print(f"SVI step {i}, loss = {loss:.4f}")
    params = svi.get_params(state)
    return guide, params


# ----------------------------
# 2) Posterior predictive mean
# ----------------------------
def predictive_mean(model, guide, params, X, rng_key, num_samples=100):
    pred = Predictive(model, guide=guide, params=params, num_samples=num_samples)
    samples = pred(rng_key, X, None)
    out = samples["obs"]  # shape (num_samples, N, [...])
    return jnp.mean(out, axis=0)  # (N, [...])


# ----------------------------
# 3) PAC-Bayes weighting
# ----------------------------
CRITERIA = {
    "pblambda": PBLambdaCriterion,
    "pbkl": PBKLCriterion,
    "tandem": TandemCriterion,
    "pbbernstein": PBBernsteinCriterion,
}


def compute_pacbayes_weights(
    preds_hold, y_hold, task, bound_type="pbkl", delta=0.05, L_max=1.0
):
    N = len(y_hold)
    M = len(preds_hold)
    losses = np.zeros(M)
    for i, ph in enumerate(preds_hold):
        if task in ("binary", "multiclass"):
            losses[i] = zero_one_loss(y_hold, np.array(ph))
        else:
            mse = mean_squared_error(y_hold, np.array(ph))
            losses[i] = min(mse, L_max) / L_max

    pi, rho = np.full(M, 1 / M), np.full(M, 1 / M)
    n_r, lam = N, max(1 / math.sqrt(N), 0.5)
    Crit = CRITERIA[bound_type]()
    prev_b = np.inf

    for _ in range(200):
        kl = float((rho * np.log(rho / pi)).sum())
        stat, bnd = Crit.compute(losses, rho, kl, n_r, delta, lam, N)
        if abs(prev_b - bnd) < 1e-6:
            break
        prev_b = bnd
        lam = 2.0 / (
            math.sqrt(1 + 2 * n_r * stat / (kl + math.log(2 * math.sqrt(n_r) / delta)))
            + 1
        )
        shift = losses.min()
        w = np.exp(-lam * n_r * (losses - shift))
        rho = w / w.sum()

    return rho, bnd


# ----------------------------
# 4) Ensemble routine
# ----------------------------
def svi_ensemble_pacbayes(
    model_fns,  # list of model callables
    X_train,
    y_train,
    X_hold,
    y_hold,
    X_test,
    y_test,
    task,  # "binary","multiclass","regression"
    rng_key,
    svi_steps=1000,
    num_samples=200,
    bound_type="pbkl",
    delta=0.05,
    L_max=1.0,
):
    optimizer = Adam(1e-3)
    M = len(model_fns)

    # 1) Fit with AutoNormal
    guides, params_list = [], []
    keys = jr.split(rng_key, M)
    for model, key in zip(model_fns, keys):
        guide, params = fit_svi(
            model, optimizer, X_train, y_train, key, num_steps=svi_steps
        )
        guides.append(guide)
        params_list.append(params)

    # 2) Hold-out predictions
    preds_hold = []
    for model, guide, params, key in zip(
        model_fns, guides, params_list, jr.split(rng_key, M)
    ):
        pm = predictive_mean(model, guide, params, X_hold, key, num_samples)
        if task == "binary":
            preds_hold.append((pm >= 0.5).astype(int))
        elif task == "multiclass":
            preds_hold.append(jnp.argmax(pm, axis=-1))
        else:
            preds_hold.append(pm)

    # 3) PAC-Bayes weights
    rho, bound = compute_pacbayes_weights(
        [np.array(p) for p in preds_hold],
        np.array(y_hold).ravel(),
        task,
        bound_type,
        delta,
        L_max,
    )

    # 4) Test predictions
    if task in ("binary", "multiclass"):
        votes = []
        for model, guide, params, key in zip(
            model_fns, guides, params_list, jr.split(rng_key, M)
        ):
            pm = predictive_mean(model, guide, params, X_test, key, num_samples)
            if task == "binary":
                votes.append((pm >= 0.5).astype(int))
            else:
                votes.append(jnp.argmax(pm, axis=-1))
        votes = np.stack([np.array(v) for v in votes], axis=0)  # (M, N)
        if task == "binary":
            votes = np.where(votes == 0, -1, +1)
            agg = (rho[:, None] * votes).sum(axis=0)
            y_pred = (agg >= 0).astype(int)
        else:
            C = int(np.max(y_hold)) + 1
            onehot = np.eye(C)[votes]  # (M, N, C)
            wavg = (rho[:, None, None] * onehot).sum(axis=0)
            y_pred = wavg.argmax(axis=1)
        test_loss = zero_one_loss(np.array(y_test).ravel(), y_pred)

    else:
        preds_test = []
        for model, guide, params, key in zip(
            model_fns, guides, params_list, jr.split(rng_key, M)
        ):
            pm = predictive_mean(model, guide, params, X_test, key, num_samples)
            preds_test.append(pm)
        preds_test = np.stack(preds_test, axis=0)
        y_pred = (rho[:, None] * preds_test).sum(axis=0)
        test_loss = mean_squared_error(np.array(y_test).ravel(), y_pred)

    return {"rho": rho, "bound": bound, "test_loss": test_loss, "y_pred": y_pred}


# ----------------------------
# 5) Simple logistic models
# ----------------------------
def model1(X, y=None):
    D = X.shape[1]
    w = numpyro.sample("w", dist.Normal(jnp.zeros(D), 1.0).to_event(1))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    logits = jnp.dot(X, w) + b
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


def model2(X, y=None):
    D = X.shape[1]
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    w = numpyro.sample("w", dist.Normal(jnp.zeros(D), sigma).to_event(1))
    b = numpyro.sample("b", dist.Normal(0.0, sigma))
    logits = jnp.dot(X, w) + b
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


# ----------------------------
# 6) Example usage
# ----------------------------
if __name__ == "__main__":
    import jax.random as jr
    from sklearn.datasets import make_classification

    key = jr.PRNGKey(0)

    # Generate binary data
    Xc, yc = make_classification(500, 10, random_state=1)
    X_train, X_hold, X_test = Xc[:300], Xc[300:400], Xc[400:]
    y_train, y_hold, y_test = yc[:300], yc[300:400], yc[400:]

    info = svi_ensemble_pacbayes(
        model_fns=[model1, model2],
        X_train=jnp.array(X_train),
        y_train=jnp.array(y_train),
        X_hold=jnp.array(X_hold),
        y_hold=jnp.array(y_hold),
        X_test=jnp.array(X_test),
        y_test=jnp.array(y_test),
        task="binary",
        rng_key=key,
        svi_steps=500,
        num_samples=100,
        bound_type="pbkl",
        delta=0.05,
    )

    print("ρ:", info["rho"])
    print("Bound:", info["bound"])
    print("Test 0-1 loss:", info["test_loss"])
