import math
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss, mean_squared_error
from quantbayes.pac_bayes_analysis.WMV import (
    PBLambdaCriterion,
    PBKLCriterion,
    TandemCriterion,
    PBBernsteinCriterion,
)
from quantbayes.stochax.trainer.train import (
    train,
    binary_loss,
    multiclass_loss,
    regression_loss,
)
from quantbayes.stochax import predict as _single_predict

__all__ = ["PacBayesEnsemble"]

CRITERIA = {
    "pblambda": PBLambdaCriterion,
    "pbkl": PBKLCriterion,
    "tandem": TandemCriterion,
    "pbbernstein": PBBernsteinCriterion,
}


class PacBayesEnsemble:
    def __init__(
        self,
        model_constructors,  # list of callables key->model
        task,  # "binary","multiclass","regression"
        loss_fn,
        optimizer,
        bound_type="pbkl",
        delta=0.05,
        L_max=1.0,
        seed=0,
    ):
        self.model_constructors = model_constructors
        self.task = task
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.bound_type = bound_type
        self.delta = delta
        self.L_max = L_max
        self.seed = seed

    def fit(
        self,
        X_train,
        y_train,
        X_hold,
        y_hold,
        X_test,
        y_test,
        m_values,
        batch_size=64,
        num_epochs=200,
        patience=20,
    ):
        rng = jr.PRNGKey(self.seed)
        self.results_ = []

        for m in m_values:
            # 1) Train m models
            models_states = []
            for i in range(m):
                rng, sk = jr.split(rng)
                model, state = eqx.nn.make_with_state(
                    lambda k: self.model_constructors[i % len(self.model_constructors)](
                        k
                    )
                )(sk)
                opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))
                bm, bs, _, _ = train(
                    model,
                    state,
                    opt_state,
                    self.optimizer,
                    self.loss_fn,
                    X_train,
                    y_train,
                    X_hold,
                    y_hold,
                    batch_size,
                    num_epochs,
                    patience,
                    sk,
                )
                models_states.append((bm, bs))

            # 2) Compute hold-out losses
            n = X_hold.shape[0]
            losses = np.zeros(m)
            y_h_np = np.array(y_hold).ravel()

            for i, (model, state) in enumerate(models_states):
                out = _single_predict(model, state, X_hold, jr.PRNGKey(1000 + i))
                if self.task == "binary":
                    preds = (jax.nn.sigmoid(out) >= 0.5).astype(int).ravel()
                    losses[i] = zero_one_loss(y_h_np, preds)
                elif self.task == "multiclass":
                    probs = jax.nn.softmax(out, axis=-1)
                    preds = np.array(jnp.argmax(probs, axis=-1))
                    losses[i] = zero_one_loss(y_h_np, preds)
                else:  # regression
                    preds = np.array(out).ravel()
                    mse = mean_squared_error(y_h_np, preds)
                    losses[i] = min(mse, self.L_max) / self.L_max

            # 3) PAC-Bayes optimize λ and ρ
            pi = np.full(m, 1 / m)
            rho = pi.copy()
            n_r = n
            lam = max(1 / math.sqrt(n_r), 0.5)
            Crit = CRITERIA[self.bound_type]()
            prevb = np.inf

            for _ in range(200):
                kl = float((rho * np.log(rho / pi)).sum())
                stat, bnd = Crit.compute(losses, rho, kl, n_r, self.delta, lam, n)
                if abs(prevb - bnd) < 1e-6:
                    break
                prevb = bnd
                lam = 2.0 / (
                    math.sqrt(
                        1
                        + 2
                        * n_r
                        * stat
                        / (kl + math.log(2 * math.sqrt(n_r) / self.delta))
                    )
                    + 1
                )
                shift = losses.min()
                w = np.exp(-lam * n_r * (losses - shift))
                rho = w / w.sum()

            # store last ensemble
            self._last_models_states = models_states
            self._last_rho = rho

            # 4) Evaluate majority-vote on test set
            y_t_np = np.array(y_test).ravel()
            if self.task == "binary":
                P = np.zeros((m, len(y_t_np)))
                for i, (model, state) in enumerate(models_states):
                    out = _single_predict(model, state, X_test, jr.PRNGKey(2000 + i))
                    pr = (jax.nn.sigmoid(out) >= 0.5).astype(int).ravel()
                    P[i] = np.where(pr == 0, -1, +1)
                agg = (rho[:, None] * P).sum(axis=0)
                test_loss = zero_one_loss(y_t_np, (agg >= 0).astype(int))

            elif self.task == "multiclass":
                C = int(jnp.max(y_h_np)) + 1
                votes = np.zeros((len(y_t_np), C))
                for i, (model, state) in enumerate(models_states):
                    out = _single_predict(model, state, X_test, jr.PRNGKey(2000 + i))
                    probs = jax.nn.softmax(out, axis=-1)
                    votes += rho[i] * np.array(probs)
                test_loss = zero_one_loss(y_t_np, votes.argmax(axis=1))

            else:  # regression
                preds = np.zeros(len(y_t_np))
                for i, (model, state) in enumerate(models_states):
                    out = _single_predict(model, state, X_test, jr.PRNGKey(2000 + i))
                    preds += rho[i] * np.array(out).ravel()
                test_loss = mean_squared_error(y_t_np, preds)

            self.results_.append(
                {
                    "m": m,
                    "hold_loss": float((rho * losses).sum()),
                    "bound": float(bnd),
                    "test_loss": float(test_loss),
                }
            )

    def predict(self, X, key):
        models_states, rho = self._last_models_states, self._last_rho
        n = X.shape[0]

        if self.task == "binary":
            P = np.zeros((len(models_states), n))
            keys = jr.split(key, len(models_states))
            for i, (model, state) in enumerate(models_states):
                out = _single_predict(model, state, X, keys[i])
                pr = (jax.nn.sigmoid(out) >= 0.5).astype(int).ravel()
                P[i] = np.where(pr == 0, -1, +1)
            agg = (rho[:, None] * P).sum(axis=0)
            return (agg >= 0).astype(int)

        elif self.task == "multiclass":
            keys = jr.split(key, len(models_states))
            C = _single_predict(
                models_states[0][0], models_states[0][1], X, keys[0]
            ).shape[-1]
            votes = np.zeros((n, C))
            for i, (model, state) in enumerate(models_states):
                out = _single_predict(model, state, X, keys[i])
                probs = jax.nn.softmax(out, axis=-1)
                votes += rho[i] * np.array(probs)
            return votes.argmax(axis=1)

        else:  # regression
            keys = jr.split(key, len(models_states))
            preds = np.zeros(n)
            for i, (model, state) in enumerate(models_states):
                out = _single_predict(model, state, X, keys[i])
                preds += rho[i] * np.array(out).ravel()
            return preds

    def summary(self):
        print("  m | hold_loss |  bound   | test_loss")
        print("--------------------------------------")
        for r in self.results_:
            print(
                f"{r['m']:3d} | {r['hold_loss']:.4f}   | {r['bound']:.4f} | {r['test_loss']:.4f}"
            )

    def plot(self):
        ms = [r["m"] for r in self.results_]
        hl = [r["hold_loss"] for r in self.results_]
        bd = [r["bound"] for r in self.results_]
        tl = [r["test_loss"] for r in self.results_]
        plt.figure(figsize=(8, 5))
        plt.plot(ms, hl, "-o", label="Hold‐out loss")
        plt.plot(ms, bd, "--s", label="PAC-Bayes bound")
        plt.plot(ms, tl, "-.^", label="Test loss")
        plt.xscale("log")
        plt.xlabel("m (ensemble size)")
        plt.ylabel("Loss / Bound")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


# ------------------------
# Define simple Equinox nets
# ------------------------
class EQXBinary(eqx.Module):
    l1: eqx.nn.Linear

    def __init__(self, in_features, key):
        self.l1 = eqx.nn.Linear(in_features, 1, key=key)

    def __call__(self, x, key, state):
        y = self.l1(x)
        return y, state


class EQXMulti(eqx.Module):
    l1: eqx.nn.Linear

    def __init__(self, in_features, n_classes, key):
        self.l1 = eqx.nn.Linear(in_features, n_classes, key=key)

    def __call__(self, x, key, state):
        y = self.l1(x)
        return y, state


class EQXReg(eqx.Module):
    l1: eqx.nn.Linear

    def __init__(self, in_features, key):
        self.l1 = eqx.nn.Linear(in_features, 1, key=key)

    def __call__(self, x, key, state):
        y = self.l1(x)
        return y, state


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import zero_one_loss

    # Binary example
    Xc, yc = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=0
    )
    X = jnp.array(Xc, dtype=jnp.float32)
    y = jnp.array(yc.reshape(-1, 1), dtype=jnp.float32)
    split = int(0.8 * len(yc))
    Xe, ye = X[:split], y[:split]
    Xh, yh = X[split:], y[split:]

    # two simple constructors
    def ctor1(key):
        return EQXBinary(X.shape[1], key)

    def ctor2(key):
        return EQXBinary(X.shape[1], jr.split(key)[1])

    ensemble = PacBayesEnsemble(
        [ctor1, ctor2],
        task="binary",
        loss_fn=binary_loss,
        optimizer=optax.adam(1e-3),
        bound_type="pbkl",
        delta=0.05,
        seed=42,
    )
    ensemble.fit(Xe, ye, Xh, yh, Xh, yh, m_values=[1, 2, 4, 8])
    ensemble.summary()
    ensemble.plot()

    # ---- test that predict() agrees with reported test_loss ----
    y_pred = ensemble.predict(Xh, jr.PRNGKey(999))
    y_h_np = np.array(yh).ravel()
    pred_err = zero_one_loss(y_h_np, y_pred)
    print(f"\n.predict() error on hold-out: {pred_err:.4f}")
