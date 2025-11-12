import math
import copy
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss, mean_squared_error

from quantbayes import bnn
from quantbayes.bnn.layers import Module

__all__ = ["PacBayesEnsembleNumPyro"]


from quantbayes.pacbayes.WMV import (
    PBLambdaCriterion,
    PBKLCriterion,
    TandemCriterion,
    PBBernsteinCriterion,
    SplitKLCriterion,
    UnexpectedBernsteinCriterion,
)

CRITERIA = {
    "pblambda": PBLambdaCriterion,
    "pbkl": PBKLCriterion,
    "tandem": TandemCriterion,
    "pbbernstein": PBBernsteinCriterion,
    "split-kl": SplitKLCriterion,
    "unexpected-bernstein": UnexpectedBernsteinCriterion,
}


class PacBayesEnsembleNumPyro:
    """
    PAC-Bayes ensemble wrapper for NumPyro-based Module classes.
    """

    def __init__(
        self,
        module_constructors,
        task,
        bound_type="pbkl",
        delta=0.05,
        L_max=1.0,
        seed=0,
    ):
        if bound_type not in CRITERIA:
            raise ValueError(
                f"Unknown bound_type '{bound_type}'. Must be one of {list(CRITERIA)}."
            )
        if task not in ("binary", "multiclass", "regression"):
            raise ValueError("`task` must be 'binary', 'multiclass', or 'regression'.")

        self.module_constructors = module_constructors
        self.task = task
        self.bound_type = bound_type
        self.delta = delta
        self.L_max = L_max
        self.seed = seed

        self.results_ = []
        self.best_m_ = None
        self.best_bound_ = np.inf
        self.best_rho_ = None
        self.best_models_states_ = None
        self.best_elbo_losses_ = None
        self.is_fitted = False

    def _compute_hold_loss(self, module, X_hold, y_hold_np, rng_key, num_samples=100):
        """
        For a trained Module `module`, compute hold-out loss on (X_hold, y_hold_np).
        """
        preds = module.predict(
            X_hold, rng_key, posterior="logits", num_samples=num_samples
        )
        if self.task == "binary":
            probs = jax.nn.sigmoid(preds.reshape((num_samples, -1))).mean(axis=0)
            preds_label = (np.array(probs) >= 0.5).astype(int)
            return float(zero_one_loss(y_hold_np, preds_label))

        elif self.task == "multiclass":
            probs = jax.nn.softmax(preds, axis=-1).mean(axis=0)
            preds_label = np.array(jnp.argmax(probs, axis=-1))
            return float(zero_one_loss(y_hold_np, preds_label))

        else:  # regression
            preds_flat = preds.reshape((num_samples, -1)).mean(axis=0)
            mse = mean_squared_error(y_hold_np, np.array(preds_flat))
            return float(min(mse, self.L_max) / self.L_max)

    def _majority_vote_test_loss(
        self, models_states, rho, X_test, y_test_np, rng_key, num_samples=100
    ):
        """
        Compute weighted majority-vote (classification) or weighted average (regression)
        on (X_test, y_test_np) using list of (module, _) in models_states.
        """
        n_test = X_test.shape[0]
        m = len(models_states)
        rng_keys = jr.split(rng_key, num=m + 1)[1:]

        if self.task == "binary":
            P_votes = np.zeros((m, n_test))
            for i, (module, _) in enumerate(models_states):
                preds = module.predict(
                    X_test, rng_keys[i], posterior="logits", num_samples=num_samples
                )
                probs = jax.nn.sigmoid(preds.reshape((num_samples, -1))).mean(axis=0)
                P_votes[i, :] = np.where(np.array(probs) >= 0.5, +1, -1)
            agg = (rho[:, None] * P_votes).sum(axis=0)
            preds_label = (agg >= 0).astype(int)
            return float(zero_one_loss(y_test_np, preds_label))

        elif self.task == "multiclass":
            C = int(y_test_np.max()) + 1
            votes = np.zeros((n_test, C))
            for i, (module, _) in enumerate(models_states):
                preds = module.predict(
                    X_test, rng_keys[i], posterior="logits", num_samples=num_samples
                )
                probs = jax.nn.softmax(preds, axis=-1).mean(axis=0)
                votes += rho[i] * np.array(probs)
            preds_label = votes.argmax(axis=1)
            return float(zero_one_loss(y_test_np, preds_label))

        else:  # regression
            preds_array = np.zeros((m, n_test))
            for i, (module, _) in enumerate(models_states):
                preds = module.predict(
                    X_test, rng_keys[i], posterior="logits", num_samples=num_samples
                )
                preds_array[i, :] = preds.reshape((num_samples, -1)).mean(axis=0)
            agg = (rho[:, None] * preds_array).sum(axis=0)
            return float(mean_squared_error(y_test_np, agg))

    def fit(
        self,
        X_train,
        y_train,
        X_hold,
        y_hold,
        X_test,
        y_test,
        m_values,
        svi_steps=500,
        num_samples=100,
    ):
        """
        Train the PAC-Bayes ensemble of NumPyro Modules.

        Parameters
        ----------
        X_train : jnp.ndarray, shape (N_train, d)
        y_train : jnp.ndarray, shape (N_train,) or (N_train,1)
        X_hold, y_hold : jnp.ndarray    # hold-out for computing empirical losses
        X_test, y_test : jnp.ndarray    # test set for final majority vote
        m_values : list of int          # ensemble sizes to try
        svi_steps : int = 500           # # SVI steps per module
        num_samples: int = 100          # # posterior samples when computing loss/prediction
        """
        # 1) Convert everything to JAX arrays (and flatten the labels for NumPy metrics)
        X_train = jnp.asarray(X_train, dtype=jnp.float32)
        y_train = jnp.asarray(y_train)

        X_hold = jnp.asarray(X_hold, dtype=jnp.float32)
        y_hold = jnp.asarray(y_hold)
        y_hold_np = np.array(y_hold).ravel()

        X_test = jnp.asarray(X_test, dtype=jnp.float32)
        y_test = jnp.asarray(y_test)
        y_test_np = np.array(y_test).ravel()

        rng = jr.PRNGKey(self.seed)
        self.results_ = []
        self.best_bound_ = np.inf
        self.best_m_ = None
        self.best_rho_ = None
        self.best_models_states_ = None
        self.best_elbo_losses_ = None

        # 2) Loop over each ensemble size m
        for m in m_values:
            models_states = []
            elbo_losses_all = []

            # ---- Step 1: Train m Bayesian Modules ----
            for i in range(m):
                rng, sk = jr.split(rng)
                constructor = self.module_constructors[
                    i % len(self.module_constructors)
                ]
                module = constructor(sk)
                module.compile()  # sets up SVI/NUTS/SteinVI
                module.fit(X_train, y_train, sk, num_steps=svi_steps)

                # We don’t need a “state” object in this wrapper—just store the trained module
                models_states.append((module, None))

                # If it was SVI, record its ELBO-loss sequence (otherwise append None)
                if module.method == "svi":
                    elbo_losses_all.append(module.get_losses)
                else:
                    elbo_losses_all.append(None)

            # Flatten for convenience
            n_hold = X_hold.shape[0]

            # ---- Step 2: Compute hold-out loss, or pairwise loss if tandem ----
            if self.bound_type == "tandem":
                # 2a) Build H: shape (m, n_hold), where H[i,t] = 1 if model i got sample t wrong
                H = np.zeros((m, n_hold), dtype=np.float32)

                # We will need one rng key per model to draw posterior samples for predictions
                rng, *all_keys = jr.split(rng, num=m + 1)
                hold_keys = all_keys[:m]

                for i, (module, _) in enumerate(models_states):
                    # "preds" is an array of shape (num_samples, n_hold, C) or (num_samples, n_hold)
                    preds = module.predict(
                        X_hold,
                        hold_keys[i],
                        posterior="logits",
                        num_samples=num_samples,
                    )
                    if self.task == "binary":
                        # preds: (num_samples, n_hold) of logits → average over posterior,
                        # then threshold at 0.5 to get a {0,1} label per sample
                        # (We do sigmoid + mean → float vector of length n_hold → threshold)
                        avg_probs = jax.nn.sigmoid(
                            preds.reshape((num_samples, -1))
                        ).mean(axis=0)
                        pred_labels = (np.array(avg_probs) >= 0.5).astype(int)
                        H[i, :] = (pred_labels != y_hold_np).astype(np.float32)

                    elif self.task == "multiclass":
                        # preds: (num_samples, n_hold, C) of logits per class → average over posterior
                        avg_probs = jax.nn.softmax(preds, axis=-1).mean(
                            axis=0
                        )  # shape (n_hold, C)
                        pred_labels = np.array(
                            jnp.argmax(avg_probs, axis=-1)
                        )  # shape (n_hold,)
                        H[i, :] = (pred_labels != y_hold_np).astype(np.float32)

                    else:  # regression
                        # preds: (num_samples, n_hold) of real values → average → compare MSE
                        avg_preds = preds.reshape((num_samples, -1)).mean(axis=0)
                        mse_vector = (avg_preds - y_hold_np) ** 2
                        clipped = np.minimum(mse_vector, self.L_max) / self.L_max
                        H[i, :] = clipped.astype(
                            np.float32
                        )  # treat that as the “loss per sample”

                # 2b) Build the m×m pairwise‐loss matrix and extract the diagonal
                pair_losses = (H @ H.T) / float(n_hold)  # shape (m, m)
                losses = np.diag(
                    pair_losses
                ).copy()  # length-m vector of marginal losses
            else:
                # For pbkl, pblambda, pbbernstein: only a 1D vector of hold-out losses is needed
                losses = np.zeros(m, dtype=np.float32)
                rng, *hold_keys = jr.split(rng, num=m + 1)
                hold_keys = hold_keys[:m]
                for i, (module, _) in enumerate(models_states):
                    losses[i] = self._compute_hold_loss(
                        module, X_hold, y_hold_np, hold_keys[i], num_samples
                    )
                pair_losses = None  # not used in these branches

            # ---- Step 3: PAC-Bayes optimize λ and ρ ----
            pi = np.full(m, 1.0 / m, dtype=np.float64)
            rho = pi.copy()
            n_r = n_hold
            lam = max(1.0 / math.sqrt(n_r), 0.5)
            Crit = CRITERIA[self.bound_type]()
            prev_bound = np.inf

            for _ in range(200):
                # 3a) Compute KL(ρ‖π)
                kl_val = float((rho * np.log(rho / pi)).sum())

                # 3b) Call the right compute(...) based on bound_type
                if self.bound_type == "tandem":
                    # Note: passes pair_losses (shape (m,m)) as first argument
                    stat, bound = Crit.compute(
                        pair_losses, rho, kl_val, n_r, self.delta, lam, n_r
                    )
                else:
                    # For pbkl, pblambda, pbbernstein: pass 1D “losses” as first argument
                    stat, bound = Crit.compute(
                        losses, rho, kl_val, n_r, self.delta, lam, n_r
                    )

                # 3c) Convergence check
                if abs(prev_bound - bound) < 1e-6:
                    break
                prev_bound = bound

                # 3d) Update λ (Thiemann’s closed-form) and re-compute ρ
                lam = 2.0 / (
                    math.sqrt(
                        1
                        + 2
                        * n_r
                        * stat
                        / (kl_val + math.log(2 * math.sqrt(n_r) / self.delta))
                    )
                    + 1
                )
                shift = losses.min()
                w = np.exp(-lam * n_r * (losses - shift))
                rho = w / w.sum()

            # ---- Step 4: Evaluate majority-vote on test set ----
            rng, test_key = jr.split(rng)
            test_loss = self._majority_vote_test_loss(
                models_states, rho, X_test, y_test_np, test_key, num_samples
            )

            # ---- Record this m’s results ----
            self.results_.append(
                {
                    "m": m,
                    "hold_loss": float((rho * losses).sum()),
                    "bound": float(bound),
                    "test_loss": float(test_loss),
                    "elbo_losses": elbo_losses_all,
                }
            )

            # Update “best” if this bound is smaller
            if bound < self.best_bound_:
                self.best_bound_ = bound
                self.best_m_ = m
                self.best_rho_ = rho.copy()
                # Deep-copy trained modules so we can use them later in predict_labels()
                self.best_models_states_ = [
                    (copy.deepcopy(mod), None) for (mod, _) in models_states
                ]
                self.best_elbo_losses_ = copy.deepcopy(elbo_losses_all)

        # Save the best ensemble for predict_labels()
        self._last_models_states = self.best_models_states_
        self._last_rho = self.best_rho_
        self.is_fitted = True
        return self

    def predict(self, X, rng_key=None, num_samples=100):
        """
        Return predicted labels (or regression values) for X using the best ensemble.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_labels().")
        if rng_key is None:
            rng_key = jr.PRNGKey(self.seed)

        n = X.shape[0]
        m = len(self._last_models_states)
        rng_keys = jr.split(rng_key, num=m + 1)[1:]
        rho = self._last_rho

        if self.task == "binary":
            P_votes = np.zeros((m, n))
            for i, (module, _) in enumerate(self._last_models_states):
                preds = module.predict(
                    X, rng_keys[i], posterior="logits", num_samples=num_samples
                )
                probs = jax.nn.sigmoid(preds.reshape((num_samples, -1))).mean(axis=0)
                P_votes[i, :] = np.where(np.array(probs) >= 0.5, +1, -1)
            agg = (rho[:, None] * P_votes).sum(axis=0)
            return (agg >= 0).astype(int)

        elif self.task == "multiclass":

            sample_preds = self._last_models_states[0][0].predict(
                X, rng_keys[0], posterior="logits", num_samples=1
            )
            C = sample_preds.shape[-1]
            votes = np.zeros((X.shape[0], C))
            for i, (module, _) in enumerate(self._last_models_states):
                preds = module.predict(
                    X, rng_keys[i], posterior="logits", num_samples=num_samples
                )
                probs = jax.nn.softmax(preds, axis=-1).mean(axis=0)
                votes += self._last_rho[i] * np.array(probs)
            return votes.argmax(axis=1)

        else:  # regression
            preds_array = np.zeros((m, n))
            for i, (module, _) in enumerate(self._last_models_states):
                preds = module.predict(
                    X, rng_keys[i], posterior="logits", num_samples=num_samples
                )
                preds_array[i, :] = preds.reshape((num_samples, -1)).mean(axis=0)
            return (rho[:, None] * preds_array).sum(axis=0)

    def predict_proba(self, X, rng_key=None, num_samples=100):
        """
        Return ensemble probabilities (or regression means) for X
        using the best ensemble weights rho.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        if rng_key is None:
            rng_key = jr.PRNGKey(self.seed)

        X = jnp.asarray(X, dtype=jnp.float32)
        n = X.shape[0]
        m = len(self._last_models_states)
        # split rng into one key per model
        keys = jr.split(rng_key, num=m + 1)[1:]
        rho = self._last_rho

        if self.task == "binary":
            # collect each model's pointwise prob
            probs = jnp.stack(
                [
                    jax.nn.sigmoid(
                        self._last_models_states[i][0]
                        .predict(
                            X, keys[i], posterior="logits", num_samples=num_samples
                        )
                        .reshape((num_samples, -1))
                    ).mean(axis=0)
                    for i in range(m)
                ]
            )  # shape (m, n)
            # weighted mixture
            return jnp.dot(rho, probs)  # shape (n,)

        elif self.task == "multiclass":
            # first predict one sample to get num classes
            sample_logits = self._last_models_states[0][0].predict(
                X, keys[0], posterior="logits", num_samples=1
            )
            C = sample_logits.shape[-1]
            # collect each model's mean class-prob vector
            probs = jnp.stack(
                [
                    jax.nn.softmax(
                        self._last_models_states[i][0].predict(
                            X, keys[i], posterior="logits", num_samples=num_samples
                        ),
                        axis=-1,
                    ).mean(
                        axis=0
                    )  # shape (n, C)
                    for i in range(m)
                ]
            )  # shape (m, n, C)
            # weigh and sum over models
            return jnp.tensordot(rho, probs, axes=([0], [0]))  # shape (n, C)

        else:  # regression
            means = jnp.stack(
                [
                    self._last_models_states[i][0]
                    .predict(X, keys[i], posterior="logits", num_samples=num_samples)
                    .reshape((num_samples, -1))
                    .mean(axis=0)
                    for i in range(m)
                ]
            )  # shape (m, n)
            return jnp.dot(rho, means)  # shape (n,)

    @property
    def results(self):
        return self.results_

    @property
    def hold_losses(self):
        return {r["m"]: r["hold_loss"] for r in self.results_}

    @property
    def bounds(self):
        return {r["m"]: r["bound"] for r in self.results_}

    @property
    def test_losses(self):
        return {r["m"]: r["test_loss"] for r in self.results_}

    @property
    def elbo_losses(self):
        return {r["m"]: r["elbo_losses"] for r in self.results_}

    def summary(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before summary().")
        header = f"{'m':>4s} | {'HoldLoss':>10s} | {'Bound':>10s} | {'TestLoss':>10s}"
        print(header)
        print("-" * len(header))
        for r in self.results_:
            test_str = f"{r['test_loss']:.4f}" if r["test_loss"] is not None else "None"
            print(
                f"{r['m']:4d} | {r['hold_loss']:.4f}   | {r['bound']:.4f}   | {test_str}"
            )


# ------------------------
# Minimal Bayesian Logistic Regression Model
# ------------------------
class BayesLogisticRegression(Module):
    def __init__(self, in_dim, method="svi"):
        super().__init__(method=method)
        self.in_dim = in_dim

    def __call__(self, X, Y=None):
        N, D = X.shape
        X = bnn.Linear(D, D)(X)

        logits = bnn.Linear(D, 1, name="name")(X)
        logits = logits.squeeze()
        numpyro.deterministic("logits", logits)
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=Y)


# ------------------------
# Test block
# ------------------------
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def plot_weak_learners_convergence(elbo_losses, m):
        """
        Plot ELBO loss convergence for each weak learner in the ensemble.

        Parameters:
        - elbo_losses: List of lists of ELBO losses, one list per weak learner.
        - m: Number of learners (should match len(elbo_losses)).
        """
        plt.figure()
        for i, losses in enumerate(elbo_losses):
            plt.plot(losses, label=f"Learner {i+1}")
        plt.xlabel("SVI Step")
        plt.ylabel("ELBO Loss")
        plt.title(f"Convergence of {m} Weak Learners")
        plt.legend()
        plt.show()

    # 1) Generate a small synthetic binary classification dataset
    Xc, yc = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        flip_y=0.1,
        class_sep=1.0,
        random_state=0,
    )
    # Split into train / hold / test (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=1, stratify=yc
    )
    X_train, X_hold, y_train, y_hold = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=2, stratify=y_temp
    )  # 0.25 * 0.8 = 0.2

    # Convert to JAX arrays
    X_train = jnp.array(X_train, dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.float32)
    X_hold = jnp.array(X_hold, dtype=jnp.float32)
    y_hold = jnp.array(y_hold, dtype=jnp.float32)
    X_test = jnp.array(X_test, dtype=jnp.float32)
    y_test = jnp.array(y_test, dtype=jnp.float32)

    y_train = jnp.asarray(y_train, dtype=jnp.int32)
    y_hold = jnp.asarray(y_hold, dtype=jnp.int32)
    y_test = jnp.asarray(y_test, dtype=jnp.int32)

    in_dim = X_train.shape[1]

    # 2) Define three identical module constructors (same BayesLogisticRegression)
    def ctor1(key):
        return BayesLogisticRegression(in_dim, method="svi")

    def ctor2(key):
        return BayesLogisticRegression(in_dim, method="svi")

    def ctor3(key):
        return BayesLogisticRegression(in_dim, method="svi")

    module_constructors = [ctor1, ctor2, ctor3]

    # 3) Build the ensemble
    ensemble = PacBayesEnsembleNumPyro(
        module_constructors=module_constructors,
        task="binary",
        bound_type="unexpected-bernstein",
        delta=0.05,
        L_max=1.0,
        seed=42,
    )

    # 4) Fit the ensemble on (train, hold, test) with m_values = [1, 2, 3]
    ensemble.fit(
        X_train,
        y_train,
        X_hold,
        y_hold,
        X_test,
        y_test,
        m_values=[1, 2, 3],
        svi_steps=10,
        num_samples=50,
    )

    # 5) Print summary of hold‐out loss, bound, and test loss
    print("\nEnsemble Summary:")
    ensemble.summary()

    # 6) Access raw dictionaries and print them
    print("\nHold losses by m:", ensemble.hold_losses)
    print("Bounds by m:    ", ensemble.bounds)
    print("Test losses by m:", ensemble.test_losses)

    # 7) Check that predict_labels works and compute in‐sample error on hold set
    rng_test = jr.PRNGKey(999)
    y_hold_preds = ensemble.predict(X_hold, rng_test, num_samples=50)
    probs = ensemble.predict_proba(X_hold, rng_test)
    from sklearn.metrics import log_loss

    y_hold_np = np.array(y_hold).ravel()
    hold_err = zero_one_loss(y_hold_np, y_hold_preds)
    loss = log_loss(y_hold_np, np.array(probs))
    print(f"\nHold‐out zero‐one error from predict_labels: {hold_err:.4f}")
    print(f"Log Loss: {loss}")

    # 8) Check predict_labels on test set
    rng_test2 = jr.PRNGKey(1001)
    y_test_preds = ensemble.predict(X_test, rng_test2, num_samples=50)
    y_test_np = np.array(y_test).ravel()

    test_err = zero_one_loss(y_test_np, y_test_preds)
    print(f"Test zero‐one error from predict_labels: {test_err:.4f}")

    # After ensemble.fit(...)
    m = ensemble.best_m_  # best ensemble size, e.g. 10
    elbo_dict = ensemble.elbo_losses  # this is a dict: {m1: [...], m2: [...], …}
    elbo_losses = elbo_dict[m]  # get the list-of-lists for ensemble size m

    # Now plot:
    plot_weak_learners_convergence(elbo_losses, m)
