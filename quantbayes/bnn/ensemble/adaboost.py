import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
from numpyro import sample, plate
import numpyro.distributions as dist
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional

from quantbayes.bnn.layers import Module


# -------------------------------------------------------------------
# 2) Define a simple Bayesian weak learner: a 2-layer Bayesian MLP
# -------------------------------------------------------------------
class BayesianMLP(Module):
    """
    A very simple Bayesian neural network for binary classification.
    - Prior: standard Normal on every weight & bias.
    - Likelihood: Bernoulli(logits = final MLP output).
    """

    def __init__(self, in_features: int, hidden: int = 16, method: str = "svi"):
        super().__init__(method=method)
        self.in_features = in_features
        self.hidden = hidden
        # NumPyro inference objects will be set in compile()
        self.compile()  # set up SVI (default) or you can override with method="nuts"

    def __call__(self, x, y=None):
        """
        Defines the Bayesian model in NumPyro syntax.
        Input:
          - x: shape [batch_size, in_features]
          - y: shape [batch_size], integer {0,1}
        """
        # Layer 1 weights & bias
        w1 = sample("w1", dist.Normal(jnp.zeros((self.in_features, self.hidden)), 1.0))
        b1 = sample("b1", dist.Normal(jnp.zeros((self.hidden,)), 1.0))
        # Layer 2 weights & bias
        w2 = sample("w2", dist.Normal(jnp.zeros((self.hidden, 1)), 1.0))
        b2 = sample("b2", dist.Normal(jnp.zeros((1,)), 1.0))

        # Forward pass
        hidden = jnp.tanh(jnp.matmul(x, w1) + b1)  # shape [batch_size, hidden]
        logits = jnp.squeeze(jnp.matmul(hidden, w2) + b2, axis=-1)  # shape [batch_size]
        logits = numpyro.deterministic("logits", logits)
        # Likelihood
        with plate("data", x.shape[0]):
            sample("obs", dist.Bernoulli(logits=logits), obs=y)

        # We return logits so that Predictive can sample "logits"
        return logits


# -------------------------------------------------------------------
# 3) AdaBoostBNN: wrap Bayesian weak learners into an AdaBoost loop
# -------------------------------------------------------------------
class AdaBoostBNN:
    """
    AdaBoost ensemble whose weak learners are Bayesian neural nets (NumPyro Modules).
    Follows the exact sample-weight update rules from scikit-learn, but uses
    posterior-mean logits to form h_b(x) = sign(mean_logits).
    """

    def __init__(
        self,
        model_constructor: Callable[[jax.random.PRNGKey], Module],
        num_estimators: int = 10,
        learning_rate: float = 1.0,
        num_posterior_samples: int = 100,
        *,
        method: str = "svi",  # can be "nuts", "svi", or "steinvi"
        svi_steps: int = 1000,
        rng_key: Optional[jax.random.PRNGKey] = None,
    ):
        """
        Parameters
        ----------
        model_constructor : function
            Given a PRNGKey, returns a fresh subclass of Module (e.g. BayesianMLP).
            That Module must implement compile()/fit()/predict().
        num_estimators : int
            Number of boost rounds (B).
        learning_rate : float
            Shrinkage factor on α_b (≤1.0).
        num_posterior_samples : int
            How many posterior samples to draw when calling predict(X).
        method : str
            Inference method to pass to each weak learner ("nuts", "svi", or "steinvi").
        svi_steps : int
            If method=="svi", how many SVI steps to run on each weak learner.
        rng_key : PRNGKey (optional)
            For reproducibility.  Defaults to jr.PRNGKey(0) if None.
        """
        self.model_constructor = model_constructor
        self.num_estimators = num_estimators
        self.learning_rate = learning_rate
        self.num_posterior_samples = num_posterior_samples
        self.method = method
        self.svi_steps = svi_steps

        # After fitting, we’ll store a list of (weak_module, α_b) tuples:
        self.weak_learners: List[Tuple[Module, float]] = []

        # For convergence plotting:
        self.train_errors: List[float] = []
        self.val_errors: List[float] = []

        # RNG
        self.rng_key = jr.PRNGKey(0) if rng_key is None else rng_key

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        *,
        X_val: Optional[jnp.ndarray] = None,
        y_val: Optional[jnp.ndarray] = None,
    ):
        """
        Train the BNN‐based AdaBoost ensemble on (X, y).  For binary classification,
        y ∈ {0,1}.  All indexing is JAX‐friendly.  We keep X, y on JAX arrays.

        Optionally pass (X_val, y_val) to track validation error at each stage.
        """
        N = X.shape[0]
        # 1) Initialize sample weights uniformly
        w = jnp.ones(N) / N

        for b in range(self.num_estimators):
            # ─── 2) Instantiate a fresh weak BNN ───
            self.rng_key, subkey = jr.split(self.rng_key)
            weak: Module = self.model_constructor(subkey)
            weak.method = self.method
            weak.compile()  # use default SVI or override via weak.method="nuts" etc.

            # 3) Fit the BNN on the weighted data. Unfortunately, NumPyro’s SVI/MCMC
            #    don’t accept sample‐weights directly, so we re‐sample (X_i,y_i) ∼ w
            #    _with_ replacement N times to approximate the weighted dataset.
            #    This is the same “bootstrap approximation” idea for BNNs.
            #    (Exact weighting would require re‐writing the ELBO’s log‐likelihood.)
            #
            #    In practice, if the dataset is not too large, this gives a good approximation.
            #
            self.rng_key, samp_key = jr.split(self.rng_key)
            idx = jr.choice(samp_key, a=N, shape=(N,), p=w)  # shape (N,)
            X_boot = X[idx]
            y_boot = y[idx]

            # Fit with the chosen inference.  If SVI, run SVI for svi_steps.  If NUTS, just run MCMC.
            if self.method == "svi":
                weak.fit(X_boot, y_boot, subkey, num_steps=self.svi_steps)
            else:
                weak.fit(X_boot, y_boot, subkey)

            # ─── 4) Get posterior‐sampled logits on the _full_ training set ───
            self.rng_key, pred_key = jr.split(self.rng_key)
            logits_samples = weak.predict(
                X, pred_key, posterior="logits", num_samples=self.num_posterior_samples
            )
            # logits_samples has shape [num_posterior_samples, N]

            # 5) Collapse to a point‐estimate: mean over posterior samples
            mean_logits = jnp.mean(logits_samples, axis=0)  # shape (N,)

            # Convert mean_logits → h_signed ∈ {−1, +1}
            h_signed = jnp.sign(mean_logits)
            h_signed = jnp.where(h_signed == 0, 1, h_signed)

            # Convert y ∈ {0,1} → y_signed ∈ {−1,+1}
            y_signed = 2 * y - 1

            # 6) Compute weighted error ε_b = sum_i w_i · 1{h_i ≠ y_i}
            incorrect = (h_signed != y_signed).astype(jnp.float32)
            epsilon = jnp.sum(w * incorrect)

            # If ε_b > 0.5, flip hypothesis:
            if epsilon > 0.5:
                h_signed = -h_signed
                epsilon = 1.0 - epsilon

            # 7) Compute α_b = ½ ln((1−ε)/ε)
            alpha = 0.5 * jnp.log((1.0 - epsilon) / (epsilon + 1e-15))

            # 8) Update sample weights: w_i ← w_i · exp(−α · y_i · h_i), then renormalize
            w = w * jnp.exp(-alpha * y_signed * h_signed)
            w = w / jnp.sum(w)

            # 9) Store this weak learner and α_b
            self.weak_learners.append((weak, float(alpha)))
            print(
                f"Weak learner {b+1}/{self.num_estimators}  –  ε = {epsilon:.4f},  α = {float(alpha):.4f}"
            )

            # ─── 10) Record staged 0–1 training error ───
            train_pred = self._staged_predict(X)  # uses first (b+1) learners
            train_err = float(jnp.mean((train_pred != y).astype(jnp.float32)))
            self.train_errors.append(train_err)

            if X_val is not None and y_val is not None:
                val_pred = self._staged_predict(X_val)
                val_err = float(jnp.mean((val_pred != y_val).astype(jnp.float32)))
                self.val_errors.append(val_err)

    def _staged_predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Run “ensemble prediction” using all stored weak learners.
        For each learner b:
          1) Draw num_posterior_samples of logits on X: shape [S, N].
          2) Average → mean_logits_b (shape [N])
          3) h_b(x) = sign(mean_logits_b) ∈ {−1,+1}
          4) Weighted vote: α_b · h_b(x)
        Finally take sign of sum over b.  Convert back to {0,1}.
        """
        N = X.shape[0]
        agg = jnp.zeros(N)

        for weak, alpha in self.weak_learners:
            self.rng_key, pred_key = jr.split(self.rng_key)
            logits_samples = weak.predict(
                X, pred_key, posterior="logits", num_samples=self.num_posterior_samples
            )
            # logits_samples: [S, N]
            mean_logits = jnp.mean(logits_samples, axis=0)  # shape (N,)
            h_signed = jnp.sign(mean_logits)
            h_signed = jnp.where(h_signed == 0, 1, h_signed)
            agg = agg + alpha * h_signed

        final_signed = jnp.sign(agg)
        final_signed = jnp.where(final_signed == 0, 1, final_signed)
        return ((final_signed + 1) // 2).astype(jnp.int32)  # {0,1}

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Same as _staged_predict, since we always want all weak learners.
        """
        return self._staged_predict(X)


# -------------------------------------------------------------------
# 4) Example usage: train & evaluate AdaBoostBNN on a toy dataset
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 4.1) Synthetic binary dataset
    X_np, y_np = make_classification(
        n_samples=500, n_features=10, n_informative=5, random_state=0
    )
    # y_np ∈ {0,1}

    # 4.2) Train/val/test splits
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=0
    )
    # Now: 60% train, 20% val, 20% test.

    # 4.3) Convert to JAX arrays
    X_train_j = jnp.array(X_train, dtype=jnp.float32)
    y_train_j = jnp.array(y_train, dtype=jnp.int32)
    X_val_j = jnp.array(X_val, dtype=jnp.float32)
    y_val_j = jnp.array(y_val, dtype=jnp.int32)
    X_test_j = jnp.array(X_test, dtype=jnp.float32)
    y_test_j = jnp.array(y_test, dtype=jnp.int32)

    # 4.4) Define a function that constructs a fresh BayesianMLP
    def make_bnn(key: jax.random.PRNGKey) -> Module:
        in_dim = X_train_j.shape[-1]
        # Here we choose SVI; if you want NUTS, you can pass method="nuts" to BayesianMLP
        return BayesianMLP(in_features=in_dim, hidden=16, method="svi")

    # 4.5) Instantiate AdaBoostBNN
    ada_bnn = AdaBoostBNN(
        model_constructor=make_bnn,
        num_estimators=10,
        learning_rate=1.0,
        num_posterior_samples=100,
        method="svi",
        svi_steps=500,
        rng_key=jr.PRNGKey(42),
    )

    # 4.6) Fit on training data, tracking validation error as well
    ada_bnn.fit(X_train_j, y_train_j, X_val=X_val_j, y_val=y_val_j)

    # 4.7) Evaluate on test set
    y_pred_j = ada_bnn.predict(X_test_j)
    y_pred_np = np.array(y_pred_j)
    test_acc = accuracy_score(y_test, y_pred_np)
    print(f"\nAdaBoostBNN Test Accuracy: {test_acc:.4f}")

    # 4.8) Plot staged training & validation error
    stages = np.arange(1, len(ada_bnn.train_errors) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(stages, ada_bnn.train_errors, label="Train Error", marker="o")
    if ada_bnn.val_errors:
        plt.plot(stages, ada_bnn.val_errors, label="Val Error", marker="s")
    plt.xlabel("Number of Weak Learners")
    plt.ylabel("0–1 Error Rate")
    plt.title("AdaBoostBNN Convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
