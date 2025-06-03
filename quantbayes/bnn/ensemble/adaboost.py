# ada_boost_numpyro_refactored.py

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
from typing import Tuple, Optional


# -------------------------------------------------------------------
# 1) Encapsulate a small Bayesian neural net (two layers) in a class
# -------------------------------------------------------------------
class BayesianBNN:
    """
    A simple Bayesian neural net (one hidden layer) for binary classification,
    with methods:
      - compile(lr): sets up AutoNormal guide and SVI
      - fit(X, y, rng_key, num_steps): runs SVI on (X, y)
      - predict_probs(X_new, rng_key, num_samples): draws posterior‐predictive samples
        and returns mean probability P(y=1 | x).
    """

    def __init__(self, in_features: int, hidden_size: int = 32, num_classes: int = 2):
        """
        :param in_features: number of input features D
        :param hidden_size: number of hidden units H (default 32)
        """
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Placeholders; will be set in compile()
        self.guide = None  # AutoNormal guide
        self.svi = None  # SVI object
        self.svi_state = None  # final SVI state
        self.params = None  # variational parameters dict

    def bnn_model(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None):
        """
        NumPyro model: two‐layer BNN with Normal priors, ReLU hidden, Bernoulli likelihood.

        :param x: shape (batch_size, D)
        :param y: shape (batch_size,) with values in {0,1}; may be None at predict time.
        """
        D = self.in_features
        H = self.hidden_size

        # 1) hidden‐layer weights + bias
        w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D, H)), 1.0).to_event(2))
        b1 = numpyro.sample("b1", dist.Normal(jnp.zeros((H,)), 1.0).to_event(1))

        # 2) output‐layer weights + bias (H → 1)
        w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((H, 1)), 1.0).to_event(2))
        b2 = numpyro.sample("b2", dist.Normal(jnp.zeros((1,)), 1.0).to_event(1))

        # 3) forward pass
        hidden = jax.nn.relu(jnp.matmul(x, w1) + b1)  # (batch_size, H)
        logits = jnp.squeeze(jnp.matmul(hidden, w2) + b2, -1)  # (batch_size,)

        # 4) Bernoulli likelihood
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

    def compile(self, lr: float = 1e-3):
        """
        Instantiate the AutoNormal guide and SVI instance.

        :param lr: learning rate for Adam‐SVI
        """
        self.guide = AutoNormal(self.bnn_model)
        optimizer = Adam(lr)
        self.svi = SVI(self.bnn_model, self.guide, optimizer, loss=Trace_ELBO())

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        num_steps: int = 1000,
    ):
        """
        Run SVI on (X, y) to learn variational parameters.

        :param X: shape (N, D), jnp.ndarray of float32
        :param y: shape (N,), jnp.ndarray of int32 in {0,1}
        :param rng_key: JAX PRNGKey
        :param num_steps: number of SVI updates
        """
        # Initialize SVI state
        self.svi_state = self.svi.init(rng_key, x=X, y=y)

        # One training step function
        def _train_step(state, _):
            state, _ = self.svi.update(state, x=X, y=y)
            return state, None

        # Run num_steps iterations of SVI
        self.svi_state, _ = jax.lax.scan(
            _train_step, self.svi_state, None, length=num_steps
        )

        # Extract variational parameters
        self.params = self.svi.get_params(self.svi_state)

    def predict_probs(
        self,
        X_new: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        num_samples: int = 50,
    ) -> jnp.ndarray:
        """
        Draw posterior‐predictive samples at X_new and return P(y=1 | x) estimate.

        :param X_new: shape (M, D)
        :param rng_key: JAX PRNGKey for Predictive
        :param num_samples: how many posterior draws
        :return: jnp.ndarray of shape (M,) giving mean probability for y=1
        """
        predictive = Predictive(
            self.bnn_model,
            guide=self.guide,
            params=self.params,
            num_samples=num_samples,
        )
        samples = predictive(rng_key, x=X_new)[
            "obs"
        ]  # shape (num_samples, M), values {0,1}
        return jnp.mean(samples, axis=0)  # (M,) mean probability


class AdaBoost:
    """
    AdaBoost where each weak learner is a BayesianBNN (trained via SVI).
    Supports both:
      • Binary classification (num_classes=2, y ∈ {0,1})
      • Multiclass classification (num_classes=K, y ∈ {0,…,K−1})
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int = 32,
        num_boost_rounds: int = 6,
        nvi_steps: int = 1000,
        posterior_samples: int = 50,
        lr: float = 1e-3,
        num_classes: int = 2,
    ):
        """
        :param in_features: input dimension D
        :param hidden_size: hidden‐layer size for each BNN
        :param num_boost_rounds: number of AdaBoost rounds (B)
        :param nvi_steps: SVI steps per weak learner
        :param posterior_samples: posterior‐predictive samples per model (for ε)
        :param lr: learning rate for each SVI
        :param num_classes: 2 for binary, K for K-way multiclass
        """
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.B = num_boost_rounds
        self.nvi_steps = nvi_steps
        self.posterior_samples = posterior_samples
        self.lr = lr
        self.num_classes = num_classes  # if 2→binary, if >2→multiclass

        self.models: list[BayesianBNN] = []
        self.alphas: list[float] = []

    @staticmethod
    def _weighted_sample_indices(
        key: jax.random.PRNGKey, w: jnp.ndarray, N: int
    ) -> jnp.ndarray:
        return jax.random.choice(key, a=N, shape=(N,), p=w)

    def fit(self, X: np.ndarray, y: np.ndarray, rng_key: jax.random.PRNGKey):
        """
        Train AdaBoost on (X, y).  For binary: y ∈ {0,1}.  For multiclass: y ∈ {0..K−1}.
        """
        # Convert to JAX
        X_j = jnp.array(X, dtype=jnp.float32)  # (N, D)
        y_j = jnp.array(y, dtype=jnp.int32)  # (N,)
        N = X_j.shape[0]
        K = self.num_classes

        # 1) Initialize sample weights
        w = jnp.ones((N,)) / N

        # 2) Draw B subkeys
        keys = jax.random.split(rng_key, self.B)

        for b in range(self.B):
            key_b = keys[b]

            # 3) Weighted bootstrap sampling
            key_b, key_samp = jax.random.split(key_b)
            idx = self._weighted_sample_indices(key_samp, w, N)  # (N,)
            X_sample = X_j[idx, :]  # shape (N, D)
            y_sample = y_j[idx]  # shape (N,)

            # 4) Build & train a fresh BayesianBNN on (X_sample, y_sample)
            model = BayesianBNN(
                self.in_features,
                hidden_size=self.hidden_size,
                num_classes=self.num_classes,
            )
            model.compile(lr=self.lr)
            key_b, key_fit = jax.random.split(key_b)
            model.fit(X_sample, y_sample, rng_key=key_fit, num_steps=self.nvi_steps)

            # 5) Compute posterior‐predictive “probabilities” on the entire training set
            key_b, key_pred = jax.random.split(key_b)
            # samples shape: (posterior_samples, N), each entry ∈ {0..K−1}
            samples = model.predict_probs(
                X_j, rng_key=key_pred, num_samples=self.posterior_samples
            )
            # But for SAMME we only need a hard label h_b(x):
            # (If predict_probs returns mean prob, you must instead do Predictive(...)[“obs”], one‐hot, etc.)
            #
            # Let’s assume predict_probs(...) returns the *mean* probability for class=1 in binary,
            # or for multiclass we might need a separate method that returns E[one_hot(obs)].
            #
            # Easiest: slightly modify predict_probs() so that it returns a length-N, length-K array of mean probs.
            # Then we do:
            #     class_probs = model.predict_probs_multiclass( … )  # shape (N, K)
            #     h = jnp.argmax(class_probs, axis=1)                 # shape (N,)
            #
            # For binary (K=2), we can threshold at 0.5:
            #     prob_pos = model.predict_probs(…)  # shape (N,), P(y=1|x)
            #     h = (prob_pos >= 0.5).astype(jnp.int32)

            if K == 2:
                # 5a) BINARY case: get P(y=1|x) as a length-N vector, then threshold
                prob_pos = (
                    samples  # shape (posterior_samples, N) → we need mean over axis=0
                )
                # Suppose predict_probs returned shape (N,) = mean P(y=1|x).
                # If it returned (posterior_samples, N), do: prob_pos = jnp.mean(samples, axis=0)
                # Here we assume predict_probs already returns (N,) directly.
                h_binary = (prob_pos >= 0.5).astype(jnp.int32)  # shape (N,)
                # Convert to ±1
                y_pm1 = 2 * y_j - 1  # {0→−1, 1→+1}
                h_pm1 = 2 * h_binary - 1  # {0→−1, 1→+1}

                # 6a) Compute weighted error ε_b
                incorrect = (h_binary != y_j).astype(jnp.float32)
                epsilon = jnp.sum(w * incorrect)

                # If ε_b > 0.5, flip h_pm1 so error → 1−ε
                if epsilon > 0.5:
                    h_pm1 = -h_pm1
                    epsilon = 1.0 - epsilon

                # α_b = ½ · ln((1−ε)/ε)
                alpha = 0.5 * jnp.log((1.0 - epsilon) / (epsilon + 1e-15))

                # 7a) Update weights: w_i ← w_i·exp(−α·y_i·h_i)
                w = w * jnp.exp(-alpha * y_pm1 * h_pm1)
                w = w / jnp.sum(w)

            else:
                # 5b) MULTICLASS case: get shape (posterior_samples, N) integer draws (each in 0..K−1)
                # We need a “mean one-hot” to get P(y=k|x), shape (N, K).
                # E.g.:
                #    samples_pp = Predictive(...)[“obs”]  # (posterior_samples, N)
                #    one_hot = jax.nn.one_hot(samples_pp, K)  # (posterior_samples, N, K)
                #    class_probs = jnp.mean(one_hot, axis=0)   # (N, K)
                #
                # For simplicity, let’s assume `model.predict_probs_multiclass(...)` returns (N, K):
                class_probs = samples  # pretend samples is (N, K)
                h = jnp.argmax(
                    class_probs, axis=1
                )  # shape (N,), predicted class ∈ {0..K−1}

                # 6b) Weighted error ε_b = Σ w_i · 1{h_i ≠ y_i}
                incorrect = (h != y_j).astype(jnp.float32)
                epsilon = jnp.sum(w * incorrect)

                # If ε_b ≥ (K−1)/K, skip this weak learner
                if epsilon >= (K - 1) / K:
                    print(
                        f"[Round {b+1}/{self.B}] ε={epsilon:.4f} ≥ (K−1)/K → skipping learner"
                    )
                    continue

                # α_b = ln((1−ε)/ε) + ln(K−1)
                alpha = jnp.log((1.0 - epsilon) / (epsilon + 1e-15)) + jnp.log(K - 1.0)

                # 7b) Update weights:
                # If correct: w_i ← w_i · exp(−α), else w_i ← w_i · exp(+α)
                correct_mask = h == y_j
                w = w * jnp.where(correct_mask, jnp.exp(-alpha), jnp.exp(alpha))
                w = w / jnp.sum(w)

            # 8) Store this round’s model and α
            self.models.append(model)
            self.alphas.append(float(alpha))

            print(f"[Round {b+1}/{self.B}] ε={epsilon:.4f}, α={alpha:.4f}")

    def predict(self, X: np.ndarray, rng_key: jax.random.PRNGKey) -> np.ndarray:
        """
        Predict {0,1} (binary) or {0..K−1} (multiclass) labels for new data X.
        """
        X_j = jnp.array(X, dtype=jnp.float32)  # (M, D)
        M = X_j.shape[0]
        K = self.num_classes

        if K == 2:
            # -------------------
            # Binary final vote: Σ α_b · h_b(x), threshold at 0
            # -------------------
            agg = jnp.zeros((M,))
            keys = jax.random.split(rng_key, len(self.models))
            for b, (model, alpha) in enumerate(zip(self.models, self.alphas)):
                key_b = keys[b]
                prob_pos = model.predict_probs(
                    X_j, rng_key=key_b, num_samples=self.posterior_samples
                )
                h_binary = (prob_pos >= 0.5).astype(jnp.int32)  # {0,1}
                h_pm1 = 2 * h_binary - 1  # {−1,+1}
                agg = agg + alpha * h_pm1

            final_pm1 = jnp.sign(agg)
            final_pm1 = jnp.where(final_pm1 == 0, 1, final_pm1)
            y_pred = (final_pm1 + 1) // 2  # {−1→0, +1→1}
            return np.array(y_pred)

        else:
            # -------------------
            # Multiclass final vote: accumulate α_b in a (M, K) score matrix, then argmax
            # -------------------
            scores = jnp.zeros((M, K))
            keys = jax.random.split(rng_key, len(self.models))
            for b, (model, alpha) in enumerate(zip(self.models, self.alphas)):
                key_b = keys[b]
                # Again, assume model.predict_probs_multiclass(...) returns (M, K):
                class_probs = model.predict_probs(
                    X_j, rng_key=key_b, num_samples=self.posterior_samples
                )
                # Hard‐label vote: h = argmax_k class_probs[i, k]
                h = jnp.argmax(class_probs, axis=1)  # shape (M,)
                one_hot_h = jax.nn.one_hot(h, K)  # (M, K)
                scores = scores + alpha * one_hot_h

            y_pred = jnp.argmax(scores, axis=1)  # (M,)
            return np.array(y_pred)


# -------------------------------------------------------------------
# 3) Example usage on synthetic data
# -------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Create toy binary data: 1000 samples, 20 features
    X_np, y_np = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=0
    )
    y_np = y_np.astype(np.int32)  # ensure {0,1}

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=0
    )

    # Build and fit AdaBoost with BNN weak learners
    in_features = X_train.shape[1]
    ada = AdaBoost(
        in_features=in_features,
        hidden_size=32,  # hidden layer size for each BNN
        num_boost_rounds=6,  # train 6 weak learners
        nvi_steps=1000,  # SVI steps per weak learner
        posterior_samples=50,  # PP draws per model when computing error
        lr=1e-3,  # learning rate for SVI
    )

    key = jax.random.PRNGKey(42)
    ada.fit(X_train, y_train, key)

    # Predict on test set
    test_key = jax.random.PRNGKey(2025)
    y_pred = ada.predict(X_test, test_key)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAdaBoost‐BNN Test Accuracy: {acc:.4f}")
