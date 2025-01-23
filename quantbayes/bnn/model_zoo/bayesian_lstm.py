import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from quantbayes.bnn.layers.base import Module
from quantbayes.bnn.layers import LSTM, Linear

class BayesianLSTM(Module):
    """
    A Bayesian LSTM regressor that produces a single scalar output `y`.
    """

    def __init__(self, input_dim, hidden_dim=32, method="svi"):
        """
        :param input_dim: Number of features per time step.
        :param hidden_dim: The dimension of the LSTM's hidden state.
        :param method: 'svi', 'nuts', or 'steinvi'.
        """
        super().__init__(method=method, task_type="regression")
        self.lstm = LSTM(input_dim, hidden_dim, name="bayes_lstm")
        self.out_layer = Linear(hidden_dim, 1, name="bayes_lstm_out")

    def __call__(self, X, y=None):
        """
        X: shape (batch_size, seq_len, input_dim)
        y: shape (batch_size,) or (batch_size,1) for regression
        """
        # 1) LSTM forward pass
        #    returns (outputs, final_state)
        outputs, (h_t, c_t) = self.lstm(X)
        # Let’s just use the final hidden state h_t
        # shape (batch_size, hidden_dim)

        # 2) Map final hidden state to scalar
        logits = self.out_layer(h_t).squeeze(-1)  # shape (batch_size,)

        # 3) Sample observation noise
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))

        # 4) Observe y with Normal(logits, sigma)
        numpyro.sample("y", dist.Normal(logits, sigma), obs=y)

        # For convenience we’ll return logits
        return logits

def test_bayesian_lstm():
    # Suppose we have data X of shape (N, seq_len, input_dim),
    # and y of shape (N,) for regression.
    # We'll make up some random data:
    rng = jax.random.PRNGKey(0)
    N, seq_len, input_dim = 50, 10, 5
    X = jax.random.normal(rng, (N, seq_len, input_dim))
    true_w = jax.random.normal(rng, (input_dim,))
    y = jnp.sum(X[:, -1, :] * true_w, axis=-1) + 0.1 * jax.random.normal(rng, (N,))

    model = BayesianLSTM(input_dim, hidden_dim=16, method="svi")
    model.compile(num_steps=2000, learning_rate=0.005)
    rng_key = jax.random.PRNGKey(42)
    model.fit(X, y, rng_key=rng_key)
    print("BayesianLSTM training done!")

if __name__ == "__main__":
    test_bayesian_lstm()