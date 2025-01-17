from time_forecast.bnn.core.base_inference import BaseInference
from time_forecast.bnn.core.base_task import BaseTask
from typing import Callable
import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import jax.numpy as jnp
import jax


class DenseMCMC(BaseTask, BaseInference):
    def __init__(
        self,
        model: str = "lstm",
        num_warmup: int = 500,
        num_samples: int = 1000,
        hidden_dim: int = 10,
    ):
        """
        Initialize DenseMCMC with model type and hyperparameters.
        """
        super().__init__()
        self.model = model
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.hidden_dim = hidden_dim
        self.mcmc = None

        # Map model names to their corresponding functions
        self.model_map = {
            "lstm": self.lstm,
            "deep_lstm": self.deep_lstm,
            "feedforward": self.feedforward,
            "deep_feedforward": self.deep_feedforward,
        }

    def get_default_model(self) -> Callable:
        """
        Return the default model based on the `model` attribute.
        """
        if self.model not in self.model_map:
            raise ValueError(
                f"Invalid model type: {self.model}. Choose from {list(self.model_map.keys())}"
            )
        return lambda X, y=None: self.model_map[self.model](X, y)

    def bayesian_inference(self, X_train, y_train, rng_key):
        """
        Perform MCMC inference.
        """
        kernel = NUTS(self.get_default_model())
        self.mcmc = MCMC(
            kernel, num_warmup=self.num_warmup, num_samples=self.num_samples
        )
        self.mcmc.run(rng_key, X=X_train, y=y_train)
        self.mcmc.print_summary()

    def retrieve_results(self):
        """
        Retrieve the posterior samples.
        """
        if not self.mcmc:
            raise RuntimeError("No inference results available. Fit the model first.")
        return self.mcmc.get_samples()

    def fit(self, X_train, y_train, rng_key, **kwargs):
        """
        Fit the model using MCMC.
        """
        self.bayesian_inference(X_train, y_train, rng_key, **kwargs)
        self.fitted = True

    def predict(self, X_test, rng_key):
        """
        Predict regression values using the posterior.
        """
        if not self.fitted or self.mcmc is None:
            raise RuntimeError("Model must be fitted before making predictions.")
        posterior_samples = self.mcmc.get_samples()
        predictive = Predictive(self.get_default_model(), posterior_samples)
        preds = predictive(rng_key=rng_key, X=X_test)
        return preds["mean"]

    def visualize(self, y_train, y_val, posterior_preds):
        """
        Visualizes the training data, validation data, and model's predictions
        with uncertainty on the validation set.

        Args:
            y_train: Training target data (shape: (N_train,)).
            y_val: Validation target data (shape: (N_val,)).
            posterior_preds: Model predictions on the validation set with uncertainty (shape: (samples, N_val)).
        """
        import matplotlib.pyplot as plt
        import jax.numpy as jnp

        # Compute prediction statistics
        mean_predictions = posterior_preds.mean(axis=0)
        lower_bound = jnp.percentile(posterior_preds, 2.5, axis=0)
        upper_bound = jnp.percentile(posterior_preds, 97.5, axis=0)

        # Generate time indices
        train_indices = jnp.arange(len(y_train))
        val_indices = jnp.arange(len(y_train), len(y_train) + len(y_val))

        # Plot the results
        plt.figure(figsize=(14, 7))

        # Plot training data
        plt.plot(
            train_indices,
            y_train,
            label="Training Data",
            color="blue",
            alpha=0.7,
        )

        # Plot validation data
        plt.plot(
            val_indices,
            y_val,
            label="Validation Data",
            color="green",
            alpha=0.7,
        )

        # Plot model predictions on validation data
        plt.plot(
            val_indices,
            mean_predictions,
            label="Predicted Mean",
            color="red",
            linestyle="--",
        )

        # Plot uncertainty bounds
        plt.fill_between(
            val_indices,
            lower_bound,
            upper_bound,
            color="gray",
            alpha=0.3,
            label="95% Confidence Interval",
        )

        plt.xlabel("Time Index")
        plt.ylabel("Target Value")
        plt.title("Time Series Predictions with Uncertainty on Validation Data")
        plt.legend()
        plt.grid(True)
        plt.show()

    def lstm(self, X, y=None):
        """
        Bayesian LSTM model with one hidden layer for time series regression using NumPyro.

        Args:
            X: Input data of shape (N, seq_len, input_size).
            y: Target data of shape (N,).
        """
        num_features = X.shape[2]  # input_size
        seq_len = X.shape[1]
        hidden_dim = self.hidden_dim

        # LSTM weights
        W_i = numpyro.sample(
            "W_i", dist.Normal(0, 1).expand([num_features, hidden_dim]).to_event(2)
        )
        U_i = numpyro.sample(
            "U_i", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        b_i = numpyro.sample("b_i", dist.Normal(0, 1).expand([hidden_dim]).to_event(1))

        W_f = numpyro.sample(
            "W_f", dist.Normal(0, 1).expand([num_features, hidden_dim]).to_event(2)
        )
        U_f = numpyro.sample(
            "U_f", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        b_f = numpyro.sample("b_f", dist.Normal(0, 1).expand([hidden_dim]).to_event(1))

        W_o = numpyro.sample(
            "W_o", dist.Normal(0, 1).expand([num_features, hidden_dim]).to_event(2)
        )
        U_o = numpyro.sample(
            "U_o", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        b_o = numpyro.sample("b_o", dist.Normal(0, 1).expand([hidden_dim]).to_event(1))

        W_c = numpyro.sample(
            "W_c", dist.Normal(0, 1).expand([num_features, hidden_dim]).to_event(2)
        )
        U_c = numpyro.sample(
            "U_c", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        b_c = numpyro.sample("b_c", dist.Normal(0, 1).expand([hidden_dim]).to_event(1))

        # Fully connected weights
        fc_weights = numpyro.sample(
            "fc_weights", dist.Normal(0, 1).expand([hidden_dim, 1]).to_event(2)
        )
        fc_bias = numpyro.sample("fc_bias", dist.Normal(0, 1).expand([1]).to_event(1))

        # LSTM forward pass
        h = jnp.zeros((X.shape[0], hidden_dim))  # Batch size x Hidden size
        c = jnp.zeros((X.shape[0], hidden_dim))  # Batch size x Hidden size
        for t in range(seq_len):
            x_t = X[:, t, :]
            i_t = jax.nn.sigmoid(jnp.dot(x_t, W_i) + jnp.dot(h, U_i) + b_i)
            f_t = jax.nn.sigmoid(jnp.dot(x_t, W_f) + jnp.dot(h, U_f) + b_f)
            o_t = jax.nn.sigmoid(jnp.dot(x_t, W_o) + jnp.dot(h, U_o) + b_o)
            g_t = jax.nn.tanh(jnp.dot(x_t, W_c) + jnp.dot(h, U_c) + b_c)
            c = f_t * c + i_t * g_t
            h = o_t * jax.nn.tanh(c)

        # Fully connected layer applied to the last time step
        y_pred = jnp.dot(h, fc_weights) + fc_bias

        # Observation noise
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

        # Observation model
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("mean", dist.Normal(y_pred.squeeze(-1), sigma), obs=y)

    def deep_lstm(self, X, y=None):
        """
        Bayesian LSTM model for time series regression using NumPyro.

        Args:
            X: Input data of shape (N, seq_len, input_size).
            y: Target data of shape (N,).
        """

        num_features = X.shape[2]  # input_size
        seq_len = X.shape[1]
        hidden_dim = self.hidden_dim

        # LSTM 1 weights
        W_i1 = numpyro.sample(
            "W_i1", dist.Normal(0, 1).expand([num_features, hidden_dim]).to_event(2)
        )
        U_i1 = numpyro.sample(
            "U_i1", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        b_i1 = numpyro.sample(
            "b_i1", dist.Normal(0, 1).expand([hidden_dim]).to_event(1)
        )

        W_f1 = numpyro.sample(
            "W_f1", dist.Normal(0, 1).expand([num_features, hidden_dim]).to_event(2)
        )
        U_f1 = numpyro.sample(
            "U_f1", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        b_f1 = numpyro.sample(
            "b_f1", dist.Normal(0, 1).expand([hidden_dim]).to_event(1)
        )

        W_o1 = numpyro.sample(
            "W_o1", dist.Normal(0, 1).expand([num_features, hidden_dim]).to_event(2)
        )
        U_o1 = numpyro.sample(
            "U_o1", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        b_o1 = numpyro.sample(
            "b_o1", dist.Normal(0, 1).expand([hidden_dim]).to_event(1)
        )

        W_c1 = numpyro.sample(
            "W_c1", dist.Normal(0, 1).expand([num_features, hidden_dim]).to_event(2)
        )
        U_c1 = numpyro.sample(
            "U_c1", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        b_c1 = numpyro.sample(
            "b_c1", dist.Normal(0, 1).expand([hidden_dim]).to_event(1)
        )

        # LSTM 2 weights
        W_i2 = numpyro.sample(
            "W_i2", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        U_i2 = numpyro.sample(
            "U_i2", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        b_i2 = numpyro.sample(
            "b_i2", dist.Normal(0, 1).expand([hidden_dim]).to_event(1)
        )

        W_f2 = numpyro.sample(
            "W_f2", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        U_f2 = numpyro.sample(
            "U_f2", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        b_f2 = numpyro.sample(
            "b_f2", dist.Normal(0, 1).expand([hidden_dim]).to_event(1)
        )

        W_o2 = numpyro.sample(
            "W_o2", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        U_o2 = numpyro.sample(
            "U_o2", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        b_o2 = numpyro.sample(
            "b_o2", dist.Normal(0, 1).expand([hidden_dim]).to_event(1)
        )

        W_c2 = numpyro.sample(
            "W_c2", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        U_c2 = numpyro.sample(
            "U_c2", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]).to_event(2)
        )
        b_c2 = numpyro.sample(
            "b_c2", dist.Normal(0, 1).expand([hidden_dim]).to_event(1)
        )

        # Fully connected weights
        fc_weights = numpyro.sample(
            "fc_weights", dist.Normal(0, 1).expand([hidden_dim, 1]).to_event(2)
        )
        fc_bias = numpyro.sample("fc_bias", dist.Normal(0, 1).expand([1]).to_event(1))

        # LSTM 1 forward pass
        h1 = jnp.zeros((X.shape[0], hidden_dim))  # Batch size x Hidden size
        c1 = jnp.zeros((X.shape[0], hidden_dim))  # Batch size x Hidden size
        for t in range(seq_len):
            x_t = X[:, t, :]
            i_t1 = jax.nn.sigmoid(jnp.dot(x_t, W_i1) + jnp.dot(h1, U_i1) + b_i1)
            f_t1 = jax.nn.sigmoid(jnp.dot(x_t, W_f1) + jnp.dot(h1, U_f1) + b_f1)
            o_t1 = jax.nn.sigmoid(jnp.dot(x_t, W_o1) + jnp.dot(h1, U_o1) + b_o1)
            g_t1 = jax.nn.tanh(jnp.dot(x_t, W_c1) + jnp.dot(h1, U_c1) + b_c1)
            c1 = f_t1 * c1 + i_t1 * g_t1
            h1 = o_t1 * jax.nn.tanh(c1)

        # LSTM 2 forward pass
        h2 = jnp.zeros((X.shape[0], hidden_dim))  # Batch size x Hidden size
        c2 = jnp.zeros((X.shape[0], hidden_dim))  # Batch size x Hidden size
        for t in range(seq_len):
            i_t2 = jax.nn.sigmoid(jnp.dot(h1, W_i2) + jnp.dot(h2, U_i2) + b_i2)
            f_t2 = jax.nn.sigmoid(jnp.dot(h1, W_f2) + jnp.dot(h2, U_f2) + b_f2)
            o_t2 = jax.nn.sigmoid(jnp.dot(h1, W_o2) + jnp.dot(h2, U_o2) + b_o2)
            g_t2 = jax.nn.tanh(jnp.dot(h1, W_c2) + jnp.dot(h2, U_c2) + b_c2)
            c2 = f_t2 * c2 + i_t2 * g_t2
            h2 = o_t2 * jax.nn.tanh(c2)

        # Fully connected layer applied to the last time step
        y_pred = jnp.dot(h2, fc_weights) + fc_bias

        # Observation noise
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

        # Observation model
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("obs", dist.Normal(y_pred.squeeze(-1), sigma), obs=y)

    def feedforward(self, X, y=None):
        hidden_dim = self.hidden_dim
        num_features = X.shape[-1]

        # Define weights and biases
        w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([num_features, hidden_dim]))
        b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([hidden_dim]))
        w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([hidden_dim, 1]))
        b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([1]))

        # Forward pass
        h1 = jnp.tanh(jnp.einsum("nfd,dh->nfh", X, w1) + b1)
        y_pred = jnp.einsum("nfh,h1->nf", h1, w2).mean(axis=1) + b2.squeeze()

        # Likelihood
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("mean", dist.Normal(y_pred, sigma), obs=y)

    def deep_feedforward(self, X, y=None):
        hidden_dim = self.hidden_dim
        num_features = X.shape[-1]

        # Define weights and biases for the first hidden layer
        w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([num_features, hidden_dim]))
        b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([hidden_dim]))

        # Define weights and biases for the second hidden layer
        w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([hidden_dim, hidden_dim]))
        b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([hidden_dim]))

        # Define weights and biases for the output layer
        w_out = numpyro.sample("w_out", dist.Normal(0, 1).expand([hidden_dim, 1]))
        b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([1]))

        # Forward pass through the first hidden layer
        h1 = jnp.tanh(jnp.einsum("nfd,dh->nfh", X, w1) + b1)

        # Forward pass through the second hidden layer
        h2 = jnp.tanh(jnp.einsum("nfh,hk->nfh", h1, w2) + b2)

        # Output layer
        y_pred = jnp.einsum("nfh,h1->nf", h2, w_out).mean(axis=1) + b_out.squeeze()

        # Likelihood
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("mean", dist.Normal(y_pred, sigma), obs=y)


if __name__ == "__main__":

    ############### Demo #############
    from time_forecast.bnn.modules.dense.mcmc import DenseMCMC
    import jax.random as jr
    from fake_data import create_synthetic_time_series

    X_train, X_val, y_train, y_val = create_synthetic_time_series()
    print(f"X_val has shape {X_val.shape} and y_val has shape {y_val.shape}")
    model = DenseMCMC(model="lstm", num_warmup=10, num_samples=100, hidden_dim=3)
    model.fit(X_train, y_train, jr.key(0))
    posterior_preds = model.predict(X_val, jr.key(2))
    print(posterior_preds.shape)
    model.visualize(y_train=y_train, y_val=y_val, posterior_preds=posterior_preds)

    from sklearn.metrics import mean_squared_error
    import numpy as np

    MSE = mean_squared_error(np.array(y_val), np.array(posterior_preds.mean(axis=0)))
    print(MSE)
