from time_forecast.bnn.core.base_inference import BaseInference
from time_forecast.bnn.core.base_task import BaseTask
from time_forecast.bnn.utils.fft_module import fft_matmul
from typing import Callable
import numpyro.distributions as dist
import numpyro
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from numpyro.optim import Adam
import jax.random as jr
import jax.numpy as jnp
import jax


class FFT_SVI(BaseTask, BaseInference):
    def __init__(
        self,
        model: str = "lstm",
        num_steps=500,
        hidden_dim: int = 10,
        track_loss: bool = False,
    ):
        """
        Initialize DenseMCMC with model type and hyperparameters.
        """
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.track_loss = track_loss
        self.svi = None
        self.params = None
        self.loss = None

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
        Perform SVI inference for the Bayesian regression model.
        """
        model = self.get_default_model()
        guide = autoguide.AutoNormal(model)
        optimizer = Adam(0.01)

        self.svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        svi_state = self.svi.init(rng_key, X_train, y_train)

        loss_progression = [] if self.track_loss else None

        for step in range(self.num_steps):
            svi_state, loss = self.svi.update(svi_state, X_train, y_train)
            if self.track_loss:
                loss_progression.append(loss)
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss:.4f}")

        self.params = self.svi.get_params(svi_state)
        if self.track_loss:
            self.loss = loss_progression

    def retrieve_results(self) -> dict:
        """
        Retrieve inference results, optionally including losses.
        """
        if not self.params:
            raise RuntimeError("No inference results available. Fit the model first.")
        if self.track_loss:
            return {"svi": self.svi, "params": self.params, "loss": self.loss}
        return {"svi": self.svi, "params": self.params}

    def fit(self, X_train, y_train, rng_key):
        """
        Fit the model using MCMC.
        """
        self.bayesian_inference(X_train, y_train, rng_key)
        self.fitted = True

    def predict(self, X_test, rng_key, num_samples=100):
        """
        Predict regression values using the posterior.
        """
        if not self.fitted or self.svi is None:
            raise RuntimeError("Model must be fitted before making predictions.")
        predictive = Predictive(
            self.svi.model,
            guide=self.svi.guide,
            params=self.params,
            num_samples=num_samples,
        )
        rng_key = jr.PRNGKey(1)
        pred_samples = predictive(rng_key, X=X_test)
        return pred_samples["mean"]

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
        Bayesian LSTM model using FFT-based circulant matrix multiplications with one hidden layer.

        Args:
            X: Input data of shape (N, seq_len, input_size).
            y: Target data of shape (N,).
        """
        num_features = X.shape[2]  # input_size
        seq_len = X.shape[1]
        hidden_size = self.hidden_dim

        # LSTM parameters
        W_i_row = numpyro.sample("W_i_row", dist.Normal(0, 1).expand([num_features]))
        U_i_row = numpyro.sample("U_i_row", dist.Normal(0, 1).expand([hidden_size]))
        b_i = numpyro.sample("b_i", dist.Normal(0, 1).expand([hidden_size]))

        W_f_row = numpyro.sample("W_f_row", dist.Normal(0, 1).expand([num_features]))
        U_f_row = numpyro.sample("U_f_row", dist.Normal(0, 1).expand([hidden_size]))
        b_f = numpyro.sample("b_f", dist.Normal(0, 1).expand([hidden_size]))

        W_o_row = numpyro.sample("W_o_row", dist.Normal(0, 1).expand([num_features]))
        U_o_row = numpyro.sample("U_o_row", dist.Normal(0, 1).expand([hidden_size]))
        b_o = numpyro.sample("b_o", dist.Normal(0, 1).expand([hidden_size]))

        W_c_row = numpyro.sample("W_c_row", dist.Normal(0, 1).expand([num_features]))
        U_c_row = numpyro.sample("U_c_row", dist.Normal(0, 1).expand([hidden_size]))
        b_c = numpyro.sample("b_c", dist.Normal(0, 1).expand([hidden_size]))

        # Fully connected layer weights
        fc_weights = numpyro.sample(
            "fc_weights", dist.Normal(0, 1).expand([hidden_size, 1])
        )
        fc_bias = numpyro.sample("fc_bias", dist.Normal(0, 1))

        # LSTM forward pass
        h = jnp.zeros((X.shape[0], hidden_size))  # Batch size x Hidden size
        c = jnp.zeros((X.shape[0], hidden_size))  # Batch size x Hidden size
        for t in range(seq_len):
            x_t = X[:, t, :]
            i_t = jax.nn.sigmoid(
                fft_matmul(W_i_row, x_t) + fft_matmul(U_i_row, h) + b_i
            )
            f_t = jax.nn.sigmoid(
                fft_matmul(W_f_row, x_t) + fft_matmul(U_f_row, h) + b_f
            )
            o_t = jax.nn.sigmoid(
                fft_matmul(W_o_row, x_t) + fft_matmul(U_o_row, h) + b_o
            )
            g_t = jax.nn.tanh(fft_matmul(W_c_row, x_t) + fft_matmul(U_c_row, h) + b_c)
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
        Bayesian LSTM model using FFT-based circulant matrix multiplications.

        Args:
            X: Input data of shape (N, seq_len, input_size).
            y: Target data of shape (N,).
        """
        num_features = X.shape[2]  # input_size
        seq_len = X.shape[1]
        hidden_dim = self.hidden_dim

        # LSTM 1 parameters
        W_i1_row = numpyro.sample("W_i1_row", dist.Normal(0, 1).expand([num_features]))
        U_i1_row = numpyro.sample("U_i1_row", dist.Normal(0, 1).expand([hidden_dim]))
        b_i1 = numpyro.sample("b_i1", dist.Normal(0, 1).expand([hidden_dim]))

        W_f1_row = numpyro.sample("W_f1_row", dist.Normal(0, 1).expand([num_features]))
        U_f1_row = numpyro.sample("U_f1_row", dist.Normal(0, 1).expand([hidden_dim]))
        b_f1 = numpyro.sample("b_f1", dist.Normal(0, 1).expand([hidden_dim]))

        W_o1_row = numpyro.sample("W_o1_row", dist.Normal(0, 1).expand([num_features]))
        U_o1_row = numpyro.sample("U_o1_row", dist.Normal(0, 1).expand([hidden_dim]))
        b_o1 = numpyro.sample("b_o1", dist.Normal(0, 1).expand([hidden_dim]))

        W_c1_row = numpyro.sample("W_c1_row", dist.Normal(0, 1).expand([num_features]))
        U_c1_row = numpyro.sample("U_c1_row", dist.Normal(0, 1).expand([hidden_dim]))
        b_c1 = numpyro.sample("b_c1", dist.Normal(0, 1).expand([hidden_dim]))

        # LSTM 2 parameters (similar to LSTM 1)
        W_i2_row = numpyro.sample("W_i2_row", dist.Normal(0, 1).expand([hidden_dim]))
        U_i2_row = numpyro.sample("U_i2_row", dist.Normal(0, 1).expand([hidden_dim]))
        b_i2 = numpyro.sample("b_i2", dist.Normal(0, 1).expand([hidden_dim]))

        W_f2_row = numpyro.sample("W_f2_row", dist.Normal(0, 1).expand([hidden_dim]))
        U_f2_row = numpyro.sample("U_f2_row", dist.Normal(0, 1).expand([hidden_dim]))
        b_f2 = numpyro.sample("b_f2", dist.Normal(0, 1).expand([hidden_dim]))

        W_o2_row = numpyro.sample("W_o2_row", dist.Normal(0, 1).expand([hidden_dim]))
        U_o2_row = numpyro.sample("U_o2_row", dist.Normal(0, 1).expand([hidden_dim]))
        b_o2 = numpyro.sample("b_o2", dist.Normal(0, 1).expand([hidden_dim]))

        W_c2_row = numpyro.sample("W_c2_row", dist.Normal(0, 1).expand([hidden_dim]))
        U_c2_row = numpyro.sample("U_c2_row", dist.Normal(0, 1).expand([hidden_dim]))
        b_c2 = numpyro.sample("b_c2", dist.Normal(0, 1).expand([hidden_dim]))

        # Fully connected layer weights
        fc_weights = numpyro.sample(
            "fc_weights", dist.Normal(0, 1).expand([hidden_dim, 1])
        )
        fc_bias = numpyro.sample("fc_bias", dist.Normal(0, 1))

        # LSTM 1 forward pass
        h1 = jnp.zeros((X.shape[0], hidden_dim))  # Batch size x Hidden size
        c1 = jnp.zeros((X.shape[0], hidden_dim))  # Batch size x Hidden size
        for t in range(seq_len):
            x_t = X[:, t, :]
            i_t1 = jax.nn.sigmoid(
                fft_matmul(W_i1_row, x_t) + fft_matmul(U_i1_row, h1) + b_i1
            )
            f_t1 = jax.nn.sigmoid(
                fft_matmul(W_f1_row, x_t) + fft_matmul(U_f1_row, h1) + b_f1
            )
            o_t1 = jax.nn.sigmoid(
                fft_matmul(W_o1_row, x_t) + fft_matmul(U_o1_row, h1) + b_o1
            )
            g_t1 = jax.nn.tanh(
                fft_matmul(W_c1_row, x_t) + fft_matmul(U_c1_row, h1) + b_c1
            )
            c1 = f_t1 * c1 + i_t1 * g_t1
            h1 = o_t1 * jax.nn.tanh(c1)

        # LSTM 2 forward pass
        h2 = jnp.zeros((X.shape[0], hidden_dim))  # Batch size x Hidden size
        c2 = jnp.zeros((X.shape[0], hidden_dim))  # Batch size x Hidden size
        for t in range(seq_len):
            i_t2 = jax.nn.sigmoid(
                fft_matmul(W_i2_row, h1) + fft_matmul(U_i2_row, h2) + b_i2
            )
            f_t2 = jax.nn.sigmoid(
                fft_matmul(W_f2_row, h1) + fft_matmul(U_f2_row, h2) + b_f2
            )
            o_t2 = jax.nn.sigmoid(
                fft_matmul(W_o2_row, h1) + fft_matmul(U_o2_row, h2) + b_o2
            )
            g_t2 = jax.nn.tanh(
                fft_matmul(W_c2_row, h1) + fft_matmul(U_c2_row, h2) + b_c2
            )
            c2 = f_t2 * c2 + i_t2 * g_t2
            h2 = o_t2 * jax.nn.tanh(c2)

        # Fully connected layer applied to the last time step
        y_pred = jnp.dot(h2, fc_weights) + fc_bias

        # Observation noise
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

        # Observation model
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("mean", dist.Normal(y_pred.squeeze(-1), sigma), obs=y)

    def feedforward(self, X, y=None):
        """
        Bayesian model using FFT-based circulant matrix multiplication.
        """
        hidden_dim = self.hidden_dim
        num_features = X.shape[-1]

        # Circulant layer parameters
        w1_first_row = numpyro.sample(
            "w1_first_row", dist.Normal(0, 1).expand([num_features])
        )
        b1_circulant = numpyro.sample(
            "b1_circulant", dist.Normal(0, 1).expand([hidden_dim])
        )

        # Circulant multiplication for the first layer
        h1_circulant = fft_matmul(w1_first_row, X)

        # Project into hidden dimensions
        w_hidden_proj = numpyro.sample(
            "w_hidden_proj", dist.Normal(0, 1).expand([num_features, hidden_dim])
        )
        h1 = jnp.einsum("nfd,dh->nfh", h1_circulant, w_hidden_proj) + b1_circulant
        h1 = jnp.tanh(h1)

        # Dense output layer parameters
        w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([hidden_dim, 1]))
        b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([1]))

        # Final prediction
        y_pred = jnp.einsum("nfh,h1->nf", h1, w2).mean(axis=1) + b2.squeeze()

        # Likelihood
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("mean", dist.Normal(y_pred, sigma), obs=y)

    def deep_feedforward(self, X, y=None):
        """
        Deep Bayesian model using FFT-based circulant matrix multiplication with two hidden layers.
        """
        hidden_dim = self.hidden_dim
        num_features = X.shape[-1]

        # Circulant layer parameters for the first hidden layer
        w1_first_row = numpyro.sample(
            "w1_first_row", dist.Normal(0, 1).expand([num_features])
        )
        b1_circulant = numpyro.sample(
            "b1_circulant", dist.Normal(0, 1).expand([hidden_dim])
        )

        # Circulant multiplication for the first layer
        h1_circulant = fft_matmul(w1_first_row, X)

        # Project into the first hidden layer dimensions
        w_hidden_proj1 = numpyro.sample(
            "w_hidden_proj1", dist.Normal(0, 1).expand([num_features, hidden_dim])
        )
        h1 = jnp.einsum("nfd,dh->nfh", h1_circulant, w_hidden_proj1) + b1_circulant
        h1 = jnp.tanh(h1)

        # Parameters for the second hidden layer
        w2_first_row = numpyro.sample(
            "w2_first_row", dist.Normal(0, 1).expand([hidden_dim])
        )
        b2_circulant = numpyro.sample(
            "b2_circulant", dist.Normal(0, 1).expand([hidden_dim])
        )

        # Circulant multiplication for the second layer
        h2_circulant = fft_matmul(w2_first_row, h1)

        # Project into the second hidden layer dimensions
        w_hidden_proj2 = numpyro.sample(
            "w_hidden_proj2", dist.Normal(0, 1).expand([hidden_dim, hidden_dim])
        )
        h2 = jnp.einsum("nfh,hk->nfh", h2_circulant, w_hidden_proj2) + b2_circulant
        h2 = jnp.tanh(h2)

        # Dense output layer parameters
        w_out = numpyro.sample("w_out", dist.Normal(0, 1).expand([hidden_dim, 1]))
        b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([1]))

        # Final prediction
        y_pred = jnp.einsum("nfh,h1->nf", h2, w_out).mean(axis=1) + b_out.squeeze()

        # Likelihood
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        numpyro.sample("mean", dist.Normal(y_pred, sigma), obs=y)


if __name__ == "__main__":

    ############### Demo #############
    from time_forecast.bnn.modules.fft.svi import FFT_SVI
    import jax.random as jr
    from fake_data import create_synthetic_time_series

    X_train, X_val, y_train, y_val = create_synthetic_time_series()
    model = FFT_SVI(model="deep_feedforward", num_steps=100, hidden_dim=10)

    model.fit(X_train, y_train, jr.key(0))
    posterior_preds = model.predict(X_val, jr.key(2))
    print(posterior_preds.shape)
    model.visualize(y_train=y_train, y_val=y_val, posterior_preds=posterior_preds)

    from sklearn.metrics import mean_squared_error
    import numpy as np

    MSE = mean_squared_error(np.array(y_val), np.array(posterior_preds.mean(axis=0)))
    print(MSE)
