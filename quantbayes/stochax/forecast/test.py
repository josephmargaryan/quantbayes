from quantbayes.fake_data import create_synthetic_time_series
from quantbayes.stochax.forecast.lstm import LSTMModel
from quantbayes.stochax.forecast.mamba import Mamba


import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
from flax.training import train_state
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Callable, Optional


# -------------------------------------------------------------------
# 1. Define Model
# -------------------------------------------------------------------


class LSTMModel(nn.Module):
    """
    A simple LSTM-based model for single-step forecasting in Flax:
      - LSTM input: (batch_size, seq_len, input_dim)
      - We take the last hidden state -> project to (batch_size, 1)
    """

    input_dim: int
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, carry=None, deterministic: bool = True):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            carry: Optional initial carry (tuple of (c, h) for each layer)

        Returns:
            y: Output tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape

        # Instantiate all LSTMCells first
        lstm_cells = [
            nn.LSTMCell(features=self.hidden_dim, name=f"lstm_cell_{layer}")
            for layer in range(self.num_layers)
        ]

        if carry is None:
            # Initialize carry for all layers using the instantiated LSTMCells
            carry = tuple(
                lstm_cell.initialize_carry(
                    jax.random.PRNGKey(layer), (batch_size, self.hidden_dim)
                )
                for layer, lstm_cell in enumerate(lstm_cells)
            )

        # Iterate through each LSTM layer
        for layer, lstm_cell in enumerate(lstm_cells):
            hidden = carry[layer]
            outputs = []
            for t in range(seq_len):
                hidden, out = lstm_cell(hidden, x[:, t, :])
                outputs.append(out)
            x = jnp.stack(outputs, axis=1)  # (batch_size, seq_len, hidden_dim)
            if self.dropout > 0.0:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
            # Update carry with the latest hidden state
            carry = carry[:layer] + (hidden,) + carry[layer + 1 :]

        # Take the last hidden state from the top layer
        last_hidden = carry[-1][1]  # (batch_size, hidden_dim)

        # Final dense layer to produce single scalar
        y = nn.Dense(features=1, name="output_dense")(last_hidden)  # (batch_size, 1)

        return y


# -------------------------------------------------------------------
# 2. Train State Creation
# -------------------------------------------------------------------
def create_train_state(
    rng: jax.random.PRNGKey,
    model: nn.Module,
    learning_rate: float,
    example_input: jnp.ndarray,
) -> train_state.TrainState:
    """
    Initializes model parameters and returns a train state.
    """
    params = model.init(rng, example_input)["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# -------------------------------------------------------------------
# 3. Loss and Single Train Step
# -------------------------------------------------------------------
def mse_loss(params, apply_fn, x, y, apply_fn_kwargs=None):
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}
    predictions = apply_fn({"params": params}, x, **apply_fn_kwargs)
    return jnp.mean((predictions - y) ** 2)


@jax.jit
def train_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    apply_fn_kwargs: Dict[str, Any],
) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """
    Performs a single training step:
      - Forward pass to compute loss
      - Compute gradients
      - Update parameters
    """
    loss, grads = jax.value_and_grad(mse_loss)(
        state.params, state.apply_fn, x, y, apply_fn_kwargs
    )
    state = state.apply_gradients(grads=grads)
    return state, loss


# -------------------------------------------------------------------
# 4. Mini-batch Data Generator
# -------------------------------------------------------------------
def data_generator(
    rng: jax.random.PRNGKey,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
):
    """
    Yields mini-batches of (X, y).
    `X` shape: (N, seq_len, input_dim)
    `y` shape: (N, 1) or (N,) for time-series forecasting
    """
    num_samples = X.shape[0]
    if shuffle:
        indices = jax.random.permutation(rng, num_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
    else:
        X_shuffled = X
        y_shuffled = y

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield X_shuffled[start:end], y_shuffled[start:end]


# -------------------------------------------------------------------
# 5. Training Function
# -------------------------------------------------------------------
def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    rng: jax.random.PRNGKey,
    apply_fn_kwargs_train: Dict[str, Any],
    apply_fn_kwargs_val: Optional[Dict[str, Any]] = None,
) -> Tuple[train_state.TrainState, list, list]:
    """
    Trains the given `model` on (X_train, y_train). Also evaluates on
    (X_val, y_val) after each epoch to compute val_loss.

    Args:
        model: Flax module.
        X_train, y_train: Training data
        X_val, y_val: Validation data
        num_epochs: number of training epochs
        batch_size: mini-batch size
        learning_rate: learning rate for optimizer
        rng: JAX random key
        apply_fn_kwargs_train: dict of extra args to pass into model.apply during training
        apply_fn_kwargs_val: dict of extra args to pass into model.apply during validation
                             (e.g. `{'deterministic': True}` or different `rngs`)

    Returns:
        state: the final trained TrainState
        train_losses: list of per-epoch training losses
        val_losses: list of per-epoch validation losses
    """
    if apply_fn_kwargs_val is None:
        apply_fn_kwargs_val = {}

    # Example input for shape initialization
    seq_len = X_train.shape[1]
    input_dim = X_train.shape[2]
    example_input = jnp.ones((1, seq_len, input_dim), dtype=jnp.float32)

    # Create train state
    state = create_train_state(rng, model, learning_rate, example_input)

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Shuffle the data each epoch by splitting rng
        rng, data_rng = jax.random.split(rng)
        epoch_losses = []

        for batch_X, batch_y in data_generator(data_rng, X_train, y_train, batch_size):
            # Ensure jnp.float32
            batch_X = jnp.array(batch_X, dtype=jnp.float32)
            batch_y = jnp.array(batch_y, dtype=jnp.float32)

            # Single training step
            state, loss = train_step(state, batch_X, batch_y, apply_fn_kwargs_train)
            epoch_losses.append(loss)

        mean_epoch_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(mean_epoch_loss.item())

        # Validation
        val_loss = evaluate_loss(
            state.params,
            state.apply_fn,
            X_val,
            y_val,
            batch_size,
            rng,  # can reuse or new rng
            apply_fn_kwargs_val,
        )
        val_losses.append(val_loss)

        if epoch % max(1, (num_epochs // 10)) == 0:
            print(
                f"Epoch {epoch}/{num_epochs}: train_loss={mean_epoch_loss:.4f}, val_loss={val_loss:.4f}"
            )

    # Plot the train/val losses
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return state, train_losses, val_losses


def evaluate_loss(
    params: flax.core.frozen_dict.FrozenDict,
    apply_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: jax.random.PRNGKey,
    apply_fn_kwargs: Dict[str, Any],
) -> float:
    """
    Compute the MSE loss over the entire dataset (X, y) in mini-batches.
    """
    losses = []
    for batch_X, batch_y in data_generator(rng, X, y, batch_size, shuffle=False):
        batch_X = jnp.array(batch_X, dtype=jnp.float32)
        batch_y = jnp.array(batch_y, dtype=jnp.float32)
        loss = mse_loss(params, apply_fn, batch_X, batch_y, apply_fn_kwargs)
        losses.append(loss)
    return float(jnp.mean(jnp.array(losses)))


# -------------------------------------------------------------------
# 6. Evaluation Function with MC Sampling
# -------------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    params: flax.core.frozen_dict.FrozenDict,
    X_val: np.ndarray,
    y_val: np.ndarray,
    rng: jax.random.PRNGKey,
    num_samples: int = 100,
    apply_fn_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate the model with Monte Carlo sampling to estimate predictive uncertainty.

    Args:
        model: Flax module (not strictly needed if only applying `params`, but
               sometimes we keep it for clarity).
        params: trained parameters
        X_val: shape (N, seq_len, input_dim)
        y_val: shape (N, 1) or (N,) for time series
        rng: JAX random key
        num_samples: how many stochastic forward passes to do
        apply_fn_kwargs: extra arguments for model.apply (e.g. rngs, deterministic, etc.)

    Returns:
        A dictionary containing:
          - "mean_predictions": (N,) average prediction across MC samples
          - "std_predictions": (N,) standard deviation across MC samples
          - "mse": float, mean squared error w.r.t y_val
          - "all_samples": (num_samples, N) array of all predictions for deeper analysis
    """
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}

    # We do multiple stochastic forward passes
    predictions_list = []
    for i in range(num_samples):
        # For each sample, we can split rng to get a new dropout key, etc.
        rng, subkey = jax.random.split(rng)
        # If your model uses dropout, you might do something like:
        # apply_fn_kwargs['rngs'] = {'dropout': subkey}

        preds = model.apply({"params": params}, X_val, **apply_fn_kwargs)
        predictions_list.append(preds)

    # predictions_list will be (num_samples, N, 1)
    predictions_stacked = jnp.stack(predictions_list, axis=0)  # (num_samples, N, 1)
    # Squeeze final dim
    predictions_stacked = predictions_stacked.squeeze(axis=-1)  # (num_samples, N)

    # Compute mean and std
    mean_predictions = jnp.mean(predictions_stacked, axis=0)  # (N,)
    std_predictions = jnp.std(predictions_stacked, axis=0)  # (N,)

    # Compute MSE
    y_val = jnp.array(y_val).squeeze(axis=-1)  # ensure shape (N,)
    mse_val = jnp.mean((mean_predictions - y_val) ** 2)

    # Time-series plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_val, label="True", color="black")
    plt.plot(mean_predictions, label="Mean Prediction", color="blue")

    # For a 95% conf interval, ~ +/- 1.96 * std -- but let's just plot +/- std for example
    lower = mean_predictions - std_predictions
    upper = mean_predictions + std_predictions
    plt.fill_between(
        x=jnp.arange(len(mean_predictions)),
        y1=lower,
        y2=upper,
        alpha=0.2,
        color="blue",
        label="Mean ± 1 std",
    )

    plt.title("Time Series Predictions (Mean + Uncertainty)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "mean_predictions": mean_predictions,
        "std_predictions": std_predictions,
        "mse": float(mse_val),
        "all_samples": predictions_stacked,
    }


# -------------------------------------------------------------------
# 7. Example Usage / Test
#    We'll create a mock time series, train the model, and evaluate.
# -------------------------------------------------------------------
if __name__ == "__main__":
    # --- Create Some Synthetic Time Series Data ---
    # For demonstration, let's say we have a simple sine wave + noise
    np.random.seed(42)
    T = 200  # total length
    t = np.arange(T)
    true_signal = np.sin(0.1 * t) + 0.1 * np.random.randn(T)

    # Make (seq_len=10) -> predict next value
    seq_len = 10
    X_data = []
    y_data = []
    for i in range(T - seq_len):
        X_data.append(true_signal[i : i + seq_len])
        y_data.append(true_signal[i + seq_len])
    X_data = np.array(X_data)  # shape: (T-seq_len, seq_len)
    y_data = np.array(y_data)  # shape: (T-seq_len,)

    # Reshape X to (N, seq_len, input_dim=1)
    X_data = X_data[..., np.newaxis]  # (N, seq_len, 1)
    y_data = y_data[..., np.newaxis]  # (N, 1)

    N = X_data.shape[0]
    train_size = int(0.8 * N)
    X_train, X_val = X_data[:train_size], X_data[train_size:]
    y_train, y_val = y_data[:train_size], y_data[train_size:]

    # --- Define Model & Training/Eval Config ---
    # model = SimpleTimeSeriesModel(hidden_dim=32)
    model = LSTMModel(1)
    rng = jax.random.PRNGKey(0)

    # In some models, you might want something like:
    # apply_fn_kwargs_train = {"deterministic": False, "rngs": {"dropout": rng}}
    # apply_fn_kwargs_val   = {"deterministic": True}
    # For this simple MLP, there's no dropout or batchnorm, so we can omit or pass empty dicts:
    apply_fn_kwargs_train = {}
    apply_fn_kwargs_val = {}
    apply_fn_kwargs_train = {"deterministic": False, "rngs": {"dropout": rng}}
    apply_fn_kwargs_val = {"deterministic": True}

    # --- Train the Model ---
    num_epochs = 200
    batch_size = 16
    learning_rate = 1e-3

    print("Starting Training...")
    state, train_losses, val_losses = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rng=rng,
        apply_fn_kwargs_train=apply_fn_kwargs_train,
        apply_fn_kwargs_val=apply_fn_kwargs_val,
    )

    # --- Evaluate the Model with MC Sampling ---
    # If we had dropout, we'd do multiple forward passes to see uncertainty.
    # Even without dropout, we can run multiple passes. They should be identical in this simple MLP.
    print("Evaluating Model...")
    results = evaluate_model(
        model=model,
        params=state.params,
        X_val=jnp.array(X_val, dtype=jnp.float32),
        y_val=jnp.array(y_val, dtype=jnp.float32),
        rng=rng,
        num_samples=50,
        apply_fn_kwargs={},
    )

    print(f"Val MSE (using mean predictions): {results['mse']:.4f}")
    # results["mean_predictions"] : shape (N_val,)
    # results["std_predictions"]  : shape (N_val,)
    # results["all_samples"]      : shape (num_samples, N_val)
