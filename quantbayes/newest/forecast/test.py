"""
timeseries_forecasting_lstm.py

4) Time-series forecasting using an LSTMCell.
Data: X ∈ ℝ^(N×seq_len×D), y ∈ ℝ^(N,).

Requirements:
- LSTM-based model in Equinox
- Sequence batching & shuffling
- MSE loss
- Evaluate RMSE, MAE
- Plot predictions vs ground truth
- Plot train & val curves
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
# Model Definition
# ----------------------------
class LSTMForecaster(eqx.Module):
    """Wraps an LSTMCell to process a single sequence, then a final linear layer."""
    lstm_cell: eqx.nn.LSTMCell
    linear: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, *, key):
        ckey, fckey = jax.random.split(key)
        self.lstm_cell = eqx.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, use_bias=True, key=ckey)
        # Final linear layer from hidden_size -> 1
        self.linear = eqx.nn.Linear(hidden_size, 1, use_bias=True, key=fckey)

    def __call__(self, x):
        """
        x: shape (seq_len, input_size)
        We'll scan over time steps:
          hidden = (h, c) where each is shape (hidden_size,)
        """
        # initial hidden states
        h = jnp.zeros((self.lstm_cell.hidden_size,))
        c = jnp.zeros((self.lstm_cell.hidden_size,))

        def step(carry, t_inp):
            (h, c) = carry
            (h, c) = self.lstm_cell(t_inp, (h, c))
            return (h, c), h  # store h for example

        (h_final, c_final), hs = jax.lax.scan(step, (h, c), x)
        # final hidden state h_final
        out = self.linear(h_final)  # shape (1,)
        return out.squeeze()  # shape ()


# ----------------------------
# Data Handling
# ----------------------------
def prepare_data(key, N=300, seq_len=12, D=1):
    """
    Create time-series: each sample is sin wave or variations, plus noise.
    X.shape = (N, seq_len, D), y.shape=(N,)
    """
    keys = jax.random.split(key, N)
    X_data = []
    y_data = []
    for i in range(N):
        k = keys[i]
        freq = jax.random.uniform(k, (), minval=0.5, maxval=1.5)
        phase = jax.random.uniform(k, (), minval=0, maxval=jnp.pi)
        t = jnp.linspace(0, 2*jnp.pi, seq_len+1)
        wave = jnp.sin(freq * t + phase)
        wave = wave + 0.1*jax.random.normal(k, shape=wave.shape)
        X_data.append(wave[:-1].reshape(seq_len, D))
        y_data.append(wave[-1])
    X_data = jnp.array(X_data)
    y_data = jnp.array(y_data)
    return X_data, y_data


def train_val_split(X, y, val_ratio=0.2, seed=42):
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    split = int(N*(1-val_ratio))
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]


# ----------------------------
# Loss
# ----------------------------
def mse_loss(model, X, y):
    # We'll vmap over N
    def predict_one(x_seq):
        return model(x_seq)
    preds = jax.vmap(predict_one)(X)
    return jnp.mean((preds - y) ** 2)


def rmse_mae(model, X, y):
    def predict_one(x_seq):
        return model(x_seq)
    preds = jax.vmap(predict_one)(X)
    rmse = jnp.sqrt(jnp.mean((preds - y)**2))
    mae = jnp.mean(jnp.abs(preds - y))
    return rmse, mae, preds


# ----------------------------
# Training
# ----------------------------
@eqx.filter_jit
def make_step(model, X, y, opt_state, optimizer):
    loss_value, grads = eqx.filter_value_and_grad(mse_loss)(model, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss_value, model, opt_state


def train_model(model, X_train, y_train, X_val, y_val, lr=1e-3, epochs=100):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        loss_train, model, opt_state = make_step(model, X_train, y_train, opt_state, optimizer)
        loss_val = mse_loss(model, X_val, y_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train MSE: {loss_train:.4f}, Val MSE: {loss_val:.4f}")
    return model, train_losses, val_losses


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(model, X_test, y_test, train_losses, val_losses):
    test_rmse, test_mae, preds = rmse_mae(model, X_test, y_test)
    print(f"Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")

    # Visualize predictions for a single sequence
    idx = 0
    seq = X_test[idx].squeeze()
    pred_val = preds[idx]
    true_val = y_test[idx]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(seq, marker="o", label="Input Sequence")
    plt.axhline(y=true_val, color="blue", linestyle="--", label="True next value")
    plt.axhline(y=pred_val, color="red", linestyle="--", label="Predicted next value")
    plt.title("Forecast Example")
    plt.legend()

    # Loss curves
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    key = jax.random.PRNGKey(3)
    X, y = prepare_data(key, N=300, seq_len=12, D=1)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2)

    model_key = jax.random.split(key, 1)[0]
    model = LSTMForecaster(input_size=1, hidden_size=16, key=model_key)

    model, train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val,
                                                  lr=1e-3, epochs=100)
    evaluate_model(model, X_val, y_val, train_losses, val_losses)


if __name__ == "__main__":
    main()
