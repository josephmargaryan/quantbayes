import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax


class ForecastingModel:
    """
    Forecasting training wrapper. Expects the model to output real-valued predictions.
    Although ForecastNet may not use BN or Dropout by default, we follow the same pattern:
      - A new random key is generated for each sample.
      - A state is passed (even if unused).
    """

    def __init__(self, lr=1e-3, loss_fn=None):
        """
        Args:
            lr: learning rate.
            loss_fn: custom loss function; if None, defaults to MSE.
                     The loss_fn should accept (preds, Y) as inputs.
        """
        self.lr = lr
        self.loss_fn = loss_fn if loss_fn is not None else self.mse_loss
        self.optimizer = optax.adam(lr)
        self.opt_state = None
        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def mse_loss(preds, Y):
        """Default MSE loss. Expects preds: [batch, ...] and Y: [batch, ...]."""
        return jnp.mean((preds - Y) ** 2)

    def _batch_forward_train(self, model, state, X, key):
        """
        Performs a forward pass on a batch in training mode.
        Splits the provided key into one key per sample.
        """
        # Split key into one per sample (assumes X has shape [B, ...])
        keys = jr.split(key, X.shape[0])

        def single_forward(m, s, x, key):
            out, new_state = m(x, state=s, key=key)
            return out, new_state

        preds, new_state = jax.vmap(
            single_forward, in_axes=(None, None, 0, 0), out_axes=(0, None)
        )(model, state, X, keys)
        return preds, new_state

    def _batch_forward_inference(self, model, state, X, key):
        """
        Performs a forward pass on a batch in inference mode.
        Again, splits the key per sample.
        """
        # Use tree_inference to set the model to inference mode.
        inf_model = eqx.tree_inference(model, value=True)
        keys = jr.split(key, X.shape[0])

        def single_forward(m, s, x, key):
            out, new_state = m(x, state=s, key=key)
            return out, new_state

        preds, new_state = jax.vmap(
            single_forward, in_axes=(None, None, 0, 0), out_axes=(0, None)
        )(inf_model, state, X, keys)
        return preds, new_state

    def _compute_batch_loss(self, model, state, X, Y, key, training=True):
        """
        Computes the loss for an entire batch.
        Uses the appropriate forward function based on training/inference mode.
        """
        if training:
            preds, new_state = self._batch_forward_train(model, state, X, key)
        else:
            preds, new_state = self._batch_forward_inference(model, state, X, key)
        loss_val = self.loss_fn(preds, Y)
        return loss_val, new_state, preds

    @eqx.filter_jit
    def _train_step(self, model, state, X, Y, key):
        """
        A single training step. Wraps the loss computation in a function that
        also returns an updated state.
        """

        def loss_wrapper(m, s):
            loss_val, new_s, _ = self._compute_batch_loss(
                m, s, X, Y, key, training=True
            )
            return loss_val, new_s

        (loss_val, new_state), grads = eqx.filter_value_and_grad(
            loss_wrapper, has_aux=True
        )(model, state)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        return loss_val, new_model, new_opt_state, new_state

    def _eval_step(self, model, state, X, Y, key):
        """
        A single evaluation step.
        """
        loss_val, new_state, _ = self._compute_batch_loss(
            model, state, X, Y, key, training=False
        )
        return loss_val

    def fit(
        self,
        model,
        state,
        X_train,
        Y_train,
        X_val,
        Y_val,
        num_epochs=50,
        patience=10,
        key=jr.PRNGKey(0),
    ):
        """
        :param y_train jax.Array of shape (num_samples, 1)
        :param y_test jax.Array of shape (num_samples, 1)
        """
        # Initialize the optimizer state.
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            key, subkey = jr.split(key)
            loss_val, model, self.opt_state, state = self._train_step(
                model, state, X_train, Y_train, subkey
            )
            self.train_losses.append(float(loss_val))
            key, subkey = jr.split(key)
            val_loss = self._eval_step(model, state, X_val, Y_val, subkey)
            self.val_losses.append(float(val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}: Train Loss = {loss_val:.4f}, Val Loss = {val_loss:.4f}"
                )

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        return model, state

    def predict(self, model, state, X, key=jr.PRNGKey(123)):
        inf_model = eqx.tree_inference(model, value=True)
        keys = jr.split(key, X.shape[0])

        def single_pred(m, x, key):
            out, _ = m(x, state=state, key=key)
            return out

        preds = jax.vmap(single_pred, in_axes=(None, 0, 0))(inf_model, X, keys)
        return preds

    def visualize(self, y_true, y_pred, title="Forecast vs. Ground Truth"):
        """
        Just a simple line plot comparing y_true and y_pred (both [batch, 1]).
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, marker="o", label="Ground Truth")
        plt.plot(y_pred, marker="x", label="Predictions")
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    import jax
    import jax.random as jr
    import jax.numpy as jnp
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    from quantbayes.fake_data import create_synthetic_time_series

    # A simple synthetic time series: shape [N, seq_len, input_dim].
    # For example, each row is a sequence of length 10 with 1 dimension.
    N, seq_len, input_dim = 50, 10, 1
    key = jr.PRNGKey(0)
    X_full = jr.normal(key, (N, seq_len, input_dim))
    # Let's say the target is the mean of each sequence (shape [N, 1]).
    Y_full = jnp.mean(X_full, axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_full, Y_full, test_size=0.2, random_state=42
    )

    X_train, X_test, Y_train, Y_test = create_synthetic_time_series()
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    print(X_train.shape)
    print(Y_train.shape)

    # Example ForecastNet model (no BN or Dropout).
    class ForecastNet(eqx.Module):
        cell: eqx.nn.GRUCell
        fc: eqx.nn.Linear

        def __init__(self, input_dim, hidden_size, key):
            k1, k2 = jr.split(key)
            self.cell = eqx.nn.GRUCell(
                input_size=input_dim, hidden_size=hidden_size, key=k1
            )
            self.fc = eqx.nn.Linear(hidden_size, 1, key=k2)

        def __call__(self, x, state=None, *, key=None):
            # x shape: [seq_len, input_dim]
            init_h = jnp.zeros(self.cell.hidden_size)

            def step(carry, xt):
                new_carry = self.cell(xt, carry)
                return new_carry, None

            final_h, _ = jax.lax.scan(step, init_h, x)
            return self.fc(final_h), state

    # Instantiate a ForecastNet and ForecastingModel from your definitions
    model = ForecastNet(input_dim, hidden_size=62, key=jr.PRNGKey(1))
    state = None  # no BN or Dropout => no separate eqx.nn.State needed

    forecasting_model = ForecastingModel(lr=1e-3)
    # Fit the model
    trained_model, state = forecasting_model.fit(
        model,
        state,
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_epochs=100,
        patience=10,
        key=jr.PRNGKey(999),
    )

    # Predict
    preds = forecasting_model.predict(trained_model, state, X_test)
    # Visualize
    forecasting_model.visualize(Y_test, preds, title="Forecast Demo")

    plt.figure(figsize=(8, 6))
    plt.plot(forecasting_model.train_losses, label="Training Loss")
    plt.plot(forecasting_model.val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    preds = forecasting_model.predict(trained_model, state, X_test)
    print(f"MSE: {mean_squared_error(np.array(Y_test), np.array(preds))}")
