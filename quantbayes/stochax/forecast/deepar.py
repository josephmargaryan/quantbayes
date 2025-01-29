import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from typing import Any, Callable, Sequence, Tuple
import optax
import numpy as np


class LSTMCell(nn.Module):
    features: int

    @nn.compact
    def __call__(self, carry: Tuple[jnp.ndarray, jnp.ndarray], inputs: jnp.ndarray) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        c, h = carry
        dense = nn.Dense(features=4 * self.features)
        gates = dense(jnp.concatenate([inputs, h], axis=-1))
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        new_c = f * c + i * g
        new_h = o * jnp.tanh(new_c)
        return (new_c, new_h), new_h

    def initialize_carry(self, rng, batch_size):
        c = jnp.zeros((batch_size, self.features))
        h = jnp.zeros((batch_size, self.features))
        return (c, h)


class DeepAR(nn.Module):
    input_dim: int
    rnn_hidden: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        batch_size, time_steps, _ = x.shape
        carry = [
            LSTMCell(features=self.rnn_hidden).initialize_carry(
                jax.random.PRNGKey(i), batch_size
            )
            for i in range(self.num_layers)
        ]

        # Pass through stacked LSTM layers
        for layer in range(self.num_layers):
            lstm = LSTMCell(features=self.rnn_hidden)
            outputs = []
            for t in range(time_steps):
                carry[layer], out = lstm(carry[layer], x[:, t, :])
                outputs.append(out)
            x = jnp.stack(outputs, axis=1)

        # Project to (mu, sigma)
        proj = nn.Dense(features=2)
        out = proj(x)
        mu = out[:, :, 0]
        sigma = nn.softplus(out[:, :, 1]) + 1e-3  # Ensure sigma > 0

        return mu, sigma

    def compute_loss(self, params, batch):
        x, y = batch
        mu, sigma = self.apply({"params": params}, x)
        loss = 0.5 * jnp.log(2 * jnp.pi * sigma**2) + ((y - mu) ** 2) / (2 * sigma**2)
        return jnp.mean(loss)  # Average loss across batch and sequence



def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    """
    Perform a single training step.

    Args:
        state: Training state containing the model parameters.
        batch: Tuple of (x, y), where x is the input and y is the target.

    Returns:
        Updated training state and the computed loss.
    """
    x, y = batch

    def loss_fn(params):
        return DeepAR(input_dim=1, rnn_hidden=32, num_layers=2).compute_loss(params, (x, y))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss



def generate_synthetic_data(batch_size, time_steps, input_dim=1):
    x = np.random.randn(batch_size, time_steps, input_dim).astype(np.float32)
    y = x.squeeze(-1) + 0.1 * np.random.randn(batch_size, time_steps).astype(np.float32)
    return x, y


def train_model():
    rng = jax.random.PRNGKey(0)
    model = DeepAR(input_dim=1, rnn_hidden=32, num_layers=2)
    state = create_train_state(rng, model, learning_rate=1e-3, input_shape=(16, 30, 1))

    B, T = 16, 30
    x, y = generate_synthetic_data(B, T)
    x = jnp.array(x)
    y = jnp.array(y)

    epochs = 100
    for epoch in range(epochs):
        state, loss = train_step(state, (x, y))
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss={loss:.4f}")



if __name__ == "__main__":
    train_model()
