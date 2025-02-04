import math
import matplotlib.pyplot as plt
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax  # https://github.com/deepmind/optax
from quantbayes.fake_data import generate_regression_data
import pandas as pd 

def dataloader(arrays, batch_size):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

df = generate_regression_data()
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
def get_data():
    return X, y

class MLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    bias: jax.Array

    def __init__(self, in_size, out_size, hidden_size, *, key):
        k1, k2 = jrandom.split(key)
        self.linear1 = eqx.nn.Linear(in_size, hidden_size, key=k1)
        self.linear2 = eqx.nn.Linear(hidden_size, out_size, key=k2)
        self.bias = jnp.zeros(out_size)

    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))  # Hidden layer with ReLU
        x = self.linear2(x) + self.bias
        return jax.nn.sigmoid(x)  # Sigmoid for binary classification
    

def main(
    dataset_size=10000,
    batch_size=32,
    learning_rate=3e-3,
    steps=200,
    hidden_size=16,
    depth=1,
    seed=5678,
):
    data_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 2)
    xs, ys = get_data()

    # Split data into training (80%) and validation (20%) sets
    split_idx = int(0.8 * dataset_size)
    xs_train, ys_train = xs[:split_idx], ys[:split_idx]
    xs_val, ys_val = xs[split_idx:], ys[split_idx:]

    train_data = dataloader((xs_train, ys_train), batch_size)
    val_data = dataloader((xs_val, ys_val), batch_size)

    model = MLP(in_size=xs.shape[1], out_size=1, hidden_size=16, key=model_key)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        pred_y = jnp.clip(pred_y, 1e-7, 1 - 1e-7)  # Avoid log(0) issues
        return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))

    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)

    train_losses = []
    val_losses = []
    val_steps = []

    for step in range(steps):
        x_train, y_train = next(train_data)
        loss, model, opt_state = make_step(model, x_train, y_train, opt_state)
        train_losses.append(loss.item())

        # Compute validation loss every 10 steps
        if step % 10 == 0:
            x_val, y_val = next(val_data)
            val_loss = compute_loss(model, x_val, y_val).item()
            val_losses.append(val_loss)
            val_steps.append(step)
            print(f"step={step}, train_loss={loss.item()}, val_loss={val_loss}")

    # Plot train vs validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(steps), train_losses, label="Train Loss", alpha=0.7)
    plt.plot(val_steps, val_losses, label="Validation Loss", linestyle="--", marker='o', alpha=0.9)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Curve")
    plt.legend()
    plt.show()

    # Compute final log loss
    pred_ys = jax.vmap(model)(xs_val)
    from sklearn.metrics import log_loss 
    print(f"Final Log Loss: {log_loss(np.array(ys_val), np.array(pred_ys))}")

if __name__ == "__main__":
    main()