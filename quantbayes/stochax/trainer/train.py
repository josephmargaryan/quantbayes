import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

__all__ = ["data_loader", "apply_model", "predict", "train", "predict_unlabeled"]

"""
Example Usage:
batch_size = 1200
train_dataset = list(data_loader(X_train, y_train, batch_size, shuffle=True))
val_dataset   = list(data_loader(X_val, y_val, batch_size, shuffle=False))

# Set up PRNG key and model.
main_key = jr.PRNGKey(42)
model_key, main_key = jr.split(main_key)
model = LogReg(in_features=9, key=model_key)
state = None  # Stateless model.

lr_schedule = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=100,
    decay_rate=0.99,
    staircase=True
)

# Create an Adam optimizer wrapped with the learning rate scheduler.
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=lr_schedule)
)
opt_state = optimizer.init(model)

# Train the model with early stopping (patience=25, up to 1000 epochs).
num_epochs = 1000
model, state, opt_state = train(model, state, optimizer, opt_state,
                                optax.losses.sigmoid_binary_cross_entropy,
                                train_dataset, val_dataset,
                                num_epochs, main_key, patience=25)

# Predict logits on validation data.
val_logits = predict(model, state, val_dataset)
probs = jax.nn.sigmoid(val_logits).flatten()
true_vals = np.concatenate([batch["y"] for batch in val_dataset])
"""


def data_loader(X, y, batch_size, shuffle=True):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, n_samples, batch_size):
        excerpt = indices[start_idx : start_idx + batch_size]
        yield {"x": X[excerpt], "y": y[excerpt]}


def apply_model(model, state, loss_fn, key, x, y):
    """
    Computes loss for a given batch.
    We call the model for each sample via vmap.
    """

    def single_loss(x, y, key):
        logits, new_state = model(x, state, key=key)
        loss = loss_fn(logits, y)
        return loss, new_state

    if x.ndim == 1:
        # Single sample case.
        loss, new_state = single_loss(x, y, key)
    else:
        # Batched input: create a separate key per sample.
        batch_size = x.shape[0]
        keys = jr.split(key, batch_size)
        losses, new_states = jax.vmap(single_loss, in_axes=(0, 0, 0))(x, y, keys)
        loss = jnp.mean(losses)
        # For stateless models, new_states is None; else combine if needed.
        new_state = None
    return loss, new_state


def train_step(model, state, opt_state, loss_fn, batch, key, optimizer):
    # Rename the inner loss_fn to avoid shadowing the outer one
    def _loss_fn(model, state, key):
        loss, new_state = apply_model(
            model, state, loss_fn, key, batch["x"], batch["y"]
        )
        return loss, new_state

    # Compute loss and gradients.
    (loss, new_state), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        model, state, key
    )  # Use _loss_fn here
    updates, opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, opt_state, loss


def predict(model, state, dataset):
    """
    Given a dataset (a list of batches), compute the logits.
    We use vmap over the batch because the model expects single samples.
    """
    logits_list = []
    # Define a function that predicts on a single sample.
    def single_predict(x):
        logit, _ = model(x, state, key=None)
        return logit

    v_single_predict = jax.vmap(single_predict, in_axes=0)
    for batch in dataset:
        batch_logits = v_single_predict(batch["x"])  # shape: (batch_size, 1)
        logits_list.append(batch_logits)
    return jnp.concatenate(logits_list, axis=0)


def compute_validation_loss(model, state, loss_fn, dataset):
    """
    Compute validation log loss for the current model.
    Applies sigmoid to convert logits to probabilities.
    """
    val_logits = predict(model, state, dataset)
    # Convert logits to probabilities.
    probs = jax.nn.sigmoid(val_logits).flatten()
    true_vals = np.concatenate([batch["y"] for batch in dataset])
    return loss_fn(true_vals, np.array(probs))


def train(
    model,
    state,
    optimizer,
    opt_state,
    loss_fn,
    train_dataset,
    val_dataset,
    num_epochs,
    key,
    patience=25,
):
    best_val_loss = float("inf")
    best_model = model
    best_state = state
    best_opt_state = opt_state
    epochs_without_improve = 0
    val_losses = []

    for epoch in range(num_epochs):
        # Train for one epoch.
        pbar = tqdm(train_dataset, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            key, subkey = jr.split(key)
            model, state, opt_state, loss = train_step(
                model, state, opt_state, loss_fn, batch, subkey, optimizer
            )
            pbar.set_postfix(loss=loss.item())
        # Evaluate on validation set.
        val_loss = compute_validation_loss(model, state, val_dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Validation log loss: {val_loss:.4f}")
        # Early stopping.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_state = state
            best_opt_state = opt_state
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print("Early stopping triggered.")
                break
    plt.figure(figsize=(10, 6))
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Validation Loss Over Epochs")
    plt.legend()
    plt.show()
    return best_model, best_state, best_opt_state


def predict_unlabeled(model, state, dataset):
    """
    Given a dataset (a list of batches containing only inputs under key "x"),
    compute the logits.
    """
    logits_list = []

    def single_predict(x):
        logit, _ = model(x, state, key=None)
        return logit

    v_single_predict = jax.vmap(single_predict, in_axes=0)
    for batch in dataset:
        batch_logits = v_single_predict(batch["x"])  # shape: (batch_size, 1)
        logits_list.append(batch_logits)
    return jnp.concatenate(logits_list, axis=0)
