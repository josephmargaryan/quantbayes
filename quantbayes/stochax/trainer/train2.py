import math
import os
import time
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from equinox import filter_jit, filter_grad, apply_updates
from equinox import tree_serialise_leaves, tree_deserialise_leaves


###############################################################################
# Helper: A simple Python-based data loader
###############################################################################
def jax_data_loader(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    batch_size: int,
    shuffle: bool = True,
    rng: Optional[jax.random.PRNGKey] = None,
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Yield batches (X_batch, Y_batch) of size batch_size."""
    assert X.shape[0] == Y.shape[0], "X and Y must have the same first dimension"
    N = X.shape[0]
    if shuffle:
        if rng is None:
            raise ValueError("Please provide a PRNGKey if shuffle=True.")
        perm = jax.random.permutation(rng, jnp.arange(N))
        X = X[perm]
        Y = Y[perm]
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        yield X[start_idx:end_idx], Y[start_idx:end_idx]


###############################################################################
# A simple BatchNorm that works on full batches without needing a named axis.
###############################################################################
class SimpleBatchNorm(eqx.Module):
    weight: Optional[jnp.ndarray]
    bias: Optional[jnp.ndarray]
    eps: float = 1e-5
    momentum: float = 0.99
    running_mean: jnp.ndarray
    running_var: jnp.ndarray
    channelwise_affine: bool = True

    def __init__(self, input_size: int, channelwise_affine: bool = True, eps: float = 1e-5, momentum: float = 0.99, dtype=jnp.float32):
        self.eps = eps
        self.momentum = momentum
        self.channelwise_affine = channelwise_affine
        if channelwise_affine:
            self.weight = jnp.ones((input_size,), dtype=dtype)
            self.bias = jnp.zeros((input_size,), dtype=dtype)
        else:
            self.weight = None
            self.bias = None
        self.running_mean = jnp.zeros((input_size,), dtype=dtype)
        self.running_var = jnp.ones((input_size,), dtype=dtype)

    def __call__(self, x: jnp.ndarray, inference: bool) -> Tuple[jnp.ndarray, "SimpleBatchNorm"]:
        """
        Args:
            x: A batched input with shape (batch_size, input_size) (for simplicity).
            inference: If True, use running statistics; otherwise, update them.
        Returns:
            A tuple of (normalized x, updated SimpleBatchNorm instance).
        """
        if inference:
            mean, var = self.running_mean, self.running_var
            new_bn = self  # No updates in inference mode.
        else:
            mean = jnp.mean(x, axis=0)
            var = jnp.var(x, axis=0)
            new_running_mean = (1 - self.momentum) * mean + self.momentum * self.running_mean
            new_running_var = (1 - self.momentum) * var + self.momentum * self.running_var
            # Use eqx.replace to create a new instance with updated running statistics.
            new_bn = eqx.replace(self, running_mean=new_running_mean, running_var=new_running_var)
        out = (x - mean) / jnp.sqrt(var + self.eps)
        if self.channelwise_affine:
            out = out * self.weight + self.bias
        return out, new_bn


###############################################################################
# A universal trainer that expects batched inputs.
###############################################################################
class EquinoxTrainer:
    """
    Trainer for Equinox models written to accept batched inputs.
    """

    def __init__(
        self,
        model: eqx.Module,
        state: Any,
        loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        optimizer: optax.GradientTransformation,
        rng: jax.random.PRNGKey,
        use_jit: bool = True,
    ):
        self.model = model
        self.state = state  # Not used in our simple model since BN is inside the model.
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        self.rng = rng
        self.use_jit = use_jit

        def train_step(model, state, opt_state, x_batch, y_batch, rng):
            def _loss_fn(m, s, x, y):
                # Model expects a full batch.
                pred_y, new_model = m(x, s, key=rng, inference=False)
                return self.loss_fn(pred_y, y), new_model

            (loss_val, new_model), grads = filter_grad(_loss_fn, has_aux=True)(
                model, state, x_batch, y_batch
            )
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_model = apply_updates(new_model, updates)
            return new_model, state, new_opt_state, loss_val

        if use_jit:
            self._train_step = eqx.filter_jit(train_step)
        else:
            self._train_step = train_step

    def fit(
        self,
        X_train: jnp.ndarray,
        Y_train: jnp.ndarray,
        batch_size: int,
        num_epochs: int,
        shuffle: bool = True,
    ) -> None:
        t0 = time.time()
        for epoch in range(num_epochs):
            self.rng, shuffle_key = jax.random.split(self.rng)
            loader = jax_data_loader(X_train, Y_train, batch_size, shuffle, shuffle_key)
            epoch_losses = []
            for x_batch, y_batch in loader:
                self.rng, subkey = jax.random.split(self.rng)
                self.model, self.state, self.opt_state, loss_val = self._train_step(
                    self.model, self.state, self.opt_state, x_batch, y_batch, subkey
                )
                epoch_losses.append(loss_val)
            mean_loss = jnp.mean(jnp.stack(epoch_losses))
            print(
                f"[Epoch {epoch+1:03d}/{num_epochs}] Loss={mean_loss.item():.4f} "
                f"(elapsed: {time.time()-t0:.1f}s)"
            )

    def evaluate(self, X_val: jnp.ndarray, Y_val: jnp.ndarray, batch_size: int) -> float:
        inference_model = eqx.nn.inference_mode(self.model)
        inference_model = eqx.Partial(inference_model, state=self.state)
        all_losses = []
        for x_batch, y_batch in jax_data_loader(X_val, Y_val, batch_size, shuffle=False):
            pred_y, _ = inference_model(x_batch)
            loss_val = self.loss_fn(pred_y, y_batch)
            all_losses.append(loss_val)
        return float(jnp.mean(jnp.stack(all_losses)))

    def predict(
        self,
        X_test: jnp.ndarray,
        batch_size: int,
        return_np: bool = True,
    ) -> Union[jnp.ndarray, Any]:
        inference_model = eqx.nn.inference_mode(self.model)
        inference_model = eqx.Partial(inference_model, state=self.state)
        preds_collected = []
        for x_batch, _ in jax_data_loader(X_test, X_test, batch_size, shuffle=False):
            pred_y, _ = inference_model(x_batch)
            preds_collected.append(pred_y)
        preds = jnp.concatenate(preds_collected, axis=0)
        if return_np:
            return preds.to_py()
        else:
            return preds

    def save(self, path: str) -> None:
        packed = (self.model, self.state, self.opt_state)
        with open(path, "wb") as f:
            tree_serialise_leaves(packed, f)

    @classmethod
    def load(cls, path: str, loss_fn=None, optimizer=None, rng=None, use_jit=True):
        with open(path, "rb") as f:
            model, state, opt_state = tree_deserialise_leaves(f)
        if loss_fn is None:
            raise ValueError("Must provide `loss_fn` to load a trainer.")
        if optimizer is None:
            optimizer = optax.adam(1e-3)
        if rng is None:
            rng = jax.random.PRNGKey(0)
        trainer = cls(
            model=model,
            state=state,
            loss_fn=loss_fn,
            optimizer=optimizer,
            rng=rng,
            use_jit=use_jit,
        )
        trainer.opt_state = opt_state
        return trainer


###############################################################################
# Model definition: TinyMLP using our SimpleBatchNorm.
###############################################################################
class TinyMLP(eqx.Module):
    bn: SimpleBatchNorm
    linear1: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear

    def __init__(self, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        input_dim = 4
        hidden_dim = 16
        output_dim = 3

        # Use our custom BatchNorm that works on full batches.
        self.bn = SimpleBatchNorm(hidden_dim, channelwise_affine=True)
        self.linear1 = eqx.nn.Linear(in_features=input_dim, out_features=hidden_dim, key=k1)
        self.dropout = eqx.nn.Dropout(p=0.2, inference=False)
        self.linear2 = eqx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, key=k2)
        self.linear3 = eqx.nn.Linear(in_features=hidden_dim, out_features=output_dim, key=k3)

    def __call__(self, x: jnp.ndarray, state: Any, *, key, inference: bool):
        # x is batched: shape (batch_size, 4)
        x = jnp.dot(x, self.linear1.weight.T) + self.linear1.bias
        x = self.dropout(x, key=key, inference=inference)
        x = jax.nn.relu(jnp.dot(x, self.linear2.weight.T) + self.linear2.bias)
        # Apply our custom BatchNorm.
        x, new_bn = self.bn(x, inference=inference)
        x = jax.nn.relu(x)
        x = jnp.dot(x, self.linear3.weight.T) + self.linear3.bias
        # In this example, we treat the updated BN as our new state.
        new_state = new_bn
        return x, new_state


###############################################################################
# Data creation and loss function.
###############################################################################
def create_synthetic_data(num_samples=1000):
    rng = jax.random.PRNGKey(42)
    X = jax.random.normal(rng, (num_samples, 4))
    sums = jnp.sum(X, axis=1)
    Y = jnp.where(sums > 2, 2, jnp.where(sums > 0, 1, 0))
    return X, Y

def multiclass_loss_fn(pred_logits, true_labels):
    return optax.softmax_cross_entropy_with_integer_labels(pred_logits, true_labels).mean()


###############################################################################
# Main test: training, evaluation, prediction, saving, and loading.
###############################################################################
def main():
    # Create synthetic data.
    X, Y = create_synthetic_data(num_samples=5000)

    # Build the model and its initial state.
    init_key = jax.random.PRNGKey(0)
    model_init_fn = eqx.nn.make_with_state(TinyMLP)
    model, state = model_init_fn(init_key)

    # Build the trainer.
    trainer = EquinoxTrainer(
        model=model,
        state=state,
        loss_fn=multiclass_loss_fn,
        optimizer=optax.adam(1e-3),
        rng=jax.random.PRNGKey(123),
    )

    # Train for 5 epochs.
    trainer.fit(X, Y, batch_size=64, num_epochs=5)

    # Evaluate the model.
    val_loss = trainer.evaluate(X, Y, batch_size=64)
    print("Validation Loss:", val_loss)

    # Predict on the first 10 samples.
    preds = trainer.predict(X[:10], batch_size=5)
    pred_labels = jnp.argmax(preds, axis=1)
    print("Predicted labels on first 10:", pred_labels)

    # Save the model.
    trainer.save("model_checkpoint.eqx")

    # Load the model and evaluate again.
    new_trainer = EquinoxTrainer.load(
        "model_checkpoint.eqx",
        loss_fn=multiclass_loss_fn,
        optimizer=optax.adam(1e-3),
        rng=jax.random.PRNGKey(999),
        use_jit=True,
    )
    val_loss_loaded = new_trainer.evaluate(X, Y, batch_size=64)
    print("Val Loss (loaded model):", val_loss_loaded)

if __name__ == "__main__":
    main()
