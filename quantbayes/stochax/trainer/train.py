import math
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

# We'll re-export these for convenience, but you can also import them directly:
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
    """
    Simple generator that yields (X_batch, Y_batch) in increments of batch_size.
    By default, it shuffles the data (requires rng if you want reproducible shuffling).

    Args:
        X: shape (N, ...) 
        Y: shape (N, ...) with matching first dimension
        batch_size: int
        shuffle: bool
        rng: optional jax.random.PRNGKey for reproducible shuffle

    Returns:
        Iterator over (X_batch, Y_batch)
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have the same first dimension"
    N = X.shape[0]

    if shuffle:
        if rng is None:
            raise ValueError("Please provide a PRNGKey if shuffle=True.")
        perm = jax.random.permutation(rng, jnp.arange(N))
        X = X[perm]
        Y = Y[perm]

    # Python-based iteration
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        yield X[start_idx:end_idx], Y[start_idx:end_idx]


###############################################################################
# A universal trainer that can handle:
#   - arbitrary Equinox models (possibly with BN, Dropout, etc.)
#   - arbitrary tasks (regression / binary / multi-class) by user-supplied `loss_fn`
#   - stateful layers (by passing around `state`)
#   - inference vs. train toggles
###############################################################################
class EquinoxTrainer:
    """
    A class to encapsulate training, evaluation, and prediction logic
    for Equinox-based models, with support for:
      - Dropout (requiring fresh PRNGKeys per step)
      - BatchNorm or other stateful layers (requiring a `state` PyTree).
      - Different tasks (regression / binary / multiclass) via user-supplied `loss_fn`.
      - Automatic or manual toggling of inference/training mode.

    Basic usage:

        trainer = EquinoxTrainer(
            model=model,       # eqx.Module
            state=state,       # eqx.nn.State or empty container if no stateful layers
            loss_fn=loss_fn,   # e.g. from optax or a custom function
            optimizer=optax.adam(1e-3),
            rng=jax.random.PRNGKey(0),
        )
        trainer.fit(X_train, Y_train, batch_size=32, num_epochs=10)
        val_loss = trainer.evaluate(X_val, Y_val, batch_size=32)
        preds = trainer.predict(X_test, batch_size=32)
        trainer.save("model.eqx")
        # ...
        trainer_loaded = EquinoxTrainer.load("model.eqx")
        test_preds_loaded = trainer_loaded.predict(X_test, batch_size=32)
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
        """
        Args:
            model: An equinox.Module (can contain BN, Dropout, etc.)
            state: The state PyTree if model is stateful (e.g. BatchNorm).
                   Typically created with `eqx.nn.make_with_state(...)`.
            loss_fn: A function (pred_y, true_y) -> scalar that returns the loss
                     for a single batch. E.g. an optax loss or custom.
            optimizer: An optax optimizer (e.g. optax.adam(1e-3)).
            rng: PRNGKey for all random operations (e.g. dropout).
            use_jit: Whether to JIT-compile the training step. Usually want True.
        """
        self.model = model
        self.state = state
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        self.rng = rng
        self.use_jit = use_jit

        # We'll define the training step as a closure:
        def train_step(model, state, opt_state, x, y, rng):
            """
            Single step over one batch. 
            Returns: (model, state, opt_state, loss_val)
            """

            def _loss_fn(m, s, x_, y_):
                # We pass rng for forward pass if using dropout, etc. 
                # Some layers (e.g. eqx.nn.Dropout) accept key=..., 
                # others ignore it. We also pass 's' for stateful layers.
                pred_y, s_out = m(x_, s, key=rng, inference=False)
                return self.loss_fn(pred_y, y_), s_out

            (loss_val, new_state), grads = filter_grad(_loss_fn, has_aux=True)(
                model, state, x, y
            )
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_model = apply_updates(model, updates)
            return new_model, new_state, new_opt_state, loss_val

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
        """
        Train the model for `num_epochs`.

        Args:
            X_train: shape (N, ...) 
            Y_train: shape (N, ...)
            batch_size: int
            num_epochs: number of epochs to train
            shuffle: whether to shuffle data each epoch
        """
        t0 = time.time()
        for epoch in range(num_epochs):
            # new subkey for data shuffling
            self.rng, shuffle_key = jax.random.split(self.rng)
            loader = jax_data_loader(X_train, Y_train, batch_size, shuffle, shuffle_key)

            epoch_losses = []
            for x_batch, y_batch in loader:
                # new subkey for forward pass
                self.rng, subkey = jax.random.split(self.rng)
                self.model, self.state, self.opt_state, loss_val = self._train_step(
                    self.model, self.state, self.opt_state, x_batch, y_batch, subkey
                )
                epoch_losses.append(loss_val)

            mean_loss = jnp.mean(jnp.stack(epoch_losses))
            print(
                f"[Epoch {epoch+1:03d}/{num_epochs}] "
                f"Loss = {mean_loss.item():.4f}  "
                f"(elapsed: {time.time()-t0:.1f}s)"
            )

    def evaluate(
        self,
        X_val: jnp.ndarray,
        Y_val: jnp.ndarray,
        batch_size: int,
    ) -> float:
        """
        Evaluate the model in inference mode (BatchNorm uses running stats, Dropout off).

        Args:
            X_val, Y_val: arrays of shape (N, ...)
            batch_size: int

        Returns:
            Mean loss over entire dataset
        """

        # Use eqx.nn.inference_mode to get a "version" of the model 
        # that has `inference=True` for e.g. BatchNorm/Dropout
        inference_model = eqx.nn.inference_mode(self.model)
        # We also partially-apply the state, so that calling inference_model(x) 
        # returns (pred_y, updated_state) but we ignore that updated_state for eval.
        # Because we just want to use fixed running stats at evaluation time.
        inference_model = eqx.Partial(inference_model, state=self.state)

        # no grads needed, so we can do a normal python loop
        all_losses = []
        for x_batch, y_batch in jax_data_loader(X_val, Y_val, batch_size, shuffle=False):
            pred_y, _ = inference_model(x_batch)  # key=None => no dropout
            loss_val = self.loss_fn(pred_y, y_batch)
            all_losses.append(loss_val)

        return float(jnp.mean(jnp.stack(all_losses)))

    def predict(
        self,
        X_test: jnp.ndarray,
        batch_size: int,
        return_np: bool = True,
    ) -> Union[jnp.ndarray, Any]:
        """
        Predict in inference mode, returning model outputs. 
        For classification tasks, you might do jnp.argmax(...) externally.

        Args:
            X_test: shape (N, ...)
            batch_size: int
            return_np: if True, return numpy array, else JAX array.

        Returns:
            The predictions. Possibly shape (N, ...).
        """
        inference_model = eqx.nn.inference_mode(self.model)
        inference_model = eqx.Partial(inference_model, state=self.state)

        preds_collected = []
        for x_batch, _ in jax_data_loader(X_test, X_test, batch_size, shuffle=False):
            pred_y, _ = inference_model(x_batch)
            preds_collected.append(pred_y)

        preds = jnp.concatenate(preds_collected, axis=0)
        if return_np:
            return preds.to_py()  # or np.array(preds) if you prefer
        else:
            return preds

    def save(self, path: str) -> None:
        """
        Saves model and state to a single file at `path`.
        The recommended extension is e.g. ".eqx" or ".pt" or ".pkl".
        """
        # We can combine everything into one PyTree
        # We only need the model and state. 
        # The optimizer state can also be saved if you want to resume training exactly.
        packed = (self.model, self.state, self.opt_state)
        with open(path, "wb") as f:
            tree_serialise_leaves(packed, f)

    @classmethod
    def load(cls, path: str, loss_fn=None, optimizer=None, rng=None, use_jit=True):
        """
        Loads model, state, and opt_state from `path`.
        This returns a new EquinoxTrainer, but you must supply the same `loss_fn`
        and `optimizer` if you want to continue training from the same point. 
        Alternatively, supply new ones if you only want to do inference, or 
        if you want to fine-tune with a different optimizer.

        Args:
            path: path to file
            loss_fn: same as in constructor
            optimizer: same as in constructor
            rng: same as in constructor
            use_jit: same as in constructor

        Returns:
            EquinoxTrainer instance
        """
        with open(path, "rb") as f:
            model, state, opt_state = tree_deserialise_leaves(f)

        if loss_fn is None:
            raise ValueError("Must provide `loss_fn` to load a trainer.")
        if optimizer is None:
            optimizer = optax.adam(1e-3)  # or user-chosen default
        if rng is None:
            rng = jax.random.PRNGKey(0)

        # create the trainer
        trainer = cls(
            model=model,
            state=state,
            loss_fn=loss_fn,
            optimizer=optimizer,
            rng=rng,
            use_jit=use_jit,
        )
        # Overwrite the loaded opt_state to continue training seamlessly
        trainer.opt_state = opt_state
        return trainer



import jax
import jax.numpy as jnp
import equinox as eqx
import optax

"""from trainer import EquinoxTrainer
from trainer import jax_data_loader  # optional usage
from trainer import tree_serialise_leaves, tree_deserialise_leaves"""

###############################################################################
# Define a small custom model using BN + Dropout + MLP
###############################################################################
class TinyMLP(eqx.Module):
    # Let's show off a stateful BatchNorm and a stochastic Dropout
    bn: eqx.nn.BatchNorm
    linear1: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear

    def __init__(self, key):
        # We'll create a small MLP for demonstration
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        # Suppose input_dim = 4, hidden_dim=16, output_dim=3
        input_dim = 4
        hidden_dim = 16
        output_dim = 3

        # eqx.nn.make_with_state can build the module plus an initial "state" 
        # at the same time, but you can also do your own arrangement.
        # BN needs an axis_name for usage with vmap. If we won't vmap here, 
        # we can pass e.g. axis_name=None, but we actually must pass *something* 
        # that won't break. We'll say "batch" for if we do vmap.
        # Alternatively if you never intend to vmap, pass a unique string 
        # and just don't use it. 
        self.bn = eqx.nn.BatchNorm(
            input_size=hidden_dim, axis_name="batch", inference=False
        )
        self.linear1 = eqx.nn.Linear(
            in_features=input_dim, out_features=hidden_dim, key=k1
        )
        # p=0.2 dropout
        self.dropout = eqx.nn.Dropout(p=0.2, inference=False)
        self.linear2 = eqx.nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, key=k2
        )
        self.linear3 = eqx.nn.Linear(
            in_features=hidden_dim, out_features=output_dim, key=k3
        )

    def __call__(self, x: jnp.ndarray, state: eqx.nn.State, *, key, inference: bool):
        # We'll manually pass inference=... for dropout & BN 
        # or we can rely on eqx.nn.inference_mode + partial usage.
        # We'll do it manually to demonstrate.
        # 
        # Step 1: apply first linear
        x = self.linear1(x, key=None)
        # Step 2: apply second linear with dropout
        x = self.dropout(x, key=key, inference=inference)
        x = jax.nn.relu(self.linear2(x, key=None))
        # Step 3: batchnorm
        # BN uses state in/out
        # The BN layer is annotated inference=..., but we can override:
        x, new_state = self.bn(x, state, key=None, inference=inference)
        x = jax.nn.relu(x)
        # Step 4: final linear
        x = self.linear3(x, key=None)
        return x, new_state


###############################################################################
# Create data
###############################################################################
def create_synthetic_data(num_samples=1000):
    rng = jax.random.PRNGKey(42)
    X = jax.random.normal(rng, (num_samples, 4))
    # Suppose we do a multiclass problem with 3 classes. We'll do integer labels [0..2].
    # We'll artificially define a label function
    # If sum of x is > 2 => label 2
    # else if sum of x is > 0 => label 1
    # else => label 0
    sums = jnp.sum(X, axis=1)
    Y = jnp.where(sums > 2, 2, jnp.where(sums > 0, 1, 0))
    return X, Y

###############################################################################
# Loss function for multiclass classification
# We'll assume labels are integer [0..out_features-1].
# We'll use "softmax_cross_entropy_with_integer_labels" from optax
###############################################################################
def multiclass_loss_fn(pred_logits, true_labels):
    # (batch, num_classes)
    # (batch,)
    return optax.softmax_cross_entropy_with_integer_labels(pred_logits, true_labels).mean()


###############################################################################
# Main test
###############################################################################
def main():
    # 1) Create data
    X, Y = create_synthetic_data(num_samples=5000)
    # 2) Build the model + state
    # We'll do eqx.nn.make_with_state to build them in one pass:
    init_key = jax.random.PRNGKey(0)
    model_init_fn = eqx.nn.make_with_state(TinyMLP)
    model, state = model_init_fn(init_key)

    # 3) Build the Trainer
    trainer = EquinoxTrainer(
        model=model,
        state=state,
        loss_fn=multiclass_loss_fn,
        optimizer=optax.adam(1e-3),
        rng=jax.random.PRNGKey(123),
    )

    # 4) Fit
    trainer.fit(X, Y, batch_size=64, num_epochs=10)

    # 5) Evaluate
    val_loss = trainer.evaluate(X, Y, batch_size=64)
    print("Validation Loss: ", val_loss)

    # 6) Predict
    preds = trainer.predict(X[:10], batch_size=5)  # shape (10, 3) logits
    pred_labels = jnp.argmax(preds, axis=1)
    print("Sample predictions on first 10:", pred_labels)

    # 7) Save
    trainer.save("model_checkpoint.eqx")

    # 8) Load
    new_trainer = EquinoxTrainer.load(
        "model_checkpoint.eqx",
        loss_fn=multiclass_loss_fn,
        optimizer=optax.adam(1e-3),
        rng=jax.random.PRNGKey(999),
        use_jit=True,
    )
    # verify loaded model
    val_loss_loaded = new_trainer.evaluate(X, Y, batch_size=64)
    print("Validation Loss (loaded model): ", val_loss_loaded)


if __name__ == "__main__":
    main()
