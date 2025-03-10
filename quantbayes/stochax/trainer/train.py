import math
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

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
# A tiny helper to replicate a tree "shallowly" across `count` copies.
###############################################################################
def replicate_tree(pytree: Any, count: int) -> List[Any]:
    """
    Returns a list of `count` copies of `pytree`. We just replicate the same 
    structure, referencing the same leaves. 
    (If you need truly distinct leaf references, you'd have to copy them yourself.)
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    return [treedef.unflatten(leaves) for _ in range(count)]


###############################################################################
# Key function: wrap forward pass in a vmap if we detect a batch dimension
###############################################################################
def maybe_vmap_forward(
    model: eqx.Module,
    state: Any,
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    rng: jax.random.PRNGKey,
    inference: bool,
):
    """
    If x_batch has a leading dimension we treat as "batch size", then vmap
    the single-sample `model(x, state, key=rng, inference=...)` over that dimension.
    Otherwise, call model once (single sample).

    Returns:
       (pred_y, new_state):
         - If vmap is used, pred_y has shape [batch_size, ...]
         - If no vmap is used, pred_y has shape model(...) would produce
         - new_state is "one" final state or a simplification. 
           (For fully correct BN merges, you'd need a more advanced approach.)
    """

    # Detect if x_batch has shape (batch_size, ...).
    if x_batch.ndim >= 1:
        batch_size = x_batch.shape[0]
    else:
        batch_size = None

    # If no batch dimension or zero, just call once.
    if batch_size is None or batch_size == 0:
        pred_y, new_state = model(x_batch, state, key=rng, inference=inference)
        return pred_y, new_state

    # Otherwise, we define a per-sample function that calls the single-sample model
    def per_sample_forward(x, y, s, r):
        # We ignore y except for matching shape
        return model(x, s, key=r, inference=inference)

    # We'll replicate the entire state for each sample. 
    # This is "toy" for BN, etc.:
    state_list = replicate_tree(state, batch_size)

    # vmap across axis=0 of x,y, state_list, but keep rng the same
    f_vmap = jax.vmap(per_sample_forward, in_axes=(0, 0, 0, None), out_axes=(0, 0))

    pred_y, new_state_batch = f_vmap(x_batch, y_batch, state_list, rng)

    # We'll just return the "last" new_state across the batch. 
    # (You could do a reduce or something else if desired.)
    new_state_single = jax.tree_map(lambda sarr: sarr[-1], new_state_batch)
    return pred_y, new_state_single


###############################################################################
# A universal trainer that can handle single-sample models with vmap
###############################################################################
class EquinoxTrainer:
    """
    A class to encapsulate training, evaluation, prediction for Equinox-based models:
      - single-sample models automatically vectorized across batch dimension 
        via maybe_vmap_forward
      - Dropout or other random layers (fresh PRNGKey)
      - Potentially stateful layers (BatchNorm) - though fully correct 
        BN merges require more advanced logic
      - Toggling inference/training mode
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
            model: An equinox.Module expecting a single sample shape
            state: The state PyTree if model is stateful (e.g. BN)
            loss_fn: A function (pred_y, true_y) -> scalar
            optimizer: An optax optimizer
            rng: PRNGKey for random layers
            use_jit: Whether to JIT-compile the training step
        """
        self.model = model
        self.state = state
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        self.rng = rng
        self.use_jit = use_jit

        def train_step(model, state, opt_state, x_batch, y_batch, rng):
            """
            Single step over one batch. 
            Returns: (model, state, opt_state, loss_val).
            """

            def _loss_fn(m, s, xb, yb):
                # forward pass
                pred_y, s_out = maybe_vmap_forward(m, s, xb, yb, rng, inference=False)
                return self.loss_fn(pred_y, yb), s_out

            # Take gradient wrt model & state
            (loss_val, new_state), grads = filter_grad(_loss_fn, has_aux=True)(
                model, state, x_batch, y_batch
            )
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_model = apply_updates(model, updates)
            return new_model, new_state, new_opt_state, loss_val

        # Possibly JIT-compile
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
        Train for `num_epochs`, automatically doing vmap if X_train has shape (N, ...).

        Args:
            X_train, Y_train: shape (N, ...)
            batch_size: int
            num_epochs: number of epochs
            shuffle: whether to shuffle
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
                f"Loss={mean_loss.item():.4f} "
                f"(elapsed: {time.time()-t0:.1f}s)"
            )

    def evaluate(self, X_val: jnp.ndarray, Y_val: jnp.ndarray, batch_size: int) -> float:
        """
        Evaluate the model in inference mode. 
        We'll do single-sample model calls but vmap if there's a leading dimension.

        Args:
            X_val, Y_val: shape (N, ...)
            batch_size: int

        Returns:
            float: Mean loss over the entire dataset
        """
        # Inference mode
        inference_model = eqx.nn.inference_mode(self.model)
        inference_model = eqx.Partial(inference_model, state=self.state)

        all_losses = []
        for x_batch, y_batch in jax_data_loader(X_val, Y_val, batch_size, shuffle=False):
            pred_y, _ = maybe_vmap_forward(
                inference_model.__self__,
                inference_model.state,
                x_batch,
                y_batch,
                rng=None,  # no dropout randomness
                inference=True
            )
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

        Args:
            X_test: shape (N, ...)
            batch_size: int
            return_np: if True => return a numpy array, else a JAX array.

        Returns:
            predictions shape (N, ...)
        """
        inference_model = eqx.nn.inference_mode(self.model)
        inference_model = eqx.Partial(inference_model, state=self.state)

        preds_collected = []
        for x_batch, _ in jax_data_loader(X_test, X_test, batch_size, shuffle=False):
            pred_y, _ = maybe_vmap_forward(
                inference_model.__self__,
                inference_model.state,
                x_batch,
                x_batch,  # dummy Y
                rng=None,
                inference=True
            )
            preds_collected.append(pred_y)

        preds = jnp.concatenate(preds_collected, axis=0)
        if return_np:
            return preds.to_py()
        else:
            return preds

    def save(self, path: str) -> None:
        """
        Save model+state+opt_state to a single file.
        """
        packed = (self.model, self.state, self.opt_state)
        with open(path, "wb") as f:
            tree_serialise_leaves(packed, f)

    @classmethod
    def load(cls, path: str, loss_fn=None, optimizer=None, rng=None, use_jit=True):
        """
        Load model+state+opt_state from a file, returning a new trainer.
        """
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
# EXAMPLE USAGE
###############################################################################
if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import optax
    import equinox as eqx

    class TinyMLP(eqx.Module):
        bn: eqx.nn.BatchNorm
        linear1: eqx.nn.Linear
        dropout: eqx.nn.Dropout
        linear2: eqx.nn.Linear
        linear3: eqx.nn.Linear

        def __init__(self, key):
            k1, k2, k3, k4, k5 = jax.random.split(key, 5)
            input_dim = 4
            hidden_dim = 16
            output_dim = 3

            self.bn = eqx.nn.BatchNorm(
                input_size=hidden_dim, axis_name="batch", inference=False
            )
            self.linear1 = eqx.nn.Linear(
                in_features=input_dim, out_features=hidden_dim, key=k1
            )
            self.dropout = eqx.nn.Dropout(p=0.2, inference=False)
            self.linear2 = eqx.nn.Linear(
                in_features=hidden_dim, out_features=hidden_dim, key=k2
            )
            self.linear3 = eqx.nn.Linear(
                in_features=hidden_dim, out_features=output_dim, key=k3
            )

        def __call__(self, x: jnp.ndarray, state: eqx.nn.State, *, key, inference: bool):
            x = self.linear1(x, key=None)
            x = self.dropout(x, key=key, inference=inference)
            x = jax.nn.relu(self.linear2(x, key=None))
            x, new_state = self.bn(x, state, key=None, inference=inference)
            x = jax.nn.relu(x)
            x = self.linear3(x, key=None)
            return x, new_state

    def create_synthetic_data(num_samples=1000):
        rng = jax.random.PRNGKey(42)
        X = jax.random.normal(rng, (num_samples, 4))
        sums = jnp.sum(X, axis=1)
        Y = jnp.where(sums > 2, 2, jnp.where(sums > 0, 1, 0))
        return X, Y

    def multiclass_loss_fn(pred_logits, true_labels):
        return optax.softmax_cross_entropy_with_integer_labels(pred_logits, true_labels).mean()

    def main():
        # create data
        X, Y = create_synthetic_data(num_samples=5000)

        # build model + state
        init_key = jax.random.PRNGKey(0)
        model_init_fn = eqx.nn.make_with_state(TinyMLP)
        model, state = model_init_fn(init_key)

        # build trainer
        trainer = EquinoxTrainer(
            model=model,
            state=state,
            loss_fn=multiclass_loss_fn,
            optimizer=optax.adam(1e-3),
            rng=jax.random.PRNGKey(123),
        )

        # train
        trainer.fit(X, Y, batch_size=64, num_epochs=5)

        # evaluate
        val_loss = trainer.evaluate(X, Y, batch_size=64)
        print("Validation Loss:", val_loss)

        # predict
        preds = trainer.predict(X[:10], batch_size=5)
        pred_labels = jnp.argmax(preds, axis=1)
        print("Predicted labels on first 10:", pred_labels)

        # save
        trainer.save("model_checkpoint.eqx")

        # load
        new_trainer = EquinoxTrainer.load(
            "model_checkpoint.eqx",
            loss_fn=multiclass_loss_fn,
            optimizer=optax.adam(1e-3),
            rng=jax.random.PRNGKey(999),
            use_jit=True,
        )
        val_loss_loaded = new_trainer.evaluate(X, Y, batch_size=64)
        print("Val Loss (loaded model):", val_loss_loaded)

    main()
