import time
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
from numpyro.infer import SVI, Trace_ELBO, Predictive
import optax

# Example usage:
"""
batch_size = 128
updates_per_epoch = X_train.shape[0] // batch_size
scheduler = make_scheduler(1e-3, updates_per_epoch)
optimizer = optax.adam(learning_rate=scheduler)
trainer = SVITrainer(model, guide, optimizer, mode="regression")
trainer.train(X_train, y_train, X_val, y_val)
trainer.save_history()
"""


def make_scheduler(
    init_value: float,
    updates_per_epoch: int,
    decay_rate: float = 0.9,
    staircase: bool = True,
) -> optax.Schedule:
    """
    Creates an exponential decay learning rate schedule.

    Args:
        init_value: initial learning rate.
        updates_per_epoch: number of steps per epoch (for decay schedule).
        decay_rate: multiplicative decay factor per epoch.
        staircase: if True, decay in discrete intervals.

    Returns:
        An optax learning rate schedule.
    """
    return optax.exponential_decay(
        init_value=init_value,
        transition_steps=updates_per_epoch,
        decay_rate=decay_rate,
        staircase=staircase,
    )


class SVITrainer:
    """
    Trainer for running SVI with mini-batching in NumPyro.

    Supports 'multiclass', 'binary', or 'regression' modes.

    You typically create an optimizer with `make_scheduler` and pass it in:

        updates_per_epoch = X_train.shape[0] // batch_size
        scheduler = make_scheduler(1e-3, updates_per_epoch)
        optimizer = optax.adam(learning_rate=scheduler)
        trainer = SVITrainer(model, guide, optimizer, mode="regression")
    """

    def __init__(
        self,
        model,
        guide,
        optimizer: optax.GradientTransformation,
        mode: str = "multiclass",
        batch_size: int = 128,
        num_epochs: int = 100,
        max_patience: int = 5,
        epsilon: float = 0.001,
        num_predictive_samples: int = 100,
        rng_seed: int = 0,
    ):
        assert mode in (
            "multiclass",
            "binary",
            "regression",
        ), "mode must be 'multiclass', 'binary', or 'regression'"
        self.model = model
        self.guide = guide
        self.optimizer = optimizer
        self.mode = mode
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_patience = max_patience
        self.epsilon = epsilon
        self.num_predictive_samples = num_predictive_samples
        self.rng_key = jr.PRNGKey(rng_seed)

        # History
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_metric_history = []

    def init_state(self, X_train: np.ndarray, y_train: np.ndarray):
        updates_per_epoch = X_train.shape[0] // self.batch_size
        svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())
        self.svi = svi
        self.updates_per_epoch = updates_per_epoch
        self.svi_state = svi.init(
            self.rng_key,
            X_train[: self.batch_size],
            y_train[: self.batch_size],
            N_total=X_train.shape[0],
        )

    def _train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        epoch_loss = 0.0
        self.rng_key, subkey = jr.split(self.rng_key)
        perm = jr.permutation(subkey, X.shape[0])
        for i in range(self.updates_per_epoch):
            idx = perm[i * self.batch_size : (i + 1) * self.batch_size]
            Xb, yb = X[idx], y[idx]
            self.svi_state, loss = self.svi.update(
                self.svi_state, Xb, yb, N_total=X.shape[0]
            )
            epoch_loss += loss
        avg_loss = float(epoch_loss / self.updates_per_epoch)
        self.train_loss_history.append(avg_loss)
        return avg_loss

    def _evaluate(
        self, X_val: np.ndarray, y_val: np.ndarray, X_train_size: int
    ) -> tuple[float, float]:
        # ELBO
        val_loss = float(
            self.svi.evaluate(self.svi_state, X_val, y_val, N_total=X_train_size)
        )
        self.val_loss_history.append(val_loss)

        # Predictive
        params = self.svi.get_params(self.svi_state)
        predictive = Predictive(
            self.model,
            guide=self.guide,
            params=params,
            num_samples=self.num_predictive_samples,
        )
        self.rng_key, subkey = jr.split(self.rng_key)
        # choose return_sites for Predictive
        if self.mode in ("multiclass", "binary"):
            return_sites = ("logits",)
        else:
            return_sites = ("mu", "sigma")
        samples = predictive(subkey, X_val, return_sites=return_sites)

        # compute metric
        if self.mode == "multiclass":
            logits = samples["logits"]  # (S,N,C)
            mean_logits = jnp.mean(logits, axis=0)
            y_pred = jnp.argmax(mean_logits, axis=-1)
            metric = float(jnp.mean(y_pred == y_val))
        elif self.mode == "binary":
            logits = samples["logits"]
            S, N, C = logits.shape
            # C==1 or 2; assume logits[...,1]
            probs = jax.nn.sigmoid(logits[..., 0] if C == 1 else logits[..., 1])
            mean_prob = jnp.mean(probs, axis=0)
            metric = float(jnp.mean((mean_prob > 0.5) == y_val))
        else:  # regression
            mu = samples["mu"]  # (S,N)
            mean_mu = jnp.mean(mu, axis=0)
            metric = float(jnp.sqrt(jnp.mean((mean_mu - y_val) ** 2)))  # RMSE

        self.val_metric_history.append(metric)
        return val_loss, metric

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True,
    ) -> None:
        self.init_state(X_train, y_train)
        best_metric = -jnp.inf if self.mode != "regression" else jnp.inf
        patience = self.max_patience

        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            train_elbo = self._train_epoch(X_train, y_train)
            val_elbo, val_metric = self._evaluate(X_val, y_val, X_train.shape[0])
            elapsed = time.time() - start_time

            if verbose:
                metric_name = "Val acc" if self.mode != "regression" else "Val RMSE"
                print(
                    f"Epoch {epoch} | "
                    f"Train ELBO={train_elbo:.2f} | "
                    f"Val ELBO={val_elbo:.2f} | "
                    f"{metric_name}={val_metric:.4f} | "
                    f"time={elapsed:.1f}s"
                )

            # early stopping
            improved = (
                val_metric > best_metric
                if self.mode != "regression"
                else val_metric < best_metric
            )
            if improved:
                best_metric = val_metric
                patience = self.max_patience
            else:
                patience -= 1
                if patience == 0:
                    if verbose:
                        print("Stopping early; no improvement.")
                    break

    def save_history(self, path: str = "training_history.npz") -> None:
        np.savez(
            path,
            train_loss=self.train_loss_history,
            val_loss=self.val_loss_history,
            val_metric=self.val_metric_history,
        )
        print(f"Saved training history to {path}")
