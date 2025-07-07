import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import augmax
from typing import Callable, Optional, Iterator, Tuple, Union, Any, List


AugmentFn = Callable[[jr.key, jnp.ndarray], jnp.ndarray]


def make_augmax_augment(transform: AugmentFn) -> AugmentFn:
    """
    Wrap an augmax.Chain (or any HWC-based transform) so it can
    take and return channel‐first batches [N, C, H, W].
    """

    @jax.jit
    def _augment(key: jr.key, batch_chw: jnp.ndarray) -> jnp.ndarray:
        subkeys = jr.split(key, batch_chw.shape[0])
        batch_hwc = jnp.transpose(batch_chw, (0, 2, 3, 1))
        aug_hwc = jax.vmap(transform)(subkeys, batch_hwc)
        return jnp.transpose(aug_hwc, (0, 3, 1, 2))

    return _augment


def data_loader(
    X: jnp.ndarray,
    y: jnp.ndarray,
    batch_size: int,
    *,
    shuffle: bool = True,
    key: Optional[jr.key] = None,
    augment_fn: Optional[Callable[[jr.key, jnp.ndarray], jnp.ndarray]] = None,
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Mini‐batch generator with optional Augmax augmentation on‐device.
    """
    n = X.shape[0]
    idx = jnp.arange(n)

    if shuffle:
        if key is None:
            raise ValueError("`shuffle=True` but no key provided.")
        key, sk = jr.split(key)
        idx = jr.permutation(sk, idx)

    for i in range(0, n, batch_size):
        batch_idx = idx[i : i + batch_size]
        xb, yb = X[batch_idx], y[batch_idx]
        if augment_fn is not None:
            if key is None:
                raise ValueError("`augment_fn` given but no PRNG key supplied.")
            key, sk = jr.split(key)
            xb = augment_fn(sk, xb)
        yield xb, yb


@eqx.filter_jit
def binary_loss(model, state, x, y, key):
    keys = jr.split(key, x.shape[0])
    logits, state = jax.vmap(
        model, in_axes=(0, 0, None), out_axes=(0, None), axis_name="batch"
    )(x, keys, state)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))
    return loss, state


@eqx.filter_jit
def multiclass_loss(model, state, x, y, key):
    keys = jr.split(key, x.shape[0])
    logits, state = jax.vmap(
        model, in_axes=(0, 0, None), out_axes=(0, None), axis_name="batch"
    )(x, keys, state)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
    return loss, state


@eqx.filter_jit
def regression_loss(model, state, x, y, key):
    keys = jr.split(key, x.shape[0])
    preds, state = jax.vmap(
        model, in_axes=(0, 0, None), out_axes=(0, None), axis_name="batch"
    )(x, keys, state)
    loss = jnp.mean((preds - y) ** 2)
    return loss, state


def global_spectral_penalty(model) -> jnp.ndarray:
    total = jnp.array(0.0, dtype=jnp.float32)

    if hasattr(model, "__spectral_penalty__"):
        raw = model.__spectral_penalty__()
        n = model._spectral_weights().size
        total += raw / n

    if hasattr(model, "delta_alpha"):
        total += jnp.mean(model.delta_alpha**2)

    if isinstance(model, eqx.Module):
        for v in vars(model).values():
            total += global_spectral_penalty(v)
    elif isinstance(model, (list, tuple)):
        for v in model:
            total += global_spectral_penalty(v)
    elif isinstance(model, dict):
        for v in model.values():
            total += global_spectral_penalty(v)

    return total


@eqx.filter_jit
def train_step(
    model,
    state,
    opt_state,
    x: jnp.ndarray,
    y: jnp.ndarray,
    key: jr.key,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    lambda_spec: float = 0.0,
) -> Tuple[Any, Any, Any, jnp.ndarray]:
    def total_loss_fn(m, s, xb, yb, k):
        base, new_s = loss_fn(m, s, xb, yb, k)
        pen = jnp.array(0.0)
        if lambda_spec > 0.0:
            pen = lambda_spec * global_spectral_penalty(m)
        return base + pen, (new_s, base, pen)

    (tot, (new_state, _, _)), grads = eqx.filter_value_and_grad(
        total_loss_fn, has_aux=True
    )(model, state, x, y, key)

    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    model = eqx.apply_updates(model, updates)

    return model, new_state, opt_state, tot


@eqx.filter_jit
def eval_step(model, state, x, y, key, loss_fn):
    loss, _ = loss_fn(model, state, x, y, key)
    return loss


def train(
    model: Any,
    state: Any,
    opt_state: Any,
    optimizer: Any,
    loss_fn: Callable,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    y_val: jnp.ndarray,
    batch_size: int,
    num_epochs: int,
    patience: int,
    key: jr.PRNGKey,
    *,
    augment_fn: Optional[Callable[[jr.PRNGKey, jnp.ndarray], jnp.ndarray]] = None,
    lambda_spec: float = 0.0,
    ckpt_path: Optional[str] = None,
    return_penalty_history: bool = False,
) -> Union[
    Tuple[Any, Any, List[float], List[float]],
    Tuple[Any, Any, List[float], List[float], List[float]],
]:
    rng, eval_rng = jr.split(key)
    train_losses: List[float] = []
    val_losses: List[float] = []
    penalty_history: List[float] = []
    best_val = float("inf")
    best_params = best_static = best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        epoch_train_loss = 0.0
        n_train = 0
        rng, perm_rng = jr.split(rng)
        for xb, yb in data_loader(
            X_train,
            y_train,
            batch_size,
            shuffle=True,
            key=perm_rng,
            augment_fn=augment_fn,
        ):
            rng, step_rng = jr.split(rng)
            model, state, opt_state, tot_loss = train_step(
                model,
                state,
                opt_state,
                xb,
                yb,
                step_rng,
                loss_fn,
                optimizer,
                lambda_spec,
            )
            epoch_train_loss += float(tot_loss) * xb.shape[0]
            n_train += xb.shape[0]
        epoch_train_loss /= n_train
        train_losses.append(epoch_train_loss)

        epoch_val_loss = 0.0
        n_val = 0
        for xb, yb in data_loader(
            X_val, y_val, batch_size, shuffle=False, key=eval_rng, augment_fn=None
        ):
            eval_rng, vk = jr.split(eval_rng)
            data_loss, _ = loss_fn(model, state, xb, yb, vk)
            epoch_val_loss += float(data_loss) * xb.shape[0]
            n_val += xb.shape[0]
        epoch_val_loss /= n_val
        val_losses.append(epoch_val_loss)

        pen = float(global_spectral_penalty(model))
        penalty_history.append(pen)

        if epoch % max(1, num_epochs // 10) == 0 or epoch == num_epochs:
            lr = (
                optimizer.state_dict()["param_states"]["learning_rate"]
                if hasattr(optimizer, "state_dict")
                else None
            )
            print(
                f"[Epoch {epoch:3d}/{num_epochs}] "
                f"Train={epoch_train_loss:.4f} | Val={epoch_val_loss:.4f} "
                f"| Pen={pen:.4f}" + (f" | LR={lr:.3e}" if lr is not None else "")
            )

        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            patience_counter = 0
            best_params = eqx.filter(model, eqx.is_inexact_array)
            best_static = eqx.filter(model, lambda x: not eqx.is_inexact_array(x))
            best_state = state
            if ckpt_path:
                eqx.tree_serialise_leaves(
                    ckpt_path, eqx.combine(best_params, best_static)
                )
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch}")
                break

    best_model = eqx.combine(best_params, best_static)
    if return_penalty_history:
        return best_model, best_state, train_losses, val_losses, penalty_history
    else:
        return best_model, best_state, train_losses, val_losses


@eqx.filter_jit
def predict(model, state, X, key):
    model = eqx.nn.inference_mode(model)
    batched_apply = jax.vmap(model, in_axes=(0, None, None))
    logits, _ = batched_apply(X, key, state)
    return logits


def predict_batched(model, state, X, key, batch_size: int = 256):
    inference_model = eqx.nn.inference_mode(model)
    n = X.shape[0]
    parts = []
    keys = jr.split(key, (n + batch_size - 1) // batch_size)
    start = 0

    @eqx.filter_jit
    def infer_batch(m, s, xb, k):
        subkeys = jr.split(k, xb.shape[0])
        preds, _ = jax.vmap(m, in_axes=(None, 0, 0))(s, xb, subkeys)
        return preds

    for k in keys:
        xb = X[start : start + batch_size]
        parts.append(infer_batch(inference_model, state, xb, k))
        start += batch_size

    return jnp.concatenate(parts, axis=0)


if __name__ == "__main__":
    """
    Synthetic vision pipeline test — runs in <10 s on CPU.
    Replace with CIFAR-10, ImageNet, etc. in real experiments.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import augmax

    # ----- fake “dataset” ---------------------------------------------------
    rng = np.random.RandomState(0)
    N, H, W, C, NUM_CLASSES = 2048, 1, 28, 28, 1
    X_np = rng.rand(N, H, W, C).astype("float32")  # [N, H, W, C]
    y_np = rng.randint(0, NUM_CLASSES, size=(N,)).astype("int32")

    # train / val split
    split = int(0.8 * N)
    X_train, X_val = X_np[:split], X_np[split:]
    y_train, y_val = y_np[:split], y_np[split:]

    # ----- augmentation pipeline -------------------------------------------
    transform = augmax.Chain(
        augmax.HorizontalFlip(),
        augmax.Rotate(angle_range=15),
    )
    augment_fn = make_augmax_augment(transform)

    # ----- simple CNN -------------------------------------------------------
    class SimpleCNN(eqx.Module):
        conv1: eqx.nn.Conv2d
        bn1: eqx.nn.BatchNorm
        conv2: eqx.nn.Conv2d
        bn2: eqx.nn.BatchNorm
        pool1: eqx.nn.MaxPool2d
        pool2: eqx.nn.MaxPool2d
        fc1: eqx.nn.Linear
        bn3: eqx.nn.BatchNorm
        fc2: eqx.nn.Linear
        drop1: eqx.nn.Dropout
        drop2: eqx.nn.Dropout
        drop3: eqx.nn.Dropout

        def __init__(self, key):
            k1, k2, k3, k4 = jr.split(key, 4)
            self.conv1 = eqx.nn.Conv2d(1, 32, kernel_size=3, padding=1, key=k1)
            self.bn1 = eqx.nn.BatchNorm(input_size=32, axis_name="batch")
            self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=3, padding=1, key=k2)
            self.bn2 = eqx.nn.BatchNorm(input_size=64, axis_name="batch")
            self.pool1 = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = eqx.nn.Linear(64 * 7 * 7, 128, key=k3)
            self.bn3 = eqx.nn.BatchNorm(input_size=128, axis_name="batch")
            self.fc2 = eqx.nn.Linear(128, 10, key=k4)
            self.drop1 = eqx.nn.Dropout(0.25)
            self.drop2 = eqx.nn.Dropout(0.25)
            self.drop3 = eqx.nn.Dropout(0.50)

        def __call__(self, x, key, state):
            d1, d2, d3 = jr.split(key, 3)
            x = self.conv1(x)
            x, state = self.bn1(x, state)
            x = jax.nn.relu(x)
            x = self.pool1(x)
            x = self.drop1(x, key=d1)
            x = self.conv2(x)
            x, state = self.bn2(x, state)
            x = jax.nn.relu(x)
            x = self.pool2(x)
            x = self.drop2(x, key=d2)
            x = x.reshape(-1)
            x = self.fc1(x)
            x, state = self.bn3(x, state)
            x = jax.nn.relu(x)
            x = self.drop3(x, key=d3)
            x = self.fc2(x)
            return x, state

    master_key = jr.PRNGKey(42)
    model_key, train_key = jr.split(master_key)

    model, state = eqx.nn.make_with_state(SimpleCNN)(model_key)
    lr_sched = optax.cosine_decay_schedule(1e-3, decay_steps=500)
    optimizer = optax.adamw(learning_rate=lr_sched, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    best_model, best_state, tr_loss, va_loss = train(
        model,
        state,
        opt_state,
        optimizer,
        multiclass_loss,
        jnp.array(X_train),
        jnp.array(y_train),
        jnp.array(X_val),
        jnp.array(y_val),
        batch_size=256,
        num_epochs=20,
        patience=5,
        key=train_key,
        augment_fn=augment_fn,
        lambda_spec=0.0,
    )

    plt.plot(tr_loss, label="train")
    plt.plot(va_loss, label="val")
    plt.legend()
    plt.show()
