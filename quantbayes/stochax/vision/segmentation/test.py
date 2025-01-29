# image_segmentation_script.py

import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
from flax.training import train_state
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Optional


# -----------------------------------------------------------
# 1. A Minimal UNet-like Segmentation Model
# -----------------------------------------------------------
class ConvBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        return x


class UNet(nn.Module):
    """
    A very simplified UNet for demonstration:
    - Downsample x2 => 2-level
    - Upsample back x2 => 2-level
    - Output segmentation logits
    """

    num_classes: int

    @nn.compact
    def __call__(self, x, **kwargs):
        # x: (batch_size, height, width, channels)
        # Encoder
        c1 = ConvBlock(32)(x)
        p1 = nn.max_pool(c1, window_shape=(2, 2), strides=(2, 2))
        c2 = ConvBlock(64)(p1)
        p2 = nn.max_pool(c2, window_shape=(2, 2), strides=(2, 2))

        # Bottleneck
        c3 = ConvBlock(128)(p2)

        # Decoder
        # upsample + concatenate skip connection
        up1 = nn.ConvTranspose(features=64, kernel_size=(2, 2), strides=(2, 2))(c3)
        concat1 = jnp.concatenate([up1, c2], axis=-1)
        c4 = ConvBlock(64)(concat1)

        up2 = nn.ConvTranspose(features=32, kernel_size=(2, 2), strides=(2, 2))(c4)
        concat2 = jnp.concatenate([up2, c1], axis=-1)
        c5 = ConvBlock(32)(concat2)

        # Output layer => logits for each class
        logits = nn.Conv(self.num_classes, kernel_size=(1, 1))(c5)
        return logits


# -----------------------------------------------------------
# 2. Create Train State
# -----------------------------------------------------------
def create_train_state(
    rng: jax.random.PRNGKey,
    model: nn.Module,
    learning_rate: float,
    example_input: jnp.ndarray,
):
    params = model.init(rng, example_input)["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# -----------------------------------------------------------
# 3. Loss (Pixelwise Cross-Entropy) & Train Step
# -----------------------------------------------------------
def segmentation_loss(
    params,
    apply_fn,
    x: jnp.ndarray,
    y: jnp.ndarray,
    apply_fn_kwargs: Dict[str, Any] = None,
) -> jnp.ndarray:
    """
    y: (batch_size, height, width) integer labels in [0, num_classes-1]
    logits: (batch_size, height, width, num_classes)
    We'll flatten for cross-entropy: (batch_size * height * width, num_classes)
    and compare with one-hot (batch_size * height * width, num_classes).
    """
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}
    logits = apply_fn({"params": params}, x, **apply_fn_kwargs)  # (B,H,W,C)
    num_classes = logits.shape[-1]

    # Flatten
    logits_flat = logits.reshape((-1, num_classes))  # (B*H*W, C)
    labels_flat = y.reshape((-1,))  # (B*H*W,)

    one_hot = jax.nn.one_hot(labels_flat, num_classes)
    loss = optax.softmax_cross_entropy(logits_flat, one_hot).mean()
    return loss


@jax.jit
def train_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    apply_fn_kwargs: Dict[str, Any],
):
    loss, grads = jax.value_and_grad(segmentation_loss)(
        state.params, state.apply_fn, x, y, apply_fn_kwargs
    )
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


# -----------------------------------------------------------
# 4. Data Generator
# -----------------------------------------------------------
def data_generator(
    rng: jax.random.PRNGKey,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
):
    num_samples = X.shape[0]
    if shuffle:
        indices = jax.random.permutation(rng, num_samples)
        X = X[indices]
        y = y[indices]

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]


# -----------------------------------------------------------
# 5. Train Function
# -----------------------------------------------------------
def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    rng: jax.random.PRNGKey,
    apply_fn_kwargs_train: Dict[str, Any],
    apply_fn_kwargs_val: Optional[Dict[str, Any]] = None,
):
    if apply_fn_kwargs_val is None:
        apply_fn_kwargs_val = {}

    example_input = jnp.ones((1,) + X_train.shape[1:], dtype=jnp.float32)
    state = create_train_state(rng, model, learning_rate, example_input)

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        rng, data_rng = jax.random.split(rng)
        batch_losses = []

        for batch_X, batch_y in data_generator(data_rng, X_train, y_train, batch_size):
            bx = jnp.array(batch_X, dtype=jnp.float32)
            by = jnp.array(batch_y, dtype=jnp.int32)

            state, loss_val = train_step(state, bx, by, apply_fn_kwargs_train)
            batch_losses.append(loss_val)

        train_loss = float(jnp.mean(jnp.array(batch_losses)))
        val_loss = evaluate_loss(
            state.params,
            state.apply_fn,
            X_val,
            y_val,
            batch_size,
            rng,
            apply_fn_kwargs_val,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % max(1, (num_epochs // 10)) == 0:
            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Plot training/validation loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Segmentation Training & Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

    return state, train_losses, val_losses


def evaluate_loss(
    params,
    apply_fn,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: jax.random.PRNGKey,
    apply_fn_kwargs: Dict[str, Any],
):
    losses = []
    for batch_X, batch_y in data_generator(rng, X, y, batch_size, shuffle=False):
        bx = jnp.array(batch_X, dtype=jnp.float32)
        by = jnp.array(batch_y, dtype=jnp.int32)
        loss_val = segmentation_loss(params, apply_fn, bx, by, apply_fn_kwargs)
        losses.append(loss_val)
    return float(jnp.mean(jnp.array(losses)))


# -----------------------------------------------------------
# 6. Evaluation Function (MC Sampling + Visualization)
# -----------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    params,
    X_val: np.ndarray,
    y_val: np.ndarray,
    rng: jax.random.PRNGKey,
    num_samples: int,
    apply_fn_kwargs: Optional[Dict[str, Any]] = None,
    num_display: int = 3,
):
    """
    Perform multiple forward passes and visualize segmentation predictions.
    We'll pick a few samples to display: input image, ground truth mask, predicted mask.
    """
    if apply_fn_kwargs is None:
        apply_fn_kwargs = {}

    def forward(params, x, **kwargs):
        return model.apply({"params": params}, x, **kwargs)

    # We'll pick some random examples to visualize
    indices = np.random.choice(X_val.shape[0], size=num_display, replace=False)

    plt.figure(figsize=(9, 3 * num_display))

    for i, idx in enumerate(indices):
        # shape (H, W, C)
        x_img = jnp.array(X_val[idx : idx + 1], dtype=jnp.float32)
        # multiple forward passes => shape (num_samples, 1, H, W, num_classes)
        logits_list = []
        for _ in range(num_samples):
            rng, subkey = jax.random.split(rng)
            logits = forward(params, x_img, **apply_fn_kwargs)
            logits_list.append(logits)

        logits_stacked = jnp.stack(logits_list, axis=0)
        mean_logits = jnp.mean(logits_stacked, axis=0)  # (1, H, W, num_classes)
        pred_mask = jnp.argmax(mean_logits, axis=-1)[0]  # (H, W)

        # Plot
        plt.subplot(num_display, 3, 3 * i + 1)
        plt.imshow(X_val[idx])
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(num_display, 3, 3 * i + 2)
        plt.imshow(y_val[idx], cmap="jet")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(num_display, 3, 3 * i + 3)
        plt.imshow(pred_mask, cmap="jet")
        plt.title("Predicted Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# 7. Example Usage
# -----------------------------------------------------------
if __name__ == "__main__":
    rng_np = np.random.RandomState(42)
    N = 50
    H, W, C = 128, 128, 3
    num_classes = 3

    # Synthetic dataset: random RGB images in [0..1]
    X_data = rng_np.rand(N, H, W, C).astype(np.float32)
    # Random masks: integer labels in [0..num_classes-1]
    y_data = rng_np.randint(0, num_classes, size=(N, H, W)).astype(np.int32)

    # Split
    train_size = int(0.8 * N)
    X_train, y_train = X_data[:train_size], y_data[:train_size]
    X_val, y_val = X_data[train_size:], y_data[train_size:]

    # Model
    model = UNet(num_classes=num_classes)
    rng = jax.random.PRNGKey(0)

    # apply_fn_kwargs (e.g., for dropout)
    apply_fn_kwargs_train = {"deterministic": False, "rngs": {"dropout": rng}}
    apply_fn_kwargs_val = {"deterministic": True}

    # Train
    num_epochs = 10
    batch_size = 4
    learning_rate = 1e-3

    print("Starting training...")
    state, train_losses, val_losses = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        rng=rng,
        apply_fn_kwargs_train=apply_fn_kwargs_train,
        apply_fn_kwargs_val=apply_fn_kwargs_val,
    )

    # Evaluate + visualize
    print("Evaluating model (MC sampling)...")
    evaluate_model(
        model=model,
        params=state.params,
        X_val=X_val,
        y_val=y_val,
        rng=rng,
        num_samples=5,
        apply_fn_kwargs=apply_fn_kwargs_train,  # test with dropout
        num_display=3,
    )
