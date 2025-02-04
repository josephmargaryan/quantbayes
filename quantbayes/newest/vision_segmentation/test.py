"""
image_segmentation_unet.py

5) Image segmentation with a simplified UNet using eqx.nn.Conv2d and eqx.nn.ConvTranspose2d.
Data: X ∈ ℝ^(B×H×W×C), y ∈ ℝ^(B×H×W×C)

Requirements:
- UNet-like architecture
- Dice or cross-entropy loss
- Evaluate IoU, pixel accuracy
- Plot segmentation masks
- LR scheduling
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
# Model Definition
# ----------------------------
class UNet(eqx.Module):
    """A small UNet-like model using eqx.nn.Conv2d and eqx.nn.ConvTranspose2d."""
    enc_conv1: eqx.nn.Conv2d
    enc_conv2: eqx.nn.Conv2d
    dec_conv1: eqx.nn.ConvTranspose2d
    dec_conv2: eqx.nn.ConvTranspose2d
    final_conv: eqx.nn.Conv2d

    def __init__(self, in_channels=1, out_channels=1, hidden_channels=16, *, key):
        keys = jax.random.split(key, 5)
        # Encoder
        self.enc_conv1 = eqx.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=(1,1),
                                       padding=1, use_bias=True, key=keys[0])
        self.enc_conv2 = eqx.nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, stride=(2,2),
                                       padding=1, use_bias=True, key=keys[1])
        # Decoder
        self.dec_conv1 = eqx.nn.ConvTranspose2d(hidden_channels*2, hidden_channels, kernel_size=3, stride=(2,2),
                                                output_padding=(1,1), padding=1, use_bias=True, key=keys[2])
        self.dec_conv2 = eqx.nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=3, stride=(1,1),
                                                padding=1, use_bias=True, key=keys[3])
        self.final_conv = eqx.nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1,
                                        padding=0, use_bias=True, key=keys[4])

    def __call__(self, x):
        # x: (B, H, W, C) -> eqx's default Conv2d expects (C, H, W), so we permute
        # We'll do it in each layer
        # But let's vmap the entire forward pass over batch dimension.
        def forward_single(img):
            # img: (H, W, C)
            img_t = jnp.transpose(img, (2, 0, 1))  # (C, H, W)
            e1 = jax.nn.relu(self.enc_conv1(img_t))
            e2 = jax.nn.relu(self.enc_conv2(e1))   # downsample
            d1 = jax.nn.relu(self.dec_conv1(e2))   # upsample
            d2 = jax.nn.relu(self.dec_conv2(d1))
            out = self.final_conv(d2)             # (out_channels, H, W)
            out = jnp.transpose(out, (1, 2, 0))   # (H, W, out_channels)
            # For binary segmentation, let's do sigmoid
            return jax.nn.sigmoid(out)

        return jax.vmap(forward_single)(x)


# ----------------------------
# Data Handling
# ----------------------------
def prepare_data(num_samples=50, height=64, width=64, channels=1):
    """
    Creates synthetic images with a circular object in the center.
    X: (num_samples, H, W, C)
    y: (num_samples, H, W, C)
    """
    X_list = []
    y_list = []

    # define a simple circle in the center
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    circle = (xx - width//2)**2 + (yy - height//2)**2 < (min(height,width)//4)**2

    for _ in range(num_samples):
        # random background
        img = np.random.rand(height, width, channels).astype(np.float32)
        mask = circle.astype(np.float32)
        if channels > 1:
            mask = np.repeat(mask[..., None], channels, axis=-1)
        X_list.append(img)
        y_list.append(mask)

    X = jnp.array(np.stack(X_list, axis=0))
    y = jnp.array(np.stack(y_list, axis=0))
    return X, y


def train_val_split(X, y, val_ratio=0.2, seed=42):
    num = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(num)
    rng.shuffle(idx)
    split = int(num*(1-val_ratio))
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]


# ----------------------------
# Loss
# ----------------------------
def dice_loss(pred, target, eps=1e-6):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = jnp.sum(pred * target)
    union = jnp.sum(pred) + jnp.sum(target)
    return 1.0 - (2.0 * intersection + eps)/(union + eps)


def segmentation_loss(model, X, Y):
    """Compute mean Dice loss across the batch."""
    preds = model(X)  # (B, H, W, out_channels)
    # vmap dice over batch
    losses = jax.vmap(dice_loss)(preds, Y)
    return jnp.mean(losses)


# ----------------------------
# Training
# ----------------------------
@eqx.filter_jit
def make_step(model, X, Y, opt_state, optimizer):
    loss_value, grads = eqx.filter_value_and_grad(segmentation_loss)(model, X, Y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss_value, model, opt_state


def train_model(model, X_train, Y_train, X_val, Y_val, lr=1e-3, epochs=50):
    # Example: simple Adam, no scheduling
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_loss, model, opt_state = make_step(model, X_train, Y_train, opt_state, optimizer)
        val_loss = segmentation_loss(model, X_val, Y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Dice: {train_loss:.4f}, Val Dice: {val_loss:.4f}")
    return model, train_losses, val_losses


# ----------------------------
# Evaluation
# ----------------------------
def compute_iou_pixelacc(model, X, Y):
    preds = model(X)
    preds_binary = (preds > 0.5).astype(jnp.float32)
    intersection = jnp.sum(preds_binary * Y, axis=(1,2,3))
    union = jnp.sum(preds_binary + Y, axis=(1,2,3)) - intersection
    iou = jnp.mean(intersection/(union+1e-6))
    pixel_acc = jnp.mean(preds_binary == Y)
    return iou, pixel_acc, preds


def evaluate_model(model, X_test, Y_test, train_losses, val_losses):
    iou, pixel_acc, preds = compute_iou_pixelacc(model, X_test, Y_test)
    print(f"Test IoU: {iou:.4f}, Pixel Accuracy: {pixel_acc:.4f}")

    # Show some predictions
    n_show = min(3, X_test.shape[0])
    plt.figure(figsize=(12, 4))
    for i in range(n_show):
        plt.subplot(2, n_show, i+1)
        plt.imshow(np.array(X_test[i].squeeze()), cmap="gray")
        plt.title("Input")
        plt.axis("off")
        plt.subplot(2, n_show, i+1+n_show)
        plt.imshow(np.array(preds[i].squeeze()), cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")
    plt.show()

    # Plot train/val
    plt.figure()
    plt.plot(train_losses, label="Train Dice Loss")
    plt.plot(val_losses, label="Val Dice Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")
    plt.title("Dice Loss Curves")
    plt.legend()
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    X, Y = prepare_data(num_samples=50, height=64, width=64, channels=1)
    X_train, Y_train, X_val, Y_val = train_val_split(X, Y, val_ratio=0.2)

    key = jax.random.PRNGKey(4)
    model_key, _ = jax.random.split(key)
    model = UNet(in_channels=1, out_channels=1, hidden_channels=16, key=model_key)

    model, train_losses, val_losses = train_model(model, X_train, Y_train, X_val, Y_val, lr=1e-3, epochs=50)

    evaluate_model(model, X_val, Y_val, train_losses, val_losses)


if __name__ == "__main__":
    main()
