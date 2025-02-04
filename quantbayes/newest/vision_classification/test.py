"""
image_classification_transformer.py

6) Image classification using a Vision Transformer style approach (MultiheadAttention).
Data: X ∈ ℝ^(B×H×W×C), y ∈ ℝ^(B×num_classes) (one-hot).

Requirements:
- eqx.nn.MultiheadAttention
- Cross-entropy
- Evaluate accuracy, confusion matrix
- Plot predictions, images, train & val curves
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
class SimpleVisionTransformer(eqx.Module):
    """
    A very simplified Vision Transformer-like classification:
    1) Flatten the spatial dimensions into a sequence of patches/tokens.
    2) Use eqx.nn.MultiheadAttention on them.
    3) Global average (or class token).
    4) MLP head to produce logits.
    """
    embed_linear: eqx.nn.Linear
    attn: eqx.nn.MultiheadAttention
    fc_head: eqx.nn.Linear
    patch_size: int
    seq_length: int
    num_classes: int

    def __init__(self, height, width, in_channels, patch_size, num_heads, emb_dim, num_classes, *, key):
        """
        height, width: image size
        patch_size: patch dimension (square)
        emb_dim: embedding dimension
        """
        # create eqx modules
        superkey, ekey, akey, hkey = jax.random.split(key, 4)
        self.patch_size = patch_size
        # number of patches in one dimension
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        self.seq_length = num_patches_h * num_patches_w
        # flatten each patch => embed => embedding dimension
        in_dim_per_patch = patch_size * patch_size * in_channels

        # A linear to embed patch -> emb_dim
        self.embed_linear = eqx.nn.Linear(in_dim_per_patch, emb_dim, key=ekey, use_bias=True)
        # MultiheadAttention from eqx
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=emb_dim,
            key_size=emb_dim,
            value_size=emb_dim,
            output_size=emb_dim,
            qk_size=emb_dim // num_heads,
            vo_size=emb_dim // num_heads,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=0.0,
            inference=False,
            key=akey
        )
        # final classification head
        self.fc_head = eqx.nn.Linear(emb_dim, num_classes, key=hkey, use_bias=True)
        self.num_classes = num_classes

    def __call__(self, x):
        """
        x: (B, H, W, C)
        We'll:
          1) reshape into patches, (B, seq_len, patch_size^2*C)
          2) embed each patch
          3) apply multi-head attn
          4) average pool the sequence dimension
          5) final linear to produce logits
        """
        # vmap over batch dimension
        def forward_image(img):
            # img: (H, W, C)
            patches = extract_patches(img, self.patch_size)  # shape (seq_len, patch_size^2*C)
            embedded = jax.vmap(self.embed_linear)(patches)  # (seq_len, emb_dim)
            # self.attn expects shape (seq_len, emb_dim) for query/key/value
            out = self.attn(embedded, embedded, embedded)    # shape (seq_len, emb_dim)
            # global average
            out = jnp.mean(out, axis=0)  # (emb_dim,)
            logits = self.fc_head(out)   # (num_classes,)
            return logits

        return jax.vmap(forward_image)(x)


def extract_patches(img, patch_size):
    """img: (H, W, C). Return shape: (num_patches, patch_size*patch_size*C)."""
    H, W, C = img.shape
    n_ph = H // patch_size
    n_pw = W // patch_size
    patches = []
    for i in range(n_ph):
        for j in range(n_pw):
            patch = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
            patches.append(patch.reshape(-1))
    return jnp.stack(patches, axis=0)


# ----------------------------
# Data Handling
# ----------------------------
def prepare_data(num_samples=200, height=32, width=32, channels=1, num_classes=3):
    """Generate synthetic images and random labels."""
    X = np.random.rand(num_samples, height, width, channels).astype(np.float32)
    y_int = np.random.randint(0, num_classes, size=(num_samples,))
    y = jax.nn.one_hot(jnp.array(y_int), num_classes)
    return jnp.array(X), y, y_int


def train_val_split(X, y, y_int, val_ratio=0.2, seed=42):
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    split = int(N*(1-val_ratio))
    return X[idx[:split]], y[idx[:split]], y_int[idx[:split]], X[idx[split:]], y[idx[split:]], y_int[idx[split:]]


# ----------------------------
# Loss
# ----------------------------
def cross_entropy_loss(model, X, Y):
    """Compute cross-entropy for classification. Y is one-hot."""
    logits = model(X)  # (B, num_classes)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(Y * log_probs, axis=-1))


def compute_accuracy_confusion(model, X, Y, Y_int):
    logits = model(X)  # (B, num_classes)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == Y_int)
    num_classes = Y.shape[1]
    cm = jnp.zeros((num_classes, num_classes), dtype=jnp.int32)
    def body(i, cm_):
        return cm_.at[Y_int[i], preds[i]].add(1)
    cm = jax.lax.fori_loop(0, X.shape[0], body, cm)
    return acc, cm


# ----------------------------
# Training
# ----------------------------
@eqx.filter_jit
def make_step(model, X, Y, opt_state, optimizer):
    loss_value, grads = eqx.filter_value_and_grad(cross_entropy_loss)(model, X, Y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss_value, model, opt_state


def train_model(model, X_train, Y_train, X_val, Y_val, lr=1e-3, epochs=100):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        loss_train, model, opt_state = make_step(model, X_train, Y_train, opt_state, optimizer)
        loss_val = cross_entropy_loss(model, X_val, Y_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train CE: {loss_train:.4f} - Val CE: {loss_val:.4f}")
    return model, train_losses, val_losses


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(model, X_test, Y_test, Y_test_int, train_losses, val_losses):
    acc, cm = compute_accuracy_confusion(model, X_test, Y_test, Y_test_int)
    print(f"Test Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Show some predictions
    preds = jnp.argmax(model(X_test), axis=-1)
    n_show = min(4, X_test.shape[0])
    plt.figure(figsize=(12,6))
    for i in range(n_show):
        plt.subplot(2, n_show, i+1)
        plt.imshow(np.array(X_test[i].squeeze()), cmap='gray')
        plt.title(f"Pred: {preds[i].item()}")
        plt.axis('off')
        plt.subplot(2, n_show, n_show + i + 1)
        plt.imshow(np.array(X_test[i].squeeze()), cmap='gray')
        plt.title(f"Label: {Y_test_int[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Plot losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CE Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    key = jax.random.PRNGKey(5)
    X, Y, Y_int = prepare_data(num_samples=200, height=32, width=32, channels=1, num_classes=3)
    X_train, Y_train, Y_train_int, X_val, Y_val, Y_val_int = train_val_split(X, Y, Y_int, val_ratio=0.2)

    model_key = jax.random.split(key, 1)[0]
    # Very small "vision transformer"
    patch_size = 4
    num_heads = 2
    emb_dim = 16
    model = SimpleVisionTransformer(height=32, width=32, in_channels=1,
                                    patch_size=patch_size,
                                    num_heads=num_heads,
                                    emb_dim=emb_dim,
                                    num_classes=3,
                                    key=model_key)

    model, train_losses, val_losses = train_model(model, X_train, Y_train, X_val, Y_val, lr=1e-3, epochs=50)
    evaluate_model(model, X_val, Y_val, Y_val_int, train_losses, val_losses)


if __name__ == "__main__":
    main()
