import jax
import os
import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def get_data(num_samples=100, num_features=1, noise_std=0.1, seed=42):

    key = jax.random.PRNGKey(seed)

    X = jax.random.normal(key, shape=(num_samples, num_features))

    true_weights = jnp.arange(1, num_features + 1)
    true_bias = 5.0

    y_true = jnp.dot(X, true_weights) + true_bias

    noise = jax.random.normal(key, shape=(num_samples,)) * noise_std
    y = y_true + noise

    return X, y


def train_test_split_jax(X, y, test_size=0.2, seed=42):

    key = jax.random.PRNGKey(seed)

    num_samples = X.shape[0]
    indices = jnp.arange(num_samples)
    shuffled_indices = jax.random.permutation(key, indices)

    test_size = int(num_samples * test_size)

    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def save_model(params, fold_idx, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"model_fold_{fold_idx}.npz")
    W, b = params
    np.savez(file_path, W=W, b=b)
    print(f"Model for fold {fold_idx} saved to {file_path}")


def model(params, X):
    W, b = params
    return jnp.dot(X, W) + b


def train_one_fold(
    train_X,
    train_y,
    val_X,
    val_y,
    learning_rate=0.01,
    epochs=1000,
    patience=10,
    noise_scale=0.0,
):
    rng = jax.random.PRNGKey(42)
    W = jax.random.normal(rng, shape=(train_X.shape[1],))
    b = jnp.zeros(1)

    params = (W, b)

    def loss_fn(params, X, y, eps=1e-6):
        preds = model(params, X)
        preds = jnp.clip(preds, 0, None)
        return jnp.mean((y - preds) ** 2)

    grad_fn = jax.grad(loss_fn)

    noise = jax.random.normal(rng, shape=params[0].shape) * noise_scale

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    patience_counter = 0

    progress_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
    for epoch in progress_bar:
        grads = grad_fn(params, train_X, train_y)
        params = (
            params[0] - learning_rate * grads[0] + noise,
            params[1] - learning_rate * grads[1],
        )

        train_loss = loss_fn(params, train_X, train_y)
        val_loss = loss_fn(params, val_X, val_y)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        progress_bar.set_postfix(val_loss=f"{val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. Best val_loss: {best_val_loss:.4f}"
            )
            break

    return params, train_losses, val_losses


X, y = get_data(num_samples=200, num_features=3, noise_std=0.2)
X_train, X_test, y_train, y_test = train_test_split_jax(X, y, test_size=0.2, seed=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = jnp.array(X)
y = jnp.array(y)

k = 3
kf = KFold(n_splits=k, shuffle=True, random_state=42)

all_train_losses = []
all_val_losses = []

for fold_idx, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"\nFold {fold_idx + 1}/{k}")

    train_X, val_X = X[train_index], X[val_index]
    train_y, val_y = y[train_index], y[val_index]

    trained_params, train_losses, val_losses = train_one_fold(
        train_X, train_y, val_X, val_y
    )

    save_model(trained_params, fold_idx)

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

    print(
        f"Finished Fold {fold_idx + 1}/{k}. Best Validation Loss: {min(val_losses):.4f}"
    )

plt.figure(figsize=(12, 6))
for fold_idx in range(k):
    plt.plot(all_train_losses[fold_idx], label=f"Fold {fold_idx + 1} - Train Loss")
    plt.plot(all_val_losses[fold_idx], label=f"Fold {fold_idx + 1} - Val Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


#######################################################################
# Predictions


def load_model(fold_idx, save_dir="models"):
    file_path = os.path.join(save_dir, f"model_fold_{fold_idx}.npz")
    data = np.load(file_path)
    W = jnp.array(data["W"])
    b = jnp.array(data["b"])
    print(f"Model for fold {fold_idx} loaded from {file_path}")
    return (W, b)


def prepare_test_data(test, scaler):
    test = scaler.transform(test)
    return jnp.array(test)


def predict_with_models(test, scaler, k, save_dir="models", aggregate="mean"):

    test_predictions_all_folds = []

    for fold_idx in range(k):
        trained_params = load_model(fold_idx, save_dir)

        test_preds = model(trained_params, test_X)

        test_predictions_all_folds.append(test_preds)

    if aggregate == "mean":
        final_predictions = jnp.mean(jnp.stack(test_predictions_all_folds), axis=0)
    elif aggregate == "median":
        final_predictions = jnp.median(jnp.stack(test_predictions_all_folds), axis=0)
    else:
        raise ValueError("Unsupported aggregation method. Use 'mean' or 'median'.")

    return final_predictions


test_X = prepare_test_data(X_test, scaler)


k = 3
final_test_predictions = predict_with_models(
    test_X, scaler, k, save_dir="models", aggregate="mean"
)

print(final_test_predictions)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, final_test_predictions, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    lw=2,
    label="Perfect Prediction",
)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
plt.legend()
plt.grid(True)
plt.show()

residuals = y_test - final_test_predictions
plt.figure(figsize=(8, 6))
plt.scatter(final_test_predictions, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(final_test_predictions, bins=30, alpha=0.7, label="Predictions")
plt.hist(y_test, bins=30, alpha=0.7, label="True Values")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.title("Distribution of Predictions and True Values")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(y_test, label="True Values", marker="o")
plt.plot(final_test_predictions, label="Predicted Values", marker="x")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("True vs Predicted Values Over Time")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, alpha=0.7, color="purple")
plt.xlabel("Residuals (True - Predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.axvline(0, color="red", linestyle="--", linewidth=1, label="Zero Error")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
sns.kdeplot(final_test_predictions, label="Predictions", fill=True, alpha=0.6)
sns.kdeplot(y_test, label="True Values", fill=True, alpha=0.6)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Prediction vs True Value Distribution")
plt.legend()
plt.grid(True)
plt.show()
