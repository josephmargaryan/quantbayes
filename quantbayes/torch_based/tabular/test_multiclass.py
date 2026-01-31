# multiclass_script_torch.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix


# ----------------------------------------------------------------
# 1. Model Definition for Multiclass
#    output_dim = num_classes
# ----------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits of shape (batch_size, num_classes)
        return self.net(x)


# ----------------------------------------------------------------
# 2. Training Function
# ----------------------------------------------------------------
def train_multiclass(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> nn.Module:
    """
    Train a multiclass classification model using CrossEntropyLoss.
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)  # class labels as long
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    def batch_generator(X, y, bs):
        n = len(X)
        indices = torch.randperm(n)
        for start in range(0, n, bs):
            end = start + bs
            batch_idx = indices[start:end]
            yield X[batch_idx], y[batch_idx]

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        for batch_x, batch_y in batch_generator(X_train_t, y_train_t, batch_size):
            optimizer.zero_grad()
            logits = model(batch_x)  # (batch_size, num_classes)
            loss = loss_fn(logits, batch_y)  # cross entropy loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % max(1, (num_epochs // 10)) == 0:
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Plot training & validation loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


def visualize_multiclass(model: nn.Module, X: np.ndarray, y: np.ndarray):
    """
    Visualizes performance for a deterministic PyTorch multiclass classification model.

    The function obtains the raw logits, applies softmax to compute probabilities,
    determines predicted classes, and then plots:
      - A confusion matrix heatmap.
      - A bar chart of average predicted probabilities per class.

    Parameters:
        model: A PyTorch nn.Module that returns logits.
        X: Input features as a NumPy array of shape (n_samples, n_features).
        y: True class labels as a NumPy array of shape (n_samples,) (integer values).
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    with torch.no_grad():
        logits = model(X_t)  # Expected shape: (n_samples, num_classes)
        probs = torch.softmax(logits, dim=-1)

    probs_np = probs.cpu().numpy()
    pred_classes = np.argmax(probs_np, axis=1)
    y_np = y_t.cpu().numpy()

    # Compute confusion matrix
    cm = confusion_matrix(y_np, pred_classes)

    # Create subplots: Confusion Matrix and Average Predicted Probabilities
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[0])
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("True")
    axs[0].set_title("Confusion Matrix")

    # Bar chart of average predicted probabilities per class.
    avg_probs = np.mean(probs_np, axis=0)
    num_classes = avg_probs.shape[0]
    axs[1].bar(range(num_classes), avg_probs, color="mediumseagreen", edgecolor="black")
    axs[1].set_xlabel("Class")
    axs[1].set_ylabel("Average Predicted Probability")
    axs[1].set_title("Average Predicted Probabilities")
    axs[1].set_xticks(range(num_classes))
    axs[1].set_xticklabels([f"Class {i}" for i in range(num_classes)])

    plt.suptitle("Multiclass Classification Performance")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ----------------------------------------------------------------
# 4. Example Test
# ----------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from quantbayes.fake_data import generate_multiclass_classification_data

    df = generate_multiclass_classification_data(n_categorical=1, n_continuous=2)

    X, y = df.drop("target", axis=1), df["target"]
    X, y = torch.tensor(X.values, dtype=torch.float32), torch.tensor(
        y.values, dtype=torch.long
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X.clone(), y.clone(), test_size=0.2, random_state=24
    )

    num_classes = 3
    model = MLPClassifier(input_dim=X.shape[-1], num_classes=num_classes, hidden_dim=32)

    # Train the model
    model = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
    )
