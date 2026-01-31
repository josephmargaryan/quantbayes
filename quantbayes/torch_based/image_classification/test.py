# image_classification_script_torch.py

import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------------------------------------------------------
# 1. Simple CNN Model (Replace with your actual model if desired)
# ----------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(
            64 * 8 * 8, 128
        )  # if input is 32x32 -> after 2 pools -> 8x8
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 3, height, width)
        output: (batch_size, num_classes)
        """
        x = torch.relu(self.conv1(x))  # (batch_size, 32, h, w)
        x = self.pool(x)  # (batch_size, 32, h/2, w/2)
        x = torch.relu(self.conv2(x))  # (batch_size, 64, h/2, w/2)
        x = self.pool(x)  # (batch_size, 64, h/4, w/4)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))  # (batch_size, 128)
        x = self.dropout(x)  # dropout (only active during training)
        logits = self.fc2(x)  # (batch_size, num_classes)
        return logits


# ----------------------------------------------------------------------------
# 2. Training Function
# ----------------------------------------------------------------------------
def train_model(
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
    Train a classification model using cross-entropy loss.
    X_*: (N, 3, H, W)  y_*: (N,) integer class labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert data to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Data generator
    def batch_generator(X, y, bs):
        n = len(X)
        indices = torch.randperm(n).to(device)
        for start in range(0, n, bs):
            end = start + bs
            batch_idx = indices[start:end]
            yield X[batch_idx], y[batch_idx]

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        # Training loop
        for batch_x, batch_y in batch_generator(X_train_t, y_train_t, batch_size):
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            val_loss = criterion(logits_val, y_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % max(1, (num_epochs // 5)) == 0:
            print(
                f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Plot train/val losses
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


# ----------------------------------------------------------------------------
# 3. Deterministic Evaluation (Accuracy)
# ----------------------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
) -> float:
    """
    Evaluate classification accuracy using a deterministic forward pass.
    """
    device = next(model.parameters()).device

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    # Set the model to evaluation mode to disable dropout and other training behaviors.
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(X_val_t), batch_size):
            batch_x = X_val_t[i : i + batch_size]
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=-1)
            predictions.append(preds.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    accuracy = (predictions == y_val).mean()
    return float(accuracy)


# ----------------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------------
def visualize_predictions(
    model: nn.Module,
    X_samples: np.ndarray,
    y_samples: np.ndarray,
    class_names: List[str],
    num_plots: int = 5,
    n_cols: int = 3,  # you can set the number of columns here
):
    """
    Display `num_plots` images with predicted vs. ground-truth labels in a grid layout.
    X_samples: (N, 3, H, W)
    """
    device = next(model.parameters()).device

    # Choose a set of random indices
    idxs = np.random.choice(len(X_samples), num_plots, replace=False)

    # Determine grid dimensions
    n_rows = math.ceil(num_plots / n_cols)
    plt.figure(figsize=(n_cols * 4, n_rows * 4))  # Adjust figure size as needed

    for i, idx in enumerate(idxs):
        img_t = torch.tensor(X_samples[idx : idx + 1], dtype=torch.float32).to(device)
        model.eval()  # Ensure deterministic behavior during inference.
        with torch.no_grad():
            logits = model(img_t)
        pred_label = int(torch.argmax(logits, dim=-1)[0].cpu().numpy())
        gt_label = int(y_samples[idx])

        # Convert image to (H, W, 3) for plotting.
        img_np = X_samples[idx].transpose(1, 2, 0)

        # Create subplot at the appropriate grid position.
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img_np, interpolation="nearest")
        plt.axis("off")
        plt.title(f"Predicted: {class_names[pred_label]}\nGT: {class_names[gt_label]}")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# 5. Example Test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data: 32x32 images, 2 classes.
    np.random.seed(42)
    N = 1000
    H, W = 32, 32
    C = 3
    X_all = np.random.rand(N, C, H, W).astype(np.float32)
    y_all = np.random.randint(low=0, high=2, size=(N,)).astype(np.int64)

    # Split data.
    train_size = int(0.8 * N)
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    y_train, y_val = y_all[:train_size], y_all[train_size:]

    # Define the model.
    model = SimpleCNN(num_classes=2)

    # Train the model.
    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-3,
    )

    # Evaluate deterministically.
    acc = evaluate_model(model, X_val, y_val, batch_size=32)
    print("Validation Accuracy:", acc)

    # Visualize predictions.
    class_names = ["Class 0", "Class 1"]
    visualize_predictions(model, X_val, y_val, class_names, num_plots=5)
