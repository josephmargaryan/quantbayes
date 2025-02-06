# image_segmentation_script_torch.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# 1. Simple U-Net for Binary Segmentation
# ----------------------------------------------------------------------------
class SimpleUNet(nn.Module):
    """
    A small 2-level U-Net for demonstration of binary segmentation.
    Input: (batch_size, 3, H, W)
    Output: (batch_size, 1, H, W) (logits for BCEWithLogitsLoss)
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Decoder
        self.up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 16 + 16 from skip
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = torch.relu(self.conv1(x))
        x1 = torch.relu(self.conv2(x1))
        x_pool = self.pool(x1)

        # Bottleneck
        x_bottleneck = torch.relu(self.conv3(x_pool))

        # Decoder
        x_up = self.up(x_bottleneck)  # (batch, 16, H/2, W/2)
        x_cat = torch.cat([x_up, x1], dim=1)  # Skip connection
        x_dec = torch.relu(self.conv4(x_cat))

        logits = self.out_conv(x_dec)  # (batch, 1, H, W)
        return logits


# ----------------------------------------------------------------------------
# 2. Training Function
# ----------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> nn.Module:
    """
    Train a simple U-Net with BCEWithLogitsLoss for binary segmentation.
    X_*: (N, 3, H, W), Y_*: (N, 1, H, W)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert data to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def batch_generator(X, Y, bs):
        n = len(X)
        indices = torch.randperm(n).to(device)
        for start in range(0, n, bs):
            end = start + bs
            idx = indices[start:end]
            yield X[idx], Y[idx]

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        for batch_x, batch_y in batch_generator(X_train_t, Y_train_t, batch_size):
            optimizer.zero_grad()
            logits = model(batch_x)  # (batch, 1, H, W)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            val_loss = criterion(logits_val, Y_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % max(1, (num_epochs // 5)) == 0:
            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Plot training & validation loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE With Logits Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


# ----------------------------------------------------------------------------
# 3. Deterministic Evaluation (Compute IoU)
# ----------------------------------------------------------------------------
def compute_iou(
    pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold=0.5
) -> float:
    """
    pred_mask, true_mask: shape (batch, H, W), values in [0,1]
    Threshold pred_mask, then compute intersection and union.
    """
    pred_bin = (pred_mask > threshold).float()
    intersection = (pred_bin * true_mask).sum().item()
    union = ((pred_bin + true_mask) > 0).sum().item()
    iou = intersection / (union + 1e-6)
    return iou


def evaluate_model(
    model: nn.Module,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    batch_size: int = 8,
) -> float:
    """
    Compute the average IoU across the entire dataset using a deterministic forward pass.
    """
    device = next(model.parameters()).device

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    def mini_batches(X, Y, bs):
        n = len(X)
        for i in range(0, n, bs):
            yield X[i : i + bs], Y[i : i + bs]

    ious = []
    model.eval()  # Ensure deterministic behavior (dropout is disabled)
    with torch.no_grad():
        for batch_x, batch_y in mini_batches(X_val_t, Y_val_t, batch_size):
            logits = model(batch_x)  # (b, 1, H, W)
            prob = torch.sigmoid(logits)  # (b, 1, H, W)
            # Remove channel dimension for IoU computation
            b, _, H, W = prob.shape
            prob_2d = prob.view(b, H, W)
            true_2d = batch_y.view(b, H, W)
            batch_iou = compute_iou(prob_2d, true_2d)
            ious.append(batch_iou)

    return float(np.mean(ious))


# ----------------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------------
def visualize_segmentation(
    model: nn.Module,
    X_samples: np.ndarray,
    Y_samples: np.ndarray,
    num_plots: int = 3,
    threshold: float = 0.5,
):
    """
    Plot side-by-side: input image, predicted mask, ground truth mask.
    X_samples: (N, 3, H, W), Y_samples: (N, 1, H, W)
    """
    device = next(model.parameters()).device

    idxs = np.random.choice(len(X_samples), size=num_plots, replace=False)
    plt.figure(figsize=(12, 4 * num_plots))

    for i, idx in enumerate(idxs):
        img = X_samples[idx]  # shape: (3, H, W)
        gt_mask = Y_samples[idx]  # shape: (1, H, W)

        model.eval()  # Deterministic inference
        with torch.no_grad():
            img_t = torch.tensor(img[None], dtype=torch.float32).to(device)
            logits = model(img_t)  # (1, 1, H, W)
            pred_prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W)
        pred_mask = (pred_prob > threshold).astype(np.float32)

        # Convert image to (H, W, 3) for plotting
        img_np = np.transpose(img, (1, 2, 0))

        plt.subplot(num_plots, 3, 3 * i + 1)
        plt.imshow(img_np)
        plt.axis("off")
        plt.title("Input Image")

        plt.subplot(num_plots, 3, 3 * i + 2)
        plt.imshow(pred_mask, cmap="gray")
        plt.axis("off")
        plt.title("Predicted Mask")

        plt.subplot(num_plots, 3, 3 * i + 3)
        plt.imshow(gt_mask[0], cmap="gray")
        plt.axis("off")
        plt.title("Ground Truth")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# 5. Example Test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data: 3-channel images, random "circle" masks
    np.random.seed(42)
    N = 50
    H, W = 64, 64
    C = 3
    X_all = np.random.rand(N, C, H, W).astype(np.float32)
    Y_all = np.zeros((N, 1, H, W), dtype=np.float32)

    # Create random circular masks
    for i in range(N):
        rr = np.random.randint(10, 20)
        center_r = np.random.randint(rr, H - rr)
        center_c = np.random.randint(rr, W - rr)
        y_grid, x_grid = np.ogrid[:H, :W]
        mask = (y_grid - center_r) ** 2 + (x_grid - center_c) ** 2 <= rr**2
        Y_all[i, 0, mask] = 1.0

    # Split the data
    train_size = int(0.8 * N)
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    Y_train, Y_val = Y_all[:train_size], Y_all[train_size:]

    # Define model
    model = SimpleUNet()

    # Train the model
    model = train_model(
        model,
        X_train,
        Y_train,
        X_val,
        Y_val,
        num_epochs=10,
        batch_size=8,
        learning_rate=1e-3,
    )

    # Evaluate deterministically
    iou_val = evaluate_model(model, X_val, Y_val, batch_size=8)
    print("Validation IoU:", iou_val)

    # Visualize segmentation results
    visualize_segmentation(model, X_val, Y_val, num_plots=3, threshold=0.5)
