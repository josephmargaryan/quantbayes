import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

class RLCNNClassifier(nn.Module):
    """
    A very simple CNN-based classifier to illustrate a RL approach to image classification.
    """
    def __init__(self, num_channels=1, num_classes=10):
        super(RLCNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*8*8, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (batch_size, num_channels, height, width)
        """
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc1(x)           # shape: (batch_size, num_classes)
        probs = self.softmax(logits)   # shape: (batch_size, num_classes)
        return probs

def generate_random_image_data(num_samples=256, num_classes=10, image_size=(1, 32, 32), seed=42):
    """
    Generate random 'images' and random labels.
    """
    np.random.seed(seed)
    images = np.random.randn(num_samples, *image_size).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_samples,))
    X = torch.tensor(images, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y

def train_rl_image_classifier(model, data_loader, optimizer, num_epochs=5):
    """
    Simple policy gradient approach for classification.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for X_batch, y_batch in data_loader:
            probs = model(X_batch)               # shape: (batch_size, num_classes)
            dist = Categorical(probs)
            actions = dist.sample()              # (batch_size,)
            log_probs = dist.log_prob(actions)   # (batch_size,)

            rewards = torch.where(actions == y_batch, 1.0, -1.0)
            loss = -(log_probs * rewards).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss / len(data_loader):.4f}")

def visualize_image_classification(X_test, y_test, y_pred, num_images=5):
    """
    Visualizes image classification results.

    Args:
        X_test (torch.Tensor): Test images (batch_size, channels, H, W).
        y_test (torch.Tensor): Ground truth labels (batch_size,).
        y_pred (torch.Tensor): Model predictions (batch_size,).
        num_images (int): Number of images to display.

    Returns:
        None
    """
    num_images = min(num_images, len(X_test))
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))

    for i in range(num_images):
        image = X_test[i].squeeze(0).numpy()  # Convert to 2D grayscale if single channel
        true_label = y_test[i].item()
        predicted_label = y_pred[i].item()

        ax = axes[i]
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Pred: {predicted_label}\nGT: {true_label}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_samples = 256
    num_classes = 10
    batch_size = 32
    lr = 0.001
    epochs = 5

    # Generate random image data
    X, y = generate_random_image_data(num_samples=num_samples, num_classes=num_classes)
    dataset = data.TensorDataset(X, y)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model & optimizer
    model = RLCNNClassifier(num_channels=1, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    train_rl_image_classifier(model, data_loader, optimizer, epochs)

    # Quick check on 5 test samples
    model.eval()
    with torch.no_grad():
        sample_X = X[:5]  # Take first 5 test samples
        sample_Y = y[:5]
        sample_probs = model(sample_X)
        sample_preds = torch.argmax(sample_probs, dim=-1)

        print("Sample Predictions:", sample_preds.tolist())
        print("Ground Truth:", sample_Y.tolist())

        # Visualize
        visualize_image_classification(sample_X, sample_Y, sample_preds, num_images=5)
