import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

class RLSegmentationModel(nn.Module):
    """
    A minimal segmentation model that outputs pixel-wise classes (foreground vs background).
    """
    def __init__(self, in_channels=1, out_channels=2):
        super(RLSegmentationModel, self).__init__()
        # Example: A small conv net producing a 2-class output for each pixel
        self.conv1 = nn.Conv2d(in_channels, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, out_channels, 1)  # direct 1x1 conv for final
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x: (batch_size, in_channels, H, W)
        returns: probability distribution (batch_size, out_channels, H, W)
        """
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        logits = self.conv3(x)  # shape: (batch_size, 2, H, W)
        probs = self.softmax(logits)  # shape: (batch_size, 2, H, W)
        return probs

def generate_random_segmentation_data(num_samples=32, image_size=(1, 16, 16), seed=42):
    """
    Generate random images (noise) and random segmentation masks (0 or 1 per pixel).
    """
    np.random.seed(seed)
    images = np.random.randn(num_samples, *image_size).astype(np.float32)
    seg_masks = np.random.randint(0, 2, size=(num_samples, image_size[1], image_size[2])) # shape: (num_samples, H, W)
    X = torch.tensor(images, dtype=torch.float32)
    Y = torch.tensor(seg_masks, dtype=torch.long)  # each pixel is 0 or 1
    return X, Y

def train_rl_segmentation(model, data_loader, optimizer, num_epochs=5):
    """
    For each pixel, the model picks a class (0 or 1).
    Reward is +1 if correct, -1 if incorrect.
    We flatten the spatial dimensions and treat them as multiple independent actions.
    This is simplistic and can be slow for large images.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for X_batch, Y_batch in data_loader:
            # X_batch shape: (batch_size, 1, H, W)
            # Y_batch shape: (batch_size, H, W)

            probs = model(X_batch)  # (batch_size, 2, H, W)
            # We'll flatten for a simpler policy gradient approach
            batch_size, num_classes, H, W = probs.shape
            probs_2d = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)  # (batch_size*H*W, 2)
            labels_1d = Y_batch.view(-1)                                   # (batch_size*H*W,)

            dist = Categorical(probs_2d)
            actions = dist.sample()               # (batch_size*H*W,)
            log_probs = dist.log_prob(actions)    # (batch_size*H*W,)

            rewards = torch.where(actions == labels_1d, 1.0, -1.0)
            loss = -(log_probs * rewards).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss / len(data_loader):.4f}")

def visualize_segmentation_results(X_test, Y_test, Y_pred, num_images=5):
    """
    Visualizes image segmentation results side by side.

    Args:
        X_test (torch.Tensor): Test images (batch_size, channels, H, W).
        Y_test (torch.Tensor): Ground truth segmentation masks (batch_size, H, W).
        Y_pred (torch.Tensor): Model predicted segmentation masks (batch_size, H, W).
        num_images (int): Number of images to display.

    Returns:
        None
    """
    num_images = min(num_images, len(X_test))
    fig, axes = plt.subplots(num_images, 3, figsize=(9, num_images * 3))

    for i in range(num_images):
        image = X_test[i].squeeze(0).numpy()  # Convert to 2D grayscale if single channel
        gt_mask = Y_test[i].numpy()           # Ground truth mask
        pred_mask = Y_pred[i].numpy()         # Predicted mask

        # Convert prediction to binary mask (0 or 1) using argmax
        pred_mask = np.argmax(pred_mask, axis=0)

        # Plot original image
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        # Plot ground truth segmentation
        axes[i, 1].imshow(gt_mask, cmap="jet", alpha=0.7)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        # Plot predicted segmentation
        axes[i, 2].imshow(pred_mask, cmap="jet", alpha=0.7)
        axes[i, 2].set_title("Predicted Segmentation")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    batch_size = 4
    epochs = 5
    lr = 0.001

    # Generate data
    X, Y = generate_random_segmentation_data(num_samples=16, image_size=(1,16,16))
    dataset = torch.utils.data.TensorDataset(X, Y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model & optimizer
    model = RLSegmentationModel(in_channels=1, out_channels=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    train_rl_segmentation(model, data_loader, optimizer, num_epochs=epochs)

    # Quick check on 1 batch
    model.eval()
    with torch.no_grad():
        sample_X, sample_Y = next(iter(data_loader))
        sample_probs = model(sample_X)
        print("Sample output shape:", sample_probs.shape)  # Should be (B, 2, H, W)
        # You could further analyze or visualize the predicted segmentation here
        # Convert softmax probabilities to segmentation class labels
        sample_preds = torch.argmax(sample_probs, dim=1)  # (batch_size, H, W)

        # Call the visualization function
        visualize_segmentation_results(sample_X, sample_Y, sample_probs, num_images=5)

