import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from models import DMM
import pandas as pd
import numpy as np
from data import create_loaders
from train import train_epoch, eval_epoch
from visualization import plot_losses, visualize_predictions

# Generate synthetic sinusidual data
data = pd.DataFrame(
    {
        "feature1": np.sin(np.linspace(0, 100, 500)),
        "feature2": np.cos(np.linspace(0, 100, 500)),
        "target": np.sin(np.linspace(0, 100, 500)) + 0.1 * np.random.randn(500),
    }
)

# Create ranfom walk data usig brownian motion
np.random.seed(42)
T = 500
steps = np.random.normal(loc=0.0, scale=0.1, size=T)
random_walk = np.cumsum(steps)
data = pd.DataFrame(
    {
        "feature1": random_walk,  # Consider this as your main observable feature
        "feature2": np.sin(np.linspace(0, 10, T)),  # Another feature, optional
        "target": random_walk,  # You can treat target as the same as feature1 for simplicity
    }
)


# Normalize the data
data[["feature1", "feature2", "target"]] = (
    data[["feature1", "feature2", "target"]] - data.mean()
) / data.std()

# Create train and validation loaders
train_loader, val_loader = create_loaders(
    data, target_column="target", sequence_length=10, batch_size=10
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dmm = DMM(x_dim=2, z_dim=2).to(device)
    pyro.clear_param_store()
    optimizer = Adam({"lr": 1e-3})
    svi = SVI(dmm.model, dmm.guide, optimizer, loss=Trace_ELBO())

    num_epochs = 100
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(dmm, svi, train_loader, device)
        val_loss = eval_epoch(dmm, svi, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    # Plot losses after training
    plot_losses(train_losses, val_losses)

    # Visualize predictions vs ground truth on validation set
    visualize_predictions(dmm, val_loader, device, num_examples=3)
