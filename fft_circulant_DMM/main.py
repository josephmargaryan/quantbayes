import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pandas as pd
import numpy as np

from data import create_loaders
from models import DMM
from train import train_epoch, eval_epoch
from inference import predict_with_uncertainty
from visualization import plot_loss, plot_predictions_with_uncertainty

if __name__ == "__main__":

    data = pd.DataFrame(
        {
            "feature1": np.sin(np.linspace(0, 100, 500)),
            "feature2": np.cos(np.linspace(0, 100, 500)),
            "target": np.sin(np.linspace(0, 100, 500)) + 0.1 * np.random.randn(500),
        }
    )

    # Normalize data
    data[["feature1", "feature2", "target"]] = (
        data[["feature1", "feature2", "target"]] - data.mean()
    ) / data.std()

    # Create loaders
    train_loader, val_loader = create_loaders(
        data, target_column="target", sequence_length=10, batch_size=10
    )

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

    # Plot loss curves
    plot_loss(train_losses, val_losses)

    # Predict with uncertainty
    gt, means, stds = predict_with_uncertainty(dmm, val_loader, device, num_samples=50)

    # Plot predictions with uncertainty
    plot_predictions_with_uncertainty(gt, means, stds, num_examples=3)
