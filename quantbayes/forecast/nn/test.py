import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from quantbayes.forecast.nn.base import (
    TimeSeriesTrainer,
    Visualizer,
)
from quantbayes.forecast.nn.models import *
from quantbayes.fake_data import create_synthetic_time_series
from quantbayes.forecast.nn.models.n_beats import NBeats


X_train, X_val, y_train, y_val = create_synthetic_time_series()
X_train, X_val, y_train, y_val = (
    np.array(X_train),
    np.array(X_val),
    np.array(y_train),
    np.array(y_val),
)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Instantiate model
model = NBeats(10)
trainer = TimeSeriesTrainer(model)
trainer.compile(
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    criterion=torch.nn.MSELoss(),
)


# Test: Training
def test_training():
    trainer.fit(train_loader, val_loader, num_epochs=100)


test_training()


# Test: Prediction with Uncertainty
def test_prediction():
    # Use validation data for testing
    X_test = X_val

    # Get predictions with uncertainty
    results = model.predict_with_uncertainty(X_test, n_samples=10)

    # Validate the output format and shape
    assert (
        "mean" in results and "uncertainty" in results
    ), "Prediction output format is incorrect"
    assert results["mean"].shape == (
        X_test.shape[0],
        1,
    ), "Prediction mean shape mismatch"
    assert results["uncertainty"].shape == (
        X_test.shape[0],
        1,
    ), "Prediction uncertainty shape mismatch"


test_prediction()


# Test: Visualization
def test_visualization():
    # Use validation data for testing
    X_test = X_val
    y_test = y_val

    # Get predictions with uncertainty
    results = model.predict_with_uncertainty(X_test, n_samples=10)

    # Visualize predictions vs actual targets
    Visualizer.visualize(
        y_train=y_train,
        y_test=y_test.squeeze(),
        predictions=results["mean"].squeeze(),
        uncertainties=results["uncertainty"].squeeze(),
    )


test_visualization()

print("All tests passed!")
