import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from time_forecast.nn.base import (
    BaseModel,
    MonteCarloMixin,
    TimeSeriesTrainer,
    Visualizer,
)


# Import modules from the refactored library
class MockModel(BaseModel, MonteCarloMixin):
    def __init__(self, input_dim, model_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(input_dim, model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, output_dim),
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        # x has shape (N, seq_len, D)
        x = torch.mean(x, dim=1)  # Reduce sequence dimension (mean pooling)
        return self.linear(x)  # Output shape: (N, output_dim)


# Generate Synthetic Time Series Data
def generate_synthetic_data(num_samples=1000, seq_len=10, input_dim=1):
    np.random.seed(42)
    X = np.random.rand(num_samples, seq_len, input_dim)
    y = X.mean(axis=1) + 0.1 * np.random.randn(num_samples, 1)
    return X, y


# Prepare data
X, y = generate_synthetic_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
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
model = MockModel(input_dim=1, model_dim=16, output_dim=1)
trainer = TimeSeriesTrainer(model)
trainer.compile(
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    criterion=torch.nn.MSELoss(),
)


# Test: Training
def test_training():
    trainer.fit(train_loader, val_loader, num_epochs=5)
    assert (
        model.linear[0].weight.grad is not None
    ), "Model weights not updated during training"


test_training()


# Test: Prediction with Uncertainty
def test_prediction():
    X_test = np.random.rand(50, 10, 1)  # Random test data
    results = model.predict_with_uncertainty(X_test, n_samples=10)
    assert (
        "mean" in results and "uncertainty" in results
    ), "Prediction output format is incorrect"
    assert results["mean"].shape == (50, 1), "Prediction mean shape mismatch"
    assert results["uncertainty"].shape == (
        50,
        1,
    ), "Prediction uncertainty shape mismatch"


test_prediction()


# Test: Visualization
def test_visualization():
    X_test = np.random.rand(50, 10, 1)  # Random test data
    y_test = X_test.mean(axis=1) + 0.1 * np.random.randn(50, 1)
    results = model.predict_with_uncertainty(X_test, n_samples=10)
    Visualizer.visualize(
        y_train=y_train[:100],
        y_test=y_test.squeeze(),
        predictions=results["mean"].squeeze(),
        uncertainties=results["uncertainty"].squeeze(),
    )


test_visualization()

print("All tests passed!")
