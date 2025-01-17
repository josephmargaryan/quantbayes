from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader


class BaseModel(ABC, torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        """Define the forward pass."""
        pass


class Visualizer:
    @staticmethod
    def visualize(y_train, y_test, predictions, uncertainties=None):
        train_range = range(len(y_train))
        test_range = range(len(y_train), len(y_train) + len(y_test))

        plt.figure(figsize=(12, 6))
        plt.plot(train_range, y_train, label="Train Targets")
        plt.plot(test_range, y_test, label="Test Targets")
        plt.plot(test_range, predictions, label="Predictions")

        if uncertainties is not None:
            plt.fill_between(
                test_range,
                predictions - 2 * uncertainties,
                predictions + 2 * uncertainties,
                color="gray",
                alpha=0.3,
                label="Uncertainty (±2 STD)",
            )

        plt.legend()
        plt.show()


class MonteCarloMixin:
    def predict_with_uncertainty(self, X, n_samples=100, batch_size=32):
        self.train()  # Enable stochastic behavior (dropout)
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size)

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                batch_preds = []
                for x_batch in loader:
                    x_batch = x_batch[0].to(self.device)
                    batch_preds.append(self(x_batch).cpu().numpy())
                predictions.append(np.concatenate(batch_preds, axis=0))

        predictions = np.array(predictions)
        mean_prediction = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        return {"mean": mean_prediction, "uncertainty": uncertainty}


class TimeSeriesTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.early_stopping_patience = 5

    def compile(self, optimizer, criterion, scheduler=None, early_stopping_patience=5):
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience

    def fit(self, train_loader, val_loader, num_epochs=100):
        self.model.to(self.device)
        best_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training Phase
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation Phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    predictions = self.model(X_batch)
                    loss = self.criterion(predictions, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(val_loader.dataset)

            if self.scheduler:
                self.scheduler.step()

            # Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        self.model.load_state_dict(best_model_state)
