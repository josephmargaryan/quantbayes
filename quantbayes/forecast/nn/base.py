from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


class BaseModel(ABC, torch.nn.Module):
    """
    Abstract base class for all models.

    Provides a structure for defining a forward pass in derived models.
    """

    def __init__(self):
        """
        Initialize the BaseModel.
        """
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        """
        Abstract method for the forward pass.

        :param x: torch.Tensor of shape (batch_size, input_dim)
            Input data for the model.
        :return: torch.Tensor of shape (batch_size, output_dim)
            Model predictions.
        """
        pass


class Visualizer:
    """
    A utility class for visualizing training, testing, and predictions with optional uncertainties.
    """

    @staticmethod
    def visualize(y_train, y_test, predictions, uncertainties=None):
        """
        Visualizes training targets, testing targets, predictions, and uncertainties.

        :param y_train: np.ndarray of shape (n_train,)
            Ground truth targets for the training set.
        :param y_test: np.ndarray of shape (n_test,)
            Ground truth targets for the test set.
        :param predictions: np.ndarray of shape (n_test,)
            Model predictions for the test set.
        :param uncertainties: np.ndarray of shape (n_test,), optional
            Uncertainty estimates for the predictions.
        :return: None
        """
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
    """
    A mixin class for enabling Monte Carlo-based prediction with uncertainty estimation.
    """

    def predict_with_uncertainty(self, X, n_samples=100, batch_size=32):
        """
        Predicts with uncertainty using Monte Carlo sampling.

        :param X: np.ndarray of shape (n_samples, input_dim)
            Input data for prediction.
        :param n_samples: int
            Number of Monte Carlo samples.
        :param batch_size: int
            Batch size for data loading.
        :return: dict
            A dictionary containing:
            - "mean": np.ndarray of shape (n_samples,)
                Mean predictions.
            - "uncertainty": np.ndarray of shape (n_samples,)
                Standard deviation of predictions.
        """
        self.train()  # Enable stochastic behavior (e.g., dropout)
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
    """
    A trainer class for time series models.

    Provides functionality for training, validation, and visualization of loss curves.
    """

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the trainer.

        :param model: BaseModel
            The model to be trained.
        :param device: str
            Device to use for training ("cuda" or "cpu").
        """
        self.model = model
        self.device = device
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.early_stopping_patience = 5

    def compile(self, optimizer, criterion, scheduler=None, early_stopping_patience=5):
        """
        Configures the trainer with optimizer, loss function, and scheduler.

        :param optimizer: torch.optim.Optimizer
            Optimizer for training the model.
        :param criterion: callable
            Loss function.
        :param scheduler: torch.optim.lr_scheduler, optional
            Learning rate scheduler.
        :param early_stopping_patience: int
            Number of epochs to wait for improvement before stopping.
        :return: None
        """
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience

    def fit(self, train_loader, val_loader, num_epochs=100):
        """
        Trains the model on the training set and validates on the validation set.

        :param train_loader: DataLoader
            DataLoader for the training set.
        :param val_loader: DataLoader
            DataLoader for the validation set.
        :param num_epochs: int
            Number of epochs to train the model.
        :return: None
        """
        self.model.to(self.device)
        best_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            # Training Phase
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                y_batch = y_batch.view(-1, 1)
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            # Validation Phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_batch = y_batch.view(-1, 1)
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    predictions = self.model(X_batch)
                    loss = self.criterion(predictions, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

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
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            label="Train Losses",
            marker="o",
        )
        plt.plot(
            range(1, len(val_losses) + 1), val_losses, label="Val Losses", marker="d"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()
