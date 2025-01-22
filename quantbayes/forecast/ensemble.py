import logging
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def safe_slice(data, indices):
    """
    Slice data based on indices, supporting both pandas and NumPy arrays.

    Args:
        data (pd.DataFrame, pd.Series, or np.ndarray): The data to slice.
        indices (list or np.ndarray): Indices for slicing.

    Returns:
        Sliced data.
    """
    if isinstance(data, (np.ndarray, list)):
        return data[indices]
    elif hasattr(data, "iloc"):
        return data.iloc[indices]
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

class EnsembleModel:
    def __init__(self, models, n_splits=5):
        """
        Initialize the EnsembleModel class.

        Args:
            models (dict): Dictionary of model instances with model names as keys.
            n_splits (int): Number of splits for TimeSeriesSplit.
        """
        self.models = models
        self.n_splits = n_splits
        self.results = {"folds": [], "ensemble_rmse": []}
        self.predictions = []
        self.ground_truth = []

    def fit_predict(self, X, y):
        """
        Perform time series cross-validation and train the ensemble.

        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix.
            y (np.ndarray or pd.Series): Target variable.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold = 1

        self.predictions = []
        self.ground_truth = []

        progress_bar = tqdm(tscv.split(X), total=self.n_splits, desc="Folds")

        for train_index, test_index in progress_bar:
            X_train, X_test = safe_slice(X, train_index), safe_slice(X, test_index)
            y_train, y_test = safe_slice(y, train_index), safe_slice(y, test_index)

            predictions = []
            fold_results = {"fold": fold, "model_rmse": {}}

            for model_name, model in self.models.items():
                logging.info(f"Training model {model_name} on fold {fold}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                fold_results["model_rmse"][model_name] = rmse
                predictions.append(y_pred)

            # Compute ensemble predictions
            ensemble_pred = np.mean(predictions, axis=0)
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

            # Store fold results
            fold_results["ensemble_rmse"] = ensemble_rmse
            self.results["folds"].append(fold_results)
            self.results["ensemble_rmse"].append(ensemble_rmse)

            # Append test set predictions and ground truth for visualization
            self.predictions.extend(ensemble_pred)
            self.ground_truth.extend(y_test)

            logging.info(f"Fold {fold} completed. Ensemble RMSE = {ensemble_rmse:.4f}")
            progress_bar.set_postfix({"Fold RMSE": ensemble_rmse})
            fold += 1

        logging.info("Cross-validation completed.")

    def plot_predictions(self):
        """
        Visualize the actual vs. predicted values for the ensemble model (test sets only).
        """
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.ground_truth, label="Actual (Ground Truth)", alpha=0.7, color="blue"
        )
        plt.plot(
            self.predictions,
            label="Ensemble Prediction (Test Sets)",
            alpha=0.7,
            color="orange",
        )
        plt.legend()
        plt.title("Actual vs. Predicted Values (Test Sets Only)")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()

    def plot_model_performance(self):
        """
        Visualize RMSE performance of individual models and ensemble across folds.
        """
        model_names = list(self.models.keys())
        folds = range(1, len(self.results["folds"]) + 1)

        for model_name in model_names:
            model_rmse = [
                fold["model_rmse"][model_name] for fold in self.results["folds"]
            ]
            plt.plot(folds, model_rmse, label=f"{model_name} RMSE", marker="o")

        ensemble_rmse = self.results["ensemble_rmse"]
        plt.plot(
            folds,
            ensemble_rmse,
            label="Ensemble RMSE",
            marker="o",
            linestyle="--",
            color="black",
        )

        plt.title("Model Performance Across Folds")
        plt.xlabel("Fold")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)
        plt.show()

    def summary(self):
        """
        Print a summary of the RMSE results for each fold.
        """
        print("\nModel Performance by Fold:")
        for fold_result in self.results["folds"]:
            print(f"Fold {fold_result['fold']}:")
            for model_name, rmse in fold_result["model_rmse"].items():
                print(f"  {model_name}: RMSE = {rmse:.4f}")
            print(f"  Ensemble: RMSE = {fold_result['ensemble_rmse']:.4f}")
        print("\nOverall Ensemble RMSE:")
        print(f"  Mean: {np.mean(self.results['ensemble_rmse']):.4f}")
        print(f"  Std Dev: {np.std(self.results['ensemble_rmse']):.4f}")

    def inference(self, X_new, weights=None):
        """
        Make predictions on new data using the trained ensemble model.

        Args:
            X_new (pd.DataFrame, np.ndarray, or list): New feature matrix for prediction.
            weights (list or np.ndarray): Weights for each model during prediction.

        Returns:
            np.array: Ensemble predictions for the new data.
        """
        if not self.models:
            raise ValueError(
                "No models trained. Please train the ensemble before inference."
            )

        if weights is None:
            weights = np.ones(len(self.models)) / len(self.models)  # Equal weights by default

        if len(weights) != len(self.models):
            raise ValueError("Length of weights must match the number of models.")

        predictions = []

        for model_name, model in self.models.items():
            logging.info(f"Generating predictions with model {model_name}...")
            y_pred = model.predict(X_new)
            predictions.append(y_pred)

        # Compute weighted ensemble predictions
        predictions = np.array(predictions)  # Shape: (num_models, num_samples)
        ensemble_pred = np.average(predictions, axis=0, weights=weights)  # Weighted average

        return ensemble_pred
