import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
        self.predictions = []  # Store ensemble predictions for visualization
        self.ground_truth = []  # Store true values for visualization

    def fit_predict(self, X, y):
        """
        Perform time series cross-validation and train the ensemble.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold = 1

        self.predictions = []
        self.ground_truth = []

        progress_bar = tqdm(tscv.split(X), total=self.n_splits, desc="Folds")

        for train_index, test_index in progress_bar:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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
            self.predictions.extend(ensemble_pred)  # Only test set predictions
            self.ground_truth.extend(y_test.values)  # Only test set ground truth

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
        plt.title("Actual vs. Predicted Closing Prices (Test Sets Only)")
        plt.xlabel("Time")
        plt.ylabel("Price")
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

    def inference(self, X_new):
        """
        Make predictions on new data using the trained ensemble model.

        Args:
            X_new (pd.DataFrame): New feature matrix for prediction.

        Returns:
            np.array: Ensemble predictions for the new data.
        """
        if not self.models:
            raise ValueError(
                "No models trained. Please train the ensemble before inference."
            )

        predictions = []

        for model_name, model in self.models.items():
            logging.info(f"Generating predictions with model {model_name}...")
            predictions.append(model.predict(X_new))

        # Compute ensemble predictions (mean of individual model predictions)
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred


def generate_synthetic_stock_data(num_stocks=10, num_days=500, seed=42):
    """
    Generate synthetic stock price data using Geometric Brownian Motion.

    Args:
        num_stocks (int): Number of stocks to simulate.
        num_days (int): Number of days to simulate.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing synthetic stock price data.
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=num_days)
    stock_data = {}

    for i in range(num_stocks):
        stock_name = f"Stock_{i+1}"
        initial_price = np.random.uniform(50, 150)  # Random initial price
        mu = np.random.uniform(0.05, 0.15)  # Random drift
        sigma = np.random.uniform(0.1, 0.3)  # Random volatility
        prices = [initial_price]

        for _ in range(1, num_days):
            dt = 1 / 252  # Assume 252 trading days in a year
            shock = np.random.normal(0, 1)
            dS = mu * prices[-1] * dt + sigma * prices[-1] * np.sqrt(dt) * shock
            prices.append(prices[-1] + dS)

        stock_data[stock_name] = prices

    return pd.DataFrame(stock_data, index=dates)


if __name__ == "__main__":
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    synthetic_data = generate_synthetic_stock_data(num_stocks=10, num_days=500)
    synthetic_data.plot(figsize=(12, 6), legend=False)
    plt.title("Synthetic Stock Price Data")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

    # Use one stock for simplicity
    stock_prices = synthetic_data["Stock_1"]
    lagged_features = pd.DataFrame(
        {
            "Lag_1": stock_prices.shift(1),
            "Lag_2": stock_prices.shift(2),
            "Lag_3": stock_prices.shift(3),
        }
    ).dropna()
    target = stock_prices.shift(-1).dropna()

    X = lagged_features.iloc[:-1]
    y = target.iloc[:-1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

    # Define models
    models = {
        "CatBoostRegressor": CatBoostRegressor(silent=True),
        "LGBMRegressor": LGBMRegressor(),
        "XGBoost": XGBRegressor(),
    }

    # Initialize EnsembleModel
    ensemble_model = EnsembleModel(models, n_splits=5)

    ensemble_model.fit_predict(X, y)

    # Example: Plot predictions and model performance
    ensemble_model.plot_predictions()
    ensemble_model.plot_model_performance()
    ensemble_model.summary()

    # Example: Inference on new data (replace X_new with your actual new dataset)
    # X_new = ...
    # predictions = ensemble_model.inference(X_new)
    # print("Predictions:", predictions)
