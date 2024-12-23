from get_data import StockData
from ensemble import EnsembleModel
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import logging
from visualizations import StockPlotter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    # Initialize StockData
    stock_data = StockData(ticker="AAPL", start_date="2023-01-01", interval="1h")

    # Get ML-ready data
    X, y = stock_data.get_ml_ready_data()

    stock_plotter = StockPlotter(X=X, y=y, ticker="AAPL")

    # Plot stock prices
    stock_plotter.plot_last_week()
    stock_plotter.plot_last_month()
    stock_plotter.plot_last_six_months()

    models = {
        "RandomForest": RandomForestRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor(),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(),
        "CatBoostRegressor": CatBoostRegressor(silent=True),
        "LGBMRegressor": LGBMRegressor(),
    }

    ensemble_model = EnsembleModel(models, n_splits=5)

    ensemble_model.fit_predict(X, y)
    ensemble_model.plot_predictions()
    ensemble_model.plot_model_performance()
    ensemble_model.summary()


if __name__ == "__main__":
    main()
