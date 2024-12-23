import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from pykalman import KalmanFilter

logging.basicConfig(level=logging.INFO)


class StatisticalModels:
    def __init__(self, data):
        self.data = data

    def fit_arima(self, order=(1, 1, 1)):
        logging.info("Fitting ARIMA model...")
        model = ARIMA(self.data, order=order)
        self.arima_result = model.fit()
        return self.arima_result

    def fit_garch(self, p=1, q=1):
        logging.info("Fitting GARCH model...")
        valid_data = self.data.replace([np.inf, -np.inf], np.nan).dropna()
        if valid_data.empty:
            raise ValueError("No valid data points available for GARCH model.")
        model = arch_model(valid_data, vol="Garch", p=p, q=q)
        self.garch_result = model.fit(disp="off")
        return self.garch_result

    def fit_kalman_filter(self):
        logging.info("Fitting Kalman Filter...")
        valid_data = self.data.replace([np.inf, -np.inf], np.nan).dropna()
        if valid_data.empty:
            raise ValueError("No valid data points available for Kalman Filter.")
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        kf = kf.em(valid_data, n_iter=10)
        (filtered_state_means, _) = kf.filter(valid_data)
        self.kalman_result = filtered_state_means
        return self.kalman_result


if __name__ == "__main__":
    import yfinance as yf

    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    data = stock.history(start="2020-01-01", interval="1d")[["Close"]]
    data = data.asfreq("B")

    stats = StatisticalModels(data["Close"])
    arima_result = stats.fit_arima()
    logging.info(f"ARIMA Summary: \n{arima_result.summary()}")

    garch_result = stats.fit_garch()
    logging.info(f"GARCH Summary: \n{garch_result.summary()}")

    kalman_result = stats.fit_kalman_filter()
    logging.info(f"Kalman Filter Result: {kalman_result[:5]}")
