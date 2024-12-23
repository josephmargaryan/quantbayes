import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model


def fit_volatility_model(data, model_type="GARCH", p=1, q=1):
    """
    Fits a volatility model to the provided financial time series.

    Parameters:
        data (array-like): Financial time series (e.g., returns).
        model_type (str): Type of model to use ('ARCH', 'GARCH', 'EGARCH', 'GARCH-M').
        p (int): Lag order for the model's AR term.
        q (int): Lag order for the model's MA term (only used in GARCH and GARCH-M).

    Returns:
        fitted_model: The fitted model object.
        forecast: Forecasted volatility.
    """
    if model_type not in ["ARCH", "GARCH", "EGARCH", "GARCH-M"]:
        raise ValueError(
            "Invalid model_type. Choose from 'ARCH', 'GARCH', 'EGARCH', 'GARCH-M'."
        )

    # Configure the model
    vol = "Garch" if model_type in ["GARCH", "GARCH-M"] else model_type
    dist = "Normal"

    model = arch_model(data, vol=vol, p=p, q=q, dist=dist, mean="Constant")

    if model_type == "GARCH-M":
        model = arch_model(data, vol="Garch", p=p, q=q, dist=dist, mean="GARCH-M")

    # Fit the model
    fitted_model = model.fit(disp="off")
    print(fitted_model.summary())

    # Forecast volatility
    forecast = fitted_model.forecast(horizon=10)
    volatility = np.sqrt(
        forecast.variance.values[-1, :]
    )  # Convert variance to standard deviation

    # Plot forecasted volatility
    plt.figure(figsize=(10, 6))
    plt.plot(volatility, marker="o", label="Forecasted Volatility")
    plt.title(f"Forecasted Volatility ({model_type})")
    plt.xlabel("Steps Ahead")
    plt.ylabel("Volatility")
    plt.legend()
    plt.show()

    return fitted_model, volatility


# Example usage
if __name__ == "__main__":
    # Generate synthetic financial returns data
    np.random.seed(42)
    returns = np.random.normal(0, 1, 1000)  # Simulated returns
    returns[500:] += np.random.normal(0, 0.5, 500)  # Add volatility clustering

    # Fit ARCH model
    print("\nFitting ARCH Model")
    fit_volatility_model(returns, model_type="ARCH", p=1)

    # Fit GARCH model
    print("\nFitting GARCH Model")
    fit_volatility_model(returns, model_type="GARCH", p=1, q=1)

    # Fit EGARCH model
    print("\nFitting EGARCH Model")
    fit_volatility_model(returns, model_type="EGARCH", p=1, q=1)

    # Fit GARCH-M model
    print("\nFitting GARCH-M Model")
    fit_volatility_model(returns, model_type="GARCH-M", p=1, q=1)
