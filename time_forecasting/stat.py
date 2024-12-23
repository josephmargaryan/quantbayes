import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

# Generate synthetic multivariate financial data
np.random.seed(42)
data = np.cumsum(np.random.normal(size=(200, 3)), axis=0)  # Random walk for 3 assets
df = pd.DataFrame(data, columns=["Asset1", "Asset2", "Asset3"])

# Split into train and test
train, test = df[:150], df[150:]

# Fit VAR model
model = VAR(train)
lag_order = model.select_order(maxlags=10).selected_orders["aic"]
model_fitted = model.fit(lag_order)

# Forecast
forecast = model_fitted.forecast(train.values[-lag_order:], steps=len(test))
forecast_df = pd.DataFrame(forecast, index=test.index, columns=test.columns)

# Plot results
plt.figure(figsize=(10, 6))
for col in df.columns:
    plt.plot(df[col], label=f"Actual {col}")
    plt.plot(forecast_df[col], linestyle="--", label=f"Forecast {col}")
plt.legend()
plt.title("VAR Model Forecast")
plt.show()
