import statsmodels.api as sm
import pandas as pd

# Example: Effect of interest rates on stock returns
data = pd.DataFrame(
    {
        "interest_rate": [0.03, 0.05, 0.02, 0.04, 0.01],
        "stock_returns": [0.02, 0.01, 0.03, -0.01, 0.04],
        "gdp_growth": [2.5, 2.0, 3.0, 2.8, 3.2],  # Confounder
    }
)

X = sm.add_constant(data[["interest_rate", "gdp_growth"]])
y = data["stock_returns"]

model = sm.OLS(y, X).fit()
print(model.summary())

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Example: Treatment is a new trading strategy, outcome is profit
data = pd.DataFrame(
    {
        "treatment": [1, 0, 1, 0, 1],
        "profit": [200, 150, 300, 100, 250],
        "experience": [5, 2, 8, 1, 7],  # Confounder
        "capital": [50000, 40000, 60000, 35000, 55000],  # Confounder
    }
)

# Estimate propensity scores
X = data[["experience", "capital"]]
y = data["treatment"]

log_reg = LogisticRegression()
data["propensity_score"] = log_reg.fit(X, y).predict_proba(X)[:, 1]

# Match treated and control groups
treated = data[data["treatment"] == 1]
control = data[data["treatment"] == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["propensity_score"]])
matches = nn.kneighbors(treated[["propensity_score"]], return_distance=False)

# Compute average treatment effect (ATE)
ate = (
    treated["profit"].values - control.iloc[matches.flatten()]["profit"].values
).mean()
print(f"Average Treatment Effect (ATE): {ate}")
