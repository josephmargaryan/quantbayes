# XGBoost Regression on Quasars Dataset
# =====================================
# This script performs the following steps:
# 1. Load the dataset (quasars.csv) and split it into training (80%) and test (20%) sets.
# 2. Further split the training set into a sub-training set (90% of training) and a hold-out validation set (10% of training).
# 3. Train an XGBoost regressor with the specified parameters (colsample_bytree=0.5, learning_rate=0.1, max_depth=4, reg_lambda=1, n_estimators=500),
#    monitoring training and validation RMSE per iteration, and plot both curves.
# 4. Evaluate the fitted model on the test set (compute RMSE and R²).
# 5. Perform a grid search (3-fold CV on the training set) over an extended parameter grid to find the best hyperparameters (minimizing RMSE).
# 6. Refit XGBoost on the entire training set with the best parameters and evaluate on the test set (RMSE and R²).
# 7. Fit a 5-nearest-neighbors regressor on the training set as a baseline and evaluate it on the test set (RMSE and R²).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor

# 1. Load data and initial train/test split
# -----------------------------------------
# Adjust the path to 'quasars.csv' as needed; here we assume it's inside the 'MLB7' folder.
df = pd.read_csv("MLB7/quasars.csv")

# The first 10 columns are features (X), the last column is the target (y).
X = df.iloc[:, :-1].values  # shape (n_samples, 10)
y = df.iloc[:, -1].values  # shape (n_samples,)

# Split into training (80%) and test (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 2. Create a hold-out validation set (10% of the training data)
# ---------------------------------------------------------------
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.10, random_state=42
)
# Now:
# - X_train_sub / y_train_sub ≈ 72% of the original data
# - X_val / y_val         ≈  8% of the original data
# - X_test / y_test       = 20% of the original data

# 3. Train XGBoost with specified parameters and monitor RMSE
# ------------------------------------------------------------
xgb_params = {
    "colsample_bytree": 0.5,
    "learning_rate": 0.1,
    "max_depth": 4,
    "reg_lambda": 1,
    "n_estimators": 500,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",  # move eval_metric into the constructor
    "random_state": 42,
}

xgb_model = XGBRegressor(**xgb_params)

# Provide eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)] to monitor RMSE
xgb_model.fit(
    X_train_sub,
    y_train_sub,
    eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)],
    verbose=False,
)

# Extract the recorded RMSE for each boosting iteration
evals_result = xgb_model.evals_result()
train_rmse = evals_result["validation_0"]["rmse"]
val_rmse = evals_result["validation_1"]["rmse"]

# Plot training and validation RMSE vs. boosting iterations
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_rmse) + 1), train_rmse, label="Train RMSE")
plt.plot(range(1, len(val_rmse) + 1), val_rmse, label="Validation RMSE")
plt.xlabel("Boosting Iteration")
plt.ylabel("RMSE")
plt.title("XGBoost Training vs. Validation RMSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Evaluate the fitted model on the test set
# ---------------------------------------------
y_pred_test = xgb_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print(f"XGBoost (fixed params) Test RMSE: {test_rmse:.4f}")
print(f"XGBoost (fixed params) Test R²:   {test_r2:.4f}")

# 5. Grid search for hyperparameter tuning (3-fold CV on training set)
# ---------------------------------------------------------------------
# We'll search over a moderately sized grid that extends the original parameters.
param_grid = {
    "colsample_bytree": [0.5, 0.7],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4, 5],
    "reg_lambda": [1, 5],
    "n_estimators": [200, 500],
}

xgb_for_grid = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    # GridSearchCV will set colsample_bytree, learning_rate, max_depth, reg_lambda, n_estimators
)

grid_search = GridSearchCV(
    estimator=xgb_for_grid,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",  # minimize MSE
    cv=3,
    verbose=1,
    n_jobs=-1,
)

# Run grid search on the full training set (X_train_full / y_train_full)
grid_search.fit(X_train_full, y_train_full)

print("Best parameters found by grid search:")
print(grid_search.best_params_)

# Convert the best (negative) MSE back to RMSE:
best_mse = -grid_search.best_score_
best_rmse_cv = np.sqrt(best_mse)
print(f"Best CV RMSE (3-fold) from grid search: {best_rmse_cv:.4f}")

# 6. Refit XGBoost on all training data with the best parameters
# --------------------------------------------------------------
best_params = grid_search.best_params_

xgb_best = XGBRegressor(**best_params, objective="reg:squarederror", random_state=42)

xgb_best.fit(X_train_full, y_train_full)

# Evaluate on the test set
y_pred_test_best = xgb_best.predict(X_test)
test_rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_test_best))
test_r2_best = r2_score(y_test, y_pred_test_best)

print(f"XGBoost (best params) Test RMSE: {test_rmse_best:.4f}")
print(f"XGBoost (best params) Test R²:   {test_r2_best:.4f}")

# 7. Baseline: 5-Nearest-Neighbors regression on the same training data
# ----------------------------------------------------------------------
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_full, y_train_full)

y_pred_knn = knn.predict(X_test)
knn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))
knn_r2 = r2_score(y_test, y_pred_knn)

print(f"KNN (k=5) Test RMSE: {knn_rmse:.4f}")
print(f"KNN (k=5) Test R²:   {knn_r2:.4f}")
