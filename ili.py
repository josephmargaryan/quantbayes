# ILI_trends exercises and solutions

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

# Adjust BASE_DIR or CSV_PATH as needed
BASE_DIR = "/Users/josephmargaryan/Downloads"
CSV_PATH = os.path.join(BASE_DIR, "ILI_trends.csv")

df = pd.read_csv(CSV_PATH)

print("Raw data:")
print(df.head())
print(df.dtypes)
print("Shape:", df.shape)

print("\n" + "=" * 80 + "\n")

# Exercise 1:
"""
Description:
Load the ILI_trends.csv file, convert the 'date' column to a proper datetime
type, sort the rows by date, and set 'date' as the DataFrame index. Then print
a concise summary of the dataset and the first few rows.
"""
# Answer:

df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
df = df.sort_values("date").set_index("date")

print("After parsing dates and setting index:")
print(df.info())
print(df.head())

print("\n" + "=" * 80 + "\n")

# Exercise 2:
"""
Description:
Create simple exploratory plots for the time series. Plot the ILI time series
over the full period, and overlay a few selected search query series
(e.g. 'flu', 'fever', 'cough') on a separate plot to visually inspect how they
relate to ILI. Also print basic descriptive statistics for the ILI variable.
"""
# Answer:

plt.figure()
df["ILI"].plot()
plt.title("Influenza-like illness (ILI) over time")
plt.xlabel("Date")
plt.ylabel("ILI")
plt.tight_layout()
plt.show()

plt.figure()
selected_terms = ["flu", "fever", "cough"]
df[selected_terms].plot()
plt.title("Selected Google search queries over time")
plt.xlabel("Date")
plt.ylabel("Search frequency (index)")
plt.tight_layout()
plt.show()

print("Descriptive statistics for ILI:")
print(df["ILI"].describe())

print("\n" + "=" * 80 + "\n")

# Exercise 3:
"""
Description:
Compute the Pearson correlation between ILI and each search query using the
same week values. List the top 15 predictors most strongly correlated with ILI.
"""
# Answer:

corr_with_ili = df.corr(numeric_only=True)["ILI"].sort_values(ascending=False)
print("Top 15 same-week correlations with ILI:")
print(corr_with_ili.head(15))

print("\n" + "=" * 80 + "\n")

# Exercise 4:
"""
Description:
Investigate whether some search queries lead ILI by a few weeks. For the top 5
non-ILI variables from Exercise 3, compute the correlation between ILI_t and
query_{t-k} for lags k = 0, 1, 2, 3 weeks. Report the correlations for each
lag and each term.
"""
# Answer:

max_lag = 3
top_terms = corr_with_ili.index[1:6]  # skip 'ILI' itself

for term in top_terms:
    print(f"Lagged correlations for term: {term}")
    for k in range(max_lag + 1):
        r = df["ILI"].corr(df[term].shift(k))
        print(f"  lag {k} weeks: corr = {r:.3f}")
    print()

print("\n" + "=" * 80 + "\n")

# Exercise 5:
"""
Description:
Create a time-based train/validation/test split of the data. Use the first
60% of observations for training, the next 20% for validation, and the final
20% for testing. Implement a simple persistence baseline that predicts the
next week's ILI as the previous week's ILI, and evaluate this baseline on the
test set using MAE and RMSE.
"""
# Answer:

target_col = "ILI"
y = df[target_col]
X = df.drop(columns=[target_col])

n = len(df)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

# Persistence baseline on the test set: y_hat_t = y_{t-1}
y_pred_persist_test = y.shift(1).iloc[val_end:]  # aligned with y_test

mae_persist = mean_absolute_error(y_test, y_pred_persist_test)
rmse_persist = root_mean_squared_error(y_test, y_pred_persist_test)

print("Persistence baseline on TEST:")
print("  MAE :", mae_persist)
print("  RMSE:", rmse_persist)

print("\n" + "=" * 80 + "\n")

# Exercise 6:
"""
Description:
Fit a multiple linear regression model that predicts ILI from all same-week
Google search query variables. Standardize the predictor variables using
only the training data. Train the model on the training set and evaluate
its performance on both the validation and test sets using MAE and RMSE.
Compare these results to the persistence baseline.
"""
# Answer:

scaler_lr = StandardScaler()
X_train_lr = scaler_lr.fit_transform(X_train)
X_val_lr = scaler_lr.transform(X_val)
X_test_lr = scaler_lr.transform(X_test)

linreg = LinearRegression()
linreg.fit(X_train_lr, y_train)

y_val_pred_lr = linreg.predict(X_val_lr)
y_test_pred_lr = linreg.predict(X_test_lr)

mae_val_lr = mean_absolute_error(y_val, y_val_pred_lr)
rmse_val_lr = root_mean_squared_error(y_val, y_val_pred_lr)

mae_test_lr = mean_absolute_error(y_test, y_test_pred_lr)
rmse_test_lr = root_mean_squared_error(y_test, y_test_pred_lr)

print("Linear regression performance:")
print(f"  Validation MAE : {mae_val_lr:.4f}")
print(f"  Validation RMSE: {rmse_val_lr:.4f}")
print(f"  Test MAE       : {mae_test_lr:.4f}")
print(f"  Test RMSE      : {rmse_test_lr:.4f}")

print("\n" + "=" * 80 + "\n")

# Exercise 7:
"""
Description:
Fit a regularised linear model using Lasso regression to automatically
perform feature selection among the search queries. Use train+validation
data to choose the regularisation strength via cross-validation. Standardize
the predictors based on train+validation only. Evaluate the final model on
the test set (MAE and RMSE) and inspect which search terms have the largest
positive coefficients and how many coefficients are shrunk exactly to zero.
"""
# Answer:

# Combine train and validation sets
X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

scaler_lasso = StandardScaler()
X_trainval_std = scaler_lasso.fit_transform(X_trainval)
X_test_lasso_std = scaler_lasso.transform(X_test)

lasso = LassoCV(cv=5, random_state=0, n_jobs=-1, max_iter=10000)
lasso.fit(X_trainval_std, y_trainval)

y_test_pred_lasso = lasso.predict(X_test_lasso_std)

mae_test_lasso = mean_absolute_error(y_test, y_test_pred_lasso)
rmse_test_lasso = root_mean_squared_error(y_test, y_test_pred_lasso)

print("Lasso regression performance (train+val -> test):")
print(f"  Test MAE : {mae_test_lasso:.4f}")
print(f"  Test RMSE: {rmse_test_lasso:.4f}")

# Inspect coefficients
coef_series = pd.Series(lasso.coef_, index=X_trainval.columns)
nonzero = (coef_series != 0).sum()
total = len(coef_series)

print(f"\nNumber of non-zero coefficients: {nonzero}/{total}")

print("\nTop 15 positive coefficients:")
print(coef_series.sort_values(ascending=False).head(15))

print("\nTop 15 negative coefficients:")
print(coef_series.sort_values(ascending=True).head(15))
