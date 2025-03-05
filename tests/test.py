import time
import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import numpyro
import numpyro.distributions as dist
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from quantbayes import bnn
from quantbayes.bnn.utils import evaluate_mcmc
from quantbayes.fake_data import generate_regression_data

# Generate synthetic regression data
df = generate_regression_data(n_continuous=3)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
target_scaler = MinMaxScaler()
feature_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)


class Test(bnn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, X, y=None):
        N, D = X.shape
        # Pre-normalize the input.
        X_norm = bnn.LayerNorm(num_features=D)(X)
        # First FFT-based circulant process.
        X1 = bnn.JVPCirculantProcess(in_features=D)(X_norm)
        # Apply a complex activation that modulates the spectral magnitude.
        X2 = jax.nn.gelu(X1)
        # Normalize the activated output.
        X2_norm = bnn.LayerNorm(num_features=D, name="norm_2")(X2)
        # Second FFT-based circulant process.
        X3 = bnn.JVPCirculantProcess(in_features=D, name="second_kernel")(X2_norm)
        # Use a learnable scaling parameter for the residual connection.
        res_scale = numpyro.param("res_scale", jnp.array(1.0))
        X_res = X3 + res_scale * X2
        # Apply the complex activation again after the residual addition.
        X_act = jax.nn.gelu(X_res)
        # Final linear mapping.
        X_out = bnn.Linear(in_features=D, out_features=1)(X_act)
        mu = X_out.squeeze()
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def complex_activation(z):
    """
    Applies a smooth nonlinearity in the spectral domain.
    Computes the magnitude, applies GELU, and rescales the input while preserving phase.
    """
    mag = jnp.abs(z)
    mag_act = jax.nn.gelu(mag)
    scale = mag_act / (mag + 1e-6)
    return z * scale


def aggregate_diagnostics(diag_list):
    # Get all keys from the first diagnostic dictionary.
    keys = diag_list[0].keys()
    agg = {}
    for key in keys:
        values = []
        for diag in diag_list:
            try:
                # Convert each value to float (if it's a string, float() will work)
                val = float(diag[key])
                values.append(val)
            except Exception as e:
                # If conversion fails, skip this key
                continue
        if values:
            agg[key] = {"mean": np.mean(values), "std": np.std(values)}
        else:
            agg[key] = {"mean": None, "std": None}
    return agg


# List of seeds for multiple runs
seeds = [0, 1, 2, 3, 4]
rmse_list = []
time_list = []
diagnostics_list = []

for seed in seeds:
    key = jr.PRNGKey(seed)
    k1, k2 = jr.split(key, 2)
    model = Test()
    model.compile(num_chains=4, num_warmup=1000, num_samples=1000)
    start_time = time.time()
    model.fit(X_train_scaled, y_train_scaled, k1)
    end_time = time.time()
    run_time = end_time - start_time

    preds = model.predict(X_test_scaled, posterior="obs", rng_key=k2)
    # Compute mean prediction across chains/samples
    mean_preds = np.array(preds).mean(axis=0)
    mean_preds = target_scaler.inverse_transform(mean_preds.reshape(-1, 1)).reshape(-1)
    targets = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(-1)
    rmse = np.sqrt(mean_squared_error(np.array(targets), mean_preds))

    diagnostics = evaluate_mcmc(model)

    rmse_list.append(rmse)
    time_list.append(run_time)
    diagnostics_list.append(diagnostics)

    print(f"Seed {seed}: RMSE: {rmse:.2f}, Time: {run_time:.2f} s")
    print("Diagnostics:", diagnostics)

# Aggregate overall metrics from multiple runs
avg_rmse = np.mean(rmse_list)
std_rmse = np.std(rmse_list)
avg_time = np.mean(time_list)
std_time = np.std(time_list)
diagnostic_summary = aggregate_diagnostics(diagnostics_list)

print("\nOverall Results:")
print(f"RMSE: {avg_rmse:.2f} ± {std_rmse:.2f}")
print(f"Time: {avg_time:.2f} ± {std_time:.2f} s")

# Print full diagnostic summary table
print("\nDiagnostic Summary (averages ± std):")
for key, stat in diagnostic_summary.items():
    if stat["mean"] is not None:
        print(f"{key}: {stat['mean']:.2f} ± {stat['std']:.2f}")
    else:
        print(f"{key}: N/A")
