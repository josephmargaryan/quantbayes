import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from quantbayes import bnn
import numpyro
import numpyro.distributions as dist
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from quantbayes.fake_data import generate_regression_data

# Generate synthetic regression data
df = generate_regression_data(n_continuous=3)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
target_scaler = MinMaxScaler()
feature_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)


# Define the model class.
class Test(bnn.Module):
    def __init__(self, prior: any, activation):
        super().__init__()
        self.prior = prior
        self.activation = activation

    def __call__(self, X, y=None):
        N, D = X.shape
        X = bnn.JVPCirculant(in_features=D, first_row_prior_fn=self.prior)(X)
        X = self.activation(X)
        X = bnn.Linear(D, 1, name="out")(X)
        mu = X.squeeze()
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


# Wrap prior functions to accept extra keyword arguments.
def gaussian_prior(shape, **kwargs):
    return dist.Normal(0.0, 1.0).expand(shape).to_event(len(shape))


def laplace_prior(shape, **kwargs):
    return dist.Laplace(0.0, 1.0).expand(shape).to_event(len(shape))


def cauchy_prior(shape, **kwargs):
    return dist.Cauchy(0.0, 1.0).expand(shape).to_event(len(shape))


priors = {"Gaussian": gaussian_prior, "Laplace": laplace_prior, "Cauchy": cauchy_prior}

activations = {"tanh": jax.nn.tanh, "SiLU": jax.nn.silu, "GELU": jax.nn.gelu}


def hyperparameter_tuning(
    X_train, y_train, X_test, y_test, priors, activations, seed=0
):
    best_rmse = float("inf")
    best_config = None
    tuning_results = []

    for prior_name, prior_fn in priors.items():
        for act_name, act_fn in activations.items():
            print(
                f"Evaluating configuration: prior={prior_name}, activation={act_name}"
            )
            key = jr.PRNGKey(seed)
            model = Test(prior=prior_fn, activation=act_fn)
            model.compile(num_chains=1, num_warmup=300, num_samples=800)

            start_time = time.time()
            model.fit(X_train, y_train, key)
            run_time = time.time() - start_time

            k1, k2 = jr.split(key, 2)
            preds = model.predict(X_test, posterior="obs", rng_key=k2)
            mean_preds = np.array(preds).mean(axis=0)

            # Inverse-transform predictions and true test targets to original scale.
            mean_preds_unscaled = target_scaler.inverse_transform(
                mean_preds.reshape(-1, 1)
            ).reshape(-1)
            y_test_unscaled = target_scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).reshape(-1)
            rmse = np.sqrt(mean_squared_error(y_test_unscaled, mean_preds_unscaled))

            config = {
                "prior": prior_name,
                "activation": act_name,
                "rmse": rmse,
                "time": run_time,
            }
            tuning_results.append(config)
            print(f"Configuration results: RMSE = {rmse:.2f}, Time = {run_time:.2f}s\n")

            if rmse < best_rmse:
                best_rmse = rmse
                best_config = config

    return best_config, tuning_results


# Run hyperparameter tuning using the pre-scaled train and test data.
# (X_train_scaled, y_train_scaled, X_test_scaled, and y_test_scaled should already be defined.)
best_config, tuning_results = hyperparameter_tuning(
    X_train_scaled,
    y_train_scaled,
    X_test_scaled,
    y_test_scaled,
    priors,
    activations,
    seed=0,
)

print("\nBest Hyperparameter Configuration:")
print(best_config)
