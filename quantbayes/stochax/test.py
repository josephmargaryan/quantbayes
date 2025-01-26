import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS

##########################################
# 1. Define the Network with Flax
##########################################
class Network(nn.Module):
    in_features: int
    hidden_dim: int
    out_features: int

    @nn.compact
    def __call__(self, x):
        # Define a simple feedforward network
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_features)(x)  # No activation at the output layer
        return x

##########################################
# 2. Training the Network with JAX
##########################################
# Create synthetic data for training
def generate_synthetic_data(n_samples=100, input_dim=2, seed=0):
    rng = jax.random.PRNGKey(seed)
    X = jax.random.normal(rng, (n_samples, input_dim))
    # True weights and bias
    W_true = jnp.array([[2.0, -1.0], [0.5, 1.5]])
    b_true = jnp.array([1.0, -0.5])
    y = jnp.dot(X, W_true.T) + b_true
    return X, y

# Training step
@jax.jit
def train_step(state, batch):
    X, y = batch

    def loss_fn(params):
        preds = state.apply_fn({"params": params}, X)
        return jnp.mean((preds - y) ** 2)  # Mean squared error

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

# Train the model
def train_flax_network():
    # Data
    X, y = generate_synthetic_data()
    batch = (X, y)

    # Initialize the model and optimizer
    model = Network(in_features=2, hidden_dim=16, out_features=2)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, X)["params"]
    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Training loop
    for step in range(1000):
        state = train_step(state, batch)
        if step % 100 == 0:
            preds = state.apply_fn({"params": state.params}, X)
            loss = jnp.mean((preds - y) ** 2)
            print(f"Step {step}: Loss = {loss:.4f}")

    return model, state.params, X, y

##########################################
# 3. Predictions with the Trained Network
##########################################
def predict_with_flax(model, params, X_new):
    return model.apply({"params": params}, X_new)

##########################################
# 4. Use NumPyro to Parameterize the Network
##########################################
def numpyro_model(X, y=None):
    # Register the Flax network
    model = flax_module("regressor", Network(in_features=2, hidden_dim=16, out_features=2), input_shape=(X.shape[0], X.shape[1]))
    
    # Sample weights and biases (implicit in NumPyro through flax_module)
    preds = model(X)
    
    # Define likelihood
    numpyro.sample("obs", dist.Normal(preds, 0.1).to_event(1), obs=y)

##########################################
# 5. Inference with SVI
##########################################
def run_svi(X, y):
    guide = lambda X, y: numpyro_model(X, y)
    optimizer = numpyro.optim.Adam(1e-3)
    svi = SVI(numpyro_model, guide, optimizer, loss=Trace_ELBO())
    svi_state = svi.init(jax.random.PRNGKey(0), X, y)

    losses = []
    for step in range(1000):
        svi_state, loss = svi.update(svi_state, X, y)
        losses.append(loss)
        if step % 100 == 0:
            print(f"Step {step}: ELBO = {-loss:.4f}")

    params = svi.get_params(svi_state)
    return params, losses

##########################################
# 6. Inference with MCMC (NUTS)
##########################################
def run_mcmc(X, y):
    kernel = NUTS(numpyro_model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
    mcmc.run(jax.random.PRNGKey(1), X, y)
    return mcmc.get_samples()

##########################################
# Main
##########################################
if __name__ == "__main__":
    # Train the deterministic network
    model, params, X, y = train_flax_network()

    # Make predictions with the trained network
    X_new = jax.random.normal(jax.random.PRNGKey(1), (5, 2))
    preds = predict_with_flax(model, params, X_new)
    print("\nPredictions with Flax Network:")
    print(preds)

    # Use SVI with NumPyro
    print("\nRunning SVI:")
    svi_params, svi_losses = run_svi(X, y)
    print("Learned Parameters with SVI:")
    print(svi_params)

    # Use MCMC with NumPyro
    print("\nRunning MCMC:")
    mcmc_samples = run_mcmc(X, y)
    print("Posterior Samples:")
    print(mcmc_samples)
