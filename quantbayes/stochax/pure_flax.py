import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
from quantbayes.fake_data import generate_regression_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import flax.serialization
import pickle

# 1. Data Preparation
df = generate_regression_data()
X, y = df.drop("target", axis=1), df["target"]

# Initialize separate scalers for X and y
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit and transform X
X = scaler_X.fit_transform(X)
X = jnp.array(X)

# Fit and transform y
y = scaler_y.fit_transform(y.to_numpy().reshape(-1, 1)).reshape(-1)
y = jnp.array(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=24, test_size=0.2
)


# 2. Model Definition
class MLP(nn.Module):
    hidden_features: int = 64  # Number of hidden units
    out_features: int = 1  # Output dimension

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_features)(x)
        return x.squeeze(-1)  # To match the shape of y


# 3. Training Setup
def create_train_state(rng, model, learning_rate, input_dim):
    params = model.init(rng, jnp.ones([1, input_dim]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# Define the loss function (Mean Squared Error)
def mse_loss(params, apply_fn, x, y):
    predictions = apply_fn({"params": params}, x)
    return jnp.mean((predictions - y) ** 2)


# Define a single training step
@jax.jit
def train_step(state, x, y):
    loss, grads = jax.value_and_grad(mse_loss)(state.params, state.apply_fn, x, y)
    state = state.apply_gradients(grads=grads)
    return state, loss


# Initialize the model and training state
rng = jax.random.PRNGKey(0)
model = MLP(hidden_features=128, out_features=1)  # You can adjust hidden units
input_dim = X_train.shape[1]
state = create_train_state(rng, model, learning_rate=1e-3, input_dim=input_dim)

# 4. Training Loop
import time

num_epochs = 1000
batch_size = 32


# Create a function to generate mini-batches
def data_generator(X, y, batch_size):
    num_samples = X.shape[0]
    indices = jax.random.permutation(rng, num_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield X_shuffled[start:end], y_shuffled[start:end]


# Training
train_losses = []
start_time = time.time()
for epoch in range(1, num_epochs + 1):
    epoch_losses = []
    for batch_X, batch_y in data_generator(X_train, y_train, batch_size):
        state, loss = train_step(state, batch_X, batch_y)
        epoch_losses.append(loss)
    mean_epoch_loss = jnp.mean(jnp.array(epoch_losses))
    train_losses.append(mean_epoch_loss)
    if epoch % 100 == 0 or epoch == 1:
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}, Loss: {mean_epoch_loss:.4f}, Time Elapsed: {elapsed:.2f}s"
        )
        start_time = time.time()

### Save the trained network
with open("flax_model_params.pkl", "wb") as f:
    pickle.dump(flax.serialization.to_state_dict(state.params), f)
"""
# Load parameters from a file
with open("flax_model_params.pkl", "rb") as f:
    loaded_params = flax.serialization.from_state_dict(model.init(rng, jnp.ones([1, input_dim]))["params"], pickle.load(f))

"""


# 5. Evaluation
# Define a prediction function
def predict(params, apply_fn, x):
    return apply_fn({"params": params}, x)


# Make predictions on the test set
predictions = predict(state.params, state.apply_fn, X_test)

# Compute MSE on the test set
test_mse = jnp.mean((predictions - y_test) ** 2)
print(f"Test MSE: {test_mse:.4f}")

# Optionally, invert scaling for better interpretability
y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
predictions_unscaled = scaler_y.inverse_transform(predictions.reshape(-1, 1)).reshape(
    -1
)

# 6. Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test_unscaled, predictions_unscaled, alpha=0.6)
plt.plot(
    [y_test_unscaled.min(), y_test_unscaled.max()],
    [y_test_unscaled.min(), y_test_unscaled.max()],
    color="red",
    linestyle="--",
    label="Ideal",
)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs. Predicted Values")
plt.legend()
plt.grid(True)
plt.show()

# Plot training loss
plt.figure(figsize=(8, 6))
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()
