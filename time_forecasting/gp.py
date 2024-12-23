import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split


# Generate synthetic data
def generate_synthetic_data():
    np.random.seed(42)
    time = np.linspace(0, 10, 200)
    data = np.sin(time) + 0.1 * np.random.normal(size=len(time))
    return time, data


# Prepare the data for training and testing
def prepare_data(time, data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])
    X = np.array(X)
    y = np.array(y)
    return X, y


# Visualize predictions with uncertainty
def visualize_predictions_with_uncertainty(y_test, y_pred, y_std):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual", linestyle="dashed", color="blue")
    plt.plot(y_pred, label="Predicted", color="orange")
    plt.fill_between(
        range(len(y_pred)),
        y_pred - 2 * y_std,
        y_pred + 2 * y_std,
        color="orange",
        alpha=0.2,
        label="Uncertainty (±2σ)",
    )
    plt.legend()
    plt.title("Gaussian Process Predictions with Uncertainty")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.show()


# Main implementation
sequence_length = 10

# Generate and prepare data
time, data = generate_synthetic_data()
X, y = prepare_data(time, data, sequence_length)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the Gaussian Process kernel and regressor
kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

# Train the Gaussian Process
print("Training Gaussian Process...")
gp.fit(X_train, y_train)

# Test the Gaussian Process
print("Testing Gaussian Process...")
y_pred, y_std = gp.predict(X_test, return_std=True)

# Visualize results with uncertainty
visualize_predictions_with_uncertainty(y_test, y_pred, y_std)

# Print the kernel parameters after training
print("Learned kernel parameters:", gp.kernel_)
