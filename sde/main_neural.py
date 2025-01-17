import torch
from stochastic_model import StochasticModel
from linear_neural_sde import LinearNeuralSDE


def main():
    # Generate synthetic data (OU process for simplicity)
    t = torch.linspace(0, 1, 100).unsqueeze(1)  # Time vector, shape (100, 1)
    y = torch.zeros(100, 1)  # One-dimensional observations
    theta = 2.0
    mu = 1.0
    sigma = 0.2
    dt = t[1] - t[0]

    for i in range(1, 100):
        dW = torch.randn(1) * torch.sqrt(dt)
        y[i] = y[i - 1] + theta * (mu - y[i - 1]) * dt + sigma * dW

    # Define the Neural SDE
    input_dim = 1
    nsde = LinearNeuralSDE(input_dim=input_dim)
    model = StochasticModel(nsde)

    # Train the Neural SDE
    print("Training the Neural SDE...")
    model.fit(t, y, lr=1e-3, epochs=1000)

    # Predict future trajectories after training
    print("Predicting trajectories after training...")
    predicted_trajectories = nsde.predict(
        t0=t[-1], y0=y[-1], T=5.0, n_paths=5, n_steps=50
    )
    print(predicted_trajectories.shape)

    # Simulate and visualize
    print("Simulating and visualizing paths...")
    model.visualize_simulation(future_horizon=0.5, n_paths=5, n_steps=50)


if __name__ == "__main__":
    main()
