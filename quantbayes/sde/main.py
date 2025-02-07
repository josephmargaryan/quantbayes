import numpy as np

from stochastic_model import StochasticModel
from sde_gbm import GeometricBrownianMotion
from sde_heston import HestonModel
from sde_merton_jump import MertonJumpDiffusion
from sde_ornstein import OrnsteinUhlenbeck
from bayesian_jump_diffusion import BayesianMertonJumpDiffusion
from bayesian_ornstein import BayesianOrnsteinUhlenbeck
from bayesian_brownian import BayesianGeometricBrownianMotion
from bayesian_heston import BayesianHestonModel


def main():
    # 1) Generate synthetic data (GBM) over [0, 1], 50 time steps
    np.random.seed(42)
    N = 50
    T = 1.0
    t = np.linspace(0, T, N)
    dt = t[1] - t[0]

    true_mu = 0.1
    true_sigma = 0.2
    y = np.zeros(N)
    y[0] = 1.0  # initial value
    for i in range(1, N):
        dW = np.random.normal(0.0, np.sqrt(dt))
        y[i] = y[i - 1] + true_mu * y[i - 1] * dt + true_sigma * y[i - 1] * dW

    # 2) Create and fit the model
    gbm = MertonJumpDiffusion()
    model = StochasticModel(gbm)
    model.fit(t, y)
    predicted_trajectories = gbm.predict(
        t0=t[-1], y0=y[-1], T=5.0, n_paths=10, n_steps=50
    )

    print(predicted_trajectories.shape)

    # 3) Visualize the model's fitted parameters
    print("Fitted mu:", gbm.mu)
    print("Fitted sigma:", gbm.sigma)

    # 4) Visualize future paths
    model.visualize_simulation(future_horizon=0.5, n_paths=30, n_steps=50)


if __name__ == "__main__":
    main()
