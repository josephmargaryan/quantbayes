import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.distributions import Normal, kl_divergence
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.decomposition import PCA
import seaborn as sns


# Function to generate fake data
def generate_fake_data(num_companies=10, num_features=2):
    """
    Generate fake data simulating company sentiment and stock prices.

    Args:
        num_companies (int): Number of companies to simulate.
        num_features (int): Number of features per company.

    Returns:
        DataFrame: Simulated data in the required format.
    """
    companies = [f"Company_{i}" for i in range(num_companies)]
    predicted_labels = np.random.choice(
        ["Positive", "Neutral", "Negative"], size=num_companies
    )
    close_prices = np.random.uniform(50, 500, size=num_companies)
    volatility = np.random.uniform(0.01, 0.05, size=num_companies)
    rsi = np.random.uniform(30, 70, size=num_companies)

    data = {
        "Company": companies,
        "predicted_labels": predicted_labels,
        "Close": close_prices,
        "Volatility": volatility,
        "RSI": rsi,
    }

    return pd.DataFrame(data)


# PCA Visualization in 3D
def pca_visualization_3d(fake_data):
    """
    Perform PCA on the dataset and visualize the first 3 principal components in 3D.

    Args:
        fake_data (DataFrame): The dataset to visualize.
    """
    features = ["Close", "Volatility", "RSI"]
    X = fake_data[features].values
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        components[:, 0],
        components[:, 1],
        components[:, 2],
        c=fake_data["predicted_labels"],
        cmap="viridis",
        s=50,
    )
    ax.set_title("PCA of Feature Space in 3D")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    fig.colorbar(sc, ax=ax, label="Sentiment Label")
    plt.show()


# Maximum Entropy Model
def maximum_entropy(fake_data):
    X = fake_data[["predicted_labels", "Close"]].values
    y = (fake_data["Close"].pct_change() > 0).astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    print("Predicted Probabilities (Maximum Entropy):\n", probabilities)


# Mutual Information
def mutual_information(fake_data):
    X = fake_data[["predicted_labels", "Close"]].values
    y = fake_data["Close"].shift(-1).fillna(method="ffill").values
    mi = mutual_info_regression(X, y)
    print("Mutual Information Scores:\n", mi)


# Covariance Matrix
def covariance_matrix(fake_data):
    cov_matrix = fake_data[["predicted_labels", "Close"]].cov()
    print("Covariance Matrix:\n", cov_matrix)
    sns.heatmap(cov_matrix, annot=True, cmap="coolwarm")
    plt.title("Covariance Matrix")
    plt.show()


# Canonical Correlation Analysis (CCA)
def canonical_correlation_analysis(fake_data):
    X = fake_data[["predicted_labels", "Close"]].values
    y = fake_data[["Volatility", "RSI"]].values
    cca = CCA(n_components=2)
    X_c, y_c = cca.fit_transform(X, y)
    print("Canonical Correlations:\n", cca.score(X, y))


# Granger Causality
def granger_causality(fake_data):
    data = fake_data[["predicted_labels", "Close"]].dropna()
    results = grangercausalitytests(data, maxlag=2, verbose=True)


# Free Energy Minimization with Visualization
def free_energy_minimization(fake_data, num_latent_features=2):
    """
    Perform free energy minimization on the sentiment and stock price data.

    Args:
        fake_data (DataFrame): Simulated data.
        num_latent_features (int): Number of latent variables to infer.

    Returns:
        torch.Tensor: Free energy for the dataset.
    """
    X = fake_data[["predicted_labels", "Close"]].values
    X = torch.tensor((X - X.mean(axis=0)) / X.std(axis=0), dtype=torch.float)

    q_mean = torch.zeros(num_latent_features, requires_grad=True)
    q_log_var = torch.zeros(num_latent_features, requires_grad=True)

    optimizer = torch.optim.Adam([q_mean, q_log_var], lr=0.01)

    free_energies = []

    for step in range(500):
        optimizer.zero_grad()

        q = Normal(q_mean, torch.exp(0.5 * q_log_var))

        prior = Normal(0, 1)
        kl = kl_divergence(q, prior).sum()

        recon_loss = ((X - q.sample([len(X)])) ** 2).mean()

        free_energy = recon_loss + kl
        free_energy.backward()
        optimizer.step()

        free_energies.append(free_energy.item())

        if step % 50 == 0:
            print(f"Step {step}, Free Energy: {free_energy.item()}")

    print("Optimization Complete.")
    print(
        f"Learned Mean: {q_mean.detach().numpy()}, Variance: {torch.exp(0.5 * q_log_var).detach().numpy()}"
    )

    # Visualization of Free Energy Minimization Path
    plt.figure(figsize=(10, 6))
    plt.plot(free_energies, label="Free Energy")
    plt.xlabel("Iteration")
    plt.ylabel("Free Energy")
    plt.title("Free Energy Minimization Path")
    plt.legend()
    plt.grid()
    plt.show()

    return free_energy.detach().item()


def entropy_surface_visualization():
    """
    Visualize entropy as a surface for a two-dimensional latent space.
    """
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    Z = -0.5 * (1 + np.log(2 * np.pi)) - 0.5 * (X**2 + Y**2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
    ax.set_title("Entropy Surface for Latent Space")
    ax.set_xlabel("Latent Variable 1")
    ax.set_ylabel("Latent Variable 2")
    ax.set_zlabel("Entropy")
    plt.show()


def main():
    # fake_data = generate_fake_data()
    fake_data = pd.read_csv("merged_data.csv")

    label_encoder = LabelEncoder()
    fake_data["predicted_labels"] = label_encoder.fit_transform(
        fake_data["predicted_labels"]
    )

    print("\n--- Running Maximum Entropy Model ---")
    maximum_entropy(fake_data)

    print("\n--- Running Mutual Information ---")
    mutual_information(fake_data)

    print("\n--- Running Covariance Matrix ---")
    covariance_matrix(fake_data)

    print("\n--- Running Canonical Correlation Analysis ---")
    canonical_correlation_analysis(fake_data)

    print("\n--- Running Granger Causality ---")
    granger_causality(fake_data)

    print("\n--- Running Free Energy Minimization ---")
    free_energy_minimization(fake_data)

    print("\n--- Running PCA Visualization in 3D ---")
    pca_visualization_3d(fake_data)
    entropy_surface_visualization()


if __name__ == "__main__":
    main()
