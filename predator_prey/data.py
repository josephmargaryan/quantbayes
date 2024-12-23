import numpy as np


def generate_synthetic_data(num_years=20):
    """Generates synthetic predator-prey data."""
    np.random.seed(42)  # For reproducibility
    years = np.arange(num_years)
    hare_population = (
        50 + 30 * np.sin(2 * np.pi * years / 10) + np.random.normal(0, 5, num_years)
    )
    lynx_population = (
        40 + 20 * np.cos(2 * np.pi * years / 10) + np.random.normal(0, 5, num_years)
    )
    hare_population = np.clip(hare_population, 10, None)  # Avoid negative populations
    lynx_population = np.clip(lynx_population, 10, None)
    data = np.stack([hare_population, lynx_population], axis=1)
    return years, data


def load_predator_prey_data():
    """Loads synthetic predator-prey data."""
    return generate_synthetic_data()
