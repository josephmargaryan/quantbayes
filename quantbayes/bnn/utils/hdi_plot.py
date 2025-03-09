import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def plot_hdi(
    predictions,
    X_test,
    credible_interval=0.95,
    ax=None,
    xlabel="X",
    ylabel="Prediction",
    feature_index=None,
):
    """
    Plot the HDI (highest density interval) for a set of regression predictions.

    Parameters
    ----------
    predictions : np.ndarray
        A 2D array of shape (n_posterior_samples, n_test_points) with posterior predictive samples.
    X_test : np.ndarray
        The test input values. For a single-feature model, this should be a 1D array (or a 2D array
        with one column). For multi-feature models, this can be any array.
    credible_interval : float, optional
        The probability mass to include in the HDI (default is 0.95).
    ax : matplotlib.axes.Axes, optional
        An axes object on which to plot. If None, a new figure and axes are created.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    feature_index : int or None, optional
        If provided and X_test is multi-dimensional, this index is used to extract the x-axis values from X_test.
        If None, and X_test is multi-dimensional, the function defaults to using the sample index.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the plot was drawn.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Ensure predictions is a numpy array
    predictions = np.array(predictions)

    # Number of test points (assumed along axis 1)
    n_posterior, n_points = predictions.shape

    # Calculate the median prediction for each test point
    median_pred = np.median(predictions, axis=0)

    # Compute the HDI for each test point
    hdi_lower = np.empty(n_points)
    hdi_upper = np.empty(n_points)
    for i in range(n_points):
        hdi_interval = az.hdi(predictions[:, i], credible_interval=credible_interval)
        hdi_lower[i] = hdi_interval[0]
        hdi_upper[i] = hdi_interval[1]

    # Determine what to use for the x-axis.
    if X_test.ndim == 1:
        x = X_test
        sort_x = True
    elif X_test.ndim == 2:
        if X_test.shape[1] == 1:
            x = X_test.flatten()
            sort_x = True
        else:
            # Multi-feature scenario: if a feature_index is specified, use that feature for x
            if feature_index is not None and 0 <= feature_index < X_test.shape[1]:
                x = X_test[:, feature_index]
                sort_x = True
            else:
                # Default to sample indices if no specific feature is chosen
                x = np.arange(n_points)
                sort_x = False
    else:
        # Fallback for unexpected dimensions
        x = np.arange(n_points)
        sort_x = False

    # Sort if necessary (e.g., when x is a continuous feature)
    if sort_x:
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        median_pred = median_pred[sort_idx]
        hdi_lower = hdi_lower[sort_idx]
        hdi_upper = hdi_upper[sort_idx]

    # Plot median predictions as a line
    ax.plot(x, median_pred, color="C0", lw=2, label="Median prediction")
    # Plot HDI as a shaded region
    ax.fill_between(
        x,
        hdi_lower,
        hdi_upper,
        color="C0",
        alpha=0.3,
        label=f"{int(credible_interval*100)}% HDI",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()

    return ax
