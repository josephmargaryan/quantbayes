import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Viz_preds:
    @staticmethod
    def visualize_unseen_data(
        test_df: pd.DataFrame,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        datetime_col: str,
        title: str = "Predictions with Uncertainty",
        figsize: tuple = (12, 6),
        interactive: bool = True,
    ):
        """
        Visualize predictions on unseen data with uncertainty.

        :param test_df: DataFrame containing the datetime column and any other relevant data.
        :param predictions: Mean predictions (numpy array).
        :param uncertainties: Uncertainty values (numpy array, e.g., standard deviations).
        :param datetime_col: Name of the datetime column in the DataFrame.
        :param title: Title of the plot.
        :param figsize: Tuple indicating the figure size.
        :param interactive: Whether to use interactive Plotly visualization.
        """
        # Ensure datetime column is properly formatted
        if not pd.api.types.is_datetime64_any_dtype(test_df[datetime_col]):
            test_df[datetime_col] = pd.to_datetime(test_df[datetime_col])

        # Align shapes of test_df and predictions
        dates = test_df[datetime_col].values[: len(predictions)]
        mean_predictions = predictions.squeeze()
        lower_bound = mean_predictions - 2 * uncertainties.squeeze()
        upper_bound = mean_predictions + 2 * uncertainties.squeeze()

        if interactive:
            # Plotly interactive visualization
            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=mean_predictions,
                    mode="lines",
                    name="Predictions",
                    line=dict(color="blue"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([dates, dates[::-1]]),
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill="toself",
                    fillcolor="rgba(0, 176, 246, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="Uncertainty",
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Predicted Value",
                template="plotly_white",
            )
            fig.show()

        else:
            # Matplotlib static visualization
            plt.figure(figsize=figsize)
            plt.plot(dates, mean_predictions, label="Predictions", color="blue")
            plt.fill_between(
                dates,
                lower_bound,
                upper_bound,
                color="blue",
                alpha=0.2,
                label="Uncertainty (±2σ)",
            )
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Predicted Value")
            plt.legend()
            plt.grid(True)
            plt.show()


if __name__ == "__main__":
    results = pd.DataFrame({"mean": np.arange(0, 100), "uncertainty": np.arage(0, 100)})
    test = pd.DataFrame({"mean": np.arange(0, 100), "uncertainty": np.arage(0, 100)})
    Viz_preds.visualize_unseen_data(
        test_df=test,  # your test DataFrame
        predictions=results["mean"],
        uncertainties=results["uncertainty"],
        datetime_col="date",  # Replace with your actual datetime column name
        title="Test Predictions with Uncertainty (Static)",
        interactive=True,  # Static Matplotlib plot
    )
