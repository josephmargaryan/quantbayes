from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import torch


def evaluate_predictions(dmm, data, device):
    """
    Evaluate predictions using MAE and RMSE.

    Args:
        dmm: Trained DMM model.
        data: Entire dataset as a Pandas DataFrame.
        device: Torch device.

    Returns:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
    """
    dmm.eval()
    ground_truth = data["Close"].values  # Target variable
    features = data.drop(columns=["Close"]).values  # Input features

    # Convert to Torch tensor
    features_tensor = (
        torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        # Get predictions from the guide
        loc, _ = dmm.guide_rnn(features_tensor)
        loc = loc.squeeze(0).cpu().numpy()  # Predicted means

    # Calculate MAE and RMSE
    mae = mean_absolute_error(ground_truth, loc[:, 0])
    rmse = np.sqrt(mean_squared_error(ground_truth, loc[:, 0]))

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse
