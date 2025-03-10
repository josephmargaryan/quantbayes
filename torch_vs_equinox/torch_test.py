import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import log_loss

from quantbayes.fake_data import generate_binary_classification_data
from quantbayes.bnn.utils import (
    plot_calibration_curve,
    plot_roc_curve,
    expected_calibration_error,
)

df = generate_binary_classification_data()
X, y = df.drop("target", axis=1), df["target"]
X_torch = torch.tensor(X.values, dtype=torch.float32)
y_torch = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)


class Generator(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


dataset = Generator(X_torch, y_torch)
train_split = int(len(dataset) * 0.8)
val_split = len(dataset) - train_split
train_set, val_set = random_split(dataset, [train_split, val_split])
train_loader = DataLoader(train_set, shuffle=True, batch_size=800)
val_loader = DataLoader(val_set, shuffle=False, batch_size=200)


class TorchNet(nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.fcl1 = nn.Linear(5, 10)
        self.fcl2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fcl1(x)
        x = F.relu(x)
        return self.fcl2(x)


def train_torch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    patience: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Move model to the appropriate device
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_state = None
    counter = 0

    for epoch in range(num_epochs):
        # ----- Training -----
        model.train()
        epoch_train_loss = 0.0
        total_train_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            epoch_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        epoch_train_loss /= total_train_samples

        # ----- Evaluation -----
        model.eval()
        epoch_val_loss = 0.0
        total_val_samples = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                batch_size = x.size(0)
                epoch_val_loss += loss.item() * batch_size
                total_val_samples += batch_size

        epoch_val_loss /= total_val_samples

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Log progress for this epoch
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}"
        )

        # ----- Early Stopping and Model Checkpointing -----
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter > patience:
                print(f"Early stopping at epoch: {epoch + 1}")
                break

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Optionally, plot training curves for further insight
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model


def eval(model: nn.Module, val_loader: DataLoader):
    preds = []
    targets = []
    model.eval()
    for x, y in val_loader:
        with torch.no_grad():
            outputs = model(x)
            outputs = F.sigmoid(outputs)
            preds.append(outputs)
            targets.append(y)
    preds = torch.cat(preds).detach().numpy()
    targets = torch.cat(targets).detach().numpy()

    loss = log_loss(targets, preds)
    ece = expected_calibration_error(targets, preds)

    return loss, ece


def predict(model: nn.Module, X: torch.Tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
    return outputs.detach().cpu().numpy()


if __name__ == "__main__":
    torch_model = TorchNet()
    trained_model = train_torch(
        model=torch_model,
        loss_fn=nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(torch_model.parameters(), lr=1e-3),
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1000,
        patience=100,
    )
    loss, ece = eval(trained_model, val_loader)
    print(f"Loss: {loss:.3f}")
    print(f"ECE: {ece:.3f}")
