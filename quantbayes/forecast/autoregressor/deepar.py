import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepAR(nn.Module):
    """
    A simplified DeepAR model for univariate time series with Gaussian likelihood.
    The model:
      - Uses an LSTM to encode the sequence
      - For each time step, outputs mu, sigma
      - Training with teacher forcing: the next input is the ground-truth
      - Negative log-likelihood loss for each step
    """

    def __init__(self, input_dim=1, rnn_hidden=64, num_layers=2):
        """
        :param input_dim: Number of input features (1 if purely univariate)
        :param rnn_hidden: Hidden dimension of LSTM
        :param num_layers: Number of LSTM layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.rnn_hidden = rnn_hidden
        self.num_layers = num_layers

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output layer -> (mu, sigma)
        # We'll produce 2 values per time step: mu_t, sigma_t
        self.proj = nn.Linear(rnn_hidden, 2)

    def forward(self, x):
        """
        Forward pass:
        x: shape (B, T, input_dim) => unrolled in time dimension T
        Returns: (mus, sigmas) each shape (B, T)
        """
        B, T, _ = x.shape
        # LSTM
        out, _ = self.lstm(x)  # (B, T, rnn_hidden)
        # project to 2 distribution parameters
        out = self.proj(out)  # (B, T, 2)
        mu = out[:, :, 0]  # (B, T)
        sigma = F.softplus(out[:, :, 1]) + 1e-3  # keep sigma > 0
        return mu, sigma

    def compute_loss(self, x, y):
        """
        Negative log-likelihood loss for a Gaussian:
          NLL = 0.5 * log(2 pi) + log sigma_t + ((y_t - mu_t)^2 / (2 sigma_t^2))
        x: (B, T, 1) input time series
        y: (B, T)   same shape in unrolled form
        returns scalar loss
        """
        mu, sigma = self.forward(x)  # each (B, T)
        # Gaussian NLL
        loss = 0.5 * torch.log(2 * torch.pi * sigma**2) + (y - mu) ** 2 / (2 * sigma**2)
        return loss.mean()

    def predict(self, x, pred_steps=1):
        """
        One-step or multi-step forecast by ancestral sampling or using the mean.
        x: shape (B, T, 1)
        returns: predictions shape (B, pred_steps)
        """
        self.eval()
        B, T, _ = x.shape

        with torch.no_grad():
            hs, (h, c) = self.lstm(x)  # (B, T, rnn_hidden)
            # h, c each has shape (num_layers, B, rnn_hidden)

        preds = []
        # last known input => shape (B, 1, 1)
        current_input = x[:, -1:, :]

        for step in range(pred_steps):
            out, (h, c) = self.lstm(current_input, (h, c))  # out: (B, 1, rnn_hidden)
            out = self.proj(out)  # (B, 1, 2)
            mu = out[:, :, 0:1]  # shape (B, 1, 1)
            sigma = F.softplus(out[:, :, 1:2]) + 1e-3  # (B, 1, 1)

            # let's do mean-based forecast => shape (B, 1, 1)
            y_pred = mu

            preds.append(
                y_pred.squeeze(2)
            )  # => (B, 1), we only remove dimension=2, keep dimension=1
            # or just do: preds.append(mu.squeeze(-1)) => (B, 1)

            # feed it back
            current_input = y_pred  # still shape (B, 1, 1)

        # now preds is a list of length pred_steps,
        # each item shape (B, 1)
        preds = torch.cat(preds, dim=1)  # => (B, pred_steps)
        return preds


if __name__ == "__main__":
    # Synthetic example
    B, T = 16, 30
    x = torch.randn(B, T, 1)  # random input
    y = x.squeeze(-1) + 0.1 * torch.randn(B, T)  # some synthetic target

    model = DeepAR(input_dim=1, rnn_hidden=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # A small training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        loss = model.compute_loss(x, y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss={loss.item():.4f}")

    # Inference for next 5 steps
    model.eval()
    with torch.no_grad():
        future_preds = model.predict(x, pred_steps=5)  # (B, 5)
    print("Future preds shape:", future_preds.shape)
