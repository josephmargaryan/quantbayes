import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


class TransformerTimeSeries(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout,
    ):
        super(TransformerTimeSeries, self).__init__()
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, model_dim))
        nn.init.normal_(
            self.positional_encoding, mean=0, std=0.02
        )  # Initialize positional encodings

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_decoder_layers
        )

        self.output_layer = nn.Linear(model_dim, 1)

    def forward(self, src, tgt):
        # Embedding input and target
        src = self.input_embedding(src) + self.positional_encoding[
            :, : src.size(1), :
        ].to(src.device)
        tgt = self.input_embedding(tgt) + self.positional_encoding[
            :, : tgt.size(1), :
        ].to(tgt.device)

        # Encoding
        memory = self.encoder(src)

        # Decoding
        output = self.decoder(tgt, memory)

        return self.output_layer(output)


# Training function
def train_model(model, dataloader, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for src, tgt, tgt_y in dataloader:
            optimizer.zero_grad()
            output = model(src, tgt)
            output = output[:, -1, :]  # Ensure the output matches target size
            loss = criterion(output.squeeze(-1), tgt_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return train_losses


# Validation function
def validate_model(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tgt, tgt_y in dataloader:
            output = model(src, tgt)
            output = output[:, -1, :]  # Ensure the output matches target size
            loss = criterion(output.squeeze(-1), tgt_y)
            val_loss += loss.item()
    avg_loss = val_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


# Visualization function
def visualize_predictions(model, dataloader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for src, tgt, tgt_y in dataloader:
            output = model(src, tgt)
            output = output[:, -1, :]  # Ensure the output matches target size
            predictions.append(output.squeeze(-1).numpy())
            actuals.append(tgt_y.numpy())
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title("Predictions vs Actual")
    plt.show()


# Data preparation
def generate_synthetic_data():
    np.random.seed(42)
    time = np.arange(0, 400, 0.1)
    data = np.sin(time) + np.random.normal(scale=0.1, size=len(time))
    return data


def prepare_data(data, sequence_length):
    src, tgt, tgt_y = [], [], []
    for i in range(len(data) - sequence_length - 1):
        src.append(data[i : i + sequence_length])
        tgt.append(data[i + 1 : i + sequence_length + 1])
        tgt_y.append(data[i + sequence_length + 1])
    src = torch.tensor(np.array(src), dtype=torch.float32).unsqueeze(
        -1
    )  # Shape: [batch, seq_len, 1]
    tgt = torch.tensor(np.array(tgt), dtype=torch.float32).unsqueeze(-1)
    tgt_y = torch.tensor(np.array(tgt_y), dtype=torch.float32)  # Shape: [batch]
    return src, tgt, tgt_y


# Hyperparameters
sequence_length = 50
data = generate_synthetic_data()
src, tgt, tgt_y = prepare_data(data, sequence_length)

dataset = TensorDataset(src, tgt, tgt_y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = 1
model_dim = 64
num_heads = 4
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.1

model = TransformerTimeSeries(
    input_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_losses = train_model(model, dataloader, criterion, optimizer, epochs=10)
visualize_predictions(model, dataloader)
