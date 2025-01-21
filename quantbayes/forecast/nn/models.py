import torch
import torch.nn as nn
from forecast.nn.base import MonteCarloMixin, BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MambaStateSpaceModel(BaseModel, MonteCarloMixin):
    def __init__(self, state_dim, input_dim, output_dim):
        super().__init__()
        self.state_transition = nn.Linear(state_dim, state_dim)
        self.input_transform = nn.Linear(input_dim, state_dim)
        self.output_layer = nn.Linear(state_dim, output_dim)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Transform the input sequence
        transformed_input = self.input_transform(x)

        # Initialize the state
        state = torch.zeros(batch_size, self.state_transition.out_features).to(x.device)

        # Use a tensor to store all states
        states = []

        for t in range(seq_len):
            input_t = transformed_input[:, t, :]  # Get the t-th time step
            state = torch.tanh(self.state_transition(state) + input_t)
            states.append(state.unsqueeze(1))  # Collect states over time

        states = torch.cat(states, dim=1)  # Shape: (batch_size, seq_len, state_dim)

        # Pass the states to the output layer to produce predictions for all time steps
        outputs = self.output_layer(states)  # Shape: (batch_size, seq_len, output_dim)

        # For LSTM-like behavior, return only the prediction of the last time step
        return outputs[:, -1, :]  # Shape: (batch_size, output_dim)


class MultivariateLSTM_SDE(BaseModel, MonteCarloMixin):
    def __init__(self, num_features):
        super(MultivariateLSTM_SDE, self).__init__()
        self.lstm1 = nn.LSTM(input_size=num_features, hidden_size=50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.fc(x[:, -1, :])
        return x


class TemporalBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, dilation, dropout
    ):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2  # Corrected padding calculation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.relu, self.dropout1, self.conv2, self.relu, self.dropout2
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCN(BaseModel, MonteCarloMixin):
    def __init__(
        self,
        input_size,
        output_size,
        num_channels,
        kernel_size,
        dropout,
    ):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        # Transpose input to match Conv1D format
        # LSTM input: (batch_size, seq_len, input_size)
        # Conv1D input: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)

        # Pass through TCN layers
        out = self.network(x)

        # Take the last time step's feature map
        out = out[:, :, -1]  # Shape: (batch_size, num_channels[-1])

        # Final linear layer for prediction
        return self.linear(out)  # Shape: (batch_size, output_size)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = self.layer1(x)
        x_ = torch.relu(x_)
        x_ = self.layer2(x_)
        gated_x = torch.sigmoid(self.gate(x_)) * x_
        return self.layer_norm(self.dropout(gated_x) + x)


class TemporalFusionTransformer(BaseModel, MonteCarloMixin):
    def __init__(
        self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 500, model_dim)
        )  # Max sequence length = 500
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        self.gate = GatedResidualNetwork(model_dim, model_dim, model_dim, dropout)
        self.fc_out = nn.Linear(model_dim, output_dim)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        seq_len = x.size(1)

        # Embedding and positional encoding
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Gated residual network
        x = self.gate(x)

        # Extract only the last time step
        x = x[:, -1, :]  # Shape: (batch_size, model_dim)

        # Final output layer
        return self.fc_out(x)  # Shape: (batch_size, output_dim)


class TransformerTimeSeriesModel(BaseModel, MonteCarloMixin):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 500, model_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(model_dim, output_dim)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])
