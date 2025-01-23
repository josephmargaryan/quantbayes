import torch
import torch.nn as nn
import numpy as np
from quantbayes.forecast.nn.base import MonteCarloMixin, BaseModel


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


