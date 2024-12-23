import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist


#############################
# FFT Circulant Layer
#############################
class FFTCirculantLayer(nn.Module):
    """
    A circulant matrix multiplication using FFT.
    Expects input of shape [B, D], transforms last dimension.
    """

    def __init__(self, n):
        super(FFTCirculantLayer, self).__init__()
        self.n = n
        # Learnable parameters defining the first row of the circulant matrix
        self.c = nn.Parameter(torch.randn(n))

    def forward(self, x):
        # x: [B, D] or [B, T, D]
        # We assume D = self.n
        # If x is 2D: [B, n], apply FFT along last dimension
        # If x is 3D: [B,T,n], we apply FFT along last dimension
        c_fft = torch.fft.fft(self.c)
        x_fft = torch.fft.fft(x, n=self.n, dim=-1)
        result_fft = c_fft * x_fft
        result = torch.fft.ifft(result_fft, dim=-1)
        return result.real


#############################
# Transition Model
#############################
class TransitionModel(nn.Module):
    """
    p(z_t | z_{t-1}) using FFT circulant layer
    """

    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        # Instead of a simple linear transform, use a circulant transform
        self.trans_layer = FFTCirculantLayer(z_dim)
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, z_prev):
        # z_prev: [B, z_dim]
        loc = self.trans_layer(z_prev)  # [B, z_dim]
        scale = torch.exp(self.log_sigma)
        return loc, scale


#############################
# Emission Model
#############################
class EmissionModel(nn.Module):
    """
    p(x_t | z_t), also using FFT circulant layer for demonstration
    """

    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.emission_layer = FFTCirculantLayer(z_dim)
        self.out = nn.Linear(z_dim, x_dim)
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, z_t):
        # z_t: [B, z_dim]
        h = self.emission_layer(z_t)  # [B, z_dim]
        loc = self.out(h)  # [B, x_dim]
        scale = torch.exp(self.log_sigma)
        return loc, scale


#############################
# Guide RNN (no circulant here for simplicity)
#############################
class GuideRNN(nn.Module):
    def __init__(self, x_dim, z_dim, rnn_dim=64, num_layers=1):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.rnn = nn.GRU(
            input_size=x_dim,
            hidden_size=rnn_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_loc = nn.Linear(2 * rnn_dim, z_dim)
        self.fc_scale = nn.Linear(2 * rnn_dim, z_dim)

    def forward(self, x):
        # x: [B, T, x_dim]
        h, _ = self.rnn(x)  # [B, T, 2*rnn_dim]
        loc = self.fc_loc(h)
        scale = torch.exp(self.fc_scale(h))
        return loc, scale


#############################
# DMM Class
#############################
class DMM(nn.Module):
    def __init__(self, x_dim=2, z_dim=2):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.trans = TransitionModel(z_dim)
        self.emission = EmissionModel(z_dim, x_dim)
        self.z_0_loc = nn.Parameter(torch.zeros(z_dim))
        self.z_0_log_scale = nn.Parameter(torch.zeros(z_dim))

        self.guide_rnn = GuideRNN(x_dim, z_dim)

    def model(self, padded_x, masks=None):
        pyro.module("dmm", self)
        batch_size, T, _ = padded_x.size()

        z_0_scale = torch.exp(self.z_0_log_scale)

        with pyro.plate("sequences", batch_size):
            z_prev = pyro.sample(
                "z_0", dist.Normal(self.z_0_loc, z_0_scale).to_event(1)
            )

            for t in pyro.markov(range(T)):
                z_loc, z_scale = self.trans(z_prev)
                z_t = pyro.sample(f"z_{t+1}", dist.Normal(z_loc, z_scale).to_event(1))

                x_loc, x_scale = self.emission(z_t)
                pyro.sample(
                    f"x_{t+1}",
                    dist.Normal(x_loc, x_scale).to_event(1),
                    obs=padded_x[:, t, :],
                )
                z_prev = z_t

    def guide(self, padded_x, masks=None):
        pyro.module("dmm", self)
        batch_size, T, _ = padded_x.size()
        loc_seq, scale_seq = self.guide_rnn(padded_x)

        z_0_scale = torch.exp(self.z_0_log_scale)

        with pyro.plate("sequences", batch_size):
            z_prev = pyro.sample(
                "z_0", dist.Normal(self.z_0_loc, z_0_scale).to_event(1)
            )

            for t in pyro.markov(range(T)):
                q_loc = loc_seq[:, t, :]
                q_scale = scale_seq[:, t, :]
                z_t = pyro.sample(f"z_{t+1}", dist.Normal(q_loc, q_scale).to_event(1))
                z_prev = z_t
