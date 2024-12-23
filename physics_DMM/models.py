import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist


# ---------------------------------------
# Simple Transition and Emission Modules
# ---------------------------------------
class TransitionModel(nn.Module):
    """
    p(z_t | z_{t-1})
    We'll use a simple linear-Gaussian model:
    z_t = A * z_{t-1} + noise
    """

    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        # Simple linear parameterization
        self.A = nn.Parameter(torch.eye(z_dim))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))  # learn noise scale

    def forward(self, z_prev):
        loc = z_prev @ self.A.T
        scale = torch.exp(self.log_sigma)
        return loc, scale


class EmissionModel(nn.Module):
    """
    p(x_t | z_t)
    We'll assume a simple linear mapping from z_t to x_t:
    x_t = C * z_t + noise
    """

    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.C = nn.Linear(z_dim, x_dim)
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, z_t):
        loc = self.C(z_t)
        scale = torch.exp(self.log_sigma)
        return loc, scale


# ---------------------------------------
# Guide (Variational Distribution) q(z_t | x_{1:T})
# ---------------------------------------
class GuideRNN(nn.Module):
    """
    q(z_{1:T} | x_{1:T})
    We'll use a bidirectional GRU to produce a hidden representation of x_{1:T},
    and from that infer the distribution of each z_t.

    q(z_t | x_{1:T}) ~ N(mu_t, sigma_t)

    We'll run the RNN over the full sequence and at each time step produce
    a mean and scale for z_t.
    """

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
        # Combine forward and backward states
        self.fc_loc = nn.Linear(2 * rnn_dim, z_dim)
        self.fc_scale = nn.Linear(2 * rnn_dim, z_dim)

    def forward(self, x):
        # x: [B, T, x_dim]
        h, _ = self.rnn(x)  # h: [B, T, 2*rnn_dim]
        loc = self.fc_loc(h)
        scale = torch.exp(self.fc_scale(h))
        return loc, scale


# ---------------------------------------
# DMM class wrapping model and guide
# ---------------------------------------
class DMM(nn.Module):
    def __init__(self, x_dim=1, z_dim=2):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.trans = TransitionModel(z_dim)
        self.emission = EmissionModel(z_dim, x_dim)
        self.z_0_loc = nn.Parameter(torch.zeros(z_dim))
        # Use a log-scale parameter to ensure positivity
        self.z_0_log_scale = nn.Parameter(torch.zeros(z_dim))

        self.guide_rnn = GuideRNN(x_dim, z_dim)

    def model(self, padded_x, masks=None):
        """
        Model p(x_{1:T}, z_{1:T}) = p(z_1)*∏_{t}p(z_t|z_{t-1})p(x_t|z_t)
        """
        pyro.module("dmm", self)
        batch_size, T, _ = padded_x.size()

        # Compute actual scale from log scale
        z_0_scale = torch.exp(self.z_0_log_scale)

        with pyro.plate("sequences", batch_size):
            # Initial latent state
            z_prev = pyro.sample(
                "z_0", dist.Normal(self.z_0_loc, z_0_scale).to_event(1)
            )

            for t in pyro.markov(range(T)):
                # p(z_t | z_{t-1})
                z_loc, z_scale = self.trans(z_prev)
                z_t = pyro.sample(f"z_{t+1}", dist.Normal(z_loc, z_scale).to_event(1))

                # p(x_t | z_t)
                x_loc, x_scale = self.emission(z_t)
                pyro.sample(
                    f"x_{t+1}",
                    dist.Normal(x_loc, x_scale).to_event(1),
                    obs=padded_x[:, t, :],
                )

                z_prev = z_t

    def guide(self, padded_x, masks=None):
        """
        Guide q(z_{1:T} | x_{1:T})
        We'll use the guide_rnn to produce q(z_t|x_{1:T}).
        """
        pyro.module("dmm", self)
        batch_size, T, _ = padded_x.size()

        loc_seq, scale_seq = self.guide_rnn(padded_x)

        # Compute actual scale from log scale
        z_0_scale = torch.exp(self.z_0_log_scale)

        with pyro.plate("sequences", batch_size):
            # Sample initial latent
            z_prev = pyro.sample(
                "z_0", dist.Normal(self.z_0_loc, z_0_scale).to_event(1)
            )

            for t in pyro.markov(range(T)):
                q_loc = loc_seq[:, t, :]
                q_scale = scale_seq[:, t, :]
                z_t = pyro.sample(f"z_{t+1}", dist.Normal(q_loc, q_scale).to_event(1))
                z_prev = z_t
