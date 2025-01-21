import jax.numpy as jnp
import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist


def var2_scan(y):
    T, K = y.shape

    # Priors
    c = numpyro.sample("c", dist.Normal(0, 1).expand([K]))
    Phi1 = numpyro.sample("Phi1", dist.Normal(0, 1).expand([K, K]).to_event(2))
    Phi2 = numpyro.sample("Phi2", dist.Normal(0, 1).expand([K, K]).to_event(2))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0).expand([K]).to_event(1))
    L_omega = numpyro.sample(
        "L_omega", dist.LKJCholesky(dimension=K, concentration=1.0)
    )
    L_Sigma = sigma[..., None] * L_omega

    def transition(carry, t):
        y_prev1, y_prev2, y_obs = carry
        m_t = c + jnp.dot(Phi1, y_prev1) + jnp.dot(Phi2, y_prev2)
        y_t = numpyro.sample(
            f"y_{t}",
            dist.MultivariateNormal(loc=m_t, scale_tril=L_Sigma),
            obs=y_obs[t],
        )
        return (y_t, y_prev1, y_obs), m_t

    init_carry = (y[1], y[0], y[2:])
    time_indices = jnp.arange(T - 2)
    _, mu = scan(transition, init_carry, time_indices)
    numpyro.deterministic("mu", mu)
