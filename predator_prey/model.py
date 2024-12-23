import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.experimental.ode import odeint


def dz_dt(z, t, theta):
    """Lotka–Volterra equations."""
    u, v = z
    alpha, beta, gamma, delta = theta
    du_dt = (alpha - beta * v) * u
    dv_dt = (-gamma + delta * u) * v
    return jnp.stack([du_dt, dv_dt])


def predator_prey_model(N, y=None):
    """Defines the probabilistic model for the predator-prey dynamics."""
    z_init = numpyro.sample("z_init", dist.LogNormal(jnp.log(10), 1).expand([2]))
    ts = jnp.arange(float(N))
    theta = numpyro.sample(
        "theta",
        dist.TruncatedNormal(
            low=0.0,
            loc=jnp.array([1.0, 0.05, 1.0, 0.05]),
            scale=jnp.array([0.5, 0.05, 0.5, 0.05]),
        ),
    )
    z = odeint(dz_dt, z_init, ts, theta, rtol=1e-6, atol=1e-5, mxstep=1000)
    sigma = numpyro.sample("sigma", dist.LogNormal(-1, 1).expand([2]))
    numpyro.sample("y", dist.LogNormal(jnp.log(z), sigma), obs=y)
