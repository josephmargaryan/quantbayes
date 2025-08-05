# test_with_wrappers.py

import numpy as np
import jax, jax.random as jr, jax.numpy as jnp
import equinox as eqx
import optax
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from quantbayes.stochax import train, binary_loss
from quantbayes.stochax.trainer.train import predict
from spectral_pacbayes import compute_pac_bound

from quantbayes.stochax.layers import SpectralCirculantLayer, SpectralDense
from custom_spectral_norm import SpectralNorm

from equinox.nn import SpectralNorm as EquinoxSpectralNorm

def test_error_binary(model, state, X, y, *, key=jr.PRNGKey(1)):
    logits = predict(model, state, X, key=key)
    return float(jnp.mean((logits > 0).astype(jnp.int32) != y))


# ───────────────────────────────────────────────────────────────────────────────
# Wrapped spectral models: two‐layer + final linear
# ───────────────────────────────────────────────────────────────────────────────
class SpecCirc(eqx.Module):
    l1: eqx.Module
    l2: eqx.nn.Linear

    def __init__(self, key, d):
        k1, k2, k3 = jr.split(key, 3)
        self.l1 = SpectralNorm(SpectralCirculantLayer(64, key=k1))
        self.l2 = eqx.nn.Linear(64, 1, key=k2)

    def __call__(self, x, key, state):
        x = jax.nn.relu(self.l1(x))
        return self.l2(x).squeeze(-1), state


class SpecSVD(eqx.Module):
    l1: eqx.Module
    l2: eqx.nn.Linear

    def __init__(self, key, d):
        k1, k2, k3 = jr.split(key, 3)
        self.l1 = SpectralNorm(SpectralDense(64, key=k1))
        self.l2 = eqx.nn.Linear(64, 1, key=k2)

    def __call__(self, x, key, state):
        x = jax.nn.relu(self.l1(x))
        return self.l2(x).squeeze(-1), state


# ───────────────────────────────────────────────────────────────────────────────
# Plain dense for comparison
# ───────────────────────────────────────────────────────────────────────────────
class SpecLinear(eqx.Module):
    spectral_linear: eqx.nn.SpectralNorm[eqx.nn.Linear]
    l2: eqx.nn.Linear

    def __init__(self, key, d):
        k1, k2, k3 = jr.split(key, 3)
        self.spectral_linear = eqx.nn.SpectralNorm(
            layer=eqx.nn.Linear(in_features=64, out_features=64, key=k3),
            weight_name="weight",
            key=k1,
        )
        self.l2 = eqx.nn.Linear(64, 1, key=k2)

    def __call__(self, x, key, state):
        x, state = self.spectral_linear(x, state=state)
        x = jax.nn.relu(x)
        return self.l2(x).squeeze(-1), state


# ───────────────────────────────────────────────────────────────────────────────
# Training + bound helper
# ───────────────────────────────────────────────────────────────────────────────
def train_and_bound(NetClass, Xtr, ytr, Xte, yte, seed=0):
    # normalize to unit ball
    rad = np.linalg.norm(Xtr, axis=1).max()
    Xtr_n, Xte_n = Xtr / rad, Xte / rad

    # init
    mk, tk = jr.split(jr.PRNGKey(seed), 2)
    model, state = eqx.nn.make_with_state(NetClass)(mk, Xtr.shape[1])
    optimizer = optax.sgd(1e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # train with soft spectral penalty
    best, best_s, *_ = train(
        model, state, opt_state, optimizer, binary_loss,
        jnp.array(Xtr_n), jnp.array(ytr),
        jnp.array(Xtr_n), jnp.array(ytr),
        batch_size=32, num_epochs=2000, patience=20,
        key=tk,
        lambda_spec=1e-3,  # softly keep σ≈1
    )

    tr_err = test_error_binary(best, best_s, jnp.array(Xtr_n), jnp.array(ytr))
    bound  = compute_pac_bound(
        best, best_s,
        jnp.array(Xtr_n), jnp.array(ytr),
        gamma=5.0, delta=0.1, key=jr.PRNGKey(0),
    )
    te_err = test_error_binary(best, best_s, jnp.array(Xte_n), jnp.array(yte))

    print(f"  → train err={tr_err:.3f}", end=" | ")
    return bound, te_err


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    mask = np.isin(y, [0, 1])
    X, y = X[mask], y[mask]
    y = (y == 1).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    nets = {
        "SpecCirc":    SpecCirc,
        "SpecSVD":     SpecSVD,
        "SpecLinear":  SpecLinear,
    }
    for name, Net in nets.items():
        print(f"\n→ Testing {name}")
        b, e = train_and_bound(Net, Xtr, ytr, Xte, yte, seed=42)
        print(f"{name:20s} → bound={b:.4f} | test err={e:.4f}")
