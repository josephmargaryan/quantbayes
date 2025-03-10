import jax
import optax
import numpy as np
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from quantbayes.fake_data import generate_binary_classification_data
from quantbayes.stochax.utils import xavier_init, apply_custom_initialization
from quantbayes.stochax import train, data_loader, binary_loss, predict
from quantbayes.bnn.utils import (
    plot_roc_curve,
    plot_calibration_curve,
    expected_calibration_error,
)


df = generate_binary_classification_data()
X, y = df.drop("target", axis=1), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)
X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)
y_train = jnp.array(y_train, dtype=jnp.float32).reshape(-1, 1)
y_test = jnp.array(y_test, dtype=jnp.float32).reshape(-1, 1)
train_loader = data_loader(X=X_train, y=y_train, batch_size=800, shuffle=True)
val_loader = data_loader(X=X_test, y=y_test, batch_size=200, shuffle=False)
key = jr.key(0)

from quantbayes.stochax.layers import JVPCirculantProcess, JVPBlockCirculant
class EQNet(eqx.Module):
    l1: eqx.Module
    l2: eqx.nn.Linear

    def __init__(self, key):
        k1, k2 = jr.split(key, 2)
        self.l1 = JVPCirculantProcess(5, key=k1)
        self.l2 = eqx.nn.Linear(5, 1, key=k2)

    def __call__(self, x, key=None, state=None):
        x = self.l1(x)
        x = jax.nn.relu(x)
        x = self.l2(x)
        return x, state


model = EQNet(key)
# model = apply_custom_initialization(model, xavier_init, key=key)
state = None
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
best_model, best_state, train_losses, val_losses = train(
    model=model,
    state=state,
    opt_state=opt_state,
    optimizer=optimizer,
    loss_fn=binary_loss,
    X_train=X_train,
    y_train=y_train,
    X_val=X_test,
    y_val=y_test,
    batch_size=800,
    num_epochs=1000,
    patience=100,
    key=key,
)

inference_model = eqx.nn.inference_mode(best_model)
preds = predict(best_model, best_state, X_test, key)
probs = jax.nn.sigmoid(preds)
loss = log_loss(np.array(y_test), np.array(probs))
ece = expected_calibration_error(y_test, probs)
print(f"ECE: {ece:.3f}")
print(f"Log loss: {loss:.3f}")

plot_calibration_curve(y_test, probs)
plot_roc_curve(y_test, probs)

plt.figure(figsize=(10, 6))
plt.plot(
    np.arange(1, len(train_losses) + 1), np.array(train_losses), label="Train Loss"
)
plt.plot(
    np.arange(1, len(val_losses) + 1), np.array(val_losses), label="Validation Loss"
)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.tight_layout()
plt.show()
