import numpy as np
import jax
import jax.random as jr
import equinox as eqx
import augmax

from quantbayes.stochax.trainer.train import make_augmax_augment

# ----- fake “dataset” ---------------------------------------------------
rng = np.random.RandomState(0)
N, H, W, C, NUM_CLASSES = 2048, 1, 28, 28, 1
X_np = rng.rand(N, H, W, C).astype("float32")  # [N, H, W, C]
y_np = rng.randint(0, NUM_CLASSES, size=(N,)).astype("int32")

# train / val split
split = int(0.8 * N)
X_train, X_val = X_np[:split], X_np[split:]
y_train, y_val = y_np[:split], y_np[split:]

# ----- augmentation pipeline -------------------------------------------
transform = augmax.Chain(
    augmax.HorizontalFlip(),
    augmax.Rotate(angle_range=15),
    input_types=[augmax.InputType.IMAGE, augmax.InputType.METADATA],
)
augment_fn = make_augmax_augment(transform)


# ----- simple CNN -------------------------------------------------------
class SimpleCNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    pool1: eqx.nn.MaxPool2d
    pool2: eqx.nn.MaxPool2d
    fc1: eqx.nn.Linear
    bn3: eqx.nn.BatchNorm
    fc2: eqx.nn.Linear
    drop1: eqx.nn.Dropout
    drop2: eqx.nn.Dropout
    drop3: eqx.nn.Dropout

    def __init__(self, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(1, 32, kernel_size=3, padding=1, key=k1)
        self.bn1 = eqx.nn.BatchNorm(input_size=32, axis_name="batch", mode="batch")
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=3, padding=1, key=k2)
        self.bn2 = eqx.nn.BatchNorm(input_size=64, axis_name="batch", mode="batch")
        self.pool1 = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = eqx.nn.Linear(64 * 7 * 7, 128, key=k3)
        self.bn3 = eqx.nn.BatchNorm(input_size=128, axis_name="batch", mode="batch")
        self.fc2 = eqx.nn.Linear(128, 10, key=k4)
        self.drop1 = eqx.nn.Dropout(0.25)
        self.drop2 = eqx.nn.Dropout(0.25)
        self.drop3 = eqx.nn.Dropout(0.50)

    def __call__(self, x, key, state):
        d1, d2, d3 = jr.split(key, 3)
        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        x = self.pool1(x)
        x = self.drop1(x, key=d1)
        x = self.conv2(x)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)
        x = self.pool2(x)
        x = self.drop2(x, key=d2)
        x = x.reshape(-1)
        x = self.fc1(x)
        x, state = self.bn3(x, state)
        x = jax.nn.relu(x)
        x = self.drop3(x, key=d3)
        x = self.fc2(x)
        return x, state
