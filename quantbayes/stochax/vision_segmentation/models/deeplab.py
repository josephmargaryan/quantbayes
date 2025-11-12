from typing import Tuple, Any
import jax, jax.numpy as jnp, jax.random as jr, jax.image
import equinox as eqx


class ConvBnRelu(eqx.Module):
    conv: eqx.nn.Conv2d
    bn: eqx.nn.BatchNorm

    def __init__(
        self,
        cin: int,
        cout: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        *,
        key,
    ):
        if padding is None:
            padding = dilation * (kernel_size // 2)
        k1, k2 = jr.split(key, 2)
        self.conv = eqx.nn.Conv2d(
            cin,
            cout,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            key=k1,
        )
        self.bn = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")

    def __call__(self, x: jnp.ndarray, *, key, state: Any) -> Tuple[jnp.ndarray, Any]:
        x = self.conv(x, key=key)
        x, state = self.bn(x, state)
        x = jax.nn.relu(x)
        return x, state


class ASPP(eqx.Module):
    b1: ConvBnRelu
    b2: ConvBnRelu
    b3: ConvBnRelu
    b4: ConvBnRelu
    pool_proj: ConvBnRelu
    project: eqx.nn.Conv2d
    bn_proj: eqx.nn.BatchNorm

    def __init__(self, cin: int, cout: int, *, key):
        ks = jr.split(key, 6)
        self.b1 = ConvBnRelu(cin, cout, kernel_size=1, key=ks[0])
        self.b2 = ConvBnRelu(cin, cout, kernel_size=3, dilation=6, key=ks[1])
        self.b3 = ConvBnRelu(cin, cout, kernel_size=3, dilation=12, key=ks[2])
        self.b4 = ConvBnRelu(cin, cout, kernel_size=3, dilation=18, key=ks[3])
        self.pool_proj = ConvBnRelu(cin, cout, kernel_size=1, key=ks[4])
        self.project = eqx.nn.Conv2d(cout * 5, cout, 1, key=ks[5])
        self.bn_proj = eqx.nn.BatchNorm(cout, axis_name="batch", mode="batch")

    def __call__(self, x: jnp.ndarray, *, key, state: Any) -> Tuple[jnp.ndarray, Any]:
        k1, k2, k3, k4, k5, k6 = jr.split(key, 6)
        y1, state = self.b1(x, key=k1, state=state)
        y2, state = self.b2(x, key=k2, state=state)
        y3, state = self.b3(x, key=k3, state=state)
        y4, state = self.b4(x, key=k4, state=state)
        gp = jnp.mean(x, axis=(1, 2), keepdims=True)  # [C, 1, 1]
        gp, state = self.pool_proj(gp, key=k5, state=state)  # [cout, 1, 1]
        gp = jax.image.resize(
            gp,
            (gp.shape[0], x.shape[1], x.shape[2]),
            method="bilinear",
        )
        cat = jnp.concatenate([y1, y2, y3, y4, gp], axis=0)
        proj = self.project(cat, key=k6)
        proj, state = self.bn_proj(proj, state)
        proj = jax.nn.relu(proj)
        return proj, state


class DeepLabV3Plus(eqx.Module):
    e1: ConvBnRelu
    e2: ConvBnRelu
    e3: ConvBnRelu
    e4: ConvBnRelu
    aspp: ASPP
    low_proj: eqx.nn.Conv2d
    low_bn: eqx.nn.BatchNorm
    dec1: ConvBnRelu
    dec2: ConvBnRelu
    out: eqx.nn.Conv2d

    def __init__(self, *, in_ch: int = 3, out_ch: int = 1, base: int = 8, key):
        ks = jr.split(key, 9)
        self.e1 = ConvBnRelu(in_ch, base, stride=2, key=ks[0])
        self.e2 = ConvBnRelu(base, base * 2, stride=2, key=ks[1])
        self.e3 = ConvBnRelu(base * 2, base * 4, stride=2, key=ks[2])
        self.e4 = ConvBnRelu(base * 4, base * 16, stride=2, key=ks[3])
        self.aspp = ASPP(base * 16, base * 16, key=ks[4])
        self.low_proj = eqx.nn.Conv2d(base * 2, 48, 1, key=ks[5])
        self.low_bn = eqx.nn.BatchNorm(48, axis_name="batch", mode="batch")
        self.dec1 = ConvBnRelu(base * 16 + 48, base * 16, key=ks[6])
        self.dec2 = ConvBnRelu(base * 16, base * 16, key=ks[7])
        self.out = eqx.nn.Conv2d(base * 16, out_ch, 1, key=ks[8])

    def __call__(self, x: jnp.ndarray, key, state: Any) -> Tuple[jnp.ndarray, Any]:
        k1, k2, k3, k4, k5, k6, k7, k8, k9 = jr.split(key, 9)
        e1, state = self.e1(x, key=k1, state=state)
        e2, state = self.e2(e1, key=k2, state=state)
        e3, state = self.e3(e2, key=k3, state=state)
        e4, state = self.e4(e3, key=k4, state=state)
        a, state = self.aspp(e4, key=k5, state=state)
        low = self.low_proj(e2, key=k6)
        low, state = self.low_bn(low, state)
        low = jax.nn.relu(low)
        a_up = jax.image.resize(
            a,
            (a.shape[0], low.shape[1], low.shape[2]),
            method="bilinear",
        )
        cat = jnp.concatenate([a_up, low], axis=0)
        d1, state = self.dec1(cat, key=k7, state=state)
        d2, state = self.dec2(d1, key=k8, state=state)
        out_logits = self.out(d2)
        logits = jax.image.resize(
            out_logits,
            (out_logits.shape[0], x.shape[1], x.shape[2]),
            method="bilinear",
        )
        return logits, state


if __name__ == "__main__":
    """
    Synthetic segmentation-pipeline smoke-test.
    Runs CPU-only in <10 s. Swap the fake data for CIFAR-10, Cityscapes,
    etc. in real experiments.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import augmax
    from augmax import InputType

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import equinox as eqx

    from quantbayes.stochax import (
        train,
        predict,
        make_augmax_augment,
        make_dice_bce_loss,
    )

    rng = np.random.RandomState(0)
    N, C, H, W, OUT_CH = 10, 3, 128, 128, 1

    X_np = rng.rand(N, C, H, W).astype("float32")

    y_np = rng.randint(0, 2, size=(N, OUT_CH, H, W)).astype("float32")

    split = int(0.8 * N)
    X_train, X_val = X_np[:split], X_np[split:]
    y_train, y_val = y_np[:split], y_np[split:]

    transform = augmax.Chain(
        augmax.HorizontalFlip(),
        augmax.Rotate(angle_range=15),
        input_types=[InputType.IMAGE, InputType.MASK],
    )
    augment_fn = make_augmax_augment(transform)

    master_key = jr.PRNGKey(42)
    model_key, train_key = jr.split(master_key)
    model, state = eqx.nn.make_with_state(DeepLabV3Plus)(
        in_ch=C,
        out_ch=OUT_CH,
        base=8,
        key=model_key,
    )

    lr_sched = optax.cosine_decay_schedule(1e-3, decay_steps=500)
    optimizer = optax.adamw(
        learning_rate=lr_sched,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=1e-4,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    best_model, best_state, tr_loss, va_loss = train(
        model=model,
        state=state,
        opt_state=opt_state,
        optimizer=optimizer,
        loss_fn=make_dice_bce_loss(),  # BCE-with-logits over 1-channel masks
        X_train=jnp.array(X_train),  # (N,C,H,W)
        y_train=jnp.array(y_train),  # (N,1,H,W)
        X_val=jnp.array(X_val),
        y_val=jnp.array(y_val),
        batch_size=32,
        num_epochs=15,
        patience=4,
        key=train_key,
        augment_fn=augment_fn,  # our NCHW ↔ NHWC/HW wrapper
        lambda_spec=0.0,
    )

    plt.plot(tr_loss, label="train")
    plt.plot(va_loss, label="val")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Synthetic TransUNet smoke-test")
    plt.show()
