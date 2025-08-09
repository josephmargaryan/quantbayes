import jax
import jax.numpy as jnp


def expected_calibration_error(logits, targets, n_bins: int = 15):
    # logits: [B,C], targets: [B] int32
    probs = jax.nn.softmax(logits, axis=-1)
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == targets).astype(jnp.float32)

    bins = jnp.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi)
        denom = jnp.maximum(1, mask.sum())
        acc = (correct * mask).sum() / denom
        avg_conf = (conf * mask).sum() / denom
        ece += (mask.mean()) * jnp.abs(acc - avg_conf)
    return ece


def brier_score(logits, targets, num_classes):
    probs = jax.nn.softmax(logits, axis=-1)
    onehot = jax.nn.one_hot(targets, num_classes)
    return jnp.mean(jnp.sum((probs - onehot) ** 2, axis=-1))
