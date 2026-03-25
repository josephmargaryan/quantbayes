# test_ebms.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from quantbayes.stochax.energy_based.base import (
    AttentionBasedEBM,
    ConvEBM,
    LSTMBasedEBM,
    MLPBasedEBM,
    RNNBasedEBM,
)
from quantbayes.stochax.energy_based.inference import detect_ood, generate_samples
from quantbayes.stochax.energy_based.train import EBMTrainer


def main():
    rng = jax.random.PRNGKey(0)

    # 1) Synthetic data (e.g. 2D for MLP)
    x_data_2d = jax.random.normal(rng, shape=(256, 2))
    # 2) Image-like data for CNN, let's say 28x28 grayscale
    x_data_img = jax.random.normal(rng, shape=(64, 1, 28, 28))  # (batch, C, H, W)

    # 3) MLP EBM
    k1, k2 = jax.random.split(rng)
    mlp_ebm = MLPBasedEBM(in_size=2, hidden_size=64, depth=2, key=k1)
    mlp_trainer = EBMTrainer(mlp_ebm, lr=1e-3)

    # Train MLP EBM for a few steps
    for step in range(10):
        subkey = jax.random.fold_in(k2, step)
        e_real, e_fake = mlp_trainer.train_step(subkey, x_data_2d)
        if step % 5 == 0:
            print(f"[MLP Step {step}] E(real)={e_real:.3f}, E(fake)={e_fake:.3f}")

    # 4) CNN EBM
    k3, k4 = jax.random.split(k2)
    conv_ebm = ConvEBM(key=k3, in_channels=1, hidden_channels=16, out_channels=32)
    conv_trainer = EBMTrainer(conv_ebm, lr=1e-3)

    # Train CNN EBM for a few steps
    for step in range(10):
        subkey = jax.random.fold_in(k4, step)
        e_real, e_fake = conv_trainer.train_step(subkey, x_data_img)
        if step % 5 == 0:
            print(f"[CNN Step {step}] E(real)={e_real:.3f}, E(fake)={e_fake:.3f}")

    # 5) Generate some samples from CNN EBM
    gen_rng = jax.random.PRNGKey(999)
    new_samples = generate_samples(
        gen_rng, conv_trainer.ebm, n_samples=8, shape=(1, 28, 28)
    )

    # Visualize a few
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i, ax in enumerate(axes):
        ax.imshow(np.squeeze(new_samples[i]), cmap="gray")
        ax.axis("off")
    plt.show()

    # 6) OOD detection on image data.
    x_ood = (
        jax.random.uniform(gen_rng, shape=(64, 1, 28, 28)) * 10.0
    )  # artificially "far"
    energies_in = conv_trainer.ebm.energy(x_data_img)
    energies_ood = conv_trainer.ebm.energy(x_ood)
    threshold = jnp.quantile(energies_in, 0.95)
    ood_mask = detect_ood(conv_trainer.ebm, x_ood, threshold)
    print("Number of OOD samples (out of 64) flagged:", jnp.sum(ood_mask))

    # ------------------------------
    # 6) RNN (GRU)-based EBM on sequence data
    # ------------------------------
    # Create synthetic sequence data for RNN-based EBM: (batch, seq_len, input_size)
    rng = jax.random.PRNGKey(30)
    k9, k10 = jax.random.split(rng)
    rnn_data = jax.random.normal(k9, shape=(128, 10, 16))
    rnn_ebm = RNNBasedEBM(input_size=16, hidden_size=32, key=k10)
    rnn_trainer = EBMTrainer(rnn_ebm, lr=1e-3)

    for step in range(10):
        subkey = jax.random.fold_in(k10, step)
        e_real, e_fake = rnn_trainer.train_step(subkey, rnn_data)
        if step % 5 == 0:
            print(f"[RNN (GRU) Step {step}] E(real)={e_real:.3f}, E(fake)={e_fake:.3f}")

    # ------------------------------
    # 4) LSTM-based EBM on sequence data
    # ------------------------------
    # Create synthetic sequence data: (batch, seq_len, input_size)
    rng = jax.random.PRNGKey(368)
    k5, k6 = jax.random.split(rng)
    seq_data = jax.random.normal(k5, shape=(128, 10, 16))
    lstm_ebm = LSTMBasedEBM(input_size=16, hidden_size=32, key=k6)
    lstm_trainer = EBMTrainer(lstm_ebm, lr=1e-3)

    for step in range(10):
        subkey = jax.random.fold_in(k6, step)
        e_real, e_fake = lstm_trainer.train_step(subkey, seq_data)
        if step % 5 == 0:
            print(f"[LSTM Step {step}] E(real)={e_real:.3f}, E(fake)={e_fake:.3f}")

    # ------------------------------
    # 5) Attention-based EBM on sequence data
    # ------------------------------
    # Create synthetic sequence data for attention: (batch, seq_len, input_size)
    rng = jax.random.PRNGKey(3310)
    k7, k8 = jax.random.split(rng)
    seq_data_attn = jax.random.normal(k7, shape=(128, 10, 16))
    # Note: now we include max_seq_len (e.g., 10)
    attn_ebm = AttentionBasedEBM(input_size=16, num_heads=4, max_seq_len=10, key=k8)
    attn_trainer = EBMTrainer(attn_ebm, lr=1e-3)

    for step in range(10):
        subkey = jax.random.fold_in(k8, step)
        e_real, e_fake = attn_trainer.train_step(subkey, seq_data_attn)
        if step % 5 == 0:
            print(f"[Attn Step {step}] E(real)={e_real:.3f}, E(fake)={e_fake:.3f}")

    # Visualize the learnable positional embeddings
    plt.imshow(attn_ebm.pos_embed, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title("Learnable Positional Embeddings")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.show()


if __name__ == "__main__":
    main()
