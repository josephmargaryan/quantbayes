from quantbayes.bnn.model_zoo import Unet, AttentionUNet, FFTUnet, ViT
import jax


# Example usage of AttentionUnet:
rng_key = jax.random.PRNGKey(42)
X = jax.random.normal(rng_key, shape=(4, 1, 32, 32))  # 4 images, 1 channel, 32X32
y = jax.random.bernoulli(rng_key, p=0.5, shape=(4, 1, 32, 32))

fft_unet = AttentionUNet(
    in_channels=1, out_channels=1, method="svi", task_type="image_segmentation"
)
fft_unet.compile(num_warmup=50, num_samples=100, num_chains=1)
fft_unet.fit(X, y, rng_key, num_steps=10)
preds_fft = fft_unet.predict(X, rng_key, posterior="logits")
print("FFT U-Net Prediction shape:", preds_fft.shape)
fft_unet.visualize(X, y)

"""
rng_key = jax.random.PRNGKey(42)
X = jax.random.normal(rng_key, shape=(8, 1, 28, 28))  # 8 images, 3 channels, 32x32 size
y = jax.random.randint(rng_key, minval=0, maxval=10, shape=(8,))  # 8 labels (10 classes)

vit_model = ViT(
    in_channels=1,
    image_size=28,
    patch_size=4,
    embed_dim=64,
    num_heads=4,
    hidden_dim=128,
    num_layers=2,
    num_classes=10,
    method="svi",
    task_type="image_classification",
)
patches = vit_model.patchify(X)
vit_model.compile(num_warmup=50, num_samples=100, num_chains=1)
vit_model.fit(patches, y, rng_key, num_steps=25)
preds_vit = vit_model.predict(patches, rng_key, posterior="logits")
print("ViT Prediction shape:", preds_vit.shape)
vit_model.visualize(X, y, num_classes=10)
"""
