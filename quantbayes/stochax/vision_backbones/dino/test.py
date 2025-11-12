"""
import equinox as eqx, jax.random as jr
from quantbayes.stochax.vision_backbones.dino.vit_dino_eqx import (
    build_dinov2_encoder, DINOClassifier
)
from quantbayes.stochax.vision_backbones.dino.dinov2_loader import load_dinov2_npz

key = jr.PRNGKey(0)
enc = build_dinov2_encoder(
    "vitb14",
    image_size=518,     # DINOv2 pretrain grid
    num_registers=4,    # must match checkpoint (0/4/8)
    use_cls=True,
    key=key,
)
enc = load_dinov2_npz(enc, "dinov2-base.npz", strict=True)

clf = DINOClassifier(num_classes=1000, encoder=enc, key=jr.fold_in(key, 1))
# logits, state = clf(x, key=jr.fold_in(key, 2), state=None)  # x: [3,H,W]

"""

"""
import jax.random as jr
from quantbayes.stochax.vision_backbones.dino.vit_dino_eqx import (
    build_dinov2_encoder, DINOSegmenter
)
from quantbayes.stochax.vision_backbones.dino.dinov2_loader import load_dinov2_npz

key = jr.PRNGKey(1)
enc = build_dinov2_encoder("vitb14", image_size=518, num_registers=4, use_cls=False, key=key)
enc = load_dinov2_npz(enc, "dinov2-base.npz", strict=True)
segmenter = DINOSegmenter(out_ch=1, encoder=enc, key=jr.fold_in(key, 2))
# logits, state = segmenter(x, key=jr.fold_in(key, 3), state=None)  # logits: [1,H,W]

"""
