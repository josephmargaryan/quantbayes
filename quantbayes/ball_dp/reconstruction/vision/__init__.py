# quantbayes/ball_dp/reconstruction/vision/__init__.py
from __future__ import annotations

from .datasets import (
    load_mnist_numpy,
    load_cifar10_numpy,
    PairedTransformDataset,
    make_mnist_paired_loader_for_resnet,
    make_cifar10_paired_loader_for_resnet,
    extract_embeddings_and_targets,
)

from .preprocess import (
    flatten_chw,
    unflatten_mnist,
    unflatten_cifar10,
    pixel_l2_bound_unit_box,
    clip01,
)

from .plotting import (
    save_recon_grid,
    save_single_grid,
)

from .encoders_torch import (
    build_resnet18_embedder,
)

from .decoders_eqx import (
    DecoderMLP,
    decode_batch,
)

from .train_decoder_eqx import (
    train_decoder_mlp,
    decode_images_from_embeddings,
)
