#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import jax

from quantbayes.ball_dp.experiments.embedding_io import default_cache_dir
from quantbayes.ball_dp.experiments._text_embeddings_common import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    load_or_create_text_classification_embeddings,
    slugify_name,
)

DEFAULT_DATASET_ID = "Yelp/yelp_review_full"


def batch_to_texts(batch: dict[str, list]) -> list[str]:
    return [str(x) for x in batch["text"]]


def default_output_path(
    output_root: str = "./data",
    model_name: str = DEFAULT_MODEL_NAME,
) -> Path:
    model_slug = slugify_name(model_name)
    return (
        default_cache_dir(output_root) / f"yelp_review_full_{model_slug}_embeddings.npz"
    )


def load_or_create_yelp_review_full_text_embeddings(
    batch_size: int = 128,
    require_jax_gpu: bool = True,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    hf_cache_dir: str | None = None,
    output_root: str = "./data",
    cache_path: str | Path | None = None,
    force_recompute: bool = False,
    dataset_id: str = DEFAULT_DATASET_ID,
):
    if cache_path is None:
        cache_path = default_output_path(output_root=output_root, model_name=model_name)

    return load_or_create_text_classification_embeddings(
        cache_path=cache_path,
        force_recompute=force_recompute,
        dataset_id=dataset_id,
        train_split="train",
        test_split="test",
        batch_to_texts=batch_to_texts,
        label_field="label",
        batch_size=batch_size,
        require_jax_gpu=require_jax_gpu,
        model_name=model_name,
        max_length=max_length,
        hf_cache_dir=hf_cache_dir,
        train_desc="Yelp Review Full train",
        test_desc="Yelp Review Full test",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute/cache canonical-split Yelp Review Full text embeddings and save JAX-ready arrays."
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="./data")
    parser.add_argument("--dataset-id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Use CPU if JAX GPU is unavailable instead of raising an error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(
            output_root=args.output_root, model_name=args.model_name
        )
    )

    X_train, y_train, X_test, y_test = load_or_create_yelp_review_full_text_embeddings(
        batch_size=args.batch_size,
        require_jax_gpu=not args.allow_cpu_fallback,
        model_name=args.model_name,
        max_length=args.max_length,
        hf_cache_dir=args.hf_cache_dir,
        output_root=args.output_root,
        cache_path=output_path,
        force_recompute=args.force_recompute,
        dataset_id=args.dataset_id,
    )

    print(f"dataset_id: {args.dataset_id}")
    print(f"model: {args.model_name}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"type(X_train): {type(X_train)}")
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"X_test shape:  {X_test.shape}, dtype: {X_test.dtype}")
    print(f"y_test shape:  {y_test.shape}, dtype: {y_test.dtype}")
    print(f"Cache path: {output_path}")
    print()
    print("Notebook usage:")
    print("from quantbayes.ball_dp.embedding_io import load_embedding_npz")
    print(
        f"X_train, y_train, X_test, y_test = "
        f'load_embedding_npz(r"{output_path}", require_jax_gpu=False)'
    )


if __name__ == "__main__":
    main()
