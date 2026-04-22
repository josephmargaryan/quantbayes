#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jax
import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from quantbayes.ball_dp.experiments.embedding_io import (
    default_cache_dir,
    get_jax_device,
    load_embedding_npz,
    save_embedding_npz,
)

BatchToTexts = Callable[[dict[str, list]], list[str]]

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_LENGTH = 256


def slugify_name(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def build_text_embedding_cache_path(
    *,
    output_root: str | Path = "./data",
    dataset_id: str,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    dataset_config_name: str | None = None,
    dataset_revision: str | None = None,
    extra_name_parts: tuple[str, ...] = (),
) -> Path:
    name_parts = [slugify_name(dataset_id)]
    if dataset_config_name is not None:
        name_parts.append(slugify_name(dataset_config_name))
    if dataset_revision is not None:
        name_parts.append(slugify_name(dataset_revision))
    for part in extra_name_parts:
        if part:
            name_parts.append(slugify_name(part))
    name_parts.append(slugify_name(model_name))
    name_parts.append(f"len{max_length}")
    filename = "_".join(name_parts) + "_embeddings.npz"
    return default_cache_dir(output_root) / filename


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, a_min=eps, a_max=None)


def safe_text(x: object) -> str:
    return "" if x is None else str(x)


def join_nonempty_texts(*parts: object, sep: str = " ") -> str:
    cleaned: list[str] = []
    for part in parts:
        text = safe_text(part).strip()
        if text:
            cleaned.append(text)
    return sep.join(cleaned)


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    last_hidden_state: [B, T, D]
    attention_mask:    [B, T]
    returns:           [B, D]
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def build_text_encoder(
    model_name: str,
    torch_device: torch.device,
    hf_cache_dir: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir=hf_cache_dir,
    )
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=hf_cache_dir,
    )
    model.eval()
    model.to(torch_device)
    return tokenizer, model


def encode_text_batch(
    texts: list[str],
    tokenizer,
    model,
    torch_device: torch.device,
    max_length: int,
) -> np.ndarray:
    tokenizer_kwargs = dict(
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if torch_device.type == "cuda":
        tokenizer_kwargs["pad_to_multiple_of"] = 8

    encoded = tokenizer(texts, **tokenizer_kwargs)
    encoded = {
        k: v.to(torch_device, non_blocking=(torch_device.type == "cuda"))
        for k, v in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(**encoded)
        embeddings = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

    return embeddings.cpu().numpy().astype(np.float32, copy=False)


def extract_embeddings(
    dataset,
    batch_to_texts: BatchToTexts,
    label_field: str,
    tokenizer,
    model,
    torch_device: torch.device,
    batch_size: int,
    max_length: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    n = len(dataset)
    for start in tqdm(range(0, n, batch_size), desc=desc):
        stop = min(start + batch_size, n)
        batch = dataset[start:stop]

        texts = batch_to_texts(batch)
        labels = np.asarray(batch[label_field], dtype=np.int32)

        if len(texts) != len(labels):
            raise ValueError(
                f"Text/label length mismatch in batch [{start}:{stop}]: "
                f"{len(texts)} texts vs {len(labels)} labels"
            )

        features = encode_text_batch(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            torch_device=torch_device,
            max_length=max_length,
        )
        all_embeddings.append(features)
        all_labels.append(labels)

    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)
    X = l2_normalize_rows(X)
    return X, y


def _load_hf_split(
    dataset_id: str,
    split: str,
    hf_cache_dir: str | None,
    dataset_config_name: str | None,
    dataset_revision: str | None,
):
    kwargs: dict[str, object] = {
        "split": split,
        "cache_dir": hf_cache_dir,
    }
    if dataset_config_name is not None:
        kwargs["name"] = dataset_config_name
    if dataset_revision is not None:
        kwargs["revision"] = dataset_revision
    return load_dataset(dataset_id, **kwargs)


def load_text_classification_embeddings(
    *,
    dataset_id: str,
    train_split: str,
    test_split: str,
    batch_to_texts: BatchToTexts,
    label_field: str,
    batch_size: int = 128,
    require_jax_gpu: bool = True,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    hf_cache_dir: str | None = None,
    dataset_config_name: str | None = None,
    dataset_revision: str | None = None,
    train_desc: str = "train",
    test_desc: str = "test",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Compute embeddings from scratch and return JAX arrays.

    With the default encoder, X_* have shape [N, 384] and dtype float32.
    y_* have shape [N] and dtype int32.
    """
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jax_device = get_jax_device(require_jax_gpu)

    tokenizer, model = build_text_encoder(
        model_name=model_name,
        torch_device=torch_device,
        hf_cache_dir=hf_cache_dir,
    )

    train_dataset = _load_hf_split(
        dataset_id=dataset_id,
        split=train_split,
        hf_cache_dir=hf_cache_dir,
        dataset_config_name=dataset_config_name,
        dataset_revision=dataset_revision,
    )
    test_dataset = _load_hf_split(
        dataset_id=dataset_id,
        split=test_split,
        hf_cache_dir=hf_cache_dir,
        dataset_config_name=dataset_config_name,
        dataset_revision=dataset_revision,
    )

    X_train_np, y_train_np = extract_embeddings(
        dataset=train_dataset,
        batch_to_texts=batch_to_texts,
        label_field=label_field,
        tokenizer=tokenizer,
        model=model,
        torch_device=torch_device,
        batch_size=batch_size,
        max_length=max_length,
        desc=train_desc,
    )
    X_test_np, y_test_np = extract_embeddings(
        dataset=test_dataset,
        batch_to_texts=batch_to_texts,
        label_field=label_field,
        tokenizer=tokenizer,
        model=model,
        torch_device=torch_device,
        batch_size=batch_size,
        max_length=max_length,
        desc=test_desc,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X_train = jax.device_put(X_train_np, device=jax_device)
    y_train = jax.device_put(y_train_np, device=jax_device)
    X_test = jax.device_put(X_test_np, device=jax_device)
    y_test = jax.device_put(y_test_np, device=jax_device)
    return X_train, y_train, X_test, y_test


def load_or_create_text_classification_embeddings(
    *,
    cache_path: str | Path,
    force_recompute: bool = False,
    dataset_id: str,
    train_split: str,
    test_split: str,
    batch_to_texts: BatchToTexts,
    label_field: str,
    batch_size: int = 128,
    require_jax_gpu: bool = True,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    hf_cache_dir: str | None = None,
    dataset_config_name: str | None = None,
    dataset_revision: str | None = None,
    train_desc: str = "train",
    test_desc: str = "test",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_recompute:
        return load_embedding_npz(cache_path, require_jax_gpu=require_jax_gpu)

    X_train, y_train, X_test, y_test = load_text_classification_embeddings(
        dataset_id=dataset_id,
        train_split=train_split,
        test_split=test_split,
        batch_to_texts=batch_to_texts,
        label_field=label_field,
        batch_size=batch_size,
        require_jax_gpu=require_jax_gpu,
        model_name=model_name,
        max_length=max_length,
        hf_cache_dir=hf_cache_dir,
        dataset_config_name=dataset_config_name,
        dataset_revision=dataset_revision,
        train_desc=train_desc,
        test_desc=test_desc,
    )
    save_embedding_npz(cache_path, X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test
