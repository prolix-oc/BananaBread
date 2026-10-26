"""Metadata and vector validation for Ollama-compatible clients."""

from numbers import Integral
from typing import Any

import numpy as np
import torch


def _positive_int(value: Any) -> int | None:
    if isinstance(value, Integral) and not isinstance(value, bool) and value > 0:
        return int(value)
    return None


def embedding_model_info(model: Any) -> dict[str, int]:
    """Describe effective runtime limits without loading weights or running inference.

    SentenceTransformers reports dimensions after truncation. Raw Qwen runtimes
    instead return their native hidden size, independent of embedding_dim.
    Unknown values are omitted rather than guessed from server-wide defaults.
    """
    dimensions = None
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, name, None)
        if callable(getter):
            dimensions = _positive_int(getter())
            if dimensions:
                break
    if dimensions is None:
        dimensions = _positive_int(getattr(model, "truncate_dim", None))
    if dimensions is None:
        runtime = getattr(model, "model", model)
        config = getattr(runtime, "config", None)
        dimensions = _positive_int(getattr(config, "hidden_size", None))
    if dimensions is None:
        session = getattr(model, "session", None)
        if session is not None:
            outputs = session.get_outputs()
            if outputs and outputs[0].shape:
                dimensions = _positive_int(outputs[0].shape[-1])

    context_length = None
    for name in ("max_seq_length", "max_length"):
        context_length = _positive_int(getattr(model, name, None))
        if context_length:
            break
    if context_length is None:
        tokenizer = getattr(model, "tokenizer", None)
        limit = _positive_int(getattr(tokenizer, "model_max_length", None))
        # HF uses enormous sentinel values when the context limit is unknown.
        if limit and limit < 1_000_000_000:
            context_length = limit

    info = {}
    if dimensions:
        info["embedding.embedding_length"] = dimensions
    if context_length:
        info["embedding.context_length"] = context_length
    return info


def checked_embedding_vectors(output: Any, count: int) -> np.ndarray:
    """Fail explicitly instead of returning a successful but unusable batch."""
    if isinstance(output, torch.Tensor):
        output = output.detach().float().cpu().numpy()
    vectors = np.asarray(output)
    if vectors.ndim != 2 or vectors.shape[0] != count or vectors.shape[1] == 0:
        raise RuntimeError(
            f"Embedding model returned shape {vectors.shape}; expected {count} nonempty vectors"
        )
    if not np.issubdtype(vectors.dtype, np.number) or np.iscomplexobj(vectors):
        raise RuntimeError("Embedding model returned non-numeric vectors")
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embedding model returned non-finite vectors")
    if not np.any(vectors != 0, axis=1).all():
        raise RuntimeError("Embedding model returned an all-zero vector")
    return vectors
