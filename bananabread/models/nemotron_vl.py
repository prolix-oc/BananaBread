"""Adapter for NVIDIA Llama Nemotron Embed VL (text + image retrieval).

The VL model is a bi-encoder that supports image and text inputs for
document indexing, with text-only queries.  It uses the same retrieval
prompts (`query:` / `passage:`) as the original Nemotron embedding model.
"""

from typing import Any, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from bananabread.models.nemotron import (
    NemotronEmbeddingModel,
    _DOCUMENT_PREFIX,
    _QUERY_PREFIX,
)

NEMOTRON_VL_MODEL_REPO = "nvidia/llama-nemotron-embed-vl-1b-v2"
NEMOTRON_VL_DIMENSIONS = {128, 256, 384, 512, 768, 1024, 1536, 2048}
NEMOTRON_VL_BACKENDS = {
    "torch",
    "torch-bnb-8bit",
    "torch-bnb-4bit",
    "torch-metal-8bit",
    "torch-metal-4bit",
}


class NemotronVLEmbeddingModel(NemotronEmbeddingModel):
    """Adapter for NVIDIA's multimodal VL embedding model."""

    def __init__(
        self,
        model_path: str = NEMOTRON_VL_MODEL_REPO,
        *,
        truncate_dim: int = 2048,
        device: str = "cpu",
        backend: str = "torch",
        compute_dtype: str = "bfloat16",
        modalities: str = "text-image",
    ) -> None:
        if truncate_dim not in NEMOTRON_VL_DIMENSIONS:
            raise ValueError(
                f"Nemotron VL dimensions must be in {sorted(NEMOTRON_VL_DIMENSIONS)}"
            )
        if backend not in NEMOTRON_VL_BACKENDS:
            raise ValueError(f"Unsupported Nemotron VL backend: {backend}")
        if compute_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(
                "Nemotron VL requires bfloat16, float16, or float32"
            )
        # Delegate initialization to the Nemotron base adapter,
        # then override the model reference with the VL repo.
        super().__init__(
            model_path=model_path,
            truncate_dim=truncate_dim,
            device=device,
            backend=backend,
            compute_dtype=compute_dtype,
        )
        self.model_path = model_path
        self.modalities = modalities
        # The VL model supports image + text inputs.
        # We expose a simple supports() interface for the dispatcher.
        self.enabled_modalities = {"text", "image"}

    def supports(self, modality: str | tuple[str, ...]) -> bool:
        parts = (modality,) if isinstance(modality, str) else modality
        return all(part in self.enabled_modalities for part in parts)

    def _prefix_text(self, texts):
        prefixed = []
        for item in texts:
            if isinstance(item, str):
                prefixed.append(f"{_DOCUMENT_PREFIX}{item}")
            else:
                prefixed.append(item)
        return prefixed

    def encode(self, texts: Sequence[Any], *args: Any, **kwargs: Any) -> Any:
        # Apply `passage:` prefix only to pure text elements;
        # multimodal objects (image/text mix) pass through unchanged.
        return self.model.encode(self._prefix_text(texts), *args, **kwargs)

    def _prefix_query_text(self, texts):
        prefixed = []
        for item in texts:
            if isinstance(item, str):
                prefixed.append(f"{_QUERY_PREFIX}{item}")
            else:
                prefixed.append(item)
        return prefixed

    def encode_query(self, texts: Sequence[Any], *args: Any, **kwargs: Any) -> Any:
        return self.model.encode(self._prefix_query_text(texts), *args, **kwargs)

