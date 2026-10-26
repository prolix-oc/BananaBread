"""Native text and image retrieval with NVIDIA Nemotron 3.5 Embed."""

from typing import Any, Sequence

import numpy as np
import torch

from bananabread.models.nemotron import Nemotron3EmbeddingModel

NEMOTRON_35_MODEL_REPOS = {
    "1B": "nvidia/Nemotron-3.5-Embed-1B-BF16",
    "8B": "nvidia/Nemotron-3.5-Embed-8B-BF16",
}
NEMOTRON_35_DIMENSIONS = {"1B": 2048, "8B": 4096}


class Nemotron35EmbeddingModel(Nemotron3EmbeddingModel):
    """Use the checkpoint's chat template and saved query/document prompts.

    The native SentenceTransformers pipeline handles text, images, and combined
    documents. Retrieval queries accept text only. Weight backends and tokenizer
    locking are shared with Nemotron-3; no remote code is needed.
    """

    def __init__(
        self,
        model_path: str,
        *,
        truncate_dim: int = 1024,
        device: str = "cpu",
        size: str = "1B",
        backend: str = "torch",
        compute_dtype: str = "bfloat16",
    ) -> None:
        if size not in NEMOTRON_35_DIMENSIONS:
            raise ValueError("Nemotron-3.5 size must be '1B' or '8B'")
        if not 1 <= truncate_dim <= NEMOTRON_35_DIMENSIONS[size]:
            raise ValueError(
                f"Nemotron-3.5 {size} dimensions must be between 1 and "
                f"{NEMOTRON_35_DIMENSIONS[size]}"
            )
        super().__init__(
            model_path,
            truncate_dim=truncate_dim,
            device=device,
            backend=backend,
            compute_dtype=compute_dtype,
        )
        self.truncate_dim = truncate_dim
        self.model.max_seq_length = 32768

    def supports(self, modality: str | tuple[str, ...]) -> bool:
        parts = (modality,) if isinstance(modality, str) else modality
        return all(part in {"text", "image"} for part in parts) and self.model.supports(modality)

    def _encode(self, inputs: Sequence[Any], *, query: bool, **kwargs: Any) -> Any:
        if query and any(
            not isinstance(item, str) and not (isinstance(item, dict) and set(item) == {"text"})
            for item in inputs
        ):
            raise ValueError("Nemotron-3.5 retrieval queries support text only")
        with self.tokenizer_lock:
            encode = self.model.encode_query if query else self.model.encode_document
            output = encode(
                inputs,
                **{**kwargs, "truncate_dim": self.truncate_dim, "normalize_embeddings": True},
            )
        # Finish normalization in FP32: the checkpoint's Normalize module may
        # run in BF16, and Matryoshka truncation changes the vector norm.
        if isinstance(output, torch.Tensor):
            if not torch.isfinite(output).all():
                raise RuntimeError("Nemotron-3.5 produced non-finite embeddings")
            return torch.nn.functional.normalize(output.float(), p=2, dim=-1)
        output = np.asarray(output, dtype=np.float32)
        if not np.isfinite(output).all():
            raise RuntimeError("Nemotron-3.5 produced non-finite embeddings")
        norms = np.linalg.norm(output, axis=-1, keepdims=True)
        return output / np.maximum(norms, np.finfo(np.float32).tiny)

    def encode(self, inputs: Sequence[Any], **kwargs: Any) -> Any:
        return self._encode(inputs, query=False, **kwargs)

    def encode_query(self, inputs: Sequence[Any], **kwargs: Any) -> Any:
        return self._encode(inputs, query=True, **kwargs)
