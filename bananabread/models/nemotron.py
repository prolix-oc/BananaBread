"""Adapter for NVIDIA's Llama Nemotron Embedding model.

The model card specifies different prefixes for retrieval queries and
documents.  Keeping that detail here prevents callers of the normal
``encode`` API from accidentally producing unprefixed document embeddings.
"""

from typing import Any, Sequence

from sentence_transformers import SentenceTransformer


NEMOTRON_MODEL_REPO = "nvidia/llama-nemotron-embed-1b-v2"
_QUERY_PREFIX = "query: "
_DOCUMENT_PREFIX = "passage: "


class NemotronEmbeddingModel:
    """SentenceTransformers adapter with NVIDIA's retrieval prompts applied."""

    def __init__(self, model_path: str, *, truncate_dim: int, device: str) -> None:
        # This model ships a custom bidirectional-Llama architecture.  Only this
        # dedicated, NVIDIA-maintained integration opts into executing it; the
        # generic ``hf`` loader remains opt-in-free.
        self.model = SentenceTransformer(
            model_path,
            truncate_dim=truncate_dim,
            device=device,
            trust_remote_code=True,
        )

    def encode(self, texts: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        """Encode corpus documents using the model's required ``passage:`` prompt."""
        return self.model.encode(
            [f"{_DOCUMENT_PREFIX}{text}" for text in texts], *args, **kwargs
        )

    def encode_query(self, texts: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        """Encode retrieval queries using the model's required ``query:`` prompt."""
        return self.model.encode(
            [f"{_QUERY_PREFIX}{text}" for text in texts], *args, **kwargs
        )

    def __getattr__(self, name: str) -> Any:
        """Expose SentenceTransformers attributes used by warmup and compilation."""
        return getattr(self.model, name)
