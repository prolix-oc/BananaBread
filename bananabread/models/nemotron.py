"""Adapter for NVIDIA's Llama Nemotron Embedding model.

The model card specifies different prefixes for retrieval queries and
documents.  Keeping that detail here prevents callers of the normal
``encode`` API from accidentally producing unprefixed document embeddings.
"""

import threading
from typing import Any, Sequence

import torch
from sentence_transformers import SentenceTransformer


NEMOTRON_MODEL_REPO = "nvidia/llama-nemotron-embed-1b-v2"
_QUERY_PREFIX = "query: "
_DOCUMENT_PREFIX = "passage: "


class NemotronEmbeddingModel:
    """SentenceTransformers adapter with NVIDIA's retrieval prompts applied."""

    def __init__(
        self,
        model_path: str,
        *,
        truncate_dim: int,
        device: str,
        backend: str = "torch",
        compute_dtype: str = "bfloat16",
    ) -> None:
        self.backend = backend
        self.compute_dtype = self._torch_dtype(compute_dtype)
        model_kwargs = self._model_kwargs(device)

        # This model ships a custom bidirectional-Llama architecture.  Only this
        # dedicated, NVIDIA-maintained integration opts into executing it; the
        # generic ``hf`` loader remains opt-in-free.
        self.model = SentenceTransformer(
            model_path,
            truncate_dim=truncate_dim,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        # Fast Hugging Face tokenizers mutate internal state during tokenization.
        # SentenceTransformers does not lock that state, so concurrent requests
        # against one model instance can otherwise fail with "Already borrowed".
        self.tokenizer_lock = threading.RLock()
        try:
            self.tokenizer = self.model[0].tokenizer
        except (AttributeError, IndexError, TypeError):
            self.tokenizer = None

    @staticmethod
    def _torch_dtype(dtype_name: str) -> torch.dtype:
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "float32":
            return torch.float32
        return torch.bfloat16

    def _model_kwargs(self, device: str) -> dict[str, Any]:
        if self.backend == "torch":
            return {}
        if self.backend not in {"torch-bnb-8bit", "torch-bnb-4bit"}:
            raise ValueError(f"Unsupported Nemotron backend: {self.backend}")
        if device.lower() != "auto" and not device.lower().startswith("cuda"):
            raise ValueError("bitsandbytes Nemotron backends require a CUDA device")

        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes quantization requires the cuda-quant extra: "
                "uv pip install bananabread-emb[cuda-quant]"
            ) from exc

        if self.backend == "torch-bnb-8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
            )

        device_map: str | dict[str, str]
        device_map = "auto" if device.lower() in {"auto", "cuda"} else {"": device}
        return {
            "dtype": self.compute_dtype,
            "device_map": device_map,
            "quantization_config": quantization_config,
        }

    def encode(self, texts: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        """Encode corpus documents using the model's required ``passage:`` prompt."""
        with self.tokenizer_lock:
            return self.model.encode(
                [f"{_DOCUMENT_PREFIX}{text}" for text in texts], *args, **kwargs
            )

    def encode_query(self, texts: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        """Encode retrieval queries using the model's required ``query:`` prompt."""
        with self.tokenizer_lock:
            return self.model.encode(
                [f"{_QUERY_PREFIX}{text}" for text in texts], *args, **kwargs
            )

    def __getattr__(self, name: str) -> Any:
        """Expose SentenceTransformers attributes used by warmup and compilation."""
        return getattr(self.model, name)
