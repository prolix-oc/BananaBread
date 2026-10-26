"""Multimodal retrieval with EmbeddingGemma 2 and optional weight quantization."""

import logging
import threading
from typing import Any, Sequence

import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer

from bananabread.models.metal import enable_batched_metal_linears

EMBEDDINGGEMMA_MODEL_REPO = "google/embeddinggemma-2"
EMBEDDINGGEMMA_DIMENSIONS = {128, 256, 512, 768}
EMBEDDINGGEMMA_BACKENDS = {"torch", "torch-bnb-4bit", "torch-metal-4bit", "torch-metal-8bit"}
logger = logging.getLogger("BananaBread-Emb")


class EmbeddingGemmaModel:
    """Apply retrieval prompts and normalize after Matryoshka truncation.

    Unused media towers can be omitted. FP16 is unsafe for this model, including
    bitsandbytes INT8's internal activation cast. NF4 keeps BF16/FP32 compute;
    Metal quantizes only weights.
    """

    def __init__(
        self,
        model_path: str,
        *,
        truncate_dim: int = 768,
        device: str = "cpu",
        backend: str = "torch",
        compute_dtype: str = "auto",
        modalities: str = "all",
    ) -> None:
        if truncate_dim not in EMBEDDINGGEMMA_DIMENSIONS:
            raise ValueError("EmbeddingGemma 2 dimensions must be 128, 256, 512, or 768")
        if backend not in EMBEDDINGGEMMA_BACKENDS:
            raise ValueError(f"Unsupported EmbeddingGemma backend: {backend}")
        if compute_dtype not in {"auto", "bfloat16", "float32"}:
            raise ValueError("EmbeddingGemma 2 requires bfloat16 or float32; float16 is unsafe")
        if modalities not in {"all", "text", "text-image", "text-audio"}:
            raise ValueError(f"Unsupported EmbeddingGemma modalities: {modalities}")

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.backend = backend
        self.truncate_dim = truncate_dim
        self.enabled_modalities = {"text"}
        if modalities in {"all", "text-image"}:
            self.enabled_modalities.add("image")
        if modalities in {"all", "text-audio"}:
            self.enabled_modalities.add("audio")
        self.compute_dtype = self._resolve_dtype(compute_dtype, device)
        model_kwargs = self._model_kwargs(device)
        config_kwargs = {}
        if "image" not in self.enabled_modalities:
            config_kwargs["vision_config"] = None
        if "audio" not in self.enabled_modalities:
            config_kwargs["audio_config"] = None
        self.model = SentenceTransformer(
            model_path,
            truncate_dim=truncate_dim,
            device=None if "device_map" in model_kwargs else device,
            trust_remote_code=False,
            config_kwargs=config_kwargs,
            model_kwargs=model_kwargs,
        )
        self.model.max_seq_length = 8192
        self.metal_layer_count = 0
        if backend.startswith("torch-metal"):
            self.metal_layer_count = enable_batched_metal_linears(self.model)
        # Share this lock with API token accounting: fast tokenizers mutate state.
        self.tokenizer_lock = threading.RLock()
        self.tokenizer = self.model.tokenizer

    @staticmethod
    def _resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
        if dtype_name == "float32":
            return torch.float32
        if device.startswith("cuda"):
            # Check the selected accelerator, rather than implicitly checking GPU 0.
            with torch.cuda.device(device):
                supports_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        elif device.startswith("mps"):
            supports_bf16 = torch.backends.mps.is_macos_or_newer(14, 0)
        else:
            # FP32 is the portable CPU default. Explicit BF16 remains available.
            supports_bf16 = dtype_name == "bfloat16"
        if supports_bf16:
            return torch.bfloat16
        if dtype_name == "bfloat16":
            logger.warning("EmbeddingGemma: BF16 unavailable on %s; using float32", device)
        return torch.float32

    def _model_kwargs(self, device: str) -> dict[str, Any]:
        kwargs = {"dtype": self.compute_dtype}
        if self.backend == "torch-bnb-4bit":
            if not device.startswith("cuda"):
                raise ValueError("bitsandbytes EmbeddingGemma requires a CUDA/HIP device")
            kwargs.update(
                device_map={"": device},
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=self.compute_dtype,
                ),
            )
        elif self.backend.startswith("torch-metal"):
            if not device.startswith("mps"):
                raise ValueError("Metal EmbeddingGemma backends require an MPS device")
            bits = 4 if self.backend == "torch-metal-4bit" else 8
            kwargs.update(
                device_map={"": device},
                quantization_config=transformers.MetalConfig(bits=bits, group_size=64),
            )
        return kwargs

    def supports(self, modality: str | tuple[str, ...]) -> bool:
        parts = (modality,) if isinstance(modality, str) else modality
        return all(part in self.enabled_modalities for part in parts) and self.model.supports(
            modality
        )

    def _encode(self, texts: Sequence[Any], prompt_name: str, **kwargs: Any) -> Any:
        # Task prefixes apply to text only. The shared dispatcher separates
        # heterogeneous inputs, so media-only batches can explicitly disable prompts.
        has_text = any(
            isinstance(item, str) or (isinstance(item, dict) and item.get("text")) for item in texts
        )
        prompt_kwargs = {"prompt_name": prompt_name} if has_text else {"prompt": ""}
        with self.tokenizer_lock:
            embeddings = self.model.encode(
                texts,
                **{
                    **kwargs,
                    **prompt_kwargs,
                    "truncate_dim": self.truncate_dim,
                    "normalize_embeddings": True,
                },
            )
        finite = (
            torch.isfinite(embeddings).all()
            if isinstance(embeddings, torch.Tensor)
            else np.isfinite(embeddings).all()
        )
        if not finite:
            raise RuntimeError(
                "EmbeddingGemma produced non-finite embeddings; check compute precision"
            )
        # ST may normalize in BF16 before converting its output to NumPy. Finish
        # in FP32 so returned Matryoshka vectors retain unit length accurately.
        if isinstance(embeddings, torch.Tensor):
            embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=-1)
        else:
            embeddings = np.asarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, np.finfo(np.float32).tiny)
        return embeddings

    def encode(self, texts: Sequence[Any], **kwargs: Any) -> Any:
        return self._encode(texts, "Document", **kwargs)

    def encode_query(self, texts: Sequence[Any], **kwargs: Any) -> Any:
        return self._encode(texts, "SearchQuery", **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)
