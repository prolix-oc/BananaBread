"""Multimodal retrieval adapter for Jina Embeddings v5 Omni.

The omni family (small / nano) maps text, image, audio, and video into
a shared embedding space.  Adapter behavior mirrors EmbeddingGemma:
optional modality flags, Matryoshka truncation, and quantized backends.
"""

import logging
import threading
from typing import Any, Sequence

import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer

from bananabread.models.metal import enable_batched_metal_linears

JINA_OMNI_MODEL_REPOS = {
    "small": "jinaai/jina-embeddings-v5-omni-small",
    "nano": "jinaai/jina-embeddings-v5-omni-nano",
}
JINA_OMNI_DIMENSIONS = {32, 64, 128, 256, 512, 768, 1024}  # Matryoshka truncation
JINA_OMNI_BACKENDS = {"torch", "torch-bnb-4bit", "torch-metal-4bit", "torch-metal-8bit"}
logger = logging.getLogger("BananaBread-Emb")


class JinaOmniEmbeddingModel:
    """Multimodal adapter for Jina v5 Omni (small or nano)."""

    def __init__(
        self,
        model_path: str,
        *,
        truncate_dim: int = 1024,
        device: str = "cpu",
        backend: str = "torch",
        compute_dtype: str = "auto",
        modalities: str = "all",
    ) -> None:
        if truncate_dim not in JINA_OMNI_DIMENSIONS:
            raise ValueError(
                f"Jina Omni dimensions must be in {sorted(JINA_OMNI_DIMENSIONS)}"
            )
        if backend not in JINA_OMNI_BACKENDS:
            raise ValueError(f"Unsupported Jina Omni backend: {backend}")
        if compute_dtype not in {"auto", "bfloat16", "float32"}:
            raise ValueError(
                "Jina Omni requires bfloat16 or float32; float16 is unsafe"
            )
        if modalities not in {"all", "text", "text-image", "text-audio"}:
            raise ValueError(f"Unsupported Jina Omni modalities: {modalities}")

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
            trust_remote_code=True,  # Jina v5 uses custom modeling code
            config_kwargs=config_kwargs,
            model_kwargs=model_kwargs,
        )
        # Jina omni uses 32K token context; keep the adapter transparent.
        self.metal_layer_count = 0
        if backend.startswith("torch-metal"):
            self.metal_layer_count = enable_batched_metal_linears(self.model)
        self.tokenizer_lock = threading.RLock()
        self.tokenizer = self.model.tokenizer

    @staticmethod
    def _resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
        if dtype_name == "float32":
            return torch.float32
        if device.startswith("cuda"):
            with torch.cuda.device(device):
                supports_bf16 = torch.cuda.is_bf16_supported(
                    including_emulation=False
                )
        elif device.startswith("mps"):
            supports_bf16 = torch.backends.mps.is_macos_or_newer(14, 0)
        else:
            supports_bf16 = dtype_name == "bfloat16"
        if supports_bf16:
            return torch.bfloat16
        if dtype_name == "bfloat16":
            logger.warning(
                "Jina Omni: BF16 unavailable on %s; using float32", device
            )
        return torch.float32

    def _model_kwargs(self, device: str) -> dict[str, Any]:
        kwargs = {"dtype": self.compute_dtype}
        if self.backend == "torch-bnb-4bit":
            if not device.startswith("cuda"):
                raise ValueError(
                    "bitsandbytes Jina Omni requires a CUDA/HIP device"
                )
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
                raise ValueError(
                    "Metal Jina Omni backends require an MPS device"
                )
            bits = 4 if self.backend == "torch-metal-4bit" else 8
            kwargs.update(
                device_map={"": device},
                quantization_config=transformers.MetalConfig(
                    bits=bits, group_size=64
                ),
            )
        return kwargs

    def supports(
        self, modality: str | tuple[str, ...]
    ) -> bool:
        parts = (modality,) if isinstance(modality, str) else modality
        return all(part in self.enabled_modalities for part in parts) and self.model.supports(modality)

    def encode(self, texts: Sequence[Any], **kwargs: Any) -> Any:
        # Delegate to the model's sentence-transformers encode.
        # Jina v5 handles task prefixes (retrieval / text-match / etc.)
        # through the adapter's saved modules.json and prompt names.
        return self.model.encode(texts, **kwargs)

    def encode_query(self, texts: Sequence[Any], **kwargs: Any) -> Any:
        return self.model.encode_query(texts, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)
