"""Adapter for Microsoft's Harrier OSS text embedding model.

Harrier is a decoder-only text embedding model using last-token pooling with
L2 normalization. It loads as a standard SentenceTransformers checkpoint with
no special retrieval prompts.
"""

import threading
from typing import Any, Sequence

import torch
from sentence_transformers import SentenceTransformer

from bananabread.bnb_logging import install_bnb_matmul_cast_filter
from bananabread.config import args
from bananabread.models.metal import enable_batched_metal_linears


HARRIER_MODEL_REPO = "microsoft/harrier-oss-v1-0.6b"
HARRIER_DIMENSIONS = {1024}  # Fixed 1024-dim
HARRIER_BACKENDS = {"torch", "torch-bnb-8bit", "torch-bnb-4bit", "torch-metal-8bit", "torch-metal-4bit"}


class HarrierEmbeddingModel:
    """SentenceTransformers adapter for Microsoft Harrier OSS."""

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
        # bitsandbytes 8-bit matmuls quantize their inputs to FP16.  Keep the
        # model compute dtype aligned when requested so it does not repeatedly
        # warn about casting BF16 activations at inference time.
        if self.backend == "torch-bnb-8bit":
            install_bnb_matmul_cast_filter()
            if args.matmul_cast_fp16:
                self.compute_dtype = torch.float16
        model_kwargs = self._model_kwargs(device)

        self.model = SentenceTransformer(
            model_path,
            truncate_dim=truncate_dim,
            # With a device_map in model_kwargs, accelerate owns placement;
            # passing `device` too only triggers SentenceTransformers'
            # "device argument is ignored" warning, so omit it in that case.
            device=None if "device_map" in model_kwargs else device,
            trust_remote_code=False,
            model_kwargs=model_kwargs,
        )
        self.model.max_seq_length = 32768
        self.metal_layer_count = 0
        if backend.startswith("torch-metal"):
            self.metal_layer_count = enable_batched_metal_linears(self.model)
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
        if self.backend in {"torch-metal-8bit", "torch-metal-4bit"}:
            if not device.lower().startswith("mps"):
                raise ValueError("Metal Harrier backends require an MPS device")
            try:
                from transformers import MetalConfig
            except ImportError as exc:
                raise ImportError(
                    "Metal quantization requires Transformers 5.15 or newer and the "
                    "metal-quant extra: uv pip install bananabread-emb[metal-quant]"
                ) from exc

            bits = 8 if self.backend == "torch-metal-8bit" else 4
            return {
                "dtype": self.compute_dtype,
                "device_map": {"": device},
                "quantization_config": MetalConfig(bits=bits, group_size=64),
            }
        if self.backend not in {"torch-bnb-8bit", "torch-bnb-4bit"}:
            raise ValueError(f"Unsupported Harrier backend: {self.backend}")
        if device.lower() != "auto" and not device.lower().startswith("cuda"):
            raise ValueError("bitsandbytes Harrier backends require a CUDA/HIP device")

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

    batch_prompt_names = {"document": None, "query": None}

    def encode(self, texts: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        with self.tokenizer_lock:
            return self.model.encode(texts, *args, **kwargs)

    def encode_query(self, texts: Sequence[str], *args: Any, **kwargs: Any) -> Any:
        with self.tokenizer_lock:
            return self.model.encode(texts, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Expose SentenceTransformers attributes used by warmup and compilation."""
        return getattr(self.model, name)
