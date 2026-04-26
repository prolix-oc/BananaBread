import sys
import threading
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from bananabread.config import args, logger


def _check_flash_attention_available() -> tuple[bool, str]:
    """Check whether Flash Attention 2 is installed and usable by Transformers."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False, "flash_attn is not installed in this Python environment"
    except Exception as e:
        import sys

        if sys.platform == 'win32':
            return False, (
                f"flash_attn failed to import on Windows: {e}. "
                "Run `uv run python install_flash_attn.py` from the repo to install "
                "a matching precompiled wheel, or install a wheel manually (see README)."
            )
        return False, f"flash_attn failed to import: {e}"

    try:
        from transformers.utils import is_flash_attn_2_available
    except ImportError:
        return True, "flash_attn imports, but this Transformers version cannot validate FA2 availability"

    if not is_flash_attn_2_available():
        return False, "Transformers reports Flash Attention 2 is unavailable for this runtime"

    return True, "available"


FLASH_ATTENTION_AVAILABLE, FLASH_ATTENTION_STATUS = _check_flash_attention_available()


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def _as_numpy(embeddings) -> np.ndarray:
    if isinstance(embeddings, np.ndarray):
        return embeddings
    if hasattr(embeddings, "detach"):
        if embeddings.dtype == torch.bfloat16:
            embeddings = embeddings.to(torch.float32)
        return embeddings.detach().cpu().numpy()
    return np.asarray(embeddings)


class BaseQwenModel:
    """Shared Qwen embedding/reranking behavior for all runtime backends."""

    backend_name = "base"

    def __init__(self, model_name: str, max_length: int = 8192):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer_lock = threading.RLock()

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    def encode(self, sentences: List[str], prompt_name: str = None, batch_size: int = 8, **kwargs):
        return self.get_embeddings(sentences, batch_size=batch_size)

    def tokenize(self, batch: List[str], return_tensors: str):
        # Fast tokenizers mutate truncation/padding state during __call__ and can
        # raise "Already borrowed" if shared across concurrent requests.
        with self.tokenizer_lock:
            return self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors=return_tensors,
                max_length=self.max_length,
            )

    def rank(self, query: str, documents: list[str], return_documents: bool = False, top_k: int = None, task_description: str = None) -> dict:
        task = task_description if task_description else 'Given a web search query, retrieve relevant passages that answer the query'
        formatted_query = self.get_detailed_instruct(task, query)

        query_emb = _as_numpy(self.get_embeddings([formatted_query]))
        candidate_embs = _as_numpy(self.get_embeddings(documents))
        scores = (query_emb @ candidate_embs.T)[0]

        top_indices = np.argsort(-scores)
        if top_k is not None:
            top_indices = top_indices[:top_k]

        results = []
        for idx in top_indices.tolist():
            result = {
                "index": idx,
                "score": float(scores[idx])
            }
            if return_documents:
                result["document"] = documents[idx]
            results.append(result)

        return {"results": results}


class QwenTorchModel(BaseQwenModel):
    """Qwen model implementation using raw Transformers AutoModel/AutoTokenizer."""

    backend_name = "torch"

    def __init__(
        self,
        model_name: str,
        device_arg: str = "cpu",
        use_flash_attention: bool = False,
        compute_dtype: str = "bfloat16",
        max_length: int = 8192,
    ):
        self.device_arg = device_arg
        self.compute_dtype = _torch_dtype(compute_dtype)

        logger.info(f"Loading Qwen torch model: {model_name}")
        super().__init__(model_name, max_length=max_length)

        kwargs = self._attention_kwargs(use_flash_attention, device_arg, self.compute_dtype)
        device_map = self._device_map(device_arg)

        if device_map:
            self.model = AutoModel.from_pretrained(
                model_name,
                dtype=self.compute_dtype,
                device_map=device_map,
                **kwargs
            )
            self.device = self.model.device
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                dtype=self.compute_dtype,
                **kwargs
            )
            if device_arg.lower() != "cpu":
                self.model.to(device_arg)
            self.device = self.model.device

        self.model.eval()
        logger.info(f"Qwen torch model initialized on {self.device} with padding_side='left'")

    @staticmethod
    def _attention_kwargs(use_flash_attention: bool, device_arg: str, compute_dtype: torch.dtype = torch.bfloat16) -> dict:
        kwargs = {}
        if use_flash_attention:
            if device_arg.lower() == "cpu":
                logger.warning("Flash Attention 2 requested for CPU. Falling back to default attention implementation.")
                return kwargs
            if not torch.cuda.is_available():
                logger.warning("Flash Attention 2 requested, but CUDA is not available. Falling back to default attention implementation.")
                return kwargs
            if compute_dtype not in (torch.float16, torch.bfloat16):
                logger.warning("Flash Attention 2 requires float16 or bfloat16 compute. Falling back to default attention implementation.")
                return kwargs
            if FLASH_ATTENTION_AVAILABLE:
                kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 enabled for Qwen model")
            else:
                logger.warning(
                    "Flash Attention 2 requested but unavailable. "
                    f"Reason: {FLASH_ATTENTION_STATUS}. "
                    "Falling back to default attention implementation."
                )
        elif device_arg.lower() != "cpu" and not FLASH_ATTENTION_AVAILABLE:
            logger.info(
                "Flash Attention 2 is not installed. For improved GPU performance with Qwen models, "
                f"consider running: uv run python install_flash_attn.py. Detection reason: {FLASH_ATTENTION_STATUS}"
            )
        return kwargs

    @staticmethod
    def _device_map(device_arg: str):
        if device_arg.lower() == "cpu":
            return None
        if device_arg.lower() in ["auto", "cuda"]:
            return "auto"
        return None

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_embeddings(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenize(batch, return_tensors="pt").to(self.model.device)

            with torch.inference_mode():
                outputs = self.model(**inputs)
                batch_embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings.append(batch_embeddings)

        if not embeddings:
            return torch.tensor([])

        all_embeddings = torch.cat(embeddings, dim=0)
        return F.normalize(all_embeddings, p=2, dim=1)


class QwenBnbModel(QwenTorchModel):
    """Qwen torch backend with bitsandbytes weight quantization for CUDA inference."""

    def __init__(
        self,
        model_name: str,
        device_arg: str = "cuda",
        quantization_bits: int = 8,
        use_flash_attention: bool = False,
        compute_dtype: str = "bfloat16",
        max_length: int = 8192,
    ):
        if device_arg.lower() == "cpu":
            raise ValueError("bitsandbytes Qwen backends require a CUDA device, not CPU")

        self.backend_name = f"torch-bnb-{quantization_bits}bit"
        self.device_arg = device_arg
        self.compute_dtype = _torch_dtype(compute_dtype)

        # Handle matmul_cast_fp16 config for 8-bit quantization
        if quantization_bits == 8:
            if args.matmul_cast_fp16:
                self.compute_dtype = torch.float16
            else:
                # Suppress the bitsandbytes cast warning when keeping bfloat16/fp32
                warnings.filterwarnings(
                    "ignore",
                    message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization",
                    category=UserWarning,
                )

        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes quantization requires the cuda-quant extra: "
                "uv pip install bananabread-emb[cuda-quant]"
            ) from exc

        # Flash Attention 2 + bitsandbytes on Windows is known to cause
        # torch.AcceleratorError: CUDA error: unknown error during inference.
        # Force SDPA fallback on Windows when BnB quantization is active.
        if sys.platform == 'win32' and use_flash_attention:
            logger.warning(
                "Flash Attention 2 is disabled for bitsandbytes backends on Windows "
                "to avoid a known CUDA compatibility issue (torch.AcceleratorError). "
                "Falling back to sdpa attention."
            )
            use_flash_attention = False

        logger.info(f"Loading Qwen {quantization_bits}-bit bitsandbytes model: {model_name}")
        BaseQwenModel.__init__(self, model_name, max_length=max_length)

        kwargs = self._attention_kwargs(use_flash_attention, device_arg, self.compute_dtype)
        device_map = "auto" if device_arg.lower() in ["auto", "cuda"] else {"": device_arg}

        if quantization_bits == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
            )

        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=self.compute_dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            **kwargs,
        )
        self.device = self.model.device
        self.model.eval()
        logger.info(f"Qwen bitsandbytes model initialized on {self.device} with backend={self.backend_name}")


class QwenOnnxModel(BaseQwenModel):
    """Qwen embedding backend using a local ONNX Runtime model."""

    backend_name = "onnx-int8"

    def __init__(
        self,
        model_name: str,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        max_length: int = 8192,
    ):
        if not model_path:
            raise ValueError("--qwen-onnx-model-path is required when --qwen-backend=onnx-int8")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX Qwen backend requires the onnx extra: uv pip install bananabread-emb[onnx]"
            ) from exc

        logger.info(f"Loading Qwen ONNX model from: {model_path}")
        super().__init__(model_name, max_length=max_length)

        resolved_model_path = self._resolve_onnx_path(model_path)
        session_options = ort.SessionOptions()
        session_options.optimized_model_filepath = str(self._optimized_onnx_path(resolved_model_path))
        # Limit ONNX Runtime threads to 1 to avoid OpenMP contention with our own threadpool.
        # BananaBread already manages concurrency via threadpool executors; letting ONNX
        # spawn its own thread army causes livelock/hang during batch requests.
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        available_providers = ort.get_available_providers()
        providers = [provider] if provider in available_providers else ["CPUExecutionProvider"]
        if provider not in available_providers:
            logger.warning(f"ONNX provider {provider} is unavailable. Falling back to CPUExecutionProvider")

        self.session = ort.InferenceSession(str(resolved_model_path), sess_options=session_options, providers=providers)
        self.input_names = [input_meta.name for input_meta in self.session.get_inputs()]
        self.device = providers[0]
        logger.info(f"Qwen ONNX model initialized with provider={providers[0]} (intra_threads=1, inter_threads=1)")

    @staticmethod
    def _resolve_onnx_path(model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if path.is_file():
            return path
        if not path.exists():
            if path.suffix.lower() == ".onnx":
                path.parent.mkdir(parents=True, exist_ok=True)
                raise FileNotFoundError(
                    f"ONNX model file does not exist: {path}. Created parent directory if needed."
                )
            path.mkdir(parents=True, exist_ok=True)
            raise FileNotFoundError(
                f"No .onnx model found in directory: {path}. Created directory if needed."
            )
        candidates = sorted(path.glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"No .onnx model found in directory: {model_path}")
        if len(candidates) > 1:
            preferred = path / "model.onnx"
            if preferred.exists():
                return preferred
        return candidates[0]

    @staticmethod
    def _optimized_onnx_path(model_path: Path) -> Path:
        optimized_dir = model_path.parent / "optimized"
        optimized_dir.mkdir(parents=True, exist_ok=True)
        return optimized_dir / model_path.name

    @staticmethod
    def last_token_pool(last_hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(axis=1) - 1
        return last_hidden_states[np.arange(last_hidden_states.shape[0]), sequence_lengths]

    def _prepare_onnx_inputs(self, encoded: dict) -> dict:
        ort_inputs = {}
        for name in self.input_names:
            if name in encoded:
                ort_inputs[name] = encoded[name].astype(np.int64)
            elif name == "position_ids":
                seq_len = encoded["input_ids"].shape[1]
                ort_inputs[name] = np.tile(np.arange(seq_len, dtype=np.int64), (encoded["input_ids"].shape[0], 1))
            elif name == "token_type_ids":
                ort_inputs[name] = np.zeros_like(encoded["input_ids"], dtype=np.int64)
            else:
                raise ValueError(f"Unsupported ONNX input '{name}' for Qwen model")
        return ort_inputs

    def get_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenize(batch, return_tensors="np")
            attention_mask = encoded["attention_mask"]
            outputs = self.session.run(None, self._prepare_onnx_inputs(encoded))
            last_hidden_state = outputs[0]

            # Validate output shape: should be (batch_size, seq_len, hidden_dim)
            if last_hidden_state.ndim != 3:
                raise ValueError(
                    f"ONNX model returned unexpected hidden-state shape {last_hidden_state.shape}. "
                    f"Expected 3-D (batch_size, seq_len, hidden_dim)."
                )
            if last_hidden_state.shape[0] != len(batch):
                raise ValueError(
                    f"ONNX model batch dimension mismatch: got {last_hidden_state.shape[0]}, expected {len(batch)}. "
                    f"The model may have been exported with a fixed batch size."
                )

            batch_embeddings = self.last_token_pool(last_hidden_state, attention_mask)
            embeddings.append(batch_embeddings)

        if not embeddings:
            return np.array([])

        all_embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
        norms = np.linalg.norm(all_embeddings, ord=2, axis=1, keepdims=True)
        return all_embeddings / np.maximum(norms, 1e-12)


QwenRawModel = QwenTorchModel


def load_qwen_model(
    model_name: str,
    backend: str = "torch",
    device_arg: str = "cpu",
    use_flash_attention: bool = False,
    compute_dtype: str = "bfloat16",
    onnx_model_path: str = None,
    onnx_provider: str = "CPUExecutionProvider",
    max_length: int = 8192,
):
    if backend == "torch":
        return QwenTorchModel(
            model_name,
            device_arg=device_arg,
            use_flash_attention=use_flash_attention,
            compute_dtype=compute_dtype,
            max_length=max_length,
        )
    if backend == "torch-bnb-8bit":
        return QwenBnbModel(
            model_name,
            device_arg=device_arg,
            quantization_bits=8,
            use_flash_attention=use_flash_attention,
            compute_dtype=compute_dtype,
            max_length=max_length,
        )
    if backend == "torch-bnb-4bit":
        return QwenBnbModel(
            model_name,
            device_arg=device_arg,
            quantization_bits=4,
            use_flash_attention=use_flash_attention,
            compute_dtype=compute_dtype,
            max_length=max_length,
        )
    if backend == "onnx-int8":
        return QwenOnnxModel(
            model_name,
            model_path=onnx_model_path,
            provider=onnx_provider,
            max_length=max_length,
        )
    raise ValueError(f"Unsupported Qwen backend: {backend}")
