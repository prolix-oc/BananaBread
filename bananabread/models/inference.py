"""Shared embedding batch limits and opt-in Transformers attention backends."""

import importlib
import sys
from typing import Any

import numpy as np
import torch
from transformers import AttentionInterface, AttentionMaskInterface, PreTrainedModel
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import bidirectional_mask_function, causal_mask_function, sdpa_mask
from transformers.utils import is_flash_attn_2_available

from bananabread.config import logger
from bananabread.models.attention import flash_attention_supported_device

COMPACT_SDPA = "bananabread_sdpa"


def compact_sdpa_mask(
    batch_size,
    q_length,
    kv_length,
    q_offset=0,
    kv_offset=0,
    mask_function=causal_mask_function,
    attention_mask=None,
    **kwargs,
):
    """Broadcast plain bidirectional padding masks; retain all other mask semantics."""
    if (
        mask_function is bidirectional_mask_function
        and q_offset == 0
        and kv_offset == 0
        and kwargs.get("local_size") is None
        and not kwargs.get("use_vmap", False)
        and (
            attention_mask is None
            or (attention_mask.ndim == 2 and attention_mask.shape[-1] == kv_length)
        )
    ):
        if attention_mask is None:
            return None
        # Every query attends to the same valid keys. Transformers has already
        # placed this 2D mask on the input device. Storage stays O(batch * length).
        return attention_mask[:, None, None, :].to(dtype=torch.bool)
    return sdpa_mask(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        **kwargs,
    )


# Register a separate backend. The standard SDPA registry entries are untouched,
# and only selected embedding model instances use this name.
AttentionInterface.register(COMPACT_SDPA, sdpa_attention_forward)
AttentionMaskInterface.register(COMPACT_SDPA, compact_sdpa_mask)


def configure_embedding_attention(model: Any, *, prefer_flash: bool = False) -> None:
    """Select backends by capability, including encoders inside multimodal models."""
    underlying = getattr(model, "model", model)
    if not isinstance(underlying, torch.nn.Module):
        return  # e.g. ONNX; batching is independent of the runtime backend.
    flash_available = prefer_flash and is_flash_attn_2_available()
    if flash_available:
        try:
            importlib.import_module("flash_attn")
            # The top-level import can succeed while varlen/padding helpers fail
            # (e.g. an incomplete installation without einops). Check before any
            # backbone is switched or Transformers caches partial FA2 imports.
            importlib.import_module("flash_attn.bert_padding")
        except Exception as exc:
            logger.warning(f"FlashAttention 2 failed to import: {exc}")
            flash_available = False
    if prefer_flash and not flash_available:
        logger.warning("FlashAttention 2 is unavailable; retaining supported attention backends")
    for native in underlying.modules():
        if not isinstance(native, PreTrainedModel) or not native._can_set_attn_implementation():
            continue
        current = native.config._attn_implementation
        if current not in {"sdpa", COMPACT_SDPA}:
            continue  # Preserve eager, existing FlashAttention, and custom backends.
        target = COMPACT_SDPA
        device = next(native.parameters(), torch.empty(0)).device
        if (
            flash_available
            and flash_attention_supported_device(device)
            and native.dtype in {torch.float16, torch.bfloat16}
            and getattr(native, "_supports_flash_attn", False)
            and not (
                sys.platform == "win32"
                and (
                    getattr(native, "is_loaded_in_4bit", False)
                    or getattr(native, "is_loaded_in_8bit", False)
                )
            )
        ):
            target = "flash_attention_2"
        # Preserve each backbone's current choice. The traversal handles those
        # independently, so a vision encoder need not support the text backend.
        choices = {"": target}
        choices.update(
            {
                key: getattr(native.config, key)._attn_implementation
                for key in native.config.sub_configs
                if getattr(native.config, key, None) is not None
            }
        )
        try:
            native.set_attn_implementation(choices)
        except (ValueError, ImportError) as exc:
            if target != "flash_attention_2":
                raise
            logger.warning(f"FlashAttention 2 unavailable for {type(native).__name__}: {exc}")
            native.set_attn_implementation({**choices, "": COMPACT_SDPA})
        logger.info(
            f"Embedding attention: {type(native).__name__} -> {native.config._attn_implementation}"
        )


def _padded_input_length(model: Any, text: str, input_type: str) -> int | None:
    """Measure the model's real preprocessing, including prompts and truncation."""
    prefixes = getattr(model, "batch_text_prefixes", {})
    text = prefixes.get(input_type, "") + text
    preprocess = getattr(model, "preprocess", None)
    if callable(preprocess):
        prompts = getattr(model, "prompts", {})
        names = getattr(model, "batch_prompt_names", {})
        name = names.get(input_type)
        if input_type not in names:
            name = "query" if input_type == "query" else getattr(model, "default_prompt_name", None)
        if name == "document" and name not in prompts:
            name = next((key for key in ("passage", "corpus") if key in prompts), None)
        if name not in prompts:
            name = getattr(model, "default_prompt_name", None)
        prompt = prompts.get(name)
        features = preprocess([text], **({"prompt": prompt} if prompt is not None else {}))
    elif callable(getattr(model, "tokenize", None)):
        # Raw Qwen torch/ONNX runtimes expose this same bounded tokenizer API.
        features = model.tokenize([text], return_tensors="pt")
    else:
        tokenizer = getattr(model, "tokenizer", None)
        if not callable(tokenizer):
            return None
        options = {"padding": True, "truncation": True, "return_tensors": "pt"}
        limit = getattr(model, "max_seq_length", None) or getattr(model, "max_length", None)
        if limit:
            options["max_length"] = limit
        features = tokenizer([text], **options)
    tokens = features.get("input_ids")
    if tokens is None:
        tokens = features.get("attention_mask")
    return tokens.shape[-1] if tokens is not None else None


def embedding_batches(model, inputs, input_type, *, batch_size, max_tokens):
    """Bound padded tokens without truncating inputs or building huge planning batches."""
    batch, longest = [], 0
    for item in inputs:
        length = _padded_input_length(model, item, input_type) if isinstance(item, str) else None
        if length is None:
            # Media/layout-dependent or unknown tokenizers are safely handled
            # individually, as in the existing multimodal API dispatcher.
            if batch:
                yield batch
                batch, longest = [], 0
            yield [item]
            continue
        if batch and (
            len(batch) >= batch_size or max(longest, length) * (len(batch) + 1) > max_tokens
        ):
            yield batch
            batch, longest = [], 0
        batch.append(item)
        longest = max(longest, length)
    if batch:
        yield batch


def encode_in_batches(model, inputs, input_type, *, batch_size, max_tokens):
    encode = getattr(model, "encode_query", None) if input_type == "query" else None
    encode = encode or model.encode
    outputs = []
    for batch in embedding_batches(
        model, inputs, input_type, batch_size=batch_size, max_tokens=max_tokens
    ):
        output = encode(batch, batch_size=len(batch), show_progress_bar=False)
        if isinstance(output, torch.Tensor):
            output = output.detach().float().cpu().numpy()
        outputs.append(output)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)
