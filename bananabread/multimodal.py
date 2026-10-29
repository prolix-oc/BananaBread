"""Model-independent inline media decoding and SentenceTransformers dispatch.

JSON media is never forwarded as a URL/path to a model. Decoded objects remain
in memory and cache/log identities use the original serializable request.
"""

import base64
import binascii
import io
import math
import threading
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from bananabread.schemas import MediaBody, MultimodalInput
from bananabread.config import args
from bananabread.models.inference import encode_batch, encode_in_batches
from bananabread.tenancy import count_text_tokens
from bananabread.utils import ExecutionTiming, timed_lock

MAX_MEDIA_BYTES = 20 * 1024 * 1024
MAX_REQUEST_MEDIA_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_AUDIO_SECONDS = 360


def _decode_body(body: str | MediaBody, modality: str) -> bytes:
    if isinstance(body, MediaBody):
        content_type, encoded = body.media_type, body.data
    else:
        header, separator, encoded = body.partition(",")
        if not separator or not header.startswith("data:") or not header.endswith(";base64"):
            raise ValueError(f"{modality} must be a base64 data URL or a data/media_type object")
        content_type = header[5:-7]
    if not content_type.startswith(f"{modality}/"):
        raise ValueError(f"Expected {modality} media type")
    if len(encoded) > 4 * math.ceil(MAX_MEDIA_BYTES / 3):
        raise ValueError("Media body exceeds the 20 MiB limit")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 media body") from exc
    if not data or len(data) > MAX_MEDIA_BYTES:
        raise ValueError("Media body must contain 1 byte to 20 MiB")
    return data


def _image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 16 megapixel limit")
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or oversized image body") from exc


def _audio(data: bytes, sampling_rate: int | None) -> dict[str, Any]:
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        with sf.SoundFile(io.BytesIO(data)) as source:
            rate = source.samplerate
            if (
                not 1 <= rate <= 192000
                or not 1 <= source.channels <= 8
                or not 0 < source.frames <= MAX_AUDIO_SECONDS * rate
            ):
                raise ValueError("Audio must be nonempty and at most 360 seconds / 8 channels")
            samples = source.read(dtype="float32", always_2d=True).mean(axis=1)
    except (sf.LibsndfileError, OSError) as exc:
        raise ValueError(
            "Invalid audio body; use a supported encoded file such as WAV or FLAC"
        ) from exc
    target_rate = sampling_rate or rate
    if target_rate != rate:
        divisor = math.gcd(rate, target_rate)
        samples = resample_poly(samples, target_rate // divisor, rate // divisor).astype(np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains non-finite samples")
    return {"array": samples, "sampling_rate": target_rate}


def _audio_sampling_rate(model: Any) -> int | None:
    underlying = getattr(model, "model", model)
    try:
        processor = underlying[0].processor
        extractor = getattr(processor, "feature_extractor", None)
        return getattr(extractor, "sampling_rate", None)
    except (AttributeError, IndexError, TypeError):
        return None


def _supports(model: Any, modalities: list[str]) -> bool:
    supports = getattr(model, "supports", None)
    if not callable(supports):
        return modalities == ["text"]
    return bool(supports(modalities[0] if len(modalities) == 1 else tuple(modalities)))


def prepare_inputs(values: list[str | MultimodalInput], model: Any) -> tuple[list[Any], list[Any]]:
    """Decode validated bodies and check the selected model's declared capabilities."""
    prepared, log_inputs = [], []
    total_bytes = 0
    sampling_rate = _audio_sampling_rate(model)
    for value in values:
        if isinstance(value, str):
            prepared.append(value)
            log_inputs.append(value)
            continue
        modalities = [key for key in ("text", "image", "audio") if getattr(value, key)]
        if not _supports(model, modalities):
            raise ValueError(
                f"The selected embedding model does not support {' + '.join(modalities)} inputs"
            )
        item, description = {}, {}
        if value.text:
            item["text"] = value.text
            description["text"] = value.text
        for modality in ("image", "audio"):
            bodies = getattr(value, modality)
            if bodies is None:
                continue
            bodies = bodies if isinstance(bodies, list) else [bodies]
            if not 1 <= len(bodies) <= 32:
                raise ValueError("Provide 1 to 32 media bodies per modality per input")
            if len(bodies) > 1 and not _supports(model, [modality, modality]):
                raise ValueError(
                    f"The selected model cannot combine multiple {modality} bodies in one input"
                )
            decoded = []
            for body in bodies:
                data = _decode_body(body, modality)
                total_bytes += len(data)
                if total_bytes > MAX_REQUEST_MEDIA_BYTES:
                    raise ValueError("Combined media exceeds the 64 MiB request limit")
                decoded.append(_image(data) if modality == "image" else _audio(data, sampling_rate))
            item[modality] = decoded[0] if len(decoded) == 1 else decoded
            description[modality] = f"<{len(decoded)} inline {modality} body/bodies>"
        # A single image should also work with models such as CLIP that do not
        # combine modalities in one embedding and expect a PIL image directly.
        if list(item) == ["text"]:
            prepared.append(item["text"])
        elif list(item) == ["image"] and not isinstance(item["image"], list):
            prepared.append(item["image"])
        else:
            prepared.append(item)
        log_inputs.append(description)
    return prepared, log_inputs


def count_input_tokens(
    inputs: list[Any], model: Any, *, timing: ExecutionTiming | None = None
) -> int:
    """Use processor masks for media, with deterministic estimates for pure encoders.

    Image-only encoders without token IDs use 16x16 patch units; audio-only
    encoders use 25 units/second. Text retains BananaBread's existing accounting.
    """
    total = 0
    preprocess = getattr(model, "preprocess", None)
    with timed_lock(getattr(model, "tokenizer_lock", None), timing):
        for item in inputs:
            if isinstance(item, str):
                total += count_text_tokens([item], getattr(model, "tokenizer", None))
                continue
            if callable(preprocess):
                features = preprocess([item])
                mask = features.get("attention_mask")
                if mask is not None:
                    units = int(
                        np.asarray(mask.cpu() if isinstance(mask, torch.Tensor) else mask).sum()
                    )
                    total += max(1, units)
                    continue
            entries = {"image": item} if isinstance(item, Image.Image) else item
            total += count_text_tokens([entries.get("text", "")], getattr(model, "tokenizer", None))
            for key in ("image", "audio"):
                parts = entries.get(key, [])
                for part in parts if isinstance(parts, list) else [parts]:
                    total += (
                        math.ceil(part.width / 16) * math.ceil(part.height / 16)
                        if key == "image"
                        else math.ceil(len(part["array"]) / part["sampling_rate"] * 25)
                    )
    return total


def encode_inputs(
    model: Any, inputs: list[Any], input_type: str, *, timing: ExecutionTiming | None = None,
    planned_batch: bool = False,
) -> Any:
    with timed_lock(getattr(model, "tokenizer_lock", None), timing):
        if planned_batch:
            return encode_batch(model, inputs, input_type)
        return encode_in_batches(
            model, inputs, input_type,
            batch_size=args.embedding_batch_size, max_tokens=args.embedding_batch_max_tokens,
        )


class SentenceTransformerAdapter:
    """Expose saved retrieval prompts and shared tokenizer locking for HF models."""

    batch_prompt_names = {"document": "document", "query": "query"}

    def __init__(self, model: Any):
        self.model = model
        self.tokenizer_lock = threading.RLock()

    def encode(self, inputs, **kwargs):
        with self.tokenizer_lock:
            return self.model.encode_document(inputs, **kwargs)

    def encode_query(self, inputs, **kwargs):
        with self.tokenizer_lock:
            return self.model.encode_query(inputs, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)
