"""Batch memory bounds and real Transformers attention equivalence."""

import argparse
import copy

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, LlamaConfig, LlamaModel
from transformers.masking_utils import (
    and_masks,
    bidirectional_mask_function,
    causal_mask_function,
    sdpa_mask,
)

from bananabread import config
from bananabread.models.inference import (
    COMPACT_SDPA,
    compact_sdpa_mask,
    configure_embedding_attention,
    embedding_batches,
    encode_in_batches,
)


class TextModel:
    prompts = {"document": "passage prefix", "query": "query"}
    batch_prompt_names = {"document": "document", "query": "query"}

    def __init__(self):
        self.seen = []

    def preprocess(self, texts, prompt=None):
        assert len(texts) == 1  # Planning must never pad the full request.
        length = len(texts[0].split()) + len((prompt or "").split())
        return {"input_ids": torch.zeros(1, 8 * ((length + 7) // 8), dtype=torch.long)}

    def encode(self, texts, **kwargs):
        self.seen.append((list(texts), kwargs))
        return np.array([[float(text.split()[0]), 1] for text in texts])

    encode_query = encode


def test_batch_limits_include_prompt_padding_and_preserve_order():
    model = TextModel()
    texts = ["0 a", "1 a", "2 " + "a " * 20, "3 a", "4 a"]
    batches = list(embedding_batches(model, texts, "document", batch_size=2, max_tokens=32))
    # 8 + 24 raw lengths would fit a sum budget, but 2 * 24 padded tokens do not.
    assert batches == [texts[:2], texts[2:3], texts[3:]]
    output = encode_in_batches(model, texts, "document", batch_size=2, max_tokens=32)
    np.testing.assert_array_equal(output[:, 0], np.arange(5))
    assert [call[1]["batch_size"] for call in model.seen] == [2, 1, 2]


def test_oversized_document_keeps_full_text_and_runs_alone():
    model = TextModel()
    texts = ["0 " + "a " * 20000, "1 a"]
    batches = list(embedding_batches(model, texts, "query", batch_size=8, max_tokens=32))
    assert batches == [[texts[0]], [texts[1]]]


def test_query_prompt_and_processor_padding_change_batch_budget():
    model = TextModel()
    texts = ["0 " + "a " * 6, "1 " + "a " * 6]
    assert list(embedding_batches(model, texts, "query", batch_size=8, max_tokens=16)) == [texts]
    assert list(embedding_batches(model, texts, "document", batch_size=8, max_tokens=16)) == [
        texts[:1],
        texts[1:],
    ]


def test_media_and_unknown_tokenizers_run_individually():
    model = TextModel()
    media = object()
    texts = ["0 a", media, "1 a"]
    assert list(embedding_batches(model, texts, "document", batch_size=8, max_tokens=8192)) == [
        [item] for item in texts
    ]
    assert list(
        embedding_batches(object(), ["a", "b"], "document", batch_size=8, max_tokens=8192)
    ) == [["a"], ["b"]]


def test_raw_torch_and_onnx_tokenizers_use_the_same_batch_limits():
    class RawModel:
        def tokenize(self, texts, return_tensors):
            assert return_tensors == "pt"
            return {"input_ids": torch.zeros(1, len(texts[0]))}

    texts = ["abcd", "efgh", "ijkl"]
    assert list(embedding_batches(RawModel(), texts, "document", batch_size=8, max_tokens=8)) == [
        texts[:2],
        texts[2:],
    ]


@pytest.mark.parametrize("key", ["embedding_batch_size", "embedding_batch_max_tokens",
                                "inference_max_requests", "inference_max_requests_per_user",
                                "inference_max_tokens", "inference_max_tokens_per_user"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_invalid_batch_config_is_rejected(monkeypatch, key, value):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    parsed = argparse.Namespace(**{**config.DEFAULTS, key: value})
    with pytest.raises(ValueError, match=key):
        config.validate_args(parsed)


@pytest.mark.parametrize("value", [-1, 1001, True, 1.5, "2"])
def test_invalid_batch_collection_delay_is_rejected(monkeypatch, value):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    parsed = argparse.Namespace(**{**config.DEFAULTS, "embedding_batch_wait_ms": value})
    with pytest.raises(ValueError, match="embedding_batch_wait_ms"):
        config.validate_args(parsed)


def test_compact_mask_has_linear_storage_and_matches_attention():
    padding = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    kwargs = dict(
        batch_size=2,
        q_length=5,
        kv_length=5,
        mask_function=bidirectional_mask_function,
        attention_mask=padding,
    )
    compact = compact_sdpa_mask(**kwargs)
    full = sdpa_mask(**kwargs)
    assert compact.shape == (2, 1, 1, 5)
    assert compact.untyped_storage().nbytes() == padding.numel()
    q, k, v = (torch.randn(2, 4, 5, 8) for _ in range(3))
    torch.testing.assert_close(
        torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=compact),
        torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=full),
    )
    long = compact_sdpa_mask(
        2,
        32768,
        32768,
        mask_function=bidirectional_mask_function,
        attention_mask=torch.ones(2, 32768, dtype=torch.bool),
    )
    assert long.untyped_storage().nbytes() == 65536


@pytest.mark.parametrize("pattern", ["causal", "sliding", "custom", "offset"])
def test_special_attention_patterns_retain_standard_masks(pattern):
    padding = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    kwargs = dict(
        batch_size=1,
        q_length=4,
        kv_length=4,
        attention_mask=padding,
        mask_function=bidirectional_mask_function,
        allow_is_causal_skip=False,
    )
    if pattern == "causal":
        kwargs["mask_function"] = causal_mask_function
    elif pattern == "sliding":
        kwargs["local_size"] = 2
        kwargs["mask_function"] = and_masks(
            bidirectional_mask_function, lambda b, h, q, k: abs(q - k) < 2
        )
    elif pattern == "custom":
        kwargs["mask_function"] = and_masks(bidirectional_mask_function, lambda b, h, q, k: q == k)
    else:
        kwargs["q_offset"] = 1
    torch.testing.assert_close(compact_sdpa_mask(**kwargs), sdpa_mask(**kwargs))


@pytest.mark.parametrize("architecture", ["bert", "llama"])
def test_real_encoder_and_causal_model_preserve_outputs_and_other_instances(architecture):
    if architecture == "bert":
        native = BertModel(
            BertConfig(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                attn_implementation="sdpa",
            )
        ).eval()
    else:
        native = LlamaModel(
            LlamaConfig(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                attn_implementation="sdpa",
            )
        ).eval()
    baseline = copy.deepcopy(native)
    configure_embedding_attention(native)
    assert native.config._attn_implementation == COMPACT_SDPA
    assert baseline.config._attn_implementation == "sdpa"
    ids = torch.tensor([[1, 2, 3, 4], [0, 0, 2, 3]])
    padding = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    with torch.inference_mode():
        expected = baseline(ids, attention_mask=padding, use_cache=False).last_hidden_state
        actual = native(ids, attention_mask=padding, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    if architecture == "llama":
        # Decoding with a cache must retain causal offsets and key-mask lengths.
        with torch.inference_mode():
            first = native(ids[:1], attention_mask=padding[:1], use_cache=True)
            expected_first = baseline(ids[:1], attention_mask=padding[:1], use_cache=True)
            actual_next = native(
                torch.tensor([[5]]),
                attention_mask=torch.ones(1, 5),
                past_key_values=first.past_key_values,
                use_cache=True,
            )
            expected_next = baseline(
                torch.tensor([[5]]),
                attention_mask=torch.ones(1, 5),
                past_key_values=expected_first.past_key_values,
                use_cache=True,
            )
        torch.testing.assert_close(actual_next.last_hidden_state, expected_next.last_hidden_state)


def test_flash_request_keeps_cpu_encoder_supported(monkeypatch):
    import bananabread.models.inference as inference

    monkeypatch.setattr(inference, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setattr(inference.importlib, "import_module", lambda name: object())
    native = BertModel(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            attn_implementation="sdpa",
        )
    ).eval()
    configure_embedding_attention(native, prefer_flash=True)
    assert native.config._attn_implementation == COMPACT_SDPA


def test_incomplete_flash_installation_falls_back_before_switching(monkeypatch, caplog):
    import bananabread.models.inference as inference

    monkeypatch.setattr(inference, "is_flash_attn_2_available", lambda: True)

    def import_flash(name):
        if name == "flash_attn.bert_padding":
            raise ImportError("No module named 'einops'")
        return object()

    monkeypatch.setattr(inference.importlib, "import_module", import_flash)
    native = BertModel(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            attn_implementation="sdpa",
        )
    ).eval()
    configure_embedding_attention(native, prefer_flash=True)
    assert native.config._attn_implementation == COMPACT_SDPA
    assert "einops" in caplog.text


@pytest.mark.parametrize(
    "supported,windows_bnb,expected",
    [(True, False, "flash_attention_2"), (False, False, COMPACT_SDPA), (True, True, COMPACT_SDPA)],
)
def test_rocm_embedding_backbone_selects_flash_by_capability(
    monkeypatch, supported, windows_bnb, expected
):
    import bananabread.models.inference as inference

    monkeypatch.setattr(torch.version, "hip", "7.2.53214")
    monkeypatch.setattr(inference, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setattr(inference.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(inference, "flash_attention_supported_device", lambda device: supported)
    if windows_bnb:
        monkeypatch.setattr(inference.sys, "platform", "win32")
    native = LlamaModel(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            attn_implementation="sdpa",
        )
    ).half()
    native.is_loaded_in_4bit = windows_bnb

    def select_backend(choices):
        native.config._attn_implementation = choices[""]

    monkeypatch.setattr(native, "set_attn_implementation", select_backend)
    configure_embedding_attention(native, prefer_flash=True)
    assert native.config._attn_implementation == expected
