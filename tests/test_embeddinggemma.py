"""EmbeddingGemma retrieval, precision safety, quantized loading and model selection."""

import argparse
from contextlib import nullcontext

import numpy as np
import pytest
import torch
import transformers

from bananabread import config
from bananabread.models import embeddinggemma, manager


class FakeSentenceTransformer:
    tokenizer = object()

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.ones((len(texts), kwargs["truncate_dim"]), dtype=np.float32)

    def modules(self):
        return []


@pytest.fixture
def fake_transformer(monkeypatch):
    monkeypatch.setattr(embeddinggemma, "SentenceTransformer", FakeSentenceTransformer)


def test_text_only_retrieval_uses_distinct_saved_prompts(fake_transformer):
    model = embeddinggemma.EmbeddingGemmaModel(
        "local/snapshot", truncate_dim=256, modalities="text"
    )
    model.encode(["a document"])
    model.encode_query(["find this"])

    assert model.model.calls == [
        (
            ["a document"],
            {"prompt_name": "Document", "truncate_dim": 256, "normalize_embeddings": True},
        ),
        (
            ["find this"],
            {"prompt_name": "SearchQuery", "truncate_dim": 256, "normalize_embeddings": True},
        ),
    ]
    assert model.model.init_kwargs["config_kwargs"] == {"vision_config": None, "audio_config": None}
    assert model.model.init_kwargs["trust_remote_code"] is False
    assert model.compute_dtype == torch.float32
    assert model.model.max_seq_length == 8192
    assert model.tokenizer is model.model.tokenizer


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("tensor", [False, True])
def test_rejects_nonfinite_embeddings(fake_transformer, monkeypatch, value, tensor):
    model = embeddinggemma.EmbeddingGemmaModel("local/snapshot")
    output = torch.full((1, 768), value) if tensor else np.full((1, 768), value)
    monkeypatch.setattr(model.model, "encode", lambda *a, **kw: output)
    with pytest.raises(RuntimeError, match="non-finite"):
        model.encode(["a document"])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"compute_dtype": "float16"}, "float16 is unsafe"),
        ({"backend": "torch-bnb-8bit"}, "Unsupported"),
        ({"truncate_dim": 384}, "dimensions"),
        ({"backend": "torch-bnb-4bit"}, "CUDA/HIP"),
        ({"backend": "torch-metal-4bit"}, "MPS"),
    ],
)
def test_rejects_unsafe_or_unsupported_loading(fake_transformer, kwargs, message):
    with pytest.raises(ValueError, match=message):
        embeddinggemma.EmbeddingGemmaModel("local/snapshot", **kwargs)


@pytest.mark.parametrize("bf16_supported", [True, False])
def test_nf4_uses_safe_compute_and_explicit_device(fake_transformer, monkeypatch, bf16_supported):
    captured = {}
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kw: bf16_supported)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", lambda **kw: captured.update(kw))
    model = embeddinggemma.EmbeddingGemmaModel(
        "local/snapshot", device="cuda:1", backend="torch-bnb-4bit"
    )

    dtype = torch.bfloat16 if bf16_supported else torch.float32
    assert captured["bnb_4bit_compute_dtype"] == dtype
    assert captured["bnb_4bit_quant_type"] == "nf4"
    assert captured["load_in_4bit"] is True
    assert model.model.init_kwargs["model_kwargs"]["dtype"] == dtype
    assert model.model.init_kwargs["model_kwargs"]["device_map"] == {"": "cuda:1"}
    assert model.model.init_kwargs["device"] is None


def test_explicit_bf16_on_older_cuda_falls_back_to_fp32(fake_transformer, monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kw: False)
    model = embeddinggemma.EmbeddingGemmaModel(
        "local/snapshot", device="cuda", compute_dtype="bfloat16"
    )
    assert model.compute_dtype == torch.float32


@pytest.mark.parametrize("bits", [4, 8])
def test_metal_quantizes_weights_without_fp16(fake_transformer, monkeypatch, bits):
    captured = {}
    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", lambda *a: True)
    monkeypatch.setattr(transformers, "MetalConfig", lambda **kw: captured.update(kw))
    model = embeddinggemma.EmbeddingGemmaModel(
        "local/snapshot", device="mps", backend=f"torch-metal-{bits}bit"
    )
    assert captured == {"bits": bits, "group_size": 64}
    assert model.compute_dtype == torch.bfloat16
    assert model.model.init_kwargs["model_kwargs"]["device_map"] == {"": "mps"}


def test_selector_adapts_shared_default_dimension_and_disables_unused_reranker(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    values = {**config.DEFAULTS, "embedding_model": "embeddinggemma", "rerank_device": "cuda"}
    parsed = config.validate_args(argparse.Namespace(**values))
    assert parsed.embedding_dim == 768


def test_selector_rejects_unsupported_dimensions():
    values = {**config.DEFAULTS, "embedding_model": "embeddinggemma", "embedding_dim": 384}
    with pytest.raises(ValueError, match="embedding_dim"):
        config.validate_args(argparse.Namespace(**values))


@pytest.mark.parametrize("pool_size", [1, 2])
def test_manager_loads_gemma_for_single_and_pooled_inference(monkeypatch, pool_size):
    parsed = argparse.Namespace(
        **{
            **config.DEFAULTS,
            "embedding_model": "embeddinggemma",
            "embedding_dim": 512,
            "embedding_device": "cuda:1",
            "num_concurrent_embedding": pool_size,
            "embeddinggemma_backend": "torch-bnb-4bit",
            "hf_access_token": "test-token",
            "hf_model_revision": "test-revision",
            "disable_warmup": True,
        }
    )
    monkeypatch.setattr(manager, "args", parsed)
    # Restore manager globals after initialize_models mutates them.
    for name in (
        "embedding_model",
        "embedding_model_pool",
        "embedding_model_name",
        "rerank_model",
        "rerank_model_pool",
    ):
        monkeypatch.setattr(manager, name, getattr(manager, name))
    downloads, models = [], []

    def download(repo, **kwargs):
        downloads.append((repo, kwargs))
        return "local/snapshot"

    def load(snapshot, **kwargs):
        models.append((snapshot, kwargs))
        return object()

    monkeypatch.setattr(manager, "download_hf_model", download)
    monkeypatch.setattr(manager, "EmbeddingGemmaModel", load)
    monkeypatch.setattr(manager, "warmup_model", lambda *a, **kw: None)
    monkeypatch.setattr(
        manager, "MxbaiRerankV2", lambda *a, **kw: pytest.fail("unused reranker loaded")
    )
    manager.initialize_models()

    assert manager.embedding_model_name == "google/embeddinggemma-2"
    assert manager.rerank_model is None
    assert len(models) == pool_size
    assert models[0] == (
        "local/snapshot",
        {
            "truncate_dim": 512,
            "device": "cuda:1",
            "backend": "torch-bnb-4bit",
            "compute_dtype": "auto",
            "modalities": "all",
        },
    )
    assert downloads[0] == (
        "google/embeddinggemma-2",
        {"storage_dir": "./models", "revision": "test-revision", "token": "test-token"},
    )


def test_real_sentence_transformer_truncates_before_normalizing(tmp_path):
    """Exercise the native architecture and ST pipeline without downloading weights."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modules.transformer import Transformer
    from sentence_transformers.sentence_transformer.modules.pooling import Pooling
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import EmbeddingGemma2Config, EmbeddingGemma2Model, PreTrainedTokenizerFast

    model_config = EmbeddingGemma2Config(
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "hidden_size_per_layer_input": 32,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "layer_types": ["sliding_attention", "full_attention"],
            "per_layer_config": {},
            "sliding_window": 16,
        }
    )
    native_path = tmp_path / "native"
    EmbeddingGemma2Model(model_config).save_pretrained(native_path)
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "hello": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        model_max_length=8192,
        processor_class="AutoTokenizer",
    ).save_pretrained(native_path)
    transformer = Transformer(str(native_path))
    saved = SentenceTransformer(
        modules=[transformer, Pooling(768)],
        prompts={
            "Document": "title: none | text: ",
            "SearchQuery": "task: search result | query: ",
        },
    )
    snapshot = tmp_path / "sentence-transformer"
    saved.save(str(snapshot))

    model = embeddinggemma.EmbeddingGemmaModel(str(snapshot), truncate_dim=128)
    for encode in (model.encode, model.encode_query):
        output = encode(["hello", "hello hello"], show_progress_bar=False)
        assert output.shape == (2, 128)
        assert np.isfinite(output).all()
        np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1.0, atol=1e-5)
