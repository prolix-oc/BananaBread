"""Nemotron 3.5 model selection, retrieval prompts, and native multimodal inference."""

import argparse

import numpy as np
import pytest
import torch
from PIL import Image

from bananabread import config
from bananabread.hf_models import resolve_model_repo_id
from bananabread.models import manager, nemotron
from bananabread.models.nemotron35 import Nemotron35EmbeddingModel
from bananabread.models.inference import COMPACT_SDPA, configure_embedding_attention
from bananabread.multimodal import encode_inputs


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []

    def supports(self, modality):
        return True

    def encode_document(self, inputs, **kwargs):
        self.calls.append(("document", inputs, kwargs))
        return np.full((len(inputs), kwargs["truncate_dim"]), 2.0, dtype=np.float32)

    def encode_query(self, inputs, **kwargs):
        self.calls.append(("query", inputs, kwargs))
        return torch.full((len(inputs), kwargs["truncate_dim"]), 2.0, dtype=torch.bfloat16)


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(nemotron, "SentenceTransformer", FakeSentenceTransformer)
    return Nemotron35EmbeddingModel("local/snapshot", truncate_dim=512)


@pytest.mark.parametrize("selector", ["nemotron-3.5", "nemotron3.5"])
@pytest.mark.parametrize("size", [None, "1B", "8B"])
def test_resolves_official_checkpoints(selector, size):
    assert resolve_model_repo_id(model_name=selector, size=size) == (
        f"nvidia/Nemotron-3.5-Embed-{size or '1B'}-BF16"
    )


def test_rejects_unsupported_size():
    with pytest.raises(ValueError, match="Nemotron-3.5 size"):
        resolve_model_repo_id(model_name="nemotron-3.5", size="4B")


def test_uses_saved_prompts_for_text_and_image_documents(fake_model):
    inputs = ["document", Image.new("RGB", (16, 16)), {"text": "caption", "image": object()}]
    output = fake_model.encode(inputs)
    query = fake_model.encode_query(["find this"])
    assert fake_model.model.calls == [
        ("document", inputs, {"truncate_dim": 512, "normalize_embeddings": True}),
        ("query", ["find this"], {"truncate_dim": 512, "normalize_embeddings": True}),
    ]
    assert fake_model.model.init_kwargs.get("trust_remote_code", False) is False
    assert fake_model.model.max_seq_length == 32768
    assert output.dtype == np.float32
    assert query.dtype == torch.float32
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1.0, atol=1e-6)
    torch.testing.assert_close(torch.linalg.vector_norm(query, dim=1), torch.ones(1))
    assert fake_model.supports("image")
    assert fake_model.supports(("text", "image"))
    assert not fake_model.supports("audio")


@pytest.mark.parametrize("combined", [True, False])
def test_rejects_image_queries(fake_model, combined):
    image = Image.new("RGB", (16, 16))
    item = {"text": "query", "image": image} if combined else image
    with pytest.raises(ValueError, match="queries support text only"):
        fake_model.encode_query([item])
    assert not fake_model.model.calls


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("query", [False, True])
def test_rejects_nonfinite_output(fake_model, monkeypatch, value, query):
    monkeypatch.setattr(
        fake_model.model,
        "encode_query" if query else "encode_document",
        lambda *a, **kw: torch.full((1, 512), value) if query else np.full((1, 512), value),
    )
    encode = fake_model.encode_query if query else fake_model.encode
    with pytest.raises(RuntimeError, match="non-finite"):
        encode(["text"])


@pytest.mark.parametrize(
    "size,dim,valid",
    [
        ("1B", 2048, True),
        ("1B", 4096, False),
        ("8B", 4096, True),
        ("8B", 4097, False),
        ("1B", 0, False),
        ("8B", 512, True),
    ],
)
def test_validates_dimensions_for_each_size(monkeypatch, size, dim, valid):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    parsed = argparse.Namespace(
        **{
            **config.DEFAULTS,
            "embedding_model": "nemotron-3.5",
            "nemotron_size": size,
            "embedding_dim": dim,
            "rerank_device": "cuda",
        }
    )
    if valid:
        assert config.validate_args(parsed).embedding_dim == dim
    else:
        with pytest.raises(ValueError, match="embedding_dim"):
            config.validate_args(parsed)


@pytest.mark.parametrize("backend", ["torch-bnb-4bit", "torch-metal-8bit"])
def test_cpu_falls_back_from_accelerator_backends(monkeypatch, backend):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    parsed = argparse.Namespace(
        **{
            **config.DEFAULTS,
            "embedding_model": "nemotron-3.5",
            "nemotron_backend": backend,
        }
    )
    assert config.validate_args(parsed).nemotron_backend == "torch"


@pytest.mark.parametrize("size", ["1B", "8B"])
@pytest.mark.parametrize("pool_size", [1, 2])
def test_manager_loads_correct_size_and_embedding_only_mode(monkeypatch, size, pool_size):
    parsed = argparse.Namespace(
        **{
            **config.DEFAULTS,
            "embedding_model": "nemotron-3.5",
            "nemotron_size": size,
            "embedding_device": "cuda:1",
            "num_concurrent_embedding": pool_size,
            "nemotron_backend": "torch-bnb-4bit",
            "hf_access_token": "test-token",
            "hf_model_revision": "test-revision",
            "disable_warmup": True,
        }
    )
    monkeypatch.setattr(manager, "args", parsed)
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
    monkeypatch.setattr(manager, "Nemotron35EmbeddingModel", load)
    monkeypatch.setattr(manager, "warmup_model", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "MxbaiRerankV2", lambda *a, **kw: pytest.fail("unused reranker"))
    manager.initialize_models()
    assert manager.embedding_model_name == f"nvidia/Nemotron-3.5-Embed-{size}-BF16"
    assert manager.rerank_model is None
    assert len(models) == pool_size
    assert models[0] == (
        "local/snapshot",
        {
            "size": size,
            "truncate_dim": 1024,
            "device": "cuda:1",
            "backend": "torch-bnb-4bit",
            "compute_dtype": "bfloat16",
        },
    )
    assert downloads[0] == (
        manager.embedding_model_name,
        {
            "storage_dir": "./models",
            "revision": "test-revision",
            "token": "test-token",
        },
    )


def test_native_mistral3_text_image_pipeline_without_downloads(tmp_path):
    """Exercise the actual architecture, processor, prompts, pooling and truncation."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modules.transformer import Transformer
    from sentence_transformers.sentence_transformer.modules.pooling import Pooling
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        Mistral3Config,
        Mistral3Model,
        PixtralImageProcessor,
        PixtralProcessor,
        PreTrainedTokenizerFast,
    )

    native_path = tmp_path / "native"
    model_config = Mistral3Config(
        image_token_index=2,
        text_config={
            "model_type": "ministral3",
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "is_causal": False,
            "use_cache": False,
            "pad_token_id": 0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "llama_4_scaling_beta": 0.1,
                "original_max_position_embeddings": 16384,
            },
        },
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "head_dim": 8,
            "image_size": 32,
            "patch_size": 8,
        },
    )
    Mistral3Model(model_config).save_pretrained(native_path)
    vocab = {
        word: idx
        for idx, word in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[IMG]",
                "[IMG_BREAK]",
                "[IMG_END]",
                "query",
                "passage",
                ":",
                "hello",
            ]
        )
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        additional_special_tokens=["[IMG]", "[IMG_BREAK]", "[IMG_END]"],
        model_max_length=32768,
    )
    template = (
        "{% for message in messages %}"
        "{% if message['content'] is string %}{{ message['content'] }} "
        "{% else %}{% for part in message['content'] %}"
        "{% if part['type'] == 'image' %}[IMG]"
        "{% elif part['type'] == 'text' %}{{ part['text'] }}{% endif %}"
        "{% endfor %}{% endif %}{% endfor %}"
    )
    PixtralProcessor(
        image_processor=PixtralImageProcessor(size={"longest_edge": 32}, patch_size=8),
        tokenizer=tokenizer,
        patch_size=8,
        spatial_merge_size=2,
        chat_template=template,
    ).save_pretrained(native_path)
    modality_config = {
        key: {"method": "forward", "method_output_name": "last_hidden_state"}
        for key in ("text", "image", "message")
    }
    modality_config["message"]["format"] = "structured"
    transformer = Transformer(
        str(native_path), modality_config=modality_config, module_output_name="token_embeddings"
    )
    saved = SentenceTransformer(
        modules=[transformer, Pooling(32, include_prompt=True)],
        prompts={"query": "query: ", "document": "passage: "},
    )
    snapshot = tmp_path / "sentence-transformer"
    saved.save(str(snapshot))
    model = Nemotron35EmbeddingModel(str(snapshot), truncate_dim=16, compute_dtype="float32")
    from bananabread.tenancy import count_text_tokens

    assert count_text_tokens(["hello"], model.tokenizer) == 1
    assert count_text_tokens(["hello " * 50], model.tokenizer) == 50
    assert count_text_tokens(["hello"] * 40, model.tokenizer) == 40
    configure_embedding_attention(model)
    assert model.model[0].auto_model.config.text_config._attn_implementation == COMPACT_SDPA
    image = Image.new("RGB", (32, 32), "red")
    inputs = ["hello", image, {"text": "hello", "image": image}]
    output = encode_inputs(model, inputs, "document")
    assert output.shape == (3, 16)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1.0, atol=1e-6)
    assert model.supports(("text", "image"))
    query = encode_inputs(model, ["hello"], "query")
    assert query.shape == (1, 16)
    assert not np.allclose(query[0], output[0])
    # The scheduler's preplanned path must retain native prompts and FP32
    # conversion while avoiding a second planning pass.
    np.testing.assert_allclose(
        encode_inputs(model, ["hello"], "document", planned_batch=True), output[:1], atol=1e-6
    )
    np.testing.assert_allclose(
        encode_inputs(model, ["hello"], "query", planned_batch=True), query, atol=1e-6
    )
    # Compare to the untruncated native pipeline to verify truncation precedes normalization.
    full = saved.encode_document(["hello"], show_progress_bar=False)
    expected = full[:, :16] / np.linalg.norm(full[:, :16], axis=1, keepdims=True)
    np.testing.assert_allclose(output[:1], expected, atol=1e-6)
