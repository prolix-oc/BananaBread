import pytest

import bananabread.models.nemotron as nemotron
from bananabread.models.nemotron import Nemotron3EmbeddingModel, NemotronEmbeddingModel
from bananabread.models.nemotron35 import Nemotron35EmbeddingModel


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls = []

    def encode(self, texts, *args, **kwargs):
        self.calls.append((texts, args, kwargs))
        return texts


def test_nemotron_applies_document_prompt_and_trusts_its_custom_code(monkeypatch):
    monkeypatch.setattr(nemotron, "SentenceTransformer", FakeSentenceTransformer)
    model = NemotronEmbeddingModel("/models/nemotron", truncate_dim=1024, device="cpu")

    assert model.model.init_kwargs == {
        "truncate_dim": 1024,
        "device": "cpu",
        "trust_remote_code": True,
        "model_kwargs": {},
    }
    assert model.encode(["a document"]) == ["passage: a document"]


def test_nemotron_applies_query_prompt(monkeypatch):
    monkeypatch.setattr(nemotron, "SentenceTransformer", FakeSentenceTransformer)
    model = NemotronEmbeddingModel("/models/nemotron", truncate_dim=2048, device="cuda")

    assert model.encode_query(["find this"]) == ["query: find this"]


@pytest.mark.parametrize(
    "model_cls", [NemotronEmbeddingModel, Nemotron3EmbeddingModel, Nemotron35EmbeddingModel]
)
def test_nemotron_builds_4bit_bitsandbytes_kwargs(monkeypatch, model_cls):
    captured = {}

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import transformers

    monkeypatch.setattr(nemotron, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", FakeBitsAndBytesConfig)
    monkeypatch.setattr(nemotron.args, "matmul_cast_fp16", True)
    model = model_cls(
        "/models/nemotron",
        truncate_dim=1024,
        device="cuda",
        backend="torch-bnb-4bit",
    )

    assert model.model.init_kwargs["device"] is None
    assert model.compute_dtype is nemotron.torch.bfloat16
    assert model.model.init_kwargs["model_kwargs"]["device_map"] == "auto"
    assert captured == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": nemotron.torch.bfloat16,
    }


@pytest.mark.parametrize(
    "model_cls", [NemotronEmbeddingModel, Nemotron3EmbeddingModel, Nemotron35EmbeddingModel]
)
@pytest.mark.parametrize("cast_fp16", [False, True])
def test_nemotron_8bit_honors_matmul_cast_setting(monkeypatch, model_cls, cast_fp16):
    captured = {}
    filter_calls = []

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import transformers

    monkeypatch.setattr(nemotron, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", FakeBitsAndBytesConfig)
    monkeypatch.setattr(nemotron.args, "matmul_cast_fp16", cast_fp16)
    monkeypatch.setattr(nemotron, "install_bnb_matmul_cast_filter", lambda: filter_calls.append(True))
    model = model_cls(
        "/models/nemotron",
        truncate_dim=1024,
        device="cuda",
        backend="torch-bnb-8bit",
        compute_dtype="bfloat16",
    )

    expected_dtype = nemotron.torch.float16 if cast_fp16 else nemotron.torch.bfloat16
    assert model.compute_dtype is expected_dtype
    assert model.model.init_kwargs["model_kwargs"]["dtype"] is expected_dtype
    assert captured == {"load_in_8bit": True}
    assert filter_calls == [True]


def test_nemotron_3_uses_saved_sentence_transformers_prompts(monkeypatch):
    class PromptAwareFakeSentenceTransformer(FakeSentenceTransformer):
        def encode_document(self, texts, *args, **kwargs):
            self.document_calls = [(texts, args, kwargs)]
            return texts

        def encode_query(self, texts, *args, **kwargs):
            self.query_calls = [(texts, args, kwargs)]
            return texts

    monkeypatch.setattr(nemotron, "SentenceTransformer", PromptAwareFakeSentenceTransformer)
    model = Nemotron3EmbeddingModel("/models/nemotron-3", truncate_dim=1024, device="cuda")

    assert model.model.init_kwargs == {
        "truncate_dim": 1024,
        "device": "cuda",
        "model_kwargs": {"dtype": nemotron.torch.bfloat16},
    }
    assert model.encode(["a document"]) == ["a document"]
    assert model.encode_query(["find this"]) == ["find this"]
    assert model.model.document_calls == [(["a document"], (), {})]
    assert model.model.query_calls == [(["find this"], (), {})]


def test_nemotron_3_omits_device_when_device_map_present(monkeypatch):
    captured = {}

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import transformers

    monkeypatch.setattr(nemotron, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", FakeBitsAndBytesConfig)
    model = Nemotron3EmbeddingModel(
        "/models/nemotron-3",
        truncate_dim=1024,
        device="cuda:1",
        backend="torch-bnb-8bit",
    )

    # accelerate owns placement via device_map, so `device` must not be passed
    assert model.model.init_kwargs["device"] is None
    assert model.model.init_kwargs["model_kwargs"]["device_map"] == {"": "cuda:1"}
    assert captured == {"load_in_8bit": True}
