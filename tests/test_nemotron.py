import bananabread.models.nemotron as nemotron
from bananabread.models.nemotron import Nemotron3EmbeddingModel, NemotronEmbeddingModel


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


def test_nemotron_builds_4bit_bitsandbytes_kwargs(monkeypatch):
    captured = {}

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import transformers

    monkeypatch.setattr(nemotron, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", FakeBitsAndBytesConfig)
    model = NemotronEmbeddingModel(
        "/models/nemotron",
        truncate_dim=1024,
        device="cuda",
        backend="torch-bnb-4bit",
    )

    assert model.model.init_kwargs["device"] is None
    assert model.model.init_kwargs["model_kwargs"]["device_map"] == "auto"
    assert captured == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": nemotron.torch.bfloat16,
    }


def test_nemotron_8bit_uses_fp16_when_matmul_cast_is_enabled(monkeypatch):
    captured = {}

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import transformers

    monkeypatch.setattr(nemotron, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", FakeBitsAndBytesConfig)
    monkeypatch.setattr(nemotron.args, "matmul_cast_fp16", True)
    model = NemotronEmbeddingModel(
        "/models/nemotron",
        truncate_dim=1024,
        device="cuda",
        backend="torch-bnb-8bit",
        compute_dtype="bfloat16",
    )

    assert model.compute_dtype is nemotron.torch.float16
    assert model.model.init_kwargs["model_kwargs"]["dtype"] is nemotron.torch.float16
    assert captured == {"load_in_8bit": True}


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
