import bananabread.models.nemotron as nemotron
from bananabread.models.nemotron import NemotronEmbeddingModel


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

    assert model.model.init_kwargs["model_kwargs"]["device_map"] == "auto"
    assert captured == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": nemotron.torch.bfloat16,
    }
