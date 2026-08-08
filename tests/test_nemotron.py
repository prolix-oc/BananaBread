from bananabread.models.nemotron import NemotronEmbeddingModel
import bananabread.models.nemotron as nemotron


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
    }
    assert model.encode(["a document"]) == ["passage: a document"]


def test_nemotron_applies_query_prompt(monkeypatch):
    monkeypatch.setattr(nemotron, "SentenceTransformer", FakeSentenceTransformer)
    model = NemotronEmbeddingModel("/models/nemotron", truncate_dim=2048, device="cuda")

    assert model.encode_query(["find this"]) == ["query: find this"]
