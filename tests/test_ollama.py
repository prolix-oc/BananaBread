"""Effective embedding metadata for Ollama model discovery."""

from types import SimpleNamespace

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules.pooling import Pooling

from bananabread.ollama import embedding_model_info


def test_sentence_transformer_metadata_uses_actual_truncation_and_context():
    pooling = Pooling(word_embedding_dimension=8, pooling_mode="mean")
    pooling.max_seq_length = 32768
    model = SentenceTransformer(modules=[pooling], truncate_dim=4)
    assert embedding_model_info(model) == {
        "embedding.embedding_length": 4,
        "embedding.context_length": 32768,
    }
    # A configured truncation larger than the native vector cannot enlarge it.
    model.truncate_dim = 16
    assert embedding_model_info(model)["embedding.embedding_length"] == 8


def test_raw_qwen_metadata_reports_native_dimensions():
    model = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=2560)),
        max_length=8192,
    )
    assert embedding_model_info(model) == {
        "embedding.embedding_length": 2560,
        "embedding.context_length": 8192,
    }


def test_onnx_metadata_uses_output_shape():
    model = SimpleNamespace(
        session=SimpleNamespace(get_outputs=lambda: [SimpleNamespace(shape=[None, None, 4096])]),
        max_length=512,
    )
    assert embedding_model_info(model) == {
        "embedding.embedding_length": 4096,
        "embedding.context_length": 512,
    }


def test_unknown_metadata_and_unbounded_tokenizer_limits_are_omitted():
    model = SimpleNamespace(tokenizer=SimpleNamespace(model_max_length=10**30))
    assert embedding_model_info(model) == {}
    model.tokenizer.model_max_length = 2048
    assert embedding_model_info(model) == {"embedding.context_length": 2048}
