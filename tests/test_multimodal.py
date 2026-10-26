import base64
import importlib
import io
import threading

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image
from pydantic import ValidationError

from bananabread.cache import UserScopedCache, get_embedding_cache_key
from bananabread.models import embeddinggemma
from bananabread.multimodal import (
    SentenceTransformerAdapter,
    count_input_tokens,
    encode_inputs,
    prepare_inputs,
)
from bananabread.schemas import EmbeddingRequest, MultimodalInput


def image_body(color="red"):
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


class Model:
    tokenizer = None
    modalities = ["text", "image", "audio", "message"]

    def __init__(self):
        self.calls = []
        self.tokenizer_lock = threading.RLock()

    def supports(self, modalities):
        return all(
            m in self.modalities
            for m in ((modalities,) if isinstance(modalities, str) else modalities)
        )

    def encode(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return np.tile([1.0, 0.0], (len(inputs), 1))

    def encode_query(self, inputs, **kwargs):
        self.calls.append((inputs, {**kwargs, "query": True}))
        return torch.tensor([[0.0, 1.0]] * len(inputs))

    def preprocess(self, inputs):
        return {"attention_mask": torch.tensor([[1, 1, 1, 0]])}


def test_inline_image_decoding_and_model_independent_mixed_dispatch():
    model = Model()
    prepared, descriptions = prepare_inputs(
        [
            "text",
            MultimodalInput(image=image_body()),
            MultimodalInput(text="caption", image=[image_body(), image_body("blue")]),
        ],
        model,
    )
    assert isinstance(prepared[1], Image.Image)
    assert prepared[1].getpixel((0, 0)) == (255, 0, 0)
    assert len(prepared[2]["image"]) == 2
    assert "base64" not in str(descriptions)
    assert count_input_tokens(prepared, model) == 7
    output = encode_inputs(model, prepared, "query")
    np.testing.assert_array_equal(output, [[0, 1], [0, 1], [0, 1]])
    assert len(model.calls) == 3


def test_audio_is_decoded_to_mono_and_preserves_sample_rate():
    stream = io.BytesIO()
    sf.write(stream, np.ones((800, 2), dtype=np.float32) * 0.1, 8000, format="WAV")
    value = MultimodalInput(
        audio={"data": base64.b64encode(stream.getvalue()).decode(), "media_type": "audio/wav"}
    )
    prepared, _ = prepare_inputs([value], Model())
    assert prepared[0]["audio"]["array"].shape == (800,)
    assert prepared[0]["audio"]["sampling_rate"] == 8000


def test_audio_is_resampled_to_the_selected_model_rate(monkeypatch):
    import bananabread.multimodal as media

    monkeypatch.setattr(media, "_audio_sampling_rate", lambda model: 16000)
    stream = io.BytesIO()
    sf.write(stream, np.zeros(800), 8000, format="WAV")
    value = MultimodalInput(
        audio={"data": base64.b64encode(stream.getvalue()).decode(), "media_type": "audio/wav"}
    )
    prepared, _ = prepare_inputs([value], Model())
    assert prepared[0]["audio"]["array"].shape == (1600,)
    assert prepared[0]["audio"]["sampling_rate"] == 16000


@pytest.mark.parametrize(
    "body",
    [
        "https://example.com/image.png",
        "/etc/passwd",
        "data:image/png;base64,%%%%",
        "data:audio/wav;base64,YQ==",
        "data:image/png;base64,YQ==",
    ],
)
def test_invalid_media_is_rejected_before_inference(body):
    with pytest.raises(ValueError):
        prepare_inputs([MultimodalInput(image=body)], Model())


def test_media_limits_are_applied(monkeypatch):
    import bananabread.multimodal as media

    monkeypatch.setattr(media, "MAX_MEDIA_BYTES", 1)
    with pytest.raises(ValueError, match="limit"):
        prepare_inputs([MultimodalInput(image=image_body())], Model())


def test_model_capability_checks_reject_unsupported_media():
    model = Model()
    model.modalities = ["text"]
    with pytest.raises(ValueError, match="does not support image"):
        prepare_inputs([MultimodalInput(image=image_body())], model)


def test_pure_image_encoder_accepts_one_image_and_rejects_combined_modalities():
    model = Model()
    model.modalities = ["text", "image"]
    # SentenceTransformers requires message support to combine modalities.
    model.supports = lambda m: isinstance(m, str) and m in model.modalities
    prepared, _ = prepare_inputs([MultimodalInput(image=image_body())], model)
    assert isinstance(prepared[0], Image.Image)
    with pytest.raises(ValueError, match="text.*image"):
        prepare_inputs([MultimodalInput(text="caption", image=image_body())], model)


def test_cache_hash_preserves_batch_boundaries_and_media_identity():
    assert get_embedding_cache_key(["a", "bc"]) != get_embedding_cache_key(["ab", "c"])
    assert get_embedding_cache_key([{"image": image_body()}]) != get_embedding_cache_key(
        [{"image": image_body("blue")}]
    )
    assert get_embedding_cache_key([{"text": "a", "image": "b"}]) == get_embedding_cache_key(
        [{"image": "b", "text": "a"}]
    )


def test_schema_accepts_mixed_batches_and_rejects_empty_or_unknown_inputs():
    request = EmbeddingRequest(input=["text", {"image": image_body()}])
    assert isinstance(request.input[1], MultimodalInput)
    for value in ({}, {"video": "unknown"}, {"image": image_body(), "typo": "x"}):
        with pytest.raises(ValidationError):
            EmbeddingRequest(input=value)


def test_gemma_media_only_inputs_have_no_text_prefix(monkeypatch):
    class FakeTransformer(Model):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(embeddinggemma, "SentenceTransformer", FakeTransformer)
    model = embeddinggemma.EmbeddingGemmaModel("local/snapshot")
    image = Image.new("RGB", (16, 16))
    model.encode([image])
    assert model.model.calls[0][1]["prompt"] == ""
    assert "prompt_name" not in model.model.calls[0][1]
    assert model.supports("image")
    text_model = embeddinggemma.EmbeddingGemmaModel("local/snapshot", modalities="text")
    assert not text_model.supports("image")


def test_hf_adapter_uses_model_document_prompt_and_lock():
    model = Model()
    model.encode_document = model.encode
    wrapped = SentenceTransformerAdapter(model)
    wrapped.encode(["a document"])
    assert model.calls[0][0] == ["a document"]
    assert wrapped.supports("image")


@pytest.fixture
def client(monkeypatch, tmp_path):
    # API initialization creates local auth files. Keep those in the test directory.
    monkeypatch.chdir(tmp_path)
    api = importlib.import_module("bananabread.api")
    api.tenant_store.close()
    model = Model()
    charged = []

    class Store:
        def check_and_consume(self, username, tokens):
            charged.append(tokens)

    monkeypatch.setattr(api, "tenant_store", Store())
    monkeypatch.setattr(api, "embedding_cache", UserScopedCache(1024 * 1024))
    monkeypatch.setattr(api, "log_embedding_result", lambda *a, **kw: None)
    monkeypatch.setattr(api.models_manager, "embedding_model", model)
    monkeypatch.setattr(api.models_manager, "embedding_model_pool", None)
    monkeypatch.setattr(api.models_manager, "embedding_model_name", "test/multimodal")
    monkeypatch.setattr(api.models_manager, "initialize_models", lambda: None)
    monkeypatch.setattr(api, "cleanup_resources", lambda: None)
    monkeypatch.setattr(api.args, "quant", "standard")
    api.app.dependency_overrides[api.get_api_user] = lambda: {"username": "test"}
    from fastapi.testclient import TestClient

    with TestClient(api.app, follow_redirects=False) as transport:
        # Suppress startup model loading while still exercising the HTTP contract.
        yield transport, model, charged
    api.app.dependency_overrides.clear()


@pytest.mark.parametrize("route", ["/v1/embeddings", "/v0/embeddings", "/api/v0/embeddings"])
def test_endpoint_image_body_order_usage_and_cache(client, route):
    transport, model, charged = client
    payload = {"input": ["a text", {"image": image_body()}], "input_type": "query"}
    response = transport.post(route, json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["model"] == "test/multimodal"
    assert [item["index"] for item in data["data"]] == [0, 1]
    assert data["usage"]["prompt_tokens"] == 5
    assert charged == [5]
    assert all(call[1]["query"] for call in model.calls)
    assert transport.post(route, json=payload).status_code == 200
    assert len(model.calls) == 2
    changed = {"input": {"image": image_body("blue")}}
    assert transport.post(route, json=changed).status_code == 200
    assert len(model.calls) == 3


def test_endpoint_rejects_unsupported_or_invalid_media_without_charging(client):
    transport, model, charged = client
    model.modalities = ["text"]
    response = transport.post("/v1/embeddings", json={"input": {"image": image_body()}})
    assert response.status_code == 400
    assert "does not support" in response.json()["detail"]
    assert charged == []
    model.modalities = ["image"]
    response = transport.post("/v1/embeddings", json={"input": {"image": "bad"}})
    assert response.status_code == 400
    assert charged == []


@pytest.mark.parametrize("route", ["/v1/embeddings", "/v0/embeddings", "/api/v0/embeddings"])
def test_endpoint_text_and_base64_output_remain_supported(client, route):
    transport, model, charged = client
    response = transport.post(
        route, json={"input": "hello", "encoding_format": "base64"}
    )
    assert response.status_code == 200
    output = np.frombuffer(
        base64.b64decode(response.json()["data"][0]["embedding"]), dtype=np.float32
    )
    np.testing.assert_array_equal(output, [1, 0])
    assert transport.post(route, json={"input": [1, 2]}).status_code == 400


@pytest.mark.parametrize("route,field", [
    ("/v1/embeddings", "input"), ("/api/embed", "input"), ("/embedding", "content"),
    ("/v0/embeddings", "input"), ("/api/v0/embeddings", "input"),
])
def test_all_embedding_routes_obey_shared_batch_limits(client, monkeypatch, route, field):
    transport, model, charged = client
    import bananabread.api as api

    monkeypatch.setattr(api.args, "embedding_batch_size", 2)
    monkeypatch.setattr(api.args, "embedding_batch_max_tokens", 8)
    texts = ["first", "second", "third", "fourth", "fifth"]
    response = transport.post(route, json={"model": "test/multimodal", field: texts})
    assert response.status_code == 200, response.text
    assert [call[0] for call in model.calls] == [texts[:2], texts[2:4], texts[4:]]
    assert [call[1]["batch_size"] for call in model.calls] == [2, 2, 1]


def test_endpoint_discovers_model_modalities_and_publishes_the_contract(client):
    transport, model, charged = client
    response = transport.get("/v1/models")
    assert response.json()["data"][0]["input_modalities"] == ["text", "image", "audio"]
    schema = transport.get("/openapi.json").json()
    assert "MultimodalInput" in schema["components"]["schemas"]
    assert "MediaBody" in schema["components"]["schemas"]
    for route in ("/v0/embeddings", "/api/v0/embeddings"):
        assert "post" in schema["paths"][route]


def test_endpoint_selects_pool_instance_on_the_inference_worker(client, monkeypatch):
    transport, model, charged = client
    from bananabread import api

    selected_on = []

    class Pool:
        def get_all_models(self):
            return [model]

        def get_model(self):
            selected_on.append(threading.current_thread().name)
            return model

    monkeypatch.setattr(api.models_manager, "embedding_model_pool", Pool())
    response = transport.post("/v1/embeddings", json={"input": {"image": image_body()}})
    assert response.status_code == 200
    assert selected_on and all(name.startswith("embedding") for name in selected_on)
