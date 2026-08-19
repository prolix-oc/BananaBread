"""Apple Silicon Metal quantization backends and configuration guards."""

import argparse

import pytest
import torch
import torch.nn.functional as F

from bananabread import config
from bananabread.models import nemotron, qwen
from bananabread.models.metal import enable_batched_metal_linears


class FakeModel:
    device = "mps:0"

    def eval(self):
        return self

    def modules(self):
        return []


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs

    def modules(self):
        return []


def test_metal_linear_compatibility_handles_batched_inputs():
    from transformers.integrations.metal_quantization import MetalLinear

    layer = MetalLinear(128, 64, bias=True, bits=8, group_size=64, dtype=None)
    layer.weight.data.normal_()
    layer.bias.data.normal_()
    expected_weight = layer.weight.detach().clone()
    expected_bias = layer.bias.detach().clone()

    assert enable_batched_metal_linears(layer) == 1

    inputs = torch.randn(3, 5, 128)
    actual = layer(inputs)
    expected = F.linear(inputs, expected_weight, expected_bias)

    assert actual.shape == (3, 5, 64)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("bits", [4, 8])
def test_qwen_metal_builds_transformers_config(monkeypatch, bits):
    captured = {}

    class FakeMetalConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    def fake_from_pretrained(*args, **kwargs):
        captured["load"] = kwargs
        return FakeModel()

    import transformers

    monkeypatch.setattr(transformers, "MetalConfig", FakeMetalConfig)
    monkeypatch.setattr(qwen.BaseQwenModel, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(qwen.AutoModel, "from_pretrained", fake_from_pretrained)

    model = qwen.QwenMetalModel(
        "Qwen/Qwen3-Embedding-0.6B",
        device_arg="mps",
        quantization_bits=bits,
    )

    assert model.backend_name == f"torch-metal-{bits}bit"
    assert model.device == "mps:0"
    assert captured["config"] == {"bits": bits, "group_size": 64}
    assert captured["load"]["device_map"] == {"": "mps"}
    assert captured["load"]["dtype"] == torch.bfloat16


def test_qwen_metal_rejects_non_mps_device():
    with pytest.raises(ValueError, match="MPS"):
        qwen.QwenMetalModel("Qwen/Qwen3-Embedding-0.6B", device_arg="cuda")


@pytest.mark.parametrize("model_class", [nemotron.NemotronEmbeddingModel, nemotron.Nemotron3EmbeddingModel])
@pytest.mark.parametrize(
    ("backend", "bits"),
    [("torch-metal-8bit", 8), ("torch-metal-4bit", 4)],
)
def test_nemotron_metal_builds_transformers_config(monkeypatch, model_class, backend, bits):
    captured = {}

    class FakeMetalConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import transformers

    monkeypatch.setattr(transformers, "MetalConfig", FakeMetalConfig)
    monkeypatch.setattr(nemotron, "SentenceTransformer", FakeSentenceTransformer)

    model = model_class(
        "/models/nemotron",
        truncate_dim=1024,
        device="mps",
        backend=backend,
    )

    assert model.model.init_kwargs["device"] is None
    assert model.model.init_kwargs["model_kwargs"]["device_map"] == {"": "mps"}
    assert model.model.init_kwargs["model_kwargs"]["dtype"] == torch.bfloat16
    assert captured == {"bits": bits, "group_size": 64}


def _metal_namespace(**overrides):
    values = {
        "qwen_backend": "torch-metal-8bit",
        "nemotron_backend": "torch-metal-4bit",
        "embedding_model": "qwen",
        "reranking_model": "none",
        "embedding_device": "mps",
        "rerank_device": "cpu",
        "qwen_compute_dtype": "bfloat16",
        "nemotron_compute_dtype": "bfloat16",
        "enable_warmup": True,
        "disable_warmup": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_validate_args_keeps_qwen_metal_on_mps():
    result = config.validate_args(_metal_namespace())
    assert result.qwen_backend == "torch-metal-8bit"


def test_validate_args_falls_back_when_qwen_metal_has_non_mps_consumer():
    result = config.validate_args(
        _metal_namespace(reranking_model="qwen", rerank_device="cpu")
    )
    assert result.qwen_backend == "torch"


def test_validate_args_keeps_nemotron_metal_on_mps():
    result = config.validate_args(_metal_namespace(embedding_model="nemotron"))
    assert result.nemotron_backend == "torch-metal-4bit"


def test_validate_args_falls_back_for_nemotron_metal_on_cuda():
    result = config.validate_args(
        _metal_namespace(embedding_model="nemotron", embedding_device="cuda")
    )
    assert result.nemotron_backend == "torch"
