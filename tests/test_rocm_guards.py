"""AMD ROCm runtime behavior, FA2 gating, and installer pyproject rewrites."""

import argparse
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import install_rocm_torch
from bananabread import config
from bananabread.models import nemotron, qwen


def _set_hip(monkeypatch, value):
    monkeypatch.setattr(torch.version, "hip", value, raising=False)


def test_is_rocm_build_detection(monkeypatch):
    _set_hip(monkeypatch, None)
    assert config.is_rocm_build() is False
    _set_hip(monkeypatch, "6.3.41935")
    assert config.is_rocm_build() is True


def test_flash_attention_disabled_on_rocm(monkeypatch):
    _set_hip(monkeypatch, "6.3.41935")
    available, reason = qwen._check_flash_attention_available()
    assert available is False
    assert "ROCm" in reason


def test_nemotron_bnb_allowed_on_rocm(monkeypatch):
    _set_hip(monkeypatch, "6.3.41935")
    captured = {}

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import transformers

    monkeypatch.setattr(transformers, "BitsAndBytesConfig", FakeBitsAndBytesConfig)
    model = object.__new__(nemotron.NemotronEmbeddingModel)
    model.backend = "torch-bnb-8bit"
    model.compute_dtype = torch.bfloat16
    kwargs = model._model_kwargs("cuda:0")
    assert kwargs["device_map"] == {"": "cuda:0"}
    assert captured == {"load_in_8bit": True}


def test_qwen_bnb_allowed_on_rocm(monkeypatch):
    _set_hip(monkeypatch, "6.3.41935")
    captured = {}

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeModel:
        device = "cuda:0"

        def eval(self):
            return self

    import transformers

    monkeypatch.setattr(transformers, "BitsAndBytesConfig", FakeBitsAndBytesConfig)
    monkeypatch.setattr(qwen.BaseQwenModel, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(qwen.QwenBnbModel, "_attention_kwargs", lambda self, *args: {})
    monkeypatch.setattr(qwen.AutoModel, "from_pretrained", lambda *args, **kwargs: FakeModel())

    model = qwen.QwenBnbModel("Qwen/Qwen3-Embedding-0.6B", device_arg="cuda:0")
    assert model.device == "cuda:0"
    assert captured == {"load_in_8bit": True}


def _bnb_namespace():
    return argparse.Namespace(
        qwen_backend="torch-bnb-8bit",
        nemotron_backend="torch-bnb-4bit",
        embedding_model="nemotron",
        reranking_model="qwen",
        embedding_device="cuda",
        rerank_device="cuda:0",
        qwen_compute_dtype="bfloat16",
        nemotron_compute_dtype="bfloat16",
        enable_warmup=True,
        disable_warmup=False,
    )


def test_validate_args_keeps_bnb_backends_on_rocm(monkeypatch):
    _set_hip(monkeypatch, "6.3.41935")
    result = config.validate_args(_bnb_namespace())
    assert result.qwen_backend == "torch-bnb-8bit"
    assert result.nemotron_backend == "torch-bnb-4bit"


def test_validate_args_keeps_bnb_backends_without_rocm(monkeypatch):
    _set_hip(monkeypatch, None)
    result = config.validate_args(_bnb_namespace())
    assert result.qwen_backend == "torch-bnb-8bit"
    assert result.nemotron_backend == "torch-bnb-4bit"


def test_validate_args_uses_float16_without_native_bf16(monkeypatch):
    _set_hip(monkeypatch, None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    parsed = _bnb_namespace()
    result = config.validate_args(parsed)
    assert result.qwen_compute_dtype == "float16"
    assert result.nemotron_compute_dtype == "float16"


def test_validate_args_keeps_bfloat16_when_supported(monkeypatch):
    _set_hip(monkeypatch, None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    parsed = _bnb_namespace()
    result = config.validate_args(parsed)
    assert result.qwen_compute_dtype == "bfloat16"
    assert result.nemotron_compute_dtype == "bfloat16"


# ----- install_rocm_torch.py: pyproject rewrite -----


def test_rewrite_replaces_url_and_preserves_other_keys():
    text = (
        "[tool.uv]\n"
        'extra-index-url = ["https://download.pytorch.org/whl/cu130"]\n'
        'index-strategy = "unsafe-best-match"\n'
    )
    out = install_rocm_torch.rewrite_extra_index_url(text, "https://example/rocm7.2")
    assert 'extra-index-url = ["https://example/rocm7.2"]' in out
    assert 'index-strategy = "unsafe-best-match"' in out
    assert "[tool.uv]" in out


def test_rewrite_only_touches_tool_uv_table():
    text = (
        '[project]\nextra-index-url = ["untouched"]\n\n'
        '[tool.uv]\nextra-index-url = ["old"]\n\n'
        "[tool.black]\nline-length = 100\n"
    )
    out = install_rocm_torch.rewrite_extra_index_url(text, "new")
    assert 'extra-index-url = ["new"]' in out
    assert '["untouched"]' in out
    assert "line-length = 100" in out


def test_rewrite_fails_loudly_without_key():
    with pytest.raises(RuntimeError, match="extra-index-url"):
        install_rocm_torch.rewrite_extra_index_url(
            '[tool.uv]\nindex-strategy = "first-index"\n', "https://example"
        )


def test_rewrite_is_idempotent():
    text = '[tool.uv]\nextra-index-url = ["a"]\n'
    once = install_rocm_torch.rewrite_extra_index_url(text, "b")
    twice = install_rocm_torch.rewrite_extra_index_url(once, "b")
    assert once == twice


def test_linux_lock_requirement_selects_exact_backend_build():
    assert install_rocm_torch.linux_torch_requirement(False) == "torch==2.11.0+rocm7.2"
    assert install_rocm_torch.linux_torch_requirement(True) == "torch==2.11.0+cu130"


def test_windows_amd_wheel_keeps_vendor_torch_version():
    assert "torch-2.9.1%2Brocm7.2.1" in install_rocm_torch.AMD_TORCH_URL
