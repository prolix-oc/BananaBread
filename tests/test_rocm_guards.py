"""AMD ROCm runtime guards: backend fallbacks, bitsandbytes guards, FA2 gating,
and the pyproject rewrite used by install_rocm_torch.py."""

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


def test_nemotron_bnb_rejected_on_rocm(monkeypatch):
    _set_hip(monkeypatch, "6.3.41935")
    model = object.__new__(nemotron.NemotronEmbeddingModel)
    model.backend = "torch-bnb-8bit"
    model.compute_dtype = torch.bfloat16
    with pytest.raises(ValueError, match="NVIDIA-only"):
        model._model_kwargs("cuda:0")


def test_qwen_bnb_rejected_on_rocm(monkeypatch):
    _set_hip(monkeypatch, "6.3.41935")
    with pytest.raises(ValueError, match="NVIDIA-only"):
        qwen.QwenBnbModel("Qwen/Qwen3-Embedding-0.6B", device_arg="cuda:0")


def _bnb_namespace():
    return argparse.Namespace(
        qwen_backend="torch-bnb-8bit",
        nemotron_backend="torch-bnb-4bit",
        embedding_model="nemotron",
        reranking_model="qwen",
        embedding_device="cuda",
        rerank_device="cuda:0",
        enable_warmup=True,
        disable_warmup=False,
    )


def test_validate_args_falls_back_to_torch_backends_on_rocm(monkeypatch):
    _set_hip(monkeypatch, "6.3.41935")
    result = config.validate_args(_bnb_namespace())
    assert result.qwen_backend == "torch"
    assert result.nemotron_backend == "torch"


def test_validate_args_keeps_bnb_backends_without_rocme(monkeypatch):
    _set_hip(monkeypatch, None)
    result = config.validate_args(_bnb_namespace())
    assert result.qwen_backend == "torch-bnb-8bit"
    assert result.nemotron_backend == "torch-bnb-4bit"


# ----- install_rocm_torch.py: pyproject rewrite -----


def test_rewrite_replaces_url_and_preserves_other_keys():
    text = (
        "[tool.uv]\n"
        'extra-index-url = ["https://download.pytorch.org/whl/cu130"]\n'
        'index-strategy = "unsafe-best-match"\n'
    )
    out = install_rocm_torch.rewrite_extra_index_url(text, "https://example/rocm6.3")
    assert 'extra-index-url = ["https://example/rocm6.3"]' in out
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
    assert install_rocm_torch.linux_torch_requirement(False) == "torch==2.9.1+rocm6.3"
    assert install_rocm_torch.linux_torch_requirement(True) == "torch==2.9.1+cu130"
