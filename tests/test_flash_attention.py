"""CUDA and HIP hardware gating for Qwen and shared embedding attention."""

from types import SimpleNamespace

import pytest
import torch

from bananabread.models import attention, qwen


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1200", "gfx1201"])
def test_rocm_uses_architecture_instead_of_nvidia_capability(monkeypatch, arch):
    monkeypatch.setattr(torch.version, "hip", "7.2.53214")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName=f"{arch}:sramecc+:xnack-"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda *a: pytest.fail("NVIDIA capability queried for AMD"),
    )
    assert attention.flash_attention_supported_device("cuda:1")


def test_unsupported_rocm_gpu_keeps_default_attention(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2.53214")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(gcnArchName="gfx1030")
    )
    assert not attention.flash_attention_supported_device("cuda")
    assert qwen.QwenTorchModel._attention_kwargs(True, "cuda") == {}


@pytest.mark.parametrize("capability,expected", [((7, 5), False), ((8, 0), True), ((9, 0), True)])
def test_cuda_preserves_ampere_requirement(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    assert attention.flash_attention_supported_device("cuda") is expected


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_non_cuda_devices_are_ineligible(device):
    assert not attention.flash_attention_supported_device(device)


def test_qwen_can_select_flash_attention_on_supported_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2.53214")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(gcnArchName="gfx1201")
    )
    monkeypatch.setattr(qwen, "FLASH_ATTENTION_AVAILABLE", True)
    assert qwen.QwenTorchModel._attention_kwargs(True, "cuda:0") == {
        "attn_implementation": "flash_attention_2"
    }
