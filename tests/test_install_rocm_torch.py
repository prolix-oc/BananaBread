"""Package selection and early diagnostics without requiring GPU hardware."""

import builtins
import importlib.metadata
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import install_rocm_torch


@pytest.mark.parametrize("restore_cuda,backend", [(False, "rocm7.2"), (True, "cu130")])
def test_linux_switch_selects_both_native_packages(monkeypatch, tmp_path, restore_cuda, backend):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.uv]\nextra-index-url = ["old"]\n')
    monkeypatch.setattr(install_rocm_torch, "PYPROJECT", pyproject)
    monkeypatch.setattr(install_rocm_torch, "require_uv", lambda: None)
    commands = []

    def run_command(command):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(install_rocm_torch, "run_cmd", run_command)
    monkeypatch.setattr(install_rocm_torch, "verify_subprocess", lambda _code: 0)

    with pytest.raises(SystemExit) as result:
        install_rocm_torch.install_linux(dry_run=False, restore_cuda=restore_cuda)

    assert result.value.code == 0
    assert commands == [
        [
            "uv",
            "lock",
            "--upgrade-package",
            f"torch==2.11.0+{backend}",
            "--upgrade-package",
            f"torchvision==0.26.0+{backend}",
        ],
        ["uv", "sync"],
    ]


def _verification_environment(monkeypatch, *, backend="rocm7.2", vision=None, native_ops=True):
    versions = {"torch": f"2.11.0+{backend}", "torchvision": vision or f"0.26.0+{backend}"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setenv("TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS", "")
    torch = SimpleNamespace(
        __version__=versions["torch"],
        version=SimpleNamespace(
            hip="7.2" if backend.startswith("rocm") else None,
            cuda="13.0" if backend == "cu130" else None,
        ),
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _id: "AMD Radeon"),
    )
    vision_imports = []
    original_import = builtins.__import__

    def import_package(name, *args, **kwargs):
        if name == "torch":
            return torch
        if name == "torchvision":
            vision_imports.append(name)
            return SimpleNamespace(extension=SimpleNamespace(_has_ops=lambda: native_ops))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_package)
    return vision_imports


@pytest.mark.parametrize("vision", ["0.26.0", "0.26.0+cu130"])
def test_rocm_verification_reports_wrong_wheel_before_import(monkeypatch, capsys, vision):
    imports = _verification_environment(monkeypatch, vision=vision)

    with pytest.raises(AssertionError, match="is not the ROCm build"):
        exec(install_rocm_torch.verification_code(), {})

    output = capsys.readouterr().out
    assert "torch 2.11.0+rocm7.2" in output
    assert f"torchvision {vision}" in output
    assert "HIP 7.2" in output
    assert imports == []


@pytest.mark.parametrize("restore_cuda,backend", [(False, "rocm7.2"), (True, "cu130")])
def test_verification_accepts_matching_native_builds(monkeypatch, restore_cuda, backend):
    imports = _verification_environment(monkeypatch, backend=backend)
    exec(install_rocm_torch.verification_code(restore_cuda), {})
    assert imports == ["torchvision"]
    assert os.environ["TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS"] == "1"


def test_verification_rejects_missing_native_operators(monkeypatch):
    _verification_environment(monkeypatch, native_ops=False)
    with pytest.raises(AssertionError, match="native operators failed to load"):
        exec(install_rocm_torch.verification_code(), {})
