"""FA2 wheel selection must match both Torch and its CUDA/HIP runtime."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import install_flash_attn


@pytest.mark.parametrize("pyver,tag", [("3.12", "cp312"), ("3.13", "cp313")])
@pytest.mark.parametrize("plat", ["win32", "linux_x86_64"])
def test_cuda_wheel_matches_runtime(plat, pyver, tag):
    backend, url = install_flash_attn.select_wheel(plat, pyver, "2.14.1", "13.0", None)
    assert backend == "cu130"
    assert f"flash_attn-2.8.3%2Bcu130torch2.14-{tag}-{tag}-" in url
    assert url.endswith("win_amd64.whl" if plat == "win32" else "linux_x86_64.whl")


@pytest.mark.parametrize("pyver,tag", [("3.12", "cp312"), ("3.13", "cp313")])
def test_rocm_wheel_matches_runtime(pyver, tag):
    backend, url = install_flash_attn.select_wheel(
        "linux_x86_64", pyver, "2.14.1", None, "7.2.53214"
    )
    assert backend == "rocm7.2"
    assert f"flash_attn-2.8.4%2Brocm7.2torch2.14git4a948e9-{tag}-{tag}-linux_x86_64.whl" in url


@pytest.mark.parametrize(
    "plat,pyver,torch_ver,cuda,hip",
    [
        ("linux_x86_64", "3.13", None, None, None),
        ("linux_x86_64", "3.13", "2.14.1", None, None),
        ("linux_x86_64", "3.13", "2.11.0", "13.0", None),
        ("linux_x86_64", "3.13", "2.14.1", "13.2", None),
        ("linux_x86_64", "3.13", "2.14.1", None, "7.1.0"),
        ("win32", "3.12", "2.9.1", None, "7.2.1"),
        ("win32", "3.12", "2.14.1", None, "7.2.1"),
        ("linux_aarch64", "3.13", "2.14.1", None, "7.2.0"),
        ("linux_x86_64", "3.14", "2.14.1", None, "7.2.0"),
    ],
)
def test_incompatible_runtime_never_selects_a_wheel(plat, pyver, torch_ver, cuda, hip):
    with pytest.raises(ValueError):
        install_flash_attn.select_wheel(plat, pyver, torch_ver, cuda, hip)


def _rocm_main(monkeypatch, *, dry_run):
    monkeypatch.setattr(sys, "argv", ["install_flash_attn.py"] + (["--dry-run"] if dry_run else []))
    monkeypatch.setattr(install_flash_attn, "get_platform_key", lambda: "linux_x86_64")
    monkeypatch.setattr(install_flash_attn, "get_python_version_key", lambda: "3.13")
    monkeypatch.setattr(install_flash_attn, "check_torch", lambda: ("2.14.1", None, "7.2.53214"))
    monkeypatch.setattr(
        install_flash_attn, "build_install_cmd", lambda url: ["uv", "pip", "install", url]
    )


def test_rocm_dry_run_does_not_install_cuda_or_execute(monkeypatch, capsys):
    _rocm_main(monkeypatch, dry_run=True)
    monkeypatch.setattr(
        install_flash_attn.subprocess, "run", lambda *a, **k: pytest.fail("executed")
    )
    install_flash_attn.main()
    output = capsys.readouterr().out
    assert "rocm7.2torch2.14git4a948e9" in output
    assert "cu130torch" not in output


def test_failed_gpu_verification_returns_failure(monkeypatch):
    _rocm_main(monkeypatch, dry_run=False)
    monkeypatch.setattr(
        install_flash_attn.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        install_flash_attn, "verify_flash_attn_gpu", lambda: SimpleNamespace(returncode=9)
    )
    with pytest.raises(SystemExit) as exc:
        install_flash_attn.main()
    assert exc.value.code == 9


def test_installer_falls_back_to_pip_without_uv(monkeypatch):
    monkeypatch.setattr(install_flash_attn.shutil, "which", lambda _name: None)
    monkeypatch.setattr(install_flash_attn, "_has_pip", lambda: True)
    assert install_flash_attn.build_install_cmd("wheel-url") == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "wheel-url",
    ]
