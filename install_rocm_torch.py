#!/usr/bin/env python3
"""
Install an AMD ROCm build of PyTorch for BananaBread.

Linux:
    Rewrites the PyTorch wheel index in pyproject.toml to the ROCm index,
    re-locks, and syncs. The lockfile itself then contains the ROCm build,
    so plain `uv run bananabread-emb` keeps working with no extra flags.

Windows:
    Installs AMD's "PyTorch on Windows" wheels (ROCm 7.2.1) directly from
    repo.radeon.com, since no pip index serves them. The lockfile cannot
    describe these wheels, so afterwards always launch with
    `uv run --no-sync` (a plain `uv run` would restore the CUDA wheel).

Usage:
    uv sync --no-install-package torch
    uv run --no-sync python install_rocm_torch.py                # install ROCm torch
    uv run --no-sync python install_rocm_torch.py --dry-run      # preview
    uv run --no-sync python install_rocm_torch.py --restore-cuda # restore CUDA (Linux)

Windows prerequisites (before running):
    uv python pin 3.12
    uv sync --no-install-package torch --no-install-package torchvision
    uv run --no-sync python install_rocm_torch.py

References:
    Linux index contents:  https://download.pytorch.org/whl/rocm7.2
    Windows wheel source:  https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Linux: PyTorch 2.11 is the first shared release published for both ROCm 7.2
# and CUDA 13.0.
ROCM_INDEX = "https://download.pytorch.org/whl/rocm7.2"
CUDA_INDEX = "https://download.pytorch.org/whl/cu130"
TORCH_VERSION = "2.11.0"
ROCM_TORCH_VERSION = f"{TORCH_VERSION}+rocm7.2"
CUDA_TORCH_VERSION = f"{TORCH_VERSION}+cu130"

# Windows: AMD's direct wheel URLs. AMD ships cp312 wheels only, and the
# torch wheel version must stay inside the project's torch pin.
AMD_ROCM_VERSION = "7.2.1"
AMD_WINDOWS_TORCH_VERSION = "2.9.1"
AMD_BASE = f"https://repo.radeon.com/rocm/windows/rocm-rel-{AMD_ROCM_VERSION}"
AMD_SDK_URLS = [
    f"{AMD_BASE}/rocm_sdk_core-{AMD_ROCM_VERSION}-py3-none-win_amd64.whl",
    f"{AMD_BASE}/rocm_sdk_devel-{AMD_ROCM_VERSION}-py3-none-win_amd64.whl",
    f"{AMD_BASE}/rocm_sdk_libraries_custom-{AMD_ROCM_VERSION}-py3-none-win_amd64.whl",
    f"{AMD_BASE}/rocm-{AMD_ROCM_VERSION}.tar.gz",
]
AMD_TORCH_URL = (
    f"{AMD_BASE}/torch-{AMD_WINDOWS_TORCH_VERSION}%2Brocm{AMD_ROCM_VERSION}"
    "-cp312-cp312-win_amd64.whl"
)
AMD_TORCHVISION_URL = (
    f"{AMD_BASE}/torchvision-0.24.1%2Brocm{AMD_ROCM_VERSION}" "-cp312-cp312-win_amd64.whl"
)
AMD_REQUIRED_PYTHON = (3, 12)


def project_python() -> Path:
    """Return the interpreter used by BananaBread's project environment."""
    if sys.platform == "win32":
        return REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return REPO_ROOT / ".venv" / "bin" / "python"


def check_torch(python_executable: str | Path | None = None):
    """Return (torch version, ROCm/HIP version) for an interpreter.

    With no argument this inspects the installer process. Passing the project
    interpreter is important when a shell alias makes ``python`` point outside
    ``.venv`` even after activation.
    """
    if python_executable is not None:
        executable = Path(python_executable)
        if not executable.exists():
            return None, None
        probe = (
            "import json, torch; "
            "print(json.dumps([torch.__version__, getattr(torch.version, 'hip', None)]))"
        )
        result = subprocess.run(
            [str(executable), "-c", probe],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None, None
        try:
            return tuple(json.loads(result.stdout.strip().splitlines()[-1]))
        except (IndexError, json.JSONDecodeError, TypeError, ValueError):
            return None, None

    try:
        import torch

        return torch.__version__, getattr(torch.version, "hip", None)
    except ImportError:
        return None, None


def rewrite_extra_index_url(text: str, url: str) -> str:
    """Replace the `extra-index-url` value inside the [tool.uv] table.

    Raises RuntimeError when the table or the key is missing, so an unexpected
    pyproject layout fails loudly instead of being silently mis-edited.
    """
    out = []
    in_tool_uv = False
    seen_key = False
    newline = "\r\n" if "\r\n" in text else "\n"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_tool_uv = stripped == "[tool.uv]"
            out.append(line)
            continue
        if in_tool_uv and re.match(r"extra-index-url\s*=", stripped):
            seen_key = True
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f'{indent}extra-index-url = ["{url}"]')
            continue
        out.append(line)
    if not seen_key:
        raise RuntimeError(
            "No `extra-index-url` key found under [tool.uv] in pyproject.toml.\n"
            "Add one first, for example:\n\n"
            f'[tool.uv]\nextra-index-url = ["{url}"]'
        )
    return newline.join(out) + newline


def linux_torch_requirement(restore_cuda: bool) -> str:
    """Return the exact build uv must select for the requested backend.

    uv lockfiles cover every supported platform. Since ROCm wheels only cover
    Linux x86_64, a loose ``torch`` upgrade can otherwise choose a
    cross-platform CUDA build instead of the requested ROCm build.
    """
    version = CUDA_TORCH_VERSION if restore_cuda else ROCM_TORCH_VERSION
    return f"torch=={version}"


def run_cmd(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), **kwargs)


def verification_code(restore_cuda: bool = False) -> str:
    """Return checks that distinguish a working ROCm build from CUDA."""
    if restore_cuda:
        return (
            "import sys, torch, torchvision; v = torch.__version__; "
            "assert '+cu130' in v, v; "
            "assert torch.version.cuda, 'CUDA runtime missing'; "
            "print('python', sys.executable, '| torch', v, '| CUDA', torch.version.cuda)"
        )
    return (
        "import sys, torch, torchvision; v = torch.__version__; "
        "assert '+rocm' in v, v; "
        "assert torch.version.hip, 'ROCm/HIP runtime missing'; "
        "assert torch.cuda.is_available(), 'AMD GPU not visible'; "
        "assert '+rocm' in torchvision.__version__, 'torchvision is not the ROCm build'; "
        "print('python', sys.executable, '| torch', v, '| HIP', torch.version.hip, "
        "'| GPU', torch.cuda.get_device_name(0))"
    )


def verify_subprocess(assertion: str, python_executable: str | Path | None = None) -> int:
    """Run checks in BananaBread's project interpreter, never a shell alias."""
    executable = Path(python_executable) if python_executable else project_python()
    if not executable.exists():
        print(f"\nVerification FAILED: project interpreter not found at {executable}")
        return 1
    cmd = [str(executable), "-c", assertion]
    print("\nVerifying installation...")
    result = run_cmd(cmd)
    if result.returncode != 0:
        print("\nVerification FAILED. See the error above.")
    return result.returncode


def require_uv():
    if shutil.which("uv") is None:
        print(
            "ERROR: uv is required (project lock/sync operations). "
            "Install it from https://docs.astral.sh/uv/getting-started/installation/"
        )
        sys.exit(1)


def install_linux(dry_run: bool, restore_cuda: bool) -> None:
    require_uv()
    target = CUDA_INDEX if restore_cuda else ROCM_INDEX
    label = "Torch 2.11 + CUDA 13.0" if restore_cuda else "Torch 2.11 + ROCm 7.2"

    original = PYPROJECT.read_text(encoding="utf-8")
    rewritten = rewrite_extra_index_url(original, target)

    print(f"pyproject.toml [tool.uv] change (target: {label}):")
    for old, new in zip(original.splitlines(), rewritten.splitlines()):
        if old != new:
            print(f"  - {old}")
            print(f"  + {new}")

    verify = verification_code(restore_cuda)
    steps = [
        ["uv", "lock", "--upgrade-package", linux_torch_requirement(restore_cuda)],
        ["uv", "sync"],
    ]

    if dry_run:
        print("\n[dry-run] Would execute:")
        print(f"  write {PYPROJECT}")
        for step in steps:
            print(f"  {' '.join(step)}")
        print(f"  {project_python()} -c {verify!r}")
        return

    PYPROJECT.write_text(rewritten, encoding="utf-8")
    for step in steps:
        result = run_cmd(step)
        if result.returncode != 0:
            print(f"\nCommand failed (exit code {result.returncode}). Aborting.")
            sys.exit(result.returncode)

    code = verify_subprocess(verify)
    if code == 0:
        print(
            "\nDone. The lockfile now pins the "
            f"{label} build, so plain `uv run bananabread-emb "
            "--embedding-device cuda` works with no extra flags."
        )
    sys.exit(code)


def install_windows(dry_run: bool) -> None:
    require_uv()
    pyver = (sys.version_info.major, sys.version_info.minor)
    if pyver != AMD_REQUIRED_PYTHON:
        print(
            f"ERROR: AMD's PyTorch on Windows wheels are cp312-only, but this "
            f"environment is Python {pyver[0]}.{pyver[1]}.\n"
            "Fix with:\n"
            "  uv python pin 3.12\n"
            "  (delete .venv if it exists)\n"
            "  uv sync --no-install-package torch --no-install-package torchvision\n"
            "  uv run --no-sync python install_rocm_torch.py"
        )
        sys.exit(1)

    target_python = project_python()
    version, hip = check_torch(target_python)
    if hip:
        print(
            f"ROCm torch already installed in {target_python}: {version}; ensuring matching torchvision"
        )

    print("Installing matching AMD ROCm torch and torchvision wheels:")
    for url in ([] if hip else AMD_SDK_URLS) + [AMD_TORCH_URL, AMD_TORCHVISION_URL]:
        print(f"  {url}")

    steps = (
        []
        if hip
        else [
            ["uv", "sync", "--no-install-package", "torch", "--no-install-package", "torchvision"],
            ["uv", "pip", "install", "--python", str(target_python), "--no-cache", *AMD_SDK_URLS],
        ]
    )
    steps.append(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(target_python),
            "--no-cache",
            AMD_TORCH_URL,
            AMD_TORCHVISION_URL,
        ]
    )

    if dry_run:
        print("\n[dry-run] Would execute:")
        for step in steps:
            print(f"  {' '.join(step)}")
        print(f"  {target_python} -c {verification_code()!r}")
        return

    for step in steps:
        result = run_cmd(step)
        if result.returncode != 0:
            print(f"\nCommand failed (exit code {result.returncode}). Aborting.")
            sys.exit(result.returncode)

    code = verify_subprocess(verification_code(), target_python)
    if code == 0:
        print(
            "\nDone. IMPORTANT: the lockfile cannot describe AMD's Windows wheels,\n"
            "so always launch with --no-sync from now on:\n"
            "  uv run --no-sync bananabread-emb --embedding-device cuda --rerank-device cuda\n"
            "A plain `uv run` would re-sync torch back to the CUDA wheel."
        )
    sys.exit(code)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Install an AMD ROCm build of PyTorch for BananaBread "
            "(Linux: re-index + re-lock; Windows: AMD wheels from repo.radeon.com)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without changing anything.",
    )
    parser.add_argument(
        "--restore-cuda",
        action="store_true",
        help="Linux only: switch the PyTorch index back to CUDA 13.0.",
    )
    args = parser.parse_args()

    installer_version, installer_hip = check_torch()
    target_python = project_python()
    target_version, target_hip = check_torch(target_python)
    print(f"Platform:         {sys.platform}")
    print(
        f"Installer Python: {sys.version_info.major}.{sys.version_info.minor} "
        f"({sys.executable})"
    )
    print(f"Project Python:   {target_python}")
    print(f"Project PyTorch:  {target_version or 'not installed'}")
    print(f"Project ROCm/HIP: {target_hip or 'none (CUDA or CPU build)'}")
    if Path(sys.executable).resolve() != target_python.resolve():
        print(
            "NOTE: the installer and BananaBread interpreters differ. All installs and "
            "verification will target Project Python."
        )
    elif installer_version != target_version or installer_hip != target_hip:
        print("NOTE: project Torch metadata changed while it was being inspected.")
    print()

    if sys.platform == "darwin":
        print("ROCm requires AMD hardware and is not available on macOS.")
        sys.exit(1)
    if sys.platform == "win32":
        if args.restore_cuda:
            print("--restore-cuda is Linux only. On Windows, delete .venv and run `uv sync`.")
            sys.exit(1)
        install_windows(args.dry_run)
    else:
        install_linux(args.dry_run, args.restore_cuda)


if __name__ == "__main__":
    main()
