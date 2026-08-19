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
    uv run python install_rocm_torch.py                # install ROCm torch
    uv run python install_rocm_torch.py --dry-run      # show what would happen
    uv run python install_rocm_torch.py --restore-cuda # back to CUDA 13.0 (Linux)

Windows prerequisites (before running):
    uv python pin 3.12
    uv sync --no-install-package torch
    uv run --no-sync python install_rocm_torch.py

References:
    Linux index contents:  https://download.pytorch.org/whl/rocm6.3
    Windows wheel source:  https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Linux: PyTorch's own index. rocm6.3 is the last index carrying torch 2.9.x;
# the rocm7.x indexes start at torch 2.10, which this project's pin
# (torch>=2.9.0,<2.10.0) excludes.
ROCM_INDEX = "https://download.pytorch.org/whl/rocm6.3"
CUDA_INDEX = "https://download.pytorch.org/whl/cu130"

# Windows: AMD's direct wheel URLs. AMD ships cp312 wheels only, and the
# torch wheel version must stay inside the project's torch pin.
AMD_ROCM_VERSION = "7.2.1"
AMD_BASE = f"https://repo.radeon.com/rocm/windows/rocm-rel-{AMD_ROCM_VERSION}"
AMD_SDK_URLS = [
    f"{AMD_BASE}/rocm_sdk_core-{AMD_ROCM_VERSION}-py3-none-win_amd64.whl",
    f"{AMD_BASE}/rocm_sdk_devel-{AMD_ROCM_VERSION}-py3-none-win_amd64.whl",
    f"{AMD_BASE}/rocm_sdk_libraries_custom-{AMD_ROCM_VERSION}-py3-none-win_amd64.whl",
    f"{AMD_BASE}/rocm-{AMD_ROCM_VERSION}.tar.gz",
]
AMD_TORCH_URL = (
    f"{AMD_BASE}/torch-2.9.1%2Brocm{AMD_ROCM_VERSION}-cp312-cp312-win_amd64.whl"
)
AMD_REQUIRED_PYTHON = (3, 12)


def check_torch():
    """Return (torch version, ROCm/HIP version or None) for this interpreter."""
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
            f"[tool.uv]\nextra-index-url = [\"{url}\"]"
        )
    return newline.join(out) + newline


def run_cmd(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), **kwargs)


def verify_subprocess(assertion: str) -> int:
    """Run `python -c <assertion>` in a fresh interpreter against the venv."""
    cmd = [sys.executable, "-c", assertion]
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
    label = "CUDA 13.0" if restore_cuda else "ROCm 6.3"

    original = PYPROJECT.read_text(encoding="utf-8")
    rewritten = rewrite_extra_index_url(original, target)

    print(f"pyproject.toml [tool.uv] change (target: {label}):")
    for old, new in zip(original.splitlines(), rewritten.splitlines()):
        if old != new:
            print(f"  - {old}")
            print(f"  + {new}")

    suffix = "+cu130" if restore_cuda else "+rocm"
    verify = (
        "import torch; v = torch.__version__; "
        f"assert {suffix!r} in v, v; print('torch', v, 'ready')"
    )
    steps = [
        ["uv", "lock", "--upgrade-package", "torch"],
        ["uv", "sync"],
    ]

    if dry_run:
        print("\n[dry-run] Would execute:")
        print(f"  write {PYPROJECT}")
        for step in steps:
            print(f"  {' '.join(step)}")
        print(f"  {sys.executable} -c {verify!r}")
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
            "  uv sync --no-install-package torch\n"
            "  uv run --no-sync python install_rocm_torch.py"
        )
        sys.exit(1)

    version, hip = check_torch()
    if hip:
        print(f"ROCm torch already installed: {version}")
        sys.exit(
            verify_subprocess(
                "import torch; assert torch.cuda.is_available(), 'GPU not visible'; "
                "print('AMD GPU visible:', torch.cuda.get_device_name(0))"
            )
        )

    print("Installing AMD ROCm SDK runtime components and torch wheel:")
    for url in AMD_SDK_URLS + [AMD_TORCH_URL]:
        print(f"  {url}")

    steps = [
        ["uv", "sync", "--no-install-package", "torch"],
        ["uv", "pip", "install", "--python", sys.executable, "--no-cache", *AMD_SDK_URLS],
        ["uv", "pip", "install", "--python", sys.executable, "--no-cache", AMD_TORCH_URL],
    ]

    if dry_run:
        print("\n[dry-run] Would execute:")
        for step in steps:
            print(f"  {' '.join(step)}")
        print(
            f"  {sys.executable} -c "
            f"'import torch; assert torch.cuda.is_available()'"
        )
        return

    for step in steps:
        result = run_cmd(step)
        if result.returncode != 0:
            print(f"\nCommand failed (exit code {result.returncode}). Aborting.")
            sys.exit(result.returncode)

    code = verify_subprocess(
        "import torch; v = torch.__version__; assert '+rocm' in v, v; "
        "assert torch.cuda.is_available(), 'AMD GPU not visible'; "
        "print('torch', v, '| GPU:', torch.cuda.get_device_name(0))"
    )
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

    version, hip = check_torch()
    print(f"Platform:  {sys.platform}")
    print(f"Python:    {sys.version_info.major}.{sys.version_info.minor} ({sys.executable})")
    print(f"PyTorch:   {version or 'not installed'}")
    print(f"ROCm/HIP:  {hip or 'none (CUDA or CPU build)'}")
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
