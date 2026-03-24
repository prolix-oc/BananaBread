#!/usr/bin/env python3
"""
Helper script to install a precompiled Flash Attention 2 wheel for your platform.

Detects your platform, Python version, and PyTorch/CUDA configuration, then
installs the correct precompiled wheel — no CUDA toolkit or C++ compiler needed.

Usage:
    python install_flash_attn.py            # auto-detect and install
    python install_flash_attn.py --dry-run  # show what would be installed
"""

import sys
import platform
import subprocess
import argparse

FLASH_ATTN_VERSION = "2.8.3"

# Windows wheels: https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows
# Linux wheels:   https://github.com/mjun0812/flash-attention-prebuild-wheels (v0.7.16)
WHEEL_URLS = {
    ("win32", "3.12"): (
        "https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/"
        "flash_attn-2.8.3+cu130torch2.9.1cxx11abiTRUE-cp312-cp312-win_amd64.whl"
    ),
    ("win32", "3.13"): (
        "https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/"
        "flash_attn-2.8.3+cu130torch2.9.1cxx11abiTRUE-cp313-cp313-win_amd64.whl"
    ),
    ("linux_x86_64", "3.12"): (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/"
        "flash_attn-2.8.3+cu130torch2.9-cp312-cp312-linux_x86_64.whl"
    ),
    ("linux_x86_64", "3.13"): (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/"
        "flash_attn-2.8.3+cu130torch2.9-cp313-cp313-linux_x86_64.whl"
    ),
}


def get_platform_key():
    if sys.platform == "win32":
        return "win32"
    elif sys.platform == "linux":
        machine = platform.machine()
        return f"linux_{machine}"
    elif sys.platform == "darwin":
        return "darwin"
    return sys.platform


def get_python_version_key():
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def check_torch():
    try:
        import torch
        cuda_version = torch.version.cuda
        torch_version = torch.__version__.split("+")[0]
        return torch_version, cuda_version
    except ImportError:
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Install a precompiled Flash Attention 2 wheel for your platform.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without installing.",
    )
    args = parser.parse_args()

    plat = get_platform_key()
    pyver = get_python_version_key()
    torch_ver, cuda_ver = check_torch()

    print(f"Platform:     {plat}")
    print(f"Python:       {pyver} ({sys.executable})")
    print(f"PyTorch:      {torch_ver or 'not installed'}")
    print(f"CUDA (torch): {cuda_ver or 'N/A'}")
    print()

    # --- Platform guard ---
    if plat == "darwin":
        print("Flash Attention 2 requires an NVIDIA GPU with CUDA. macOS is not supported.")
        sys.exit(1)

    # --- Warnings ---
    if torch_ver is None:
        print(
            "WARNING: PyTorch is not installed yet. Flash Attention requires PyTorch.\n"
            "  Install PyTorch first, then re-run this script.\n"
            "  See: https://pytorch.org/get-started/locally/\n"
        )

    if cuda_ver and not cuda_ver.startswith("13."):
        print(
            f"WARNING: Your PyTorch reports CUDA {cuda_ver}, but the precompiled\n"
            f"  wheels bundled with this project target CUDA 13.0. The wheel may\n"
            f"  not be compatible. If you hit errors at runtime, you may need a\n"
            f"  wheel built for your CUDA version.\n"
        )

    if torch_ver and not torch_ver.startswith("2.9"):
        print(
            f"WARNING: Your PyTorch version is {torch_ver}, but the precompiled\n"
            f"  wheels target PyTorch 2.9.x. Consider aligning your PyTorch version.\n"
        )

    # --- Look up the wheel ---
    key = (plat, pyver)
    url = WHEEL_URLS.get(key)

    if url is None:
        print(f"No precompiled wheel available for {plat} + Python {pyver}.\n")
        print("Supported combinations:")
        for (p, pv) in sorted(WHEEL_URLS):
            print(f"  {p}  Python {pv}")
        print()
        print("Options:")
        print("  - Use Python 3.12 or 3.13")
        if "win" in plat:
            print(
                "  - Browse https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows\n"
                "    for other Windows wheels"
            )
        else:
            print(
                "  - Browse https://github.com/mjun0812/flash-attention-prebuild-wheels/releases\n"
                "    for other Linux wheels"
            )
            print("  - On Linux you can also compile from source: pip install flash-attn")
        sys.exit(1)

    print(f"Selected: flash-attn {FLASH_ATTN_VERSION}")
    print(f"Wheel:    {url}")
    print()

    if args.dry_run:
        print("[dry-run] Would execute:")
        print(f"  {sys.executable} -m pip install \"{url}\"")
        return

    cmd = [sys.executable, "-m", "pip", "install", url]
    print("Installing...")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\nFlash Attention 2 installed successfully!")
        print("Enable it with: --qwen-flash-attention")
    else:
        print(f"\nInstallation failed (exit code {result.returncode}).")
        print("You can try manually:")
        print(f"  pip install \"{url}\"")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
