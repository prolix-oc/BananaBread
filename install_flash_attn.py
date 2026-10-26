#!/usr/bin/env python3
"""
Helper script to install a precompiled Flash Attention 2 wheel for your platform.

Detects your platform, Python version, and PyTorch CUDA/ROCm configuration, then
installs the matching precompiled wheel without compiling from source.

Usage:
    python install_flash_attn.py            # auto-detect and install
    python install_flash_attn.py --dry-run  # show what would be installed
"""

import sys
import platform
import subprocess
import argparse
import shutil
import textwrap

TORCH_VERSION = "2.14"
FLASH_ATTN_VERSIONS = {"cu130": "2.8.3", "rocm7.2": "2.8.4 (upstream commit 4a948e9)"}

# Wheels: https://github.com/mjun0812/flash-attention-prebuild-wheels
WHEEL_URLS = {
    ("win32", "3.12", "cu130"): (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.10.2/"
        "flash_attn-2.8.3%2Bcu130torch2.14-cp312-cp312-win_amd64.whl"
    ),
    ("win32", "3.13", "cu130"): (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.10.2/"
        "flash_attn-2.8.3%2Bcu130torch2.14-cp313-cp313-win_amd64.whl"
    ),
    ("linux_x86_64", "3.12", "cu130"): (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.10.0/"
        "flash_attn-2.8.3%2Bcu130torch2.14-cp312-cp312-linux_x86_64.whl"
    ),
    ("linux_x86_64", "3.13", "cu130"): (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.10.0/"
        "flash_attn-2.8.3%2Bcu130torch2.14-cp313-cp313-linux_x86_64.whl"
    ),
    ("linux_x86_64", "3.12", "rocm7.2"): (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v1.0.0/"
        "flash_attn-2.8.4%2Brocm7.2torch2.14git4a948e9-cp312-cp312-linux_x86_64.whl"
    ),
    ("linux_x86_64", "3.13", "rocm7.2"): (
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v1.0.0/"
        "flash_attn-2.8.4%2Brocm7.2torch2.14git4a948e9-cp313-cp313-linux_x86_64.whl"
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
        return torch_version, cuda_version, getattr(torch.version, "hip", None)
    except ImportError:
        return None, None, None


def select_wheel(plat, pyver, torch_ver, cuda_ver, hip_ver):
    """Reject incompatible builds before invoking any package manager."""
    if not torch_ver:
        raise ValueError("Install PyTorch first, then rerun this script.")
    if ".".join(torch_ver.split(".")[:2]) != TORCH_VERSION:
        raise ValueError(f"These wheels require PyTorch {TORCH_VERSION}.x; found {torch_ver}.")
    if hip_ver:
        if plat != "linux_x86_64":
            raise ValueError("Prebuilt ROCm FA2 wheels are Linux x86_64 only; use SDPA here.")
        backend = "rocm" + ".".join(hip_ver.split(".")[:2])
    elif cuda_ver:
        backend = "cu" + "".join(cuda_ver.split(".")[:2])
    else:
        raise ValueError("PyTorch has no CUDA/HIP runtime; these wheels require GPU PyTorch.")
    url = WHEEL_URLS.get((plat, pyver, backend))
    if url is None:
        raise ValueError(f"No matching prebuilt wheel for {plat}, Python {pyver}, {backend}.")
    return backend, url


def verification_code():
    """Exercise both FA2 entry points used by Transformers on the actual GPU."""
    return textwrap.dedent("""\
        import torch
        import flash_attn
        import flash_attn.bert_padding
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        assert torch.cuda.is_available(), 'CUDA/HIP GPU is not visible'
        q, k, v = [torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.float16) for _ in range(3)]
        with torch.inference_mode():
            actual = flash_attn_func(q, k, v)
            expected = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
            lengths = torch.tensor([0, 16], device='cuda', dtype=torch.int32)
            varlen = flash_attn_varlen_func(
                q[0], k[0], v[0], lengths, lengths, 16, 16
            )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(varlen, expected[0], atol=5e-3, rtol=5e-3)
        print('flash_attn', flash_attn.__version__, '| GPU', torch.cuda.get_device_name(0),
              '| fixed and variable-length attention verified')
        """)


def verify_flash_attn_gpu():
    cmd = [
        sys.executable,
        "-c",
        verification_code(),
    ]
    return subprocess.run(cmd)


def _has_pip():
    """Check whether the current interpreter has pip available."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
    )
    return result.returncode == 0


def _has_uv():
    """Check whether the `uv` command is available on PATH."""
    return shutil.which("uv") is not None


def build_install_cmd(url: str):
    """Return the package-install command for the current environment.

    Prefers `uv pip install` when `uv` is available (common for uv-managed
    venvs that don't ship pip). Falls back to `python -m pip install` when
    pip is present. Otherwise returns ``None``.
    """
    if _has_uv():
        return ["uv", "pip", "install", "--python", sys.executable, url]
    if _has_pip():
        return [sys.executable, "-m", "pip", "install", url]
    return None


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
    torch_ver, cuda_ver, hip_ver = check_torch()

    print(f"Platform:     {plat}")
    print(f"Python:       {pyver} ({sys.executable})")
    print(f"PyTorch:      {torch_ver or 'not installed'}")
    print(f"CUDA (torch): {cuda_ver or 'N/A'}")
    print(f"HIP (torch):  {hip_ver or 'N/A'}")
    print()

    # --- Platform guard ---
    if plat == "darwin":
        print("These Flash Attention 2 wheels require CUDA or ROCm. macOS is not supported.")
        sys.exit(1)

    # --- Look up the wheel ---
    try:
        backend, url = select_wheel(plat, pyver, torch_ver, cuda_ver, hip_ver)
    except ValueError as exc:
        print(f"ERROR: {exc}\n")
        print("Supported combinations:")
        for p, pv, build in sorted(WHEEL_URLS):
            print(f"  {p}  Python {pv}  Torch {TORCH_VERSION}  {build}")
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

    print(f"Selected: flash-attn {FLASH_ATTN_VERSIONS[backend]} ({backend})")
    print(f"Wheel:    {url}")
    print()

    cmd = build_install_cmd(url)

    if args.dry_run:
        print("[dry-run] Would execute:")
        if cmd:
            print(f"  {' '.join(cmd)}")
        else:
            print("  (no package manager found – neither uv nor pip available)")
        return

    if cmd is None:
        print(
            "ERROR: No package manager found.\n"
            "  - If you're using uv, ensure `uv` is on your PATH.\n"
            "  - If you're using pip, ensure pip is installed in this environment.\n"
        )
        sys.exit(1)

    print(f"Using: {' '.join(cmd[:2])}")
    print("Installing...")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\nFlash Attention 2 installed. Verifying GPU attention...")
        verify_result = verify_flash_attn_gpu()
        if verify_result.returncode == 0:
            print("\nFlash Attention 2 installed and GPU-verified successfully!")
            print("Enable it with: --embedding-flash-attention (or --qwen-flash-attention)")
        else:
            print("\nInstallation completed, but Flash Attention 2 GPU verification failed.")
            print("Check the error above for a wheel/runtime mismatch or unsupported GPU.")
            sys.exit(verify_result.returncode)
    else:
        print(f"\nInstallation failed (exit code {result.returncode}).")
        print("You can try manually:")
        print(f'  uv pip install --python {sys.executable} "{url}"')
        print("or")
        print(f'  pip install "{url}"')
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
