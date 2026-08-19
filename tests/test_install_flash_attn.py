"""Regression tests for the prebuilt Flash Attention wheel matrix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import install_flash_attn


def test_prebuilt_wheels_match_torch_211_and_cuda_13():
    assert install_flash_attn.WHEEL_URLS
    for url in install_flash_attn.WHEEL_URLS.values():
        assert "flash_attn-2.8.3%2Bcu130torch2.11" in url


def test_prebuilt_wheels_cover_supported_python_platforms():
    assert set(install_flash_attn.WHEEL_URLS) == {
        ("linux_x86_64", "3.12"),
        ("linux_x86_64", "3.13"),
        ("win32", "3.12"),
        ("win32", "3.13"),
    }
