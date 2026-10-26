"""Exercise allocator environment defaults through real application startup."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("primary", "legacy", "expected_primary"),
    [
        (None, None, "expandable_segments:True"),
        ("backend:native", None, "backend:native"),
        (None, "max_split_size_mb:256", None),
        ("backend:native", "max_split_size_mb:256", "backend:native"),
    ],
)
def test_startup_defaults_allow_block_reuse_and_preserve_overrides(
    tmp_path, primary, legacy, expected_primary
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"embedding_device": "cpu", "rerank_device": "cpu"}))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key, value in (
        ("PYTORCH_ALLOC_CONF", primary),
        ("PYTORCH_CUDA_ALLOC_CONF", legacy),
    ):
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    script = """
import json, os, sys
sys.argv = ["bananabread", "--config", sys.argv[1]]
import bananabread.config
print(json.dumps([os.environ.get("PYTORCH_ALLOC_CONF"),
                  os.environ.get("PYTORCH_CUDA_ALLOC_CONF")]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == [expected_primary, legacy]
