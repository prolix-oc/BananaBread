import time
import sys
import json
import datetime
import psutil
import asyncio
from typing import Optional, Any

from tqdm.auto import tqdm as auto_tqdm

from bananabread.config import logger, args, EMBEDDING_LOGGING_ENABLED, EMBEDDING_LOG_FILE


# ----- Uniform tqdm -> logger routing -----
# Raw tqdm bars (Hugging Face downloads, transformers weight loading) paint
# ANSI escape sequences straight onto stderr, which garbles the structured
# log when lines interleave.  Every bar that would have rendered is routed
# through the logger instead: throttled progress lines while running and one
# completion line at close.  Bars created disabled (e.g. SentenceTransformers
# encode with show_progress_bar=False) and instantly-finished bars (cached
# snapshots) stay fully silent.  Patching the shared tqdm.auto base class
# covers hub's and transformers' subclasses alike.

TQDM_LOG_INTERVAL_SECONDS = 2.0


def _humanize(value: float, unit, unit_scale: bool) -> str:
    unit = unit if isinstance(unit, str) and unit not in ("it", "") else ""
    if not unit_scale:
        shown = int(value) if float(value).is_integer() else round(value, 1)
        return f"{shown}{unit}"
    for factor, prefix in ((1024**3, "G"), (1024**2, "M"), (1024, "K")):
        if value >= factor:
            return f"{value / factor:.1f}{prefix}{unit}"
    return f"{value:.1f}{unit}"


def _emit_tqdm_line(info, *, final: bool) -> None:
    try:
        elapsed = max(time.monotonic() - info["start"], 1e-9)
        # A disabled stdlib bar is a shell: __init__/update return early and
        # never populate desc/n/total, so the tracker owns the real counts.
        desc = info["desc"] or "progress"
        n = info["n"]
        total = info["total"]
        rate = f"{_humanize(n / elapsed, info['unit'], info['unit_scale'])}/s"
        if total:
            pct = min(100.0, 100.0 * n / total)
            text = (
                f"{desc}: {_humanize(n, info['unit'], info['unit_scale'])}/"
                f"{_humanize(total, info['unit'], info['unit_scale'])} ({pct:.0f}%, {rate})"
            )
        else:
            text = f"{desc}: {_humanize(n, info['unit'], info['unit_scale'])} ({rate})"
        if final:
            logger.info(f"✅ {text} in {elapsed:.1f}s")
        else:
            info["logged"] = True
            logger.info(f"⏳ {text}")
    except Exception:
        # Progress logging must never break the operation it is reporting.
        pass


def route_tqdm_to_logger() -> None:
    """Suppress rendered tqdm bars process-wide; log progress through the logger."""
    if getattr(auto_tqdm, "_bb_routed", False):
        return
    original_init = auto_tqdm.__init__
    original_update = auto_tqdm.update
    original_close = auto_tqdm.close

    def _init(self, *args, **kwargs):
        caller_disabled = bool(kwargs.get("disable"))
        if not caller_disabled:
            kwargs["disable"] = True  # no ANSI rendering; logger takes over
        original_init(self, *args, **kwargs)
        if not caller_disabled:
            self._bb_progress = {
                "start": time.monotonic(),
                "last": time.monotonic(),
                "logged": False,
                "n": 0,
                "desc": (kwargs.get("desc") or "").strip(),
                "total": kwargs.get("total"),
                "unit": kwargs.get("unit") or "",
                "unit_scale": bool(kwargs.get("unit_scale")),
            }

    def _update(self, n=1):
        info = getattr(self, "_bb_progress", None)
        if info is not None:
            info["n"] += n
            now = time.monotonic()
            if now - info["last"] >= TQDM_LOG_INTERVAL_SECONDS:
                info["last"] = now
                _emit_tqdm_line(info, final=False)
        original_update(self, n)

    def _close(self):
        info = getattr(self, "_bb_progress", None)
        if info is not None:
            self._bb_progress = None
            if info["logged"]:
                _emit_tqdm_line(info, final=True)
        original_close(self)

    auto_tqdm.__init__ = _init
    auto_tqdm.update = _update
    auto_tqdm.close = _close
    auto_tqdm._bb_routed = True


# ----- Embedding Logging Function -----

def log_embedding_result(inputs: list[str], embeddings: list, metadata: dict):
    """
    Log embedding query and results to a JSON file.
    
    Args:
        inputs: List of input strings that were embedded
        embeddings: The resulting embeddings
        metadata: Additional metadata to log
    """
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "inputs": inputs,
            "metadata": metadata
        }
        with open(EMBEDDING_LOG_FILE, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
    except Exception as e:
        logger.error(f"Failed to log embeddings: {e}")

# ----- Memory Profiling Helper Functions -----

import os

def get_process_memory_usage():
    """Return the current process memory usage (RSS) in bytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def get_model_memory_usage(model):
    """
    Approximate the memory used by a PyTorch model (by summing parameter sizes).
    If the model is wrapped (or not a pure nn.Module), try to access its .parameters().
    """
    total = 0
    try:
        # For QwenRawModel or SentenceTransformer
        actual_model = model
        if hasattr(model, "model"):
            actual_model = model.model
            
        for param in actual_model.parameters():
            total += param.numel() * param.element_size()
    except Exception:
        total = None
    return total

# ----- Optimized Threadpool Execution Functions -----

async def run_in_threadpool_with_executor(executor, func, *args, **kwargs):
    """
    Run a function in a specific threadpool executor for better CPU utilization.
    This allows us to dedicate threads to specific types of operations.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args, **kwargs)
