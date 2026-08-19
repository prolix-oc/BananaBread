import time
import sys
import json
import datetime
import psutil
import asyncio
from typing import Optional, Any

from bananabread.config import logger, args, EMBEDDING_LOGGING_ENABLED, EMBEDDING_LOG_FILE


# ----- Embedding Logging Function -----

def log_embedding_result(inputs: list[str], embeddings: list, metadata: dict):
    """
    Log embedding query and results to a JSON file.
    
    Args:
        inputs: List of input texts that were embedded
        embeddings: List of embedding vectors (as lists of floats)
        metadata: Dictionary containing additional metadata (model, quantization, etc.)
    """
    if not EMBEDDING_LOGGING_ENABLED:
        return
    
    try:
        # Create log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": metadata.get("model", "unknown"),
            "quantization": metadata.get("quantization", args.quant),
            "embedding_dimensions": metadata.get("embedding_dimensions", "unknown"),
            "num_inputs": len(inputs),
            "inputs": inputs,
            "embeddings": embeddings
        }
        
        # Append to log file (create if doesn't exist)
        with open(EMBEDDING_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
        logger.debug(f"📝 Logged {len(inputs)} embeddings to {EMBEDDING_LOG_FILE}")
        
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
