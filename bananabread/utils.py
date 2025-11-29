import time
import sys
import json
import datetime
import psutil
import asyncio
import tqdm
from typing import Optional, Any

from bananabread.config import logger, args, EMBEDDING_LOGGING_ENABLED, EMBEDDING_LOG_FILE

# ----- Custom Progress Tracking -----

class CustomProgressTracker:
    """Custom progress tracking with statistics for embedding operations"""
    def __init__(self, total_items: int, operation_name: str = "Embedding"):
        self.total_items = total_items
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.current_item = 0
        self.last_log_time = 0.0
        self.log_interval = 0.5  # Log every 0.5 seconds
        
    def start(self):
        """Start progress tracking"""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        logger.info(f"🚀 Starting {self.operation_name} operation for {self.total_items} items")
        
    def update(self, current_item: int):
        """Update progress and log statistics"""
        self.current_item = current_item
        current_time = time.time()
        
        # Log progress at intervals
        if current_time - self.last_log_time >= self.log_interval:
            self._log_progress(current_time)
            self.last_log_time = current_time
            
    def _log_progress(self, current_time: float):
        """Log current progress with statistics"""
        if self.start_time is None:
            return
            
        elapsed_time = current_time - self.start_time
        progress_percent = (self.current_item / self.total_items) * 100
        items_per_second = self.current_item / elapsed_time if elapsed_time > 0 else 0
        remaining_items = self.total_items - self.current_item
        eta_seconds = remaining_items / items_per_second if items_per_second > 0 else 0
        
        logger.info(
            f"📊 {self.operation_name} Progress: {progress_percent:.1f}% "
            f"({self.current_item}/{self.total_items}) | "
            f"Speed: {items_per_second:.2f} items/sec | "
            f"ETA: {eta_seconds:.1f}s"
        )
        
    def finish(self):
        """Finish progress tracking and log final statistics"""
        if self.start_time is None:
            return
            
        total_time = time.time() - self.start_time
        items_per_second = self.total_items / total_time if total_time > 0 else 0
        
        logger.info(
            f"✅ {self.operation_name} completed: {self.total_items} items "
            f"in {total_time:.2f}s ({items_per_second:.2f} items/sec)"
        )

def suppress_tqdm_progress():
    """Suppress all tqdm progress bars and replace with custom tracking"""
    # Override tqdm to disable progress bars
    original_tqdm = tqdm.tqdm
    original_trange = tqdm.trange
    
    def custom_tqdm(iterable=None, desc=None, total=None, disable=False, **kwargs):
        """Custom tqdm that intercepts MixedBread progress and shows our custom tracking"""
        # Check if this is a MixedBread embedding operation
        if desc and ('embedding' in desc.lower() or 'encode' in desc.lower()):
            # Create our custom progress tracker for this operation
            if total and total > 0:
                progress_tracker = CustomProgressTracker(total, "MixedBread Embedding")
                progress_tracker.start()
                
                # Wrap the iterable to update our progress tracker
                if iterable is not None:
                    def tracked_iterable():
                        for i, item in enumerate(iterable):
                            progress_tracker.update(i + 1)
                            yield item
                        progress_tracker.finish()
                    
                    return tracked_iterable()
                else:
                    # If no iterable, just return a dummy tqdm that updates our tracker
                    class DummyTqdm:
                        def __init__(self, *args, **kwargs):
                            self.n = 0
                            self.total = total
                            self.progress_tracker = progress_tracker
                        
                        def update(self, n=1):
                            self.n += n
                            self.progress_tracker.update(self.n)
                        
                        def close(self):
                            self.progress_tracker.finish()
                        
                        def __enter__(self):
                            return self
                        
                        def __exit__(self, *args):
                            self.close()
                    
                    return DummyTqdm()
        
        # For non-embedding operations, just disable the progress bar
        return original_tqdm(iterable=iterable, desc=desc, total=total, disable=True, **kwargs)
    
    def custom_trange(*args, **kwargs):
        """Custom trange that disables progress bars"""
        return original_trange(*args, disable=True, **kwargs)
    
    # Replace tqdm functions
    tqdm.tqdm = custom_tqdm
    tqdm.trange = custom_trange
    
    # Also disable autonotebook versions
    try:
        from tqdm import autonotebook
        autonotebook.tqdm = custom_tqdm
        autonotebook.trange = custom_trange
    except ImportError:
        pass

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
