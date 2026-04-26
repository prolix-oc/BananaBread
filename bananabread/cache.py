import time
import threading
import hashlib
import sys
import torch
from collections import OrderedDict
from typing import Optional, Any

from bananabread.config import logger

# ----- CUDA Cache Manager -----

class CUDACacheManager:
    """Manages CUDA cache with TTL-based automatic clearing"""
    
    def __init__(self, ttl_seconds: int = 300, min_clear_interval: int = 60, 
                 memory_threshold: int = 80, enabled: bool = False):
        
        self.ttl_seconds = ttl_seconds
        self.min_clear_interval = min_clear_interval
        self.memory_threshold = memory_threshold
        self.enabled = enabled
        
        # Track last inference and clear times
        self.last_inference_time = time.time()
        self.last_clear_time = time.time()
        self.clear_count = 0
        
        # Thread safety
        self.activity_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.monitor_thread = None
        
        # CUDA availability
        self.cuda_available = torch.cuda.is_available()
        
        if self.enabled and self.cuda_available:
            logger.info(f"🧹 CUDA Cache Manager enabled:")
            logger.info(f"   - TTL: {ttl_seconds}s")
            logger.info(f"   - Min clear interval: {min_clear_interval}s")
            logger.info(f"   - Memory threshold: {memory_threshold}%")
            self.start_monitor_thread()
        elif self.enabled and not self.cuda_available:
            logger.warning("⚠️  CUDA Cache Manager enabled but CUDA not available")
        else:
            logger.info("CUDA Cache Manager disabled (use --cuda-cache-ttl-enabled to enable)")
    
    def mark_inference_activity(self):
        """Mark that an inference operation just occurred"""
        with self.activity_lock:
            self.last_inference_time = time.time()
    
    def get_cuda_memory_stats(self):
        """Get CUDA memory statistics"""
        if not self.cuda_available:
            return None
        
        try:
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            
            # Calculate percentage if we can get max memory
            try:
                max_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_pct = (allocated / max_memory) * 100 if max_memory > 0 else 0
                reserved_pct = (reserved / max_memory) * 100 if max_memory > 0 else 0
            except:
                allocated_pct = 0
                reserved_pct = 0
            
            return {
                "allocated": allocated,
                "reserved": reserved,
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "allocated_pct": allocated_pct,
                "reserved_pct": reserved_pct
            }
        except Exception as e:
            logger.error(f"Error getting CUDA memory stats: {e}")
            return None
    
    def should_clear_cache(self):
        """Determine if CUDA cache should be cleared"""
        if not self.cuda_available:
            return False
        
        current_time = time.time()
        
        # Check minimum clear interval
        time_since_last_clear = current_time - self.last_clear_time
        if time_since_last_clear < self.min_clear_interval:
            return False
        
        # Check if idle for TTL duration
        time_since_inference = current_time - self.last_inference_time
        if time_since_inference < self.ttl_seconds:
            return False
        
        # Check memory threshold
        stats = self.get_cuda_memory_stats()
        if stats and stats["reserved_pct"] < self.memory_threshold:
            return False
        
        return True
    
    def clear_cuda_cache(self, reason: str = "TTL"):
        """Clear CUDA cache and log statistics"""
        if not self.cuda_available:
            return
        
        # Get memory stats before clearing
        stats_before = self.get_cuda_memory_stats()
        
        try:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Get memory stats after clearing
            stats_after = self.get_cuda_memory_stats()
            
            # Update tracking
            self.last_clear_time = time.time()
            self.clear_count += 1
            
            # Log the clearing operation
            if stats_before and stats_after:
                freed_gb = stats_before["reserved_gb"] - stats_after["reserved_gb"]
                logger.info(
                    f"🧹 CUDA cache cleared ({reason}): "
                    f"Reserved {stats_before['reserved_gb']:.2f}GB → {stats_after['reserved_gb']:.2f}GB "
                    f"(freed {freed_gb:.2f}GB) | "
                    f"Allocated {stats_before['allocated_gb']:.2f}GB → {stats_after['allocated_gb']:.2f}GB | "
                    f"Clear #{self.clear_count}"
                )
            else:
                logger.info(f"🧹 CUDA cache cleared ({reason}) | Clear #{self.clear_count}")
                
        except Exception as e:
            logger.error(f"Error clearing CUDA cache: {e}")
    
    def monitor_loop(self):
        """Background monitoring loop for automatic cache clearing"""
        logger.info("🔄 CUDA cache monitor thread started")
        
        while not self.stop_event.is_set():
            try:
                # Check every 10 seconds
                if self.stop_event.wait(timeout=10):
                    break
                
                # Determine if we should clear cache
                if self.should_clear_cache():
                    # Use lock to prevent clearing during inference
                    with self.activity_lock:
                        # Double-check after acquiring lock
                        if self.should_clear_cache():
                            self.clear_cuda_cache(reason="TTL")
                
            except Exception as e:
                logger.error(f"Error in CUDA cache monitor loop: {e}")
        
        logger.info("🔄 CUDA cache monitor thread stopped")
    
    def start_monitor_thread(self):
        """Start the background monitor thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(
                target=self.monitor_loop,
                name="cuda-cache-monitor",
                daemon=True
            )
            self.monitor_thread.start()
    
    def stop_monitor_thread(self):
        """Stop the background monitor thread"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.info("Stopping CUDA cache monitor thread...")
            self.stop_event.set()
            self.monitor_thread.join(timeout=5)
            if self.monitor_thread.is_alive():
                logger.warning("⚠️  CUDA cache monitor thread did not stop gracefully")
    
    def get_stats(self):
        """Get statistics about CUDA cache management"""
        current_time = time.time()
        return {
            "enabled": self.enabled,
            "cuda_available": self.cuda_available,
            "ttl_seconds": self.ttl_seconds,
            "min_clear_interval": self.min_clear_interval,
            "memory_threshold": self.memory_threshold,
            "clear_count": self.clear_count,
            "last_clear_time": self.last_clear_time,
            "last_inference_time": self.last_inference_time,
            "time_since_last_clear": current_time - self.last_clear_time,
            "time_since_last_inference": current_time - self.last_inference_time,
            "cuda_memory": self.get_cuda_memory_stats()
        }

# ----- Limited Cache -----

def get_cache_size(obj):
    """
    Fast structural size estimate in bytes.

    Previously this walked every element of nested lists/dicts recursively,
    which for an OpenAI-shaped embeddings response (thousands of 1024-dim
    float lists) meant millions of sys.getsizeof() calls on the event loop
    thread — easily seconds of stall per request. The eviction path called
    it a second time per evicted value.

    This version short-circuits on the common cached-response shape:
      {"object": "list", "data": [{"embedding": [...], ...}, ...], ...}
    and falls back to a bounded shallow walk for anything else. Accuracy
    only needs to be within an order of magnitude for eviction to behave
    sensibly, since ``cache_limit_bytes`` is a soft target.
    """
    # Fast path: OpenAI embeddings response
    if isinstance(obj, dict):
        data = obj.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "embedding" in first:
                emb = first["embedding"]
                if isinstance(emb, list):
                    # Python float ~ 24 bytes; list overhead ~ 56 + 8/elem
                    dim = len(emb)
                    per_row = 56 + dim * (8 + 24)
                    return len(data) * (per_row + 200)  # 200 for dict/object overhead
                if isinstance(emb, (bytes, str)):
                    # base64 / bytes path
                    return len(data) * (len(emb) + 256)

    # Fallback: bounded shallow estimate (no deep recursion over floats)
    try:
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sys.getsizeof(obj[0]) * len(obj) if obj else 0
        return size
    except Exception:
        return 1024  # harmless default

class LimitedCache(OrderedDict):
    def __init__(self, cache_limit_bytes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_limit_bytes = cache_limit_bytes
        self.current_size = 0
        self.lock = threading.RLock()
    
    def __setitem__(self, key, value):
        with self.lock:
            # If key already exists, remove its previous size.
            if OrderedDict.__contains__(self, key):
                self.current_size -= get_cache_size(OrderedDict.__getitem__(self, key))
            OrderedDict.__setitem__(self, key, value)
            self.current_size += get_cache_size(value)
            self._evict_if_needed()
    
    def __getitem__(self, key):
        with self.lock:
            value = OrderedDict.__getitem__(self, key)
            self.move_to_end(key)
            return value

    def __contains__(self, key):
        with self.lock:
            return OrderedDict.__contains__(self, key)

    def clear(self):
        with self.lock:
            OrderedDict.clear(self)
            self.current_size = 0

    def set_limit(self, cache_limit_bytes: int):
        with self.lock:
            self.cache_limit_bytes = cache_limit_bytes
            self._evict_if_needed()

    def stats(self):
        with self.lock:
            return {
                "entries": len(self),
                "current_size": self.current_size,
                "limit": self.cache_limit_bytes,
            }
    
    def _evict_if_needed(self):
        # Evict the oldest items until current size is within the limit.
        # Cap iterations to prevent any theoretical infinite loop.
        max_evictions = len(self)
        evicted = 0
        while self.current_size > self.cache_limit_bytes and len(self) > 0 and evicted < max_evictions:
            old_key, old_value = OrderedDict.popitem(self, last=False)
            size_removed = get_cache_size(old_value)
            self.current_size = max(0, self.current_size - size_removed)
            evicted += 1


class UserScopedCache:
    def __init__(self, global_limit_bytes: int):
        self.global_limit_bytes = global_limit_bytes
        self.default_limit_bytes = global_limit_bytes
        self.scope = "global"
        self.user_limits: dict[str, int] = {}
        self.global_cache = LimitedCache(global_limit_bytes)
        self.user_caches: dict[str, LimitedCache] = {}
        self.lock = threading.RLock()

    def configure(
        self,
        scope: str = "global",
        default_limit_bytes: int | None = None,
        user_limits: dict[str, int] | None = None,
    ):
        if scope not in {"global", "per_user"}:
            raise ValueError("cache scope must be 'global' or 'per_user'")

        with self.lock:
            self.scope = scope
            self.default_limit_bytes = default_limit_bytes or self.global_limit_bytes
            self.user_limits = user_limits or {}
            self.global_cache.set_limit(self.default_limit_bytes)
            for username, cache in self.user_caches.items():
                cache.set_limit(self._limit_for_user_locked(username))

    def get(self, username: str, key: str):
        cache = self._cache_for_user(username)
        if key not in cache:
            return None
        return cache[key]

    def set(self, username: str, key: str, value: Any):
        self._cache_for_user(username)[key] = value

    def clear(self):
        with self.lock:
            self.global_cache.clear()
            for cache in self.user_caches.values():
                cache.clear()

    def stats(self):
        with self.lock:
            return {
                "scope": self.scope,
                "global": self.global_cache.stats(),
                "users": {
                    username: cache.stats()
                    for username, cache in self.user_caches.items()
                },
            }

    def total_size(self) -> int:
        with self.lock:
            return self.global_cache.current_size + sum(
                cache.current_size for cache in self.user_caches.values()
            )

    def _cache_for_user(self, username: str) -> LimitedCache:
        with self.lock:
            if self.scope == "global":
                return self.global_cache
            if username not in self.user_caches:
                self.user_caches[username] = LimitedCache(self._limit_for_user_locked(username))
            return self.user_caches[username]

    def _limit_for_user_locked(self, username: str) -> int:
        return self.user_limits.get(username, self.default_limit_bytes)

# ----- Cache Key Helpers -----

def get_rerank_cache_key(query: str, documents: list[str], top_k: int, return_documents: bool, task_description: Optional[str] = None) -> str:
    m = hashlib.sha256()
    m.update(query.encode("utf-8"))
    for doc in documents:
        m.update(doc.encode("utf-8"))
    m.update(str(top_k).encode("utf-8"))
    m.update(str(return_documents).encode("utf-8"))
    if task_description:
        m.update(task_description.encode("utf-8"))
    return m.hexdigest()

def get_embedding_cache_key(input_data: list[str], encoding_format: str = "float", quantization: str = "standard") -> str:
    m = hashlib.sha256()
    for item in input_data:
        m.update(item.encode("utf-8"))
    m.update(encoding_format.encode("utf-8"))
    m.update(quantization.encode("utf-8"))
    return m.hexdigest()
