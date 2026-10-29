import time
import threading
import hashlib
import json
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
    and otherwise walks containers, treating numeric vectors in constant time.
    Nested rerank results include their document strings in the estimate too.
    This budgets estimated Python storage, not total process RSS.
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

    # Never walk an embedding float-by-float, including non-OpenAI shapes.
    try:
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum(sys.getsizeof(k) + get_cache_size(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            if obj and isinstance(obj[0], (int, float)):
                size += max(32, sys.getsizeof(obj[0])) * len(obj)
            else:
                size += sum(get_cache_size(value) for value in obj)
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
    """Tenant-isolated LRU entries sharing one aggregate memory budget.

    A single lock covers lookup, recency, accounting and eviction. In particular,
    a concurrent insert cannot evict an entry between a contains/get pair.
    """

    def __init__(self, global_limit_bytes: int):
        self.global_limit_bytes = global_limit_bytes
        self.default_limit_bytes = global_limit_bytes
        self.scope = "global"
        self.user_limits: dict[str, int] = {}
        self.entries = OrderedDict()
        self.recency: dict[str | None, OrderedDict] = {}
        self.sizes: dict[str | None, int] = {}
        self.current_size = 0
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
            if scope != self.scope:
                self.clear()
            self.scope = scope
            self.default_limit_bytes = (
                self.global_limit_bytes if default_limit_bytes is None else default_limit_bytes
            )
            self.user_limits = dict(user_limits or {})
            self._evict_locked()

    def get(self, username: str, key: str):
        with self.lock:
            identity = (None if self.scope == "global" else username, key)
            entry = self.entries.get(identity)
            if entry is None:
                return None
            self.entries.move_to_end(identity)
            self.recency[identity[0]].move_to_end(identity)
            return entry[0]

    def set(self, username: str, key: str, value: Any):
        size = get_cache_size(value)
        with self.lock:
            namespace = None if self.scope == "global" else username
            identity = (namespace, key)
            if identity in self.entries:
                self._remove_locked(identity)
            # An oversized result should not flush everyone else's useful entries.
            if size > min(self.global_limit_bytes, self._limit_locked(namespace)):
                return
            self.entries[identity] = (value, size)
            self.recency.setdefault(namespace, OrderedDict())[identity] = None
            self.sizes[namespace] = self.sizes.get(namespace, 0) + size
            self.current_size += size
            self._evict_locked(namespace)

    def _limit_locked(self, namespace):
        return self.user_limits.get(namespace, self.default_limit_bytes)

    def _remove_locked(self, identity):
        _, size = self.entries.pop(identity)
        namespace = identity[0]
        del self.recency[namespace][identity]
        self.current_size -= size
        self.sizes[namespace] -= size
        if self.sizes[namespace] == 0:
            del self.sizes[namespace]
            del self.recency[namespace]

    def _evict_locked(self, namespace=None):
        # First enforce tenant shares; then apply LRU across all remaining entries.
        namespaces = list(self.sizes) if namespace is None else [namespace]
        for user in namespaces:
            while self.sizes.get(user, 0) > self._limit_locked(user):
                self._remove_locked(next(iter(self.recency[user])))
        while self.current_size > self.global_limit_bytes and self.entries:
            self._remove_locked(next(iter(self.entries)))

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.recency.clear()
            self.sizes.clear()
            self.current_size = 0

    def stats(self):
        with self.lock:
            counts = {}
            for namespace, _ in self.entries:
                counts[namespace] = counts.get(namespace, 0) + 1

            def namespace_stats(namespace):
                return {
                    "entries": counts.get(namespace, 0),
                    "current_size": self.sizes.get(namespace, 0),
                    "limit": self._limit_locked(namespace),
                }

            return {
                "scope": self.scope,
                "aggregate_limit": self.global_limit_bytes,
                "current_size": self.current_size,
                "global": namespace_stats(None),
                "users": {
                    user: namespace_stats(user) for user in counts if user is not None
                },
            }

    def total_size(self) -> int:
        with self.lock:
            return self.current_size

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

def get_embedding_cache_key(
    input_data: list[Any],
    encoding_format: str = "float",
    quantization: str = "standard",
    input_type: str = "document",
) -> str:
    m = hashlib.sha256()
    # Preserve item boundaries and media bodies: ["ab", "c"] differs from ["a", "bc"].
    m.update(json.dumps(input_data, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    m.update(encoding_format.encode("utf-8"))
    m.update(quantization.encode("utf-8"))
    m.update(input_type.encode("utf-8"))
    return m.hexdigest()
