import os
import json
import secrets
import hashlib
import sys
import psutil
import torch
import sys
import argparse
import uvicorn
import logging
import multiprocessing
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

from transformers import pipeline
from collections import OrderedDict
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from mxbai_rerank import MxbaiRerankV2
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings


# ----- Enhanced Logging Configuration -----show_progress_bar=False
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log output"""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

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

# Suppress tqdm progress bars globally
import tqdm
import time

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
    
    # Remove the problematic __globals__ patching that causes attribute errors

# Suppress progress bars on module import
suppress_tqdm_progress()

def setup_pretty_logging(level=logging.INFO):
    """Setup pretty colored logging with uvicorn handler management"""
    # Remove uvicorn handlers completely to prevent duplicate logging
    logging.getLogger("uvicorn.access").handlers.clear() 
    logging.getLogger("uvicorn.access").propagate = False
    logging.getLogger("uvicorn").handlers.clear()
    logging.getLogger("uvicorn").propagate = False
    logging.getLogger("uvicorn.error").handlers.clear()
    logging.getLogger("uvicorn.error").propagate = False
    
    # Configure root logger
    logging.root.propagate = False
    
    # Clear any existing handlers on root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create console handler with custom formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    logging.root.addHandler(console_handler)
    logging.root.setLevel(level)

# Setup pretty logging
setup_pretty_logging()
logger = logging.getLogger("BananaBread-Emb")

# ----- CPU and System Information -----

def get_cpu_info():
    """Get detailed CPU information including socket and core topology"""
    try:
        # Try to get CPU info from lscpu on Linux
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        if result.returncode == 0:
            cpu_info = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    cpu_info[key.strip()] = value.strip()
            return cpu_info
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Fallback to basic info
    return {
        'CPU(s)': str(multiprocessing.cpu_count()),
        'Model name': 'Unknown',
        'Socket(s)': '1'
    }

def get_available_cores():
    """Get list of available CPU cores with their socket information"""
    try:
        # Try to read from /proc/cpuinfo on Linux
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        
        cores = []
        current_core = {}
        for line in cpuinfo.split('\n'):
            if line.strip() == '':
                if current_core and 'processor' in current_core:
                    cores.append(current_core)
                current_core = {}
            elif ':' in line:
                key, value = line.split(':', 1)
                current_core[key.strip()] = value.strip()
        
        # Group cores by physical id (socket)
        socket_cores = {}
        for core in cores:
            if 'physical id' in core and 'processor' in core:
                socket_id = core['physical id']
                if socket_id not in socket_cores:
                    socket_cores[socket_id] = []
                socket_cores[socket_id].append(int(core['processor']))
        
        return socket_cores
    except (FileNotFoundError, KeyError):
        # Fallback: assume all cores are on socket 0
        return {'0': list(range(multiprocessing.cpu_count()))}

# ----- CPU and Threadpool Optimization Configuration -----

# Get system information
CPU_INFO = get_cpu_info()
AVAILABLE_SOCKETS = get_available_cores()
TOTAL_PHYSICAL_CORES = multiprocessing.cpu_count()

logger.info("BananaBread-Emb CPU Optimization Starting...")
logger.info(f"System Info: {CPU_INFO.get('Model name', 'Unknown CPU')}")
logger.info(f"Total Physical Cores: {TOTAL_PHYSICAL_CORES}")
logger.info(f"Available Sockets: {len(AVAILABLE_SOCKETS)}")
for socket_id, cores in AVAILABLE_SOCKETS.items():
    logger.info(f"  - Socket {socket_id}: Cores {min(cores)}-{max(cores)} ({len(cores)} cores)")

# ----- Enhanced Argument Parsing -----

parser = argparse.ArgumentParser(
    description="BananaBread-Emb - Optimized MixedBread AI Server",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Use all CPU cores (default)
  python server.py
  
  # Use only 8 cores
  python server.py --use-cores 8
  
  # Pin to specific socket (0 or 1)
  python server.py --cpu-socket 1
  
  # Custom thread configuration
  python server.py --use-cores 16 --embedding-threads 16 --rerank-threads 16 --cache-limit 4096
  
  # Load models on GPU (if available)
  python server.py --embedding-device cuda --rerank-device cuda
  
  # Load models on different GPUs
  python server.py --embedding-device cuda:0 --rerank-device cuda:1
  
  # Load embedding model on GPU, rerank model on CPU
  python server.py --embedding-device cuda --rerank-device cpu
  
  # Use Apple Silicon GPU (MPS)
  python server.py --embedding-device mps --rerank-device mps
    """
)

# CPU and core selection arguments
parser.add_argument("--use-cores", type=int, default=None, 
                   help=f"Number of physical CPU cores to use (default: all {TOTAL_PHYSICAL_CORES} cores)")
parser.add_argument("--cpu-socket", type=int, choices=list(range(len(AVAILABLE_SOCKETS))), default=None,
                   help="Pin operations to specific CPU socket (for multi-socket systems)")

# Existing threadpool arguments
parser.add_argument("--cache-limit", type=int, default=1024, help="Cache limit in MB")
parser.add_argument("--embedding-threads", type=int, default=None, 
                   help="Number of threads for embedding operations (default: auto-detected)")
parser.add_argument("--rerank-threads", type=int, default=None,
                   help="Number of threads for reranking operations (default: auto-detected)")
parser.add_argument("--classification-threads", type=int, default=None,
                   help="Number of threads for classification operations (default: auto-detected)")
parser.add_argument("--general-threads", type=int, default=None,
                   help="Number of threads for general operations (default: auto-detected)")

# Device selection arguments
parser.add_argument("--embedding-device", type=str, default="cpu",
                   help="Device to load embedding model on (cpu, cuda, cuda:0, cuda:1, mps, etc.)")
parser.add_argument("--rerank-device", type=str, default="cpu",
                   help="Device to load rerank model on (cpu, cuda, cuda:0, cuda:1, mps, etc.)")

# Logging arguments
parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                   help="Set logging level (default: INFO)")

args, remaining_args = parser.parse_known_args()

# Setup logging level
setup_pretty_logging(getattr(logging, args.log_level))

# ----- CPU Core Selection Logic -----

def select_cpu_cores():
    """Select which CPU cores to use based on arguments"""
    available_cores = list(range(TOTAL_PHYSICAL_CORES))
    
    if args.cpu_socket is not None:
        # Use cores from specific socket
        socket_cores = AVAILABLE_SOCKETS.get(str(args.cpu_socket), available_cores)
        available_cores = socket_cores
        logger.info(f"Pinning to Socket {args.cpu_socket}: Cores {min(socket_cores)}-{max(socket_cores)}")
    
    if args.use_cores is not None:
        # Limit to specified number of cores
        if args.use_cores > len(available_cores):
            logger.warning(f"⚠️  Requested {args.use_cores} cores but only {len(available_cores)} available")
            selected_cores = available_cores
        else:
            selected_cores = available_cores[:args.use_cores]
        logger.info(f"Using {len(selected_cores)} cores: {min(selected_cores)}-{max(selected_cores)}")
    else:
        selected_cores = available_cores
        logger.info(f"Using all {len(selected_cores)} cores: {min(selected_cores)}-{max(selected_cores)}")
    
    return selected_cores

SELECTED_CORES = select_cpu_cores()
CPU_COUNT = len(SELECTED_CORES)

# Calculate optimal thread counts based on selected cores
EMBEDDING_THREADS = args.embedding_threads or CPU_COUNT
RERANK_THREADS = args.rerank_threads or CPU_COUNT
CLASSIFICATION_THREADS = args.classification_threads or max(1, CPU_COUNT // 2)
GENERAL_THREADS = args.general_threads or CPU_COUNT * 2

logger.info("Threadpool Configuration:")
logger.info(f"  - Embedding threads: {EMBEDDING_THREADS}")
logger.info(f"  - Rerank threads: {RERANK_THREADS}")
logger.info(f"  - Classification threads: {CLASSIFICATION_THREADS}")
logger.info(f"  - General threads: {GENERAL_THREADS}")

# Create optimized threadpool executors with selected cores
embedding_executor = ThreadPoolExecutor(max_workers=EMBEDDING_THREADS, thread_name_prefix="embedding")
rerank_executor = ThreadPoolExecutor(max_workers=RERANK_THREADS, thread_name_prefix="rerank")
classification_executor = ThreadPoolExecutor(max_workers=CLASSIFICATION_THREADS, thread_name_prefix="classification")
general_executor = ThreadPoolExecutor(max_workers=GENERAL_THREADS, thread_name_prefix="general")

# ----- API Key Authentication Setup -----

API_KEYS_FILE = "./api_keys.json"
api_keys = {}

def load_api_keys():
    global api_keys
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as f:
            api_keys = json.load(f)
    else:
        api_keys = {}
    updated = False
    # For each user in the file, if no API key is provided, generate one.
    for user, key in api_keys.items():
        if not key:
            new_key = secrets.token_hex(16)
            api_keys[user] = new_key
            updated = True
    if updated:
        with open(API_KEYS_FILE, "w") as f:
            json.dump(api_keys, f)

load_api_keys()

def get_api_key(authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    api_key = authorization.split("Bearer ")[-1]
    if api_key not in api_keys.values():
        raise HTTPException(status_code=401, detail="Unauthorized")
    return api_key

# ----- Initialize FastAPI and Models -----

app = FastAPI(
    title="BananaBread-Emb", # Set your desired project name here
    description="A way to slip MixedBread's reranker and embedder into a lot of places it doesn't belong.",
    version="0.5.2"
)

# Store threadpool configuration in app state for access in endpoints
app.state.cpu_count = CPU_COUNT
app.state.selected_cores = SELECTED_CORES
app.state.total_physical_cores = TOTAL_PHYSICAL_CORES
app.state.cpu_socket = args.cpu_socket
app.state.embedding_executor = embedding_executor
app.state.rerank_executor = rerank_executor
app.state.classification_executor = classification_executor
app.state.general_executor = general_executor

# Reranking model initialization with device specification
logger.info("Initializing models...")
logger.info(f"Loading rerank model on device: {args.rerank_device}")
rerank_model = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-base-v2", device=args.rerank_device)

# Embedding model initialization with truncation to 1024 dimensions and device specification
logger.info(f"Loading embedding model on device: {args.embedding_device}")
embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=1024, device=args.embedding_device)
logger.info("Models initialized successfully")

def get_cache_size(obj):
    """
    Recursively estimate the size in bytes of a Python object.
    This is a rough approximation.
    """
    seen_ids = set()
    def inner(obj):
        if id(obj) in seen_ids:
            return 0
        seen_ids.add(id(obj))
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum(inner(k) + inner(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(inner(i) for i in obj)
        return size
    return inner(obj)

class LimitedCache(OrderedDict):
    def __init__(self, cache_limit_bytes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_limit_bytes = cache_limit_bytes
        self.current_size = 0
    
    def __setitem__(self, key, value):
        # If key already exists, remove its previous size.
        if key in self:
            self.current_size -= get_cache_size(self[key])
        super().__setitem__(key, value)
        self.current_size += get_cache_size(value)
        self._evict_if_needed()
    
    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value
    
    def _evict_if_needed(self):
        # Evict the oldest items until current size is within the limit.
        while self.current_size > self.cache_limit_bytes and len(self) > 0:
            old_key, old_value = self.popitem(last=False)
            self.current_size -= get_cache_size(old_value)

# --- Load the classification pipeline globally ---

classifier = pipeline(
    'text-classification',
    model='SamLowe/roberta-base-go_emotions',
    top_k=None
)

# ----- Logging Setup and Argument Parsing -----

cache_limit_bytes = args.cache_limit * 1024 * 1024

logger.info(f"💾 Using cache limit: {args.cache_limit} MB ({cache_limit_bytes} bytes)")

# Update thread counts based on command line arguments if provided
if args.embedding_threads and args.embedding_threads != EMBEDDING_THREADS:
    embedding_executor = ThreadPoolExecutor(max_workers=args.embedding_threads, thread_name_prefix="embedding")
    logger.info(f"Updated embedding threads to {args.embedding_threads}")

if args.rerank_threads and args.rerank_threads != RERANK_THREADS:
    rerank_executor = ThreadPoolExecutor(max_workers=args.rerank_threads, thread_name_prefix="rerank")
    logger.info(f"Updated rerank threads to {args.rerank_threads}")

if args.classification_threads and args.classification_threads != CLASSIFICATION_THREADS:
    classification_executor = ThreadPoolExecutor(max_workers=args.classification_threads, thread_name_prefix="classification")
    logger.info(f"Updated classification threads to {args.classification_threads}")

if args.general_threads and args.general_threads != GENERAL_THREADS:
    general_executor = ThreadPoolExecutor(max_workers=args.general_threads, thread_name_prefix="general")
    logger.info(f"Updated general threads to {args.general_threads}")

# ----- Initialize Caches with LimitedCache -----

# Replace previous dictionary caches with instances of LimitedCache.
rerank_cache = LimitedCache(cache_limit_bytes)
embedding_cache = LimitedCache(cache_limit_bytes)

# ----- Request Schemas -----

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = 3
    return_documents: bool = False

# Embedding request schema matching OpenAI's format:
class EmbeddingRequest(BaseModel):
    model: str = "mixedbread-ai/mxbai-embed-large-v1"
    input: list[str]

class ClassificationRequest(BaseModel):
    input: str
    top_k: int | None = None   # Optional; if provided, only the top K results will be returned.
    sorted: bool = False       # If True, sort results from highest to lowest score.

# ----- Ollama Compatible Schemas -----

class OllamaMessage(BaseModel):
    role: str
    content: str

class OllamaRequest(BaseModel):
    model: str
    messages: list[OllamaMessage]
    stream: bool = False
    options: dict = {}
    format: str = None
    keep_alive: str = None

class OllamaResponse(BaseModel):
    model: str = "mixedbread-ai/mxbai-embed-large-v1"
    created_at: str = ""
    message: OllamaMessage
    done_reason: str = "stop"
    done: bool = True
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0

# ----- Llama.cpp Compatible Schemas -----

class LlamaCppEmbeddingRequest(BaseModel):
    content: str
    model: str = "mixedbread-ai/mxbai-embed-large-v1"
    normalize: bool = True
    truncate: bool = True

class LlamaCppEmbeddingResponse(BaseModel):
    embedding: list[float]
    model: str

# ----- Utility Functions for Caching -----

def get_rerank_cache_key(query: str, documents: list[str], top_k: int, return_documents: bool) -> str:
    m = hashlib.sha256()
    m.update(query.encode("utf-8"))
    for doc in documents:
        m.update(doc.encode("utf-8"))
    m.update(str(top_k).encode("utf-8"))
    m.update(str(return_documents).encode("utf-8"))
    return m.hexdigest()

def get_embedding_cache_key(input_data: list[str]) -> str:
    m = hashlib.sha256()
    for item in input_data:
        m.update(item.encode("utf-8"))
    return m.hexdigest()

# ----- Memory Profiling Helper Functions -----

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
        for param in model.parameters():
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
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args, **kwargs)

# ----- API Endpoints -----

@app.post("/v1/rerank")
async def rerank_endpoint(request: RerankRequest, api_key: str = Depends(get_api_key)):
    if not request.query or not request.documents:
        raise HTTPException(status_code=400, detail="Query or documents must be provided")
    
    key = get_rerank_cache_key(request.query, request.documents, request.top_k, request.return_documents)
    if key in rerank_cache:
        return rerank_cache[key]
    
    # Use dedicated rerank threadpool for optimal CPU utilization
    result = await run_in_threadpool_with_executor(
        rerank_executor, 
        rerank_model.rank,
        request.query,
        request.documents,
        return_documents=request.return_documents,
        top_k=request.top_k
    )
    rerank_cache[key] = result
    return result

@app.post("/v1/embeddings")
async def embedding_endpoint(request: EmbeddingRequest, api_key: str = Depends(get_api_key)):
    inputs = request.input if isinstance(request.input, list) else [request.input]
    if not inputs:
        raise HTTPException(status_code=400, detail="Input must be provided")
    
    key = get_embedding_cache_key(inputs) if inputs else ""
    if key in embedding_cache:
        return embedding_cache[key]
    
    # Use dedicated embedding threadpool for optimal CPU utilization
    # Add custom progress tracking for large batches
    if len(inputs) > 10:  # Only track progress for batches larger than 10 items
        progress_tracker = CustomProgressTracker(len(inputs), "Embedding")
        progress_tracker.start()
        
        # Process in smaller chunks to show progress
        chunk_size = max(1, len(inputs) // 10)  # Process in 10% chunks
        all_embeddings = []
        
        for i in range(0, len(inputs), chunk_size):
            chunk = inputs[i:i + chunk_size]
            chunk_embeddings = await run_in_threadpool_with_executor(
                embedding_executor,
                embedding_model.encode,
                chunk
            )
            all_embeddings.append(chunk_embeddings)
            progress_tracker.update(min(i + chunk_size, len(inputs)))
        
        # Combine all chunks
        import numpy as np
        docs_embeddings = np.concatenate(all_embeddings, axis=0)
        progress_tracker.finish()
    else:
        # For small batches, process normally
        docs_embeddings = await run_in_threadpool_with_executor(
            embedding_executor,
            embedding_model.encode,
            inputs
        )
    
    # Create wrapper function for quantize_embeddings to handle keyword arguments
    def quantize_embeddings_wrapper(embeddings):
        return quantize_embeddings(embeddings, precision="ubinary")
    
    binary_docs_embeddings = await run_in_threadpool_with_executor(
        embedding_executor,
        quantize_embeddings_wrapper,
        docs_embeddings
    )
    
    # Format response to mimic OpenAI's embeddings API.
    data = []
    for idx, emb in enumerate(binary_docs_embeddings.tolist()):
        data.append({
            "object": "embedding",
            "embedding": emb,
            "index": idx
        })
    
    # Dummy usage info – OpenAI normally returns token counts.
    usage = {"prompt_tokens": 0, "total_tokens": 0}
    
    result = {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": usage
    }
    embedding_cache[key] = result
    return result

@app.post("/v1/classify")
async def classify_endpoint(request: ClassificationRequest, api_key: str = Depends(get_api_key)):
    # Use dedicated classification threadpool for optimal CPU utilization
    raw_result = await run_in_threadpool_with_executor(
        classification_executor,
        classifier,
        request.input
    )
    
    # The pipeline returns a list of results for each input.
    # For a single input, we might get a list of one element (which is itself a list of dicts).
    if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], list):
        results = raw_result[0]
    else:
        results = raw_result
    
    # Normalize: limit precision of the score values to 9 digits.
    normalized_results = [
        {"label": item["label"], "score": round(item["score"], 9)}
        for item in results
    ]
    
    # Sort the list if requested (from highest to lowest score)
    if request.sorted:
        normalized_results = sorted(normalized_results, key=lambda x: x["score"], reverse=True)
    
    # Apply top_k slicing if specified.
    if request.top_k is not None:
        normalized_results = normalized_results[:request.top_k]
    
    return {"result": normalized_results}

# ----- Ollama Compatible Embedding Endpoint -----

class OllamaEmbeddingRequest(BaseModel):
    model: str
    prompt: Optional[str] = None
    input: Optional[str | list[str]] = None
    truncate: bool = True
    options: dict = {}
    keep_alive: str = "5m"

class OllamaEmbeddingResponse(BaseModel):
    embedding: list[float]
    model: str

@app.post("/api/embeddings")
async def ollama_embeddings_endpoint(request: OllamaEmbeddingRequest):
    """
    Ollama-compatible endpoint for embeddings.
    This endpoint does not require API key authentication.
    Follows Ollama API specification for embeddings.
    """
    # Determine input source (either 'prompt' or 'input' field)
    if request.prompt:
        inputs = [request.prompt]
    elif request.input:
        if isinstance(request.input, str):
            inputs = [request.input]
        elif isinstance(request.input, list):
            inputs = request.input
        else:
            raise HTTPException(status_code=400, detail="Input must be a string or list of strings")
    else:
        raise HTTPException(status_code=400, detail="Either 'prompt' or 'input' must be provided")
    
    if not inputs:
        raise HTTPException(status_code=400, detail="Input must be provided")
    
    # Generate embeddings using the existing embedding model
    # Use dedicated embedding threadpool for optimal performance
    docs_embeddings = await run_in_threadpool_with_executor(
        embedding_executor,
        embedding_model.encode,
        inputs
    )
    
    # For single input, return single embedding array
    # For multiple inputs, Ollama typically returns the first embedding
    # (this matches Ollama's behavior where embeddings are generated one at a time)
    if len(inputs) == 1:
        embedding = docs_embeddings[0].tolist()
    else:
        # For multiple inputs, return the first embedding (Ollama convention)
        embedding = docs_embeddings[0].tolist()
    
    # Ensure embedding is a list[float] for type safety
    if not isinstance(embedding, list):
        embedding = [float(embedding)] if isinstance(embedding, (int, float)) else []
    else:
        # Convert all elements to float
        embedding = [float(x) for x in embedding]
    
    # Format response according to Ollama API specification
    response = OllamaEmbeddingResponse(
        embedding=embedding,
        model=request.model
    )
    
    return response

@app.post("/api/embed")
async def ollama_embed_endpoint(request: OllamaEmbeddingRequest):
    """
    Alias for /api/embeddings - some Ollama clients use /api/embed
    This endpoint does not require API key authentication.
    """
    return await ollama_embeddings_endpoint(request)

# ----- Llama.cpp Compatible Embedding Endpoint -----

@app.post("/embedding")
async def llamacpp_embedding_endpoint(request: LlamaCppEmbeddingRequest):
    """
    Llama.cpp-compatible endpoint for embeddings.
    This endpoint does not require API key authentication.
    Follows llama.cpp server API specification for embeddings.
    """
    if not request.content:
        raise HTTPException(status_code=400, detail="Content must be provided")
    
    # Generate embedding using the existing embedding model
    # Use dedicated embedding threadpool for optimal performance
    inputs = [request.content]
    docs_embeddings = await run_in_threadpool_with_executor(
        embedding_executor,
        embedding_model.encode,
        inputs
    )
    
    # Get the embedding and convert to list
    embedding = docs_embeddings[0].tolist()
    
    # Ensure embedding is always a proper list[float] for type safety
    # Handle nested lists by flattening if necessary
    def flatten_to_float_list(obj):
        """Recursively flatten nested lists and convert to float list"""
        if isinstance(obj, (int, float)):
            return [float(obj)]
        elif isinstance(obj, list):
            result = []
            for item in obj:
                result.extend(flatten_to_float_list(item))
            return result
        else:
            return [0.0]  # Fallback for unsupported types
    
    # Convert embedding to proper list[float] format
    embedding = flatten_to_float_list(embedding)
    
    # Apply normalization if requested (llama.cpp typically normalizes by default)
    if request.normalize:
        # Normalize the embedding to unit length
        import numpy as np
        embedding_array = np.array(embedding, dtype=np.float64)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm
            embedding = embedding_array.tolist()
            # Ensure the result is always a list[float] after normalization
            if isinstance(embedding, list):
                embedding = [float(x) if isinstance(x, (int, float)) else 0.0 for x in embedding]
            elif isinstance(embedding, (int, float)):
                embedding = [float(embedding)]
            else:
                embedding = [0.0]
    
    # Format response according to llama.cpp API specification
    response = LlamaCppEmbeddingResponse(
        embedding=embedding,
        model=request.model
    )
    
    return response

@app.post("/v1/embeddings/llamacpp")
async def llamacpp_embedding_endpoint_v1(request: LlamaCppEmbeddingRequest):
    """
    Alternative llama.cpp-compatible endpoint with v1 prefix.
    This endpoint does not require API key authentication.
    """
    return await llamacpp_embedding_endpoint(request)

@app.get("/v1/memory")
async def memory_usage(api_key: str = Depends(get_api_key)):
    """
    Returns a JSON payload with memory usage statistics:
      - Process memory usage (RSS)
      - Estimated memory used by the rerank and embedding models
      - Estimated size of the embedding cache
      - GPU memory allocated (if CUDA is available)
      - CPU and threadpool configuration
    """
    data = {}
    data["process_memory"] = get_process_memory_usage()
    # Try to get memory usage from each model. For rerank_model, attempt to use its underlying model if available.
    if hasattr(rerank_model, "model"):
        data["rerank_model_memory"] = get_model_memory_usage(rerank_model.model)
    else:
        data["rerank_model_memory"] = get_model_memory_usage(rerank_model)
    data["embedding_model_memory"] = get_model_memory_usage(embedding_model)
    data["embedding_cache_memory"] = get_cache_size(embedding_cache)
    
    # Add CPU and threadpool information
    data["cpu_cores"] = CPU_COUNT
    data["total_physical_cores"] = TOTAL_PHYSICAL_CORES
    data["selected_cores"] = SELECTED_CORES
    data["cpu_socket"] = args.cpu_socket
    data["embedding_threads"] = EMBEDDING_THREADS
    data["rerank_threads"] = RERANK_THREADS
    data["classification_threads"] = CLASSIFICATION_THREADS
    data["general_threads"] = GENERAL_THREADS
    
    if torch.cuda.is_available():
        # Report GPU memory allocated on device 0.
        data["gpu_memory_allocated"] = torch.cuda.memory_allocated(0)
    
    return data

@app.get("/v1/health")
async def health(api_key: str = Depends(get_api_key)):
    return {"status": "healthy", "cpu_cores": CPU_COUNT, "optimized": True}

@app.get("/v1/models")
async def model(api_key: str = Depends(get_api_key)):
    return {"object": "list",
            "data": [
                {
                    "id": "mixedbread-large-v1",
                    "object": "model",
                    "created": "176257272",
                    "owned_by": "urmom"
                }
            ]}

@app.get("/", name="BananaBread-Emb Works")
async def read_root():
    return {
        "message": "🍞 BananaBread-Emb is running with optimized CPU utilization!",
        "cpu_cores": CPU_COUNT,
        "total_physical_cores": TOTAL_PHYSICAL_CORES,
        "optimized_threadpools": True,
        "cpu_socket": args.cpu_socket,
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "rerank": "/v1/rerank", 
            "classify": "/v1/classify",
            "memory": "/v1/memory",
            "health": "/v1/health",
            "models": "/v1/models"
        }
    }

# ----- Cleanup and Resource Management -----

import atexit
import signal
import threading

# Global shutdown flag
shutdown_event = threading.Event()

def cleanup_threadpools():
    """Clean up threadpool executors on shutdown"""
    try:
        logger.info("Shutting down threadpool executors...")
        embedding_executor.shutdown(wait=True, cancel_futures=True)
        rerank_executor.shutdown(wait=True, cancel_futures=True)
        classification_executor.shutdown(wait=True, cancel_futures=True)
        general_executor.shutdown(wait=True, cancel_futures=True)
        logger.info("Threadpool executors shut down successfully")
    except Exception as e:
        logger.error(f"Error during threadpool cleanup: {e}")

def cleanup_resources():
    """Comprehensive cleanup of all resources"""
    if not shutdown_event.is_set():
        shutdown_event.set()
        logger.info("Initiating graceful shutdown...")
        
        # Clean up threadpools
        cleanup_threadpools()
        
        # Clean up models and caches
        try:
            logger.info("Cleaning up model resources...")
            global rerank_cache, embedding_cache
            rerank_cache.clear()
            embedding_cache.clear()
            logger.info("Model resources cleaned up")
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
        
        logger.info("Graceful shutdown completed")
        
        # Force process termination after cleanup
        import os
        os._exit(0)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully without interrupting the event loop"""
    signal_name = signal.Signals(signum).name
    logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")
    
    # Set shutdown flag instead of calling sys.exit directly
    if 'server' in globals():
        server.should_exit = True
    cleanup_resources()

# Register signal handlers for graceful shutdown - we'll override these in main
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

# Register cleanup function for normal exit
atexit.register(cleanup_resources)

# ----- Custom Uvicorn Server Launch Configuration -----

class CustomServer(uvicorn.Server):
    """Custom uvicorn server to control startup logging timing and graceful shutdown"""
    
    def __init__(self, config):
        super().__init__(config)
        self._shutdown_requested = False
    
    async def startup(self, sockets=None):
        """Override startup to add our custom logging at the right time"""
        # Call the original startup
        await super().startup(sockets)
        
        # Now that the server is actually starting up, log our messages
        logger.info("BananaBread-Emb server is now running!")
        logger.info(f"Server available at: http://{self.config.host}:{self.config.port}")
        logger.info(f"API Documentation: http://{self.config.host}:{self.config.port}/docs")
        logger.info(f"ReDoc Documentation: http://{self.config.host}:{self.config.port}/redoc")
    
    async def shutdown(self, sockets=None):
        """Override shutdown to ensure graceful cleanup"""
        if not self._shutdown_requested:
            self._shutdown_requested = True
            logger.info("Server shutdown initiated...")
            
            # Call our custom cleanup
            cleanup_resources()
            
            # Call the original shutdown
            await super().shutdown(sockets)
            
            logger.info("Server shutdown completed")
    
    def handle_exit(self, sig, frame):
        """Handle exit signals more gracefully"""
        logger.info(f"Received exit signal {sig}, shutting down server...")
        self.should_exit = True

def main():
    """Main entry point for the bananabread-emb console script"""
    # Create uvicorn configuration
    config = uvicorn.Config(
        "server:app",
        host="0.0.0.0",
        port=8008,
        reload=False,
        log_config=None  # Disable uvicorn's default logging
    )
    
    # Create custom server
    server = CustomServer(config)
    
    # Set up signal handling for the server
    def server_signal_handler(signum, frame):
        """Handle signals for server shutdown"""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name} signal, initiating server shutdown...")
        
        # Set the shutdown flag to trigger uvicorn's graceful shutdown
        server.should_exit = True
        
        # Call our comprehensive cleanup
        cleanup_resources()
    
    # Override the signal handlers to use our server-specific handler
    signal.signal(signal.SIGINT, server_signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, server_signal_handler)  # Termination signal
    
    logger.info("Initializing BananaBread-Emb server...")
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down gracefully...")
        cleanup_resources()
    except Exception as e:
        logger.error(f"Unexpected error during server execution: {e}")
        cleanup_resources()
        raise
    finally:
        # Ensure cleanup always runs
        cleanup_resources()

if __name__ == "__main__":
    main()
