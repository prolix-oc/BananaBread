import os

# Set PyTorch CUDA memory optimization (before torch import)
# Reduces VRAM fragmentation and improves memory efficiency
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

import json
import secrets
import hashlib
import sys
import psutil
import torch
import argparse
import uvicorn
import logging
import multiprocessing
import subprocess
import threading
import atexit
import signal
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

from transformers import pipeline, AutoTokenizer, AutoModel
from collections import OrderedDict
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from mxbai_rerank import MxbaiRerankV2
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
import torch.nn.functional as F


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

# ----- Qwen Reranking Helper Functions -----

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last token from hidden states based on attention mask"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction for Qwen reranking"""
    return f'Instruct: {task_description}\nQuery: {query}'

# ----- Qwen Reranker Class -----

class QwenReranker:
    """Qwen-based reranking using embedding similarity"""
    
    def __init__(self, model_name: str, device: str = "cpu", use_flash_attention: bool = False, 
                 rerank_instruction: str = "Given the following storytelling documents, which of these best add to the user's message?"):
        """
        Initialize Qwen reranker
        
        Args:
            model_name: Name of the Qwen model to use
            device: Device to load model on (cpu, cuda, etc.)
            use_flash_attention: Whether to use flash_attention_2
            rerank_instruction: Instruction template for reranking tasks
        """
        self.device = device
        self.rerank_instruction = rerank_instruction
        self.max_length = 8192
        
        logger.info(f"Loading Qwen reranker model: {model_name}")
        
        # Initialize tokenizer with left padding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # Initialize model
        if use_flash_attention:
            logger.info("Enabling flash_attention_2 for Qwen reranker")
            self.model = AutoModel.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16
            ).to(device)
        else:
            self.model = AutoModel.from_pretrained(model_name).to(device)
        
        self.model.eval()
        logger.info(f"Qwen reranker initialized on {device}")
    
    def rank(self, query: str, documents: list[str], return_documents: bool = False, top_k: int = None) -> dict:
        """
        Rank documents based on query using embedding similarity
        
        Args:
            query: Query string
            documents: List of document strings to rank
            return_documents: Whether to include document text in results
            top_k: Number of top results to return (None = all)
        
        Returns:
            Dictionary with ranked results
        """
        # Format query with instruction
        formatted_query = get_detailed_instruct(self.rerank_instruction, query)
        
        # Combine query and documents for batch processing
        input_texts = [formatted_query] + documents
        
        # Tokenize - using inference_mode() for better memory efficiency
        with torch.inference_mode():
            batch_dict = self.tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
            
            # Get embeddings
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Calculate scores (query vs documents)
            query_embedding = embeddings[0:1]  # First embedding is the query
            doc_embeddings = embeddings[1:]     # Rest are documents
            scores = (query_embedding @ doc_embeddings.T).squeeze(0)
            
            # Convert to list and create results
            scores_list = scores.cpu().tolist()
        
        # Clear CUDA cache to reduce fragmentation if using GPU
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create results list with scores and indices
        results = []
        for idx, score in enumerate(scores_list):
            result = {
                "index": idx,
                "score": float(score)
            }
            if return_documents:
                result["document"] = documents[idx]
            results.append(result)
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply top_k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return {"results": results}

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
  
  # Use ubinary quantization for embeddings
  python server.py --quant ubinary
  
  # Use int8 quantization for embeddings
  python server.py --quant int8
  
  # Use custom embedding dimensions (512 instead of default 1024)
  python server.py --embedding-dim 512
  
  # Use maximum embedding dimensions (no truncation)
  python server.py --embedding-dim 2048
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

# Quantization arguments
parser.add_argument("--quant", type=str, choices=['standard', 'ubinary', 'int8'], default='standard',
                   help="Quantization precision for embeddings (default: standard)")

# Embedding dimensions argument
parser.add_argument("--embedding-dim", type=int, default=1024,
                   help="Embedding dimensions to truncate to (default: 1024)")

# Embedding logging argument
parser.add_argument("--log-embeddings", action='store_true',
                   help="Enable logging of embedding queries and results to embeddings.log")

# CUDA cache management arguments
parser.add_argument("--cuda-cache-ttl", type=int, default=300,
                   help="Time in seconds before clearing CUDA cache when idle (default: 300)")
parser.add_argument("--cuda-cache-ttl-enabled", action='store_true',
                   help="Enable automatic CUDA cache clearing based on TTL")
parser.add_argument("--cuda-min-clear-interval", type=int, default=60,
                   help="Minimum seconds between CUDA cache clears (default: 60)")
parser.add_argument("--cuda-memory-threshold", type=int, default=80,
                   help="Only clear CUDA cache if memory usage exceeds this percentage (default: 80)")

# Model concurrency arguments
parser.add_argument("--num-concurrent-embedding", type=int, default=1,
                   help="Number of concurrent embedding model instances to load on GPU (default: 1)")
parser.add_argument("--num-concurrent-rerank", type=int, default=1,
                   help="Number of concurrent reranking model instances to load on GPU (default: 1)")

# Torch compilation arguments
parser.add_argument("--enable-torch-compile", action='store_true',
                   help="Enable torch.compile() for models (requires PyTorch 2.0+)")
parser.add_argument("--torch-compile-mode", type=str, 
                   choices=['default', 'reduce-overhead', 'max-autotune'], 
                   default='default',
                   help="Torch compilation mode (default: default)")
parser.add_argument("--torch-compile-backend", type=str, default='inductor',
                   help="Torch compilation backend (default: inductor)")

# Model warmup arguments
parser.add_argument("--enable-warmup", action='store_true', default=True,
                   help="Enable model warmup on startup (default: True)")
parser.add_argument("--disable-warmup", action='store_true',
                   help="Disable model warmup on startup")
parser.add_argument("--warmup-samples", type=int, default=3,
                   help="Number of warmup inference samples to run (default: 3)")

# Embedding model selection arguments
parser.add_argument("--embedding-model", type=str, choices=['mixedbread', 'qwen'], default='mixedbread',
                   help="Embedding model to use (default: mixedbread)")
parser.add_argument("--qwen-size", type=str, choices=['0.6B', '4B', '8B'], default='0.6B',
                   help="Qwen model size to use when --embedding-model=qwen (default: 0.6B)")
parser.add_argument("--qwen-flash-attention", action='store_true',
                   help="Enable flash_attention_2 for Qwen models (requires compatible GPU)")

# Reranking model selection arguments
parser.add_argument("--reranking-model", type=str, choices=['mixedbread', 'qwen'], default=None,
                   help="Reranking model to use (default: mixedbread, or qwen if --embedding-model=qwen)")

args, remaining_args = parser.parse_known_args()

# Setup logging level
setup_pretty_logging(getattr(logging, args.log_level))

# Embedding logging setup
EMBEDDING_LOG_FILE = "./embeddings.log"
EMBEDDING_LOGGING_ENABLED = args.log_embeddings

if EMBEDDING_LOGGING_ENABLED:
    logger.info(f"📝 Embedding logging enabled: {EMBEDDING_LOG_FILE}")

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
# IMPORTANT: For GPU models, use single-threaded executor to avoid CUDA context issues
# PyTorch CUDA models are NOT thread-safe - multiple threads cause separate CUDA allocations
EMBEDDING_THREADS = args.embedding_threads or CPU_COUNT
RERANK_THREADS = args.rerank_threads or CPU_COUNT
CLASSIFICATION_THREADS = args.classification_threads or max(1, CPU_COUNT // 2)
GENERAL_THREADS = args.general_threads or CPU_COUNT * 2

# Detect if we'll be using GPU - adjust threadpool size for concurrent model instances
using_gpu_embedding = args.embedding_device != "cpu"
using_gpu_rerank = args.rerank_device != "cpu"

if using_gpu_embedding:
    # Set threads equal to number of concurrent model instances
    EMBEDDING_THREADS = args.num_concurrent_embedding
    logger.info(f"🔄 Embedding model on GPU ({args.embedding_device}) - using {EMBEDDING_THREADS} thread(s) for {args.num_concurrent_embedding} concurrent model instance(s)")
    
if using_gpu_rerank:
    # Set threads equal to number of concurrent model instances
    RERANK_THREADS = args.num_concurrent_rerank
    logger.info(f"🔄 Rerank model on GPU ({args.rerank_device}) - using {RERANK_THREADS} thread(s) for {args.num_concurrent_rerank} concurrent model instance(s)")

logger.info("Threadpool Configuration:")
logger.info(f"  - Embedding threads: {EMBEDDING_THREADS} {'(GPU-safe)' if using_gpu_embedding else '(CPU-optimized)'}")
logger.info(f"  - Rerank threads: {RERANK_THREADS} {'(GPU-safe)' if using_gpu_rerank else '(CPU-optimized)'}")
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

# Determine which reranking model to use
# If no reranking model is specified but embedding model is qwen, use qwen for reranking too
if args.reranking_model is None:
    if args.embedding_model == 'qwen':
        reranking_model_choice = 'qwen'
        logger.info("No reranking model specified, using qwen (same as embedding model)")
    else:
        reranking_model_choice = 'mixedbread'
else:
    reranking_model_choice = args.reranking_model

logger.info("Initializing models...")

# ----- Model Pool for Concurrent GPU Execution -----

class ModelPool:
    """
    Pool of model instances for concurrent GPU execution.
    Each thread in the threadpool gets its own model instance to avoid CUDA context issues.
    """
    
    def __init__(self, num_instances: int, model_loader_func, model_name: str):
        """
        Initialize model pool
        
        Args:
            num_instances: Number of model instances to create
            model_loader_func: Function that loads and returns a model instance
            model_name: Name of the model (for logging)
        """
        self.num_instances = num_instances
        self.model_name = model_name
        self.models = []
        self.model_index = 0
        self.lock = threading.Lock()
        
        logger.info(f"🔄 Creating model pool for {model_name} with {num_instances} instance(s)...")
        
        # Load multiple model instances
        for i in range(num_instances):
            logger.info(f"  Loading model instance {i+1}/{num_instances}...")
            model = model_loader_func()
            self.models.append(model)
            
        logger.info(f"✅ Model pool initialized with {len(self.models)} instance(s)")
    
    def get_model(self):
        """
        Get next model from pool using round-robin selection.
        Thread-safe.
        """
        with self.lock:
            model = self.models[self.model_index]
            self.model_index = (self.model_index + 1) % self.num_instances
            return model
    
    def get_all_models(self):
        """Get all models in the pool (for cleanup, etc.)"""
        return self.models

# ----- Torch Compilation and Warmup Functions -----

def compile_model_if_enabled(model, model_name: str):
    """
    Apply torch.compile() to a model if enabled via CLI flags.
    
    Args:
        model: The model to compile
        model_name: Name of the model (for logging)
    
    Returns:
        The compiled model if compilation is enabled, otherwise the original model
    """
    if not args.enable_torch_compile:
        return model
    
    # Check PyTorch version
    torch_version = torch.__version__.split('+')[0]  # Remove any +cu118 suffix
    major, minor, patch = torch_version.split('.')[:3]
    
    if int(major) < 2:
        logger.warning(f"⚠️  torch.compile() requires PyTorch 2.0+, current version: {torch.__version__}")
        logger.warning(f"⚠️  Skipping compilation for {model_name}")
        return model
    
    try:
        logger.info(f"🔥 Compiling {model_name} with torch.compile()")
        logger.info(f"   Mode: {args.torch_compile_mode}")
        logger.info(f"   Backend: {args.torch_compile_backend}")
        
        # For SentenceTransformer models, compile the underlying encode method
        if hasattr(model, 'encode'):
            # Compile the model's encode function
            compiled_model = model
            original_encode = model.encode
            
            # Create a wrapper that compiles the internal operations
            def compiled_encode(*args, **kwargs):
                return original_encode(*args, **kwargs)
            
            # Note: torch.compile works best on the underlying transformer model
            # For SentenceTransformer, we compile the first module (transformer)
            if hasattr(model, '_first_module'):
                try:
                    transformer = model._first_module()
                    if hasattr(transformer, 'auto_model'):
                        logger.info(f"   Compiling underlying transformer model...")
                        transformer.auto_model = torch.compile(
                            transformer.auto_model,
                            mode=args.torch_compile_mode,
                            backend=args.torch_compile_backend
                        )
                        logger.info(f"✅ {model_name} compiled successfully")
                    else:
                        logger.warning(f"⚠️  Could not access auto_model for compilation")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to compile transformer: {e}")
            
            return compiled_model
        
        # For QwenReranker or other models with a .model attribute
        elif hasattr(model, 'model'):
            logger.info(f"   Compiling underlying model...")
            model.model = torch.compile(
                model.model,
                mode=args.torch_compile_mode,
                backend=args.torch_compile_backend
            )
            logger.info(f"✅ {model_name} compiled successfully")
            return model
        
        else:
            # Direct compilation
            compiled = torch.compile(
                model,
                mode=args.torch_compile_mode,
                backend=args.torch_compile_backend
            )
            logger.info(f"✅ {model_name} compiled successfully")
            return compiled
            
    except Exception as e:
        logger.error(f"❌ Failed to compile {model_name}: {e}")
        logger.warning(f"⚠️  Continuing with uncompiled model")
        return model

def warmup_model(model, model_type: str, model_name: str, num_samples: int = 3):
    """
    Perform warmup inference on a model to trigger compilation and cache warming.
    
    Args:
        model: The model to warm up
        model_type: Type of model ('embedding', 'rerank', 'classification')
        model_name: Name of the model (for logging)
        num_samples: Number of warmup samples to run
    """
    if args.disable_warmup:
        logger.info(f"⏭️  Skipping warmup for {model_name} (disabled via --disable-warmup)")
        return
    
    logger.info(f"🔥 Warming up {model_name} ({model_type})...")
    logger.info(f"   Running {num_samples} warmup inference(s)...")
    
    start_time = time.time()
    
    try:
        with torch.inference_mode():
            for i in range(num_samples):
                if model_type == 'embedding':
                    # Warmup with dummy text of varying lengths
                    dummy_texts = [
                        "This is a warmup sample.",
                        "This is a longer warmup sample with more tokens to test batching behavior.",
                        "Short warmup."
                    ]
                    
                    if hasattr(model, 'encode'):
                        # SentenceTransformer
                        if args.embedding_model == 'qwen':
                            _ = model.encode([dummy_texts[i % len(dummy_texts)]], prompt_name="query")
                        else:
                            _ = model.encode([dummy_texts[i % len(dummy_texts)]])
                    else:
                        logger.warning(f"⚠️  Model does not have encode method, skipping warmup")
                        return
                
                elif model_type == 'rerank':
                    # Warmup with dummy query and documents
                    dummy_query = "What is machine learning?"
                    dummy_docs = [
                        "Machine learning is a subset of artificial intelligence.",
                        "Deep learning uses neural networks with multiple layers.",
                        "Python is a popular programming language for data science."
                    ]
                    
                    if hasattr(model, 'rank'):
                        _ = model.rank(dummy_query, dummy_docs, top_k=2)
                    else:
                        logger.warning(f"⚠️  Model does not have rank method, skipping warmup")
                        return
                
                elif model_type == 'classification':
                    # Warmup with dummy text
                    dummy_text = "This is a test sentence for classification warmup."
                    
                    if callable(model):
                        _ = model(dummy_text)
                    else:
                        logger.warning(f"⚠️  Model is not callable, skipping warmup")
                        return
        
        # Clear CUDA cache after warmup if using GPU
        if torch.cuda.is_available():
            device_str = ""
            if hasattr(model, 'device'):
                device_str = str(model.device)
            elif model_type == 'embedding':
                device_str = args.embedding_device
            elif model_type == 'rerank':
                device_str = args.rerank_device
            
            if device_str and (device_str.startswith('cuda') or device_str == 'mps'):
                torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Warmup completed for {model_name} in {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Warmup failed for {model_name}: {e}")
        logger.warning(f"⚠️  Continuing anyway, but first inference may be slower")

# Embedding model initialization with truncation to specified dimensions and device specification
logger.info(f"Loading embedding model on device: {args.embedding_device}")
logger.info(f"Using embedding model: {args.embedding_model}")

# Define model loader function for pooling
def load_embedding_model():
    """Load a single embedding model instance"""
    if args.embedding_model == 'qwen':
        qwen_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
        if args.qwen_flash_attention:
            model = SentenceTransformer(
                qwen_model_name,
                model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
                tokenizer_kwargs={"padding_side": "left"},
                device=args.embedding_device
            )
        else:
            model = SentenceTransformer(qwen_model_name, device=args.embedding_device)
    else:
        # MixedBread model
        model = SentenceTransformer(
            "mixedbread-ai/mxbai-embed-large-v1",
            truncate_dim=args.embedding_dim,
            device=args.embedding_device
        )
    
    # Apply torch.compile() if enabled
    model = compile_model_if_enabled(model, f"Embedding-{args.embedding_model}")
    
    return model

# Determine if we should use a shared pool for both embedding and reranking
# This happens when using Qwen for both tasks on the same device
use_shared_qwen_pool = (
    args.embedding_model == 'qwen' and
    reranking_model_choice == 'qwen' and
    args.embedding_device == args.rerank_device and
    not args.qwen_flash_attention and
    (using_gpu_embedding or using_gpu_rerank) and
    (args.num_concurrent_embedding > 1 or args.num_concurrent_rerank > 1)
)

if use_shared_qwen_pool:
    # Create a shared pool for both embedding and reranking
    # Pool size is the maximum of the two concurrent settings
    shared_pool_size = max(args.num_concurrent_embedding, args.num_concurrent_rerank)
    qwen_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
    embedding_model_name = qwen_model_name
    
    logger.info(f"♻️  Creating SHARED model pool for both embedding and reranking")
    logger.info(f"   Using {shared_pool_size} Qwen instance(s) for both tasks")
    logger.info(f"   Memory savings: Loading {shared_pool_size} models instead of {args.num_concurrent_embedding + args.num_concurrent_rerank}")
    
    # Create shared pool
    shared_qwen_pool = ModelPool(
        shared_pool_size,
        load_embedding_model,
        embedding_model_name
    )
    
    embedding_model_pool = shared_qwen_pool
    embedding_model = None
    
elif using_gpu_embedding and args.num_concurrent_embedding > 1:
    # Use separate model pool for embedding only
    if args.embedding_model == 'qwen':
        embedding_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
        logger.info(f"Qwen model (native dimensions)")
    else:
        embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
        logger.info(f"Using embedding dimensions: {args.embedding_dim}")
    
    embedding_model_pool = ModelPool(
        args.num_concurrent_embedding,
        load_embedding_model,
        embedding_model_name
    )
    embedding_model = None  # We'll use the pool instead
    shared_qwen_pool = None
else:
    # Single model instance (traditional approach)
    if args.embedding_model == 'qwen':
        qwen_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
        logger.info(f"Loading Qwen model: {qwen_model_name}")
        
        if args.qwen_flash_attention:
            logger.info("Enabling flash_attention_2 for Qwen model")
            embedding_model = SentenceTransformer(
                qwen_model_name,
                model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
                tokenizer_kwargs={"padding_side": "left"},
                device=args.embedding_device
            )
        else:
            embedding_model = SentenceTransformer(qwen_model_name, device=args.embedding_device)
        
        embedding_model_name = qwen_model_name
        logger.info(f"Qwen model loaded (native dimensions)")
    else:
        embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
        logger.info(f"Using embedding dimensions: {args.embedding_dim}")
        embedding_model = SentenceTransformer(embedding_model_name, truncate_dim=args.embedding_dim, device=args.embedding_device)
    
    embedding_model_pool = None
    shared_qwen_pool = None

# Apply torch.compile() to single embedding model if enabled
if embedding_model is not None and not embedding_model_pool:
    embedding_model = compile_model_if_enabled(embedding_model, f"Embedding-{args.embedding_model}")

# Warmup embedding model(s)
if embedding_model_pool:
    # Warmup all models in pool
    logger.info(f"🔥 Warming up embedding model pool ({embedding_model_pool.num_instances} instance(s))...")
    for i, model in enumerate(embedding_model_pool.get_all_models()):
        logger.info(f"   Warming up embedding model instance {i+1}/{embedding_model_pool.num_instances}")
        warmup_model(model, 'embedding', f"{embedding_model_name}-{i+1}", num_samples=args.warmup_samples)
elif embedding_model:
    # Warmup single model
    warmup_model(embedding_model, 'embedding', embedding_model_name, num_samples=args.warmup_samples)

# Reranking model initialization with device specification
logger.info(f"Using reranking model: {reranking_model_choice}")

# Create pool-aware reranker wrapper for shared Qwen pool
class PooledQwenReranker:
    """Wrapper for Qwen reranker that uses a model pool"""
    def __init__(self, model_pool, rerank_instruction: str):
        self.model_pool = model_pool
        self.rerank_instruction = rerank_instruction
        self.max_length = 8192
        logger.info(f"♻️  Qwen reranker initialized using shared model pool")
    
    def rank(self, query: str, documents: list[str], return_documents: bool = False, top_k: int = None) -> dict:
        """Rank using a model from the shared pool"""
        # Get a model from the pool
        model = self.model_pool.get_model()
        
        # Format query with instruction
        formatted_query = get_detailed_instruct(self.rerank_instruction, query)
        
        # Combine query and documents for batch processing
        input_texts = [formatted_query] + documents
        
        # Tokenize and process using the pooled model
        with torch.inference_mode():
            batch_dict = model.tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            # Get the underlying transformer model from SentenceTransformer
            if hasattr(model, '_first_module'):
                transformer_model = model._first_module().auto_model
                batch_dict = {k: v.to(transformer_model.device) for k, v in batch_dict.items()}
                
                # Get embeddings
                outputs = transformer_model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # Calculate scores
                query_embedding = embeddings[0:1]
                doc_embeddings = embeddings[1:]
                scores = (query_embedding @ doc_embeddings.T).squeeze(0)
                
                # Convert to list
                scores_list = scores.cpu().tolist()
            else:
                raise ValueError("Cannot access underlying transformer model from SentenceTransformer")
        
        # Create results
        results = []
        for idx, score in enumerate(scores_list):
            result = {
                "index": idx,
                "score": float(score)
            }
            if return_documents:
                result["document"] = documents[idx]
            results.append(result)
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply top_k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return {"results": results}

if reranking_model_choice == 'qwen':
    if use_shared_qwen_pool:
        # Use the shared pool for reranking
        logger.info("♻️  Reranker using SHARED pool with embedding model")
        rerank_model = PooledQwenReranker(
            shared_qwen_pool,
            "Given the following storytelling documents, which of these best add to the user's message?"
        )
        rerank_model_pool = None  # Using shared pool
        
    elif (args.embedding_model == 'qwen' and 
          args.embedding_device == args.rerank_device and
          not args.qwen_flash_attention and
          args.num_concurrent_embedding == 1 and
          args.num_concurrent_rerank == 1):
        # Traditional single model sharing (only when both are single instance)
        logger.info("♻️  Reusing Qwen embedding model for reranking (memory optimization - no additional model loaded)")
        qwen_reranker_model_name = qwen_model_name
        
        # Access the underlying transformers model from SentenceTransformer
        if hasattr(embedding_model, '_first_module'):
            qwen_tokenizer =embedding_model.tokenizer
            qwen_transformer_model = embedding_model._first_module().auto_model
            
            # Create a custom reranker instance that shares the model
            class SharedQwenReranker(QwenReranker):
                def __init__(self, tokenizer, model, rerank_instruction):
                    self.device = model.device
                    self.rerank_instruction = rerank_instruction
                    self.max_length = 8192
                    self.tokenizer = tokenizer
                    self.model = model
                    logger.info(f"Qwen reranker initialized using shared single model")
            
            rerank_model = SharedQwenReranker(
                qwen_tokenizer,
                qwen_transformer_model,
                "Given the following storytelling documents, which of these best add to the user's message?"
            )
        else:
            logger.warning("Could not access wrapped model, loading separate reranker")
            rerank_model = QwenReranker(
                f"Qwen/Qwen3-Embedding-{args.qwen_size}",
                device=args.rerank_device,
                use_flash_attention=args.qwen_flash_attention
            )
        rerank_model_pool = None
        
    else:
        # Load separate Qwen reranker (or pool if num_concurrent_rerank > 1)
        qwen_reranker_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
        
        if using_gpu_rerank and args.num_concurrent_rerank > 1:
            # Create separate pool for reranking
            logger.info(f"Loading rerank model pool on device: {args.rerank_device}")
            logger.info(f"Loading separate Qwen reranker pool: {qwen_reranker_model_name}")
            
            def load_qwen_reranker():
                return QwenReranker(
                    qwen_reranker_model_name,
                    device=args.rerank_device,
                    use_flash_attention=args.qwen_flash_attention
                )
            
            rerank_model_pool = ModelPool(
                args.num_concurrent_rerank,
                load_qwen_reranker,
                qwen_reranker_model_name
            )
            rerank_model = None  # Using pool
        else:
            # Single reranker instance
            logger.info(f"Loading rerank model on device: {args.rerank_device}")
            logger.info(f"Loading separate Qwen reranker: {qwen_reranker_model_name}")
            rerank_model = QwenReranker(
                qwen_reranker_model_name,
                device=args.rerank_device,
                use_flash_attention=args.qwen_flash_attention
            )
            rerank_model_pool = None
else:
    # Use MixedBread reranker (default)
    logger.info(f"Loading rerank model on device: {args.rerank_device}")
    rerank_model = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-base-v2", device=args.rerank_device)
    rerank_model_pool = None

# Apply torch.compile() to single rerank model if enabled
if rerank_model is not None and not rerank_model_pool and reranking_model_choice != 'qwen':
    # MixedBread reranker compilation
    rerank_model = compile_model_if_enabled(rerank_model, f"Rerank-{reranking_model_choice}")
elif rerank_model is not None and not rerank_model_pool and reranking_model_choice == 'qwen':
    # Qwen reranker compilation (only if not shared)
    if not use_shared_qwen_pool:
        rerank_model = compile_model_if_enabled(rerank_model, f"Rerank-Qwen")

# Warmup reranking model(s)
if rerank_model_pool:
    # Warmup all models in pool
    logger.info(f"🔥 Warming up reranking model pool ({rerank_model_pool.num_instances} instance(s))...")
    for i, model in enumerate(rerank_model_pool.get_all_models()):
        logger.info(f"   Warming up reranking model instance {i+1}/{rerank_model_pool.num_instances}")
        warmup_model(model, 'rerank', f"{reranking_model_choice}-{i+1}", num_samples=args.warmup_samples)
elif rerank_model and not use_shared_qwen_pool:
    # Warmup single model (skip if using shared pool, already warmed up)
    warmup_model(rerank_model, 'rerank', f"Rerank-{reranking_model_choice}", num_samples=args.warmup_samples)
elif use_shared_qwen_pool:
    # Shared pool warmup already done during embedding warmup
    logger.info(f"⏭️  Skipping rerank warmup (using shared pool already warmed up)")

logger.info("Models initialized successfully")

# ----- CUDA Cache Manager -----

class CUDACacheManager:
    """Manages CUDA cache with TTL-based automatic clearing"""
    
    def __init__(self, ttl_seconds: int = 300, min_clear_interval: int = 60, 
                 memory_threshold: int = 80, enabled: bool = False):
        """
        Initialize CUDA cache manager
        
        Args:
            ttl_seconds: Time in seconds before clearing cache when idle
            min_clear_interval: Minimum seconds between cache clears
            memory_threshold: Only clear if GPU memory usage > this percentage
            enabled: Whether automatic clearing is enabled
        """
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

# Initialize CUDA cache manager
cuda_cache_manager = CUDACacheManager(
    ttl_seconds=args.cuda_cache_ttl,
    min_clear_interval=args.cuda_min_clear_interval,
    memory_threshold=args.cuda_memory_threshold,
    enabled=args.cuda_cache_ttl_enabled
)

def get_cache_size(obj):
    """
    Recursively estimate the size in bytes of a Python object.
    This is a rough approximation.z
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

# --- Lazy-load the classification pipeline ---
# The classifier will be loaded only when first needed to save memory

classifier = None

def get_classifier():
    """Lazy-load the classification pipeline on first use"""
    global classifier
    if classifier is None:
        logger.info("🔄 Loading classification model (first use)...")
        classifier = pipeline(
            'text-classification',
            model='SamLowe/roberta-base-go_emotions',
            top_k=None
        )
        logger.info("✅ Classification model loaded")
    return classifier

# ----- Logging Setup and Argument Parsing -----

cache_limit_bytes = args.cache_limit * 1024 * 1024

logger.info(f"💾 Using cache limit: {args.cache_limit} MB ({cache_limit_bytes} bytes)")
logger.info(f"🔢 Embedding quantization: {args.quant}")

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
        import datetime
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": metadata.get("model", embedding_model_name),
            "quantization": metadata.get("quantization", args.quant),
            "embedding_dimensions": metadata.get("embedding_dimensions", args.embedding_dim if args.embedding_model == 'mixedbread' else "native"),
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
    
    # Mark inference activity for CUDA cache manager
    cuda_cache_manager.mark_inference_activity()
    
    # Get reranker model (from pool if using concurrent instances, otherwise use global)
    def get_rerank_model_and_rank(query, documents, return_documents, top_k):
        """Get model from pool and perform ranking"""
        if rerank_model_pool:
            model = rerank_model_pool.get_model()
        else:
            model = rerank_model
        
        return model.rank(query, documents, return_documents=return_documents, top_k=top_k)
    
    # Use dedicated rerank threadpool for optimal CPU utilization
    result = await run_in_threadpool_with_executor(
        rerank_executor, 
        get_rerank_model_and_rank,
        request.query,
        request.documents,
        request.return_documents,
        request.top_k
    )
    
    # Clear CUDA cache immediately after inference if using GPU
    if torch.cuda.is_available() and (args.rerank_device.startswith('cuda') or args.rerank_device == 'mps'):
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache after reranking")
    
    rerank_cache[key] = result
    return result

@app.post("/v1/embeddings")
async def embedding_endpoint(request: EmbeddingRequest, api_key: str = Depends(get_api_key)):
    inputs = request.input if isinstance(request.input, list) else [request.input]
    if not inputs:
        raise HTTPException(status_code=400, detail="Input must be provided")
    
    logger.info(f"📄 Processing {len(inputs)} documents for embeddings")
    
    key = get_embedding_cache_key(inputs) if inputs else ""
    if key in embedding_cache:
        return embedding_cache[key]
    
    # Mark inference activity for CUDA cache manager
    cuda_cache_manager.mark_inference_activity()
    
    # Get model (from pool if using concurrent instances, otherwise use global)
    def get_embedding_model_and_encode(inputs):
        """Get model from pool and encode inputs"""
        if embedding_model_pool:
            model = embedding_model_pool.get_model()
        else:
            model = embedding_model
        
        # For Qwen models, use prompt_name="query" for better retrieval performance
        if args.embedding_model == 'qwen':
            return model.encode(inputs, prompt_name="query")
        else:
            return model.encode(inputs)
    
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
                get_embedding_model_and_encode,
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
            get_embedding_model_and_encode,
            inputs
        )
    
    # Apply quantization based on CLI argument
    if args.quant != 'standard':
        # Create wrapper function for quantize_embeddings to handle keyword arguments
        def quantize_embeddings_wrapper(embeddings):
            return quantize_embeddings(embeddings, precision=args.quant)
        
        docs_embeddings = await run_in_threadpool_with_executor(
            embedding_executor,
            quantize_embeddings_wrapper,
            docs_embeddings
        )
        logger.debug(f"Applied {args.quant} quantization to embeddings")
    
    # Ensure embeddings are on CPU before converting to list (prevents GPU memory leaks in cache)
    if hasattr(docs_embeddings, 'cpu'):
        docs_embeddings = docs_embeddings.cpu()
    
    # Convert embeddings to list format
    embeddings_list = docs_embeddings.tolist()
    
    # Clear CUDA cache immediately after inference if using GPU
    if torch.cuda.is_available() and (args.embedding_device.startswith('cuda') or args.embedding_device == 'mps'):
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache after embedding")
    
    # Log the embedding results if logging is enabled
    log_embedding_result(
        inputs=inputs,
        embeddings=embeddings_list,
        metadata={
            "model": embedding_model_name,
            "quantization": args.quant,
            "embedding_dimensions": args.embedding_dim if args.embedding_model == 'mixedbread' else "native"
        }
    )
    
    # Format response to mimic OpenAI's embeddings API.
    data = []
    for idx, emb in enumerate(embeddings_list):
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
        "model": embedding_model_name,
        "usage": usage
    }
    embedding_cache[key] = result
    return result

@app.post("/v1/classify")
async def classify_endpoint(request: ClassificationRequest, api_key: str = Depends(get_api_key)):
    # Use dedicated classification threadpool for optimal CPU utilization
    # Lazy-load classifier on first use
    clf = get_classifier()
    raw_result = await run_in_threadpool_with_executor(
        classification_executor,
        clf,
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
    
    logger.info(f"📄 Processing {len(inputs)} documents for Ollama embeddings")
    
    # Get model (from pool if using concurrent instances, otherwise use global)
    def get_embedding_model_and_encode(inputs):
        """Get model from pool and encode inputs"""
        if embedding_model_pool:
            model = embedding_model_pool.get_model()
        else:
            model = embedding_model
        
        # For Qwen models, use prompt_name="query" for better retrieval performance
        if args.embedding_model == 'qwen':
            return model.encode(inputs, prompt_name="query")
        else:
            return model.encode(inputs)
    
    # Generate embeddings using the existing embedding model
    # Use dedicated embedding threadpool for optimal performance
    docs_embeddings = await run_in_threadpool_with_executor(
        embedding_executor,
        get_embedding_model_and_encode,
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
        model=embedding_model_name
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
    
    logger.info(f"📄 Processing 1 document for Llama.cpp embeddings")
    
    # Get model (from pool if using concurrent instances, otherwise use global)
    def get_embedding_model_and_encode(inputs):
        """Get model from pool and encode inputs"""
        if embedding_model_pool:
            model = embedding_model_pool.get_model()
        else:
            model = embedding_model
        
        # For Qwen models, use prompt_name="query" for better retrieval performance  
        if args.embedding_model == 'qwen':
            return model.encode(inputs, prompt_name="query")
        else:
            return model.encode(inputs)
    
    # Generate embedding using the existing embedding model
    # Use dedicated embedding threadpool for optimal performance
    inputs = [request.content]
    docs_embeddings = await run_in_threadpool_with_executor(
        embedding_executor,
        get_embedding_model_and_encode,
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
        model=embedding_model_name
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
      - CUDA cache management statistics
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
    
    # Add CUDA cache management statistics
    data["cuda_cache_manager"] = cuda_cache_manager.get_stats()
    
    if torch.cuda.is_available():
        # Report GPU memory allocated on device 0.
        data["gpu_memory_allocated"] = torch.cuda.memory_allocated(0)
        data["gpu_memory_reserved"] = torch.cuda.memory_reserved(0)
    
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
        "embedding_model": args.embedding_model,
        "embedding_model_name": embedding_model_name,
        "embedding_quantization": args.quant,
        "embedding_dimensions": args.embedding_dim if args.embedding_model == 'mixedbread' else "native",
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

# Global shutdown flag and lock to prevent multiple cleanup attempts
shutdown_event = threading.Event()
cleanup_lock = threading.Lock()
cleanup_completed = False

# Timeout for thread cleanup (in seconds)
SHUTDOWN_TIMEOUT = 5.0

def cleanup_threadpools():
    """Clean up threadpool executors on shutdown with timeout"""
    executors = [
        ("embedding", embedding_executor),
        ("rerank", rerank_executor),
        ("classification", classification_executor),
        ("general", general_executor)
    ]
    
    for name, executor in executors:
        try:
            logger.info(f"Shutting down {name} threadpool executor...")
            # Cancel all pending futures first
            executor.shutdown(wait=False, cancel_futures=True)
            
            # Wait for threads to complete with timeout
            # Use a separate thread to implement timeout on shutdown
            shutdown_thread = threading.Thread(
                target=lambda: executor.shutdown(wait=True),
                daemon=True
            )
            shutdown_thread.start()
            shutdown_thread.join(timeout=SHUTDOWN_TIMEOUT)
            
            if shutdown_thread.is_alive():
                logger.warning(f"⚠️  {name} executor shutdown timed out after {SHUTDOWN_TIMEOUT}s")
            else:
                logger.info(f"✅ {name} executor shut down successfully")
        except Exception as e:
            logger.error(f"Error during {name} executor cleanup: {e}")

def cleanup_resources():
    """Comprehensive cleanup of all resources with safety checks"""
    global cleanup_completed
    
    # Use lock to ensure cleanup only runs once
    with cleanup_lock:
        if cleanup_completed:
            logger.debug("Cleanup already completed, skipping")
            return
        
        if shutdown_event.is_set():
            logger.debug("Shutdown already in progress, skipping duplicate cleanup")
            return
        
        shutdown_event.set()
        logger.info("🔄 Initiating graceful shutdown...")
        
        # Stop CUDA cache manager thread
        try:
            if cuda_cache_manager.enabled and cuda_cache_manager.cuda_available:
                cuda_cache_manager.stop_monitor_thread()
                # Final CUDA cache clear on shutdown
                if torch.cuda.is_available():
                    cuda_cache_manager.clear_cuda_cache(reason="shutdown")
        except Exception as e:
            logger.error(f"Error during CUDA cache manager cleanup: {e}")
        
        # Clean up threadpools with timeout
        try:
            cleanup_threadpools()
        except Exception as e:
            logger.error(f"Error during threadpool cleanup: {e}")
        
        # Clean up models and caches
        try:
            logger.info("Cleaning up model resources...")
            global rerank_cache, embedding_cache
            rerank_cache.clear()
            embedding_cache.clear()
            logger.info("✅ Model resources cleaned up")
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
        
        # Mark cleanup as completed
        cleanup_completed = True
        logger.info("✅ Graceful shutdown completed")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully - minimal work in signal handler"""
    signal_name = signal.Signals(signum).name
    logger.info(f"📡 Received {signal_name} signal")
    
    # Set shutdown flag - actual cleanup will happen in main shutdown flow
    shutdown_event.set()
    
    # Signal uvicorn server to exit if it exists
    if 'server' in globals():
        server.should_exit = True

# Register signal handlers for graceful shutdown
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
        self._shutdown_lock = threading.Lock()
    
    async def startup(self, sockets=None):
        """Override startup to add our custom logging at the right time"""
        # Call the original startup
        await super().startup(sockets)
        
        # Now that the server is actually starting up, log our messages
        logger.info("🍞 BananaBread-Emb server is now running!")
        logger.info(f"🌐 Server available at: http://{self.config.host}:{self.config.port}")
        logger.info(f"📚 API Documentation: http://{self.config.host}:{self.config.port}/docs")
        logger.info(f"📖 ReDoc Documentation: http://{self.config.host}:{self.config.port}/redoc")
    
    async def shutdown(self, sockets=None):
        """Override shutdown to ensure graceful cleanup"""
        with self._shutdown_lock:
            if self._shutdown_requested:
                logger.debug("Shutdown already in progress, skipping duplicate shutdown")
                return
            
            self._shutdown_requested = True
            logger.info("🛑 Server shutdown initiated...")
        
        # Call the original shutdown first to stop accepting new requests
        try:
            await super().shutdown(sockets)
        except Exception as e:
            logger.error(f"Error during uvicorn shutdown: {e}")
        
        # Then call our custom cleanup
        cleanup_resources()
        
        logger.info("✅ Server shutdown completed")
    
    def handle_exit(self, sig, frame):
        """Handle exit signals more gracefully"""
        signal_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else str(sig)
        logger.info(f"📡 Received exit signal {signal_name}, shutting down server...")
        self.should_exit = True

def main():
    """Main entry point for the bananabread-emb console script"""
    global server
    
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
        """Handle signals for server shutdown - minimal work here"""
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.info(f"📡 Received {signal_name} signal, initiating server shutdown...")
        
        # Only set the shutdown flag - don't call cleanup directly
        # Cleanup will be handled by uvicorn's shutdown sequence
        server.should_exit = True
    
    # Override the signal handlers to use our server-specific handler
    signal.signal(signal.SIGINT, server_signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, server_signal_handler)  # Termination signal
    
    # Also handle SIGBREAK on Windows
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, server_signal_handler)
    
    logger.info("🚀 Initializing BananaBread-Emb server...")
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("⌨️  Received KeyboardInterrupt, shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Unexpected error during server execution: {e}")
        raise
    finally:
        # Ensure cleanup always runs at the end
        logger.info("🧹 Running final cleanup...")
        cleanup_resources()

if __name__ == "__main__":
    main()
