import os
import sys
import json
import logging
import argparse
import multiprocessing
import subprocess
import random
import torch
from typing import Dict, Any
from transformers import set_seed

# ----- Enhanced Logging Configuration -----
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

# Setup pretty logging initially (will be updated after args parse)
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

# ----- GPU/CUDA Optimizations (Environment Variables) -----
# Set these before importing other heavy internal modules if possible, 
# keeping them here or in the main init.
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
os.environ['TORCHINDUCTOR_CUDAGRAPH_TREES'] = '0'

# ----- Config File and Argument Parsing -----

CONFIG_FILE = "config.json"

# Define defaults for arguments (moved here so it's available for config creation)
DEFAULTS = {
    "cache_limit": 1024,
    "embedding_device": "cpu",
    "rerank_device": "cpu",
    "log_level": "INFO",
    "quant": "standard",
    "embedding_dim": 1024,
    "cuda_cache_ttl": 300,
    "cuda_min_clear_interval": 60,
    "cuda_memory_threshold": 80,
    "num_concurrent_embedding": 1,
    "num_concurrent_rerank": 1,
    "torch_compile_mode": "default",
    "torch_compile_backend": "inductor",
    "warmup_samples": 3,
    "embedding_model": "mixedbread",
    "qwen_size": "0.6B",
    "seed": 65,
    # None defaults for optional overrides
    "use_cores": None,
    "cpu_socket": None,
    "embedding_threads": None,
    "rerank_threads": None,
    "classification_threads": None,
    "general_threads": None,
    "reranking_model": None,
    "qwen_flash_attention": False,
    "enable_torch_compile": False,
    "enable_warmup": True,
    "cuda_cache_ttl_enabled": False,
    "log_embeddings": False
}

def create_default_config() -> Dict[str, Any]:
    """Create default config.json file on first startup"""
    # Only include non-None defaults that users would want to customize
    config_defaults = {k: v for k, v in DEFAULTS.items() if v is not None}
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_defaults, f, indent=4)
        logger.info(f"📝 Created default configuration file: {CONFIG_FILE}")
        return config_defaults
    except Exception as e:
        logger.warning(f"⚠️  Failed to create config file: {e}")
        return {}

def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if it doesn't exist"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                logger.info(f"📄 Loaded configuration from {CONFIG_FILE}")
                return config
        except Exception as e:
            logger.warning(f"⚠️  Failed to load config file: {e}")
            return {}
    else:
        return create_default_config()

CPU_INFO = get_cpu_info()
AVAILABLE_SOCKETS = get_available_cores()
TOTAL_PHYSICAL_CORES = multiprocessing.cpu_count()

def parse_args():
    parser = argparse.ArgumentParser(
        description="BananaBread-Emb - Optimized MixedBread AI Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      python server.py
      python server.py --use-cores 8
      python server.py --embedding-device cuda --rerank-device cuda
        """
    )

    # CPU and core selection arguments
    parser.add_argument("--use-cores", type=int, default=DEFAULTS.get("use_cores"), 
                       help=f"Number of physical CPU cores to use (default: all {TOTAL_PHYSICAL_CORES} cores)")
    parser.add_argument("--cpu-socket", type=int, choices=list(range(len(AVAILABLE_SOCKETS))), default=DEFAULTS.get("cpu_socket"),
                       help="Pin operations to specific CPU socket (for multi-socket systems)")

    # Threadpool arguments
    parser.add_argument("--cache-limit", type=int, default=DEFAULTS["cache_limit"], help=f"Cache limit in MB (default: {DEFAULTS['cache_limit']})")
    parser.add_argument("--embedding-threads", type=int, default=DEFAULTS.get("embedding_threads"), 
                       help="Number of threads for embedding operations (default: auto-detected)")
    parser.add_argument("--rerank-threads", type=int, default=DEFAULTS.get("rerank_threads"),
                       help="Number of threads for reranking operations (default: auto-detected)")
    parser.add_argument("--classification-threads", type=int, default=DEFAULTS.get("classification_threads"),
                       help="Number of threads for classification operations (default: auto-detected)")
    parser.add_argument("--general-threads", type=int, default=DEFAULTS.get("general_threads"),
                       help="Number of threads for general operations (default: auto-detected)")

    # Device selection arguments
    parser.add_argument("--embedding-device", type=str, default=DEFAULTS["embedding_device"],
                       help=f"Device to load embedding model on (default: {DEFAULTS['embedding_device']})")
    parser.add_argument("--rerank-device", type=str, default=DEFAULTS["rerank_device"],
                       help=f"Device to load rerank model on (default: {DEFAULTS['rerank_device']})")

    # Logging arguments
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=DEFAULTS["log_level"],
                       help=f"Set logging level (default: {DEFAULTS['log_level']})")

    # Quantization arguments
    parser.add_argument("--quant", type=str, choices=['standard', 'ubinary', 'int8'], default=DEFAULTS["quant"],
                       help=f"Quantization precision for embeddings (default: {DEFAULTS['quant']})")

    # Embedding dimensions argument
    parser.add_argument("--embedding-dim", type=int, default=DEFAULTS["embedding_dim"],
                       help=f"Embedding dimensions to truncate to (default: {DEFAULTS['embedding_dim']})")

    # Embedding logging argument
    parser.add_argument("--log-embeddings", action='store_true', default=DEFAULTS.get("log_embeddings"),
                       help="Enable logging of embedding queries and results to embeddings.log")

    # CUDA cache management arguments
    parser.add_argument("--cuda-cache-ttl", type=int, default=DEFAULTS["cuda_cache_ttl"],
                       help=f"Time in seconds before clearing CUDA cache when idle (default: {DEFAULTS['cuda_cache_ttl']})")
    parser.add_argument("--cuda-cache-ttl-enabled", action='store_true', default=DEFAULTS.get("cuda_cache_ttl_enabled"),
                       help="Enable automatic CUDA cache clearing based on TTL")
    parser.add_argument("--cuda-min-clear-interval", type=int, default=DEFAULTS["cuda_min_clear_interval"],
                       help=f"Minimum seconds between CUDA cache clears (default: {DEFAULTS['cuda_min_clear_interval']})")
    parser.add_argument("--cuda-memory-threshold", type=int, default=DEFAULTS["cuda_memory_threshold"],
                       help=f"Only clear CUDA cache if memory usage exceeds this percentage (default: {DEFAULTS['cuda_memory_threshold']})")

    # Model concurrency arguments
    parser.add_argument("--num-concurrent-embedding", type=int, default=DEFAULTS["num_concurrent_embedding"],
                       help=f"Number of concurrent embedding model instances to load on GPU (default: {DEFAULTS['num_concurrent_embedding']})")
    parser.add_argument("--num-concurrent-rerank", type=int, default=DEFAULTS["num_concurrent_rerank"],
                       help=f"Number of concurrent reranking model instances to load on GPU (default: {DEFAULTS['num_concurrent_rerank']})")

    # Torch compilation arguments
    parser.add_argument("--enable-torch-compile", action='store_true', default=DEFAULTS.get("enable_torch_compile"),
                       help="Enable torch.compile() for models (requires PyTorch 2.0+)")
    parser.add_argument("--torch-compile-mode", type=str, 
                       choices=['default', 'reduce-overhead', 'max-autotune'], 
                       default=DEFAULTS["torch_compile_mode"],
                       help=f"Torch compilation mode (default: {DEFAULTS['torch_compile_mode']})")
    parser.add_argument("--torch-compile-backend", type=str, default=DEFAULTS["torch_compile_backend"],
                       help=f"Torch compilation backend (default: {DEFAULTS['torch_compile_backend']})")

    # Model warmup arguments
    parser.add_argument("--enable-warmup", action='store_true', default=DEFAULTS.get("enable_warmup"),
                       help="Enable model warmup on startup (default: True)")
    parser.add_argument("--disable-warmup", action='store_true',
                       help="Disable model warmup on startup")
    parser.add_argument("--warmup-samples", type=int, default=DEFAULTS["warmup_samples"],
                       help=f"Number of warmup inference samples to run (default: {DEFAULTS['warmup_samples']})")

    # Embedding model selection arguments
    parser.add_argument("--embedding-model", type=str, choices=['mixedbread', 'qwen'], default=DEFAULTS["embedding_model"],
                       help=f"Embedding model to use (default: {DEFAULTS['embedding_model']})")
    parser.add_argument("--qwen-size", type=str, choices=['0.6B', '4B', '8B'], default=DEFAULTS["qwen_size"],
                       help=f"Qwen model size to use when --embedding-model=qwen (default: {DEFAULTS['qwen_size']})")
    parser.add_argument("--qwen-flash-attention", action='store_true', default=DEFAULTS.get("qwen_flash_attention"),
                       help="Enable flash_attention_2 for Qwen models (requires compatible GPU)")

    # Reranking model selection arguments
    parser.add_argument("--reranking-model", type=str, choices=['mixedbread', 'qwen'], default=DEFAULTS.get("reranking_model"),
                       help="Reranking model to use (default: mixedbread, or qwen if --embedding-model=qwen)")

    # Determinism
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                        help="Random seed for reproducibility. Set to -1 for random seed. (default: 42)")

    # Load configuration and apply to parser defaults
    config = load_config()
    if config:
        parser.set_defaults(**config)
        logger.info("⚙️  Applied configuration defaults from file")

    args, _ = parser.parse_known_args()
    return args

# Global args instance
args = parse_args()

# Apply seed
if args.seed == -1:
    args.seed = random.randint(0, 2**32 - 1)
    logger.info(f"🎲 Using random seed: {args.seed}")
else:
    logger.info(f"🎲 Using fixed seed: {args.seed}")

set_seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Apply logging level from args
setup_pretty_logging(getattr(logging, args.log_level))

# Embedding logging setup
EMBEDDING_LOG_FILE = "./embeddings.log"
EMBEDDING_LOGGING_ENABLED = args.log_embeddings

if EMBEDDING_LOGGING_ENABLED:
    logger.info(f"📝 Embedding logging enabled: {EMBEDDING_LOG_FILE}")
