import os
import sys
import json
import logging
import argparse
import multiprocessing
import subprocess
import random
from pathlib import Path
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

    # Suppress noisy third-party network loggers (httpx, etc.) that flood the console
    for noisy in ("httpx", "httpcore", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

# Setup pretty logging initially (will be updated after args parse)
setup_pretty_logging()
logger = logging.getLogger("BananaBread-Emb")

# Warn about Python pre-release versions (known Pydantic compatibility issues)
if sys.version_info.releaselevel != 'final':
    logger.warning(
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} "
        f"({sys.version_info.releaselevel}). Pre-release Python versions may cause errors with Pydantic "
        f"(e.g., '_eval_type() got an unexpected keyword argument'). "
        f"Please upgrade to the stable release of Python {sys.version_info.major}.{sys.version_info.minor}."
    )

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
    "qwen_backend": "torch",
    "qwen_compute_dtype": "bfloat16",
    "qwen_onnx_model_path": None,
    "qwen_onnx_provider": "CPUExecutionProvider",
    "qwen_max_length": 8192,
    "model_storage_dir": "./models",
    "hf_model_slug": None,
    "hf_model_revision": None,
    "hf_access_token": None,
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
    "matmul_cast_fp16": False,
    "enable_torch_compile": False,
    "enable_warmup": True,
    "cuda_cache_ttl_enabled": False,
    "log_embeddings": False,
    "seed_management_key": None,
    "seed_user_key": None,
}

CONFIG_CHOICES = {
    "embedding_model": {"mixedbread", "qwen", "hf"},
    "reranking_model": {"mixedbread", "qwen", None},
    "qwen_size": {"0.6B", "4B", "8B"},
    "qwen_backend": {"torch", "torch-bnb-8bit", "torch-bnb-4bit", "onnx-int8"},
    "qwen_compute_dtype": {"bfloat16", "float16", "float32"},
    "quant": {"standard", "ubinary", "int8"},
    "log_level": {"DEBUG", "INFO", "WARNING", "ERROR"},
}


def resolve_config_path(config_path: str | None = None) -> Path:
    """Resolve the config path from CLI/env/default without depending on import cwd."""
    requested = config_path or os.environ.get("BANANABREAD_CONFIG") or CONFIG_FILE
    path = Path(requested).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def normalize_config(config: Dict[str, Any], source: Path) -> Dict[str, Any]:
    """Normalize config keys and ignore invalid null/unknown values before argparse uses them."""
    normalized = {}
    valid_keys = set(DEFAULTS)

    for raw_key, value in config.items():
        key = raw_key.replace("-", "_")
        if key not in valid_keys:
            logger.warning(f"⚠️  Ignoring unknown config key in {source}: {raw_key}")
            continue
        if value is None and DEFAULTS[key] is not None:
            logger.warning(f"⚠️  Ignoring null config value for '{key}' in {source}; using default {DEFAULTS[key]!r}")
            continue
        normalized[key] = value

    return normalized

def create_default_config(config_path: Path) -> Dict[str, Any]:
    """Create default config.json file on first startup"""
    # Only include non-None defaults that users would want to customize
    config_defaults = {k: v for k, v in DEFAULTS.items() if v is not None}
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_defaults, f, indent=4)
        logger.info(f"📝 Created default configuration file: {config_path}")
        return config_defaults
    except Exception as e:
        logger.warning(f"⚠️  Failed to create config file: {e}")
        return {}

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if it doesn't exist"""
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if not isinstance(config, dict):
                    logger.warning(f"⚠️  Config file must contain a JSON object: {config_path}")
                    return {}
                logger.info(f"📄 Loaded configuration from {config_path}")
                return normalize_config(config, config_path)
        except Exception as e:
            logger.warning(f"⚠️  Failed to load config file {config_path}: {e}")
            return {}
    else:
        return create_default_config(config_path)


def validate_args(parsed_args):
    """Validate config-derived values because argparse choices do not validate defaults."""
    for key, choices in CONFIG_CHOICES.items():
        value = getattr(parsed_args, key, None)
        if value not in choices:
            logger.warning(f"⚠️  Invalid value for '{key}': {value!r}. Using default {DEFAULTS[key]!r}")
            setattr(parsed_args, key, DEFAULTS[key])

    for key in ("embedding_device", "rerank_device"):
        value = getattr(parsed_args, key, None)
        if not isinstance(value, str) or not value.strip():
            logger.warning(f"⚠️  Invalid value for '{key}': {value!r}. Using default {DEFAULTS[key]!r}")
            setattr(parsed_args, key, DEFAULTS[key])
        else:
            setattr(parsed_args, key, value.strip().lower())

    active_qwen_devices = []
    reranking_model = parsed_args.reranking_model
    if reranking_model is None and parsed_args.embedding_model == "qwen":
        reranking_model = "qwen"
    if parsed_args.embedding_model == "qwen":
        active_qwen_devices.append(parsed_args.embedding_device)
    if reranking_model == "qwen":
        active_qwen_devices.append(parsed_args.rerank_device)
    if parsed_args.qwen_backend.startswith("torch-bnb") and any(device == "cpu" for device in active_qwen_devices):
        logger.warning(
            "⚠️  qwen_backend uses bitsandbytes, but at least one active Qwen device is CPU. "
            "Falling back to qwen_backend='torch' instead of attempting CUDA quantized loading."
        )
        parsed_args.qwen_backend = "torch"

    if not getattr(parsed_args, "enable_warmup", True):
        parsed_args.disable_warmup = True
    if getattr(parsed_args, "disable_warmup", False):
        parsed_args.enable_warmup = False

    return parsed_args

CPU_INFO = get_cpu_info()
AVAILABLE_SOCKETS = get_available_cores()
TOTAL_PHYSICAL_CORES = multiprocessing.cpu_count()

def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=os.environ.get("BANANABREAD_CONFIG", CONFIG_FILE))
    pre_args, _ = pre_parser.parse_known_args()
    config_path = resolve_config_path(pre_args.config)

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
    parser.add_argument("--config", type=str, default=str(config_path),
                       help=f"Path to config JSON file (default: {CONFIG_FILE}, or BANANABREAD_CONFIG)")

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
    parser.add_argument("--embedding-model", type=str, choices=['mixedbread', 'qwen', 'hf'], default=DEFAULTS["embedding_model"],
                       help=f"Embedding model to use (default: {DEFAULTS['embedding_model']})")
    parser.add_argument("--qwen-size", type=str, choices=['0.6B', '4B', '8B'], default=DEFAULTS["qwen_size"],
                       help=f"Qwen model size to use when --embedding-model=qwen (default: {DEFAULTS['qwen_size']})")
    parser.add_argument("--qwen-backend", type=str, choices=['torch', 'torch-bnb-8bit', 'torch-bnb-4bit', 'onnx-int8'], default=DEFAULTS["qwen_backend"],
                       help=f"Qwen runtime backend (default: {DEFAULTS['qwen_backend']})")
    parser.add_argument("--qwen-compute-dtype", type=str, choices=['bfloat16', 'float16', 'float32'], default=DEFAULTS["qwen_compute_dtype"],
                       help=f"Compute dtype for torch Qwen backends (default: {DEFAULTS['qwen_compute_dtype']})")
    parser.add_argument("--qwen-onnx-model-path", type=str, default=DEFAULTS.get("qwen_onnx_model_path"),
                       help="Local ONNX model directory or .onnx file for --qwen-backend=onnx-int8")
    parser.add_argument("--qwen-onnx-provider", type=str, default=DEFAULTS["qwen_onnx_provider"],
                       help=f"ONNX Runtime execution provider (default: {DEFAULTS['qwen_onnx_provider']})")
    parser.add_argument("--qwen-max-length", type=int, default=DEFAULTS["qwen_max_length"],
                       help=f"Maximum token length for Qwen embedding inputs (default: {DEFAULTS['qwen_max_length']})")
    parser.add_argument("--qwen-flash-attention", action='store_true', default=DEFAULTS.get("qwen_flash_attention"),
                       help="Enable flash_attention_2 for Qwen models (requires compatible GPU)")
    parser.add_argument("--matmul-cast-fp16", action='store_true', default=DEFAULTS.get("matmul_cast_fp16"),
                       help="Cast compute dtype to float16 during bitsandbytes 8-bit quantization instead of keeping bfloat16 (default: False)")
    parser.add_argument("--model-storage-dir", type=str, default=DEFAULTS["model_storage_dir"],
                       help=f"Directory for explicitly downloaded model snapshots (default: {DEFAULTS['model_storage_dir']})")
    parser.add_argument("--hf-model-slug", type=str, default=DEFAULTS.get("hf_model_slug"),
                       help="Hugging Face model slug to use when --embedding-model=hf, e.g. sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--hf-model-revision", type=str, default=DEFAULTS.get("hf_model_revision"),
                       help="Optional Hugging Face model revision for --embedding-model=hf")
    parser.add_argument("--hf-access-token", type=str, default=DEFAULTS.get("hf_access_token"),
                       help="Hugging Face access token for private or gated models")

    # Reranking model selection arguments
    parser.add_argument("--reranking-model", type=str, choices=['mixedbread', 'qwen'], default=DEFAULTS.get("reranking_model"),
                       help="Reranking model to use (default: mixedbread, or qwen if --embedding-model=qwen)")

    # Determinism
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                        help="Random seed for reproducibility. Set to -1 for random seed. (default: 42)")

    # Seeding
    parser.add_argument("--seed-management-key", type=str, default=DEFAULTS.get("seed_management_key"),
                        help="If set, automatically configure the management key on first startup when api_keys.json does not exist")
    parser.add_argument("--seed-user-key", type=str, default=DEFAULTS.get("seed_user_key"),
                        help="If set, override the auto-generated user API key on first startup when api_keys.json does not exist")

    # Load configuration and apply to parser defaults
    config = load_config(config_path)
    if config:
        parser.set_defaults(**config)
        logger.info("⚙️  Applied configuration defaults from file")

    args, _ = parser.parse_known_args()
    args.config = str(resolve_config_path(args.config))
    return validate_args(args)

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
