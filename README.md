# BananaBread-Emb

A local embedding, reranking, and classification server that speaks the same API as OpenAI, Ollama, and llama.cpp. Drop it in front of tools like SillyTavern and they won't know the difference. Works with [Lumiverse](https://github.com/prolix-oc/Lumiverse) as well!

## What does it do?

BananaBread downloads a specialized model to your machine and runs a local server that turns text into **embeddings** (numeric representations used for search and similarity), **reranks** search results by relevance, and **classifies** text by emotion. It runs on CPU out of the box, with optional GPU acceleration.

**Supported models:**
| Model | Type | Notes |
|---|---|---|
| [MixedBread mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) | Embedding | Default. Lightweight, great on CPU. |
| [MixedBread mxbai-rerank-base-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) | Reranking | Default reranker. |
| [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Embedding + Reranking | One model does both. Sizes: 0.6B, 4B, 8B. |
| [RoBERTa go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) | Classification | Emotion classification. Loaded on first use. |

---

## Prerequisites

- **Python 3.12 or 3.13** (3.14 stable also works — see [Troubleshooting](#troubleshooting) if you hit issues)
- **uv** (recommended) — install it from [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/)

> **Why uv?** It handles virtual environments, dependency resolution, and precompiled binary downloads automatically. You _can_ use pip, but you'll need to manage more things yourself — especially on Windows with GPU support.

---

## Quick Start

### 1. Clone and run

```bash
git clone https://github.com/prolix-oc/BananaBread
cd BananaBread
uv run bananabread-emb
```

That's it. On the first run, uv will:
1. Create a virtual environment
2. Install all dependencies
3. Download the default models from HuggingFace
4. Start the server on **http://localhost:8008**

### 2. Set up API keys

On first launch, BananaBread creates an `api_keys.json` file with a generated key. You can also create it yourself beforehand:

```json
{
    "user": ""
}
```

Leave the value empty and BananaBread will fill in a random key on startup. You can add multiple users:

```json
{
    "alice": "",
    "bob": "",
    "shared_service": ""
}
```

Each empty value gets a unique key generated automatically. Use the key as a Bearer token for authenticated endpoints (see [API Endpoints](#api-endpoints)).

If you want predictable keys on first startup without manually editing `api_keys.json`, set them in `config.json`:

```json
{
    "seed_management_key": "your-management-key",
    "seed_user_key": "your-user-key"
}
```

These are only applied when `api_keys.json` does not exist. Once the file is created, changing these values in `config.json` has no effect.

### 3. Test it

Once the server is running, open **http://localhost:8008/docs** in your browser to see the interactive API docs, or send a quick test request:

```bash
curl -X POST http://localhost:8008/embedding \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello world!"}'
```

---

## Configuration

BananaBread can be configured via command-line flags or a `config.json` file (created automatically on first run). Flags override the config file.

### Common flags

```bash
# See all available flags
uv run bananabread-emb -h

# Increase the embedding cache (keeps vectors in memory for faster repeat lookups)
uv run bananabread-emb --cache-limit 4096  # 4 GB, default is 1024 MB

# Limit CPU core usage
uv run bananabread-emb --use-cores 8

# Change log verbosity
uv run bananabread-emb --log-level DEBUG
```

### Using Qwen models

Qwen can handle both embedding and reranking with a single model, which saves memory:

```bash
# Use Qwen for everything (embedding + reranking, shares one model)
uv run bananabread-emb --embedding-model qwen

# Pick a larger Qwen model (0.6B, 4B, or 8B)
uv run bananabread-emb --embedding-model qwen --qwen-size 4B

# Load Qwen with CUDA 8-bit weight quantization
uv run --extra cuda-quant bananabread-emb --embedding-model qwen --qwen-size 4B --embedding-device cuda --qwen-backend torch-bnb-8bit

# Run a pre-exported INT8 ONNX Qwen model on CPU
uv run --extra onnx bananabread-emb --embedding-model qwen --qwen-backend onnx-int8 --qwen-onnx-model-path ./models/qwen3-embedding-4b-int8-onnx

# Mix and match: Qwen embeddings, MixedBread reranking
uv run bananabread-emb --embedding-model qwen --reranking-model mixedbread
```

`qwen_backend` controls how the model itself runs. `quant` controls the returned embedding vectors after inference. For example, `qwen_backend=onnx-int8` uses an INT8 ONNX model, while `quant=int8` returns scalar-int8 embedding vectors to clients.

### Using a Hugging Face embedding model

You can run a SentenceTransformers or embedding-capable Hugging Face model by slug. BananaBread resolves and validates the Hub metadata, downloads the snapshot into `model_storage_dir`, then loads the local snapshot with SentenceTransformers:

```bash
uv run bananabread-emb --embedding-model hf --hf-model-slug sentence-transformers/all-MiniLM-L6-v2
```

Use `--hf-model-revision` to pin a branch, tag, or commit SHA. Use `--hf-access-token` for private or gated models:

```bash
uv run bananabread-emb --embedding-model hf --hf-model-slug org/private-model --hf-access-token hf_...
```

Existing `mixedbread` and `qwen` run commands keep their standard behavior.

### Config file

BananaBread creates a `config.json` in the current working directory on first run. You can also copy `config.default.json` to `config.json` and edit it. The keys use the same names as the flags (with underscores instead of dashes). Hyphenated keys are also accepted and normalized. Command-line flags override config file values.

If your launcher starts BananaBread from a different working directory, pass the config path explicitly:

```bash
uv run bananabread-emb --config /path/to/config.json
```

You can also set `BANANABREAD_CONFIG=/path/to/config.json`.

All available options (grouped by purpose):

| Option | Default | Description |
|---|---|---|
| `config` | `"config.json"` | Path to the config JSON file, or set `BANANABREAD_CONFIG` |
| **Model selection** | | |
| `embedding_model` | `"mixedbread"` | `"mixedbread"`, `"qwen"`, or `"hf"` |
| `reranking_model` | `null` | `"mixedbread"`, `"qwen"`, or `null` (auto: matches embedding model) |
| `qwen_size` | `"0.6B"` | Qwen model size: `"0.6B"`, `"4B"`, or `"8B"` |
| `qwen_backend` | `"torch"` | Qwen runtime: `"torch"`, `"torch-bnb-8bit"`, `"torch-bnb-4bit"`, or `"onnx-int8"` |
| `qwen_compute_dtype` | `"bfloat16"` | Compute dtype for torch Qwen backends: `"bfloat16"`, `"float16"`, or `"float32"` |
| `qwen_onnx_model_path` | `null` | Local `.onnx` file or directory for `qwen_backend="onnx-int8"` |
| `qwen_onnx_provider` | `"CPUExecutionProvider"` | ONNX Runtime execution provider |
| `qwen_max_length` | `8192` | Maximum token length for Qwen embedding inputs |
| `qwen_flash_attention` | `false` | Enable Flash Attention 2 for Qwen models |
| `matmul_cast_fp16` | `false` | Cast compute dtype to `float16` during bitsandbytes 8-bit quantization (suppresses bfloat16→fp16 warning) |
| `model_storage_dir` | `"./models"` | Directory used by `/v1/models/download` for explicit model snapshots |
| `hf_model_slug` | `null` | Hugging Face repo id used when `embedding_model="hf"` |
| `hf_model_revision` | `null` | Optional branch, tag, or commit SHA for `hf_model_slug` |
| `hf_access_token` | `null` | Optional Hugging Face token for private or gated model metadata/downloads |
| **Device placement** | | |
| `embedding_device` | `"cpu"` | `"cpu"`, `"cuda"`, `"cuda:0"`, etc. |
| `rerank_device` | `"cpu"` | Same options as above |
| **Embedding cache** | | |
| `cache_limit` | `1024` | Max cache size in MB for each cache (embedding and rerank) |
| `quant` | `"standard"` | Embedding quantization: `"standard"`, `"int8"`, or `"ubinary"` |
| `embedding_dim` | `1024` | Embedding dimensions (MixedBread only, truncation) |
| **CPU / threading** | | |
| `use_cores` | `null` | Limit to N CPU cores (`null` = use all) |
| `cpu_socket` | `null` | Pin to a specific CPU socket (multi-socket systems) |
| `embedding_threads` | `null` | Thread count for embedding ops (`null` = auto) |
| `rerank_threads` | `null` | Thread count for rerank ops (`null` = auto) |
| `classification_threads` | `null` | Thread count for classification ops (`null` = auto) |
| `general_threads` | `null` | Thread count for general ops (`null` = auto) |
| **GPU concurrency** | | |
| `num_concurrent_embedding` | `1` | Number of embedding model instances on GPU |
| `num_concurrent_rerank` | `1` | Number of rerank model instances on GPU |
| **CUDA cache management** | | |
| `cuda_cache_ttl_enabled` | `false` | Enable automatic VRAM cleanup when idle |
| `cuda_cache_ttl` | `300` | Seconds of idle time before clearing VRAM |
| `cuda_min_clear_interval` | `60` | Minimum seconds between VRAM clears |
| `cuda_memory_threshold` | `80` | Only clear if VRAM usage exceeds this % |
| **Torch compilation** | | |
| `enable_torch_compile` | `false` | Enable `torch.compile()` (PyTorch 2.0+) |
| `torch_compile_mode` | `"default"` | `"default"`, `"reduce-overhead"`, or `"max-autotune"` |
| `torch_compile_backend` | `"inductor"` | Compilation backend |
| **Warmup** | | |
| `enable_warmup` | `true` | Run warmup inference on startup |
| `warmup_samples` | `3` | Number of warmup inferences per model |
| **Logging** | | |
| `log_level` | `"INFO"` | `"DEBUG"`, `"INFO"`, `"WARNING"`, or `"ERROR"` |
| `log_embeddings` | `false` | Log embedding queries and results to `embeddings.log` |
| **Miscellaneous** | | |
| `seed` | `65` | Random seed for reproducibility (`-1` for random) |
| **Seeding (first startup only)** | | |
| `seed_management_key` | `null` | Auto-configure the management key when `api_keys.json` does not exist |
| `seed_user_key` | `null` | Override the auto-generated user API key when `api_keys.json` does not exist |

---

## GPU Setup (Optional)

If you have an NVIDIA GPU, you can run models on it for significantly faster inference.

### Basic GPU usage

```bash
# Run both models on GPU
uv run bananabread-emb --embedding-device cuda --rerank-device cuda

# Or a specific GPU
uv run bananabread-emb --embedding-device cuda:0 --rerank-device cuda:1
```

### Flash Attention 2

Flash Attention 2 makes GPU inference faster and uses less memory. It's optional — everything works without it, just a bit slower on GPU.

**Requires:** NVIDIA GPU with compute capability 7.5+ (RTX 2000 series or newer).

#### Installing a prebuilt wheel

Use the helper script to install only the precompiled wheel for your current platform and Python version:

```bash
uv run python install_flash_attn.py
```

Preview what it would install:

```bash
uv run python install_flash_attn.py --dry-run
```

The script detects your platform, Python version, and PyTorch setup, then installs the right precompiled wheel. This avoids declaring every platform wheel in `pyproject.toml`, which can make uv download multiple large Flash Attention wheels while resolving the project.

If you're using pip outside uv, run `python install_flash_attn.py` from the cloned repo. **Do not** run `pip install flash-attn` on Windows; it will try to compile from source and almost certainly fail.

You can also install a wheel directly:

```bash
# Windows + Python 3.13
pip install https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.8.3+cu130torch2.9.1cxx11abiTRUE-cp313-cp313-win_amd64.whl

# Windows + Python 3.12
pip install https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/resolve/main/flash_attn-2.8.3+cu130torch2.9.1cxx11abiTRUE-cp312-cp312-win_amd64.whl
```

Once installed, enable it:

```bash
uv run bananabread-emb --embedding-model qwen --qwen-flash-attention --embedding-device cuda
```

At startup, BananaBread only enables Flash Attention 2 when `flash_attn` imports in the same Python environment, Transformers reports FA2 support, CUDA is available, the target device is not CPU, and Qwen compute dtype is `float16` or `bfloat16`. If any check fails, BananaBread logs the reason and falls back to the default attention implementation.

> **Windows + bitsandbytes users:** Flash Attention 2 is automatically disabled when using `torch-bnb-8bit` or `torch-bnb-4bit` on Windows because the combination triggers a known CUDA compatibility issue (`torch.AcceleratorError: CUDA error: unknown error`). BananaBread falls back to SDPA attention automatically.

### Qwen CUDA quantization

Qwen supports optional bitsandbytes weight quantization on CUDA. This reduces model VRAM usage; it is separate from `quant`, which quantizes the returned embedding vectors.

```bash
# Safer VRAM reduction
uv run --extra cuda-quant bananabread-emb --embedding-model qwen --qwen-size 4B --embedding-device cuda --qwen-backend torch-bnb-8bit

# Lower VRAM, more quality-sensitive
uv run --extra cuda-quant bananabread-emb --embedding-model qwen --qwen-size 4B --embedding-device cuda --qwen-backend torch-bnb-4bit
```

For quantized CUDA backends, keep `num_concurrent_embedding` and `num_concurrent_rerank` at `1` unless you have enough VRAM for multiple model copies.

You can also set these options in `config.json`:

```json
{
    "embedding_model": "qwen",
    "qwen_size": "4B",
    "qwen_backend": "torch-bnb-8bit",
    "qwen_compute_dtype": "bfloat16",
    "matmul_cast_fp16": false,
    "embedding_device": "cuda"
}
```

For the 4-bit NF4 backend, change `qwen_backend` to `"torch-bnb-4bit"`. Set `matmul_cast_fp16` to `true` if you want to use `float16` compute during 8-bit quantization instead of keeping `bfloat16` (this avoids the bfloat16→float16 cast warning entirely).

### Qwen ONNX INT8

For CPU deployments, export and quantize Qwen to ONNX ahead of time, then point BananaBread at the local model file or directory:

```bash
uv run --extra onnx bananabread-emb --embedding-model qwen --qwen-size 4B --qwen-backend onnx-int8 --qwen-onnx-model-path ./models/qwen3-embedding-4b-int8-onnx
```

The ONNX backend runs the model through ONNX Runtime, then applies the same last-token pooling and normalization as the torch backend.

---

## API Endpoints

BananaBread exposes multiple API formats so it can slot into different tools without adapter code.

### Authentication

Endpoints under `/v1/` require a Bearer token (from `api_keys.json`):
```
Authorization: Bearer <your-api-key>
```

Ollama and llama.cpp-compatible endpoints do **not** require authentication.

### Endpoint reference

| Endpoint | Auth | Format | Description |
|---|---|---|---|
| `POST /v1/embeddings` | Yes | OpenAI | Generate embeddings (single or batch) |
| `POST /v1/rerank` | Yes | Custom | Rerank documents against a query |
| `POST /v1/classify` | Yes | Custom | Classify text by emotion |
| `POST /v1/models/download` | Yes | Custom | Resolve, inspect, and optionally download a Hugging Face model snapshot |
| `POST /v1/models` | Yes | OpenAI | List loaded models |
| `POST /v1/health` | Yes | Custom | Health check |
| `POST /v1/memory` | Yes | Custom | Memory and thread usage stats |
| `POST /api/embeddings` | No | Ollama | Ollama-compatible embedding |
| `POST /api/embed` | No | Ollama | Ollama-compatible embedding (alt) |
| `POST /embedding` | No | llama.cpp | llama.cpp-compatible embedding |
| `GET /` | No | — | Server info and status |

Full interactive docs are available at **http://localhost:8008/docs** (Swagger) and **http://localhost:8008/redoc** (ReDoc) while the server is running.

### Management endpoints (optional)

These endpoints require the **management key** (set via `seed_management_key` in config, or through the web admin panel). If no management key is configured, these endpoints return `403`.

| Endpoint | Method | Description |
|---|---|---|
| `GET /v1/management/config` | GET | View current tenants, tiers, limits, and cache config |
| `PATCH /v1/management/config` | PATCH | Update management key, default limits, tiers, or cache settings |
| `POST /v1/management/users` | POST | Create a new user with optional tier and custom limits |
| `GET /management` | GET | Web admin panel for managing users, tiers, and cache settings |

Token limits are enforced per user and automatically reset at daily and weekly intervals. Limits can be set at three levels (most specific wins):

1. **Per-user limits** — set when creating a user or via the admin panel
2. **Tier limits** — shared limits for a group of users
3. **Default limits** — fallback for all users without overrides

Cache scope can be `global` (default, shared across all users) or `per_user` (each user gets an isolated cache with its own limit).

### Request examples

#### Embeddings (OpenAI format)
```bash
curl -X POST http://localhost:8008/v1/embeddings \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "Another sentence"],
    "model": "mixedbread-ai/mxbai-embed-large-v1"
  }'
```

#### Reranking
```bash
curl -X POST http://localhost:8008/v1/rerank \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of AI.",
      "Python is a programming language.",
      "Neural networks learn from data."
    ],
    "top_k": 2
  }'
```

#### Embeddings (llama.cpp format, no auth)
```bash
curl -X POST http://localhost:8008/embedding \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Hello world!",
    "normalize": true
  }'
```

#### Classification
```bash
curl -X POST http://localhost:8008/v1/classify \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "I am so happy today!",
    "top_k": 3,
    "sorted": true
  }'
```

#### Download a Hugging Face embedding model
```bash
curl -X POST http://localhost:8008/v1/models/download \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "author": "sentence-transformers",
    "path": "all-MiniLM-L6-v2"
  }'
```

The downloader accepts `author` + `path`, a direct repo id in `path`, or standard selectors such as `{"model_name":"qwen","size":"0.6B"}`. It checks Hub metadata for SentenceTransformers or embedding capability before downloading unless `require_embedding_capable` is set to `false`.

For private or gated models, pass `hf_access_token` in the request body or configure `hf_access_token`/`--hf-access-token` on the server. The token is used for Hub metadata and snapshot download calls and is not returned in the response.

### Python example

```python
import requests

API_KEY = "your-api-key-from-api_keys.json"
BASE_URL = "http://localhost:8008"

# Embeddings
response = requests.post(f"{BASE_URL}/v1/embeddings",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"input": ["Hello world"], "model": "mixedbread-ai/mxbai-embed-large-v1"}
)
embeddings = response.json()["data"][0]["embedding"]
print(f"Embedding dimensions: {len(embeddings)}")

# Reranking
response = requests.post(f"{BASE_URL}/v1/rerank",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "query": "What is AI?",
        "documents": ["AI is artificial intelligence.", "The sky is blue."],
        "top_k": 1
    }
)
print(response.json())
```

---

## Integration with Other Tools

BananaBread is designed to impersonate other servers. It responds to model listing requests and formats responses to match what client software expects.

- **SillyTavern**: Point it at `http://localhost:8008` as a llama.cpp backend. BananaBread responds to the models endpoint, so SillyTavern will treat it as a full llama.cpp instance.
- **OpenAI-compatible clients**: Use the `/v1/embeddings` endpoint with your API key.
- **Ollama-compatible clients**: Use `/api/embeddings` or `/api/embed` — no authentication needed.

---

## Troubleshooting

### Flash Attention fails to install on Windows

`pip install flash-attn` tries to compile from source on Windows and will usually fail. Use the helper script instead:

```bash
uv run python install_flash_attn.py
```

See [Flash Attention 2](#flash-attention-2) for more details.

### `_eval_type() got an unexpected keyword argument 'prefer_fwd_module'`

This happens when running a **pre-release** version of Python 3.14 (alpha, beta, or release candidate). Pydantic relies on internal Python APIs that changed between the RC and the final release.

**Fix:** Upgrade to the stable release of Python 3.14.0 (or use Python 3.12/3.13).

### Models are slow on first request

This is normal. BananaBread runs warmup inference on startup by default, but the first real request may still be slightly slower as caches warm up. Subsequent requests will be faster. You can control warmup with `--warmup-samples` or disable it with `--disable-warmup`.

---

## License

MIT
