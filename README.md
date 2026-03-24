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

# Mix and match: Qwen embeddings, MixedBread reranking
uv run bananabread-emb --embedding-model qwen --reranking-model mixedbread
```

### Config file

BananaBread creates a `config.json` on first run. You can also copy `config.default.json` to `config.json` and edit it. The keys use the same names as the flags (with underscores instead of dashes). Command-line flags override config file values.

All available options (grouped by purpose):

| Option | Default | Description |
|---|---|---|
| **Model selection** | | |
| `embedding_model` | `"mixedbread"` | `"mixedbread"` or `"qwen"` |
| `reranking_model` | `null` | `"mixedbread"`, `"qwen"`, or `null` (auto: matches embedding model) |
| `qwen_size` | `"0.6B"` | Qwen model size: `"0.6B"`, `"4B"`, or `"8B"` |
| `qwen_flash_attention` | `false` | Enable Flash Attention 2 for Qwen models |
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

#### Installing with uv (recommended)

If you're running from the cloned repo, uv will automatically download a precompiled wheel for your platform:

```bash
uv run --extra flash-attn bananabread-emb --embedding-model qwen --qwen-flash-attention --embedding-device cuda
```

#### Installing without uv

If you're using pip, **do not** run `pip install flash-attn` on Windows — it will try to compile from source and almost certainly fail. Use the included helper script instead:

```bash
python install_flash_attn.py
```

The script detects your platform, Python version, and PyTorch setup, then installs the right precompiled wheel. Run `python install_flash_attn.py --dry-run` to preview what it would do.

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
| `POST /v1/models` | Yes | OpenAI | List loaded models |
| `POST /v1/health` | Yes | Custom | Health check |
| `POST /v1/memory` | Yes | Custom | Memory and thread usage stats |
| `POST /api/embeddings` | No | Ollama | Ollama-compatible embedding |
| `POST /api/embed` | No | Ollama | Ollama-compatible embedding (alt) |
| `POST /embedding` | No | llama.cpp | llama.cpp-compatible embedding |
| `GET /` | No | — | Server info and status |

Full interactive docs are available at **http://localhost:8008/docs** (Swagger) and **http://localhost:8008/redoc** (ReDoc) while the server is running.

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
python install_flash_attn.py
```

Or install with `uv`, which automatically pulls precompiled wheels:

```bash
uv run --extra flash-attn bananabread-emb
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
