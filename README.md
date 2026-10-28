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
| [NVIDIA Llama Nemotron Embed 1B v2](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) | Embedding | Multilingual retrieval, 8K-token inputs, and Matryoshka dimensions (384–2048). |
| [NVIDIA Nemotron-3 Embed](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) | Embedding | Multilingual retrieval. Official BF16 checkpoints in 1B and 8B sizes, 32K-token inputs, and 2,048-dimensional Matryoshka embeddings. |
| [NVIDIA Nemotron-3.5 Embed 1B](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16) / [8B](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16) | Embedding | Text + image documents, text queries, 32K-token inputs. Matryoshka embeddings up to 2,048 dimensions (1B) or 4,096 (8B). |
| [Jina Embeddings v5 Omni Small](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small) | Embedding | Multimodal (text, image, audio, video), 1024-dim, ~1.74B params, 32K tokens. |
| [Jina Embeddings v5 Omni Nano](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano) | Embedding | Multimodal (text, image, audio, video), 768-dim, ~1.04B params, 8K tokens. |
| [NVIDIA Llama Nemotron Embed VL 1B v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) | Embedding | Text + image retrieval, 2048-dim, 10240-token budget with image tiles. |
| [Microsoft Harrier OSS v1 0.6B](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | Embedding | Multilingual text retrieval, 1024-dim, 32K tokens, last-token pooling. |
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

BananaBread uses Hugging Face's Xet downloader automatically when it is
available. To explicitly select the chunked Xet path for all supplied and
custom models, start the server with:

```bash
uv run bananabread-emb --hf-download-backend xet
```

Set `hf_download_backend` to `"xet"` in `config.json` for the same behavior.
`"http"` forces the standard HTTP downloader, while `"auto"` (the default) honors the Hugging Face environment and installed capabilities. Hub files
that do not expose Xet metadata still download over HTTP.

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

To allow API requests without a key, add this setting to `config.json` and restart the server:

```json
{
    "disable_api_key_auth": true
}
```

You can also launch with `uv run bananabread-emb --disable-api-key-auth`. Authentication is enabled by default. With this setting enabled, API requests ignore the Authorization header and use a shared `anonymous` user, created automatically in `api_keys.json` if needed. Token usage, token limits, and per-user cache settings apply to that shared user. Endpoints that require the management key still require authentication.

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

### Embedding inference memory

All embedding models and OpenAI, Ollama, and llama.cpp embedding routes split requests into inference batches of at most **8 inputs** and **8,192 padded tokens** by default. Configure these limits with `--embedding-batch-size` / `--embedding-batch-max-tokens`, or `embedding_batch_size` / `embedding_batch_max_tokens` in `config.json`. Both must be positive integers. The planner measures each text with its tokenizer or processor, including retrieval prompts, truncation, and padding. Media and models without a measurable tokenizer run individually.

The token budget limits batching, not document length: an input over budget runs alone with the model's existing context limit. Response order is preserved. Smaller limits reduce activation memory even with quantized weights; concurrent model instances each have their own budget.

Compatible Transformers encoders use a compact, broadcastable key padding mask for plain bidirectional SDPA attention. Causal, sliding-window, packed, and other custom patterns retain the standard mask implementation. This optimization applies by capability across embedding models, including their multimodal backbones, and does not require FlashAttention.

CUDA inference defaults to `expandable_segments:True` with no block split limit, allowing the allocator to reuse memory across changing batch shapes. Explicit `PYTORCH_ALLOC_CONF` or `PYTORCH_CUDA_ALLOC_CONF` settings are preserved. GPU monitors include the allocator's reserved memory, which can exceed the memory occupied by active tensors; changing allocator settings requires restarting the process.

### Using Qwen models

Qwen can handle both embedding and reranking with a single model, which saves memory:

```bash
# Use Qwen for everything (embedding + reranking, shares one model)
uv run bananabread-emb --embedding-model qwen

# Pick a larger Qwen model (0.6B, 4B, or 8B)
uv run bananabread-emb --embedding-model qwen --qwen-size 4B

# Load Qwen with GPU 8-bit weight quantization
uv run --extra cuda-quant bananabread-emb --embedding-model qwen --qwen-size 4B --embedding-device cuda --qwen-backend torch-bnb-8bit

# Load Qwen with 8-bit Metal weight quantization on Apple Silicon
uv run --extra metal-quant bananabread-emb --embedding-model qwen --qwen-size 4B --embedding-device mps --qwen-backend torch-metal-8bit

# Run a pre-exported INT8 ONNX Qwen model on CPU
uv run --extra onnx bananabread-emb --embedding-model qwen --qwen-backend onnx-int8 --qwen-onnx-model-path ./models/qwen3-embedding-4b-int8-onnx

# Mix and match: Qwen embeddings, MixedBread reranking
uv run bananabread-emb --embedding-model qwen --reranking-model mixedbread
```

`qwen_backend` controls how the model itself runs. `quant` controls the returned embedding vectors after inference. For example, `qwen_backend=onnx-int8` uses an INT8 ONNX model, while `quant=int8` returns scalar-int8 embedding vectors to clients.

### Using NVIDIA Nemotron Embedding

NVIDIA's Llama Nemotron Embed 1B v2 is a 2048-dimensional multilingual retrieval model with an 8192-token input limit. Start it with:

```bash
uv run bananabread-emb --embedding-model nemotron --embedding-device cuda --embedding-dim 2048
```

The model uses a distinct retrieval prompt for each side of a search. BananaBread applies `passage:` for the default `document` input type and `query:` when you set the BananaBread extension `input_type` to `query` on `POST /v1/embeddings`. Use the same `embedding_dim` for indexing and querying; supported Matryoshka sizes are 384, 512, 768, 1024, and 2048.

```bash
# Index a corpus document (the default input_type is document)
curl -X POST http://localhost:8008/v1/embeddings \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"input":"A summit is the highest point of a mountain.","input_type":"document"}'

# Embed a search query
curl -X POST http://localhost:8008/v1/embeddings \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"input":"summit define","input_type":"query"}'
```

Nemotron requires NVIDIA's custom bidirectional-Llama implementation, so this dedicated model selector enables `trust_remote_code` only for NVIDIA's fixed model repository. Review the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) before use.

For lower accelerator-memory use, the dedicated backends quantize the model weights with bitsandbytes on NVIDIA CUDA or AMD ROCm, or with Metal kernels on Apple Silicon (separate from the response-vector `quant` option):

```bash
# 8-bit weights
uv run --extra cuda-quant bananabread-emb --embedding-model nemotron --nemotron-backend torch-bnb-8bit --embedding-device cuda --reranking-model none

# 4-bit NF4 weights with BF16 compute
uv run --extra cuda-quant bananabread-emb --embedding-model nemotron --nemotron-backend torch-bnb-4bit --nemotron-compute-dtype bfloat16 --embedding-device cuda --reranking-model none

# Apple Silicon Metal 8-bit
uv run --extra metal-quant bananabread-emb --embedding-model nemotron --nemotron-backend torch-metal-8bit --embedding-device mps --reranking-model none
```

Nemotron defaults to embedding-only mode; use `--reranking-model mixedbread` only when you need reranking too. For any model, `--reranking-model none` disables reranking. The `/v1/rerank` endpoint then returns HTTP 503 instead of loading a reranker. Nemotron also serializes requests that share one model instance to avoid the Hugging Face fast-tokenizer `RuntimeError: Already borrowed`; use `num_concurrent_embedding` greater than one on CUDA to load independent model instances when you need parallel inference.

### Using NVIDIA Nemotron-3 Embed

Nemotron-3 Embed is available from Hugging Face as the official `1B` and `8B` BF16 checkpoints. Both are 2,048-dimensional multilingual retrieval models with a 32,768-token input limit. They run with the saved SentenceTransformers query/document prompts and do not enable remote code.

```bash
# 1B BF16 checkpoint (the default Nemotron-3 size)
uv run bananabread-emb --embedding-model nemotron-3 --nemotron-size 1B --embedding-device cuda --embedding-dim 2048

# 8B BF16 checkpoint
uv run bananabread-emb --embedding-model nemotron-3 --nemotron-size 8B --embedding-device cuda --embedding-dim 2048
```

Use the BananaBread `input_type` extension exactly as for the earlier Nemotron model: `document` (the default) uses the model's saved `passage:` prompt, and `query` uses its saved `query:` prompt. Keep the same `embedding_dim` for corpus indexing and queries. The checkpoints require Transformers 5.2+ and SentenceTransformers 5.4.1+, which BananaBread installs automatically.

### Using NVIDIA Nemotron-3.5 Embed

Nemotron-3.5 Embed supports text, image, and combined image/text documents with text-only retrieval queries. The official BF16 checkpoints have a 32,768-token context, including image tokens. The `1B` model produces up to 2,048 dimensions; `8B` produces up to 4,096. Both load natively without remote code.

```bash
# 1B, full-size vectors
uv run bananabread-emb --embedding-model nemotron-3.5 --nemotron-size 1B --embedding-device cuda --embedding-dim 2048

# 8B, full-size vectors
uv run bananabread-emb --embedding-model nemotron-3.5 --nemotron-size 8B --embedding-device cuda --embedding-dim 4096

# 8B with 4-bit NF4 weights and smaller returned vectors
uv run --extra cuda-quant bananabread-emb --embedding-model nemotron-3.5 --nemotron-size 8B --embedding-device cuda --nemotron-backend torch-bnb-4bit --embedding-dim 1024

# Apple Silicon with Metal 8-bit weights
uv run --extra metal-quant bananabread-emb --embedding-model nemotron-3.5 --embedding-device mps --nemotron-backend torch-metal-8bit
```

The shared `--nemotron-backend` and `--nemotron-compute-dtype` options apply to this series. Reranking is disabled by default; add `--reranking-model mixedbread` if needed. The default size is `1B`, and the default returned dimension is `1024`. Set `embedding_model` to `"nemotron-3.5"` and `nemotron_size` to `"1B"` or `"8B"` in `config.json` for the same behavior. Hub token, revision, storage, and download-backend settings also apply.

On `POST /v1/embeddings`, `input_type="document"` applies the checkpoint's saved `passage:` prompt, including for image-only inputs. `input_type="query"` applies `query:` and accepts text only. Use the [multimodal request format](#multimodal-request-bodies) to send inline images:

```json
{
  "input": [
    "A document containing plain text",
    {"image": "data:image/png;base64,<encoded-image>"},
    {"text": "A caption for this page", "image": "data:image/png;base64,<encoded-image>"}
  ],
  "input_type": "document"
}
```

Each entry produces one vector. Matryoshka truncation supports any positive dimension up to the selected model's native size, with L2 normalization after truncation. Keep the model and dimensions consistent between indexing and querying. BananaBread uses Transformers 5.20.0 and SentenceTransformers 6.1+ for this native pipeline, including the Transformers fixes for FlashAttention 2 image inputs. The models use the [OpenMDW 1.1 license](https://openmdw.ai).

### Using Google EmbeddingGemma 2

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), released October 6, 2026, is Google's 740M-parameter multimodal embedding model. BananaBread loads its text, image, and audio encoders by default and runs the native architecture without remote code. For a smaller text-only deployment, use `--embeddinggemma-modalities text` (270M parameters); `text-image` and `text-audio` omit one media encoder.

```bash
# CPU text embeddings; defaults to FP32 compute and 768 dimensions
uv run bananabread-emb --embedding-model embeddinggemma

# CUDA / ROCm: 4-bit NF4 weights, BF16 compute when the GPU supports it
uv run --extra cuda-quant bananabread-emb --embedding-model embeddinggemma --embedding-device cuda --embeddinggemma-backend torch-bnb-4bit

# Apple Silicon: 4-bit Metal weights (torch-metal-8bit is also available)
uv run --extra metal-quant bananabread-emb --embedding-model embeddinggemma --embedding-device mps --embeddinggemma-backend torch-metal-4bit

# Smaller normalized vectors for a vector database
uv run bananabread-emb --embedding-model embeddinggemma --embedding-dim 256

# Smallest deployment: omit both media encoders
uv run bananabread-emb --embedding-model embeddinggemma --embeddinggemma-modalities text
```

The selector defaults to embedding-only mode. Add `--reranking-model mixedbread` if needed. Existing API formats continue to work. On `/v1/embeddings`, the default `input_type="document"` applies the saved `Document` prompt (`title: none | text: `), and `input_type="query"` applies `SearchQuery` (`task: search result | query: `):

```bash
curl -X POST http://localhost:8008/v1/embeddings \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"input":"How do I bake banana bread?","input_type":"query"}'
```

Supported dimensions are **128, 256, 512, and 768**, with normalization after truncation. The shared `embedding_dim=1024` default maps to 768 for this selector. Use the same dimensions and model for indexing and querying; switching from another model requires re-embedding the corpus. Inputs are truncated to 8,192 tokens including the prompt.

`embeddinggemma_backend` quantizes model weights; the independent `quant` setting quantizes returned vectors. `embeddinggemma_compute_dtype="auto"` uses BF16 on supported accelerators and FP32 elsewhere. You can explicitly choose `bfloat16` or `float32`; GPUs without native BF16 use FP32. **FP16 is unsafe for this model**, so the FP16 compute option, bitsandbytes INT8 backend (which casts activations to FP16), and `matmul_cast_fp16` override are not used by this adapter. NF4 and Metal keep BF16/FP32 compute, and non-finite outputs are rejected.

BananaBread installs Transformers 5.20.0, SentenceTransformers 6.1+, and the image/audio processing dependencies. Quantized backends require the corresponding extra and device; CPU inference uses the standard `torch` backend. `hf_access_token`, `hf_model_revision`, `hf_download_backend`, and `model_storage_dir` also apply to the dedicated loader.

### Multimodal request bodies

`POST /v1/embeddings` accepts structured inputs alongside the existing text format. Each object produces **one vector**, even when it combines text, multiple images, and audio. An array produces one vector per entry, in request order. This shared path works with EmbeddingGemma, Nemotron-3.5, and compatible models loaded through `--embedding-model hf --hf-model-slug ...`, using SentenceTransformers' declared modality support. `GET /v1/models` lists the active model's `input_modalities`.

```json
{
  "input": {
    "text": "A product photo",
    "image": "data:image/png;base64,<encoded-image>"
  },
  "input_type": "document"
}
```

Use an image-only input with `{"input":{"image":"data:image/jpeg;base64,<encoded-image>"}}`. An `image` or `audio` field also accepts a list of bodies, or an object containing raw base64 in `data` and its MIME type in `media_type`:

```json
{
  "input": [
    "Find the sound of a dog barking",
    {"audio": {"data": "<base64-wav-file>", "media_type": "audio/wav"}},
    {"image": "data:image/png;base64,<encoded-image>"}
  ],
  "input_type": "query"
}
```

Audio files (for example WAV or FLAC) are decoded to mono and resampled to the model processor's declared sampling rate when available; EmbeddingGemma uses 16 kHz. Retrieval prefixes apply only to inputs containing text. Media-only inputs receive no text prefix. Model-specific context limits still apply; EmbeddingGemma shares its 8,192-token budget across all modalities.

The JSON extension accepts inline bodies, with limits of 20 MiB per file, 64 MiB of combined media per request, 16 megapixels per image, 360 seconds / 8 channels per audio file, and 32 bodies per modality per input. It does not fetch URLs or read server paths. Malformed or unsupported media returns HTTP 400; invalid JSON/schema shapes return HTTP 422. Video and multipart uploads are not exposed by this extension. Other embedding model processors may require additional dependencies specified by their model cards.

Media affects cache identity and usage limits. Usage uses the model processor's attention mask where available; pure encoders without token masks use 16×16 image patch units or 25 units per audio second. Text keeps the existing token accounting. Logged inputs contain media counts instead of base64 payloads. Float, base64, INT8, and binary vector response options work as before.

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
| `embedding_model` | `"mixedbread"` | `"mixedbread"`, `"qwen"`, `"nemotron"`, `"nemotron-3"`, `"nemotron-3.5"`, `"embeddinggemma"`, or `"hf"` |
| `reranking_model` | `null` | `"mixedbread"`, `"qwen"`, `"none"`, or `null` (auto: Qwen uses Qwen; Nemotron and EmbeddingGemma disable reranking; otherwise MixedBread) |
| `qwen_size` | `"0.6B"` | Qwen model size: `"0.6B"`, `"4B"`, or `"8B"` |
| `qwen_backend` | `"torch"` | Qwen runtime: `"torch"`, `"torch-bnb-8bit"`, `"torch-bnb-4bit"`, `"torch-metal-8bit"`, `"torch-metal-4bit"`, or `"onnx-int8"` |
| `qwen_compute_dtype` | `"bfloat16"` | Compute dtype for torch Qwen backends: `"bfloat16"`, `"float16"`, or `"float32"` |
| `nemotron_backend` | `"torch"` | Nemotron runtime: `"torch"`, `"torch-bnb-8bit"`, `"torch-bnb-4bit"`, `"torch-metal-8bit"`, or `"torch-metal-4bit"` |
| `nemotron_compute_dtype` | `"bfloat16"` | Compute dtype for Nemotron torch backends |
| `nemotron_size` | `"1B"` | Nemotron-3 / Nemotron-3.5 checkpoint size: `"1B"` or `"8B"` |
| `embeddinggemma_backend` | `"torch"` | EmbeddingGemma 2 runtime: `"torch"`, `"torch-bnb-4bit"`, `"torch-metal-4bit"`, or `"torch-metal-8bit"` |
| `embeddinggemma_compute_dtype` | `"auto"` | `"auto"`, `"bfloat16"`, or `"float32"`; FP16 is unsafe for EmbeddingGemma 2 |
| `embeddinggemma_modalities` | `"all"` | Encoders to load: `"all"`, `"text"`, `"text-image"`, or `"text-audio"` |
| `qwen_onnx_model_path` | `null` | Local `.onnx` file or directory for `qwen_backend="onnx-int8"` |
| `qwen_onnx_provider` | `"CPUExecutionProvider"` | ONNX Runtime execution provider |
| `qwen_max_length` | `8192` | Maximum token length for Qwen embedding inputs |
| `qwen_flash_attention` | `false` | Enable Flash Attention 2 for Qwen models |
| `matmul_cast_fp16` | `false` | Cast Qwen and Nemotron compute dtype to `float16` during bitsandbytes 8-bit quantization (suppresses bfloat16→fp16 warning) |
| `model_storage_dir` | `"./models"` | Directory used by `/v1/models/download` for explicit model snapshots |
| `hf_download_backend` | `"auto"` | Hub download backend: `"auto"`, `"xet"`, or `"http"`; applies to supplied and custom models |
| `hf_model_slug` | `null` | Hugging Face repo id used when `embedding_model="hf"` |
| `hf_model_revision` | `null` | Optional branch, tag, or commit SHA for `hf_model_slug` or the dedicated EmbeddingGemma / Nemotron-3.5 loaders |
| `hf_access_token` | `null` | Optional Hugging Face token for private or gated model metadata/downloads |
| **Device placement** | | |
| `embedding_device` | `"cpu"` | `"cpu"`, `"cuda"`, `"cuda:0"`, or `"mps"` |
| `rerank_device` | `"cpu"` | Same options as above |
| **Embedding cache** | | |
| `cache_limit` | `1024` | Max cache size in MB for each cache (embedding and rerank) |
| `quant` | `"standard"` | Embedding quantization: `"standard"`, `"int8"`, or `"ubinary"` |
| `embedding_dim` | `1024` | Output dimensions for MixedBread, Nemotron, EmbeddingGemma, and HF models (truncation where supported); EmbeddingGemma maps the shared default to 768 |
| `embedding_batch_size` | `8` | Maximum inputs per embedding inference batch, for every model |
| `embedding_batch_max_tokens` | `8192` | Maximum padded tokens per batch; longer individual inputs run alone |
| `embedding_flash_attention` | `false` | Prefer FlashAttention 2 on compatible CUDA/HIP embedding backbones; requires `flash_attn` |
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
| **Authentication** | | |
| `disable_api_key_auth` | `false` | Allow API requests without a key using a shared `anonymous` user; management authentication is still required |
| **Seeding (first startup only)** | | |
| `seed_management_key` | `null` | Auto-configure the management key when `api_keys.json` does not exist |
| `seed_user_key` | `null` | Override the auto-generated user API key when `api_keys.json` does not exist |

---

## GPU Setup (Optional)

If you have an NVIDIA or AMD GPU, or an Apple Silicon Mac, you can run models on its accelerator. NVIDIA Turing and newer GPUs work out of the box with the Torch 2.14 + CUDA 13 wheel pinned in `uv.lock`. AMD uses PyTorch's ROCm builds — see [AMD GPUs (ROCm)](#amd-gpus-rocm). Apple Silicon uses PyTorch MPS — see [Apple Silicon (Metal)](#apple-silicon-metal). On GPUs without native bfloat16 support, including Turing, BananaBread automatically changes model compute dtype to float16 for Qwen and Nemotron; EmbeddingGemma uses FP32 because FP16 is unsafe for it.

### Basic GPU usage

```bash
# Run both models on GPU
uv run bananabread-emb --embedding-device cuda --rerank-device cuda

# Or a specific GPU
uv run bananabread-emb --embedding-device cuda:0 --rerank-device cuda:1
```

### Apple Silicon (Metal)

The Torch 2.14 macOS wheels require macOS 14 or newer.

Apple Silicon Macs can run Qwen, Nemotron, Nemotron-3, and EmbeddingGemma on MPS with optional 8-bit or 4-bit affine weight quantization. The Metal kernels keep quantized linear weights on the GPU and are downloaded from Hugging Face on first use. BananaBread loads version 2 of `kernels-community/mlx-quantization-metal-kernels` with `kernels>=0.17`, avoiding Transformers' older version-1 pin and its upgrade warning. Use 8-bit when embedding quality is the priority; 4-bit saves more unified memory but should be evaluated against your retrieval corpus.

```bash
# Qwen 8-bit Metal
uv run --extra metal-quant bananabread-emb --embedding-model qwen --qwen-size 4B --embedding-device mps --qwen-backend torch-metal-8bit

# Qwen 4-bit Metal
uv run --extra metal-quant bananabread-emb --embedding-model qwen --qwen-size 4B --embedding-device mps --qwen-backend torch-metal-4bit

# Nemotron-3 8-bit Metal
uv run --extra metal-quant bananabread-emb --embedding-model nemotron-3 --nemotron-size 1B --embedding-device mps --nemotron-backend torch-metal-8bit --reranking-model none
```

Metal quantization requires an Apple Silicon Mac with MPS available. Flash Attention 2 is not used on Metal; Transformers' MPS attention implementation remains active. Keep model concurrency at `1` unless the Mac has enough unified memory for additional model copies.

### AMD GPUs (ROCm)

PyTorch's ROCm builds expose AMD GPUs through the same `cuda` device API, so every `--embedding-device cuda` / `--rerank-device cuda` flag works unchanged, and BananaBread detects ROCm builds at startup. Linux x86_64 ROCm 7.2 installations can also use the optional Radeon-capable Flash Attention 2 wheels described below; SDPA remains the default. The `cuda-quant` extra retains its existing name but installs bitsandbytes 0.50.1 or newer, whose prebuilt wheels support ROCm 7.2 and the `gfx1100`, `gfx1200`, and `gfx1201` GPU targets.

#### Linux

Run the installer — it rewrites the PyTorch index in `pyproject.toml` to the ROCm index, re-locks, syncs, and verifies the result:

```bash
uv sync --no-install-package torch --no-install-package torchvision # avoid downloading CUDA first
uv run --no-sync python install_rocm_torch.py --dry-run
uv run --no-sync python install_rocm_torch.py
```

The helper always installs into and verifies BananaBread's `.venv`, even if a
shell alias named `python` points at a different interpreter. Its final output
must show a non-empty HIP version, the AMD GPU name, and the `.venv` Python path.

Because the lockfile itself now contains the ROCm build, plain `uv run bananabread-emb --embedding-device cuda` works afterwards — no extra flags. To switch back to NVIDIA CUDA later: `uv run python install_rocm_torch.py --restore-cuda`.

Why the index matters: the installer routes both packages through a named PyTorch index and selects the exact `torch==2.14.1+rocm7.2` and `torchvision==0.29.1+rocm7.2` builds so missing or incompatible ROCm wheels produce a resolution error instead of silently falling back to another backend. Both packages must be switched together: torchvision's `torch==2.14.1` dependency also accepts ROCm torch, even when torchvision itself came from PyPI or a CUDA index. The official ROCm wheels are Linux x86_64. AMD supports the RX 7900 XTX (`gfx1100`), RX 9070 series (`gfx1201`), and RX 9060 series (`gfx1200`) on Ubuntu 24.04.3, Ubuntu 22.04.5, RHEL 10.1, and RHEL 9.7; see the [ROCm 7.2 system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/system-requirements.html#supported-gpus).

The ROCm 7.2 index contains multiple Torch releases. Inspect the full wheel
filename when checking availability; commands that sort the index and use `tail`
only display the last few entries.

#### Windows

Windows AMD users are no longer stuck on CPU: AMD ships an official **PyTorch on Windows** build (ROCm 7.2.1, `torch 2.9.1+rocm7.2.1`). This remains a supported exception to BananaBread's shared Torch 2.14 baseline. The wheels aren't on a pip index, though; they're direct downloads from `repo.radeon.com`, so `install_rocm_torch.py` swaps them into the venv for you:

```powershell
# If you already have a .venv from the NVIDIA/CUDA setup, delete it first.
uv python pin 3.12                       # AMD's wheels are cp312-only
uv sync --no-install-package torch --no-install-package torchvision
uv run --no-sync python install_rocm_torch.py
```

The script installs AMD's ROCm SDK runtime wheels plus matching ROCm torch and torchvision wheels, then verifies torchvision imports and the GPU is visible. Re-running it also repairs torchvision in an existing ROCm environment. To double-check yourself:

```powershell
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.9.1+rocm7.2.1 True
```

Always launch with `uv run --no-sync` on this setup — a plain `uv run` re-syncs torch back to the CUDA wheel from the lockfile:

```powershell
uv run --no-sync bananabread-emb --embedding-device cuda --rerank-device cuda
```

**Requirements:** Windows 11, the [AMD Adrenalin 26.2.2 driver](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html) or newer, Python 3.12, and a supported GPU — Radeon RX 7900 XTX, RX 9070 / 9070 XT / RX 9060 XT, AI PRO R9700, PRO W7900, or Ryzen AI Max / Ryzen AI 300 iGPUs. See AMD's [release notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-2-1.html) for the full compatibility list. If you hit missing-DLL errors, install the [Visual C++ 2015–2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), and per AMD's known-issues list turn off WDAG and Smart App Control.

Other Windows notes from AMD: it's inference-only (fine for BananaBread), and LLM batch size 1 is what's officially supported — keep `num_concurrent_embedding` / `num_concurrent_rerank` low. Full list: [AMD limitations](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsryz.html).

### Flash Attention 2

Flash Attention 2 makes GPU inference faster and uses less memory. It's optional — everything works without it, just a bit slower on GPU.

**Requires:** Torch 2.14.x and Python 3.12 or 3.13, plus either NVIDIA Ampere or newer with CUDA 13.0, or Linux x86_64 with ROCm 7.2 and a supported AMD GPU. NVIDIA Turing GPUs use SDPA. AMD's native Windows Torch 2.9.1 build also continues to use SDPA; these ROCm FA2 wheels are Linux-only.

The helper selects **FA2 2.8.3** for CUDA and the **2.8.4 ROCm build pinned to upstream commit `4a948e9`** for AMD. The ROCm wheel is built with Composable Kernel and includes CDNA targets `gfx90a`, `gfx942`, `gfx950` and Radeon RDNA targets `gfx1100`, `gfx1101`, `gfx1102`, `gfx1150`, `gfx1151`, `gfx1200`, `gfx1201`. Radeon support comes from that upstream commit rather than a released FA2 tag. See the [wheel repository](https://github.com/mjun0812/flash-attention-prebuild-wheels#packages) for build details.

#### Installing a prebuilt wheel

Use the helper script to install only the precompiled wheel for your current platform and Python version:

```bash
uv run --no-sync python install_flash_attn.py
```

Preview what it would install:

```bash
uv run --no-sync python install_flash_attn.py --dry-run
```

Install ROCm torch first with `install_rocm_torch.py` on AMD Linux. On NVIDIA, run `uv sync` to apply the Torch 2.14 upgrade before using the helper. The script detects your platform, Python version, and CUDA/HIP build, rejects incompatible combinations, and installs the matching precompiled wheel. It then runs fixed-length and variable-length GPU attention and compares the results with PyTorch SDPA; an import alone does not establish GPU support. This avoids declaring every platform wheel in `pyproject.toml`, which can make uv download multiple large Flash Attention wheels while resolving the project.

If you're using pip outside uv, run `python install_flash_attn.py` from the cloned repo. **Do not** run `pip install flash-attn` on Windows; it will try to compile from source and almost certainly fail.

You can also install a wheel directly:

```bash
# Windows + Python 3.13
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.10.2/flash_attn-2.8.3%2Bcu130torch2.14-cp313-cp313-win_amd64.whl

# Windows + Python 3.12
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.10.2/flash_attn-2.8.3%2Bcu130torch2.14-cp312-cp312-win_amd64.whl
```

Once installed, enable it:

```bash
# Any compatible embedding model, including Nemotron-3.5
uv run bananabread-emb --embedding-model nemotron-3.5 --embedding-device cuda --embedding-flash-attention

# Existing Qwen-specific flag remains available
uv run bananabread-emb --embedding-model qwen --qwen-flash-attention --embedding-device cuda
```

The shared `--embedding-flash-attention` option selects FlashAttention 2 per compatible CUDA/HIP backbone using FP16 or BF16. Unsupported runtimes or backbones retain their supported attention implementation, including compact SDPA masks. Transformers is pinned to **5.20.0** to include the Nemotron-3.5 image-input fixes. Installing Transformers alone does not install `flash_attn`; run the installer above in the same environment. CPU, Apple Silicon, unsupported GPUs, and installations without working FA2 retain their supported backends.

At startup, BananaBread only enables Flash Attention 2 when `flash_attn` imports in the same Python environment, Transformers reports FA2 support, a supported CUDA/HIP device is available, and Qwen compute dtype is `float16` or `bfloat16`. If any check fails, BananaBread logs the reason and falls back to the default attention implementation.

> **Windows + bitsandbytes users:** Flash Attention 2 is automatically disabled when using `torch-bnb-8bit` or `torch-bnb-4bit` on Windows because the combination triggers a known CUDA compatibility issue (`torch.AcceleratorError: CUDA error: unknown error`). BananaBread falls back to SDPA attention automatically.

### Qwen GPU quantization

Qwen supports optional bitsandbytes weight quantization on NVIDIA CUDA and AMD ROCm GPUs. This reduces model VRAM usage; it is separate from `quant`, which quantizes the returned embedding vectors.

```bash
# Safer VRAM reduction
uv run --extra cuda-quant bananabread-emb --embedding-model qwen --qwen-size 4B --embedding-device cuda --qwen-backend torch-bnb-8bit

# Lower VRAM, more quality-sensitive
uv run --extra cuda-quant bananabread-emb --embedding-model qwen --qwen-size 4B --embedding-device cuda --qwen-backend torch-bnb-4bit
```

For quantized GPU backends, keep `num_concurrent_embedding` and `num_concurrent_rerank` at `1` unless you have enough VRAM for multiple model copies.

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

For the 4-bit NF4 backend, change `qwen_backend` to `"torch-bnb-4bit"`. Set `matmul_cast_fp16` to `true` if you want Qwen or Nemotron to use `float16` compute during 8-bit quantization instead of keeping `bfloat16` (this avoids the bfloat16→float16 cast warning entirely).

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

API endpoints require a Bearer token (from `api_keys.json`) by default, including Ollama and llama.cpp-compatible endpoints:
```
Authorization: Bearer <your-api-key>
```

Set `"disable_api_key_auth": true` in `config.json` (or use `--disable-api-key-auth`) and restart to allow requests without a key on endpoints marked **User key** below. All such requests use the shared `anonymous` user. Endpoints marked **Management key** still require a management key or a management login cookie.

### Endpoint reference

| Endpoint | Auth | Format | Description |
|---|---|---|---|
| `POST /v1/embeddings` | User key | OpenAI | Generate embeddings (single or batch) |
| `POST /api/v0/embeddings` | User key | LM Studio | LM Studio v0-compatible embeddings |
| `POST /v0/embeddings` | User key | LM Studio | Short alias for `/api/v0/embeddings` |
| `POST /v1/rerank` | User key | Custom | Rerank documents against a query |
| `POST /v1/classify` | Management key | Custom | Classify text by emotion |
| `POST /v1/models/download` | Management key | Custom | Resolve, inspect, and optionally download a Hugging Face model snapshot |
| `GET /v1/models` | User key | OpenAI | List loaded models |
| `GET /v1/health` | User key | Custom | Health check |
| `GET /v1/memory` | Management key | Custom | Memory and thread usage stats |
| `POST /api/embeddings` | User key | Ollama | Legacy `prompt` request; returns a single `embedding` vector |
| `POST /api/embed` | User key | Ollama | `input` string or batch; returns an `embeddings` array in input order |
| `GET /api/tags` | User key | Ollama | Discover the loaded embedding model |
| `POST /api/show` | User key | Ollama | Read the loaded model's effective vector dimensions and context limit |
| `POST /embedding` | User key | llama.cpp | llama.cpp-compatible embedding |
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

#### Embeddings (LM Studio v0 format)

[LM Studio's documented endpoint](https://lmstudio.ai/docs/developer/rest/endpoints#post-apiv0embeddings) is `/api/v0/embeddings`. BananaBread also accepts `/v0/embeddings` as a short alias. Both use the same embedding handler as `/v1/embeddings`:

```bash
curl -X POST http://localhost:8008/api/v0/embeddings \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Nemotron-3.5-Embed-1B-BF16",
    "input": "Some text to embed"
  }'
```

The response contains `object: "list"`, a `data` array of indexed embedding vectors, the loaded `model` identifier, and `usage.prompt_tokens` / `usage.total_tokens`. The `model` request field is accepted for compatibility; BananaBread uses its configured model. A string or array of strings is supported, along with the existing `encoding_format`, `input_type`, and structured image/audio extensions. Authentication, token limits, batching, and caching follow the same settings as `/v1/embeddings`; omit the Authorization header when `disable_api_key_auth` is enabled.

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

#### Embeddings (llama.cpp format)
```bash
curl -X POST http://localhost:8008/embedding \
  -H "Authorization: Bearer <your-api-key>" \
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

The downloader accepts `author` + `path`, a direct repo id in `path`, or standard selectors such as `{"model_name":"qwen","size":"0.6B"}`, `{"model_name":"nemotron"}`, `{"model_name":"nemotron-3","size":"8B"}`, and `{"model_name":"nemotron-3.5","size":"8B"}`. It checks Hub metadata for SentenceTransformers or embedding capability before downloading unless `require_embedding_capable` is set to `false`.

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
- **LM Studio-compatible clients**: Use `/api/v0/embeddings` or `/v0/embeddings` with the same request body as `/v1/embeddings`.
- **Ollama-compatible clients**: Use `/api/embed` with `{"model":"your-model","input":["First text","Second text"]}`. The response includes `embeddings`, with one vector per input, plus `model`, `total_duration`, `load_duration`, and `prompt_eval_count`. Durations are nanoseconds; `load_duration` is zero because models load at startup. Legacy clients can use `/api/embeddings` with `prompt` to receive a single `embedding` vector. Send your API key, or enable `disable_api_key_auth` for clients that cannot send one.

### Open Connections for Obsidian

Select **Ollama** as the embedding provider and set its host to `http://127.0.0.1:8008` (use the server's address if it runs on another machine). Enter the host without `/api` or an endpoint suffix. Open Connections adds `/api/embed`, `/api/tags`, and `/api/show` itself.

Its Ollama adapter sends no Authorization header. Set `"disable_api_key_auth": true` in BananaBread's loaded config and restart the server, then refresh the extension's model list and select the loaded embedding model. Discovery endpoints follow the same authentication setting as embeddings.

`/api/show` reports the runtime's output dimensions after truncation and its configured text context limit, where available. These values come from the loaded model, including pooled models, without running inference. Unknown metadata is omitted. `/api/tags` lists the loaded model; `/api/show` also accepts its name with `:latest` appended.

If Open Connections reports **“Embedding adapter returned no usable vectors”**, click Retry after updating and restarting BananaBread. The server rejects empty, missing, non-finite, or all-zero output vectors with an explicit HTTP 500 instead of returning a successful unusable batch. If no `/api/embed` request reaches the server, the extension may have discarded every input during preprocessing; server response changes cannot repair that path.

---

## Troubleshooting

### Embedding requests stall or Windows reports `WinError 64`

Token counting shares a tokenizer lock with inference. BananaBread runs token
counting and quota accounting on workers so a request waiting for that lock
does not block health checks, connections, or other responses.

The `⏱ embed` log records `tokens`, total `account` time, `account_queue` and
`account_lock` waits, plus `encode_queue`, `encode_lock`, and `encode` work time.
`account` includes its queue and lock waits; those fields should not be added
to it. `encode` excludes queue and lock waits, but includes batching,
tokenization, inference, and output conversion inside the worker. It is not a
GPU-only measurement. Cache hits also log accounting times.

If `encode` itself is very slow, include the token count, model/backend, batch
limits, and dedicated/shared GPU memory usage in a report. Long lock waits
can explain delayed requests without explaining the original slow inference.
On Windows, an `IocpProactor.accept` / `WinError 64` traceback can also indicate
[CPython issue #93821](https://github.com/python/cpython/issues/93821), which can
close the listener after a client disconnect. Record the Python version and
restart the server if it stops accepting connections; the accounting fix does
not repair that separate asyncio failure.

### Embedding request returns `Method Not Allowed` or `Not Found`

For a JSON body with `"content"`, use `POST /embedding`. `/embed` is not an endpoint. For OpenAI-format bodies with `"input"`, use `POST /v1/embeddings`; Ollama clients use `POST /api/embed` or `POST /api/embeddings`.

With `"disable_api_key_auth": true` in the loaded config and the server restarted, this request needs no Authorization header:

```bash
curl http://127.0.0.1:8008/embedding \
  -H 'Content-Type: application/json' \
  -d '{"content":"Hello world","normalize":true}'
```

Older versions could report 405 for unknown paths because the catch-all OPTIONS route masked the 404 response. A valid endpoint called with the wrong HTTP method still returns 405.

### `operator torchvision::nms does not exist` during ROCm verification

This means torchvision's native operators were unavailable when it imported.
One cause is a PyPI or CUDA torchvision wheel paired with ROCm torch. Check
package metadata without importing torchvision (which would trigger the error):

```bash
uv run --no-sync python -c "import importlib.metadata as m; print('torch', m.version('torch')); print('torchvision', m.version('torchvision'))"
```

On Linux, the installer selects `torch 2.14.1+rocm7.2` together with
`torchvision 0.29.1+rocm7.2`. Update your checkout and rerun
`uv run --no-sync python install_rocm_torch.py` to repair a mismatched pair.
Verification prints both installed versions before importing torchvision and
enables its native-extension loading warnings. If the versions already match,
the warning before the traceback can identify a missing shared library or
another loader error; include that warning when reporting the failure.

### `Can't initialize NVML` on an AMD GPU

NVML is NVIDIA's management library. On an AMD host this warning normally means
the BananaBread process loaded the NVIDIA/CUDA Torch wheel instead of the ROCm
wheel. Check the exact project interpreter—not an activated-shell `python` alias:

```bash
uv run --no-sync python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

For Linux, the expected result contains `.venv`, `+rocm7.2`, a non-empty HIP
version, and `True`. If it does not, rerun
`uv run --no-sync python install_rocm_torch.py`. BananaBread now stops before
model loading with these runtime details when an explicit `cuda` device is not
available, instead of leaving the NVML warning as the only clue.

### Flash Attention fails to install on Windows

`pip install flash-attn` tries to compile from source on Windows and will usually fail. Use the helper script instead:

```bash
uv run --no-sync python install_flash_attn.py
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
