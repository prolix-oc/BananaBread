# llama.cpp-Compatible Embedding Endpoint

This server now includes a llama.cpp-compatible embedding endpoint that allows you to use llama.cpp clients and tools with the mixedbread embedding models.

## Auth Setup
Create an empty `api_keys.json` file, and add the following:
```json
{
    "user": ""
}
```
The server will automatically fill any missing API keys for each user entry. This server does support multiple users for embeddings, reranking, and classification.

## How to Install and Run
**Recommended virtual environment manager: UV**

```
git clone https://github.com/prolix-oc/BananaBread-Emb
cd BananaBread-Emb
uv run bananabread-emb
```

This will begin a download of all the prerequesite models from HuggingFace. This will download `mxbai-embed-large-v1`, `mxbai-rerank-base-v2`, and `SamLowe/roberta-base-go_emotions`, then start the endpoint on port 8008.

## Advanced Usage
To see all advanced flags:
```
uv run bananabread-emb -h
```
To raise the LRU cache limit (vectors held in memory for faster access):
```
uv run bananabread-emb --cache-limit [size-in-MB, default 1024]
```
As an example, if you have 32GB of RAM + 16 CPU cores, and would like to allocated 8GiB of memory + 8 cores for embedding, you would enter:
```
uv run bananabread-emb --cache-limit 8192 --embedding-cores 8
```
This will keep up to 8GiB of vector embeddings in memory and use half of the CPU for calculating embeddings, dramatically decreasing request latency for already-embedded documents.

## Documentation
Access ReDoc at `https://<ip-of-host>:8008/redoc` for a better understanding of how to interact with the many endpoints of BananaBread.

## Endpoint Details

### `POST /embedding`
**No API key required**

Test llama.cpp-compatible endpoint for generating embeddings.

### `POST /v1/embeddings/llamacpp`
**No API key required**

Alternative llama.cpp-compatible endpoint with v1 prefix.

## Request Format

The endpoint accepts requests in llama.cpp server API format:

```json
{
  "content": "Your text to embed",
  "model": "mixedbread-ai/mxbai-embed-large-v1",
  "normalize": true,
  "truncate": true
}
```

### Request Parameters

- `content` (required): The text content to generate embeddings for
- `model` (optional): The model to use (defaults to "mixedbread-ai/mxbai-embed-large-v1")
- `normalize` (optional): Whether to normalize the embedding to unit length (defaults to true)
- `truncate` (optional): Whether to truncate input text if needed (defaults to true)

## Response Format

The response follows the llama.cpp server API specification:

```json
{
  "embedding": [0.123, -0.456, 0.789, ...],
  "model": "mixedbread-ai/mxbai-embed-large-v1"
}
```

### Response Fields

- `embedding`: The generated embedding vector as a list of floats
- `model`: The model used to generate the embedding

## Usage Examples

### Using curl

```bash
# Basic embedding request
curl -X POST http://localhost:8008/embedding \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Hello world, this is a test sentence.",
    "model": "mixedbread-ai/mxbai-embed-large-v1"
  }'

# With normalization disabled
curl -X POST http://localhost:8008/embedding \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Test sentence without normalization.",
    "model": "mixedbread-ai/mxbai-embed-large-v1",
    "normalize": false
  }'
```

### Using Python

```python
import requests
import math

# Test the llama.cpp-compatible endpoint
response = requests.post('http://localhost:8008/embedding', json={
    'content': 'Hello world, this is a test sentence for embedding.',
    'model': 'mixedbread-ai/mxbai-embed-large-v1',
    'normalize': True
})

if response.status_code == 200:
    result = response.json()
    print(f"Embedding length: {len(result['embedding'])}")
    print(f"Model: {result['model']}")
    
    # Check if embedding is normalized
    embedding = result['embedding']
    norm = math.sqrt(sum(x*x for x in embedding))
    print(f"Embedding norm: {norm:.6f}")
else:
    print(f"Error: {response.text}")
```

## Comparison with Other Formats

### OpenAI Format (requires API key)
- Endpoint: `POST /v1/embeddings`
- Requires: `Authorization: Bearer <api_key>` header
- Input: `{"input": ["text1", "text2"], "model": "model-name"}`
- Output: `{"object": "list", "data": [{"object": "embedding", "embedding": [...], "index": 0}]}`

### Ollama Format (no API key required)
- Endpoint: `POST /api/embeddings` or `POST /api/embed`
- No authentication required
- Input: `{"prompt": "text", "model": "model-name"}` or `{"input": "text", "model": "model-name"}`
- Output: `{"embedding": [...], "model": "model-name"}`

### llama.cpp Format (API key required)
- Endpoint: `POST /embedding` or `POST /v1/embeddings/llamacpp`
- Requires: `Authorization: Bearer <api_key>` header
- Input: `{"content": "text", "model": "model-name", "normalize": true}`
- Output: `{"embedding": [...], "model": "model-name"}`

## Key Features

### Normalization Support
The llama.cpp endpoint supports optional embedding normalization:
- When `normalize: true` (default), embeddings are normalized to unit length
- When `normalize: false`, raw embeddings are returned
- Normalized embeddings are useful for cosine similarity calculations

### Multiple Endpoints
Two endpoints are provided for compatibility:
- `/embedding` - Primary endpoint matching llama.cpp server default
- `/v1/embeddings/llamacpp` - Alternative with v1 prefix for consistency

## Model Information

The endpoint relies on the `mixedbread-ai/mxbai-embed-large-v1` model with:
- 1024-dimensional embeddings (truncated)
- CPU-based inference (can use GPU)
- Built-in LRU caching with eviction for performance
- Optional normalization support

## Notes

- The endpoint is fully compatible with llama.cpp clients and tools
- No API key authentication is required
- The embedding dimension is 1024 (truncated from the original model)
- Normalization is enabled by default but can be disabled
- The endpoint includes proper error handling and validation
- Both endpoints provide identical functionality, choose based on your client requirements

## Integration with llama.cpp

This endpoint is designed to work with llama.cpp server clients that expect the standard llama.cpp embedding API format. The response structure and parameter names match the llama.cpp server specification, making it easy to integrate with existing llama.cpp-based applications and tools. It also reports back via the "models" request, further tricking software like SillyTavern into thinking it's a full llama.cpp instance.
