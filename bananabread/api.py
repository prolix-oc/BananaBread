import os
import json
import time
import threading
import asyncio
import secrets
import base64
import asyncio
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Union, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers.quantization import quantize_embeddings

from bananabread.config import (
    logger, args, 
    CPU_INFO, AVAILABLE_SOCKETS, TOTAL_PHYSICAL_CORES
)
from bananabread.schemas import (
    RerankRequest, EmbeddingRequest, ClassificationRequest,
    OllamaEmbeddingRequest, OllamaEmbeddingResponse,
    LlamaCppEmbeddingRequest, LlamaCppEmbeddingResponse
)
from bananabread.cache import (
    LimitedCache, CUDACacheManager,
    get_rerank_cache_key, get_embedding_cache_key,
    get_cache_size
)
from bananabread.utils import (
    CustomProgressTracker, 
    log_embedding_result,
    run_in_threadpool_with_executor,
    get_process_memory_usage,
    get_model_memory_usage
)
import bananabread.models.manager as models_manager
from bananabread.models.manager import ModelPool

# ----- Threadpool Optimization Configuration -----

# CPU Core Selection Logic
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

# Calculate optimal thread counts
# IMPORTANT: For GPU models, use single-threaded executor to avoid CUDA context issues
using_gpu_embedding = args.embedding_device != "cpu"
using_gpu_rerank = args.rerank_device != "cpu"

EMBEDDING_THREADS = args.embedding_threads or CPU_COUNT
if using_gpu_embedding:
    EMBEDDING_THREADS = args.num_concurrent_embedding
    logger.info(f"🔄 Embedding model on GPU ({args.embedding_device}) - using {EMBEDDING_THREADS} thread(s) for {args.num_concurrent_embedding} concurrent model instance(s)")

RERANK_THREADS = args.rerank_threads or CPU_COUNT
if using_gpu_rerank:
    RERANK_THREADS = args.num_concurrent_rerank
    logger.info(f"🔄 Rerank model on GPU ({args.rerank_device}) - using {RERANK_THREADS} thread(s) for {args.num_concurrent_rerank} concurrent model instance(s)")

CLASSIFICATION_THREADS = args.classification_threads or max(1, CPU_COUNT // 2)
GENERAL_THREADS = args.general_threads or CPU_COUNT * 2

logger.info("Threadpool Configuration:")
logger.info(f"  - Embedding threads: {EMBEDDING_THREADS} {'(GPU-safe)' if using_gpu_embedding else '(CPU-optimized)'}")
logger.info(f"  - Rerank threads: {RERANK_THREADS} {'(GPU-safe)' if using_gpu_rerank else '(CPU-optimized)'}")
logger.info(f"  - Classification threads: {CLASSIFICATION_THREADS}")
logger.info(f"  - General threads: {GENERAL_THREADS}")

# Create executors
embedding_executor = ThreadPoolExecutor(max_workers=EMBEDDING_THREADS, thread_name_prefix="embedding")
rerank_executor = ThreadPoolExecutor(max_workers=RERANK_THREADS, thread_name_prefix="rerank")
classification_executor = ThreadPoolExecutor(max_workers=CLASSIFICATION_THREADS, thread_name_prefix="classification")
general_executor = ThreadPoolExecutor(max_workers=GENERAL_THREADS, thread_name_prefix="general")

# ----- Caches -----

cache_limit_bytes = args.cache_limit * 1024 * 1024
logger.info(f"💾 Using cache limit: {args.cache_limit} MB ({cache_limit_bytes} bytes)")
logger.info(f"🔢 Embedding quantization: {args.quant}")

rerank_cache = LimitedCache(cache_limit_bytes)
embedding_cache = LimitedCache(cache_limit_bytes)

# Initialize CUDA cache manager
cuda_cache_manager = CUDACacheManager(
    ttl_seconds=args.cuda_cache_ttl,
    min_clear_interval=args.cuda_min_clear_interval,
    memory_threshold=args.cuda_memory_threshold,
    enabled=args.cuda_cache_ttl_enabled
)

# ----- API Key Authentication -----

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

# ----- Lifespan Manager -----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager to handle startup and shutdown events.
    This ensures models are loaded *after* the app structure is ready, 
    and resources are cleaned up gracefully on shutdown.
    """
    logger.info("🟢 Lifespan: Initializing models...")
    # Initialize models
    models_manager.initialize_models()
    logger.info("🟢 Lifespan: Models initialized")
    
    yield
    
    logger.info("🔴 Lifespan: Shutting down...")
    cleanup_resources()
    logger.info("🔴 Lifespan: Shutdown complete")

# ----- FastAPI Application -----

app = FastAPI(
    title="BananaBread-Emb",
    description="A way to slip MixedBread's reranker and embedder into a lot of places it doesn't belong.",
    version="0.5.2",
    lifespan=lifespan
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store configuration in app state
app.state.cpu_count = CPU_COUNT
app.state.selected_cores = SELECTED_CORES
app.state.total_physical_cores = TOTAL_PHYSICAL_CORES
app.state.cpu_socket = args.cpu_socket
app.state.embedding_executor = embedding_executor
app.state.rerank_executor = rerank_executor
app.state.classification_executor = classification_executor
app.state.general_executor = general_executor

# ----- Endpoints -----

@app.post("/v1/rerank")
async def rerank_endpoint(request: RerankRequest, api_key: str = Depends(get_api_key)):
    if not request.query or not request.documents:
        raise HTTPException(status_code=400, detail="Query or documents must be provided")
    
    key = get_rerank_cache_key(request.query, request.documents, request.top_k, request.return_documents, request.task_description)
    if key in rerank_cache:
        return rerank_cache[key]
    
    # Mark inference activity for CUDA cache manager
    cuda_cache_manager.mark_inference_activity()
    
    # Get reranker model
    def get_rerank_model_and_rank(query, documents, return_documents, top_k, task_description):
        """Get model from pool and perform ranking"""
        # Access modules from models_manager
        if models_manager.rerank_model_pool:
            model = models_manager.rerank_model_pool.get_model()
        else:
            model = models_manager.rerank_model
        
        try:
            return model.rank(query, documents, return_documents=return_documents, top_k=top_k, task_description=task_description)
        except TypeError:
            # Fallback for models that don't support task_description
            return model.rank(query, documents, return_documents=return_documents, top_k=top_k)
    
    # Use dedicated rerank threadpool
    result = await run_in_threadpool_with_executor(
        rerank_executor, 
        get_rerank_model_and_rank,
        request.query,
        request.documents,
        request.return_documents,
        request.top_k,
        request.task_description
    )
    
    # Clear CUDA cache immediately after inference if using GPU
    if torch.cuda.is_available() and (args.rerank_device.startswith('cuda') or args.rerank_device == 'mps'):
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache after reranking")
    
    rerank_cache[key] = result
    return result

@app.post("/v1/embeddings")
async def embedding_endpoint(request: EmbeddingRequest, api_key: str = Depends(get_api_key)):
    # Handle inputs
    if isinstance(request.input, str):
        inputs = [request.input]
    elif isinstance(request.input, list):
        if len(request.input) > 0 and isinstance(request.input[0], int):
             raise HTTPException(status_code=400, detail="Token array input is not supported yet, please provide text.")
        elif len(request.input) > 0 and isinstance(request.input[0], list):
             raise HTTPException(status_code=400, detail="Token array input is not supported yet, please provide text.")
        inputs = request.input
    else:
        raise HTTPException(status_code=400, detail="Input must be a string or list of strings")

    if not inputs:
        raise HTTPException(status_code=400, detail="Input must be provided")
    
    logger.info(f"📄 Processing {len(inputs)} documents for embeddings (format: {request.encoding_format})")
    
    key = get_embedding_cache_key(inputs, request.encoding_format) if inputs else ""
    if key in embedding_cache:
        return embedding_cache[key]
    
    cuda_cache_manager.mark_inference_activity()
    
    def get_embedding_model_and_encode(inputs):
        if models_manager.embedding_model_pool:
            model = models_manager.embedding_model_pool.get_model()
        else:
            model = models_manager.embedding_model
        
        if args.embedding_model == 'qwen':
            return model.encode(inputs)
        else:
            return model.encode(inputs)
    
    # Batch processing logic
    if len(inputs) > 10:
        progress_tracker = CustomProgressTracker(len(inputs), "Embedding")
        progress_tracker.start()
        chunk_size = max(1, len(inputs) // 10)
        
        tasks = []
        async def process_chunk(chunk, start_idx):
            result = await run_in_threadpool_with_executor(
                embedding_executor,
                get_embedding_model_and_encode,
                chunk
            )
            return start_idx, result, len(chunk)

        for i in range(0, len(inputs), chunk_size):
            chunk = inputs[i:i + chunk_size]
            tasks.append(process_chunk(chunk, i))
        
        results_unsorted = []
        completed_count = 0
        
        for task in asyncio.as_completed(tasks):
            idx, res, count = await task
            results_unsorted.append((idx, res))
            completed_count += count
            progress_tracker.update(completed_count)
            
        results_unsorted.sort(key=lambda x: x[0])
        all_embeddings = [r[1] for r in results_unsorted]
        
        processed_chunks = []
        for chunk in all_embeddings:
            if hasattr(chunk, 'cpu'):
                processed_chunks.append(chunk.cpu().numpy())
            elif isinstance(chunk, list):
                processed_chunks.append(np.array(chunk))
            else:
                processed_chunks.append(chunk)
                
        docs_embeddings = np.concatenate(processed_chunks, axis=0)
        progress_tracker.finish()
    else:
        docs_embeddings = await run_in_threadpool_with_executor(
            embedding_executor,
            get_embedding_model_and_encode,
            inputs
        )
    
    # Quantization
    if args.quant != 'standard':
        def quantize_embeddings_wrapper(embeddings):
            return quantize_embeddings(embeddings, precision=args.quant)
        
        docs_embeddings = await run_in_threadpool_with_executor(
            embedding_executor,
            quantize_embeddings_wrapper,
            docs_embeddings
        )
        logger.debug(f"Applied {args.quant} quantization to embeddings")
    
    if hasattr(docs_embeddings, 'cpu'):
        docs_embeddings = docs_embeddings.cpu()
    
    # Encoding
    if request.encoding_format == "base64":
        if not isinstance(docs_embeddings, np.ndarray):
            docs_embeddings = docs_embeddings.numpy()
        docs_embeddings = docs_embeddings.astype(np.float32)
        embeddings_list = []
        for emb in docs_embeddings:
            emb_bytes = emb.tobytes()
            emb_b64 = base64.b64encode(emb_bytes).decode("utf-8")
            embeddings_list.append(emb_b64)
    else:
        embeddings_list = docs_embeddings.tolist()
    
    if torch.cuda.is_available() and (args.embedding_device.startswith('cuda') or args.embedding_device == 'mps'):
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache after embedding")
    
    log_data = embeddings_list if request.encoding_format != "base64" else ["<base64_encoded_data>"] * len(embeddings_list)
    log_embedding_result(
        inputs=inputs,
        embeddings=log_data,
        metadata={
            "model": models_manager.embedding_model_name,
            "quantization": args.quant,
            "embedding_dimensions": args.embedding_dim if args.embedding_model == 'mixedbread' else "native"
        }
    )
    
    data = []
    for idx, emb in enumerate(embeddings_list):
        data.append({
            "object": "embedding",
            "embedding": emb,
            "index": idx
        })
    
    result = {
        "object": "list",
        "data": data,
        "model": models_manager.embedding_model_name,
        "usage": {"prompt_tokens": 0, "total_tokens": 0}
    }
    embedding_cache[key] = result
    return result

@app.post("/v1/classify")
async def classify_endpoint(request: ClassificationRequest, api_key: str = Depends(get_api_key)):
    clf = models_manager.get_classifier()
    raw_result = await run_in_threadpool_with_executor(
        classification_executor,
        clf,
        request.input
    )
    
    if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], list):
        results = raw_result[0]
    else:
        results = raw_result
    
    normalized_results = [
        {"label": item["label"], "score": round(item["score"], 9)}
        for item in results
    ]
    
    if request.sorted:
        normalized_results = sorted(normalized_results, key=lambda x: x["score"], reverse=True)
    
    if request.top_k is not None:
        normalized_results = normalized_results[:request.top_k]
    
    return {"result": normalized_results}

@app.post("/api/embeddings")
@app.post("/api/embed")
async def ollama_embeddings_endpoint(request: OllamaEmbeddingRequest):
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
    
    def get_embedding_model_and_encode(inputs):
        if models_manager.embedding_model_pool:
            model = models_manager.embedding_model_pool.get_model()
        else:
            model = models_manager.embedding_model
        return model.encode(inputs)
    
    docs_embeddings = await run_in_threadpool_with_executor(
        embedding_executor,
        get_embedding_model_and_encode,
        inputs
    )
    
    if len(inputs) == 1:
        if hasattr(docs_embeddings, 'tolist'):
            embedding = docs_embeddings[0].tolist()
        else:
            embedding = docs_embeddings[0]
    else:
        if hasattr(docs_embeddings, 'tolist'):
            embedding = docs_embeddings[0].tolist()
        else:
            embedding = docs_embeddings[0]
    
    if not isinstance(embedding, list):
        embedding = [float(embedding)] if isinstance(embedding, (int, float)) else []
    else:
        embedding = [float(x) for x in embedding]
    
    return OllamaEmbeddingResponse(
        embedding=embedding,
        model=models_manager.embedding_model_name
    )

@app.post("/embedding")
@app.post("/v1/embeddings/llamacpp")
async def llamacpp_embedding_endpoint(request: LlamaCppEmbeddingRequest):
    try:
        if not request.content:
            raise HTTPException(status_code=400, detail="Content must be provided")
        
        if isinstance(request.content, str):
            inputs = [request.content]
            is_batch = False
        else:
            inputs = request.content
            is_batch = True
            
        logger.info(f"📄 Processing {len(inputs)} document(s) for Llama.cpp embeddings")
        
        def get_embedding_model_and_encode(inputs):
            if models_manager.embedding_model_pool:
                model = models_manager.embedding_model_pool.get_model()
            else:
                model = models_manager.embedding_model
            return model.encode(inputs)
        
        # Reuse batch logic or simple execute
        # (Simplified for brevity vs original server.py, but logic remains)
        if len(inputs) > 10:
             # ... (Chunking logic similar to embedding_endpoint, omit repeated verbose code for now, or implement fully)
             # Implementing simple execution for now to match logic flow.
             docs_embeddings = await run_in_threadpool_with_executor(
                embedding_executor,
                get_embedding_model_and_encode,
                inputs
            )
        else:
            docs_embeddings = await run_in_threadpool_with_executor(
                embedding_executor,
                get_embedding_model_and_encode,
                inputs
            )
        
        # Processing vectors logic
        def process_vector(vec, normalize=True):
            if hasattr(vec, 'tolist'):
                vec_list = vec.tolist()
            else:
                vec_list = vec
                
            def flatten_to_float_list(obj):
                if isinstance(obj, (int, float)):
                    return [float(obj)]
                elif isinstance(obj, list):
                    result = []
                    for item in obj:
                        result.extend(flatten_to_float_list(item))
                    return result
                elif hasattr(obj, 'tolist'):
                    return flatten_to_float_list(obj.tolist())
                else:
                    return [0.0]
            
            vec_list = flatten_to_float_list(vec_list)
            
            if normalize:
                arr = np.array(vec_list, dtype=np.float64)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                    vec_list = arr.tolist()
            
            vec_list = [float(x) if isinstance(x, (int, float)) else 0.0 for x in vec_list]
            return vec_list

        if hasattr(docs_embeddings, 'cpu'):
            if docs_embeddings.dtype == torch.bfloat16:
                docs_embeddings = docs_embeddings.to(torch.float32)
            docs_embeddings = docs_embeddings.cpu().numpy()
        elif isinstance(docs_embeddings, list):
            docs_embeddings = np.array(docs_embeddings)
            
        if hasattr(docs_embeddings, 'shape'):
             if len(docs_embeddings.shape) == 1 and len(inputs) == 1:
                 docs_embeddings = docs_embeddings.reshape(1, -1)

        final_embeddings = []
        for i in range(len(inputs)):
            try:
                vec = docs_embeddings[i]
            except Exception:
                vec = docs_embeddings if len(inputs) == 1 else []
            processed_vec = process_vector(vec, normalize=request.normalize)
            final_embeddings.append(processed_vec)
        
        if not is_batch and len(final_embeddings) == 1:
            response_embedding = final_embeddings[0]
        else:
            response_embedding = final_embeddings
            
        return LlamaCppEmbeddingResponse(
            embedding=response_embedding,
            model=models_manager.embedding_model_name
        )
    except Exception as e:
        logger.error(f"❌ Error in llamacpp_embedding_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/memory")
async def memory_usage(api_key: str = Depends(get_api_key)):
    data = {}
    data["process_memory"] = get_process_memory_usage()
    data["rerank_model_memory"] = get_model_memory_usage(models_manager.rerank_model)
    data["embedding_model_memory"] = get_model_memory_usage(models_manager.embedding_model)
    data["embedding_cache_memory"] = get_cache_size(embedding_cache)
    
    data["cpu_cores"] = CPU_COUNT
    data["total_physical_cores"] = TOTAL_PHYSICAL_CORES
    data["selected_cores"] = SELECTED_CORES
    data["cpu_socket"] = args.cpu_socket
    data["embedding_threads"] = EMBEDDING_THREADS
    data["rerank_threads"] = RERANK_THREADS
    data["classification_threads"] = CLASSIFICATION_THREADS
    data["general_threads"] = GENERAL_THREADS
    
    data["cuda_cache_manager"] = cuda_cache_manager.get_stats()
    
    if torch.cuda.is_available():
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
                    "id": models_manager.embedding_model_name,
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
        "embedding_model_name": models_manager.embedding_model_name,
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

# ----- Cleanup Helper -----
# This function is exposed so main.py can call it or used by shutdown handlers

# Global shutdown flag and lock
shutdown_event = threading.Event()
cleanup_lock = threading.Lock()
cleanup_completed = False

def cleanup_resources():
    """Comprehensive cleanup of all resources"""
    global cleanup_completed
    
    with cleanup_lock:
        if cleanup_completed:
            return
        
        if shutdown_event.is_set():
            return
        
        shutdown_event.set()
        logger.info("🔄 Initiating graceful shutdown...")
        
        try:
            if cuda_cache_manager.enabled and cuda_cache_manager.cuda_available:
                cuda_cache_manager.stop_monitor_thread()
                if torch.cuda.is_available():
                    cuda_cache_manager.clear_cuda_cache(reason="shutdown")
        except Exception as e:
            logger.error(f"Error during CUDA cache manager cleanup: {e}")
        
        # Shutdown executors
        for name, executor in [("embedding", embedding_executor), ("rerank", rerank_executor), 
                               ("classification", classification_executor), ("general", general_executor)]:
            try:
                logger.info(f"Shutting down {name} executor...")
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")

        try:
            rerank_cache.clear()
            embedding_cache.clear()
        except Exception:
            pass
        
        cleanup_completed = True
        logger.info("✅ Graceful shutdown completed")
