import threading
import time
import base64
import torch
import numpy as np
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, RedirectResponse
from sentence_transformers.quantization import quantize_embeddings

from bananabread.config import (
    logger, args, 
    CPU_INFO, AVAILABLE_SOCKETS, TOTAL_PHYSICAL_CORES
)
from bananabread.schemas import (
    RerankRequest, EmbeddingRequest, ClassificationRequest,
    OllamaEmbeddingRequest, OllamaEmbeddingResponse,
    LlamaCppEmbeddingRequest, LlamaCppEmbeddingResponse,
    HFModelDownloadRequest, HFModelDownloadResponse,
    CreateUserRequest, CreateUserResponse, ManagementConfigUpdate,
    ManagementLoginRequest, BulkRegenerateRequest,
)
from bananabread.cache import (
    UserScopedCache, CUDACacheManager,
    get_rerank_cache_key, get_embedding_cache_key,
)
from bananabread.utils import (
    log_embedding_result,
    run_in_threadpool_with_executor,
    get_process_memory_usage,
    get_model_memory_usage
)
from bananabread.hf_models import download_hf_model, inspect_hf_model, resolve_model_repo_id
from bananabread.tenancy import TenantStore, count_text_tokens
from bananabread.admin_panel import ADMIN_PANEL_HTML, LOGIN_HTML
from bananabread.auth import (
    create_jwt, verify_jwt, set_auth_cookie, clear_auth_cookie
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

# Separate executor for embedding quantization. Quantization is CPU-bound
# (numpy) and must not contend with the embedding_executor, or the
# quantization pass can queue behind another request's GPU encode and appear
# to clients as a post-inference hang.
QUANT_THREADS = max(2, CPU_COUNT // 4)
quant_executor = ThreadPoolExecutor(max_workers=QUANT_THREADS, thread_name_prefix="quant")
logger.info(f"  - Quantization threads: {QUANT_THREADS}")

# ----- Caches -----

cache_limit_bytes = args.cache_limit * 1024 * 1024
logger.info(f"💾 Using cache limit: {args.cache_limit} MB ({cache_limit_bytes} bytes)")
logger.info(f"🔢 Embedding quantization: {args.quant}")

rerank_cache = UserScopedCache(cache_limit_bytes)
embedding_cache = UserScopedCache(cache_limit_bytes)

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
tenant_store = TenantStore(API_KEYS_FILE)

def load_api_keys():
    global api_keys
    was_fresh = not tenant_store.path.exists()
    tenant_store.load()
    if was_fresh:
        if getattr(args, "seed_management_key", None) and not tenant_store.data.get("management_key"):
            tenant_store.update_config(management_key=args.seed_management_key)
            logger.info("🔑 Seeded management key from config")
        if getattr(args, "seed_user_key", None) and "user" in tenant_store.data.get("users", {}):
            tenant_store.set_user_api_key("user", args.seed_user_key)
            logger.info("🔑 Seeded user API key from config")
    apply_cache_config()
    api_keys = {
        username: record["api_key"]
        for username, record in tenant_store.data.get("users", {}).items()
    }
    logger.info(f"🔑 Loaded API keys from {API_KEYS_FILE}")

def mb_to_bytes(value: int | None, fallback_bytes: int) -> int:
    return fallback_bytes if value is None else value * 1024 * 1024

def apply_cache_config():
    cache_config = tenant_store.cache_config()
    embedding_cache.configure(
        scope=cache_config["scope"],
        default_limit_bytes=mb_to_bytes(cache_config.get("default_embedding_mb"), cache_limit_bytes),
        user_limits={
            username: mb_to_bytes(limits.get("embedding_mb"), cache_limit_bytes)
            for username, limits in cache_config.get("users", {}).items()
            if limits.get("embedding_mb") is not None
        },
    )
    rerank_cache.configure(
        scope=cache_config["scope"],
        default_limit_bytes=mb_to_bytes(cache_config.get("default_rerank_mb"), cache_limit_bytes),
        user_limits={
            username: mb_to_bytes(limits.get("rerank_mb"), cache_limit_bytes)
            for username, limits in cache_config.get("users", {}).items()
            if limits.get("rerank_mb") is not None
        },
    )

load_api_keys()

def get_api_key(authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return authorization.split("Bearer ")[-1]

def get_api_user(api_key: str = Depends(get_api_key)):
    username = tenant_store.authenticate_user(api_key)
    return {"api_key": api_key, "username": username}

def get_management_key(request: Request, authorization: str = Header(None)):
    # Try Bearer token first
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization.split("Bearer ")[-1]
        tenant_store.authenticate_management(api_key)
        return api_key

    # Fall back to JWT cookie
    token = request.cookies.get("bananabread_auth")
    if token:
        try:
            payload = verify_jwt(token)
            if payload.get("mgmt"):
                return token
        except Exception:
            pass

    raise HTTPException(status_code=401, detail="Management authentication required")

def get_embedding_model_for_token_count():
    model = models_manager.embedding_model
    if model is None and models_manager.embedding_model_pool:
        models = models_manager.embedding_model_pool.get_all_models()
        model = models[0] if models else None
    return model

def get_embedding_tokenizer():
    return getattr(get_embedding_model_for_token_count(), "tokenizer", None)

def count_usage_tokens(texts: List[str]) -> int:
    model = get_embedding_model_for_token_count()
    tokenizer = getattr(model, "tokenizer", None)
    tokenizer_lock = getattr(model, "tokenizer_lock", None)
    if tokenizer_lock is None:
        return count_text_tokens(texts, tokenizer)
    with tokenizer_lock:
        return count_text_tokens(texts, tokenizer)

def consume_user_tokens(username: str, texts: List[str]) -> int:
    tokens = count_usage_tokens(texts)
    tenant_store.check_and_consume(username, tokens)
    return tokens

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
# We allow all origins since BananaBread is typically accessed from various frontends
# Authorization is handled via Bearer token in headers, not cookies
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Using Authorization header, not cookies
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Explicit OPTIONS handler for all routes - ensures CORS preflight works even behind reverse proxies
# This catches any OPTIONS request and returns proper CORS headers
@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    """
    Handle CORS preflight requests explicitly.
    This ensures OPTIONS requests work correctly even behind reverse proxies
    that might not forward them properly to the CORSMiddleware.
    """
    origin = request.headers.get("origin", "*")
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": origin if origin != "*" else "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD",
            "Access-Control-Allow-Headers": request.headers.get("access-control-request-headers", "*"),
            "Access-Control-Max-Age": "86400",  # Cache preflight for 24 hours
        }
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

@app.get("/management/login", response_class=HTMLResponse)
async def management_login_page():
    return LOGIN_HTML

@app.post("/management/auth")
async def management_auth(request_data: ManagementLoginRequest, response: Response):
    try:
        tenant_store.authenticate_management(request_data.key)
    except HTTPException:
        raise HTTPException(status_code=401, detail="Invalid management key")

    token = create_jwt({"mgmt": True})
    set_auth_cookie(response, token)
    return {"status": "ok"}

@app.get("/management/logout")
async def management_logout():
    response = RedirectResponse(url="/management/login", status_code=302)
    clear_auth_cookie(response)
    return response

@app.get("/management", response_class=HTMLResponse)
async def management_panel(request: Request):
    token = request.cookies.get("bananabread_auth")
    if not token:
        return RedirectResponse(url="/management/login", status_code=302)
    try:
        verify_jwt(token)
    except Exception:
        return RedirectResponse(url="/management/login", status_code=302)
    return ADMIN_PANEL_HTML

@app.get("/v1/management/config")
async def get_management_config(api_key: str = Depends(get_management_key)):
    return tenant_store.snapshot()

@app.patch("/v1/management/config")
async def update_management_config(request: ManagementConfigUpdate, api_key: str = Depends(get_management_key)):
    default_limits = request.default_limits.model_dump() if request.default_limits else None
    tiers = {
        tier: limits.model_dump()
        for tier, limits in request.tiers.items()
    } if request.tiers is not None else None
    snapshot = tenant_store.update_config(
        management_key=request.management_key,
        default_limits=default_limits,
        tiers=tiers,
        cache=request.cache.model_dump() if request.cache else None,
    )
    if request.cache is not None:
        apply_cache_config()
    return snapshot

@app.post("/v1/management/users", response_model=CreateUserResponse)
async def create_user_endpoint(request: CreateUserRequest, api_key: str = Depends(get_management_key)):
    limits = request.limits.model_dump() if request.limits else None
    return tenant_store.create_user(request.username, tier=request.tier, limits=limits)

@app.post("/v1/management/users/{username}/regenerate")
async def regenerate_user_key_endpoint(username: str, api_key: str = Depends(get_management_key)):
    new_key = tenant_store.regenerate_user_api_key(username)
    return {"username": username, "api_key": new_key}

@app.post("/v1/management/users/regenerate")
async def bulk_regenerate_user_keys_endpoint(request: BulkRegenerateRequest, api_key: str = Depends(get_management_key)):
    return tenant_store.bulk_regenerate_user_api_keys(request.usernames)

@app.post("/v1/rerank")
async def rerank_endpoint(
    request: RerankRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(get_api_user),
):
    if not request.query or not request.documents:
        raise HTTPException(status_code=400, detail="Query or documents must be provided")

    t_start = time.perf_counter()
    consume_user_tokens(auth["username"], [request.query, *request.documents])

    key = get_rerank_cache_key(request.query, request.documents, request.top_k, request.return_documents, request.task_description)
    cached = rerank_cache.get(auth["username"], key)
    if cached is not None:
        return cached

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

    t_rank = time.perf_counter()
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
    rank_s = time.perf_counter() - t_rank

    # Clear CUDA cache immediately after inference if using GPU
    if torch.cuda.is_available() and (args.rerank_device.startswith('cuda') or args.rerank_device == 'mps'):
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache after reranking")

    # Defer cache insertion until after the response is sent (see embeddings
    # endpoint for rationale).
    background_tasks.add_task(rerank_cache.set, auth["username"], key, result)

    logger.info(
        f"⏱ rerank docs={len(request.documents)} rank={rank_s:.3f}s "
        f"total={time.perf_counter()-t_start:.3f}s"
    )
    return result

@app.post("/v1/embeddings")
async def embedding_endpoint(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(get_api_user),
):
    # ---- timing instrumentation ----
    # Use perf_counter throughout so we can diff between the inference finishing
    # and the response actually being sent. If the client is timing out *after*
    # the batch is processed, the gap between `t_inference_done` and `return`
    # is what we care about.
    t_start = time.perf_counter()

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
    prompt_tokens = consume_user_tokens(auth["username"], inputs)

    logger.info(f"📄 Processing {len(inputs)} documents for embeddings (format: {request.encoding_format})")

    key = get_embedding_cache_key(inputs, request.encoding_format) if inputs else ""
    cached = embedding_cache.get(auth["username"], key)
    if cached is not None:
        cached_result = deepcopy(cached)
        cached_result["usage"] = {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}
        logger.info(f"⏱ embed cache-hit total={time.perf_counter()-t_start:.3f}s n={len(inputs)}")
        return cached_result

    cuda_cache_manager.mark_inference_activity()

    def get_embedding_model_and_encode(inputs):
        if models_manager.embedding_model_pool:
            model = models_manager.embedding_model_pool.get_model()
        else:
            model = models_manager.embedding_model
        return model.encode(inputs)

    t_encode = time.perf_counter()
    docs_embeddings = await run_in_threadpool_with_executor(
        embedding_executor,
        get_embedding_model_and_encode,
        inputs
    )
    encode_s = time.perf_counter() - t_encode

    t_post = time.perf_counter()
    if hasattr(docs_embeddings, 'cpu'):
        # Convert BFloat16 to Float32 before numpy conversion / quantization
        # (numpy and sentence_transformers quantize_embeddings don't support bf16)
        if docs_embeddings.dtype == torch.bfloat16:
            docs_embeddings = docs_embeddings.to(torch.float32)
        docs_embeddings = docs_embeddings.cpu()

    # Quantization
    quant_s = 0.0
    if args.quant != 'standard':
        def quantize_embeddings_wrapper(embeddings):
            kwargs = {"precision": args.quant}
            if args.quant == "int8" and models_manager.calibration_embeddings is not None:
                kwargs["calibration_embeddings"] = models_manager.calibration_embeddings
            return quantize_embeddings(embeddings, **kwargs)

        t_quant = time.perf_counter()
        docs_embeddings = await run_in_threadpool_with_executor(
            quant_executor,
            quantize_embeddings_wrapper,
            docs_embeddings
        )
        quant_s = time.perf_counter() - t_quant
        logger.debug(f"Applied {args.quant} quantization to embeddings")

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
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}
    }
    post_s = time.perf_counter() - t_post

    # Defer disk logging and cache insertion until *after* the response is
    # flushed to the client. Previously both ran synchronously on the event
    # loop after inference completed, which is what made clients time out
    # even though the batch had already been processed.
    log_data = embeddings_list if request.encoding_format != "base64" else ["<base64_encoded_data>"] * len(embeddings_list)
    background_tasks.add_task(
        log_embedding_result,
        inputs,
        log_data,
        {
            "model": models_manager.embedding_model_name,
            "quantization": args.quant,
            "embedding_dimensions": args.embedding_dim if args.embedding_model in {'mixedbread', 'hf'} else "native",
        },
    )
    background_tasks.add_task(embedding_cache.set, auth["username"], key, result)

    total_s = time.perf_counter() - t_start
    logger.info(
        f"⏱ embed n={len(inputs)} encode={encode_s:.3f}s "
        f"quant={quant_s:.3f}s post={post_s:.3f}s total={total_s:.3f}s"
    )
    return result

@app.post("/v1/classify")
async def classify_endpoint(request: ClassificationRequest, api_key: str = Depends(get_management_key)):
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

@app.post("/v1/models/download", response_model=HFModelDownloadResponse)
async def download_model_endpoint(request: HFModelDownloadRequest, api_key: str = Depends(get_management_key)):
    try:
        repo_id = resolve_model_repo_id(
            author=request.author,
            path=request.path,
            model_name=request.model_name,
            size=request.size,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def inspect_and_download():
        hf_access_token = request.hf_access_token or args.hf_access_token
        metadata = inspect_hf_model(repo_id, revision=request.revision, token=hf_access_token)
        if request.require_embedding_capable and not metadata["is_embedding_capable"]:
            raise ValueError(
                f"Model '{repo_id}' does not appear to be SentenceTransformers or embedding capable. "
                "Set require_embedding_capable=false to download anyway."
            )

        local_path = None
        if request.download:
            local_path = download_hf_model(
                repo_id,
                storage_dir=request.storage_dir or args.model_storage_dir,
                revision=request.revision,
                token=hf_access_token,
                allow_patterns=request.allow_patterns,
                ignore_patterns=request.ignore_patterns,
            )

        return {
            "repo_id": repo_id,
            "local_path": local_path,
            "downloaded": request.download,
            "metadata": metadata,
        }

    try:
        return await run_in_threadpool_with_executor(general_executor, inspect_and_download)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error downloading Hugging Face model '{repo_id}': {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/api/embeddings")
@app.post("/api/embed")
async def ollama_embeddings_endpoint(request: OllamaEmbeddingRequest, auth: dict = Depends(get_api_user)):
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
    consume_user_tokens(auth["username"], inputs)
    
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

    # Convert BFloat16 to Float32 before tolist() (numpy/tolist doesn't support bf16)
    if hasattr(docs_embeddings, 'dtype') and docs_embeddings.dtype == torch.bfloat16:
        docs_embeddings = docs_embeddings.to(torch.float32)

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
async def llamacpp_embedding_endpoint(request: LlamaCppEmbeddingRequest, auth: dict = Depends(get_api_user)):
    try:
        if not request.content:
            raise HTTPException(status_code=400, detail="Content must be provided")
        
        if isinstance(request.content, str):
            inputs = [request.content]
            is_batch = False
        else:
            inputs = request.content
            is_batch = True
        consume_user_tokens(auth["username"], inputs)
            
        logger.info(f"📄 Processing {len(inputs)} document(s) for Llama.cpp embeddings")
        
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
async def memory_usage(api_key: str = Depends(get_management_key)):
    data = {}
    data["process_memory"] = get_process_memory_usage()
    data["rerank_model_memory"] = get_model_memory_usage(models_manager.rerank_model)
    data["embedding_model_memory"] = get_model_memory_usage(models_manager.embedding_model)
    data["embedding_cache_memory"] = embedding_cache.total_size()
    data["cache_stats"] = {
        "embedding": embedding_cache.stats(),
        "rerank": rerank_cache.stats(),
    }
    
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
async def health(api_key: str = Depends(get_api_user)):
    return {"status": "healthy", "cpu_cores": CPU_COUNT, "optimized": True}

@app.get("/v1/models")
async def model(api_key: str = Depends(get_api_user)):
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
        "embedding_dimensions": args.embedding_dim if args.embedding_model in {'mixedbread', 'hf'} else "native",
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "rerank": "/v1/rerank", 
            "classify": "/v1/classify",
            "management_panel": "/management",
            "management_config": "/v1/management/config",
            "create_user": "/v1/management/users",
            "download_model": "/v1/models/download",
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
                               ("quant", quant_executor),
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

        try:
            tenant_store.close()
        except Exception as e:
            logger.error(f"Error closing tenant store: {e}")

        cleanup_completed = True
        logger.info("✅ Graceful shutdown completed")
