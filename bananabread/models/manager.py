import time
import threading
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from mxbai_rerank import MxbaiRerankV2

from bananabread.config import logger, args
from bananabread.hf_models import download_hf_model, inspect_hf_model, resolve_model_repo_id
from bananabread.models.qwen import QwenRawModel, load_qwen_model

# Global model references
embedding_model = None
embedding_model_pool = None
rerank_model = None
rerank_model_pool = None
embedding_model_name = ""

# Lazy-loaded classifier
classifier = None

# Pre-computed calibration embeddings for stable int8 quantization
calibration_embeddings = None

_CALIBRATION_TEXTS = [
    "This is a short sentence.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The quick brown fox jumps over the lazy dog.",
    "In 1492, Christopher Columbus sailed across the Atlantic Ocean and reached the Americas.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "Python is a high-level programming language known for its readability and versatility.",
    "The capital of France is Paris, a city renowned for its art, culture, and cuisine.",
    "Quantum mechanics describes the behavior of matter and energy at the smallest scales.",
    "A healthy diet includes a variety of fruits, vegetables, whole grains, and lean proteins.",
    "The Great Wall of China is one of the most impressive architectural feats in human history.",
    "Climate change refers to long-term shifts in temperatures and weather patterns around the world.",
    "Shakespeare wrote many famous plays, including Hamlet, Macbeth, and Romeo and Juliet.",
    "The human brain contains approximately 86 billion neurons that communicate through synapses.",
    "Artificial neural networks are inspired by the structure and function of biological brains.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Internet has revolutionized communication, commerce, and access to information globally.",
    "Einstein's theory of relativity fundamentally changed our understanding of space and time.",
    "Regular exercise is essential for maintaining physical health and mental well-being.",
    "The Mona Lisa is a portrait painted by Leonardo da Vinci in the early 16th century.",
    "Cryptocurrencies like Bitcoin use blockchain technology to maintain decentralized ledgers.",
]

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

# ----- Compilation and Warmup -----

def compile_model_if_enabled(model, model_name: str):
    """
    Apply torch.compile() to a model if enabled via CLI flags.
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
            # Check if it's our QwenRawModel wrapper - compilation support for raw HF models
            if isinstance(model, QwenRawModel):
                logger.info(f"   Compiling underlying HF model for QwenRawModel...")
                model.model = torch.compile(
                    model.model,
                    mode=args.torch_compile_mode,
                    backend=args.torch_compile_backend
                )
                logger.info(f"✅ {model_name} compiled successfully")
                return model

            # Compile the model's encode function for SentenceTransformers
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
                        # SentenceTransformer or QwenRawModel
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

class PooledQwenReranker:
    """Wrapper for Qwen reranker that uses a model pool"""
    def __init__(self, model_pool):
        self.model_pool = model_pool
        logger.info(f"♻️  Qwen reranker initialized using shared model pool")
    
    def rank(self, query: str, documents: list[str], return_documents: bool = False, top_k: int = None, task_description: str = None) -> dict:
        """Rank using a model from the shared pool"""
        # Get a model from the pool
        model = self.model_pool.get_model()
        # Use the rank method of the pooled QwenRawModel
        return model.rank(query, documents, return_documents=return_documents, top_k=top_k, task_description=task_description)

# ----- Initialization Logic -----

def load_hf_embedding_model():
    if not args.hf_model_slug:
        raise ValueError("--hf-model-slug is required when --embedding-model=hf")

    repo_id = resolve_model_repo_id(path=args.hf_model_slug)
    metadata = inspect_hf_model(repo_id, revision=args.hf_model_revision, token=args.hf_access_token)
    if not metadata["is_embedding_capable"]:
        raise ValueError(
            f"Hugging Face model '{repo_id}' does not appear to be SentenceTransformers or embedding capable"
        )

    local_path = download_hf_model(
        repo_id,
        storage_dir=args.model_storage_dir,
        revision=args.hf_model_revision,
        token=args.hf_access_token,
    )
    logger.info(f"📦 Loaded Hugging Face embedding snapshot: {repo_id} -> {local_path}")
    return repo_id, SentenceTransformer(local_path, truncate_dim=args.embedding_dim, device=args.embedding_device)

def load_embedding_model_instance():
    """Load a single embedding model instance based on args"""
    if args.embedding_model == 'qwen':
        qwen_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
        model = load_qwen_model(
            qwen_model_name,
            backend=args.qwen_backend,
            device_arg=args.embedding_device,
            use_flash_attention=args.qwen_flash_attention,
            compute_dtype=args.qwen_compute_dtype,
            onnx_model_path=args.qwen_onnx_model_path,
            onnx_provider=args.qwen_onnx_provider,
            max_length=args.qwen_max_length,
        )
    elif args.embedding_model == 'hf':
        _, model = load_hf_embedding_model()
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

def initialize_models():
    global embedding_model, embedding_model_pool, rerank_model, rerank_model_pool, embedding_model_name
    
    logger.info("Initializing models...")
    
    # Detect if we'll be using GPU
    using_gpu_embedding = args.embedding_device != "cpu"
    using_gpu_rerank = args.rerank_device != "cpu"
    
    # Determine reranking model choice
    if args.reranking_model is None:
        if args.embedding_model == 'qwen':
            reranking_model_choice = 'qwen'
            logger.info("No reranking model specified, using qwen (same as embedding model)")
        else:
            reranking_model_choice = 'mixedbread'
    else:
        reranking_model_choice = args.reranking_model

    # Determine if we use shared pool
    use_shared_qwen_pool = (
        args.embedding_model == 'qwen' and
        reranking_model_choice == 'qwen' and
        args.embedding_device == args.rerank_device and
        (using_gpu_embedding or using_gpu_rerank) and
        (args.num_concurrent_embedding > 1 or args.num_concurrent_rerank > 1)
    )
    
    # ----- Embedding Model Init -----
    logger.info(f"Loading embedding model on device: {args.embedding_device}")
    logger.info(f"Using embedding model: {args.embedding_model}")
    if args.embedding_model == 'qwen':
        logger.info(f"Using Qwen backend: {args.qwen_backend}")
    elif args.embedding_model == 'hf':
        logger.info(f"Using Hugging Face model slug: {args.hf_model_slug}")
    
    shared_qwen_pool = None
    
    if use_shared_qwen_pool:
        shared_pool_size = max(args.num_concurrent_embedding, args.num_concurrent_rerank)
        qwen_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
        embedding_model_name = qwen_model_name
        
        logger.info(f"♻️  Creating SHARED model pool for both embedding and reranking")
        logger.info(f"   Using {shared_pool_size} Qwen instance(s) for both tasks")
        
        shared_qwen_pool = ModelPool(
            shared_pool_size,
            load_embedding_model_instance,
            embedding_model_name
        )
        embedding_model_pool = shared_qwen_pool
        embedding_model = None
        
    elif using_gpu_embedding and args.num_concurrent_embedding > 1:
        if args.embedding_model == 'qwen':
            embedding_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
        elif args.embedding_model == 'hf':
            embedding_model_name = resolve_model_repo_id(path=args.hf_model_slug) if args.hf_model_slug else "hf"
        else:
            embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
        
        embedding_model_pool = ModelPool(
            args.num_concurrent_embedding,
            load_embedding_model_instance,
            embedding_model_name
        )
        embedding_model = None
        
    else:
        # Single instance
        if args.embedding_model == 'qwen':
            embedding_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
            embedding_model = load_qwen_model(
                embedding_model_name,
                backend=args.qwen_backend,
                device_arg=args.embedding_device,
                use_flash_attention=args.qwen_flash_attention,
                compute_dtype=args.qwen_compute_dtype,
                onnx_model_path=args.qwen_onnx_model_path,
                onnx_provider=args.qwen_onnx_provider,
                max_length=args.qwen_max_length,
            )
        elif args.embedding_model == 'hf':
            embedding_model_name, embedding_model = load_hf_embedding_model()
        else:
            embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
            embedding_model = SentenceTransformer(embedding_model_name, truncate_dim=args.embedding_dim, device=args.embedding_device)
        
        # Apply compile to single instance
        embedding_model = compile_model_if_enabled(embedding_model, f"Embedding-{args.embedding_model}")
        embedding_model_pool = None

    # Warmup embedding
    if embedding_model_pool:
        logger.info(f"🔥 Warming up embedding model pool...")
        for i, model in enumerate(embedding_model_pool.get_all_models()):
            warmup_model(model, 'embedding', f"{embedding_model_name}-{i+1}", num_samples=args.warmup_samples)
    elif embedding_model:
        warmup_model(embedding_model, 'embedding', embedding_model_name, num_samples=args.warmup_samples)

    # ----- Rerank Model Init -----
    logger.info(f"Using reranking model: {reranking_model_choice}")
    
    if reranking_model_choice == 'qwen':
        if use_shared_qwen_pool:
            logger.info("♻️  Reranker using SHARED pool with embedding model")
            rerank_model = PooledQwenReranker(shared_qwen_pool)
            rerank_model_pool = None
        elif (args.embedding_model == 'qwen' and 
              args.embedding_device == args.rerank_device and
              args.num_concurrent_embedding == 1 and
              args.num_concurrent_rerank == 1):
             logger.info("♻️  Reusing Qwen embedding model for reranking")
             rerank_model = embedding_model
             rerank_model_pool = None
        else:
            qwen_reranker_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
            if using_gpu_rerank and args.num_concurrent_rerank > 1:
                def load_qwen_reranker():
                    return load_qwen_model(
                        qwen_reranker_model_name,
                        backend=args.qwen_backend,
                        device_arg=args.rerank_device,
                        use_flash_attention=args.qwen_flash_attention,
                        compute_dtype=args.qwen_compute_dtype,
                        onnx_model_path=args.qwen_onnx_model_path,
                        onnx_provider=args.qwen_onnx_provider,
                        max_length=args.qwen_max_length,
                    )
                rerank_model_pool = ModelPool(
                    args.num_concurrent_rerank,
                    load_qwen_reranker,
                    qwen_reranker_model_name
                )
                rerank_model = None
            else:
                rerank_model = load_qwen_model(
                    qwen_reranker_model_name,
                    backend=args.qwen_backend,
                    device_arg=args.rerank_device,
                    use_flash_attention=args.qwen_flash_attention,
                    compute_dtype=args.qwen_compute_dtype,
                    onnx_model_path=args.qwen_onnx_model_path,
                    onnx_provider=args.qwen_onnx_provider,
                    max_length=args.qwen_max_length,
                )
                rerank_model_pool = None
    else:
        # MixedBread
        rerank_model = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-base-v2", device=args.rerank_device)
        rerank_model_pool = None

    # Apply compile to single rerank instance
    if rerank_model is not None and not rerank_model_pool:
        if not use_shared_qwen_pool and rerank_model != embedding_model: # Don't re-compile if shared
            name = f"Rerank-{reranking_model_choice}" if reranking_model_choice != 'qwen' else "Rerank-Qwen"
            rerank_model = compile_model_if_enabled(rerank_model, name)

    # Warmup reranker
    if rerank_model_pool:
        logger.info(f"🔥 Warming up reranking model pool...")
        for i, model in enumerate(rerank_model_pool.get_all_models()):
            warmup_model(model, 'rerank', f"{reranking_model_choice}-{i+1}", num_samples=args.warmup_samples)
    elif rerank_model and not use_shared_qwen_pool and rerank_model != embedding_model:
        warmup_model(rerank_model, 'rerank', f"Rerank-{reranking_model_choice}", num_samples=args.warmup_samples)

    # Pre-compute calibration embeddings for stable int8 quantization
    if args.quant == 'int8':
        if embedding_model is not None:
            _compute_calibration_embeddings(embedding_model)
        elif embedding_model_pool is not None:
            _compute_calibration_embeddings(embedding_model_pool.get_model())

    logger.info("Models initialized successfully")

def _compute_calibration_embeddings(model):
    """Generate calibration embeddings for stable int8 quantization ranges."""
    global calibration_embeddings
    logger.info("🔧 Computing int8 calibration embeddings from dummy texts...")
    try:
        if hasattr(model, 'encode'):
            emb = model.encode(_CALIBRATION_TEXTS)
        elif hasattr(model, 'get_embeddings'):
            emb = model.get_embeddings(_CALIBRATION_TEXTS)
        else:
            logger.warning("⚠️  Could not compute calibration embeddings: model has no encode/get_embeddings method")
            return

        # Convert torch tensors to numpy float32
        if hasattr(emb, 'cpu'):
            if emb.dtype == torch.bfloat16:
                emb = emb.to(torch.float32)
            emb = emb.cpu().numpy()

        calibration_embeddings = emb
        logger.info(f"✅ Computed {len(_CALIBRATION_TEXTS)} calibration embeddings for int8 quantization")
    except Exception as e:
        logger.warning(f"⚠️  Failed to compute calibration embeddings: {e}")

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
